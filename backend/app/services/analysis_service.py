"""
Orchestrates a single-slice AI segmentation analysis.

Call ``run_case_analysis()`` from an API endpoint.  It is *synchronous*;
the endpoint itself runs in a FastAPI worker thread (plain ``def``), so the
server event loop is never blocked.

RESEARCH DISCLAIMER
-------------------
All measurements produced by this service are geometric estimates from a
research prototype model.  They have NOT been validated for clinical use.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.models.models import AnalysisResult, Case, CaseStatus, Image
from app.services import storage
from app.services.ai_engine import ai_engine

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimum foreground pixels before vessel / stenosis analysis is attempted.
_MIN_VESSEL_PIXELS = 10


def _coerce_prediction_mask(prediction: dict, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the native-resolution binary mask and probability map for the selected slice.

    The runtime AI engine returns a dict keyed by `mask_native` and `probability_map_native`.
    These are the only safe values to use downstream because they correspond to the actual
    selected slice rather than the 128x128 model input resolution.
    """
    if not isinstance(prediction, dict):
        raise TypeError(f"AI prediction output must be a dict, got {type(prediction).__name__}")

    mask_native = prediction.get("mask_native")
    if mask_native is None:
        raise ValueError("AI prediction output missing required 'mask_native' field")

    mask = np.asarray(mask_native)
    if mask.ndim != 2:
        raise ValueError(f"AI prediction mask must be 2D, got shape {mask.shape}")
    if mask.shape != expected_shape:
        raise ValueError(
            f"AI prediction mask shape {mask.shape} does not match selected slice shape {expected_shape}"
        )

    mask_binary = (mask > 0).astype(np.uint8)

    prob_map = prediction.get("probability_map_native")
    if prob_map is not None:
        prob = np.asarray(prob_map, dtype=np.float32)
        if prob.ndim != 2:
            raise ValueError(f"Probability map must be 2D, got shape {prob.shape}")
        if prob.shape != expected_shape:
            raise ValueError(
                f"Probability map shape {prob.shape} does not match selected slice shape {expected_shape}"
            )
    else:
        prob = None

    return mask_binary, prob


# ---------------------------------------------------------------------------
# Overlay generation helper
# ---------------------------------------------------------------------------

def generate_overlay_png(ct_slice: np.ndarray, mask: np.ndarray) -> bytes:
    """Return a PNG (bytes) of *ct_slice* with the segmentation *mask* overlaid in red.

    *ct_slice* – 2-D float array in [0, 1] (platform-normalised values).
    *mask*     – 2-D binary (uint8/bool) array, same H × W.
    """
    # Grayscale CT → RGB
    ct_uint8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([ct_uint8, ct_uint8, ct_uint8], axis=-1)

    # Red overlay where mask is positive
    alpha = 0.45
    red = np.array([220, 30, 30], dtype=np.float32)
    mask_bool = mask.astype(bool)
    for c in range(3):
        rgb[:, :, c] = np.where(
            mask_bool,
            np.clip(ct_uint8 * (1 - alpha) + red[c] * alpha, 0, 255).astype(np.uint8),
            rgb[:, :, c],
        )

    buf = io.BytesIO()
    PILImage.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def run_case_analysis(
    image: Image,
    slice_index: int | None,
    model_name: str,
    user_id: int,
    db: Session,
) -> AnalysisResult:
    """Run segmentation inference on one axial slice of *image*.

    Creates and flushes (but does NOT commit) an :class:`AnalysisResult` row.
    The caller is responsible for committing the session so that case-status
    updates and the analysis record land in the same transaction.

    Parameters
    ----------
    image:        ORM ``Image`` with a valid ``processed_path``.
    slice_index:  Axial slice index (z).  ``None`` → middle slice.
    model_name:   ``"attention_unet"`` or ``"unet"``.
    user_id:      ID of the user triggering the analysis.
    db:           Active SQLAlchemy session.
    """
    result_kwargs: dict = dict(
        case_id=image.case_id,
        image_id=image.id,
        model_name=model_name,
        created_by=user_id,
        physical_spacing_available=False,
    )

    try:
        # ------------------------------------------------------------------ #
        # 1. Load the processed NIfTI volume
        # ------------------------------------------------------------------ #
        processed_path = image.processed_path
        if not processed_path or not Path(processed_path).is_file():
            raise FileNotFoundError(f"Processed NIfTI not found: {processed_path}")

        sitk_img = sitk.ReadImage(processed_path)
        arr = sitk.GetArrayFromImage(sitk_img)   # shape (Z, Y, X), values in [0, 1]

        # ------------------------------------------------------------------ #
        # 2. Choose slice
        # ------------------------------------------------------------------ #
        z_max = arr.shape[0]
        if slice_index is None:
            slice_index = z_max // 2
        if not (0 <= slice_index < z_max):
            raise ValueError(f"slice_index {slice_index} out of range [0, {z_max - 1}]")

        ct_slice = arr[slice_index].astype(np.float32)   # (Y, X), platform-normalised [0,1]

        # ------------------------------------------------------------------ #
        # 3. Determine pixel spacing from stored metadata
        # ------------------------------------------------------------------ #
        pixel_spacing_mm: float | None = None
        physical_spacing_available = False

        ps = image.processed_spacing_mm         # JSON list [x, y, z] in mm, or None
        if ps and len(ps) >= 2:
            # In-plane spacing: average of x and y spacings
            pixel_spacing_mm = float((ps[0] + ps[1]) / 2.0)
            physical_spacing_available = True
        else:
            # Fall back to original voxel spacing if processed spacing is missing
            vs = image.voxel_spacing_mm
            if vs and len(vs) >= 2:
                pixel_spacing_mm = float((vs[0] + vs[1]) / 2.0)
                physical_spacing_available = True

        result_kwargs["pixel_spacing_mm"] = pixel_spacing_mm
        result_kwargs["physical_spacing_available"] = physical_spacing_available

        # ------------------------------------------------------------------ #
        # 4. Run AI inference
        # ------------------------------------------------------------------ #
        pred = ai_engine.predict_slice(
            ct_slice,
            source="platform_normalized",
            model_name=model_name,
        )

        if "inference_duration_ms" not in pred or pred["inference_duration_ms"] is None:
            raise ValueError("AI prediction output missing required 'inference_duration_ms' field")

        mask, prob_map = _coerce_prediction_mask(pred, ct_slice.shape)
        inference_duration_ms: float = float(pred["inference_duration_ms"])
        foreground_pixels: int = int(mask.sum())

        result_kwargs.update(
            slice_index=slice_index,
            inference_duration_ms=inference_duration_ms,
            foreground_pixels=foreground_pixels,
        )

        # ------------------------------------------------------------------ #
        # 5. Save mask + overlay PNGs
        # ------------------------------------------------------------------ #
        img_dir = storage.image_dir(image.case_id, image.storage_key)
        img_dir.mkdir(parents=True, exist_ok=True)

        mask_fname = f"mask_slice_{slice_index}.png"
        overlay_fname = f"overlay_slice_{slice_index}.png"
        mask_path = img_dir / mask_fname
        overlay_path = img_dir / overlay_fname

        # Mask PNG (8-bit grayscale, 0 / 255)
        mask_uint8 = (mask.astype(np.uint8)) * 255
        PILImage.fromarray(mask_uint8, mode="L").save(str(mask_path))

        # Overlay PNG
        overlay_bytes = generate_overlay_png(ct_slice, mask)
        overlay_path.write_bytes(overlay_bytes)

        result_kwargs["mask_path"] = str(mask_path)
        result_kwargs["overlay_path"] = str(overlay_path)

        # ------------------------------------------------------------------ #
        # 6. Vessel morphology analysis (only when vessel is present)
        # ------------------------------------------------------------------ #
        centerline_status: str | None = None
        centerline_px: int | None = None
        mean_diameter_px: float | None = None
        mean_diameter_mm: float | None = None

        if foreground_pixels >= _MIN_VESSEL_PIXELS:
            try:
                from app.ml.vessel_analysis import analyze_vessel_morphology
                vessel = analyze_vessel_morphology(mask)

                centerline_status = vessel.get("centerline_status")
                centerline_px = vessel.get("centerline_length_px")
                mean_diameter_px = vessel.get("mean_diameter_px")

                if mean_diameter_px is not None and pixel_spacing_mm:
                    mean_diameter_mm = round(mean_diameter_px * pixel_spacing_mm, 3)

            except Exception as exc:
                logger.warning("Vessel analysis failed: %s", exc)
                centerline_status = "ANALYSIS_FAILED"

        result_kwargs.update(
            centerline_status=centerline_status,
            centerline_px=centerline_px,
            mean_diameter_px=mean_diameter_px,
            mean_diameter_mm=mean_diameter_mm,
        )

        # ------------------------------------------------------------------ #
        # 7. Stenosis detection (only when vessel morphology succeeded)
        # ------------------------------------------------------------------ #
        mld_px: float | None = None
        mld_mm: float | None = None
        reference_diameter_px: float | None = None
        reference_diameter_mm: float | None = None
        stenosis_pct: float | None = None
        severity_category: str | None = None

        suitable = (
            centerline_status is not None
            and "SUITABLE" in (centerline_status or "")
            and foreground_pixels >= _MIN_VESSEL_PIXELS
        )
        if suitable:
            try:
                from app.ml.stenosis_analysis import run_stenosis_detection
                stenosis = run_stenosis_detection(mask, pixel_spacing_mm=pixel_spacing_mm)

                mld_px = stenosis.get("mld_px")
                mld_mm = stenosis.get("mld_mm")
                reference_diameter_px = stenosis.get("reference_diameter_px")
                reference_diameter_mm = stenosis.get("reference_diameter_mm")
                stenosis_pct = stenosis.get("stenosis_pct")
                severity_category = stenosis.get("severity_category")

            except Exception as exc:
                logger.warning("Stenosis detection failed: %s", exc)

        result_kwargs.update(
            mld_px=mld_px,
            mld_mm=mld_mm,
            reference_diameter_px=reference_diameter_px,
            reference_diameter_mm=reference_diameter_mm,
            stenosis_pct=stenosis_pct,
            severity_category=severity_category,
        )

        # ------------------------------------------------------------------ #
        # 8. Build the ORM record (success)
        # ------------------------------------------------------------------ #
        result_kwargs["status"] = "success"
        analysis = AnalysisResult(**result_kwargs)

    except Exception as exc:
        logger.error("Analysis failed for image %s: %s", getattr(image, "id", "?"), exc, exc_info=True)
        result_kwargs.setdefault("slice_index", slice_index if slice_index is not None else None)
        result_kwargs["status"] = "failed"
        result_kwargs["error_message"] = str(exc)
        analysis = AnalysisResult(**result_kwargs)

    db.add(analysis)
    db.flush()   # populate analysis.id without committing
    return analysis
