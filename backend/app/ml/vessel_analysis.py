"""Vessel Segmentation and Morphological Analysis (Phase 16.1 Refined).

Performs:
1. Binary vessel mask thresholding & configurable post-processing (raw default min_size=0)
2. Preserves both raw and cleaned masks to maintain separation
3. Morphological skeletonization & vessel centerline extraction with quality verification
4. Rigorous centerline acceptance criteria (rejection of <5 px unsuitable fragments)
5. Euclidean Distance Transform (EDT) radius estimation (2D in-plane approximation)
6. Dynamic physical pixel spacing ingestion accounting for actual input and resized dimensions:
   effective_spacing = native_spacing * (orig_dim / target_dim)
7. Vessel continuity, component count, endpoint, and bifurcation identification
8. Candidate narrowing-region identification
9. Multi-panel composite visualization & reporting
10. Strict nullable representation (None) for unavailable measurements (no silent zeroing)
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from app.ml.config import config
except ImportError:
    try:
        from .config import config
    except ImportError:
        from config import config


def compute_effective_spacing(
    native_spacing: Tuple[float, float],
    orig_shape: Tuple[int, int],
    target_shape: Tuple[int, int]
) -> Tuple[float, float, float]:
    """Calculate effective in-plane spacing accounting for actual original and resized dimensions.

    Parameters:
        native_spacing: (spacing_y, spacing_x) in mm from image metadata.
        orig_shape: (orig_height, orig_width) of input image.
        target_shape: (target_height, target_width) after model resizing.

    Returns:
        (eff_spacing_y, eff_spacing_x, isotropic_effective_spacing) in mm/pixel.
    """
    scale_y = float(orig_shape[0]) / float(target_shape[0])
    scale_x = float(orig_shape[1]) / float(target_shape[1])
    eff_sy = float(native_spacing[0]) * scale_y
    eff_sx = float(native_spacing[1]) * scale_x
    isotropic_eff = float(np.sqrt(eff_sy * eff_sx))
    return eff_sy, eff_sx, isotropic_eff


def clean_vessel_mask(bin_mask: np.ndarray, min_size: int = 0) -> Tuple[np.ndarray, int]:
    """Perform connected-component noise filtering on a binary mask.

    If min_size <= 0: returns original raw mask unchanged (preserving small distal branches).
    If min_size > 0: removes connected components smaller than min_size pixels.
    """
    if min_size <= 0:
        return bin_mask.copy(), int(bin_mask.sum())

    labels, _ = ndi.label(bin_mask)
    if labels.max() == 0:
        return bin_mask.copy(), 0

    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False  # background
    cleaned = keep[labels]
    return cleaned, int(cleaned.sum())


def extract_centerline_and_topology(
    cleaned_mask: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract vessel medial-axis centerline and evaluate topological quality.

    Engineering Safeguard Acceptance Criteria:
    - num_skeleton_px == 0: 'NO_VESSEL_DETECTED'
    - 0 < num_skeleton_px < 5: 'UNSUITABLE_FRAGMENT' (too short for continuous medial-axis profile)
    - 5 <= num_skeleton_px < 10: 'SHORT_SEGMENT' (suitable for width check, unsuitable for percentile stenosis)
    - num_skeleton_px >= 10: 'SUITABLE_FOR_CENTERLINE_ANALYSIS'

    Note: These thresholds are engineering safeguards, not clinically validated standards.
    Passing the threshold does not establish anatomical measurement accuracy.
    """
    skeleton = skeletonize(cleaned_mask > 0)
    num_skeleton_px = int(skeleton.sum())

    if num_skeleton_px == 0:
        return skeleton, {
            "num_skeleton_px": 0,
            "endpoints_count": 0,
            "junctions_count": 0,
            "skeleton_components": 0,
            "status": "NO_VESSEL_DETECTED",
            "is_suitable_for_centerline": False,
            "is_suitable_for_stenosis": False,
            "message": "Empty vessel mask; no centerline could be extracted."
        }

    # Connected components of the skeleton itself (detect multi-vessel fragments)
    skel_struct = np.ones((3, 3), dtype=int)
    _, num_skel_components = ndi.label(skeleton, structure=skel_struct)

    # Count endpoints and junctions via 3x3 8-neighborhood kernel
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neighbor_count = ndi.convolve(skeleton.astype(int), kernel, mode="constant")
    endpoints = int(((neighbor_count == 1) & skeleton).sum())
    junctions = int(((neighbor_count >= 3) & skeleton).sum())

    if num_skeleton_px < 5:
        status = "UNSUITABLE_FRAGMENT"
        is_centerline_valid = False
        is_stenosis_valid = False
        msg = f"Centerline length ({num_skeleton_px} px) is too short for reliable profile analysis (< 5 px threshold)."
    elif num_skeleton_px < 10:
        status = "SHORT_SEGMENT"
        is_centerline_valid = True
        is_stenosis_valid = False
        msg = f"Centerline length ({num_skeleton_px} px) is sufficient for local width, but too short for stenosis estimation (< 10 px threshold)."
    else:
        status = "SUITABLE_FOR_CENTERLINE_ANALYSIS"
        is_centerline_valid = True
        is_stenosis_valid = True
        msg = f"Valid continuous centerline tract ({num_skeleton_px} px, {num_skel_components} component(s))."

    topo_info = {
        "num_skeleton_px": num_skeleton_px,
        "endpoints_count": endpoints,
        "junctions_count": junctions,
        "skeleton_components": num_skel_components,
        "status": status,
        "is_suitable_for_centerline": is_centerline_valid,
        "is_suitable_for_stenosis": is_stenosis_valid,
        "message": msg
    }
    return skeleton, topo_info


def analyze_vessel_morphology(
    mask_arr: np.ndarray,
    image_arr: np.ndarray = None,
    gt_arr: np.ndarray = None,
    pixel_spacing_mm: Optional[float] = None,
    min_size: int = 0,
    case_name: str = "sample"
) -> Tuple[Dict[str, Any], Path]:
    """Perform complete 2D morphological analysis on a vessel binary mask.

    Parameters:
        mask_arr: 2D binary or probability mask.
        image_arr: Optional 2D CT slice image array for visualization.
        gt_arr: Optional ground truth mask array.
        pixel_spacing_mm: Physical pixel spacing in mm. If None or <=0, mm conversions are NOT performed.
        min_size: Connected component filtering threshold (default 0: raw prediction preserved).
        case_name: Case stem for saving figures.
    """
    # 1. Ensure binary mask & preserve raw mask
    bin_mask = (mask_arr > 0).astype(bool)
    raw_fg_pixels = int(bin_mask.sum())

    # 2. Configurable Connected-Component Cleaning
    cleaned, cleaned_fg_pixels = clean_vessel_mask(bin_mask, min_size=min_size)

    # 3. Skeletonization and Topological Evaluation
    skeleton, topo = extract_centerline_and_topology(cleaned)
    num_skeleton_px = topo["num_skeleton_px"]

    # 4. Euclidean Distance Transform (EDT)
    distance_map = ndi.distance_transform_edt(cleaned)
    centerline_coords = np.where(skeleton)

    if num_skeleton_px > 0:
        radius_px = distance_map[centerline_coords]
        diameter_px = 2.0 * radius_px
        mean_diam_px = float(np.mean(diameter_px))
        min_diam_px = float(np.min(diameter_px))
        max_diam_px = float(np.max(diameter_px))
        median_diam_px = float(np.median(diameter_px))
        ref_diam_px = float(np.percentile(diameter_px, config.analysis.reference_percentile))
    else:
        diameter_px = np.array([])
        mean_diam_px = min_diam_px = max_diam_px = median_diam_px = ref_diam_px = None

    # 5. Physical Spacing Verification (NO silent fallback!)
    has_physical_spacing = (pixel_spacing_mm is not None and float(pixel_spacing_mm) > 0.0)

    if has_physical_spacing and num_skeleton_px > 0:
        pixel_spacing_val = float(pixel_spacing_mm)
        mean_diam_mm = round(mean_diam_px * pixel_spacing_val, 2)
        min_diam_mm = round(min_diam_px * pixel_spacing_val, 2)
        max_diam_mm = round(max_diam_px * pixel_spacing_val, 2)
        median_diam_mm = round(median_diam_px * pixel_spacing_val, 2)
        ref_diam_mm = round(ref_diam_px * pixel_spacing_val, 2)
        spacing_note = f"Valid patient physical spacing ({pixel_spacing_val:.4f} mm/pixel)."
    else:
        pixel_spacing_val = float(pixel_spacing_mm) if has_physical_spacing else None
        mean_diam_mm = min_diam_mm = max_diam_mm = median_diam_mm = ref_diam_mm = None
        spacing_note = "Physical spacing unavailable. Measurements reported in pixels only." if not has_physical_spacing else "No centerline detected."

    # 6. Candidate Narrowing Points (>25% reduction relative to reference diameter)
    candidate_narrowings = 0
    narrowing_mask = np.zeros_like(skeleton, dtype=bool)
    if topo["is_suitable_for_centerline"] and ref_diam_px is not None and ref_diam_px > 0:
        narrow_indices = np.where(diameter_px < (0.75 * ref_diam_px))[0]
        candidate_narrowings = len(narrow_indices)
        for idx in narrow_indices:
            r, c = centerline_coords[0][idx], centerline_coords[1][idx]
            narrowing_mask[r, c] = True
    else:
        narrow_indices = np.array([], dtype=int)

    results = {
        "case_name": case_name,
        "raw_foreground_pixels": raw_fg_pixels,
        "cleaned_foreground_pixels": cleaned_fg_pixels,
        "foreground_pixels": cleaned_fg_pixels,  # backward compatibility
        "filter_min_size": min_size,
        "skeleton_centerline_pixels": num_skeleton_px,
        "skeleton_components": topo["skeleton_components"],
        "endpoints_count": topo["endpoints_count"],
        "junctions_count": topo["junctions_count"],
        "centerline_status": topo["status"],
        "is_suitable_for_centerline": topo["is_suitable_for_centerline"],
        "is_suitable_for_stenosis": topo["is_suitable_for_stenosis"],
        "centerline_message": topo["message"],
        "mean_diameter_px": round(mean_diam_px, 2) if mean_diam_px is not None else None,
        "min_diameter_px": round(min_diam_px, 2) if min_diam_px is not None else None,
        "max_diameter_px": round(max_diam_px, 2) if max_diam_px is not None else None,
        "median_diameter_px": round(median_diam_px, 2) if median_diam_px is not None else None,
        "reference_diameter_px": round(ref_diam_px, 2) if ref_diam_px is not None else None,
        "physical_spacing_available": has_physical_spacing,
        "pixel_spacing_mm": round(pixel_spacing_val, 4) if has_physical_spacing else None,
        "mean_diameter_mm": mean_diam_mm,
        "min_diameter_mm": min_diam_mm,
        "max_diameter_mm": max_diam_mm,
        "reference_diameter_mm": ref_diam_mm,
        "candidate_narrowing_points": candidate_narrowings,
        "spacing_note": spacing_note,
        "notes": "2D in-plane EDT diameter approximation. Not an orthogonal 3D lumen cross-section."
    }

    # 7. Multi-panel composite visualization
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=130)

    # Panel 1: Original Image
    if image_arr is not None:
        axes[0, 0].imshow(image_arr, cmap="gray")
        axes[0, 0].set_title("1. Original CT Slice")
    else:
        axes[0, 0].text(0.5, 0.5, "Image Not Provided", ha="center")
        axes[0, 0].set_title("1. Original CT Slice")
    axes[0, 0].axis("off")

    # Panel 2: Ground Truth Mask
    if gt_arr is not None:
        axes[0, 1].imshow(gt_arr > 0, cmap="gray")
        axes[0, 1].set_title(f"2. Ground-Truth Mask ({int((gt_arr > 0).sum())} px)")
    else:
        axes[0, 1].text(0.5, 0.5, "GT Not Provided", ha="center")
        axes[0, 1].set_title("2. Ground Truth")
    axes[0, 1].axis("off")

    # Panel 3: Predicted Mask (Raw vs Cleaned)
    axes[0, 2].imshow(cleaned, cmap="gray")
    if min_size > 0:
        axes[0, 2].set_title(f"3. Cleaned Mask ({cleaned_fg_pixels} px | Raw: {raw_fg_pixels} px)")
    else:
        axes[0, 2].set_title(f"3. Raw Predicted Mask ({raw_fg_pixels} px)")
    axes[0, 2].axis("off")

    # Panel 4: Segmentation Overlay
    if image_arr is not None:
        axes[1, 0].imshow(image_arr, cmap="gray")
        masked = np.ma.masked_where(~cleaned, cleaned)
        axes[1, 0].imshow(masked, cmap="autumn", alpha=0.55)
        axes[1, 0].set_title("4. Lumen Segmentation Overlay")
    else:
        axes[1, 0].imshow(cleaned, cmap="autumn")
        axes[1, 0].set_title("4. Vessel Overlay")
    axes[1, 0].axis("off")

    # Panel 5: Centerline Skeleton & Quality Badge
    if image_arr is not None:
        axes[1, 1].imshow(image_arr, cmap="gray")
    else:
        axes[1, 1].imshow(np.zeros_like(cleaned), cmap="gray")

    if num_skeleton_px > 0:
        skel_masked = np.ma.masked_where(~skeleton, skeleton)
        axes[1, 1].imshow(skel_masked, cmap="spring", alpha=0.85)
        if candidate_narrowings > 0:
            axes[1, 1].scatter(
                centerline_coords[1][narrow_indices],
                centerline_coords[0][narrow_indices],
                c="cyan", s=30, marker="x", label="Narrowing Point"
            )
            axes[1, 1].legend(loc="upper right", fontsize=8)

    badge = f"[{topo['status']}]"
    axes[1, 1].set_title(f"5. Centerline ({num_skeleton_px} px)\n{badge}")
    axes[1, 1].axis("off")

    # Panel 6: Local Diameter Map (Distance Transform)
    if has_physical_spacing and num_skeleton_px > 0:
        diam_map = distance_map * 2.0 * pixel_spacing_val
        unit_lbl = "Diameter (mm)"
        ref_str = f"Ref: {ref_diam_mm:.1f} mm" if ref_diam_mm is not None else "Ref: N/A"
        min_str = f"Min: {min_diam_mm:.1f} mm" if min_diam_mm is not None else "Min: N/A"
    elif num_skeleton_px > 0:
        diam_map = distance_map * 2.0
        unit_lbl = "Diameter (pixels)"
        ref_str = f"Ref: {ref_diam_px:.1f} px"
        min_str = f"Min: {min_diam_px:.1f} px"
    else:
        diam_map = np.zeros_like(cleaned, dtype=float)
        unit_lbl = "Diameter (pixels)"
        ref_str = "Ref: N/A"
        min_str = "Min: N/A"

    im6 = axes[1, 2].imshow(diam_map, cmap="hot")
    axes[1, 2].set_title(f"6. Diameter Map\n({ref_str}, {min_str})")
    axes[1, 2].axis("off")
    fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04, label=unit_lbl)

    plt.suptitle(f"Phase 16.1 Vessel Morphological & Centerline Analysis: {case_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_dir = config.paths.analysis_dir
    legacy_dir = config.paths.legacy_outputs_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / f"{case_name}_morphology.png"
    legacy_fig_path = legacy_dir / f"{case_name}_morphology.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    fig.savefig(legacy_fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Save cleaned mask and centerline images
    Image.fromarray((cleaned * 255).astype(np.uint8)).save(out_dir / f"{case_name}_cleaned_mask.png")
    Image.fromarray((skeleton * 255).astype(np.uint8)).save(out_dir / f"{case_name}_centerline.png")

    return results, fig_path


def main():
    parser = argparse.ArgumentParser(description="Phase 16.1 Vessel Morphological Analysis")
    parser.add_argument("--mask", default=None, help="Path to binary mask PNG")
    parser.add_argument("--image", default=None, help="Path to original image PNG")
    parser.add_argument("--gt", default=None, help="Path to ground-truth mask PNG")
    parser.add_argument("--min-size", type=int, default=0, help="Min component size (0=raw default)")
    parser.add_argument("--spacing", type=float, default=None, help="Physical pixel spacing in mm")
    args = parser.parse_args()

    config.setup_environment()
    out_dir = config.paths.analysis_dir
    legacy_dir = config.paths.legacy_outputs_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    if not args.mask:
        pred_masks = list(config.paths.predictions_dir.glob("*_mask.png"))
        if not pred_masks:
            pred_masks = list(config.paths.legacy_outputs_dir.glob("predictions/*_mask.png"))
        if not pred_masks:
            raise FileNotFoundError("No predicted masks found. Run predict.py first.")
        mask_path = pred_masks[0]
    else:
        mask_path = Path(args.mask)

    case_stem = mask_path.stem.replace("_mask", "")
    print(f"Analyzing vessel morphology for case: {case_stem} ({mask_path})", flush=True)

    mask_arr = np.asarray(Image.open(mask_path).convert("L"))

    image_arr = None
    if args.image and Path(args.image).exists():
        image_arr = np.asarray(Image.open(args.image).convert("L"))
    else:
        candidate_img = config.paths.test_dir / "images" / f"{case_stem}.png"
        if candidate_img.exists():
            image_arr = np.asarray(Image.open(candidate_img).convert("L"))

    gt_arr = None
    if args.gt and Path(args.gt).exists():
        gt_arr = np.asarray(Image.open(args.gt).convert("L"))
    else:
        candidate_gt = config.paths.test_dir / "masks" / f"{case_stem}.png"
        if candidate_gt.exists():
            gt_arr = np.asarray(Image.open(candidate_gt).convert("L"))

    stats, fig_path = analyze_vessel_morphology(
        mask_arr, image_arr, gt_arr, pixel_spacing_mm=args.spacing, min_size=args.min_size, case_name=case_stem
    )

    df = pd.DataFrame([stats])
    csv_path = out_dir / "vessel_measurements.csv"
    legacy_csv_path = legacy_dir / "vessel_measurements.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(legacy_csv_path, index=False)

    print("\n--- Vessel Morphological Analysis Results ---", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nSaved composite visualization to {fig_path}", flush=True)
    print(f"Saved measurements to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
