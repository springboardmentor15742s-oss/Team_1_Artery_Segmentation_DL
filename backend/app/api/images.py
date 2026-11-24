import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, writers
from app.core.config import settings
from app.db.session import get_db
from app.models.models import Case, CaseStatus, Image, User
from app.schemas.schemas import ImageOut
from app.services import audit, slices, storage
from app.services.preprocessing import InvalidVolumeError, detect_format, run_pipeline

router = APIRouter(prefix="/api", tags=["images"])

CHUNK = 1024 * 1024


def _stream_to_disk(upload: UploadFile, dest: Path) -> int:
    """Copy the upload to disk in chunks, enforcing the size limit without loading it into memory."""
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(CHUNK):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB} MB")
            out.write(chunk)
    return size


@router.post("/cases/{case_id}/images", response_model=ImageOut, status_code=201)
def upload_image(case_id: int, file: UploadFile = File(...), db: Session = Depends(get_db),
                 user: User = Depends(writers)):
    # Plain `def`: FastAPI runs it in a worker thread, so heavy preprocessing does not block the server.
    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    fmt = detect_format(file.filename or "")
    if fmt is None:
        raise HTTPException(400, "Unsupported file type. Allowed: .nii, .nii.gz, .zip (DICOM series)")

    key = uuid.uuid4().hex  # the client filename is never used on disk
    folder = storage.image_dir(case_id, key)
    folder.mkdir(parents=True, exist_ok=True)
    original = folder / f"original.{fmt}"
    processed = folder / "processed.nii.gz"
    try:
        size = _stream_to_disk(file, original)
        meta = run_pipeline(original, fmt, processed)
    except InvalidVolumeError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, str(exc))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    image = Image(case_id=case_id, storage_key=key, original_path=str(original), processed_path=str(processed),
                  file_format=fmt, size_bytes=size, uploaded_by=user.id, **meta)
    case.status = CaseStatus.preprocessed
    db.add(image)
    db.commit()
    audit.log(db, user.id, "upload", "image", image.id)
    return image


@router.get("/cases/{case_id}/images", response_model=list[ImageOut])
def list_images(case_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case.images


def _get_image(db: Session, image_id: int) -> Image:
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    return image


@router.get("/images/{image_id}/file")
def download_volume(image_id: int, processed: bool = True, db: Session = Depends(get_db),
                    _: User = Depends(get_current_user)):
    image = _get_image(db, image_id)
    path = Path(image.processed_path if processed and image.processed_path else image.original_path)
    if not path.exists():
        raise HTTPException(404, "File missing on storage")
    return FileResponse(path, filename=path.name)


@router.get("/images/{image_id}/slice")
def get_slice(image_id: int, axis: Literal["axial", "coronal", "sagittal"] = "axial",
              index: int | None = Query(None, ge=0), source: Literal["processed", "original"] = "processed",
              db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    image = _get_image(db, image_id)
    if source == "processed":
        path, window = Path(image.processed_path or ""), (0.0, 1.0)
    else:
        path, window = Path(image.original_path), (settings.HU_MIN, settings.HU_MAX)
    if source == "original" and image.file_format == "zip":
        raise HTTPException(400, "Original DICOM zip cannot be sliced; use source=processed")
    if not path.is_file():
        raise HTTPException(404, "File missing on storage")
    try:
        png = slices.render_slice_png(path, axis, index, window)
    except IndexError as exc:
        raise HTTPException(422, str(exc))
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/images/{image_id}/overlay")
def get_overlay(
    image_id: int,
    slice_index: int | None = Query(None, ge=0),
    analysis_id: int | None = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Return a segmentation overlay PNG for the given image.

    - If *analysis_id* is provided, retrieves that specific result.
    - Otherwise, returns the overlay from the most recent analysis of this image.
    - If the overlay file exists on disk it is served directly; otherwise it is
      generated dynamically from the stored mask.

    **RESEARCH PROTOTYPE – NOT FOR CLINICAL USE.**
    """
    from sqlalchemy import select as sa_select

    from app.models.models import AnalysisResult
    from app.services.analysis_service import generate_overlay_png

    image = _get_image(db, image_id)

    # Fetch the requested (or latest) analysis
    if analysis_id is not None:
        result = db.get(AnalysisResult, analysis_id)
        if not result or result.image_id != image_id:
            raise HTTPException(404, "Analysis result not found for this image")
    else:
        result = db.scalar(
            sa_select(AnalysisResult)
            .where(AnalysisResult.image_id == image_id)
            .order_by(AnalysisResult.id.desc())
        )
        if not result:
            raise HTTPException(404, "No analysis found for this image; run /analyze first")

    # Determine which slice to show
    target_slice = slice_index if slice_index is not None else result.slice_index

    # --- Fast path: serve saved overlay if it exists ---
    if result.overlay_path and Path(result.overlay_path).is_file():
        saved_path = Path(result.overlay_path)
        # If a different slice_index is requested we fall through to dynamic generation
        if slice_index is None or (
            result.slice_index is not None and slice_index == result.slice_index
        ):
            return FileResponse(str(saved_path), media_type="image/png")

    # --- Dynamic path: generate from stored mask or re-run on the slice ---
    if not image.processed_path or not Path(image.processed_path).is_file():
        raise HTTPException(404, "Processed NIfTI not found")

    import SimpleITK as sitk
    import numpy as np

    arr = sitk.GetArrayFromImage(sitk.ReadImage(image.processed_path))
    z_max = arr.shape[0]
    idx = target_slice if target_slice is not None else z_max // 2
    if not 0 <= idx < z_max:
        raise HTTPException(422, f"slice_index must be between 0 and {z_max - 1}")
    ct_slice = arr[idx].astype(np.float32)

    # Build mask: use saved mask file if available and slice matches
    mask: np.ndarray | None = None
    if (
        result.mask_path
        and Path(result.mask_path).is_file()
        and (slice_index is None or slice_index == result.slice_index)
    ):
        from PIL import Image as PILImage
        mask_img = PILImage.open(result.mask_path).convert("L")
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    if mask is None:
        # No saved mask: return the CT slice alone (grayscale)
        import io
        from PIL import Image as PILImage
        ct_uint8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(ct_uint8, mode="L").save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    # Resize mask to match CT slice if needed
    if mask.shape != ct_slice.shape:
        from PIL import Image as PILImage
        m_img = PILImage.fromarray(mask * 255, mode="L").resize(
            (ct_slice.shape[1], ct_slice.shape[0]), PILImage.NEAREST
        )
        mask = (np.array(m_img) > 127).astype(np.uint8)

    overlay_bytes = generate_overlay_png(ct_slice, mask)
    return Response(overlay_bytes, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})
