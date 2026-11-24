"""3D CT preprocessing.

Pipeline: load -> validate -> reorient (LPS) -> resample to target spacing
-> clip to HU window -> normalize to [0, 1] -> save as NIfTI.
Inputs: NIfTI (.nii, .nii.gz) or a zipped DICOM series (.zip).
"""
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from app.core.config import settings

MB = 1024 * 1024


class InvalidVolumeError(ValueError):
    """The uploaded file is not a usable 3D CT volume."""


def detect_format(filename: str) -> str | None:
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return "nii.gz"
    if name.endswith(".nii"):
        return "nii"
    if name.endswith(".zip"):
        return "zip"
    return None


def _safe_extract(zip_path: Path, dest: Path) -> None:
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise InvalidVolumeError("File is not a valid zip archive") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > settings.MAX_ZIP_FILES:
            raise InvalidVolumeError("Zip contains too many files")
        if sum(i.file_size for i in infos) > settings.MAX_UNZIPPED_MB * MB:
            raise InvalidVolumeError("Zip is too large once extracted")
        root = dest.resolve()
        for info in infos:  # zip-slip protection
            if not (root / info.filename).resolve().is_relative_to(root):
                raise InvalidVolumeError("Zip contains an unsafe path")
        zf.extractall(dest)


def _read_dicom_series(folder: Path) -> sitk.Image:
    best: list[str] = []
    for current, _dirs, _files in os.walk(folder):
        for series_id in sitk.ImageSeriesReader.GetGDCMSeriesIDs(current):
            files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(current, series_id))
            if len(files) > len(best):
                best = files
    if not best:
        raise InvalidVolumeError("No DICOM series found in the zip")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best)
    return reader.Execute()


def load_volume(path: Path, fmt: str) -> sitk.Image:
    try:
        if fmt == "zip":
            with tempfile.TemporaryDirectory() as tmp:
                _safe_extract(path, Path(tmp))
                return _read_dicom_series(Path(tmp))
        return sitk.ReadImage(str(path))
    except InvalidVolumeError:
        raise
    except Exception as exc:  # SimpleITK raises RuntimeError with file details
        raise InvalidVolumeError(f"Could not read the volume: {str(exc).splitlines()[0]}") from exc


def validate(img: sitk.Image) -> None:
    if img.GetDimension() != 3:
        raise InvalidVolumeError(f"Expected a 3D volume, got {img.GetDimension()}D")
    if img.GetNumberOfComponentsPerPixel() != 1:
        raise InvalidVolumeError("Expected a single-channel volume")
    if min(img.GetSize()) < settings.MIN_DIM_VOXELS:
        raise InvalidVolumeError(f"Volume too small (minimum {settings.MIN_DIM_VOXELS} voxels per axis)")
    spacing = np.array(img.GetSpacing())
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise InvalidVolumeError("Volume has invalid voxel spacing")
    arr = sitk.GetArrayViewFromImage(img)
    if not np.all(np.isfinite(arr)):
        raise InvalidVolumeError("Volume contains NaN or infinite values")
    if float(arr.min()) == float(arr.max()):
        raise InvalidVolumeError("Volume is blank (no intensity variation)")


def reorient(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")


def resample(img: sitk.Image, target_mm: list[float]) -> sitk.Image:
    size = np.array(img.GetSize(), dtype=float)
    spacing = np.array(img.GetSpacing(), dtype=float)
    target = np.array(target_mm, dtype=float)
    new_size = np.maximum(np.round(size * spacing / target).astype(int), 1)
    if int(np.prod(new_size)) > settings.MAX_PROCESSED_VOXELS:
        raise InvalidVolumeError("Volume is too large after resampling")
    return sitk.Resample(
        img,
        [int(s) for s in new_size],
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        [float(t) for t in target],
        img.GetDirection(),
        -1024.0,
        sitk.sitkFloat32,
    )


def window_and_normalize(img: sitk.Image, hu_min: float, hu_max: float) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img)
    arr = (np.clip(arr, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
    if float(arr.max()) == float(arr.min()):
        raise InvalidVolumeError("No intensities fall inside the HU window; is this a CT in HU?")
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.CopyInformation(img)
    return out


def run_pipeline(src: Path, fmt: str, dst: Path) -> dict:
    img = sitk.Cast(load_volume(src, fmt), sitk.sitkFloat32)
    validate(img)
    meta = {
        "original_shape": [int(s) for s in img.GetSize()],
        "voxel_spacing_mm": [round(float(s), 4) for s in img.GetSpacing()],
    }
    img = reorient(img)
    img = resample(img, settings.TARGET_SPACING_MM)
    img = window_and_normalize(img, settings.HU_MIN, settings.HU_MAX)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(dst), useCompression=True)
    meta["processed_shape"] = [int(s) for s in img.GetSize()]
    meta["processed_spacing_mm"] = [round(float(s), 4) for s in img.GetSpacing()]
    return meta
