"""Render single 2D slices of a 3D volume as PNG for the web viewer."""
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image as PILImage

AXES = {"axial": 0, "coronal": 1, "sagittal": 2}  # numpy array order is (z, y, x)


@lru_cache(maxsize=3)
def _load_array(path: str, mtime_ns: int) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def slice_count(shape_xyz: list[int], axis: str) -> int:
    x, y, z = shape_xyz
    return {"axial": z, "coronal": y, "sagittal": x}[axis]


def render_slice_png(path: Path, axis: str, index: int | None, window: tuple[float, float]) -> bytes:
    arr = _load_array(str(path), path.stat().st_mtime_ns)
    ax = AXES[axis]
    n = arr.shape[ax]
    if index is None:
        index = n // 2
    if not 0 <= index < n:
        raise IndexError(f"index must be between 0 and {n - 1}")
    sl = np.take(arr, index, axis=ax)
    if axis != "axial":
        sl = np.flipud(sl)  # keep head-to-foot (z) pointing up
    lo, hi = window
    img8 = (np.clip((sl.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(img8, mode="L").save(buf, format="PNG")
    return buf.getvalue()
