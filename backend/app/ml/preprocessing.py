"""Inference Preprocessing and Post-processing Utilities.

Harmonizes differences between platform preprocessing ([-200, 800] HU)
and model training-time preprocessing ([-100, 700] HU, 128x128 resolution).
Prevents double-normalization and guarantees exact tensor specifications.
"""
from typing import Literal, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from app.ml.config import config


def normalize_to_model_hu(
    arr: np.ndarray,
    source: Literal["raw_hu", "platform_normalized", "image_uint8", "auto"] = "auto",
) -> np.ndarray:
    """Normalize input 2D array to model's exact [-100, 700] HU window in [0.0, 1.0].

    Parameters:
        arr: 2D numpy array.
        source:
            - 'raw_hu': Direct Hounsfield Units from original CT volume.
            - 'platform_normalized': Volume already normalized to [0, 1] via [-200, 800] HU.
            - 'image_uint8': Standard 8-bit image in [0, 255].
            - 'auto': Heuristically determines format from min/max/dtype.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got array with shape {arr.shape}")

    hu_min = config.preprocessing.hu_min  # -100.0
    hu_max = config.preprocessing.hu_max  # 700.0

    if source == "auto":
        amin, amax = float(arr.min()), float(arr.max())
        if amin < -50.0 or amax > 255.0:
            source = "raw_hu"
        elif 0.0 <= amin and amax <= 1.0:
            source = "platform_normalized"
        else:
            source = "image_uint8"

    if source == "raw_hu":
        # Direct training window [-100, 700] HU
        norm = np.clip((arr - hu_min) / (hu_max - hu_min), 0.0, 1.0)
    elif source == "platform_normalized":
        # Platform window was [-200, 800] HU: invert to HU first, then map to model window
        # hu = norm * (800 - (-200)) + (-200)
        hu = arr * 1000.0 - 200.0
        norm = np.clip((hu - hu_min) / (hu_max - hu_min), 0.0, 1.0)
    elif source == "image_uint8":
        norm = np.clip(arr / 255.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown source type: {source}")

    return norm.astype(np.float32)


def prepare_slice_tensor(
    slice_2d: np.ndarray,
    source: Literal["raw_hu", "platform_normalized", "image_uint8", "auto"] = "auto",
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess a 2D slice into a model-ready PyTorch tensor [1, 1, H, W].

    Returns:
        tensor: torch.FloatTensor of shape (1, 1, 128, 128) in [0.0, 1.0].
        original_shape: (H, W) before resizing.
    """
    if target_size is None:
        target_size = (config.preprocessing.image_size, config.preprocessing.image_size)

    norm_2d = normalize_to_model_hu(slice_2d, source=source)
    orig_shape = norm_2d.shape

    # Resize to target resolution (128, 128) using bilinear interpolation
    if norm_2d.shape != target_size:
        pil_im = Image.fromarray((norm_2d * 255.0).astype(np.uint8))
        pil_resized = pil_im.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        resized_arr = np.asarray(pil_resized, dtype=np.float32) / 255.0
    else:
        resized_arr = norm_2d

    tensor = torch.from_numpy(resized_arr[None, None])  # [1, 1, H, W]
    return tensor, orig_shape


def postprocess_prediction(
    logits: torch.Tensor,
    original_shape: Optional[Tuple[int, int]] = None,
    threshold: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Post-process model output logits into probability maps and binary masks.

    Parameters:
        logits: Raw logits tensor from model forward pass [1, 1, 128, 128].
        original_shape: (orig_h, orig_w) if upsampling to native scan resolution is desired.
        threshold: Decision threshold tau in [0.0, 1.0].

    Returns:
        prob_128: Float32 probability map at model resolution [128, 128].
        mask_128: Uint8 binary mask at model resolution [128, 128] (0 or 255).
        prob_orig: Optional Float32 probability map at original resolution (or None).
        mask_orig: Optional Uint8 binary mask at original resolution (or None).
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

    mask_128 = (probs >= threshold).astype(np.uint8) * 255

    prob_orig = None
    mask_orig = None

    if original_shape is not None and original_shape != probs.shape:
        # Resize probability map back to original scan resolution
        prob_pil = Image.fromarray((probs * 255.0).astype(np.uint8))
        prob_resized = prob_pil.resize((original_shape[1], original_shape[0]), Image.Resampling.BILINEAR)
        prob_orig = (np.asarray(prob_resized, dtype=np.float32) / 255.0)
        mask_orig = (prob_orig >= threshold).astype(np.uint8) * 255

    return probs, mask_128, prob_orig, mask_orig
