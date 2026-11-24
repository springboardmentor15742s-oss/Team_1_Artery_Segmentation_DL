"""AI Inference Engine Service.

Provides a thread-safe, cached singleton inference engine for 2D artery segmentation.
Supports both Attention U-Net (default) and Standard U-Net executing on CPU.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from app.ml.config import config
from app.ml.models import build_model
from app.ml.preprocessing import postprocess_prediction, prepare_slice_tensor

log = logging.getLogger("artery.ai_engine")


class AIEngine:
    """In-memory cached AI inference engine for coronary artery segmentation."""

    _models: Dict[str, torch.nn.Module] = {}
    _device: str = "cpu"

    @classmethod
    def get_supported_models(cls) -> list[str]:
        return list(config.model.available_models)

    @classmethod
    def load_model(cls, model_name: str = "attention_unet") -> torch.nn.Module:
        """Load and cache model weights in memory on CPU.

        Subsequent calls return the cached model instance immediately.
        """
        model_key = model_name.lower().replace("-", "_")
        if model_key in cls._models:
            return cls._models[model_key]

        if model_key not in config.model.checkpoints:
            raise ValueError(
                f"Unsupported model '{model_name}'. Available: {list(config.model.checkpoints.keys())}"
            )

        ckpt_filename = config.model.checkpoints[model_key]
        ckpt_path = config.paths.checkpoints_dir / ckpt_filename

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt_path}. Ensure model weights are present."
            )

        log.info("Loading %s checkpoint from %s onto CPU...", model_key, ckpt_path.name)
        t0 = time.perf_counter()

        # Build architecture
        model = build_model(model_key, base=config.model.base_filters)

        # Load weights on CPU
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)

        # Set evaluation mode
        model.eval()

        # CPU thread limits
        torch.set_num_threads(config.hardware.num_threads)

        cls._models[model_key] = model
        duration = (time.perf_counter() - t0) * 1000.0
        log.info("Successfully loaded and cached %s in %.1f ms", model_key, duration)
        return model

    @classmethod
    def preload_defaults(cls) -> None:
        """Pre-load the default Attention U-Net model during application startup."""
        cls.load_model(config.model.default_model)

    @classmethod
    def predict_slice(
        cls,
        slice_data: Union[np.ndarray, Image.Image],
        model_name: str = "attention_unet",
        threshold: float = 0.50,
        source: Literal["raw_hu", "platform_normalized", "image_uint8", "auto"] = "auto",
        return_native_resolution: bool = True,
    ) -> Dict[str, Any]:
        """Execute single-slice deep learning inference.

        Parameters:
            slice_data: 2D numpy array or PIL Image representing an axial CT slice.
            model_name: 'attention_unet' (default) or 'unet'.
            threshold: Decision probability threshold tau in [0.0, 1.0].
            source: Source intensity range ('raw_hu', 'platform_normalized', 'image_uint8', 'auto').
            return_native_resolution: Whether to resample outputs back to original slice dimensions.

        Returns:
            Dictionary containing binary masks, probability maps, duration, and metadata.
        """
        if isinstance(slice_data, Image.Image):
            arr = np.asarray(slice_data.convert("L"), dtype=np.float32)
        else:
            arr = np.asarray(slice_data, dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(f"Inference requires a 2D slice, but got array with shape {arr.shape}")
        if min(arr.shape) < 8:
            raise ValueError(f"Input slice dimensions {arr.shape} are too small for segmentation.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input slice contains NaN or infinite values.")

        # 1. Fetch cached model
        model = cls.load_model(model_name)

        # 2. Input tensor preparation (HU windowing and 128x128 resize)
        tensor, orig_shape = prepare_slice_tensor(arr, source=source)

        # 3. CPU forward pass under torch.inference_mode()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(tensor)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Post-processing (probability mapping and thresholding)
        target_orig = orig_shape if return_native_resolution else None
        prob_128, mask_128, prob_orig, mask_orig = postprocess_prediction(
            logits, original_shape=target_orig, threshold=threshold
        )

        fg_128 = int((mask_128 > 0).sum())
        fg_native = int((mask_orig > 0).sum()) if mask_orig is not None else fg_128

        return {
            "model_name": model_name,
            "threshold": threshold,
            "device": cls._device,
            "inference_duration_ms": round(inference_time_ms, 2),
            "input_dimensions": {
                "original_shape": [int(s) for s in orig_shape],
                "model_shape": [config.preprocessing.image_size, config.preprocessing.image_size],
            },
            "foreground_pixels_128": fg_128,
            "foreground_pixels_native": fg_native,
            "mask_128": mask_128,
            "probability_map_128": prob_128,
            "mask_native": mask_orig if mask_orig is not None else mask_128,
            "probability_map_native": prob_orig if prob_orig is not None else prob_128,
        }


# Convenience module-level singleton instance
ai_engine = AIEngine
