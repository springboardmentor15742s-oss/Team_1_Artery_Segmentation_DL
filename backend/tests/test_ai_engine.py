"""Unit and Integration Tests for AI Inference Engine Service.

Verifies:
1. Standard U-Net checkpoint loading.
2. Attention U-Net checkpoint loading.
3. Model caching mechanism (singleton persistence).
4. CPU single-slice inference forward pass.
5. Exact input preprocessing and HU normalization.
6. Probability map values in [0.0, 1.0] and binary mask values in {0, 255}.
7. Invalid input error handling (dimensions, non-finite values, unknown model).
8. Real test slice validation on ImageCAS test data.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image

from app.services.ai_engine import ai_engine
from app.services.analysis_service import run_case_analysis
from app.ml.config import config
from app.ml.preprocessing import normalize_to_model_hu, prepare_slice_tensor, postprocess_prediction


def test_checkpoint_files_exist():
    """Verify that both integrated model weights are present in backend/app/ml/checkpoints."""
    att_path = config.paths.checkpoints_dir / "attention_unet_opt_10ep_best.pth"
    unet_path = config.paths.checkpoints_dir / "unet_opt_10ep_best.pth"
    assert att_path.exists(), f"Missing checkpoint: {att_path}"
    assert unet_path.exists(), f"Missing checkpoint: {unet_path}"
    assert att_path.stat().st_size > 1_000_000, "Attention U-Net checkpoint file size suspicious"
    assert unet_path.stat().st_size > 1_000_000, "Standard U-Net checkpoint file size suspicious"


def test_load_models_and_caching():
    """Verify loading checkpoints into models and confirming in-memory cache."""
    # 1. Load Attention U-Net
    model_att = ai_engine.load_model("attention_unet")
    assert model_att is not None
    assert not model_att.training, "Model must be in eval mode"

    # Verify cache hit
    model_att_2 = ai_engine.load_model("attention_unet")
    assert model_att is model_att_2, "Model must be cached in memory"

    # 2. Load Standard U-Net
    model_unet = ai_engine.load_model("unet")
    assert model_unet is not None
    assert not model_unet.training, "Model must be in eval mode"
    assert model_unet is not model_att, "Different models must have distinct instances"


def test_preprocessing_normalization():
    """Test HU windowing and harmonization between platform and model."""
    # Raw HU CT slice
    raw_hu = np.array([[-1024.0, -100.0, 300.0, 700.0, 1500.0]], dtype=np.float32).repeat(10, axis=0)
    norm = normalize_to_model_hu(raw_hu, source="raw_hu")
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert np.isclose(norm[0, 1], 0.0)  # -100 HU maps to 0.0
    assert np.isclose(norm[0, 3], 1.0)  # 700 HU maps to 1.0
    assert np.isclose(norm[0, 2], 0.5)  # 300 HU maps to (300 - (-100))/800 = 0.5

    # Platform normalized slice (0.0 to 1.0 representing -200 to 800 HU)
    # HU 300 corresponds to (300 - (-200))/(800 - (-200)) = 500/1000 = 0.5 in platform
    plat_val = np.full((10, 10), 0.5, dtype=np.float32)
    norm_from_plat = normalize_to_model_hu(plat_val, source="platform_normalized")
    assert np.allclose(norm_from_plat, 0.5)


def test_prepare_slice_tensor():
    """Verify tensor shape, dtype and dimensions."""
    dummy_slice = np.random.uniform(0.0, 255.0, size=(256, 256)).astype(np.float32)
    tensor, orig_shape = prepare_slice_tensor(dummy_slice, source="image_uint8")
    assert orig_shape == (256, 256)
    assert tensor.shape == (1, 1, 128, 128)
    assert tensor.dtype.is_floating_point


def test_predict_slice_attention_unet():
    """Test single-slice inference using Attention U-Net on synthetic input."""
    dummy_slice = np.random.uniform(-100.0, 700.0, size=(128, 128)).astype(np.float32)
    res = ai_engine.predict_slice(dummy_slice, model_name="attention_unet", threshold=0.50)

    assert res["model_name"] == "attention_unet"
    assert res["device"] == "cpu"
    assert res["inference_duration_ms"] > 0
    assert res["probability_map_128"].shape == (128, 128)
    assert res["mask_128"].shape == (128, 128)
    assert res["probability_map_128"].min() >= 0.0
    assert res["probability_map_128"].max() <= 1.0
    # Binary mask must contain only 0 and 255
    unique_vals = set(np.unique(res["mask_128"]))
    assert unique_vals.issubset({0, 255})


def test_prediction_contract_contains_native_resolution_fields():
    """Runtime prediction output must expose native-resolution mask/probability fields."""
    dummy_slice = np.random.uniform(-100.0, 700.0, size=(128, 128)).astype(np.float32)
    res = ai_engine.predict_slice(dummy_slice, model_name="attention_unet", threshold=0.50)

    for field in ("mask_native", "mask_128", "probability_map_native", "probability_map_128", "inference_duration_ms"):
        assert field in res, f"Missing runtime field: {field}"
    assert res["mask_native"].shape == dummy_slice.shape
    assert res["probability_map_native"].shape == dummy_slice.shape


def test_run_case_analysis_uses_native_mask_and_persists_overlay(tmp_path):
    """The orchestration layer must consume the native mask, save it, and store overlay paths."""
    volume = np.zeros((12, 32, 32), dtype=np.float32)
    volume[:, 10:20, 10:20] = 1.0
    path = tmp_path / "sample.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(volume), str(path))

    image = SimpleNamespace(
        case_id=7,
        id=42,
        storage_key="abc123",
        processed_path=str(path),
        processed_spacing_mm=[0.5, 0.5, 1.0],
        voxel_spacing_mm=[0.5, 0.5, 1.0],
    )

    native_mask = np.zeros((32, 32), dtype=np.uint8)
    native_mask[12:18, 12:18] = 255
    prediction = {
        "mask_native": native_mask,
        "mask_128": np.zeros((128, 128), dtype=np.uint8),
        "probability_map_native": (native_mask > 0).astype(np.float32),
        "probability_map_128": np.zeros((128, 128), dtype=np.float32),
        "inference_duration_ms": 22.5,
    }

    with patch("app.services.analysis_service.ai_engine.predict_slice", return_value=prediction):
        result = run_case_analysis(image, slice_index=4, model_name="attention_unet", user_id=1, db=MagicMock())

    assert result.status == "success"
    assert Path(result.mask_path).exists()
    assert Path(result.overlay_path).exists()
    assert Image.open(result.mask_path).size == (32, 32)
    assert Image.open(result.overlay_path).size == (32, 32)
    assert result.foreground_pixels == 36


def test_run_case_analysis_raises_for_missing_native_mask(tmp_path):
    """Malformed prediction output should fail safely instead of silently substituting a blank mask."""
    volume = np.zeros((12, 32, 32), dtype=np.float32)
    path = tmp_path / "sample.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(volume), str(path))

    image = SimpleNamespace(
        case_id=7,
        id=43,
        storage_key="xyz789",
        processed_path=str(path),
        processed_spacing_mm=[0.5, 0.5, 1.0],
        voxel_spacing_mm=[0.5, 0.5, 1.0],
    )

    missing_mask_prediction = {
        "mask_128": np.zeros((128, 128), dtype=np.uint8),
        "probability_map_128": np.zeros((128, 128), dtype=np.float32),
        "inference_duration_ms": 5.0,
    }

    with patch("app.services.analysis_service.ai_engine.predict_slice", return_value=missing_mask_prediction):
        result = run_case_analysis(image, slice_index=4, model_name="attention_unet", user_id=1, db=MagicMock())

    assert result.status == "failed"
    assert "mask_native" in (result.error_message or "")


def test_predict_slice_standard_unet():
    """Test single-slice inference using Standard U-Net on synthetic input."""
    dummy_slice = np.random.uniform(-100.0, 700.0, size=(128, 128)).astype(np.float32)
    res = ai_engine.predict_slice(dummy_slice, model_name="unet", threshold=0.50)

    assert res["model_name"] == "unet"
    assert res["probability_map_128"].shape == (128, 128)
    assert res["mask_128"].shape == (128, 128)


def test_invalid_input_handling():
    """Verify that invalid inputs raise appropriate ValueErrors."""
    # 1D array
    with pytest.raises(ValueError, match="2D slice"):
        ai_engine.predict_slice(np.ones(100))

    # 3D array
    with pytest.raises(ValueError, match="2D slice"):
        ai_engine.predict_slice(np.ones((10, 10, 10)))

    # NaN input
    with pytest.raises(ValueError, match="NaN or infinite"):
        bad = np.ones((64, 64), dtype=np.float32)
        bad[5, 5] = np.nan
        ai_engine.predict_slice(bad)

    # Unknown model name
    with pytest.raises(ValueError, match="Unsupported model"):
        ai_engine.predict_slice(np.ones((64, 64)), model_name="resnet50")


def test_inference_on_real_sample_slice():
    """Test inference on actual ImageCAS test slice case_014_slice_084.png."""
    sample_path = Path(r"C:\Users\somes\Documents\Model\processed_data\test\images\case_014_slice_084.png")
    if not sample_path.exists():
        pytest.skip("Test sample slice not available at expected path.")

    im = Image.open(sample_path).convert("L")
    res_att = ai_engine.predict_slice(im, model_name="attention_unet", threshold=0.50)
    res_unet = ai_engine.predict_slice(im, model_name="unet", threshold=0.50)

    # Verify both models detect coronary vessels on slice 084
    assert res_att["foreground_pixels_128"] > 0, "Attention U-Net should detect vessel on test slice 084"
    assert res_unet["foreground_pixels_128"] > 0, "Standard U-Net should detect vessel on test slice 084"
    print(f"\n[Real Sample Test] Attention U-Net: {res_att['foreground_pixels_128']} px in {res_att['inference_duration_ms']:.1f} ms")
    print(f"[Real Sample Test] Standard U-Net:  {res_unet['foreground_pixels_128']} px in {res_unet['inference_duration_ms']:.1f} ms")
