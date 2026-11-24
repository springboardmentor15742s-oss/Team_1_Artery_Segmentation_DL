"""Centralized AI/ML Engine Configuration.

Optimized for low-memory CPU execution (4 GB RAM Windows environment).
Defines model parameters, HU windows, thresholds, and analysis tiers.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


ML_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ML_DIR / "checkpoints"


@dataclass
class HardwareConfig:
    device: str = "cpu"
    num_threads: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    batch_size: int = 1


@dataclass
class PathConfig:
    ml_root: Path = ML_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    analysis_dir: Path = ML_DIR / "output" / "analysis"
    reports_dir: Path = ML_DIR / "output" / "reports"
    predictions_dir: Path = ML_DIR / "output" / "predictions"
    legacy_outputs_dir: Path = ML_DIR / "output"
    test_dir: Path = ML_DIR / "output"

    def create_directories(self) -> None:
        for d in (self.checkpoints_dir, self.analysis_dir, self.reports_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessingConfig:
    image_size: int = 128
    hu_min: float = -100.0
    hu_max: float = 700.0
    normalization_method: str = "minmax"
    filter_empty_slices: bool = True
    background_slice_ratio: float = 0.10
    random_seed: int = 42


@dataclass
class ModelConfig:
    default_model: str = "attention_unet"
    available_models: List[str] = field(default_factory=lambda: ["attention_unet", "unet"])
    in_channels: int = 1
    out_channels: int = 1
    base_filters: int = 16
    prediction_threshold: float = 0.50

    checkpoints: Dict[str, str] = field(
        default_factory=lambda: {
            "attention_unet": "attention_unet_opt_10ep_best.pth",
            "unet": "unet_opt_10ep_best.pth",
        }
    )


@dataclass
class AnalysisConfig:
    min_component_size: int = 0  # 0 preserves raw distal vessels
    centerline_algorithm: str = "skeletonize"
    reference_percentile: float = 90.0
    min_centerline_len: int = 10  # minimum skeleton px to avoid discretization noise

    severity_bins: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "Normal / Insignificant": (0.0, 25.0),
            "Mild Stenosis": (25.0, 50.0),
            "Moderate Stenosis": (50.0, 70.0),
            "Severe Stenosis": (70.0, 99.0),
            "Total Occlusion": (99.0, 100.0),
        }
    )


class MLAppConfig:
    def __init__(self):
        self.hardware = HardwareConfig()
        self.paths = PathConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.analysis = AnalysisConfig()

    def setup_environment(self) -> None:
        try:
            import torch
            torch.set_num_threads(self.hardware.num_threads)
        except Exception:
            pass
        self.paths.create_directories()


config = MLAppConfig()
