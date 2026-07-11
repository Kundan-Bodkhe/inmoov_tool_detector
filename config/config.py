"""
Central configuration for the InMoov Mechanical Tool Detection project.

Every tunable parameter used across training, evaluation, and live webcam
inference lives here so the rest of the codebase never hard-codes values.
Edit this file (or override via CLI flags in train.py / evaluate.py /
webcam_detect.py) to change behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Project root (this file lives in <project_root>/config/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Single source of truth for all project settings."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = PROJECT_ROOT / "dataset"
    data_yaml: Path = PROJECT_ROOT / "dataset" / "data.yaml"
    models_dir: Path = PROJECT_ROOT / "models"
    runs_dir: Path = PROJECT_ROOT / "runs"
    logs_dir: Path = PROJECT_ROOT / "logs"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # Path to the weights used for inference / evaluation. After training,
    # point this at runs/detect/<experiment>/weights/best.pt, or leave the
    # default and copy best.pt into models/best.pt.
    weights_path: Path = PROJECT_ROOT / "models" / "best.pt"

    # Pretrained checkpoint used as the starting point for transfer learning.
    pretrained_weights: str = "yolo11n.pt"

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    epochs: int = 150
    patience: int = 25          # early stopping patience (epochs w/o improvement)
    batch_size: int = 16
    image_size: int = 640
    optimizer: str = "AdamW"    # SGD, Adam, AdamW, NAdam, RAdam, RMSProp, auto
    learning_rate: float = 0.001
    lr_final_factor: float = 0.01  # lrf: final LR = lr0 * lrf (cosine schedule)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    seed: int = 42
    workers: int = 2

    # Augmentation (Ultralytics native hyperparameters)
    augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1

    # ------------------------------------------------------------------
    # Inference / detection
    # ------------------------------------------------------------------
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45     # NMS IoU threshold
    max_detections: int = 50

    # ------------------------------------------------------------------
    # Webcam
    # ------------------------------------------------------------------
    camera_index: int = 0
    window_name: str = "InMoov Tool Detector"
    window_width: int = 1280
    window_height: int = 720
    show_fps: bool = True
    show_inference_time: bool = True
    flip_horizontal: bool = False   # set True if camera mirrors the image

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: str = "auto"  # "auto", "cpu", "0", "0,1", "mps"

    # ------------------------------------------------------------------
    # Visualization colors (BGR, OpenCV convention). Extend as needed;
    # unlisted classes fall back to `default_color`.
    # ------------------------------------------------------------------
    default_color: Tuple[int, int, int] = (0, 255, 0)
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "hammer": (0, 128, 255),
        "screwdriver": (255, 0, 0),
        "pliers": (0, 255, 255),
        "wrench": (255, 0, 255),
        "spanner": (255, 128, 0),
        "allen_key": (128, 255, 0),
        "tape_measure": (0, 255, 128),
        "scissors": (128, 0, 255),
    })

    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2

    def __post_init__(self) -> None:
        for d in (self.models_dir, self.runs_dir, self.logs_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Return a display color for a class name, case-insensitive."""
        return self.class_colors.get(class_name.lower(), self.default_color)


# Environment variable overrides let you tweak key values without editing
# code, e.g. `CAMERA_INDEX=1 python webcam_detect.py`
def _load_env_overrides(cfg: Config) -> Config:
    if os.getenv("CAMERA_INDEX") is not None:
        cfg.camera_index = int(os.getenv("CAMERA_INDEX"))
    if os.getenv("CONF_THRESHOLD") is not None:
        cfg.confidence_threshold = float(os.getenv("CONF_THRESHOLD"))
    if os.getenv("WEIGHTS_PATH") is not None:
        cfg.weights_path = Path(os.getenv("WEIGHTS_PATH"))
    if os.getenv("DEVICE") is not None:
        cfg.device = os.getenv("DEVICE")
    return cfg


config = _load_env_overrides(Config())
