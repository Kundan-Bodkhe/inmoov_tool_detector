"""
Automatic compute-device selection (CUDA GPU -> Apple MPS -> CPU).
"""

from __future__ import annotations

from typing import Tuple

import torch


def get_device(preferred: str = "auto") -> str:
    """
    Resolve the best available device string for Ultralytics/PyTorch.

    Args:
        preferred: "auto" to autodetect, or an explicit device string
                   ("cpu", "0", "0,1", "mps").

    Returns:
        A device string usable directly by Ultralytics YOLO(..., device=...).
    """
    if preferred != "auto":
        return preferred

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """Return a human-readable description of the selected device."""
    if device == "cpu":
        return "CPU"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    if device.replace(",", "").isdigit():
        try:
            idx = int(device.split(",")[0])
            name = torch.cuda.get_device_name(idx)
            mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            return f"GPU {device} - {name} ({mem_gb:.1f} GB)"
        except Exception:
            return f"GPU {device}"
    return device


def gpu_memory_usage() -> Tuple[float, float]:
    """Return (allocated_gb, reserved_gb) for the current CUDA device, or (0, 0)."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved
