"""TensorRT FP16 model export utilities for YOLO inference acceleration.

On RTX 5090 (SM 12.0, 680 5th-gen Tensor Cores), TensorRT FP16 provides
2-4x inference speedup over standard PyTorch.

Usage:
    model_path = ensure_tensorrt_engine("yolo11x-pose.pt")
    # Returns "yolo11x-pose.engine" if TensorRT export succeeds,
    # otherwise returns the original "yolo11x-pose.pt"
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def ensure_tensorrt_engine(model_name: str) -> str:
    """Ensure a TensorRT FP16 engine exists for the given YOLO model.

    If a .engine file already exists alongside the .pt file, returns the
    engine path. Otherwise attempts a one-time export (takes 2-5 min).
    Falls back to the original .pt path if export fails.

    Args:
        model_name: Path to the YOLO .pt model file.

    Returns:
        Path string to use for YOLO() constructor — either .engine or .pt.
    """
    if not torch.cuda.is_available():
        return model_name

    pt_path = Path(model_name)
    if pt_path.suffix != ".pt":
        return model_name

    engine_path = pt_path.with_suffix(".engine")
    if engine_path.exists():
        try:
            import tensorrt  # noqa: F401
            logger.info("TensorRT engine found: %s", engine_path)
            return str(engine_path)
        except ImportError:
            logger.warning("TensorRT engine %s exists but tensorrt not installed, using .pt", engine_path)
            return model_name

    try:
        from ultralytics import YOLO

        logger.info(
            "Exporting TensorRT FP16 engine from %s (one-time cost, ~2-5 min)...",
            model_name,
        )
        base_model = YOLO(model_name)
        export_path = base_model.export(format="engine", half=True)
        logger.info("TensorRT export complete: %s", export_path)
        return str(export_path)
    except Exception as exc:
        logger.warning(
            "TensorRT export failed (%s), using PyTorch model. "
            "Install tensorrt for faster inference: pip install tensorrt",
            exc,
        )
        return model_name
