"""Auto-detect the best available compute device (cuda > mps > cpu)."""

from __future__ import annotations

import torch


def get_best_device() -> str:
    """Return the best available PyTorch device string.

    Priority: cuda (NVIDIA GPU) > mps (Apple Silicon) > cpu.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
