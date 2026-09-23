"""Reproducibility and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a torch device.

    Args:
        preference: ``auto``, ``cpu``, ``cuda`` or ``mps``. ``auto`` picks CUDA,
            then Apple MPS, then CPU. An unavailable explicit choice falls back
            to CPU instead of crashing.
    """
    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if preference == "auto":
        return torch.device("cuda" if cuda_ok else "mps" if mps_ok else "cpu")
    if preference == "cuda" and cuda_ok:
        return torch.device("cuda")
    if preference == "mps" and mps_ok:
        return torch.device("mps")
    return torch.device("cpu")
