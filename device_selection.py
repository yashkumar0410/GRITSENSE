"""Shared CPU, NVIDIA CUDA, and Intel XPU device selection."""

from __future__ import annotations

from typing import Literal

DeviceChoice = Literal["auto", "cpu", "cuda", "xpu"]
DEVICE_CHOICES = ("auto", "cpu", "cuda", "xpu")


def resolve_device(choice: str = "auto"):
    """Resolve a requested device, raising instead of silently falling back."""
    import torch

    if isinstance(choice, torch.device):
        if choice.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"NVIDIA CUDA device {choice} is unavailable.")
        if choice.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
            raise RuntimeError(f"Intel XPU device {choice} is unavailable.")
        return choice
    choice = choice.strip().lower()
    if choice in ("intel", "arc", "xpu:0"):
        choice = "xpu"
    elif choice == "cuda:0":
        choice = "cuda"
    if choice not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device {choice!r}; choose one of {', '.join(DEVICE_CHOICES)}.")
    if choice == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA was selected, but PyTorch reports no CUDA GPU available.")
        return torch.device("cuda:0")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU was selected, but PyTorch reports no Intel GPU available.")
    return torch.device("xpu:0")


def ultralytics_device(device):
    """Convert torch.device to the device value expected by Ultralytics."""
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return str(device)

