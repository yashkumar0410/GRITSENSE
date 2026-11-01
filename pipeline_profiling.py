"""Opt-in timing helpers for the video-processing pipeline."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import os
import time

_enabled = os.environ.get("GRITSENSE_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


def enable(enabled: bool = True) -> None:
    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    return _enabled


def _synchronize(device) -> None:
    if device is None:
        return
    device_name = str(device).lower()
    if not isinstance(device, int) and not device_name.startswith(("xpu", "cuda")):
        return
    import torch

    if isinstance(device, int) and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_name.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize(device)
    elif device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def record(name: str, elapsed_seconds: float, count: int = 1) -> None:
    if not _enabled:
        return
    _totals[name] += elapsed_seconds
    _counts[name] += count


def measure(name: str, device=None):
    if not _enabled:
        return nullcontext()
    return _Measure(name, device)


class _Measure:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device

    def __enter__(self):
        _synchronize(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        try:
            if exc_info[0] is None:
                _synchronize(self.device)
        finally:
            record(self.name, time.perf_counter() - self.started)
        return False


def verified_model_device(model) -> str | None:
    """Return the model backend's selected execution device when available."""
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    backend_device = getattr(backend, "device", None)
    if backend_device is not None:
        return str(backend_device)
    predictor_device = getattr(predictor, "device", None)
    if predictor_device is not None:
        return str(predictor_device)
    parameters = getattr(backend, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return str(next(parameters()).device)
    except StopIteration:
        return None


def report_model_device(name: str, model, requested_device) -> None:
    actual_device = verified_model_device(model)
    if actual_device is None:
        print(f"{name} backend device: not verified (requested {requested_device})")
        return
    print(f"{name} backend device: {actual_device} (requested {requested_device})")
    expected_device = f"cuda:{requested_device}" if isinstance(requested_device, int) else str(requested_device)
    if expected_device == "xpu":
        expected_device = "xpu:0"
    if actual_device != expected_device:
        print(f"WARNING: {name} backend is not on the requested device ({expected_device}).")


def report(title: str) -> None:
    if not _enabled:
        return
    print(f"\n[profile] {title}")
    print("[profile] Nested measurements overlap; do not sum these rows.")
    print(f"{'measurement':34} {'total (s)':>11} {'calls':>9} {'avg (ms)':>11} {'fps':>9}")
    for name, elapsed in _totals.items():
        count = _counts[name]
        average_ms = elapsed * 1000 / count if count else 0.0
        fps = count / elapsed if elapsed > 0 and name.endswith((".frame_total", ".frame_average")) else 0.0
        fps_display = f"{fps:.2f}" if fps else "-"
        print(f"{name:34} {elapsed:11.3f} {count:9d} {average_ms:11.2f} {fps_display:>9}")
