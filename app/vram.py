# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wykrywanie i raportowanie pamieci GPU (VRAM).

Kolejnosc prob: nvidia-ml-py (pynvml) -> torch.cuda -> no GPU.
Uzywane do: panelu statusu, ostrzezen o braku VRAM i doboru wariantow modeli.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuInfo:
    available: bool
    name: str = "no GPU"
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0

    def summary(self) -> str:
        if not self.available:
            return "GPU: not detected (CPU mode; generation will be very slow)"
        return (
            f"GPU: {self.name} | VRAM: {self.free_gb:.1f} GB free "
            f"of {self.total_gb:.1f} GB"
        )


def _via_pynvml() -> Optional[GpuInfo]:
    try:
        import pynvml  # nvidia-ml-py
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gb = 1024 ** 3
        return GpuInfo(
            available=True,
            name=name,
            total_gb=mem.total / gb,
            used_gb=mem.used / gb,
            free_gb=mem.free / gb,
        )
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _via_torch() -> Optional[GpuInfo]:
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        gb = 1024 ** 3
        total = props.total_memory / gb
        reserved = torch.cuda.memory_reserved(idx) / gb
        return GpuInfo(
            available=True,
            name=props.name,
            total_gb=total,
            used_gb=reserved,
            free_gb=max(0.0, total - reserved),
        )
    except Exception:
        return None


def get_gpu_info() -> GpuInfo:
    """Zwraca aktualny stan GPU (odswieza sie przy kazdym wywolaniu)."""
    return _via_pynvml() or _via_torch() or GpuInfo(available=False)


def can_fit(required_gb: float, reserve_gb: float = 1.0) -> bool:
    """Czy model o danym zapotrzebowaniu VRAM zmiesci sie z zapasem."""
    info = get_gpu_info()
    if not info.available:
        return False
    return (info.total_gb - reserve_gb) >= required_gb


def low_vram_hint(required_gb: float) -> str:
    """Podpowiedz przy zbyt malym VRAM."""
    info = get_gpu_info()
    if not info.available:
        return (
            "No NVIDIA GPU detected. Check your driver and CUDA-enabled PyTorch installation "
            "(install.bat). A supported GPU is required for practical generation."
        )
    return (
        f"The model requires approximately {required_gb:.0f} GB VRAM; your GPU has "
        f"{info.total_gb:.1f} GB. Try Fast mode, a shorter clip or fewer "
        f"frames, and close other GPU applications (games, OBS or hardware-accelerated browsers)."
    )
