# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Powiekszanie do 1080p - SeedVR2 3B (Apache 2.0, wbudowany w ComfyUI).

Wan 2.2 A14B jest uczony do 720p - generowanie wprost w 1080p psuje obraz (powtorzenia,
rozjechane twarze) i nie miesci sie w 16 GB. Dlatego wideo powstaje w 720p, a SeedVR2
w jednym kroku dorysowuje ostre szczegoly w 1080p (zamiast rozmywac jak zwykle skalowanie).
Krotszy bok zawsze 1080 px: 16:9 -> 1920x1080, 9:16 -> 1080x1920, 1:1 -> 1080x1080.
Gdy model jest niedostepny albo zabraknie pamieci, zostaje skalowanie lanczos w FFmpeg.
"""
from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Callable, Optional, Tuple

from . import model_manager, templates
from .comfy_client import ComfyClient

MODEL_ID = "seedvr2-3b"
SHORT_SIDE = 1080
_CHUNK = 21    # klatki na kawalek wideo (4n+1); auto na 16 GB dawalo po 1 klatce
_SMALL_CHUNK = 9  # GPUs under 15 GB: 21-frame chunks at 1080p peaked at 15.4 GB (measured)
_OVERLAP = 2  # klatki latentu wspolne dla sasiednich kawalkow (plynne przejscia)

ProgressCb = Callable[[float, str], None]


# Standardowe formaty -> dokladny rozmiar jak w obrazach (klatka Wan 3:4 to 832x1104,
# bez zaokraglenia wyszloby 1080x1434 zamiast 1080x1440)
_STANDARD = {16 / 9: (1920, 1080), 9 / 16: (1080, 1920), 1.0: (1080, 1080),
             3 / 4: (1080, 1440), 4 / 3: (1440, 1080)}


def target_size(width: int, height: int) -> Tuple[int, int]:
    """Rozmiar 1080p o tych samych proporcjach (parzyste wymiary dla kodeka wideo)."""
    for ratio, size in _STANDARD.items():
        if abs(width / height - ratio) < 0.01:
            return size
    scale = SHORT_SIDE / min(width, height)
    return int(round(width * scale / 2) * 2), int(round(height * scale / 2) * 2)


def _mapping(width: int, height: int) -> dict:
    return {"%UNET%": model_manager.find_file("diffusion_models", "seedvr2_3b"),
            "%VAE%": model_manager.find_file("vae", "seedvr2_ema_vae"),
            "%WIDTH%": width, "%HEIGHT%": height, "%SEED%": random.randint(0, 2 ** 31 - 1)}


def _lanczos(src: str, out: Path, width: int, height: int) -> Tuple[bool, str]:
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf",
                        f"scale={width}:{height}:flags=lanczos", "-c:v", "libx264", "-crf", "16",
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-an", str(out)],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-300:]


def video(client: ComfyClient, src: str, out_path: str | Path, width: int, height: int,
          fps: float, progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """Wideo (bez dzwieku) -> 1080p. Zwraca (ok, uwaga) - uwaga pusta, gdy SeedVR2 zadzialal."""
    out_path = Path(out_path)
    tw, th = target_size(width, height)
    if model_manager.is_installed(MODEL_ID):
        try:
            mapping = _mapping(tw, th)
            mapping.update({"%VIDEO%": client.upload_image(src), "%FPS%": fps,
                            "%OVERLAP%": _OVERLAP,
                            "%CHUNK%": _SMALL_CHUNK if model_manager.small_gpu() else _CHUNK})

            def ncb(node: str, value: int, maximum: int) -> None:
                if progress_cb and node == "10":
                    progress_cb(value / max(maximum, 1), "1080p (SeedVR2)")

            client.prepare_pipeline("upscale")
            result = client.run_and_wait(templates.build("seedvr2_video", mapping),
                                         node_progress_cb=ncb)
            # LoadVideo tez zwraca podglad (wejsciowe 720p) - bierzemy plik z VHS_VideoCombine
            item = next((it for it in result.files
                         if (it.get("filename") or "").startswith("aistudio_1080p")
                         and it["filename"].lower().endswith(".mp4")), None)
            if item is not None:
                out_path.write_bytes(client.download_output(item))
                return True, ""
            note = "SeedVR2 returned no video"
        except Exception as e:  # noqa: BLE001 - zostaje zwykle skalowanie
            note = f"SeedVR2 unavailable ({str(e)[:120]})"
    else:
        note = "SeedVR2 model missing"
    ok, msg = _lanczos(src, out_path, tw, th)
    return ok, f"1080p with standard scaling: {note}." if ok else f"Upscaling failed: {msg}"


def image(client: ComfyClient, src: str, out_path: str | Path,
          size: Optional[Tuple[int, int]] = None) -> Tuple[bool, str]:
    """Obraz -> 1080p (albo dokladnie `size`). Zwraca (ok, uwaga)."""
    from PIL import Image

    out_path = Path(out_path)
    with Image.open(src) as im:
        tw, th = size or target_size(im.width, im.height)
    if model_manager.is_installed(MODEL_ID):
        try:
            mapping = _mapping(tw, th)
            mapping["%IMAGE%"] = client.upload_image(src)
            client.prepare_pipeline("upscale")
            result = client.run_and_wait(templates.build("seedvr2_image", mapping))
            if result.images:
                out_path.write_bytes(result.images[0])
                return True, ""
            note = "SeedVR2 returned no image"
        except Exception as e:  # noqa: BLE001
            note = f"SeedVR2 unavailable ({str(e)[:120]})"
    else:
        note = "SeedVR2 model missing"
    with Image.open(src) as im:
        im.convert("RGB").resize((tw, th), Image.LANCZOS).save(out_path)
    return True, f"1080p with standard scaling: {note}."
