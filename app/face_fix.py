# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Poprawka twarzy postaci z wlasna LoRA (jak ADetailer).

Twarz glownej osoby (najwieksza w kadrze, MediaPipe Face Landmarker - Apache 2.0) wycinamy
z zapasem, powiekszamy do 1024 px, przerysowujemy Z-Image + LoRA postaci przy sredniej sile
i wklejamy z miekka krawedzia - reszta kadru zostaje nietknieta. Uzywane tam, gdzie twarz
postaci nie pochodzi wprost z LoRA:
  - obrazy skladane ze zdjeciami pomocniczymi (Qwen-Image-Edit upraszczal rysy),
  - klatki startowe ujec skladanych ze zdjeciami,
  - poczatek kolejnych czesci dlugiego filmu (ostatnia klatka poprzedniej czesci) - bez tego
    twarz mogla po kilku czesciach odplynac od postaci.
"""
from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Optional, Tuple

from . import config, prefs, workflows

LANDMARKER = config.ROOT / "models" / "mediapipe" / "face_landmarker.task"
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")
CROP_SIZE = 1024     # twarz przerysowywana w tej rozdzielczosci (szczegoly jak na portrecie)
MARGIN = 2.2         # wycinek = rozmiar twarzy x MARGIN (czolo, wlosy, broda w kontekscie)
MIN_FACE = 24        # mniejszych twarzy nie ruszamy (daleki plan - LoRA i tak nie pomoze)
_DETECTOR = None
_LOCK = threading.RLock()  # main_face trzyma blokade i wola _detector() - musi byc RLock


def _detector():
    global _DETECTOR
    with _LOCK:
        if _DETECTOR is None:
            import mediapipe as mp

            if not LANDMARKER.exists():
                import urllib.request

                LANDMARKER.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER)
            _DETECTOR = mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=5))
        return _DETECTOR


def main_face(image) -> Optional[Tuple[int, int, int, int]]:
    """Ramka (x0, y0, x1, y1) najwiekszej twarzy na obrazie PIL albo None."""
    import mediapipe as mp
    import numpy as np

    rgb = np.ascontiguousarray(np.array(image.convert("RGB")))
    h, w = rgb.shape[:2]
    with _LOCK:
        res = _detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    boxes = []
    for lm in res.face_landmarks:
        xs, ys = [p.x * w for p in lm], [p.y * h for p in lm]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    if not boxes:
        return None
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return box if min(box[2] - box[0], box[3] - box[1]) >= MIN_FACE else None


def fix(client, ch: dict, src: str | Path, out_path: str | Path, denoise: float = 0.5,
        seed: Optional[int] = None, text: str = "") -> Tuple[bool, str]:
    """Twarz postaci na obrazie src -> out_path. (False, powod) gdy nie ma czego poprawiac
    (brak twarzy, twarz za mala) albo silnik zawiodl - wtedy uzyj obrazu src bez zmian.
    text - opis sceny (mina, swiatlo), zeby poprawka nie zmieniala wyrazu twarzy."""
    from PIL import Image, ImageDraw, ImageFilter

    from . import characters, image_gen

    try:
        image = Image.open(src).convert("RGB")
        box = main_face(image)
    except Exception as e:  # noqa: BLE001 - brak MediaPipe/modelu = bez poprawki
        return False, f"Face detection failed ({e})"
    if box is None:
        return False, "No clear face was found"
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(min(max(x1 - x0, y1 - y0) * MARGIN, image.width, image.height))
    left = int(min(max(cx - side / 2, 0), image.width - side))
    top = int(min(max(cy - side / 2, 0), image.height - side))
    crop = image.crop((left, top, left + side, top + side)).resize((CROP_SIZE, CROP_SIZE),
                                                                    Image.LANCZOS)
    tmp = Path(out_path).with_name(Path(out_path).stem + "_face.png")
    crop.save(tmp)
    models = [m for m in image_gen.available_models() if workflows.detect_family(m) == "zimage"]
    if not models:
        tmp.unlink(missing_ok=True)
        return False, "Z-Image-Turbo is required to restore character identity."
    seed = random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)
    try:
        wf = workflows.img2img(image_gen.resolve_spec(models[0]), client.upload_image(str(tmp)),
                               prefs.styled(characters.positive(
                                   ch, "close-up photo of her face, natural skin texture, "
                                       "detailed eyes" + (f". Scene: {text}" if text else "")),
                                   "image"),
                               "", denoise, seed, prefs.IMAGE_STEPS, 1.0,
                               loras=characters.loras(ch))
        wf["9"]["inputs"]["filename_prefix"] = "aistudio_face"
        client.prepare_pipeline("image")
        result = client.run_and_wait(wf)
    except Exception as e:  # noqa: BLE001
        return False, f"Engine: {e}"
    finally:
        tmp.unlink(missing_ok=True)
    if not result.images:
        return False, "The engine returned no image"
    import io

    face = Image.open(io.BytesIO(result.images[0])).convert("RGB").resize((side, side),
                                                                            Image.LANCZOS)
    # miekka elipsa: srodek z poprawki, brzegi wycinka z oryginalu (bez widocznego szwu)
    feather = max(6, side // 10)
    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).ellipse([feather, feather, side - feather, side - feather], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather / 2))
    image.paste(face, (left, top), mask)
    image.save(out_path)
    return True, str(out_path)
