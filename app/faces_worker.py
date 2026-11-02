# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Proces roboczy: twarze w wideo dla ruchu ust wielu postaci (uruchamiany przez lipsync.py).

MediaPipe Face Landmarker (Apache 2.0) wykrywa do 10 twarzy w kazdej klatce, twarze
laczymy w "slady" po pokryciu ramek miedzy klatkami, a plec kazdej postaci (do
dobrania glosu z dialogu) ocenia CLIP ViT-B/32 (OpenAI, MIT) bez douczania - porownujac
twarz z opisami "a photo of a man" / "a photo of a woman". Wynik: JSON ze sladami.

Uzycie: faces_worker.py <wideo> <wynik.json>
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKER = os.path.join(ROOT, "models", "mediapipe", "face_landmarker.task")
CLIP_DIR = os.path.join(ROOT, "models", "clip-vit-base-patch32")
sys.path.append(ROOT)
from app.aux_models import ensure_clip, ensure_landmarker
MIN_PRESENCE = 0.4   # slad musi byc widoczny w co najmniej 40% klatek

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detect(video):
    import mediapipe as mp

    ensure_landmarker()
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=10))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracks, frames, n = [], {}, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        for lm in res.face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in lm])
            box = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                   float(pts[:, 0].max()), float(pts[:, 1].max())]
            best = max(tracks, key=lambda t: _iou(t["last"], box), default=None)
            if best is not None and best["seen_at"] != n and _iou(best["last"], box) > 0.2:
                track = best
            else:
                track = {"boxes": [], "last": box, "seen_at": -1}
                tracks.append(track)
            track["boxes"].append((n, box))
            track["last"], track["seen_at"] = box, n
            if box[2] - box[0] > frames.get(id(track), (0, None, None))[0]:
                frames[id(track)] = (box[2] - box[0], n, rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy())
        n += 1
    cap.release()
    out = []
    for t in tracks:
        if len(t["boxes"]) < MIN_PRESENCE * max(n, 1):
            continue
        b = np.array([bx for _, bx in t["boxes"]])
        out.append({"presence": round(len(t["boxes"]) / n, 2),
                    "bbox": [float(b[:, 0].min()), float(b[:, 1].min()),
                             float(b[:, 2].max()), float(b[:, 3].max())],
                    "size": float(np.median(b[:, 2] - b[:, 0])),
                    "cx": float(np.median((b[:, 0] + b[:, 2]) / 2)),
                    "crop": frames[id(t)][2]})
    return out, n, fps, (w if n else 0), (h if n else 0)


def genders(crops):
    """Prawdopodobienstwo 'kobieta' dla kazdej twarzy (CLIP zero-shot)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    ensure_clip()
    model = CLIPModel.from_pretrained(CLIP_DIR).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR)
    texts = ["a close-up photo of a man's face", "a close-up photo of a woman's face"]
    with torch.no_grad():
        inp = proc(text=texts, images=list(crops), return_tensors="pt", padding=True)
        probs = model(**inp).logits_per_image.softmax(dim=1)[:, 1]
    return [float(p) for p in probs]


def main():
    video, out_json = sys.argv[1], sys.argv[2]
    tracks, n, fps, w, h = detect(video)
    if tracks:
        for t, p in zip(tracks, genders([t["crop"] for t in tracks])):
            t["p_female"] = round(p, 3)
            t["gender"] = "female" if p >= 0.5 else "male"
    for t in tracks:
        del t["crop"]
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"frames": n, "fps": fps, "width": w, "height": h, "tracks": tracks}, fh)
    print(f"Faces detected: {len(tracks)}", flush=True)


if __name__ == "__main__":
    main()
