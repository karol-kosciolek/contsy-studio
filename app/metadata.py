# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Oznaczanie plikow wyjsciowych jako wygenerowane przez AI.

Wersja podstawowa: zapis znacznikow do PNG (tEXt) i EXIF do JPEG. Pelne C2PA
(podpisany manifest) wymaga biblioteki `c2pa` - mozna je dodac pozniej; znaczniki
ponizej sa czytelne dla czlowieka i prostych narzedzi i spelniaja minimum
"oznacz tresc AI".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

try:
    from PIL import Image, PngImagePlugin
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import piexif
except Exception:  # pragma: no cover
    piexif = None  # type: ignore

AI_TAG = "AI-generated"
GENERATOR = "Contsy Studio"


def _tag_png(path: Path, info: Dict) -> None:
    img = Image.open(path)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("ai_generated", "true")
    meta.add_text("digital_source_type", "trainedAlgorithmicMedia")
    meta.add_text("generator", GENERATOR)
    meta.add_text("parameters", json.dumps(info, ensure_ascii=False))
    img.save(path, pnginfo=meta)


def _tag_jpeg(path: Path, info: Dict) -> None:
    if piexif is None:
        return
    try:
        exif = piexif.load(str(path))
    except Exception:
        exif = {"0th": {}, "Exif": {}, "1st": {}, "GPS": {}, "Interop": {}}
    exif["0th"][piexif.ImageIFD.Software] = GENERATOR.encode("utf-8")
    comment = f"{AI_TAG}; {json.dumps(info, ensure_ascii=False)}"
    exif["Exif"][piexif.ExifIFD.UserComment] = b"ASCII\x00\x00\x00" + \
        comment.encode("ascii", "ignore")
    piexif.insert(piexif.dump(exif), str(path))


def embed_metadata(path: str | Path, info: Dict) -> None:
    """Dodaje znaczniki AI do pliku graficznego (PNG/JPEG). Bledy nie przerywaja pracy."""
    if Image is None:
        return
    p = Path(path)
    try:
        suffix = p.suffix.lower()
        if suffix == ".png":
            _tag_png(p, info)
        elif suffix in (".jpg", ".jpeg"):
            _tag_jpeg(p, info)
    except Exception as e:  # noqa: BLE001 - oznaczanie nie moze wywrocic generacji
        print(f"[metadata] Could not label {p.name}: {e}", flush=True)
