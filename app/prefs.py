# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Ustawienia uzytkownika (autozapis) i staly styl obrazu.

Styl jest jeden: ultra realistyczny obraz, wideo 1080p (Wan 2.2 + SeedVR2; tryb Jakosc albo
Szybki - app/video_gen.py MODES), obrazy 1080p. Do promptu dopisujemy styl, zeby wynik zawsze
wygladal jak prawdziwe zdjecie / nagranie. Autozapis zapisywany w config/user_prefs.json.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Dict, List

from . import config
from .persistence import atomic_text

_FILE = config.CONFIG_DIR / "user_prefs.json"
_LOCK = threading.RLock()

STYLE = {
    "video": "ultra realistic, photorealistic real life footage, natural lighting, cinematic, "
             "highly detailed, sharp focus",
    "image": "ultra realistic photo, photorealistic, natural light, highly detailed, "
             "sharp focus",
}
IMAGE_STEPS = 8   # Z-Image-Turbo uczony na 8 krokach

# character: slug postaci; qc: kontrola jakosci wynikow postaci (app/qc.py);
# autopilot: sesje modelek planowane w tle (app/autopilot.py) - wlaczany w zakladce Studio;
# tier: model set "full" or "lite" ("" = chosen from the GPU, see model_manager.tier)
_DEFAULTS = {"autosave": False, "folder": "", "character": "", "qc": True, "autopilot": False,
             "tier": ""}


def load() -> Dict:
    try:
        data = json.loads(_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {**_DEFAULTS, **{k: v for k, v in (data if isinstance(data, dict) else {}).items() if k in _DEFAULTS}}


def save(**changes) -> Dict:
    with _LOCK:
        data = {**load(), **{k: v for k, v in changes.items() if k in _DEFAULTS}}
        atomic_text(_FILE, json.dumps(data, ensure_ascii=False, indent=2))
        return data


def styled(prompt: str, kind: str) -> str:
    """Prompt z dopisanym stylem ultra realistycznym (kind: 'video' / 'image')."""
    return f"{prompt.strip().rstrip('.')}, {STYLE[kind]}" if prompt and prompt.strip() else prompt


def autosave(files: List[str]) -> str:
    """Kopiuje gotowe pliki do folderu autozapisu (gdy wlaczony). Zwraca opis albo ''."""
    p = load()
    if not p["autosave"] or not p["folder"] or not files:
        return ""
    folder = Path(p["folder"])
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = folder / Path(f).name
            if target.resolve() == Path(f).resolve():
                continue
            number = 1
            while target.exists():
                number += 1
                target = folder / f"{Path(f).stem}_{number}{Path(f).suffix}"
            shutil.copy2(f, target)
    except OSError as e:
        return f"Autosave failed: {e}"
    return f"Saved to folder: {folder}"


def choose_folder(current: str = "") -> str:
    """Systemowe okno wyboru folderu (aplikacja dziala lokalnie, wiec okno pojawia sie
    na ekranie uzytkownika). Zwraca wybrany folder albo dotychczasowy."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=current or str(Path.home()),
                                         title="Autosave folder")
        root.destroy()
    except Exception:  # noqa: BLE001 - brak okna = zostaje stary folder
        chosen = ""
    return chosen or current
