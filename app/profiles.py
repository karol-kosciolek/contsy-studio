# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Profile postaci - spojny wyglad tej samej (fikcyjnej) postaci we wszystkich trybach.

Profil to opis + tagi do promptu + opcjonalna LoRA + seed + obraz referencyjny
+ wybrany glos TTS. Przechowywany jako JSON w config/profiles/<slug>.json.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from . import config
from .persistence import atomic_text

PROFILES_DIR = config.CONFIG_DIR / "profiles"


@dataclass
class Profile:
    name: str
    description: str = ""
    positive_tags: str = ""      # dopisywane do promptu (wyglad postaci)
    negative_tags: str = ""
    lora: str = ""               # nazwa pliku LoRA (opcjonalna)
    lora_strength: float = 0.8
    seed: int = -1               # -1 = losowy
    reference_image: str = ""    # sciezka do zdjecia postaci (do i2v/i2i)
    voice: str = ""              # nazwa glosu TTS
    created: float = field(default_factory=time.time)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
    return s.strip("_") or "postac"


def _path(name: str) -> Path:
    return PROFILES_DIR / f"{_slug(name)}.json"


def list_profiles() -> List[str]:
    """Nazwy profili tak, jak je wpisano (load/delete przyjmuja je bezposrednio)."""
    if not PROFILES_DIR.exists():
        return []
    names = []
    for p in PROFILES_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            names.append(str(data.get("name") or p.stem) if isinstance(data, dict) else p.stem)
        except (OSError, ValueError):
            names.append(p.stem)
    return sorted(names, key=str.lower)


def load(name: str) -> Optional[Profile]:
    p = _path(name)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return Profile(**{k: v for k, v in data.items() if k in Profile.__annotations__})
    except (OSError, ValueError, TypeError):
        return None


def save(profile: Profile, reference_src: Optional[str] = None) -> Profile:
    """Zapisuje profil. Jesli podano reference_src, kopiuje zdjecie do folderu profili."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    if reference_src:
        src = Path(reference_src)
        if src.exists():
            target = PROFILES_DIR / f"{_slug(profile.name)}{src.suffix.lower()}"
            shutil.copy2(src, target)
            profile.reference_image = str(target)
    atomic_text(_path(profile.name), json.dumps(asdict(profile), ensure_ascii=False, indent=2))
    return profile


def delete(name: str) -> bool:
    """Usuwa profil razem z jego kopia zdjecia referencyjnego."""
    profile = load(name)
    p = _path(name)
    if not p.exists():
        return False
    p.unlink()
    if profile and profile.reference_image:
        ref = Path(profile.reference_image)
        if ref.parent == PROFILES_DIR:
            ref.unlink(missing_ok=True)
    return True


def apply_to_prompt(profile: Optional[Profile], positive: str, negative: str):
    """Laczy prompt uzytkownika z tagami profilu. Zwraca (positive, negative)."""
    if profile is None:
        return positive, negative
    pos = ", ".join(x for x in (profile.positive_tags, positive) if x.strip())
    neg = ", ".join(x for x in (profile.negative_tags, negative) if x.strip())
    return pos, neg
