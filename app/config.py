# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Konfiguracja i sciezki projektu.

Laduje config/settings.json (z opcjonalnym nadpisaniem przez settings.local.json)
oraz katalog modeli config/models.json. Wszystkie sciezki licza sie od katalogu
glownego projektu (rodzic folderu app/).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

# Katalog glowny = rodzic folderu app/
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Rekurencyjnie nadpisuje slownik base wartosciami z override."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_settings() -> Dict[str, Any]:
    """Wczytuje settings.json i (jesli jest) settings.local.json jako nadpisanie."""
    with open(CONFIG_DIR / "settings.json", encoding="utf-8") as f:
        settings = json.load(f)
    local = CONFIG_DIR / "settings.local.json"
    if local.exists():
        with open(local, encoding="utf-8") as f:
            settings = _deep_merge(settings, json.load(f))
    for section in ("app", "comfyui"):
        cfg = settings[section]
        if cfg.get("host") not in ("127.0.0.1", "localhost"):
            raise ValueError(f"{section}.host must be 127.0.0.1 or localhost. Contsy is a local-only application.")
        port = cfg.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError(f"{section}.port must be an integer between 1024 and 65535.")
    if settings["app"]["port"] == settings["comfyui"]["port"]:
        raise ValueError("The app and ComfyUI must use different ports.")
    args = settings["comfyui"].get("extra_args", [])
    forbidden = ("--listen", "--port", "--enable-cors-header", "--base-directory", "--output-directory", "--input-directory", "--extra-model-paths-config")
    if not isinstance(args, list) or any(not isinstance(a, str) or
            a.split("=", 1)[0] in forbidden for a in args):
        raise ValueError("ComfyUI extra_args contains an unsupported networking or filesystem override.")
    # Where users get the source code of this exact version, shown in Settings (About).
    source_url = settings["app"].setdefault("source_url", "")
    if not isinstance(source_url, str) or (
            source_url and not re.fullmatch(r"https?://[^\s'\"<>]+", source_url)):
        raise ValueError("app.source_url must be empty or the http(s) address of this version's source code.")
    return settings


def load_model_catalog() -> Dict[str, Any]:
    """Wczytuje katalog modeli (config/models.json)."""
    with open(CONFIG_DIR / "models.json", encoding="utf-8") as f:
        return json.load(f)


SETTINGS: Dict[str, Any] = load_settings()
CATALOG: Dict[str, Any] = load_model_catalog()


def abs_path(rel: str) -> Path:
    """Zamienia sciezke wzgledna z konfiguracji na absolutna w katalogu projektu."""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


# Czesto uzywane sciezki
COMFYUI_DIR = abs_path(SETTINGS["paths"]["comfyui_dir"])
MODELS_DIR = abs_path(SETTINGS["paths"]["models_dir"])
OUTPUTS_DIR = abs_path(SETTINGS["paths"]["outputs_dir"])
HISTORY_DIR = abs_path(SETTINGS["paths"]["history_dir"])


def destination_dir(dest_key: str) -> Path:
    """Zwraca absolutny katalog docelowy dla wpisu 'dest' z katalogu modeli."""
    rel = CATALOG["destinations"].get(dest_key, dest_key)
    return abs_path(rel)


def models_by_category() -> Dict[str, List[dict]]:
    """Grupuje modele z katalogu wg kategorii."""
    grouped: Dict[str, List[dict]] = {}
    for m in CATALOG["models"]:
        grouped.setdefault(m["category"], []).append(m)
    return grouped


def get_model(model_id: str) -> dict | None:
    for m in CATALOG["models"]:
        if m["id"] == model_id:
            return m
    return None


def ensure_dirs() -> None:
    """Tworzy podstawowe foldery robocze, jesli ich nie ma."""
    for d in (MODELS_DIR, OUTPUTS_DIR, HISTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)
