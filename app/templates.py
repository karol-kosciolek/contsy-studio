# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Silnik szablonow workflow dla wideo.

Szablony to pliki JSON w formacie API ComfyUI z tokenami %...% (np. %POSITIVE%,
%SEED%, %WIDTH%). build() podstawia pod nie wartosci:
  - token stojacy samodzielnie jako wartosc -> zastapiony wartoscia z zachowaniem
    typu (int/float/str),
  - token wewnatrz dluzszego napisu -> zwykle podstawienie tekstowe.

Dzieki temu, gdy nazwa wezla zmieni sie w przyszlej wersji ComfyUI, wystarczy
podmienic plik szablonu (wyeksportowany z ComfyUI przez „Zapisz (API)”),
zachowujac tokeny.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from . import config

TEMPLATES_DIR = config.ROOT / "workflows"


def available() -> List[str]:
    if not TEMPLATES_DIR.exists():
        return []
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.json"))


def load_raw(name: str) -> Dict[str, Any]:
    p = TEMPLATES_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Workflow template is missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _sub(obj: Any, mapping: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _sub(v, mapping) for k, v in obj.items() if k != "_note"}
    if isinstance(obj, list):
        return [_sub(v, mapping) for v in obj]
    if isinstance(obj, str):
        if obj in mapping:            # token samodzielny -> zachowaj typ
            return mapping[obj]
        out = obj
        for token, value in mapping.items():
            if token in out:
                out = out.replace(token, str(value))
        return out
    return obj


def gguf_loaders(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Load GGUF weights (the Lite model set) through ComfyUI-GGUF instead of the core loaders."""
    for node in graph.values():
        inputs = node.get("inputs", {})
        if node.get("class_type") == "CLIPLoader" and str(inputs.get("clip_name", "")).endswith(".gguf"):
            node["class_type"] = "CLIPLoaderGGUF"
            node["inputs"] = {"clip_name": inputs["clip_name"], "type": inputs["type"]}
        elif node.get("class_type") == "UNETLoader" and str(inputs.get("unet_name", "")).endswith(".gguf"):
            node["class_type"] = "UnetLoaderGGUF"
            node["inputs"] = {"unet_name": inputs["unet_name"]}
    return graph


def build(name: str, mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Wczytuje szablon i podstawia tokeny. Zwraca workflow w formacie API."""
    return gguf_loaders(_sub(load_raw(name), mapping))
