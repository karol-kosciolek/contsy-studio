# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Modelki (postacie z LoRA) jako profile studia: co nalezy do ktorej - zadania kolejki
i wyniki z historii - oraz liczby na Pulpit.

Profil = postac wybrana w panelu bocznym. Wtedy Kolejka, Galeria i Pulpit pokazuja tylko ja;
"Wszystkie (bez postaci)" pokazuje wszystko. Kolejka fizycznie jest jedna (jedna karta), wiec
"kolejka modelki" to widok na jej zadania. Wyniki zostaja w outputs/ - do kogo naleza, mowi
historia: nowe wpisy maja "profile" (slug), starsze "character" (nazwa postaci).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import characters, config, content, history, queue_manager

MEDIA = (".png", ".jpg", ".jpeg", ".webp", ".mp4")


def current() -> str:
    """Profil wybrany w panelu bocznym (slug) albo "" - wszystkie."""
    ch = characters.selected()
    return ch["slug"] if ch else ""


def name(profile: str) -> str:
    ch = characters.get(profile) if profile else None
    return ch["name"] if ch else ""


def jobs(profile: str, snapshot: Optional[List[queue_manager.Job]] = None
         ) -> List[queue_manager.Job]:
    """Zadania profilu (wszystkie, gdy profile == "")."""
    snap = queue_manager.snapshot() if snapshot is None else snapshot
    return [j for j in snap if not profile or (j.params or {}).get("character") == profile]


_CHAR_LORA = re.compile(r"aistudio_char_(.+)\.safetensors$")


def _owner(entry: Dict, by_name: Dict[str, str]) -> str:
    """Profil wpisu historii: "profile" (nowe wpisy), nazwa postaci (wideo) albo LoRA postaci
    w "loras" (starsze obrazy nie zapisywaly postaci, tylko jej LoRA)."""
    if entry.get("profile"):
        return entry["profile"]
    if by_name.get(entry.get("character") or ""):
        return by_name[entry["character"]]
    for lora in entry.get("loras") or []:
        m = _CHAR_LORA.match(str(lora.get("name", "") if isinstance(lora, dict) else lora))
        if m:
            return m.group(1)
    return ""


def file_owners(entries: Optional[List[Dict]] = None) -> Dict[str, str]:
    """Nazwa pliku wyniku -> profil (slug) wg historii; wynik bez postaci -> ""."""
    by_name = {c["name"]: c["slug"] for c in characters.all_characters()}
    owners: Dict[str, str] = {}
    for e in reversed(entries if entries is not None else history.recent(5000)):
        slug = _owner(e, by_name)
        for f in e.get("files") or []:
            owners[Path(f).name] = slug
    return owners


def _media_key(path: str) -> str:
    return str(Path(path).resolve()).casefold()


def _legacy(entry: Dict) -> bool:
    """Recorded before the mandatory SFW check (no qc.safe) or by the removed adult mode."""
    qc_result = entry.get("qc")
    return content.legacy_adult(entry) or not isinstance(qc_result, dict) or "safe" not in qc_result


def has_legacy_results(entries: Optional[List[Dict]] = None) -> bool:
    """Whether history lists files that stay hidden as legacy records."""
    return any(isinstance(entry, dict) and entry.get("files") and _legacy(entry)
               for entry in (history.recent(5000) if entries is None else entries))


def approved_files(entries: Optional[List[Dict]] = None) -> set[str]:
    """Only explicit SFW approvals allow media to be displayed.

    Earlier records containing only qc.ok have no confirmed mandatory safety review.
    Files and legacy history remain unchanged on disk.
    """
    approved = set()
    seen = set()
    for entry in history.recent(5000) if entries is None else entries:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files") or []
        if not isinstance(files, list):
            continue
        safe = not _legacy(entry) and entry["qc"].get("safe") is True
        for path in files:
            if not isinstance(path, str):
                continue
            key = _media_key(path)
            if key not in seen and safe:
                approved.add(key)
            seen.add(key)
    return approved


def is_approved(path: str, entries: Optional[List[Dict]] = None) -> bool:
    return _media_key(path) in approved_files(entries)


def results(profile: str = "", exts=MEDIA, limit: Optional[int] = None,
            owners: Optional[Dict[str, str]] = None) -> List[str]:
    """Wyniki z outputs/ (najnowsze pierwsze), przy profilu - tylko jego."""
    d = config.OUTPUTS_DIR
    if not d.exists():
        return []
    approved = approved_files()
    files = [p for p in d.iterdir() if p.is_file() and _media_key(str(p)) in approved and p.suffix.lower() in exts
             and not p.stem.endswith("_raw") and "_part" not in p.stem]
    if profile:
        owners = file_owners() if owners is None else owners
        files = [p for p in files if owners.get(p.name) == profile]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


def overview(snapshot: Optional[List[queue_manager.Job]] = None,
             entries: Optional[List[Dict]] = None, now: Optional[float] = None) -> List[Dict]:
    """Wiersz na kazda postac (plus "bez postaci"): stan, zadania w kolejce, wyniki dzis
    i razem, ostatni wynik, uwagi. Liczy tylko istniejace pliki."""
    snap = queue_manager.snapshot() if snapshot is None else snapshot
    entries = history.recent(5000) if entries is None else entries
    owners = file_owners(entries)
    midnight = time.mktime(time.localtime(now or time.time())[:3] + (0, 0, 0, 0, 0, -1))
    approved = approved_files(entries)
    seen: Dict[str, Dict] = {}
    for e in entries:
        for f in e.get("files") or []:
            name_ = Path(f).name
            if name_ in seen or not Path(f).exists() or _media_key(f) not in approved:
                continue
            seen[name_] = {"owner": owners.get(name_, ""), "ts": float(e.get("ts") or 0)}

    def row(slug: str, ch: Optional[dict]) -> Dict:
        mine = [v for v in seen.values() if v["owner"] == slug]
        queued = [j for j in snap if j.status in (queue_manager.PENDING, queue_manager.RUNNING)
                  and (j.params or {}).get("character", "") == slug]
        preview = Path(characters.DIR / slug / "preview.png") if slug else None
        return {"slug": slug, "name": ch["name"] if ch else "No character",
                "status": (ch or {}).get("status", ""), "usable": bool(slug and
                                                                       characters.usable(slug)),
                "preview": str(preview) if preview and preview.exists() and (ch or {}).get("preview_safe") is True else "",
                "queued": len(queued), "running": any(j.status == queue_manager.RUNNING
                                                      for j in queued),
                "eta": sum((j.eta() or 0.0) if j.status == queue_manager.RUNNING
                           else j.estimate() for j in queued),
                "today": sum(1 for v in mine if v["ts"] >= midnight), "total": len(mine),
                "last": max((v["ts"] for v in mine), default=0.0),
                "note": (ch or {}).get("error") or (ch or {}).get("warning") or ""}

    rows = [row(c["slug"], c) for c in characters.all_characters()]
    return rows + [row("", None)]
