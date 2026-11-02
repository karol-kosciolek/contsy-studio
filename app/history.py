# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Historia generacji - zapis promptow, seedow, parametrow i plikow wynikowych.

Przechowujemy jako JSON Lines (history/history.jsonl) - dopisywanie jest tanie,
a odczyt ostatnich wpisow prosty.
"""
from __future__ import annotations

import functools
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .persistence import atomic_text

_HISTORY_FILE = config.HISTORY_DIR / "history.jsonl"
_CTX = threading.local()
_LOCK = threading.RLock()


def _synchronized(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return wrapped


def job_context(params: Optional[Dict]) -> None:
    """Polecenie uzytkownika, dane autopilota i profil (postac) biezacego zadania kolejki -
    dopisywane do wpisow historii z tego watku (generatory zapisuja historie gleboko, bez
    dostepu do zadania)."""
    _CTX.extra = {k: params[k] for k in ("idea", "autopilot", "day", "session")
                  if params and params.get(k)}
    if params and params.get("character"):
        _CTX.extra["profile"] = params["character"]


@_synchronized
def record(entry: Dict) -> Dict:
    """Dopisuje wpis do historii. Uzupelnia znacznik czasu, jesli brak."""
    entry = {**getattr(_CTX, "extra", {}), **entry}
    entry.setdefault("ts", time.time())
    entry.setdefault("time_str", time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(entry["ts"])))
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


@_synchronized
def recent(limit: int = 50) -> List[Dict]:
    """Zwraca ostatnie wpisy (najnowsze pierwsze)."""
    if not _HISTORY_FILE.exists():
        return []
    lines = _HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    out: List[Dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict):
                continue
            out.append(entry)
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def _same_file(left: str, right: str) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if Path(right).is_absolute():
        return str(Path(left).resolve()).casefold() == str(Path(right).resolve()).casefold()
    return Path(left).name == Path(right).name


@_synchronized
def annotate(path: str, **fields) -> None:
    """Dopisuje pola do wpisu historii pliku (np. werdykt kontroli jakosci "qc")."""
    if not _HISTORY_FILE.exists():
        return
    out = []
    for line in _HISTORY_FILE.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if not isinstance(entry, dict):
            out.append(line)
            continue
        if any(_same_file(f, path) for f in entry.get("files") or []):
            line = json.dumps({**entry, **fields}, ensure_ascii=False) + "\n"
        out.append(line)
    atomic_text(_HISTORY_FILE, "".join(out))


def entry_for(path: str, limit: int = 1000) -> Optional[Dict]:
    """Najnowszy wpis historii z tym plikiem (po nazwie) albo None."""
    return next((e for e in recent(limit)
                 if any(_same_file(f, path) for f in e.get("files") or [])), None)


@_synchronized
def clear() -> None:
    if _HISTORY_FILE.exists():
        _HISTORY_FILE.unlink()


@_synchronized
def forget(paths: List[str]) -> None:
    """Usuwa z historii wpisy usunietych plikow (Galeria -> Usun)."""
    if not _HISTORY_FILE.exists() or not paths:
        return
    out = []
    for line in _HISTORY_FILE.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            entry = json.loads(line)
        except (ValueError, TypeError):
            out.append(line)
            continue
        if not isinstance(entry, dict):
            out.append(line)
            continue
        files = entry.get("files") or []
        remaining = [f for f in files if not any(_same_file(f, path) for path in paths)]
        if len(remaining) == len(files):
            out.append(line)
        elif remaining:
            out.append(json.dumps({**entry, "files": remaining}, ensure_ascii=False) + "\n")
    atomic_text(_HISTORY_FILE, "".join(out))
