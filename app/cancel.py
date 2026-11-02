# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Przerywanie biezacego zadania ("⏹ Przerwij bieżące" w zakladce Kolejka).

Prosba o przerwanie ustawia flage i natychmiast: zatrzymuje zadanie w ComfyUI (/interrupt)
i zabija procesy robocze (glos, ruch ust). Kod generowania wywoluje check() przy kazdej
aktualizacji postepu - wtedy leci wyjatek Cancelled, ktory przechodzi przez zwykle
"except Exception" (dziedziczy po BaseException) az do kolejki, a pliki tymczasowe sprzata
video_gen.generate.
"""
from __future__ import annotations

import subprocess
import threading
from typing import Set

_FLAG = threading.Event()
_PROCS: Set[subprocess.Popen] = set()
_LOCK = threading.Lock()


class Cancelled(BaseException):
    """Zadanie przerwane przez uzytkownika."""


def reset() -> None:
    _FLAG.clear()


def requested() -> bool:
    return _FLAG.is_set()


def check() -> None:
    if _FLAG.is_set():
        raise Cancelled()


def track(proc: subprocess.Popen) -> None:
    with _LOCK:
        _PROCS.add(proc)


def untrack(proc: subprocess.Popen) -> None:
    with _LOCK:
        _PROCS.discard(proc)
    if proc.poll() is None:  # wyjatek w trakcie (np. Cancelled) - proces nie moze zostac
        proc.kill()


def request() -> None:
    """Przerywa biezace zadanie: flaga, procesy robocze, silnik ComfyUI."""
    _FLAG.set()
    with _LOCK:
        procs = list(_PROCS)
    for p in procs:
        if p.poll() is None:
            p.kill()
    from . import config
    from .comfy_client import ComfyClient

    cfg = config.SETTINGS["comfyui"]
    ComfyClient(cfg["host"], cfg["port"]).interrupt()
