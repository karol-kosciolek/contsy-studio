# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Synteza mowy (narracja i dialogi postaci) modelem VoxCPM2 - polskie glosy 48 kHz.

Model dziala w osobnym procesie (app/tts_worker.py), ktory po zakonczeniu oddaje cala
pamiec GPU. Kazdy glos to stala probka barwy (assets/voices), wiec postac brzmi tak
samo we wszystkich filmach.
Narracja jest opcjonalna: bez modelu zwracamy komunikat, a wideo i tak powstaje.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import cancel, config, model_manager

MODEL_ID = "voxcpm2"
_WORKER = Path(__file__).with_name("tts_worker.py")

# nazwa w UI -> probka barwy w assets/voices (wybrane z 24 glosow modelu wg wysokosci
# glosu F0 i oceny jakosci UTMOS: kobiety 259 i 178 Hz, mezczyzni 117 i 149 Hz)
VOICES = {
    "Woman 1 — higher voice": "kobieta_1.wav",
    "Woman 2 — lower voice": "kobieta_2.wav",
    "Man 1 — low voice": "mezczyzna_1.wav",
    "Man 2 — medium voice": "mezczyzna_2.wav",
}
_VOICES_DIR = config.ROOT / "assets" / "voices"


def available() -> bool:
    return model_manager.is_installed(MODEL_ID)


def available_voices() -> List[str]:
    return list(VOICES) if available() else []


def synthesize(text: str, voice: Optional[str], out_path: str | Path,
               progress_cb: Optional[Callable[[float, str], None]] = None) -> Tuple[bool, str]:
    """Generuje plik WAV (48 kHz) z jednym tekstem. Zwraca (ok, sciezka_lub_komunikat)."""
    if not text or not text.strip():
        return False, "Enter some text."
    return synthesize_many([(text, voice, str(out_path))], progress_cb)


def synthesize_many(items: List[Tuple[str, Optional[str], str]],
                    progress_cb: Optional[Callable[[float, str], None]] = None
                    ) -> Tuple[bool, str]:
    """Kilka kwestii (tekst, glos, plik) w jednym procesie - model laduje sie raz."""
    if not items or any(not str(text or "").strip() for text, _, _ in items):
        return False, "Enter some text for each spoken line."
    if not available():
        return False, "Voice skipped: install the VoxCPM2 model (Video + Audio package)."
    legacy_voices = {"Kobieta 1 — wyższy głos": "Woman 1 — higher voice",
                     "Kobieta 2 — niższy głos": "Woman 2 — lower voice",
                     "Mężczyzna 1 — niski głos": "Man 1 — low voice",
                     "Mężczyzna 2 — średni głos": "Man 2 — medium voice"}
    items = [(text, legacy_voices.get(voice, voice), out) for text, voice, out in items]
    job = [{"text": text.strip(),
            "reference": str(_VOICES_DIR / VOICES[voice if voice in VOICES else next(iter(VOICES))]),
            "output": str(Path(out).resolve())} for text, voice, out in items]
    workdir = tempfile.mkdtemp(prefix="aistudio_tts_")
    job_file = Path(workdir) / "job.json"
    job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    log = deque(maxlen=15)
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(_WORKER), str(job_file)],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "TQDM_DISABLE": "1"})
        cancel.track(proc)  # "Przerwij biezace" zabija proces od razu
        for line in proc.stdout:
            line = line.rstrip()
            match = re.search(r"PROGRESS (\S+) (.*)", line)
            if match:
                if progress_cb:
                    progress_cb(float(match.group(1)), match.group(2))
            elif line:
                log.append(line)
        proc.wait()
    finally:
        if proc is not None:
            cancel.untrack(proc)
        shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode != 0 or not all(Path(j["output"]).exists() for j in job):
        return False, "Speech synthesis failed: " + (" | ".join(list(log)[-3:]) or f"exit code {proc.returncode}")
    return True, job[-1]["output"]
