# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Dialogi postaci zapisane w prompcie wideo.

Kwestie piszemy w cudzyslowie wprost w opisie sceny, np.:
    A woman in a cafe smiles at the camera and says „Cześć, miło Cię widzieć!”
parse() rozdziela prompt na:
  - opis obrazu dla Wan (bez tekstu kwestii - cytat w prompcie konczy sie czesto napisem
    na ekranie; zamiast niego dopisujemy, ze postac mowi),
  - opis dzwieku tla dla Stable Audio (bez mowy - ta dochodzi osobno),
  - liste kwestii z plcia mowiacego (z najblizszego slowa przed cytatem: woman/man,
    kobieta/mezczyzna, she/he...), wedlug ktorej dobieramy glos.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

# „...”  “...”  "..."  «...»
_QUOTE = re.compile(r'[„“"«]([^„“”"«»]{2,}?)[”"»]')
_FEMALE = re.compile(r"\b(woman|women|girl|lady|she|her|female|mother|grandmother|kobiet\w*|"
                     r"dziewczyn\w*|pani|ona|matk\w*|babci\w*)\b", re.I)
_MALE = re.compile(r"\b(man|men|boy|guy|he|him|his|male|father|grandfather|m[eę][żz]czyzn\w*|"
                   r"ch[łl]opak\w*|chłopiec|pan|on|ojciec|dziadek)\b", re.I)
# "says:", "mówi:" itp. tuz przed cytatem - usuwamy razem z cytatem
_SAYS = re.compile(r"\b(says|said|asks|asked|shouts|whispers|replies|answers|mówi|pyta|"
                   r"krzyczy|szepcze|odpowiada)\s*:?\s*$", re.I)

FEMALE_VOICES = ["Woman 1 — higher voice", "Woman 2 — lower voice"]
MALE_VOICES = ["Man 1 — low voice", "Man 2 — medium voice"]


# "two women" / "dwie kobiety" w opisie - kwestie tej plci przypisujemy na zmiane dwom osobom
_PAIR = {"female": re.compile(r"\b(two|2)\s+(women|girls|ladies)\b|\bdwie\s+(kobiety|dziewczyny)", re.I),
         "male": re.compile(r"\b(two|2)\s+(men|boys|guys)\b|\bdw(ó|o)ch\s+(m[eę][żz]czyzn|ch[łl]opak)", re.I)}


@dataclass
class Line:
    speaker: str   # plec: "female" / "male"
    text: str
    who: str = ""  # konkretna osoba: "female#0", "female#1", "male#0"...


def _gender(context: str) -> Optional[str]:
    """Plec z ostatniego slowa-wskazowki w kontekscie przed cytatem."""
    last_f = max((m.end() for m in _FEMALE.finditer(context)), default=-1)
    last_m = max((m.end() for m in _MALE.finditer(context)), default=-1)
    if last_f < 0 and last_m < 0:
        return None
    return "female" if last_f > last_m else "male"


def parse(prompt: str) -> Tuple[str, str, List[Line]]:
    """Zwraca (opis_obrazu, opis_dzwieku_tla, kwestie)."""
    lines: List[Line] = []
    visual, sound, pos = [], [], 0
    has_f, has_m = bool(_FEMALE.search(prompt)), bool(_MALE.search(prompt))
    for m in _QUOTE.finditer(prompt):
        before = prompt[pos:m.start()]
        text = m.group(1).strip()
        speaker = _gender(before)
        if speaker is None:
            if lines and has_f and has_m:      # dialog bez wskazowek - na zmiane
                speaker = "male" if lines[-1].speaker == "female" else "female"
            elif lines:
                speaker = lines[-1].speaker
            else:
                speaker = "male" if has_m and not has_f else "female"
        lines.append(Line(speaker, text))
        # "says „...”" -> "talks" w tym samym miejscu; kropka, jesli cytat konczyl zdanie
        stop = "." if text[-1] in ".!?" else ""
        if _SAYS.search(before):
            visual.append(_SAYS.sub("talks", before) + stop)
        elif before.strip():
            visual.append(before.rstrip() + " talking" + stop)
        # kolejny cytat tuz po poprzednim - "talking" juz jest w opisie
        sound.append(_SAYS.sub("", before) + stop)
        pos = m.end()
    visual.append(prompt[pos:])
    sound.append(prompt[pos:])

    def tidy(parts: List[str], drop_fragments: bool = False) -> str:
        s = re.sub(r"\s{2,}", " ", "".join(parts))
        if drop_fragments:  # "and." / "he." po usunietej kwestii
            s = re.sub(r"\b(and|i|oraz|he|she|they|on|ona|oni)\s*([.,!?]|$)", r"\2", s)
        s = re.sub(r"\s+([,.!?])", r"\1", s)
        s = re.sub(r"([.!?,])[.!?,]+", r"\1", s)
        if drop_fragments:  # urywki ("The man.", "She.") nic nie wnosza do opisu dzwieku
            s = " ".join(z for z in re.split(r"(?<=[.!?])\s+", s) if len(z.split()) > 2)
        return s.strip(" ,:")

    last: dict = {}
    for line in lines:
        g = line.speaker
        idx = (1 - last[g] if g in last else 0) if _PAIR[g].search(prompt) else 0
        last[g] = idx
        line.who = f"{g}#{idx}"
    if not lines:
        return prompt.strip(), prompt.strip(), []
    return f"{tidy(visual).rstrip('.!?')}, natural lip movement", tidy(sound, True), lines


def voices_for(lines: List[Line], preferred: Optional[str] = None) -> List[str]:
    """Glos dla kazdej kwestii: kazda osoba swoj glos (druga kobieta - drugi glos kobiecy).

    preferred (np. glos z profilu postaci) dostaje pierwsza osoba pasujacej plci.
    """
    voices = {"female": list(FEMALE_VOICES), "male": list(MALE_VOICES)}
    for g in voices:
        if preferred in voices[g]:
            voices[g].remove(preferred)
            voices[g].insert(0, preferred)
    return [voices[l.speaker][int(l.who.split("#")[1]) % 2] for l in lines]


def assemble(wavs: List[str], out_path: str | Path, start: float = 0.3,
             gap: float = 0.5) -> Tuple[float, List[Tuple[float, float]]]:
    """Skleja kwestie w jedna sciezke (krotka cisza przed i miedzy).

    Zwraca (dlugosc w s, [(poczatek, koniec) kazdej kwestii]) - czasy sa potrzebne do
    ruchu ust, gdy mowi kilka postaci.
    """
    parts, rate = [], None
    for w in wavs:
        data, sr = sf.read(w, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        rate = rate or sr
        parts.append(data)
    silence = lambda s: np.zeros(int(rate * s), dtype=np.float32)  # noqa: E731
    track, timings, t = [silence(start)], [], start
    for i, p in enumerate(parts):
        timings.append((t, t + len(p) / rate))
        t += len(p) / rate + (gap if i + 1 < len(parts) else 0)
        track += [p] + ([silence(gap)] if i + 1 < len(parts) else [])
    audio = np.concatenate(track)
    sf.write(str(out_path), audio, rate)
    return len(audio) / rate, timings


def speaker_track(dialog_wav: str | Path, timings: List[Tuple[float, float]], keep: List[int],
                  out_path: str | Path, length: float) -> None:
    """Sciezka jednej postaci: tylko jej kwestie (indeksy `keep`), reszta cisza, dlugosc
    `length` s - na tej podstawie LatentSync rusza ustami tylko tej osoby."""
    audio, rate = sf.read(str(dialog_wav), dtype="float32", always_2d=False)
    out = np.zeros(int(length * rate) + 1, dtype=np.float32)
    for i in keep:
        a, b = (int(x * rate) for x in timings[i])
        b = min(b, len(audio), len(out))
        out[a:b] = audio[a:b]
    sf.write(str(out_path), out, rate)
