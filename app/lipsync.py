# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Lip-sync (LatentSync 1.6) w osobnym procesie - po zakonczeniu cala pamiec GPU wraca do systemu.

Przed uruchomieniem wywolujacy zwalnia modele ComfyUI (ComfyClient.free_memory), zeby
LatentSync dostal pusta karte. Postep procesu roboczego (app/lipsync_worker.py) trafia
do progress_cb, a w razie bledu zwracamy koncowke jego logu.
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

import soundfile as sf

from . import cancel, dialogue, ffmpeg_utils, model_manager

MODEL_ID = "latentsync-16"
_WORKER = Path(__file__).with_name("lipsync_worker.py")
_FACES = Path(__file__).with_name("faces_worker.py")
MIN_FACE_PX = 48   # mniejsza twarz: ruchu ust i tak nie widac, a LatentSync jej nie znajdzie


def available() -> bool:
    return model_manager.is_installed(MODEL_ID)


def find_faces(video: str, workdir: str) -> dict:
    """Twarze w wideo (app/faces_worker.py): slady z ramka, rozmiarem i plcia."""
    out = Path(workdir) / "faces.json"
    subprocess.run([sys.executable, str(_FACES), str(video), str(out)], cwd=workdir,
                   capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tracks": []}


def run_dialogue(video: str, dialog_wav: str, timings: List[Tuple[float, float]],
                 who: List[str], output: str,
                 progress_cb: Optional[Callable[[float, str], None]] = None
                 ) -> Tuple[bool, str, List[str]]:
    """Ruch ust dla kazdej mowiacej postaci w kadrze - bez limitu liczby osob.

    who[i] - osoba mowiaca kwestie i ("female#0", "male#0"...), timings[i] - jej czas.
    Osoby dobieramy do twarzy wg plci (CLIP) i wielkosci twarzy. Jedna twarz: caly kadr
    (najlepsza jakosc). Kilka: wycinek wokol kazdej mowiacej twarzy idzie do LatentSync
    z dzwiekiem tylko jej kwestii (reszta cisza - usta zamkniete) i wraca na miejsce.
    Zwraca (ok, sciezka_lub_komunikat, uwagi).
    """
    notes: List[str] = []
    work = tempfile.mkdtemp(prefix="aistudio_dialog_")
    try:
        if progress_cb:
            progress_cb(0.0, "Lip sync: finding faces")
        v25 = Path(work) / "v25.mp4"   # LatentSync pracuje w 25 fps - wszystkie warstwy tez
        ffmpeg_utils._run(["ffmpeg", "-y", "-i", str(video), "-vf", "fps=25", "-c:v", "libx264",
                           "-crf", "14", "-pix_fmt", "yuv420p", "-an", str(v25)])
        total = round((ffmpeg_utils.probe_duration(v25) or 0.0) * 25)
        # Tylko fragment z mowa (+0,4 s zapasu): LatentSync przerywa, gdy w ktorejkolwiek
        # klatce nie ma twarzy - a w innym ujeciu postac moze stac tylem albo biec.
        f0 = max(0, int((min(a for a, _ in timings) - 0.4) * 25))
        f1 = min(total, int((max(b for _, b in timings) + 0.4) * 25) + 1)
        if f0 < 5 and f1 > total - 5:
            f0, f1 = 0, total
        clip, wav = v25, Path(dialog_wav)
        if (f0, f1) != (0, total):
            clip, wav = Path(work) / "clip.mp4", Path(work) / "clip.wav"
            ffmpeg_utils._run(["ffmpeg", "-y", "-i", str(v25), "-vf",
                               f"trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS",
                               "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-an",
                               str(clip)])
            audio, rate = sf.read(str(dialog_wav), dtype="float32", always_2d=False)
            sf.write(str(wav), audio[int(f0 / 25 * rate):int(f1 / 25 * rate)], rate)
            timings = [(a - f0 / 25, b - f0 / 25) for a, b in timings]
        out_clip = Path(work) / "clip_ls.mp4" if clip != v25 else Path(output)
        ok, msg = _dialogue_clip(clip, str(wav), timings, who, str(out_clip), (f1 - f0) / 25,
                                 work, progress_cb, notes)
        if not ok or clip == v25:
            return ok, msg, notes
        ok, msg = _splice(v25, out_clip, f0, f1, total, Path(output))
        return ok, (output if ok else msg), notes
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _splice(full: Path, clip: Path, f0: int, f1: int, total: int, out: Path) -> Tuple[bool, str]:
    """Wkleja przetworzony fragment [f0, f1) z powrotem w pelne wideo 25 fps."""
    parts, n = [], 0
    graph = []
    if f0 > 0:
        graph.append(f"[0:v]trim=end_frame={f0},setpts=PTS-STARTPTS[p{n}]")
        parts.append(f"[p{n}]")
        n += 1
    graph.append(f"[1:v]fps=25,setpts=PTS-STARTPTS,trim=end_frame={f1 - f0}[p{n}]")
    parts.append(f"[p{n}]")
    n += 1
    if f1 < total:
        graph.append(f"[0:v]trim=start_frame={f1},setpts=PTS-STARTPTS[p{n}]")
        parts.append(f"[p{n}]")
        n += 1
    graph.append(f"{''.join(parts)}concat=n={n}:v=1:a=0[v]")
    return ffmpeg_utils._run(["ffmpeg", "-y", "-i", str(full), "-i", str(clip), "-filter_complex",
                              ";".join(graph), "-map", "[v]", "-c:v", "libx264", "-crf", "14",
                              "-pix_fmt", "yuv420p", str(out)])


def _dialogue_clip(v25: Path, dialog_wav: str, timings: List[Tuple[float, float]],
                   who: List[str], output: str, length: float, work: str,
                   progress_cb: Optional[Callable[[float, str], None]],
                   notes: List[str]) -> Tuple[bool, str]:
    """Ruch ust na fragmencie 25 fps (patrz run_dialogue). Zwraca (ok, sciezka/blad)."""
    faces = find_faces(str(v25), work)
    tracks = sorted((t for t in faces.get("tracks", []) if t["size"] >= MIN_FACE_PX),
                    key=lambda t: -t["size"] * t["presence"])
    if not tracks:
        return False, ("No face in the frame is large enough for lip sync; "
                       "use a close-up or medium shot")
    # osoba -> twarz: najpierw pasujaca plec, w kolejnosci wielkosci twarzy
    assign = {}
    for person in dict.fromkeys(who):
        free = [t for t in tracks if t not in assign.values()]
        pick = [t for t in free if t.get("gender") == person.split("#")[0]] or free
        if pick:
            assign[person] = pick[0]
        else:
            notes.append("Lip sync: there are more speakers than visible faces; "
                         "some lines will be heard off-screen.")
    if len(tracks) == 1 and len(assign) == 1:
        person = next(iter(assign))
        audio = Path(work) / "a0.wav"
        dialogue.speaker_track(dialog_wav, timings, [i for i, p in enumerate(who) if p == person],
                               audio, length)
        ok, msg = run(str(v25), str(audio), output, progress_cb)
        return ok, (output if ok else msg)

    W, H = faces["width"], faces["height"]
    layers = []
    for k, (person, t) in enumerate(assign.items()):
        x0, y0, x1, y1 = t["bbox"]
        side = max(x1 - x0, y1 - y0) * 1.8   # twarz z zapasem na ruch glowy
        w, h = min(int(side) // 2 * 2, W // 2 * 2), min(int(side) // 2 * 2, H // 2 * 2)
        x = int(min(max((x0 + x1 - w) / 2, 0), W - w))
        y = int(min(max((y0 + y1 - h) / 2, 0), H - h))
        crop, audio, out = (Path(work) / f"{n}{k}.{e}" for n, e in
                            (("crop", "mp4"), ("a", "wav"), ("ls", "mp4")))
        ffmpeg_utils._run(["ffmpeg", "-y", "-i", str(v25), "-vf", f"crop={w}:{h}:{x}:{y}",
                           "-c:v", "libx264", "-crf", "14", "-an", str(crop)])
        dialogue.speaker_track(dialog_wav, timings,
                               [i for i, p in enumerate(who) if p == person], audio, length)

        def cb(f, d, k=k):
            if progress_cb:
                progress_cb((k + f) / len(assign), f"Lip sync: person {k + 1}/{len(assign)} - {d}")

        ok, msg = run(str(crop), str(audio), str(out), cb)
        if ok:
            layers.append((str(out), x, y, w, h))
        else:
            notes.append(f"Lip sync for person {k + 1} skipped ({msg[:120]}).")
    if not layers:
        return False, "LatentSync did not process any faces"
    ok, msg = ffmpeg_utils.overlay_layers(v25, layers, output, work)
    return ok, (output if ok else msg)


def run(video: str, audio: str, output: str,
        progress_cb: Optional[Callable[[float, str], None]] = None) -> Tuple[bool, str]:
    """Synchronizuje usta na wideo z narracja. Zwraca (ok, komunikat). Wynik ma dzwiek 16 kHz."""
    workdir = tempfile.mkdtemp(prefix="aistudio_lipsync_")
    log = deque(maxlen=15)
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(_WORKER), str(Path(video).resolve()),
             str(Path(audio).resolve()), str(Path(output).resolve()),
             *(["--low-vram"] if model_manager.small_gpu() else [])],
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
    if proc.returncode != 0 or not Path(output).exists():
        return False, " | ".join(list(log)[-4:]) or f"Process exited with code {proc.returncode}"
    return True, output
