# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Mandatory SFW checks and optional face/hand quality checks.

Every generated output needs a complete vision response. Missing or malformed responses
fail closed. Unsafe and unverified files are preserved in outputs/odrzucone with an audit
record. Quality failures may keep the final result only after SFW coverage is confirmed.
Video validation samples the full duration at two-second intervals plus the start,
end (and midpoint for short clips); it cannot guarantee every frame between samples is safe.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import characters, config, history, llm, model_manager, prefs

MODEL_ID = "face-check"
FACE_MIN = 0.40
MAX_TRIES = 3
NUDITY = "nudity"
UNCHECKED = "unchecked"
REJECTED_DIR = config.OUTPUTS_DIR / "odrzucone"
_IMAGES = (".png", ".jpg", ".jpeg", ".webp")

# Nagosc w opisie stroju ("bare shoulders" to jeszcze nie nagosc). Model pisze w obu szykach:
# "bare breast" i "right breast bare", "nipple visible" (test 30.10).
_PART = r"(?:breasts?|chest|nipples?|areolas?|buttocks?|butt|bottom|crotch|genitals?|torso)"
_BARE = re.compile(
    r"\b(?:naked|nude|nudity|topless|bottomless|uncovered|genitals?|"
    rf"(?:bare|exposed|visible|uncovered) (?:\w+ )?{_PART}|"
    rf"{_PART} (?:\w+ )?(?:is |are )?(?:bare|exposed|visible|uncovered|showing)|"
    r"nothing|no (?:top|bra|bikini top|clothing|clothes|bottoms?|underwear|pants))\b", re.I)
# nie liczy sie jako nagosc: "no nipple visible", "nothing visible", "not visible"
_NEGATED = re.compile(r"\b(?:no|not|without|none)\b[^,;.]{0,25}\b(?:visible|exposed|bare)\b", re.I)
_BAD_HANDS = re.compile(r"\b(?:extra|missing|six|seven|deformed|malformed|fused|mangled|"
                        r"distorted|merged|too many)\b", re.I)


@dataclass
class Verdict:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    face: Optional[float] = None
    coverage: Optional[dict] = None

    def summary(self) -> str:
        return "; ".join(self.reasons) or "ok"

    @property
    def nude(self) -> bool:
        return any(r.startswith((NUDITY, "nagość")) for r in self.reasons)

    @property
    def unchecked(self) -> bool:
        return any(r.startswith((UNCHECKED, "niesprawdzone")) for r in self.reasons)

    @property
    def safe(self) -> bool:
        """Confirmed SFW, independently of optional face/hand quality checks."""
        return not self.nude and not self.unchecked

    def record(self) -> dict:
        return {"ok": self.ok, "safe": self.safe, "reasons": self.reasons,
                "face": None if self.face is None else round(self.face, 3),
                "coverage": self.coverage}


def enabled() -> bool:
    return bool(prefs.load().get("qc", True))


def bare(text: str) -> bool:
    """Czy opis stroju mowi o nagosci (po odrzuceniu zaprzeczen typu 'no nipple visible')."""
    return bool(_BARE.search(_NEGATED.sub(" ", text or "")))


def judge(face: Optional[float], cov: Optional[dict], quality: bool = True) -> Verdict:
    """Werdykt z pomiarow: opis stroju (zawsze - straz SFW), a przy kontroli jakosci tez
    podobienstwo twarzy (None = nie sprawdzono) i dlonie."""
    reasons: List[str] = []
    if quality and face is not None and face < FACE_MIN:
        reasons.append(f"different face (similarity {face:.2f} < {FACE_MIN:.2f})")
    if cov:
        if bare(cov.get("chest", "")) or bare(cov.get("hips", "")):
            reasons.append(f"{NUDITY} (chest: {cov.get('chest')}; hips: {cov.get('hips')})")
        if quality and _BAD_HANDS.search(str(cov.get("hands", ""))):
            reasons.append(f"hands: {cov.get('hands')}")
    return Verdict(ok=not reasons, reasons=reasons, face=face, coverage=cov)


# ----------------------------- twarz (YuNet + SFace) -----------------------------

_LOCK = threading.Lock()
_NETS: Dict[str, object] = {}
_REFS: Dict[str, Tuple[float, list]] = {}  # slug -> (znacznik zdjec, osadzenia twarzy)


def _nets():
    import cv2

    if not _NETS:
        d = config.destination_dir("qc")
        _NETS["det"] = cv2.FaceDetectorYN.create(str(d / "face_detection_yunet_2023mar.onnx"),
                                                 "", (320, 320), 0.8, 0.3, 5000)
        _NETS["rec"] = cv2.FaceRecognizerSF.create(
            str(d / "face_recognition_sface_2021dec.onnx"), "")
    return _NETS["det"], _NETS["rec"]


def _embed(path) -> Optional[object]:
    """Osadzenie najwiekszej twarzy na obrazie albo None (brak twarzy / pliku)."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 1280 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
    det, rec = _nets()
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    return rec.feature(rec.alignCrop(img, face))


def _references(ch: dict) -> list:
    """Osadzenia twarzy postaci: portret probny + zdjecia treningowe (liczone raz)."""
    folder = characters.DIR / ch["slug"]
    files = [folder / "preview.png"] + sorted((folder / "photos").glob("*"))
    files = [f for f in files if f.suffix.lower() in _IMAGES and f.exists()][:40]
    stamp = sum(f.stat().st_mtime for f in files)
    cached = _REFS.get(ch["slug"])
    if cached and cached[0] == stamp:
        return cached[1]
    embs = [e for e in (_embed(f) for f in files) if e is not None]
    _REFS[ch["slug"]] = (stamp, embs)
    return embs


def face_score(path: str, ch: dict) -> Optional[float]:
    """Najwieksze podobienstwo twarzy z wyniku do twarzy postaci; None - brak twarzy."""
    import cv2

    with _LOCK:
        refs = _references(ch)
        emb = _embed(path)
        if emb is None or not refs:
            return None
        _, rec = _nets()
        return max(float(rec.match(emb, r, cv2.FaceRecognizerSF_FR_COSINE)) for r in refs)


# ----------------------------- sprawdzanie i odrzucanie -----------------------------

def coverage(path: str) -> Optional[dict]:
    """Opis stroju; druga, niezalezna proba, gdy pierwsza nie widzi nagosci - model raz na
    kilka razy pomijal odsloniety biust (test 31.10: 1 z 3 opisow nagiego zdjecia)."""
    def valid(value):
        return isinstance(value, dict) and all(
            isinstance(value.get(key), str) and value[key].strip()
            for key in ("chest", "hips"))

    cov = llm.coverage(path)
    if not valid(cov):
        return None
    if bare(cov["chest"]) or bare(cov["hips"]):
        return cov
    second = llm.coverage(path)
    if not valid(second):
        return None
    return second if bare(second["chest"]) or bare(second["hips"]) else cov


def check(path: str, ch: Optional[dict], quality: bool = True) -> Verdict:
    """Require SFW coverage; face matching is optional and requires a character."""
    face = None
    if quality and ch and not model_manager.ensure([MODEL_ID]):
        try:
            face = face_score(path, ch)
        except Exception:  # noqa: BLE001 - brak OpenCV / zly plik: bez sprawdzenia twarzy
            face = None
    try:
        cov = coverage(path)
    except Exception:
        cov = None
    valid = isinstance(cov, dict) and all(isinstance(cov.get(k), str) and cov[k].strip() for k in ("chest", "hips"))
    verdict = judge(face, cov if valid else None, quality)
    if not valid:
        verdict.reasons.append(f"{UNCHECKED} (vision model did not confirm clothing coverage)")
        verdict.ok = False
    return verdict


def check_video(path: str, ch: Optional[dict], quality: bool = True) -> Verdict:
    """Check the start, every two seconds, and the end, with at least three frames.

    Every required frame must be extracted and validated. Temporal sampling reduces
    missed scene changes, but cannot guarantee the content of frames between samples.
    """
    reasons, faces, cov = [], [], None
    dur = _duration(path)
    if not math.isfinite(dur) or dur <= 0:
        return Verdict(ok=False, reasons=[f"{UNCHECKED} (could not read video duration)"])
    end = max(0.0, dur - min(0.1, dur / 10))
    moments = {0.0, end}
    moments.update(float(second) for second in range(2, math.ceil(dur), 2) if second < end)
    if len(moments) < 3:
        moments.add(dur / 2)
    with tempfile.TemporaryDirectory() as tmp:
        for i, at in enumerate(sorted(moments)):
            frame = Path(tmp) / f"frame_{i}.png"
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.6f}", "-i",
                                str(path), "-frames:v", "1", str(frame)],
                               capture_output=True, check=False, timeout=30,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.TimeoutExpired):
                pass
            if not frame.exists() or frame.stat().st_size == 0:
                reasons.append(f"{UNCHECKED} (could not extract video frame at {at:.2f}s)")
                continue
            verdict = check(str(frame), ch, quality)
            reasons += [reason for reason in verdict.reasons if reason not in reasons]
            if verdict.face is not None:
                faces.append(verdict.face)
            cov = cov or verdict.coverage
    return Verdict(ok=not reasons, reasons=reasons, face=min(faces) if faces else None,
                   coverage=cov)


def _duration(path: str) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                            "csv=p=0", str(path)], capture_output=True, text=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(r.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def reject(path: str, verdict: Verdict, label: str = "", keep_original: bool = False) -> str:
    """Wynik do outputs/odrzucone/ (z wpisem w qc.jsonl) i z historii. keep_original - kopia
    (np. klatka startowa, ktora i tak jest plikiem tymczasowym). Zwraca nowa sciezke."""
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    target = REJECTED_DIR / Path(path).name
    while target.exists():
        target = REJECTED_DIR / f"{Path(path).stem}_{time.time_ns()}{Path(path).suffix}"
    if keep_original:
        shutil.copy2(path, target)
    else:
        shutil.move(path, target)
        history.forget([path])
    with open(REJECTED_DIR / "qc.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "file": target.name, "job": label,
                            **verdict.record()}, ensure_ascii=False) + "\n")
    return str(target)


Make = Callable[[], Tuple[bool, List[str], str]]


def _nude_fail(what: str) -> str:
    return f"⛔ SFW check failed: {what} was quarantined in {REJECTED_DIR}."


def generate_checked(make: Make, ch: Optional[dict], notes: List[str],
                     progress_cb: Optional[Callable[[float, str], None]] = None,
                     label: str = "", guard: Optional[bool] = None
                     ) -> Tuple[bool, List[str], str]:
    """Check every output. SFW validation is mandatory, even without a character."""
    quality = enabled() and bool(ch)
    for attempt in range(1, MAX_TRIES + 1):
        ok, files, msg = make()
        if not (ok and files):
            return ok, files, msg
        if progress_cb:
            progress_cb(0.98, f"Checking results ({attempt}/{MAX_TRIES})...")
        verdicts = [check(path, ch, quality) for path in files]
        for path, verdict in zip(files, verdicts):
            history.annotate(path, qc=verdict.record())
        unsafe = [v for v in verdicts if not v.safe]
        if unsafe:
            # Keep a complete failed batch together, including otherwise safe siblings.
            for path, verdict in zip(files, verdicts):
                reject(path, verdict, label)
            if any(v.unchecked for v in unsafe) or attempt == MAX_TRIES:
                return False, [], _nude_fail("unverified or unsafe output")
            notes.append("SFW check rejected a result; generating again.")
            continue
        if all(v.ok for v in verdicts):
            if attempt > 1:
                notes.append(f"Check passed on attempt {attempt}.")
            return True, files, msg
        if attempt == MAX_TRIES:
            notes.append(f"⚠ Quality check: review the last result ({'; '.join(v.summary() for v in verdicts if not v.ok)}).")
            return True, files, msg
        for path, verdict in zip(files, verdicts):
            reject(path, verdict, label)
        notes.append("Quality check rejected a result; generating again.")
    return False, [], "No verified output was produced."


def render_checked(client, ch: dict, text: str, width: int, height: int, out_path: Path,
                   seed: int, notes: List[str],
                   progress_cb: Optional[Callable[[float, str], None]] = None
                   ) -> Tuple[bool, str]:
    """Validate a generated start frame before video inference can use it."""
    quality = enabled()
    for attempt in range(1, MAX_TRIES + 1):
        ok, msg = characters.render(client, ch, text, width, height, out_path,
                                    seed + 1000 * (attempt - 1), progress_cb)
        if not ok:
            return ok, msg
        verdict = check(msg, ch, quality)
        if verdict.ok:
            if attempt > 1:
                notes.append(f"Start frame passed on attempt {attempt}.")
            return True, msg
        if not verdict.safe:
            reject(msg, verdict, "start frame")
            if verdict.unchecked or attempt == MAX_TRIES:
                return False, _nude_fail("start frame")
        elif attempt == MAX_TRIES:
            notes.append(f"⚠ Start frame quality: {verdict.summary()}.")
            return True, msg
        else:
            reject(msg, verdict, "start frame")
        notes.append(f"Start frame rejected ({verdict.summary()}); generating again.")
    return False, "No verified start frame was produced."
