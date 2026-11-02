# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Kolejka zadan bez limitu - zapisywana na dysku, wykonywana po kolei w tle.

Klikniecie "Generuj" tylko dodaje zadanie (add) i od razu wraca, wiec mozna dodac dziesiatki
filmow. Jeden watek roboczy bierze zadania po kolei (na karcie 16 GB zawsze jedno naraz).
Oczekujace zadania i ich zalaczniki (kopie w queue_files/) przetrwaja restart aplikacji -
przerwane w polowie zadanie wraca na poczatek kolejki. Zadania mozna usuwac pojedynczo,
wszystkie naraz, wstrzymac kolejke albo przerwac biezace (app/cancel.py).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import cancel, config, content, history
from .persistence import atomic_text

PENDING, RUNNING, DONE, ERROR, STOPPED = "w kolejce", "generowanie", "gotowe", "blad", "przerwane"
_STATE = config.CONFIG_DIR / "queue.json"
FILES_DIR = config.ROOT / "queue_files"
_KEEP_FINISHED = 50

# runner(params, progress_cb) -> (ok, pliki, uwagi, blad)
Runner = Callable[[dict, Callable[[float, str], None]], Tuple[bool, List[str], List[str], str]]
_RUNNERS: Dict[str, Runner] = {}
# szacunek czasu przed pierwszym pomiarem: wideo na sekunde filmu (test 10 s 1080p: 28 min),
# obraz, narracja - potem srednia z ostatnich zadan. Wideo w trybach Szybki i Podglad ma wlasne
# srednie (app/video_gen.py MODES; pomiar 5 s: Szybki 6,5 min, z tego SeedVR2 ok. 3,8 min -
# Podglad bez niego) - wspolna skakalaby miedzy trybami.
_RATE = {"wideo": 170.0, "wideo-szybkie": 80.0, "wideo-podglad": 35.0, "obraz": 30.0,
         "narracja": 40.0, "trening": 4.5, "zestaw": 12.0}
_SPEED_KEYS = {"fast": "wideo-szybkie", "preview": "wideo-podglad"}  # tryby z app/video_gen.py


@dataclass
class Job:
    id: int
    kind: str          # wideo | obraz | narracja | trening
    label: str
    params: dict
    status: str = PENDING
    progress: float = 0.0
    desc: str = ""
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    error: str = ""
    files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    mark_t: Optional[float] = None  # chwila, gdy postep przekroczyl 10% (tempo do ETA)
    mark_p: float = 0.0

    def elapsed(self) -> float:
        return ((self.finished or time.time()) - self.started) if self.started else 0.0

    def units(self) -> float:
        """Jednostki pracy do szacowania czasu (wideo: sekundy filmu, trening: kroki)."""
        if self.kind == "zestaw":
            return float(self.params.get("count", 20))
        if self.kind == "trening":
            return float(self.params.get("steps", 2000))
        return float(self.params.get("seconds", 1)) if self.kind == "wideo" else 1.0

    def estimate(self) -> float:
        return rate(self.kind, self.params) * self.units()

    def eta(self) -> Optional[float]:
        if self.status != RUNNING:
            return None
        # trening: tempo samych krokow - szybkie sprawdzanie zdjec na poczatku zanizalo ETA
        if self.kind == "trening" and self.mark_t and self.progress - self.mark_p > 0.01:
            rate = (self.progress - self.mark_p) / max(time.time() - self.mark_t, 1.0)
            return max(0.0, (1.0 - self.progress) / rate)
        total = self.elapsed() / self.progress if self.progress > 0.05 else self.estimate()
        return max(0.0, total - self.elapsed())


_JOBS: List[Job] = []
_COND = threading.Condition()
_STATE_INFO = {"next_id": 1, "paused": False}
_REQUEUE = threading.Event()  # przerwanie przez pause_now: zadanie wraca do kolejki
_started = False


# ----------------------------- zapis na dysku -----------------------------

def _save() -> None:
    """Atomically save the queue without stopping its worker on a filesystem error."""
    data = {**_STATE_INFO, "rates": _RATE, "jobs": [asdict(j) for j in _JOBS]}
    try:
        atomic_text(_STATE, json.dumps(data, ensure_ascii=False, indent=1))
    except OSError as exc:
        print(f"[queue] Could not save {_STATE.name}: {exc}", flush=True)


def _load() -> None:
    try:
        data = json.loads(_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    try:
        _STATE_INFO["next_id"] = max(1, int(data.get("next_id", 1)))
    except (ValueError, TypeError, OverflowError):
        _STATE_INFO["next_id"] = 1
    _STATE_INFO["paused"] = bool(data.get("paused", False))
    rates = data.get("rates", {})
    if isinstance(rates, dict):
        for key, value in rates.items():
            try:
                value = float(value)
                if math.isfinite(value) and value > 0:
                    _RATE[key] = value
            except (ValueError, TypeError, OverflowError):
                continue
    seen = {j.id for j in _JOBS}
    for item in data.get("jobs", []) if isinstance(data.get("jobs"), list) else []:
        try:
            if not isinstance(item, dict) or not isinstance(item.get("params"), dict):
                continue
            job = Job(**{k: v for k, v in item.items() if k in Job.__dataclass_fields__})
            if not isinstance(job.id, int) or isinstance(job.id, bool) or job.id < 1 or job.id in seen:
                continue
            if not isinstance(job.kind, str) or not isinstance(job.label, str):
                continue
            if not isinstance(job.status, str) or job.status not in (PENDING, RUNNING, DONE, ERROR, STOPPED):
                continue
            if not all(isinstance(value, str) for value in (job.desc, job.error)):
                continue
            if not all(isinstance(value, list) and all(isinstance(entry, str) for entry in value)
                       for value in (job.files, job.notes)):
                continue
            for key in ("progress", "created", "mark_p"):
                number = float(getattr(job, key))
                if not math.isfinite(number):
                    raise ValueError("Invalid saved job timestamp or progress.")
                setattr(job, key, number)
            job.progress = min(max(job.progress, 0.0), 1.0)
            for key in ("started", "finished", "mark_t"):
                value = getattr(job, key)
                if value is not None:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("Invalid saved job timestamp.")
                    setattr(job, key, number)
            for key in ("steps", "seconds", "count"):
                if key in job.params:
                    number = float(job.params[key])
                    if not math.isfinite(number) or number <= 0:
                        raise ValueError("Invalid job size.")
            seen.add(job.id)
            if job.status == RUNNING:
                job.status, job.progress, job.desc, job.started = PENDING, 0.0, "", None
            if job.status == PENDING:
                refusal = _prompt_refusal(job.params)
                if refusal:
                    job.status, job.error, job.finished = ERROR, refusal, time.time()
            _JOBS.append(job)
        except (TypeError, ValueError, OverflowError):
            continue
    _STATE_INFO["next_id"] = max([_STATE_INFO["next_id"], *[j.id + 1 for j in _JOBS]])


def _prompt_refusal(params: dict) -> str:
    if not isinstance(params, dict):
        return "Invalid job parameters."
    if content.legacy_adult(params):
        return content.REFUSAL
    keys = {"idea", "positive", "prompt", "text", "story", "description", "theme",
            "scene", "caption", "script", "dialogue", "scenes", "lines"}
    def texts(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for entry in value:
                yield from texts(entry)
        elif isinstance(value, dict):
            for key, entry in value.items():
                if key in keys:
                    yield from texts(entry)
    values = list(texts(params))
    from . import llm  # Lazy import avoids the queue/LLM initialization cycle.
    if llm.blocked(*[content.normalize_for_check(value) for value in values]):
        return llm.BLOCK_MSG
    return content.check(*values)


def status_label(status: str) -> str:
    return {PENDING: "Queued", RUNNING: "Generating", DONE: "Done", ERROR: "Error",
            STOPPED: "Stopped"}.get(status, status)


# ----------------------------- API dla interfejsu -----------------------------

def _rate_key(kind: str, params: dict) -> str:
    return _SPEED_KEYS.get(params.get("speed"), kind) if kind == "wideo" else kind


def rate(kind: str, params: Optional[dict] = None) -> float:
    """Sekundy pracy karty na jednostke (srednia z ostatnich zadan) dla zadania z params."""
    return _RATE.get(_rate_key(kind, params or {}), 60.0)


def register(kind: str, runner: Runner) -> None:
    _RUNNERS[kind] = runner


def add(kind: str, label: str, params: dict, files: Optional[Dict[str, object]] = None) -> Job:
    """Dodaje zadanie. `files` - parametry z plikami (sciezka albo lista sciezek): kopiujemy
    je do queue_files/<id>/, bo pliki tymczasowe przegladarki znikaja po restarcie."""
    refusal = _prompt_refusal(params)
    if refusal:
        raise ValueError(refusal)
    with _COND:
        job = Job(id=_STATE_INFO["next_id"], kind=kind, label=label, params=dict(params))
        _STATE_INFO["next_id"] += 1
        folder = FILES_DIR / str(job.id)
        for key, value in (files or {}).items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", key):
                raise ValueError("Invalid attachment field name.")
            paths = value if isinstance(value, list) else [value]
            copied = []
            for i, src in enumerate(p for p in paths if p):
                folder.mkdir(parents=True, exist_ok=True)
                dst = folder / f"{key}_{i}{Path(src).suffix.lower() or '.png'}"
                shutil.copy2(src, dst)
                copied.append(str(dst))
            job.params[key] = copied if isinstance(value, list) else (copied[0] if copied else None)
        _JOBS.append(job)
        _save()
        _COND.notify_all()
        return job


def position(job_id: int) -> int:
    """Ile zadan jest przed danym (lacznie z biezacym)."""
    with _COND:
        ahead = [j for j in _JOBS if j.status in (PENDING, RUNNING) and j.id < job_id]
        return len(ahead)


def remove(job_ids: List[int]) -> None:
    """Usuwa oczekujace zadania; zaznaczone biezace - przerywa."""
    ids = set(job_ids)
    with _COND:
        if any(j.id in ids and j.status == RUNNING for j in _JOBS):
            cancel.request()
        for j in [j for j in _JOBS if j.id in ids and j.status == PENDING]:
            _JOBS.remove(j)
            shutil.rmtree(FILES_DIR / str(j.id), ignore_errors=True)
        _save()


def clear_pending() -> None:
    remove([j.id for j in snapshot() if j.status == PENDING])


def stop_current(job_id: Optional[int] = None) -> bool:
    """Przerywa biezace zadanie - przy job_id tylko gdy to wlasnie ono trwa (spoznione
    klikniecie nie zatrzyma kolejnego zadania). Zwraca False, gdy nic nie przerwano."""
    with _COND:
        run = next((j for j in _JOBS if j.status == RUNNING), None)
        if run is None or (job_id is not None and run.id != job_id):
            return False
    cancel.request()  # poza blokada - wola silnik przez HTTP
    return True


def set_paused(paused: bool) -> None:
    with _COND:
        _STATE_INFO["paused"] = paused
        _save()
        _COND.notify_all()


def pause_now() -> Optional[int]:
    """Pauza "od razu" (przycisk na dole panelu bocznego, np. na czas gry): kolejka staje, biezace
    zadanie jest przerywane i wraca na swoje miejsce (po wznowieniu liczy sie od nowa),
    a silnik oddaje pamiec karty. Nic nie dzieje sie samo - tylko na klikniecie.
    Zwraca id przerwanego zadania albo None."""
    set_paused(True)
    with _COND:
        run = next((j for j in _JOBS if j.status == RUNNING), None)
    if run is not None:
        _REQUEUE.set()
        cancel.request()  # poza blokada - wola silnik przez HTTP
    threading.Thread(target=_free_gpu, name="pauza-vram", daemon=True).start()
    return run.id if run else None


def _free_gpu(timeout: float = 60.0) -> None:
    """Po pauzie: gdy przerwane zadanie sie zwinie, modele ComfyUI wychodza z pamieci karty
    (procesy glosu i ruchu ust zabija juz cancel.request)."""
    deadline = time.time() + timeout
    while time.time() < deadline and any(j.status == RUNNING for j in snapshot()):
        time.sleep(0.5)
    from .comfy_client import ComfyClient

    cfg = config.SETTINGS["comfyui"]
    client = ComfyClient(cfg["host"], cfg["port"])
    if client.system_stats():
        client.free_memory()


def paused() -> bool:
    return bool(_STATE_INFO["paused"])


def clear_finished() -> None:
    with _COND:
        _JOBS[:] = [j for j in _JOBS if j.status in (PENDING, RUNNING)]
        _save()


def snapshot() -> List[Job]:
    with _COND:
        return list(_JOBS)


def latest_finished(kind: str) -> Optional[Job]:
    done = [j for j in snapshot() if j.kind == kind and j.status in (DONE, ERROR, STOPPED)]
    return max(done, key=lambda j: j.finished or 0) if done else None


def remaining_seconds() -> float:
    """Szacowany czas do konca calej kolejki."""
    total = 0.0
    for j in snapshot():
        if j.status == RUNNING:
            total += j.eta() or 0.0
        elif j.status == PENDING:
            total += j.estimate()
    return total


def fmt_time(secs: Optional[float]) -> str:
    if secs is None:
        return "—"
    secs = int(secs)
    if secs >= 3600:
        return f"{secs // 3600} h {secs % 3600 // 60:02d} min"
    return f"{round(secs / 60)} min" if secs >= 90 else f"{secs} s"


# ----------------------------- watek roboczy -----------------------------

def _next() -> Optional[Job]:
    if _STATE_INFO["paused"]:
        return None
    pending = sorted((j for j in _JOBS if j.status == PENDING), key=lambda j: j.id)
    # reczne zadania przed zadaniami autopilota (app/autopilot.py)
    return next((j for j in pending if not (j.params or {}).get("autopilot")),
                pending[0] if pending else None)


def _run(job: Job) -> None:
    def cb(frac: float, desc: str = "") -> None:
        cancel.check()
        job.progress = min(max(frac, 0.0), 1.0)
        if job.mark_t is None and job.progress >= 0.1:
            job.mark_t, job.mark_p = time.time(), job.progress
        if desc:
            job.desc = desc

    runner = _RUNNERS.get(job.kind)
    history.job_context(job.params)  # poziom tresci i polecenie trafiaja do historii
    try:
        refusal = _prompt_refusal(job.params)
        if refusal:
            raise ValueError(refusal)
        if runner is None:
            raise RuntimeError(f"Unknown job type: {job.kind}")
        ok, files, notes, error = runner(job.params, cb)
    except cancel.Cancelled:
        ok, files, notes, error = False, [], [], ""
    except Exception as e:  # noqa: BLE001 - blad jednego zadania nie zatrzymuje kolejki
        ok, files, notes, error = False, [], [], str(e)
    finally:
        history.job_context(None)
    with _COND:
        job.finished = time.time()
        job.files, job.notes = files, notes
        # pauza "od razu" (pause_now): niedokonczone zadanie wraca na swoje miejsce; gotowe
        # w chwili pauzy zostaje gotowe
        requeue = cancel.requested() and _REQUEUE.is_set() and not ok
        if requeue:
            job.status, job.progress, job.desc, job.error = PENDING, 0.0, "", ""
            job.started = job.finished = job.mark_t = None
            job.mark_p, job.files, job.notes = 0.0, [], []
        elif cancel.requested() and not _REQUEUE.is_set():
            job.status, job.error = STOPPED, "Stopped."
        elif ok:
            job.status, job.progress = DONE, 1.0
            # srednia ruchoma czasu na jednostke pracy - dokladniejsze ETA kolejki
            key = _rate_key(job.kind, job.params)
            _RATE[key] = 0.7 * _RATE.get(key, job.elapsed()) + \
                0.3 * job.elapsed() / max(job.units(), 1.0)
        else:
            job.status, job.error = ERROR, error or "Unknown error."
        _REQUEUE.clear()
        if not requeue:  # zadanie wracajace do kolejki potrzebuje swoich zdjec
            shutil.rmtree(FILES_DIR / str(job.id), ignore_errors=True)
        finished = [j for j in _JOBS if j.status in (DONE, ERROR, STOPPED)]
        for old in finished[:-_KEEP_FINISHED]:
            _JOBS.remove(old)
        _save()


def _worker() -> None:
    while True:
        job = None
        try:
            with _COND:
                job = _next()
                while job is None:
                    _COND.wait(timeout=2.0)
                    job = _next()
                job.status, job.started, job.progress, job.desc = RUNNING, time.time(), 0.0, ""
                job.mark_t, job.mark_p = None, 0.0
                cancel.reset()
                _REQUEUE.clear()
                _save()
            from . import llm  # pomocnik promptu na karcie oddaje VRAM przed generowaniem

            llm.release_gpu()
            _run(job)
        except BaseException as e:  # noqa: BLE001 - watek kolejki nie moze umrzec
            print(f"[queue] Worker error: {e!r}", flush=True)
            with _COND:
                if job is not None and job.status == RUNNING:
                    job.status, job.error, job.finished = ERROR, f"Queue error: {e}", time.time()
                    _save()
            time.sleep(1.0)


def start() -> None:
    """Wczytuje zapisana kolejke i uruchamia watek roboczy (raz, przy starcie aplikacji)."""
    global _started
    if _started:
        return
    _started = True
    with _COND:
        _load()
    threading.Thread(target=_worker, name="kolejka", daemon=True).start()
