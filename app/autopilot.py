# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Autopilot modelek: karta persony, rytm publikacji i zapas tresci do przodu.

Kazda modelka (postac z LoRA) moze miec persone - characters/<slug>/persona.json: wiek, pochodzenie,
charakter, styl, garderoba, miejsca, tematy, rytm (zdjec i filmow dziennie, dlugosc filmu)
i zapas (na ile dni do przodu tresci maja byc gotowe). Sesje sa tylko SFW: temat w stroju
z garderoby, prompty z regula SFW (app/content.py), a kazdy wynik przechodzi straz SFW
(app/qc.py) - nagi nie liczy sie do zapasu i nie trafia do paczki.

Kalendarz zamiast publikacji: kazde zadanie autopilota dostaje dzien ("day"), na ktory jest.
Zapas = ile kolejnych dni od dzisiaj ma juz komplet gotowych tresci (po kontroli jakosci).
Dni mijaja, okno przesuwa sie do przodu i autopilot dorabia brakujace.

Watek w tle co TICK sekund, gdy autopilot jest wlaczony (Pulpit), nie ma pauzy i kolejka
jest pusta: wybiera modelke z autopilotem i najkrotszym zapasem, a dla pierwszego niepelnego dnia
model tekstowy pisze temat jednej sesji z persony (inny niz ostatnie), planer - prompty zdjec
i filmow, wszystko trafia do kolejki. Reczne zadania maja pierwszenstwo (queue_manager._next),
a nowa sesja powstaje dopiero, gdy kolejka jest pusta. Nic nie publikuje sie samo.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import characters, content, history, llm, packs, planner, prefs, queue_manager, video_gen

TICK = 30.0            # co ile sekund autopilot sprawdza zapas
HOURS_PER_DAY = 16.0   # ile godzin pracy karty na dobe zakladamy (noc + czas bez gry)
# narzut straży SFW (pomiar 31.10: opis stroju ok. 35 s, czysty wynik - dwa opisy; film - klatka
# startowa i dwie klatki gotowego filmu)
QC_SECONDS = {"image": 70.0, "video": 210.0}

DEFAULTS = {"age": 24, "origin": "", "personality": "", "style": "", "wardrobe": [],
            "places": [], "pillars": [], "photos_per_day": 3, "videos_per_day": 1,
            "video_seconds": 6, "buffer_days": 7, "enabled": False,
            "language": "en",  # jezyk opisow postow (app/packs.py)
            "video_speed": video_gen.QUALITY}  # tryb filmow (app/video_gen.py MODES)
_LISTS = ("wardrobe", "places", "pillars")
_LIMITS = {"age": (21, 60), "photos_per_day": (0, 20), "videos_per_day": (0, 5),
           "video_seconds": (2, 15), "buffer_days": (1, 30)}

ProgressCb = Callable[[float, str], None]


# ----------------------------- persona -----------------------------

def _file(slug: str) -> Path:
    return characters.DIR / slug / "persona.json"


def persona(slug: str) -> dict:
    try:
        data = json.loads(_file(slug).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return clean(data if isinstance(data, dict) else {})


def clean(data: dict) -> dict:
    """Persona w dozwolonych granicach: liczby w zakresach, listy bez pustych pozycji. Pola
    spoza DEFAULTS (np. dawny udzial sesji 18+) odpadaja."""
    out = {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
    for key, (lo, hi) in _LIMITS.items():
        try:
            out[key] = int(min(max(int(float(out[key])), lo), hi))
        except (TypeError, ValueError):
            out[key] = DEFAULTS[key]
    for key in _LISTS:
        items = (out[key].splitlines() if isinstance(out[key], str)
                 else out[key] if isinstance(out[key], (list, tuple)) else [])
        out[key] = [" ".join(str(i).split()) for i in items if str(i).strip()][:20]
    for key in ("origin", "personality", "style"):
        out[key] = " ".join(str(out[key] or "").split())[:300]
    out["enabled"] = bool(out["enabled"])
    out["language"] = "pl" if str(out.get("language", "")).lower().startswith(("pl", "pol")) \
        else "en"
    out["video_speed"] = video_gen.FAST if out.get("video_speed") == video_gen.FAST \
        else video_gen.QUALITY
    return out


def save_persona(slug: str, data: dict) -> dict:
    p = clean(data)
    _file(slug).write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


_PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {"type": "integer", "minimum": 21, "maximum": 45},
        "origin": {"type": "string"}, "personality": {"type": "string"},
        "style": {"type": "string"},
        "wardrobe": {"type": "array", "minItems": 5, "maxItems": 10, "items": {"type": "string"}},
        "places": {"type": "array", "minItems": 5, "maxItems": 10, "items": {"type": "string"}},
        "pillars": {"type": "array", "minItems": 3, "maxItems": 6, "items": {"type": "string"}}},
    "required": ["age", "origin", "personality", "style", "wardrobe", "places", "pillars"]}
_PERSONA_SYSTEM = (
    "You create a persona for a fictional virtual influencer (an AI model aged 21 or "
    "more) who posts SFW content on Instagram and TikTok. Write every value in simple, "
    "correct English (short phrases, "
    "real words only). Preserve the supplied character's gender and appearance. Return JSON: "
    "age - their age (21 or more); origin - where they are from and "
    "where they live (real places); personality - two or three traits; style - their fashion "
    "style in a few words; wardrobe - 5 to 10 signature outfits, each one simple and concrete "
    "(for example 'white linen midi dress', 'black swimwear', 'beige sweater and light "
    "jeans'); places - 5 to 10 real, photogenic places where they shoot photos (for example "
    "'Tulum beach', 'a hotel terrace in Dubai', 'a yacht on the French Riviera'); pillars - 3 to 6 "
    "content themes of their account (for example 'travel', 'beach fashion', 'morning "
    "routine'). Make it coherent and specific. Follow the user's idea when there is one."
    + content.RULE)


def suggest(ch: dict, hint: str = "",
            progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, object]:
    """Persona zaproponowana przez model tekstowy (z wygladu postaci i pomyslu uzytkownika).
    Zwraca (True, persona do sprawdzenia) albo (False, komunikat)."""
    if llm.blocked(hint):
        return False, llm.BLOCK_MSG
    refusal = content.check(hint)
    if refusal:
        return False, refusal
    err = llm.ensure(progress_cb)
    if err:
        return False, err
    user = (f"Character gender: {ch.get('gender') or 'not specified'}. "
            f"Character appearance: {ch.get('look') or 'adult person'}.\n"
            f"User's idea (any language, may be empty): {hint.strip() or '-'}")
    ok, text = llm.ask(_PERSONA_SYSTEM, user, schema=_PERSONA_SCHEMA, max_tokens=700,
                       progress_cb=progress_cb, desc="Writing the persona...", temp=0.4)
    data = planner._json(text) if ok else None
    if not data:
        return False, text if not ok else "The model returned no persona. Try again."
    return True, clean({**persona(ch["slug"]), **data})


# ----------------------------- kalendarz i zapas -----------------------------

def _entry_kind(entry: dict) -> str:
    return "video" if str(entry.get("mode", "")).startswith("video") else "image"


def day_counts(slug: str, entries: Optional[List[dict]] = None,
               jobs: Optional[List[queue_manager.Job]] = None,
               queued: bool = True) -> Dict[str, Dict[str, int]]:
    """Dzien -> {"image": n, "video": n}: gotowe tresci autopilota tej modelki (pliki na dysku,
    bez tych, ktore nie przeszly kontroli), a przy queued - takze zadania w kolejce."""
    out: Dict[str, Dict[str, int]] = {}
    for e in history.recent(5000) if entries is None else entries:
        if not (e.get("autopilot") and e.get("profile") == slug and e.get("day")):
            continue
        if content.legacy_adult(e) or (e.get("qc") or {}).get("safe") is not True:
            continue
        if not any(Path(f).exists() for f in e.get("files") or []):
            continue
        out.setdefault(e["day"], {"image": 0, "video": 0})[_entry_kind(e)] += 1
    if queued:
        for j in queue_manager.snapshot() if jobs is None else jobs:
            p = j.params or {}
            if j.status in (queue_manager.PENDING, queue_manager.RUNNING) \
                    and p.get("autopilot") and p.get("character") == slug and p.get("day"):
                out.setdefault(p["day"], {"image": 0, "video": 0})[
                    "video" if j.kind == "wideo" else "image"] += 1
    return out


def _days(today: dt.date, n: int) -> List[str]:
    return [(today + dt.timedelta(days=d)).isoformat() for d in range(n)]


def missing(p: dict, counts: Dict[str, Dict[str, int]], day: str) -> Tuple[int, int]:
    have = counts.get(day, {})
    return (max(p["photos_per_day"] - have.get("image", 0), 0),
            max(p["videos_per_day"] - have.get("video", 0), 0))


def buffer_days(p: dict, counts: Dict[str, Dict[str, int]], today: dt.date) -> int:
    """Ile kolejnych dni od dzisiaj ma komplet tresci."""
    n = 0
    for day in _days(today, 60):
        if missing(p, counts, day) != (0, 0):
            break
        n += 1
    return n


def gap(p: dict, counts: Dict[str, Dict[str, int]],
        today: dt.date) -> Optional[Tuple[str, int, int]]:
    """Pierwszy niepelny dzien w oknie zapasu: (dzien, brakujace zdjecia, brakujace filmy)."""
    for day in _days(today, p["buffer_days"]):
        photos, videos = missing(p, counts, day)
        if photos or videos:
            return day, photos, videos
    return None


def daily_seconds(p: dict) -> float:
    """Szacowany czas pracy karty na jeden dzien tresci tej modelki."""
    image = queue_manager.rate("obraz") + QC_SECONDS["image"]
    video = (queue_manager.rate("wideo", {"speed": p.get("video_speed")}) * p["video_seconds"]
             + QC_SECONDS["video"])
    return p["photos_per_day"] * image + p["videos_per_day"] * video


def capacity() -> Tuple[float, bool]:
    """(godziny pracy karty na dobe dla wszystkich modelek z autopilotem, czy sie miesci)."""
    hours = sum(daily_seconds(persona(c["slug"])) for c in characters.all_characters()
                if persona(c["slug"])["enabled"]) / 3600
    return hours, hours <= HOURS_PER_DAY


# ----------------------------- sesja -----------------------------

_SESSION_SCHEMA = {"type": "object", "properties": {"theme": {"type": "string"}},
                   "required": ["theme"]}
_SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer",
            7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def _sessions_file(slug: str) -> Path:
    return characters.DIR / slug / "sessions.jsonl"


def recent_themes(slug: str, n: int = 8) -> List[str]:
    try:
        lines = _sessions_file(slug).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line)["theme"])
        except (ValueError, KeyError):
            continue
        if len(out) >= n:
            break
    return out


def _summary(p: dict) -> str:
    lists = {k: "; ".join(p[k]) or "-" for k in _LISTS}
    return (f"Age: {p['age']}. Origin: {p['origin'] or '-'}. Personality: "
            f"{p['personality'] or '-'}. Style: {p['style'] or '-'}. Wardrobe: "
            f"{lists['wardrobe']}. Places: {lists['places']}. Content themes: {lists['pillars']}.")


def session_theme(p: dict, day: str, recent: List[str],
                  progress_cb: Optional[ProgressCb] = None,
                  character: Optional[dict] = None) -> Tuple[bool, str]:
    """Temat jednej sesji: miejsce, stroj, czynnosc, nastroj - z persony."""
    season = _SEASONS[dt.date.fromisoformat(day).month]
    system = ("You plan one SFW photo and video session for a fictional adult virtual influencer. "
              "Reply with JSON: theme - one or two sentences in English: "
              "where the person is, what they wear, what they do, the mood and the light. Use their "
              "persona: a place from their places and an outfit from their wardrobe (or something "
              f"that clearly fits them) and one of their content themes. It is {season}. They are "
              "dressed in an outfit from their wardrobe (name it in the theme). Preserve the "
              "persona's exact age and the supplied character's gender and appearance. Make it clearly "
              "different from their recent sessions. Do not use their name." + content.RULE)
    user = (f"Character identity: {_identity(character or {}, p)}\n"
            f"Persona: {_summary(p)}\nRecent sessions (do not repeat): "
            + ("; ".join(recent) if recent else "-"))
    ok, text = llm.ask(system, user, schema=_SESSION_SCHEMA, max_tokens=300,
                       progress_cb=progress_cb, desc="Autopilot: planning a session...")
    data = planner._json(text) if ok else None
    theme = " ".join(str((data or {}).get("theme", "")).split())
    if not theme:
        return False, text if not ok else "The model returned no session theme."
    if llm.blocked(theme):
        return False, llm.BLOCK_MSG
    if content.asks_nudity(theme):  # model dopisal nagosc - takiej sesji nie planujemy
        return False, "The suggested theme failed the SFW check. Autopilot will retry on the next cycle."
    return True, theme


def _identity(ch: dict, p: dict) -> str:
    return (f"exactly {p['age']} years old; gender: {ch.get('gender') or 'not specified'}; "
            f"appearance: {ch.get('look') or 'not specified'}. Keep this exact identity in every prompt.")


def plan_session(ch: dict, p: dict, day: str, photos: int, videos: int,
                 progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """Jedna sesja modelki na dany dzien: temat, prompty, zadania w kolejce."""
    ok, theme = session_theme(p, day, recent_themes(ch["slug"]), progress_cb, character=ch)
    if not ok:
        return False, theme
    total = photos + videos
    context = f"Character identity: {_identity(ch, p)}\nSession: {theme}"
    rows: List[list] = []
    if photos:
        rows += [[planner.KINDS["image"], "Portrait 3:4", 0, "—", pr]
                 for pr in planner.write_series("image", theme, photos, 0, total, progress_cb,
                                                context)]
    if videos:
        rows += [[planner.KINDS["video"], "Portrait 9:16", p["video_seconds"], "Yes", pr]
                 for pr in planner.write_series("video", theme, videos, photos, total,
                                                progress_cb, context)]
    if not rows:
        return False, "The model returned no prompts for the session."
    session = f"{ch['slug']}-{day}-{int(time.time())}"
    added, _ = planner.enqueue(rows, [], theme, character=ch, prefix=f"🤖 {day}",
                               extra_params={"autopilot": True, "day": day,
                                             "session": session},
                               speed=p.get("video_speed", video_gen.QUALITY))
    if not added:
        return False, "No jobs were added: all suggested prompts were rejected."
    with open(_sessions_file(ch["slug"]), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "day": day, "session": session,
                            "theme": theme, "jobs": added}, ensure_ascii=False) + "\n")
    return True, f"{ch['name']}: session for {day} ({added} jobs) — {theme}"


# ----------------------------- watek w tle -----------------------------

STATUS = {"last": "", "when": 0.0}
_started = False


def enabled() -> bool:
    return bool(prefs.load().get("autopilot", False))


def tick(today: Optional[dt.date] = None) -> str:
    """Jeden obieg: zaplanuj sesje modelki z najkrotszym zapasem (gdy wolno). Zwraca opis."""
    if not enabled():
        return "disabled"
    if queue_manager.paused():
        return "paused"
    if any(j.status in (queue_manager.PENDING, queue_manager.RUNNING)
           for j in queue_manager.snapshot()):
        return "queue busy — the content buffer will be checked when it is empty"
    today = today or dt.date.today()
    active = [(ch, persona(ch["slug"])) for ch in characters.all_characters()
              if characters.usable(ch["slug"])]
    active = [(ch, p) for ch, p in active if p["enabled"]]
    # najpierw paczka dla dnia, ktory ma juz komplet gotowych tresci (app/packs.py)
    for ch, p in active:
        ready = day_counts(ch["slug"], queued=False)
        for day in _days(today, p["buffer_days"]):
            if missing(p, ready, day) == (0, 0) and not packs.day_done(ch["slug"], day):
                ok, msg = packs.build_day(ch["slug"], day)
                return msg if ok else f"❌ {ch['name']}: {msg}"
    candidates = []
    for ch, p in active:
        hole = gap(p, day_counts(ch["slug"]), today)
        if hole:
            ready = buffer_days(p, day_counts(ch["slug"], queued=False), today)
            candidates.append((ready, ch["name"], ch, p, hole))
    if not candidates:
        return "content buffer full"
    _, _, ch, p, (day, photos, videos) = min(candidates, key=lambda c: (c[0], c[1]))
    ok, msg = plan_session(ch, p, day, photos, videos)
    return msg if ok else f"❌ {ch['name']}: {msg}"


def _loop() -> None:
    while True:
        time.sleep(TICK)
        try:
            msg = tick()
        except Exception as e:  # noqa: BLE001 - autopilot nie moze zabic aplikacji
            msg = f"❌ autopilot error: {e}"
        if msg != STATUS["last"]:
            STATUS.update(last=msg, when=time.time())


def start() -> None:
    """Watek autopilota (raz, przy starcie aplikacji). Sam nic nie robi, dopoki autopilot nie
    zostanie wlaczony w zakladce Studio."""
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_loop, name="autopilot", daemon=True).start()
