# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Seria z polecenia (dawniej Auto-plan): polecenie po polsku -> lista unikalnych promptow ->
cala seria do kolejki.

Przyklad: "7 filmów nad morzem w bikini i 19 zdjęć w stroju koronkowym". Model tekstowy
(app/llm.py) najpierw rozbija polecenie na grupy (rodzaj, liczba, temat, format, dlugosc,
dzwiek - sam dobiera, czego polecenie nie mowi), potem dla kazdej grupy pisze prompty
paczkami po kilka - kazdy inny (poza, akcja, kadr, swiatlo, detale stroju i miejsca), z lista
juz napisanych, zeby sie nie powtarzaly. Odpowiedzi sa wymuszone schematem JSON, wiec lista
zawsze sie zgadza. Do polecenia mozna dolaczyc do 10 zdjec-referencji ([1], [2]...) - trafiaja
do kazdego zadania, ktore je wskazuje.

Tylko tresci SFW (app/content.py): polecenie z prosba o nagosc jest odrzucane, model dostaje
regule SFW, a kazdy prompt jest sprawdzany. Zadanie w kolejce pamieta cale polecenie (params
"idea"; trafia tez do historii).

Planowanie trwa od kilkudziesieciu sekund do kilku minut, dlatego dziala w tle (start /
status) - interfejs pokazuje postep i nie blokuje sie. Granica z llm.blocked() obowiazuje
dla polecenia i kazdego promptu.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Callable, List, Optional, Tuple

from . import characters, content, llm, queue_manager, refs, video_gen

IMAGE_FORMATS = ["Square 1:1", "Landscape 16:9", "Portrait 9:16", "Portrait 3:4"]
VIDEO_FORMATS = ["Landscape 16:9", "Portrait 9:16", "Square 1:1", "Portrait 3:4"]
KINDS = {"image": "Photo", "video": "Video"}
_VIDEO_KINDS = {"video", "film", "wideo"}
_FORMAT_ALIASES = {"Poziomo 16:9": "Landscape 16:9", "Pionowo 9:16": "Portrait 9:16",
                   "Kwadrat 1:1": "Square 1:1", "Portret 3:4": "Portrait 3:4"}
MAX_ITEMS = 300
_BATCH = {"image": 5, "video": 4}

ProgressCb = Callable[[float, str], None]

_ORDER_SCHEMA = {
    "type": "object",
    "properties": {"groups": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
        "type": "object",
        "properties": {
            "kind": {"enum": ["image", "video"]},
            "count": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS},
            "theme": {"type": "string"},
            "format": {"enum": ["Portrait 9:16", "Landscape 16:9", "Square 1:1", "Portrait 3:4"]},
            "seconds": {"type": "integer", "minimum": 0, "maximum": 15},
            "sound": {"type": "boolean"}},
        "required": ["kind", "count", "theme", "format", "seconds", "sound"]}}},
    "required": ["groups"]}

_ORDER_SYSTEM = (
    "You turn a user's order (written in any language, possibly colloquial - take every word "
    "literally) into a production plan of photos and videos. Split it into groups; each "
    "group is one kind ('image' for photos, 'video' for videos) with one "
    "theme. For each group give: count - exactly the number the user asked for (1 if no "
    "number); theme - a self-contained description in English of what to show, keeping EVERY "
    "detail and style word the user gave (place, outfit, mood, people, reference photo "
    "markers like [1]) but without the count, format or length; if a group "
    "refers to another one ('from the same session', 'in the same style', 'also'), copy that "
    "group's full description into its theme; format - what the user asked for ('vertical' = "
    "'Portrait 9:16', 'horizontal' = 'Landscape 16:9', 'square' = 'Square 1:1', 'portrait' = "
    "'Portrait 3:4'); if they did not say, choose the best one for the content: "
    "vertical 'Portrait 9:16' for people, fashion and social media, horizontal 'Landscape 16:9' "
    "for landscapes, places and cinematic scenes, 'Square 1:1' for products. A format attached "
    "to photos applies only to those photos, and a format attached to videos applies only "
    "to those videos: 'one square photo and one landscape video' means Square 1:1 for the "
    "image group and Landscape 16:9 for the video group. Only a format explicitly stated "
    "for the whole order applies to every group; seconds - for videos the length the user "
    "asked for, otherwise choose 3-15 to fit the action (5 for one simple action, more for "
    "several actions or speech); 0 for photos; sound - false only if the user asks for silent "
    "videos (without audio, silent), otherwise true. When the user says 'X photos and X films' "
    "about one theme, make two groups with the same theme. Follow the order exactly."
    + content.RULE)

_BATCH_RULES = (
    "\n\nNow you write a SERIES of different prompts for one theme. Each prompt follows all "
    "rules above and is complete on its own. Make every prompt clearly different from the "
    "others: pose or action, camera angle and framing, place details, lighting and time of day, "
    "outfit details and expression - while keeping the theme and its style. A video prompt is "
    "ONE single shot - one paragraph of 50-110 words, no shot labels. Every array item is "
    "one complete prompt. Preserve any exact age, gender and appearance supplied in the "
    "character identity or theme; never invent a different age or gender. If the user requests "
    "a silent video, describe no sounds or speech and end with 'No audio.' Reply with JSON only.")
_NO_REFS = " There are no reference photos here: never use [1], [2] markers."


def _prompts_schema(n: int) -> dict:
    return {"type": "object", "properties": {"prompts": {
        "type": "array", "minItems": n, "maxItems": n,
        "items": {"type": "string", "minLength": 80}}}, "required": ["prompts"]}


def _json(text: str) -> Optional[dict]:
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return None


def parse_order(order: str, progress_cb: Optional[ProgressCb] = None
                ) -> Tuple[bool, object]:
    """Grupy z polecenia: [{kind, count, theme, format, seconds, sound}] albo komunikat."""
    ok, text = llm.ask(_ORDER_SYSTEM, f"Order: {order}", schema=_ORDER_SCHEMA, max_tokens=1000,
                       check=lambda t: _json(t) is not None, progress_cb=progress_cb,
                       desc="Planning the series...", frac=0.02)
    if not ok:
        return False, text
    data = _json(text)
    if not data or not data.get("groups"):
        return False, "Could not create a plan. Try a simpler request, such as “5 photos of …”."
    return True, data["groups"]


def write_series(kind: str, theme: str, count: int, done: int, total: int,
                 progress_cb: Optional[ProgressCb] = None, order: str = "",
                 n_refs: int = 0) -> List[str]:
    """`count` unikalnych promptow dla tematu (paczkami). done/total - do paska postepu.
    order - cale polecenie uzytkownika jako kontekst ("3 filmy z tej samej sesji" nie ma
    tresci bez reszty polecenia, a model przy niejasnym temacie odplywal w inne klimaty).
    n_refs - liczba dolaczonych zdjec: ich numery zostaja w promptach, inne znikaja.
    Regula SFW dla modelu; prompt z nagoscia odpada (app/content.py)."""
    out: List[str] = []
    system = llm._SYSTEM[kind] + _BATCH_RULES + (llm.refs_rule(n_refs) if n_refs else _NO_REFS) \
        + content.RULE
    tries = 3 * (count // _BATCH[kind] + 1)  # odrzucone prompty (np. po polsku) - nie w kolko
    while len(out) < count and tries > 0:
        tries -= 1
        n = min(_BATCH[kind], count - len(out))
        used = "\n".join(f"- {' '.join(p.split()[:14])}…" for p in out[-12:])
        user = ((f"Whole order from the user: {order}\nThis group is only the "
                 f"{'videos' if kind == 'video' else 'photos'}; take place, outfit, people "
                 "and style from the order. Do not add anything the order does not ask for.\n"
                 if order else "")
                + f"Theme: {theme}\nWrite {n} different prompts in English (numbers "
                f"{len(out) + 1}-{len(out) + n} of {count})."
                + (f"\nAlready written - do not repeat these ideas:\n{used}" if used else ""))
        ok, text = llm.ask(system, user, schema=_prompts_schema(n), max_tokens=260 * n + 200,
                           check=lambda t: _json(t) is not None, progress_cb=progress_cb,
                           desc=f"Writing prompts {done + len(out) + 1}–{done + len(out) + n} "
                                f"of {total}...",
                           frac=0.05 + 0.95 * (done + len(out)) / max(total, 1))
        data = _json(text) if ok else None
        if not data:
            break  # blad silnika - zwracamy to, co juz jest
        for p in data.get("prompts", [])[:n]:
            # bez etykiet ujec i numerow nieistniejacych zdjec; urywki (np. sam dzwiek) odpadaja
            p = re.sub(r"(?im)^\s*(?:Uj[ęe]cie|Shot) \d+\s*:\s*", "", llm._clean(p)).strip()
            p = re.sub(r"\s*\[(\d{1,2})\]",
                       lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_refs else "", p)
            p = content.enforce(p) or ""
            if (len(p.split()) >= 25 and llm.english(p) and not llm.blocked(p)
                    and " ".join(p.split()).casefold() not in {x.casefold() for x in out}):
                out.append(" ".join(p.split()))
    return out[:count]


# Czy polecenie w ogole mowi o formacie; jesli nie, a temat to osoba - pion (model mimo
# instrukcji wybieral kwadrat dla zdjec modelek)
_FORMAT_WORDS = re.compile(r"pion|poziom|kwadrat|portret|panoram|9\s*:\s*16|16\s*:\s*9|1\s*:\s*1|"
                           r"3\s*:\s*4|square|vertical|horizontal|landscape|portrait", re.I)
_PERSON = re.compile(r"kobiet|dziewczyn|modelk|mężczyzn|facet|chłopak|osob|postać|postaci|"
                     r"influencer|\bona\b|\bon\b|woman|girl|model|\bman\b|person|\[\d+\]", re.I)


def _auto_format(fmt: str, theme: str, order: str) -> str:
    if fmt == "Square 1:1" and not _FORMAT_WORDS.search(order) and _PERSON.search(theme):
        return "Portrait 9:16"
    return fmt


def _explicit_media_formats(order: str) -> dict:
    """Recover unambiguous formats directly attached to a media kind in the order.

    Leave complex orders with multiple formats for the same kind to the group parser.
    In particular, a photo's explicit format must not leak into a video group.
    """
    formats = {
        "Square 1:1": r"square|kwadrat\w*|1\s*:\s*1",
        "Landscape 16:9": r"landscape|horizontal|poziom\w*|16\s*:\s*9",
        "Portrait 9:16": r"vertical|pion\w*|9\s*:\s*16",
        "Portrait 3:4": r"portrait|portret\w*|3\s*:\s*4",
    }
    kinds = {"image": r"photos?|photographs?|images?|zdjęci\w*|fotk\w*",
             "video": r"videos?|films?|film\w*|wideo"}
    found = {kind: set() for kind in kinds}
    for kind, media in kinds.items():
        for fmt, words in formats.items():
            # Optional modifiers cover "square product photo" and "photo in 1:1 format".
            before = rf"\b(?:{words})\s+(?:(?:product|silent|produkt\w*)\s+)?(?:{media})\b"
            after = rf"\b(?:{media})\s+(?:(?:in|w)\s+)?(?:(?:format|formacie)\s+)?(?:{words})\b"
            if re.search(before, order, re.I) or re.search(after, order, re.I):
                found[kind].add(fmt)
    return {kind: next(iter(values)) for kind, values in found.items() if len(values) == 1}


def make_plan(order: str, n_refs: int = 0,
              progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, object]:
    """Caly plan: wiersze [rodzaj, format, sekundy, dzwiek, prompt] albo komunikat bledu."""
    order = (order or "").strip()
    if not order:
        return False, "Enter a request, such as “10 beach photos at sunset and 3 videos”."
    if llm.blocked(order):
        return False, llm.BLOCK_MSG
    refusal = content.check(order)
    if refusal:
        return False, refusal
    err = llm.ensure(progress_cb)
    if err:
        return False, err
    ok, groups = parse_order(order, progress_cb)
    if not ok:
        return False, groups
    total = sum(int(g["count"]) for g in groups)
    if total > MAX_ITEMS:
        return False, f"The plan contains {total} items. The limit is {MAX_ITEMS} per plan."
    rows: List[list] = []
    explicit_formats = {kind: fmt for kind, fmt in _explicit_media_formats(order).items()
                        if sum(g["kind"] == kind for g in groups) == 1}
    for g in groups:
        kind = g["kind"]
        if kind == "video":
            fmt = g["format"] if g["format"] in VIDEO_FORMATS else "Portrait 9:16"
            secs = min(max(int(g.get("seconds") or 5), 2), video_gen.MAX_SECONDS)
            sound = "Yes" if g.get("sound", True) else "No"
        else:
            fmt, secs, sound = g["format"], 0, "—"
        fmt = _auto_format(explicit_formats.get(kind, fmt), g["theme"], order)
        prompts = write_series(kind, g["theme"], int(g["count"]), len(rows), total, progress_cb,
                               order, n_refs)
        rows += [[KINDS[kind], fmt, secs, sound, p] for p in prompts]
    if not rows:
        return False, "The model returned no usable prompts. Try again."
    if progress_cb:
        progress_cb(1.0, "Plan ready")
    return True, rows


def estimate(rows: List[list], speed: str = video_gen.QUALITY) -> float:
    """Szacowany czas generowania calego planu (s), wg sredniej z kolejki."""
    video = queue_manager.rate("wideo", {"speed": speed})
    return sum(video * max(int(float(r[2] or 5)), 1) if str(r[0]).lower() in _VIDEO_KINDS
               else queue_manager.rate("obraz") for r in rows)


def enqueue(rows: List[list], ref_paths: Optional[List[str]] = None,
            order: str = "", character: Optional[dict] = None,
            extra_params: Optional[dict] = None, prefix: str = "plan",
            speed: str = video_gen.QUALITY) -> Tuple[int, int]:
    """Dodaje wiersze planu [rodzaj, format, sekundy, dzwiek, prompt] do kolejki. Zdjecia-
    referencje dostaje kazde zadanie, ktore je wskazuje numerem (albo kazde, gdy zaden prompt
    nie ma numerow). order - polecenie uzytkownika, zapamietane w kazdym zadaniu. Pomija puste,
    zablokowane i z nagoscia (tylko SFW). character - postac podana wprost (autopilot) zamiast
    wybranej w panelu bocznym; extra_params - dodatkowe parametry zadan (autopilot: dzien,
    sesja); prefix - poczatek nazwy zadania; speed - tryb filmow (app/video_gen.py MODES).
    Zwraca (dodane, pominiete)."""
    ref_paths = list(ref_paths or [])
    any_marks = any(refs.numbers(str(r[-1] or "")) for r in rows)
    added = skipped = 0
    for r in rows:
        r = list(r)
        if len(r) >= 6:  # wiersz z dawna kolumna "Tresc" (przed usunieciem trybu 18+)
            del r[4]
        kind, fmt, secs, sound, prompt = (r + [None] * 5)[:5]
        prompt = str(prompt or "").strip()
        fmt = _FORMAT_ALIASES.get(fmt, fmt)
        if not prompt or llm.blocked(prompt) or content.asks_nudity(prompt):
            skipped += 1
            continue
        files = {"refs": ref_paths} if ref_paths and (refs.numbers(prompt) or not any_marks) \
            else None
        extra = {**({"idea": order} if order else {}), **(extra_params or {})}
        if str(kind).lower() in _VIDEO_KINDS:
            secs = min(max(int(float(secs or 5)), 1), video_gen.MAX_SECONDS)
            params = {"prompt": prompt, "seconds": secs,
                      "aspect": fmt if fmt in VIDEO_FORMATS else "Portrait 9:16",
                      "sound": str(sound).strip().lower() not in ("nie", "no", "false"),
                      "speed": speed, **extra}
            who = _who(params, character)
            queue_manager.add("wideo", f"{prefix} · video {secs} s{video_gen.mode_tag(speed)}"
                                       f"{who}: {prompt}", params, files=files)
        else:
            params = {"prompt": prompt,
                      "format": fmt if fmt in IMAGE_FORMATS else "Portrait 9:16", **extra}
            who = _who(params, character)
            queue_manager.add("obraz", f"{prefix} · image {fmt}{who}: {prompt}", params,
                              files=files)
        added += 1
    return added, skipped


def _who(params: dict, character: Optional[dict]) -> str:
    """Postac zadania: podana wprost (autopilot) albo wybrana w panelu bocznym."""
    if character is None:
        return characters.tag(params)
    params["character"] = character["slug"]
    return f" · {character['name']}"


# ----------------------------- planowanie w tle -----------------------------

_RUN = {"id": 0, "running": False, "progress": 0.0, "desc": "", "ok": None, "msg": "",
        "rows": [], "refs": [], "added": 0, "auto_add": False, "order": "",
        "speed": video_gen.QUALITY}
_RUN_LOCK = threading.Lock()


def start(order: str, ref_paths: List[str], auto_add: bool,
          speed: str = video_gen.QUALITY) -> Tuple[bool, str]:
    """Uruchamia planowanie w tle. speed - tryb filmow. Zwraca (ok, komunikat)."""
    order = llm.mark_refs((order or "").strip(), len(ref_paths))  # "z drugiego zdjęcia" -> [2]
    if not order:
        return False, "Enter a request, such as “10 beach photos at sunset and 3 videos”."
    if llm.blocked(order):
        return False, llm.BLOCK_MSG
    refusal = content.check(order)
    if refusal:
        return False, refusal
    if len(ref_paths) > refs.MAX_REFS:
        return False, f"Up to {refs.MAX_REFS} reference images are supported; {len(ref_paths)} were attached."
    missing = [n for n in refs.numbers(order) if n > len(ref_paths)]
    if missing:
        return False, (f"The request refers to image [{missing[0]}], but only {len(ref_paths)} "
                       "reference images were attached.")
    with _RUN_LOCK:
        if _RUN["running"]:
            return False, "A plan is already being prepared. Wait for it to finish."
        _RUN.update(id=_RUN["id"] + 1, running=True, progress=0.0, desc="Start...", ok=None,
                    msg="", rows=[], refs=list(ref_paths), added=0, auto_add=auto_add,
                    order=order, speed=speed)
    threading.Thread(target=_work, args=(order, list(ref_paths), auto_add, speed),
                     name="auto-plan", daemon=True).start()
    return True, ""


def _work(order: str, ref_paths: List[str], auto_add: bool,
          speed: str = video_gen.QUALITY) -> None:
    def cb(frac: float, desc: str = "") -> None:
        _RUN["progress"] = min(max(frac, 0.0), 1.0)
        if desc:
            _RUN["desc"] = desc

    try:
        ok, rows = make_plan(order, len(ref_paths), cb)
    except Exception as e:  # noqa: BLE001 - blad planu nie moze zabic aplikacji
        ok, rows = False, f"Planning failed: {e}"
    added = 0
    if ok and auto_add:
        added, _ = enqueue(rows, ref_paths, order, speed=speed)
    _RUN.update(running=False, ok=ok, rows=rows if ok else [], msg="" if ok else str(rows),
                added=added, progress=1.0)


def status() -> dict:
    return dict(_RUN)
