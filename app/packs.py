# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""SFW content packs for Instagram and TikTok.

New packs use English artifact names and post types. The existing paczki/ library
location is retained without renaming legacy exports. Captions follow the selected
persona language and every pack includes an AI disclosure and publication guidance.
Unverified media requires a fresh safety check before it can be packaged.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import (characters, config, content, ffmpeg_utils, history, llm, metadata, planner, qc,
               trash, video_gen)

PACKS_DIR = config.ROOT / "paczki"
MAX_CAROUSEL = 20
AI_LINE = {"pl": "🤖 AI-generated character.", "en": "🤖 AI-generated character."}
AI_TAGS = ["aiinfluencer", "aimodel", "aigenerated"]
TIMES = {"carousel": "19:00", "reel": "12:30", "tiktok": "18:00"}
_IMAGES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class Item:
    path: str
    kind: str                      # image | video
    group: str                     # sesja (autopilot) albo wspolna grupa zaznaczenia
    theme: str = ""
    dialogue: List[str] = field(default_factory=list)
    ratio: str = ""                # proporcje obrazu, np. "3:4"
    prompt: str = ""               # prompt i polecenie z historii (ocena ryzyka)
    legacy_adult: bool = False     # wynik oznaczony jeszcze jako 18+ (sprzed usuniecia trybu)


# Scena, w ktorej "ubranie" bywa tylko rekwizytem (test 31.10: zdjecie spod koldry z doklejonym
# "fully clothed" wyszlo nago, a model wizyjny raz na kilka razy to przeoczal) - do paczki nie idzie
_RISKY = re.compile(r"\b(?:sheets?|duvet|blankets?|towels?|bath(?:tub)?|shower|naked|nude|"
                    r"topless|thong|undress\w*|nipples?|genital\w*|private (?:areas|parts))\b",
                    re.I)


def _ratio(path: str) -> str:
    from PIL import Image

    try:
        with Image.open(path) as im:
            w, h = im.size
    except OSError:
        return ""
    r = w / h
    return min({"1:1": 1.0, "3:4": 0.75, "4:5": 0.8, "9:16": 0.5625, "16:9": 1.7778}.items(),
               key=lambda kv: abs(kv[1] - r))[0]


def _plain(text: str) -> str:
    """Male litery bez polskich znakow (porownywanie imienia postaci w odmianie)."""
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text).lower().replace("ł", "l"))
    return "".join(c for c in text if not unicodedata.combining(c))


def _scene(prompt: str) -> str:
    """Sama scena z promptu: bez "Photo of <slowo-klucz> woman.", opisu wygladu postaci i stylu
    (slowo-klucz LoRA trafialo do hashtagow)."""
    text = re.sub(r"^Photo of \S+ (?:woman|man)\.\s*", "", str(prompt or ""))
    text = re.split(r"\bThe (?:woman|man) has\b|, ultra realistic", text)[0]
    text = re.sub(r"\b\w+_aist\b", "", text)
    return " ".join(text.split()[:70])


def _item(path: str, entry: dict, group: str) -> Item:
    kind = "video" if path.lower().endswith(".mp4") else "image"
    # temat do opisu: polecenie (sesja autopilota), a bez niego - prompt zdjecia (inaczej opis
    # wymyslal stroj z persony, ktorego na zdjeciu nie ma)
    theme = entry.get("idea") or _scene(entry.get("positive", ""))
    return Item(path=path, kind=kind, group=group,
                theme=theme, dialogue=list(entry.get("dialogue") or []),
                ratio=_ratio(path) if kind == "image" else "",
                prompt=f"{entry.get('positive', '')} {entry.get('idea', '')}",
                legacy_adult=content.legacy_adult(entry))


def items_for_day(slug: str, day: str, entries: Optional[List[dict]] = None) -> List[Item]:
    """Gotowe tresci autopilota modelki na dany dzien (bez tych, ktore nie przeszly kontroli)."""
    out = []
    for e in reversed(history.recent(5000) if entries is None else entries):
        if not (e.get("autopilot") and e.get("profile") == slug and e.get("day") == day):
            continue
        if (e.get("qc") or {}).get("safe") is not True or content.legacy_adult(e):
            continue
        for f in e.get("files") or []:
            if Path(f).exists():
                out.append(_item(f, e, e.get("session") or day))
    return out


def items_for_files(paths: List[str], entries: Optional[List[dict]] = None) -> List[Item]:
    """Zaznaczone pliki (Historia) - jedna grupa; poziom i temat z historii."""
    by_name = {}
    for e in history.recent(5000) if entries is None else entries:
        for f in e.get("files") or []:
            by_name.setdefault(Path(f).name, e)
    return [_item(p, by_name.get(Path(p).name, {}), "zaznaczone") for p in paths
            if Path(p).exists() and p.lower().endswith(_IMAGES + (".mp4",))]


def safety_check(items: List[Item], entries: Optional[List[dict]] = None
                  ) -> Tuple[List[Item], List[str]]:
    """Exclude adult legacy records and risky scenes; recheck unverified media.

    Videos use the mandatory full-duration sampling policy from qc.check_video.
    Returns approved items and notes explaining exclusions.
    """
    verdicts = {}
    for e in history.recent(5000) if entries is None else entries:
        for f in e.get("files") or []:
            if e.get("qc"):
                verdicts.setdefault(str(Path(f).resolve()).casefold(), e["qc"])
    kept: List[Item] = []
    notes: List[str] = []
    for it in items:
        name = Path(it.path).name
        if it.legacy_adult:
            notes.append(f"{name}: legacy 18+ output — skipped")
            continue
        risky = _RISKY.search(it.prompt or "")
        if risky:
            notes.append(f"{name}: risk in description (‘{risky.group(0)}’) — skipped")
            continue
        known = verdicts.get(str(Path(it.path).resolve()).casefold()) or {}
        if any(str(r).startswith((qc.NUDITY, "nagość")) for r in known.get("reasons") or []):
            notes.append(f"{name}: check: nudity — skipped")
            continue
        if known.get("safe") is not True:  # bez werdyktu albo niesprawdzony - sprawdzamy teraz
            v = qc.check_video(it.path, None, quality=False) if it.kind == "video" else \
                qc.check(it.path, None, quality=False)
            if not v.safe:
                notes.append(f"{name}: {'check: nudity' if v.nude else 'validation unavailable'}"
                             " — skipped")
                continue
        kept.append(it)
    return kept, notes


# ----------------------------- opisy -----------------------------

_CAPTION_SCHEMA = {
    "type": "object",
    "properties": {"instagram": {"type": "string"}, "tiktok": {"type": "string"},
                   "hashtags": {"type": "array", "minItems": 5, "maxItems": 10,
                                "items": {"type": "string"}}},
    "required": ["instagram", "tiktok", "hashtags"]}


def captions(persona: dict, theme: str, language: str = "pl",
             name: str = "") -> Dict[str, str]:
    """Opisy postow glosem persony (pierwsza osoba) z oznaczeniem AI. Blad modelu = opisy
    zastepcze (tylko oznaczenie i hashtagi), zeby paczka zawsze powstala. name - nazwa
    postaci: jej czesci nie trafiaja do hashtagow (model odmienial ja po polsku)."""
    from . import autopilot

    lang = "Polish" if language == "pl" else "English"
    summary = autopilot._summary(autopilot.clean(persona))
    banned = [_plain(w) for w in re.findall(r"\w{3,}", name)] + ["aist"]
    # imie postaci (takze odmienione, "Mią") nie trafia ani do tematu, ani do hashtagow
    theme = " ".join(w for w in str(theme or "").split()
                     if not any(_plain(w).startswith(b) for b in banned))
    system = (
        "You write SFW social media captions for a fictional virtual influencer (a woman aged "
        f"21 or more), in her voice (first person), in simple, correct {lang}. Her persona: "
        f"{summary} Return JSON: instagram - two or three short sentences with one or two emojis "
        "that end with a question to the followers; tiktok - one short hook sentence; "
        "hashtags - 6 to 10 relevant hashtags "
        "without the # sign. Do not mention AI, the app adds that label." + content.RULE)
    ok, text = llm.ask(system, f"Photo session: {theme or 'lifestyle photos'}",
                       schema=_CAPTION_SCHEMA, max_tokens=500, desc="Writing captions...",
                       temp=0.5)
    data = (planner._json(text) if ok else None) or {}
    if not isinstance(data, dict) or content.check(*[v for v in data.values() if isinstance(v, str)]):
        data = {}
    tags = [re.sub(r"[^\w]", "", str(t)) for t in data.get("hashtags") or []]
    tags = [t for t in dict.fromkeys(tags + AI_TAGS)
            if t and not any(b in _plain(t) for b in banned)]
    line = AI_LINE.get(language, AI_LINE["pl"])

    def compose(text_: str, n_tags: int) -> str:
        body = " ".join(str(text_ or "").split())
        return "\n\n".join(x for x in (body, line, " ".join(f"#{t}" for t in tags[:n_tags])) if x)

    return {"instagram": compose(data.get("instagram"), 12),
            "tiktok": compose(data.get("tiktok"), 5)}


# ----------------------------- pliki -----------------------------

def _jpg(src: str, dst: Path) -> None:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dst, quality=92)
    metadata.embed_metadata(dst, {"source": Path(src).name})


def srt(lines: List[str], duration: float) -> str:
    """Napisy: kwestie po kolei, rowno w czasie filmu (aplikacja zna tekst, nie zna chwil)."""
    lines = [" ".join(t.split()) for t in lines if t and t.strip()]
    step = duration / max(len(lines), 1)

    def ts(s: float) -> str:
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    return "\n".join(f"{i + 1}\n{ts(i * step)} --> {ts((i + 1) * step - 0.05)}\n{t}\n"
                     for i, t in enumerate(lines))


def _video(src: str, dst: Path, lines: List[str]) -> str:
    """Kopia filmu; z kwestiami - napisy wypalone na dole (nad paskiem przyciskow aplikacji)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        shutil.copy2(src, dst)
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "subtitles.srt").write_text(
            srt(lines, ffmpeg_utils.probe_duration(src) or 5.0), encoding="utf-8")
        style = ("FontName=Arial,FontSize=13,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                 "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=70")
        ok, err = ffmpeg_utils._run(["ffmpeg", "-y", "-i", src, "-vf",
                                     f"subtitles=subtitles.srt:force_style='{style}'", "-c:a", "copy",
                                     "-metadata", f"comment={ffmpeg_utils.AI_COMMENT}",
                                     str(dst)], cwd=tmp)
    if not ok:
        shutil.copy2(src, dst)
        return f"subtitles failed ({err[-120:]}) — video has no subtitles"
    return ""


def build(items: List[Item], ch: Optional[dict], persona: dict, folder: Path,
          day: str = "", notes: Optional[List[str]] = None) -> dict:
    """Paczka z elementow do nowego folderu (notes - uwagi o pominietych przez straz SFW).
    Zwraca plan (zapisany tez w plan.json). Przerwane skladanie nie zostawia polowy paczki."""
    folder.mkdir(parents=True)
    try:
        return _fill(items, ch, persona, folder, day, list(notes or []))
    except BaseException:  # takze Cancelled (app/cancel.py) - sprzatamy i dalej w gore
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _fill(items: List[Item], ch: Optional[dict], persona: dict, folder: Path,
          day: str, notes: List[str]) -> dict:
    language = persona.get("language", "pl")
    plan = {"model": ch["name"] if ch else "", "day": day, "posts": [], "notes": notes}
    groups: Dict[str, List[Item]] = {}
    for it in items:
        groups.setdefault(it.group, []).append(it)
    n_car = n_reel = 0
    for gi, (group, its) in enumerate(groups.items(), 1):
        theme = next((i.theme for i in its if i.theme), "")
        cap = captions(persona, theme, language, ch["name"] if ch else "")
        # Instagram: karuzele (zdjecia tej sesji, po proporcjach) i rolki; TikTok: filmy
        by_ratio: Dict[str, List[Item]] = {}
        for i in its:
            if i.kind == "image":
                by_ratio.setdefault(i.ratio, []).append(i)
        for ratio, imgs in by_ratio.items():
            for start in range(0, len(imgs), MAX_CAROUSEL):
                n_car += 1
                files = []
                for k, i in enumerate(imgs[start:start + MAX_CAROUSEL], 1):
                    rel = Path("instagram") / f"carousel_{n_car}" / f"{k:02d}.jpg"
                    _jpg(i.path, folder / rel)
                    files.append(rel.as_posix())
                plan["posts"].append({"platform": "instagram", "type": "carousel",
                                      "files": files, "ratio": ratio,
                                      "caption": cap["instagram"], "time": TIMES["carousel"]})
        for i in (i for i in its if i.kind == "video"):
            n_reel += 1
            for platform, name, key in (("instagram", f"reel_{n_reel}.mp4", "reel"),
                                        ("tiktok", f"video_{n_reel}.mp4", "tiktok")):
                rel = Path(platform) / name
                warn = _video(i.path, folder / rel, i.dialogue)
                if warn:
                    plan["notes"].append(f"{Path(i.path).name}: {warn}")
                plan["posts"].append({"platform": platform,
                                      "type": "reel" if platform == "instagram" else "video",
                                      "files": [rel.as_posix()],
                                      "caption": cap["instagram" if platform == "instagram"
                                                     else "tiktok"],
                                      "time": TIMES[key]})
    (folder / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _texts(folder, plan, language)
    return plan


def _texts(folder: Path, plan: dict, language: str) -> None:
    lines = []
    for post in plan["posts"]:
        lines += [f"=== {post['platform']} · {post['type']} · {post['time']} ===",
                  "Files: " + ", ".join(post["files"]), "", post["caption"], ""]
    if plan["notes"]:
        lines += ["=== Notes ===", *plan["notes"]]
    (folder / "captions.txt").write_text("\n".join(lines), encoding="utf-8")
    (folder / "README.txt").write_text(
        "How to use this pack\n\n"
        "- Instagram: upload each carousel folder as one post, or the reel MP4.\n"
        "- TikTok: upload the video MP4 and enable the AI-generated content label.\n"
        "- Captions and suggested times: captions.txt; planner data: plan.json.\n"
        "- Check each platform's current AI labeling requirements before posting.\n"
        "Media includes an AI-generated metadata label. Review every file before publishing.\n"
        "Automated visual checks can miss unsafe content and do not guarantee correctness.\n",
        encoding="utf-8")


def _target(slug: str, name: str) -> Path:
    if slug:
        characters._folder(slug)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid pack name.")
    target = PACKS_DIR / (slug or "general") / name
    if not target.resolve().is_relative_to(PACKS_DIR.resolve()):
        raise ValueError("Pack folder is outside the pack library.")
    return target


def _fresh(folder: Path) -> Path:
    """Pierwsza wolna nazwa (folder, folder_2, folder_3...) - nowa paczka nie zastepuje starej."""
    path, n = folder, 1
    while path.exists():
        n += 1
        path = folder.with_name(f"{folder.name}_{n}")
    return path


def build_day(slug: str, day: str) -> Tuple[bool, str]:
    """Paczka dnia modelki z tresci autopilota. Zwraca (ok, komunikat ze sciezka).
    Poprzednia paczka tego dnia (mogly w niej byc poprawiane opisy) idzie do Kosza Windows."""
    from . import autopilot

    ch = characters.usable(slug)
    items = items_for_day(slug, day)
    if not (ch and items):
        return False, f"No completed autopilot content for {day}."
    items, skipped = safety_check(items)
    if not items:
        return False, f"No content passed the SFW check for {day}: " + "; ".join(skipped)
    folder = _target(slug, day)
    if folder.exists() and not trash.move(folder):
        return False, (f"Could not move the previous pack ({folder}) to the Recycle Bin. Close "
                       "its open files and try again.")
    plan = build(items, ch, autopilot.persona(slug), folder, day, skipped)
    return True, _summary(plan, folder)


def build_selection(paths: List[str], slug: str = "") -> Tuple[bool, str]:
    """Paczka z zaznaczonych plikow (Galeria); slug - modelka (do opisow i kontroli).
    Podglady 480p (tryb Podglad, bez 1080p) nie trafiaja do paczek."""
    import time

    from . import autopilot

    entries = history.recent(5000)
    previews = {Path(f).name for e in entries if e.get("speed") == video_gen.PREVIEW
                for f in e.get("files") or []}
    skipped = [p for p in paths if Path(p).name in previews]
    items = items_for_files([p for p in paths if Path(p).name not in previews], entries)
    note = (f" Skipped 480p previews: {len(skipped)}. Render in Quality "
            "or Fast mode for publishing." if skipped else "")
    if not items:
        return False, "Select images or videos." + note
    ch = characters.usable(slug) if slug else None
    items, skipped = safety_check(items, entries)
    if not items:
        return False, "No content passed the SFW check: " + "; ".join(skipped) + note
    folder = _fresh(_target(slug, time.strftime("%Y-%m-%d_%H%M")))
    persona = autopilot.persona(slug) if slug else dict(autopilot.DEFAULTS)
    plan = build(items, ch, persona, folder, notes=skipped)
    return True, _summary(plan, folder) + note


def _summary(plan: dict, folder: Path) -> str:
    count = {p: sum(1 for x in plan["posts"] if x["platform"] == p)
             for p in ("instagram", "tiktok")}
    notes = f" Notes: {len(plan['notes'])} (in captions.txt)." if plan["notes"] else ""
    return (f"📦 Pack ready: {folder} — Instagram: {count['instagram']} post(s), "
            f"TikTok: {count['tiktok']}.{notes}")


def day_done(slug: str, day: str) -> bool:
    return (_target(slug, day) / "plan.json").exists()
