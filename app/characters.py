# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Postacie (modelki) - presety z wlasna LoRA (strona Trening, wybor w panelu bocznym).

"Wszystkie (bez postaci)" = generowanie bez LoRA. Postac powstaje ze zdjec jednej osoby: zadanie
"trening" w kolejce sprawdza zdjecia (model wizyjny: czy jest osoba, wiek, opis sceny), uczy
LoRA Z-Image (lora_train) i robi portret probny. Wybrana postac dziala w:
  - obrazach: Z-Image + LoRA postaci (slowo-klucz w prompcie),
  - obrazach ze zdjec pomocniczych: postac z LoRA jako obraz 1, zdjecia dokladaja rzeczy,
  - wideo: klatka startowa kazdego ujecia z Z-Image + LoRA, dalej Wan 2.2 (obraz -> wideo).

Folder postaci characters/<slug>/: character.json, photos/ (zdjecia treningowe), preview.png;
train/ - robocze, kasowane po treningu. LoRA: ComfyUI/models/loras/aistudio_char_<slug>.

Granica stala: postac z osoba wygladajaca na niepelnoletnia nie zostanie wytrenowana -
source photos are preserved and training is blocked. Zdjecia z nagoscia sa pomijane (straz SFW) - LoRA uczy sie tylko
na zdjeciach SFW. Zgoda (postac fikcyjna / AI albo dorosla osoba, ktora sie zgodzila)
jest wymagana przy tworzeniu.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import cancel, config, content, llm, lora_train, prefs, queue_manager, refs, trash, workflows
from .persistence import atomic_text

ProgressCb = Callable[[float, str], None]

DIR = config.ROOT / "characters"
LORA_DIR = config.COMFYUI_DIR / "models" / "loras"
KIND = "trening"
MIN_PHOTOS, MAX_PHOTOS = 8, 40
MIN_USABLE = 6
MAX_PHOTO_BYTES = 50 * 1024 * 1024
MAX_PHOTO_PIXELS = 40_000_000
DEFAULT_LABEL = "All characters (no LoRA)"
# statusy zapisane w character.json (w kolejce / trening liczone na biezaco z kolejki)
NEW, READY, FAILED, STOPPED = "nowa", "gotowa", "błąd", "przerwana"
_PREVIEW = ("close-up portrait photo, looking at the camera with a soft smile, soft natural "
            "daylight, plain light background")


# ----------------------------- zapis -----------------------------

def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.replace("ł", "l").replace("Ł", "L"))
    text = re.sub(r"[^a-z0-9]+", "_", text.encode("ascii", "ignore").decode().lower()).strip("_")
    return text[:24] or "postac"


def _folder(slug: str) -> Path:
    if not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", slug):
        raise ValueError("Invalid character identifier.")
    folder = DIR / slug
    if folder.resolve().parent != DIR.resolve():
        raise ValueError("Character folder is outside the character library.")
    return folder


def _file(slug: str) -> Path:
    return _folder(slug) / "character.json"


def _save(ch: dict) -> None:
    path = _file(ch["slug"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps(ch, ensure_ascii=False, indent=1))


def get(slug: str) -> Optional[dict]:
    if not slug:
        return None
    try:
        data = json.loads(_file(slug).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("slug") != slug or not isinstance(data.get("name"), str):
            return None
        return data
    except (OSError, ValueError):
        return None


def all_characters() -> List[dict]:
    if not DIR.exists():
        return []
    found = [get(p.name) for p in DIR.iterdir() if p.is_dir()]
    return sorted((c for c in found if c), key=lambda c: c.get("created", 0))


def lora_name(slug: str) -> str:
    _folder(slug)
    return f"aistudio_char_{slug}.safetensors"


def usable(slug: str) -> Optional[dict]:
    """Postac gotowa do generowania (wytrenowana, plik LoRA na miejscu) albo None."""
    ch = get(slug)
    if ch and ch.get("status") == READY and (LORA_DIR / lora_name(slug)).exists():
        return ch
    return None


def _job(slug: str) -> Optional[queue_manager.Job]:
    return next((j for j in queue_manager.snapshot() if j.kind == KIND
                 and j.params.get("slug") == slug
                 and j.status in (queue_manager.PENDING, queue_manager.RUNNING)), None)


def state(ch: dict) -> str:
    """Status do wyswietlenia (z kolejka: w kolejce / trening N%)."""
    job = _job(ch["slug"])
    if job and job.status == queue_manager.RUNNING:
        return f"Training {job.progress * 100:.0f}%"
    if job:
        return "Queued"
    return {NEW: "New", READY: "Ready", FAILED: "Error", STOPPED: "Stopped"}.get(ch.get("status", NEW), "Unknown")


def choices() -> List[Tuple[str, str]]:
    """Lista do wyboru postaci: domyslna + wytrenowane."""
    return [(DEFAULT_LABEL, "")] + [(c["name"], c["slug"]) for c in all_characters()
                                    if usable(c["slug"])]


def selected() -> Optional[dict]:
    """Postac wybrana w panelu bocznym (None = wszystkie / bez postaci)."""
    return usable(prefs.load().get("character", ""))


def tag(params: dict) -> str:
    """Przy dodawaniu zadania: wybrana postac trafia do parametrow (zadanie w kolejce zostaje
    przy tej postaci, nawet gdy wybor sie zmieni). Zwraca dopisek do nazwy zadania."""
    ch = selected()
    if not ch:
        return ""
    params["character"] = ch["slug"]
    return f" · {ch['name']}"


def for_job(p: dict) -> Tuple[Optional[dict], str]:
    """W zadaniu: (postac albo None, uwaga gdy postac zniknela od dodania zadania)."""
    slug = p.get("character") or ""
    ch = usable(slug)
    return ch, ("" if ch or not slug else
                "The selected character is unavailable; the default was used.")


# ----------------------------- tworzenie / usuwanie -----------------------------

def create(name: str, photos: List[str], consent: bool) -> Tuple[Optional[dict], str]:
    """Nowa postac ze zdjec + zadanie treningu w kolejce. Zwraca (postac, komunikat)."""
    from PIL import Image, ImageOps

    name = (name or "").strip()
    photos = [p for p in (photos or []) if p and Path(p).exists()]
    if not name:
        return None, "Enter a character name."
    if llm.blocked(name):
        return None, llm.BLOCK_MSG
    if not consent:
        return None, ("Confirm that the character is fictional (AI-generated) "
                      "or an adult who consented to training.")
    if not MIN_PHOTOS <= len(photos) <= MAX_PHOTOS:
        return None, (f"Add {MIN_PHOTOS} to {MAX_PHOTOS} photos of the same person "
                      f"(15–25 recommended; currently {len(photos)}).")
    if any(c["name"].strip().lower() == name.lower() for c in all_characters()):
        return None, (f"Character ‘{name}’ already exists. Choose another name "
                      "or manage the existing character in the library.")
    err = lora_train.ready()
    if err:
        return None, err
    base, slug, i = slugify(name), slugify(name), 2
    while (DIR / slug).exists():
        slug, i = f"{base}_{i}", i + 1
    folder = DIR / slug / "photos"
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src in photos:
        try:
            if Path(src).stat().st_size > MAX_PHOTO_BYTES:
                continue
            with Image.open(src) as im:
                if im.width * im.height > MAX_PHOTO_PIXELS:
                    continue
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1536, 1536), Image.LANCZOS)
                if min(im.size) < 384:
                    continue  # za male - nic nie wnosi do treningu
                saved += 1
                im.save(folder / f"{saved:02d}.png")
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    if saved < MIN_PHOTOS:
        shutil.rmtree(DIR / slug, ignore_errors=True)
        return None, (f"Not enough valid photos ({saved}); at least {MIN_PHOTOS} are required "
                      "(minimum 384 px).")
    steps = lora_train.steps_for(saved)
    ch = {"name": name, "slug": slug, "trigger": f"{slug}_aist", "gender": "woman",
          "status": NEW, "photos": saved, "steps": steps, "look": "", "created": time.time(),
          "trained": None, "error": ""}
    _save(ch)
    job = enqueue(ch)
    return ch, (f"Character ‘{name}’ added ({saved} photos). Training queued (job "
                f"#{job.id}, about {queue_manager.fmt_time(job.estimate())}).")


def enqueue(ch: dict) -> queue_manager.Job:
    return queue_manager.add(KIND, f"Character training: {ch['name']}",
                             {"slug": ch["slug"], "steps": ch["steps"]})


def retrain(slug: str) -> str:
    ch = get(slug)
    if not ch:
        return "Character not found."
    if _job(slug):
        return "Training for this character is already queued."
    if not any((DIR / slug / "photos").glob("*.png")):
        return "No source photos are available for this character."
    job = enqueue(ch)
    return f"Training ‘{ch['name']}’ queued (job #{job.id})."


def delete(slug: str) -> str:
    ch = get(slug)
    if not ch:
        return "Character not found."
    job = _job(slug)
    if job and job.status == queue_manager.RUNNING:
        return "This character is being trained. Stop training in the Queue first."
    if job:
        queue_manager.remove([job.id])
    lora = LORA_DIR / lora_name(slug)
    if lora.exists() and not trash.move(lora):
        return "Could not move the character LoRA to the Recycle Bin. Close files and try again."
    if not trash.move(_folder(slug)):
        return "Could not move the character folder to the Recycle Bin."
    if prefs.load().get("character") == slug:
        prefs.save(character="")
    return f"Character ‘{ch['name']}’ moved to the Recycle Bin."


# ----------------------------- generowanie z postacia -----------------------------

def loras(ch: dict, strength: float = 1.0) -> List[dict]:
    return [{"name": lora_name(ch["slug"]), "strength_model": strength,
             "strength_clip": strength}]


# Skape stroje w opisach zdjec treningowych (llm.check_photo)
_REVEALING = re.compile(r"\b(?:bikini\w*|swim\w*|lingerie|underwear|bra|bras|bralette|panties|"
                        r"thong)\b", re.I)
# Opis stroju w prompcie. Bez niego LoRA postaci (starsze zdjecia treningowe w bikini i akty)
# dawala w tescie nagosc przy "spacer po plazy" - stroj dopisujemy wtedy sami.
# Tylko ubrania: "master suite", "on top of the bed", "swimming", "shirtless" to nie stroj.
_CLOTHING = re.compile(
    r"\b(?:\w*wear\w*|dress(?:es|ed)?|cloth(?:es|ing|ed)?|outfits?|shirts?|t-shirts?|"
    r"(?<!on )tops?(?!\s+of\b)|tank|bikinis?|swim(?:suit|wear)s?|lingerie|bras?|bralette|"
    r"jeans|pants|trousers|skirts?|shorts|sweaters?|hoodies?|coats?|jackets?|\w*suits?|"
    r"uniforms?|robes?|towels?|leggings|blouses?|gowns?|costumes?|underwear|panties|"
    r"stockings|sweatshirts?|cardigans?|kimonos?|aprons?|armou?r|corsets?|pajamas|pyjamas)\b",
    re.I)


# Konkretny stroj, gdy prompt zadnego nie podaje - pierwszy pasujacy do sceny. Ogolne
# "fully clothed in an outfit that fits the scene" przegrywalo z LoRA uczona tez na aktach
# (test 30.10, Virtual Mia w salonie: 7 z 8 nago); nazwana sukienka - 12 z 12 ubrana.
_OUTFITS = [
    (re.compile(r"\b(?:beach\w*|pool\w*|ocean|sea|seaside|yacht\w*|boat\w*|swimming|lake|"
                r"tropical|sunbath\w*)\b", re.I), "a black one-piece swimsuit"),
    (re.compile(r"\b(?:gym|workout\w*|fitness|yoga|running|jogging|treadmill)\b", re.I),
     "a fitted long-sleeve sports top and leggings"),
    (re.compile(r"\b(?:snow\w*|ski|skiing|skier|winter|mountain\w*|cabin|chalet|fireplace)\b",
                re.I), "a cozy knit sweater and jeans"),
    (re.compile(r"\b(?:bed|beds|bedroom\w*|duvet|sheets|pillows?)\b", re.I),
     "long-sleeve satin pajamas"),
]
_DEFAULT_OUTFIT = "a fitted black knit dress with long sleeves"


def outfit_for(scene: str) -> str:
    return next((outfit for rx, outfit in _OUTFITS if rx.search(scene)), _DEFAULT_OUTFIT)


def positive(ch: dict, text: str) -> str:
    """Prompt Z-Image z postacia: slowo-klucz z treningu na poczatku, opis bez cech wygladu
    (wyglad daje LoRA), stroj i opis wygladu postaci na koncu. Konkretny stroj pasujacy do
    sceny, gdy opis go nie podaje."""
    body = content.strip_hiding(refs.neutral_look(refs.strip(text))).strip().rstrip(" .")
    look = (ch.get("look") or "").rstrip(".")
    extra = "" if _CLOTHING.search(body) else \
        f" The {ch['gender']} is wearing {outfit_for(body)}."
    return (f"Photo of {ch['trigger']} {ch['gender']}. {body}.{extra}"
            + (f" The {ch['gender']} has {look[:1].lower() + look[1:]}." if look else ""))


def render(client, ch: dict, text: str, width: int, height: int, out_path: Path,
           seed: Optional[int] = None,
           progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """Obraz postaci (Z-Image + LoRA) w rozmiarze width x height - bez wpisu w historii
    (klatka startowa wideo, baza do skladania ze zdjeciami). Zwraca (ok, sciezka/blad)."""
    from PIL import Image

    from . import image_gen

    refusal = content.check(text, ch.get("look", ""))
    if refusal:
        return False, refusal
    models = image_gen.available_models()
    if not models:
        return False, "The Z-Image model is unavailable."
    seed = random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)
    w16, h16 = -(-width // 16) * 16, -(-height // 16) * 16
    try:
        wf = workflows.txt2img(image_gen.resolve_spec(models[0]), positive(ch, text), "",
                               w16, h16, seed, prefs.IMAGE_STEPS, 1.0,
                               loras=loras(ch, content.LORA_STRENGTH))
        wf["9"]["inputs"]["filename_prefix"] = "aistudio_char"
        client.prepare_pipeline("image")
        result = client.run_and_wait(wf, progress_cb=progress_cb)
    except Exception as e:  # noqa: BLE001
        return False, f"Character generation failed ({e})."
    if not result.images:
        return False, "Character generation: the engine returned no image."
    out_path = Path(out_path)
    out_path.write_bytes(result.images[0])
    if (w16, h16) != (width, height):
        with Image.open(out_path) as im:
            left, top = (w16 - width) // 2, (h16 - height) // 2
            im.crop((left, top, left + width, top + height)).save(out_path)
    return True, str(out_path)


# ----------------------------- zadanie treningu (kolejka) -----------------------------

def _nude(photo: Path) -> bool:
    """Straz SFW dla zdjecia treningowego (app/qc.py) - LoRA uczona na aktach rozbierala
    postac nawet przy zwyklych scenach."""
    from . import qc  # qc importuje ten modul

    return not qc.check(str(photo), None, quality=False).safe


def _fail(ch: dict, status: str, error: str) -> Tuple[bool, list, list, str]:
    ch.update(status=status, error=error)
    _save(ch)
    return False, [], [], error


def run_job(p: dict, cb) -> tuple:
    """Trening postaci: sprawdzenie zdjec i opisy (model wizyjny) -> LoRA -> portret probny."""
    from .comfy_client import ComfyClient

    ch = get(p.get("slug", ""))
    if not ch:
        return False, [], [], "The character is unavailable."
    ch.update(status=NEW, error="")  # ponowny trening - bez starego bledu na liscie
    _save(ch)
    folder = DIR / ch["slug"]
    photos = sorted((folder / "photos").glob("*.png"))
    work = folder / f"train_{time.time_ns()}"
    images = work / "img"
    images.mkdir(parents=True)
    notes: List[str] = []
    try:
        # --- 1. Zdjecia: osoba, wiek (twarda blokada), opis sceny bez cech wygladu ---
        genders: Dict[str, int] = {}
        kept: List[Tuple[Path, str]] = []
        for i, photo in enumerate(photos):
            cb(0.04 * i / len(photos), f"Checking photos {i + 1}/{len(photos)}...")
            info = llm.check_photo(str(photo))
            if info is None:
                notes.append(f"Photo {photo.stem}: validation unavailable; skipped.")
                continue
            if info["person"] and info["age"] < 18:
                return _fail(ch, FAILED, (
                    f"⛔ Photo {photo.stem} appears to show a minor (estimated age {info['age']}). "
                    "Training was blocked. All source photos were preserved."))
            if not info["person"]:
                notes.append(f"Photo {photo.stem}: no clear person; skipped.")
                continue
            if content.asks_nudity(info["caption"]) or _nude(photo):
                notes.append(f"Photo {photo.stem}: unsafe or unverified; skipped (training uses only "
                             "verified SFW photos).")
                continue
            genders[info["gender"]] = genders.get(info["gender"], 0) + 1
            kept.append((photo, info["caption"]))
        if len(kept) < MIN_USABLE:
            return _fail(ch, FAILED, f"Not enough verified photos with a clear adult person ({len(kept)}); "
                                     f"at least {MIN_USABLE} are required.")
        gender = max(genders, key=genders.get)
        # postac przejmuje styl strojow ze zdjec - przy przewadze bikini/bielizny ostrzegamy
        revealing = sum(bool(_REVEALING.search(caption)) for _, caption in kept)
        warning = "" if revealing < 0.4 * len(kept) else (
            f"{revealing} of {len(kept)} photos show swimwear or underwear. The character may "
            "repeat those outfits. For everyday scenes, add photos in everyday clothes "
            "and retrain the character.")
        if warning:
            notes.append(warning)
        for photo, caption in kept:
            shutil.copy(photo, images / photo.name)
            text = f"photo of {ch['trigger']} {gender}" + (f", {caption}" if caption else "")
            (images / f"{photo.stem}.txt").write_text(text, encoding="utf-8")
        cb(0.045, "Describing character appearance...")
        look = llm.describe_person(str(kept[0][0]))
        ch.update(gender=gender, look=look, error="", warning=warning)
        _save(ch)

        # --- 2. Trening (silnik ComfyUI oddaje cala karte) ---
        cfg = config.SETTINGS["comfyui"]
        client = ComfyClient(cfg["host"], cfg["port"])
        if client.system_stats():
            client.free_memory()
        steps = int(p.get("steps") or ch["steps"])

        def tcb(frac: float, desc: str) -> None:
            cb(0.05 + 0.92 * frac, desc)

        ok, err = lora_train.train(work, images, steps, LORA_DIR / lora_name(ch["slug"]), tcb)
        if not ok:
            return _fail(ch, FAILED, err)

        # --- 3. Portret probny z gotowej LoRA ---
        ch.update(status=READY, trained=time.time(), steps=steps)
        _save(ch)
        cb(0.98, "Generating character preview...")
        preview = folder / "preview.png"
        ch["preview_safe"] = False
        _save(ch)
        if client.system_stats():
            from . import qc
            if preview.exists():
                archive = folder / f"preview_{time.time_ns()}.png"
                shutil.copy2(preview, archive)
            ok, msg = qc.render_checked(client, ch, prefs.styled(_PREVIEW, "image"),
                                         832, 1216, preview, 42, notes)
            if not ok:
                return _fail(ch, FAILED, msg)
            ch["preview_safe"] = True
            _save(ch)
        else:
            notes.append("Character preview skipped: ComfyUI is unavailable.")
        notes.insert(0, f"Character ‘{ch['name']}’ is ready. Select it in the left sidebar "
                        "(Character).")
        return True, [str(preview)] if preview.exists() and ch.get("preview_safe") is True else [], notes, ""
    except cancel.Cancelled:
        _fail(ch, STOPPED, "Training stopped.")
        raise


queue_manager.register(KIND, run_job)
