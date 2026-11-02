# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Obrazy pomocnicze (referencje) i ujecia w prompcie wideo.

Uzytkownik dolacza zdjecia (np. bluza, osoba, budynek) i wskazuje je w opisie numerem:
    A young man wearing the hoodie [1] walks on a rooftop.
    Ujęcie 2: he jumps onto the glass building [2].
Kazde ujecie (tekst po "Ujęcie N:" / "Shot N:") dostaje klatke startowa zlozona przez
Qwen-Image-Edit-2511 z obrazow, ktore wskazuje (do 10; model bierze 3 naraz, wiec wiecej
sklada w przebiegach - compose_refs) - z zachowaniem wzoru, kolorow i logo - a Wan 2.2
ozywia ja jako wideo. Kolejne ujecie sklada sie na bazie ostatniej
klatki poprzedniego, wiec osoba i stroj zostaja te same. Bez numerow wszystkie obrazy trafiaja
do pierwszego ujecia.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import model_manager, templates
from .comfy_client import ComfyClient

MODEL_ID = "qwen-image-edit-2511"
MAX_IMAGES = 3  # TextEncodeQwenImageEditPlus przyjmuje do 3 obrazow na przebieg
MAX_REFS = 10   # obrazow pomocniczych na zadanie (wiecej niz 3 - skladanie w przebiegach)

# [1] albo @1 (numer obrazu pomocniczego)
_MARK = re.compile(r"\s*(?:\[(\d{1,2})\]|@(\d{1,2})\b)")
# "Ujęcie 2:", "Ujecie 2 -", "Shot 2:", "Scena 2:", "Scene 2." na poczatku linii
# Na poczatku linii (takze "Ujęcie 2." z kropka) albo w srodku tekstu z dwukropkiem/myslnikiem -
# wpisane w jednej linii "... okno. Ujęcie 2: wychodzi ..." sklejalo caly opis w jedno ujecie
_SHOT = re.compile(r"(?:^\s*(?:uj[ęe]cie|shot|scen[ae])\s*(\d+)\s*[:.\-–—]|"
                   r"(?<!\w)(?:uj[ęe]cie|shot|scen[ae])\s*(\d+)\s*[:\-–—])\s*", re.I | re.M)

# Wskazowki dla modelu edycji: wierne odwzorowanie referencji, zdjecie, nie kolaz
_KEEP = ("Photorealistic photo, one coherent real scene. Every person from the reference images "
         "stays exactly the same person: same face, facial features, hairstyle, hair color, "
         "skin tone and body - only pose, expression, clothing and setting change as described. "
         "Keep the exact design, colors, patterns, logos, text and proportions of the items from "
         "the reference images.")
_CONTINUE = ("Image 1 is the previous moment of the same video: keep the same person, face, "
             "outfit, lighting and style.")
_STEP = ("Image 1 is the picture made so far: keep its scene, people, faces, poses, framing "
         "and everything already in it unchanged; only add or apply what comes from the other "
         "images.")
# Baza to postac z wlasnej LoRA (characters.render): to ona jest glowna osoba, obrazy
# pomocnicze dokladaja rzeczy, miejsca i inne osoby
CHARACTER = ("Image 1 shows the main person: keep exactly her or his face, facial features, "
             "hairstyle, hair color, skin tone and body from image 1. Do not tan, darken, "
             "lighten or restyle her or him.")
# Na poczatku promptu - model edycji najmocniej slucha pierwszych zdan
_IDENTITY = ("The person must be exactly the same person as in the reference image: identical "
             "face, skin tone, hair color, hairstyle, eye color and body. Do not tan, darken, "
             "lighten or restyle her or him.")

# Opisy wygladu, ktore przy osobie ze zdjecia wygrywaly ze zdjeciem (test: blada modelka
# z platynowymi wlosami + "deep tan, sun-kissed skin" w prompcie = opalona blondynka)
_LOOK = [
    (r"\b(?:a |an )?\d{2}[- ]year[- ]old\b", ""),
    (r"\b(?:aged|age) \d{2}\b", ""),
    (r"\bin (?:her|his) (?:early |mid |late )?(?:twenties|thirties|forties|fifties)\b", ""),
    (r"\b(?:the )?(?:deep |rich |warm |golden |dark )?colou?r of (?:her|his) skin\b", "skin"),
    (r"\b(?:deep(?:ly)?|golden|bronzed?|sun[- ]?kissed|sun[- ]?bleached|sun[- ]?tanned|tanned|"
     r"tan|olive|pale|fair|porcelain|ivory|dark|ebony|caramel|honey|alabaster)[- ]"
     r"(?:skin(?:ned)?|complexion|tan|glow)\b", "skin"),
    (r"\b(?:deep|golden|bronze|dark|light|healthy|even) tan\b", ""),
    (r"\b(?:tanned|sun-kissed|sunkissed|bronzed|sun-bleached|suntanned)\b", ""),
    (r"\b(?:(?:long|short|shoulder-length|wavy|straight|curly|flowing|messy|silky|sleek|"
     r"windswept|wet|thick)[ ,]+)*(?:blonde?|brunette|platinum|silver|white|black|brown|red|"
     r"auburn|chestnut|golden|dark|light|ginger|copper|honey|ash|strawberry|jet-black|raven)"
     r"(?:[- ]blonde?)?[- ]hair(?:ed)?\b", "hair"),
    (r"\b(?:blue|green|brown|hazel|grey|gray|amber|dark|light|ice-blue|piercing blue) eyes\b",
     "eyes"),
]
_LOOK_RE = [(re.compile(p, re.I), r) for p, r in _LOOK]


def neutral_look(text: str) -> str:
    """Opis bez wieku, koloru skory, opalenizny, koloru wlosow i oczu - przy osobie ze zdjecia
    wyglad ma pochodzic ze zdjecia, nie z tekstu."""
    for rx, repl in _LOOK_RE:
        text = rx.sub(repl, text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def numbers(text: str) -> List[int]:
    """Numery obrazow wskazane w tekscie (1 = pierwszy), w kolejnosci, bez powtorzen."""
    out: List[int] = []
    for m in _MARK.finditer(text):
        n = int(m.group(1) or m.group(2))
        if n not in out:
            out.append(n)
    return out


def strip(text: str) -> str:
    """Tekst bez numerow obrazow (dla Wan i dla opisu dzwieku)."""
    return re.sub(r"\s{2,}", " ", _MARK.sub("", text)).strip()


def plain(prompt: str) -> str:
    """Caly prompt bez oznaczen ujec i numerow obrazow (do dialogow i dzwieku tla)."""
    return strip(_SHOT.sub(" ", prompt))


def split_shots(prompt: str) -> List[str]:
    """Ujecia oznaczone "Ujęcie N:" / "Shot N:"; bez oznaczen - jedno ujecie.
    Gdy pierwsze oznaczenie to "Ujęcie 1:", tekst przed nim (np. tytul, opis calosci)
    dolacza do kazdego ujecia jako wspolny kontekst; gdy prompt zaczyna sie od razu opisem,
    a dalej jest "Ujęcie 2:", ten opis jest ujeciem 1."""
    marks = list(_SHOT.finditer(prompt))
    if not marks:
        return [prompt.strip()]
    head = prompt[:marks[0].start()].strip()
    intro = head if int(marks[0].group(1) or marks[0].group(2)) <= 1 else ""
    shots = [head] if head and not intro else []
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = prompt[m.end():nxt.start() if nxt else len(prompt)].strip()
        if body:
            shots.append(f"{intro}\n{body}".strip() if intro else body)
    return shots or [prompt.strip()]


def split_seconds(total: int, shots: int) -> List[int]:
    """Rowny podzial dlugosci filmu na ujecia (kazde min. 2 s)."""
    shots = max(1, min(shots, max(1, total // 2)))
    base, rest = divmod(total, shots)
    return [base + (1 if i < rest else 0) for i in range(shots)]


def edit_prompt(text: str, used: List[int], has_base: bool, continuing: bool,
                step: bool = False, looks: Optional[Dict[int, str]] = None,
                base_lead: str = "") -> str:
    """Prompt dla Qwen-Image-Edit: [n] -> "from image k" (k = miejsce na liscie obrazow);
    numery spoza `used` (juz dodane albo dodawane w kolejnym przebiegu) znikaja.
    step - kolejny przebieg: obraz 1 to dotychczasowy wynik, ktory ma zostac bez zmian.
    looks - {numer: wyglad osoby} (llm.describe_person): tekst wspiera zdjecie zamiast z nim
    walczyc - bez tego slonce na plazy opalalo blada modelke ze zdjecia.
    base_lead - zdanie o bazie (CHARACTER: baza to postac, ktora ma zostac glowna osoba)."""
    offset = 1 if has_base else 0

    def repl(m: re.Match) -> str:
        n = int(m.group(1) or m.group(2))
        return f" from image {used.index(n) + 1 + offset}" if n in used else ""

    body = re.sub(r"\s{2,}", " ", _MARK.sub(repl, text)).strip()
    if used:  # wyglad osoby ma pochodzic ze zdjecia, nie z tekstu
        body = neutral_look(body)
    lead = _STEP if step else (_CONTINUE if continuing else "")
    if not numbers(text) and used:
        # obrazy bez numerow w tekscie: na poczatku (najwazniejsze dla modelu), ze osoby
        # i rzeczy z opisu to te ze zdjec - inaczej opis wyglądu wygrywal ze zdjeciem
        listed = " and ".join(f"image {i + 1 + offset}" for i in range(len(used)))
        verb = "Add" if step else "Use"
        if base_lead and has_base:
            lead = (f"{lead} {verb} the exact items, places and any other people shown in "
                    f"{listed}; the main person is the one from image 1.").strip()
        else:
            lead = (f"{lead} {verb} the exact person and items shown in {listed}: the people "
                    "and things in the description below are them.").strip()
    looks = {n: look for n, look in (looks or {}).items() if look and n in used}
    who = " ".join(f"The person from image {used.index(n) + 1 + offset} has {look} - keep "
                   "exactly this look in any light." for n, look in looks.items())
    # powtorzone na koncu (model slucha poczatku i konca); jasna karnacja - wprost "nie opalona",
    # bo samo slonce/plaza w opisie opalalo osobe w czesci prob
    tail = " ".join(f"Her or his look stays exactly: {look}"
                    + (", not tanned" if re.search(r"pale|porcelain|fair|light|ivory|white",
                                                    look, re.I) else "") + "."
                    for look in looks.values())
    ident = base_lead if (base_lead and has_base) else (_IDENTITY if used else "")
    return " ".join(x for x in (ident, who, lead, _KEEP, body, tail) if x)


def compose_refs(client: ComfyClient, base: Optional[str], ref_list: List[Tuple[int, str]],
                 text: str, width: int, height: int, out_path: str | Path,
                 seed: Optional[int] = None, continuing: bool = False,
                 progress_cb: Optional[Callable[[float, str], None]] = None,
                 looks: Optional[Dict[int, str]] = None,
                 base_lead: str = "") -> Tuple[bool, str]:
    """Klatka z dowolnej liczby obrazow pomocniczych (do MAX_REFS). Qwen-Image-Edit bierze 3
    obrazy naraz, wiec przy wiekszej liczbie sklada w przebiegach: pierwszy z baza (zdjecie
    startowe / ostatnia klatka) i pierwszymi obrazami, kazdy kolejny dokłada 2 nastepne do
    dotychczasowego wyniku, ktory ma zostac bez zmian. text - opis z numerami [n];
    ref_list - [(numer, sciezka)]. Zwraca (ok, sciezka/blad)."""
    if not ref_list:
        return False, "No reference images were provided."
    passes, rest = [], list(ref_list)
    first = MAX_IMAGES - (1 if base else 0)
    passes.append(rest[:first])
    rest = rest[first:]
    while rest:
        passes.append(rest[:MAX_IMAGES - 1])
        rest = rest[MAX_IMAGES - 1:]
    current = base
    for i, chunk in enumerate(passes):
        used = [n for n, _ in chunk]
        prompt = edit_prompt(text, used, bool(current), continuing and i == 0, step=i > 0,
                             looks=looks, base_lead=base_lead if i == 0 else "")
        target = Path(out_path) if i == len(passes) - 1 else \
            Path(out_path).with_name(f"{Path(out_path).stem}_p{i + 1}.png")

        def cb(frac: float, desc: str, i=i) -> None:
            if progress_cb:
                label = f"pass {i + 1}/{len(passes)}, " if len(passes) > 1 else ""
                progress_cb((i + frac) / len(passes), label + desc)

        images = ([current] if current else []) + [p for _, p in chunk]
        ok, msg = compose(client, images, prompt, width, height, target,
                          None if seed is None else seed + i, cb)
        if i > 0 and current and current != base:
            Path(current).unlink(missing_ok=True)  # posredni wynik poprzedniego przebiegu
        if not ok:
            return False, msg
        current = str(target)
    return True, current


def _workflow(images: List[str], prompt: str, width: int, height: int, seed: int) -> dict:
    """Graf jak w oficjalnym szablonie image_qwen_image_edit_2511_int8 (LoRA Lightning:
    4 kroki, cfg 1, shift 3.1, CFGNorm, referencje metoda index_timestep_zero); rozmiar
    wyniku = rozmiar klatki wideo (pusty latent zamiast latentu pierwszego obrazu)."""
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": model_manager.model_file(MODEL_ID, "diffusion_models"),
            "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "strength_model": 1.0,
            "lora_name": model_manager.model_file(MODEL_ID, "loras", "lightning")}},
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.1}},
        "4": {"class_type": "CFGNorm", "inputs": {"model": ["3", 0], "strength": 1.0}},
        "5": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": model_manager.model_file(MODEL_ID, "text_encoders"),
            "type": "qwen_image", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {
            "vae_name": model_manager.model_file(MODEL_ID, "vae")}},
        "24": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}},
        "25": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0, "model": ["4", 0], "positive": ["22", 0], "negative": ["23", 0],
            "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["6", 0]}},
        "27": {"class_type": "SaveImage", "inputs": {"images": ["26", 0],
                                                     "filename_prefix": "aistudio_ref"}},
    }
    pos = {"clip": ["5", 0], "vae": ["6", 0], "prompt": prompt}
    neg = {"clip": ["5", 0], "vae": ["6", 0], "prompt": ""}
    for i, name in enumerate(images[:MAX_IMAGES]):
        wf[str(11 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        pos[f"image{i + 1}"] = neg[f"image{i + 1}"] = [str(11 + i), 0]
    wf["20"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": pos}
    wf["21"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": neg}
    for node, src in (("22", "20"), ("23", "21")):
        wf[node] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": [src, 0], "reference_latents_method": "index_timestep_zero"}}
    return templates.gguf_loaders(wf)


def compose(client: ComfyClient, images: List[str], prompt: str, width: int, height: int,
            out_path: str | Path, seed: Optional[int] = None,
            progress_cb: Optional[Callable[[float, str], None]] = None) -> Tuple[bool, str]:
    """Sklada obraz (width x height) z 1-3 obrazow wedlug promptu. Zwraca (ok, sciezka/blad)."""
    from . import content, llm, qc

    refusal = content.check(prompt)
    if refusal or llm.blocked(prompt):
        return False, refusal or llm.BLOCK_MSG
    if not images:
        return False, "No reference images were provided."
    if len(images) > MAX_IMAGES:
        return False, f"A composition pass supports at most {MAX_IMAGES} images."
    if not model_manager.is_installed(MODEL_ID):
        return False, "The Qwen-Image-Edit-2511 model is missing."
    seed = random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)
    try:
        names = [client.upload_image(p) for p in images[:MAX_IMAGES]]
        client.prepare_pipeline("edit")

        def ncb(node: str, value: int, maximum: int) -> None:
            if progress_cb and node == "25":
                progress_cb(value / max(maximum, 1), f"step {value}/{maximum}")

        result = client.run_and_wait(_workflow(names, prompt, width, height, seed),
                                     node_progress_cb=ncb)
    except Exception as e:  # noqa: BLE001
        return False, f"Reference composition failed: {e}"
    if not result.images:
        return False, "The engine returned no image."
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(result.images[0])
    verdict = qc.check(str(out_path), None, quality=False)
    if not verdict.safe:
        qc.reject(str(out_path), verdict, f"reference composition: {prompt[:80]}")
        return False, "Reference image failed the safety check: " + verdict.summary()
    return True, str(out_path)
