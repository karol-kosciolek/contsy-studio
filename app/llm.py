# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Local prompt writing and visual safety checks using the cataloged Qwen model.

Generation prompts use English; quoted speech retains the user's original language.
Model files and the pinned llama.cpp runtime are installed on first use.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import requests

from . import config, content, model_manager, refs

MODEL_ID = "prompt-llm"
# llama.cpp fits as many layers as free VRAM allows (all of them on 16 GB, fewer on small GPUs)
GPU_LAYERS = "auto"
LLAMA_BUILD = "b11900"  # przypieta wersja llama.cpp
_ZIP_URL = (f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/"
            f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip")
_RUNTIME_SIZE = 37_051_563
_RUNTIME_SHA256 = "baf0183b1906d7cc0cfe9a0bd3fae34ceb4048acecc371ff3b7a91456783d051"
_RUNTIME_MAX_DOWNLOAD = 64 * 1024 * 1024
_RUNTIME_MAX_UNPACKED = 256 * 1024 * 1024
_INSTALL_LOCK = threading.Lock()
BIN_DIR = config.ROOT / "vendor" / "llama.cpp"
_EXE = BIN_DIR / "llama-completion.exe"
_VISION_EXE = BIN_DIR / "llama-mtmd-cli.exe"  # model + mmproj: opis osoby ze zdjecia

ProgressCb = Callable[[float, str], None]

BLOCK_MSG = ("Blocked: sexualized depictions of minors are not allowed. "
             "Characters in intimate clothing must be clearly adult.")

# --- granica stala: nieletni + tresci seksualne (PL i EN) ---
_MINOR = re.compile(
    r"\b(child|children|kids?|minors?|underage|under-age|teens?|teenagers?|teenage|preteens?|"
    r"pre-teens?|loli|lolita|shota|schoolgirls?|schoolboys?|little (girl|boy)s?|young (girl|boy)s?|"
    r"toddlers?|infants?|newborns?|juveniles?|pubescent|prepubescent|"
    r"([1-9]|1[0-7])[ -]?(yo|y/o|years? old|year-old)|"
    r"dzieck\w*|dzieci\w*|nastolat\w*|nieletni\w*|niepełnolet\w*|małolat\w*|uczennic\w*|"
    r"dziewczyn(ka|ki|ce|kę|eczk\w*)|chłop(czyk\w*|iec|ca|cem)|niemowl\w*|gimnazjalist\w*|"
    r"podstawówk\w*|([1-9]|1[0-7])[ -]?(lat|latk\w*|letni\w*))\b", re.I)
_SEXUAL = re.compile(
    r"\b(nsfw|nude|naked|nudity|sex\w*|erotic\w*|porn\w*|explicit|topless|bottomless|lingerie|"
    r"underwear|bikini|breasts?|boobs?|nipples?|genital\w*|penis|vagina|pussy|aroused|orgasm\w*|"
    r"bdsm|fetish\w*|seductive|sensual|undress\w*|strip(ping|tease|per|pers)|"
    r"nago|nag(a|i|ie|ich|y)|nagość\w*|seks\w*|eroty\w*|porno\w*|bielizn\w*|bieliź\w*|piersi|"
    r"biust\w*|sutk\w*|podnieca\w*|rozebran\w*|rozbier\w*|zmysłow\w*|uwodzic\w*)\b", re.I)


def blocked(*texts: str) -> bool:
    """True, gdy tekst laczy nieletnich z tresciami seksualnymi."""
    text = content.normalize_for_check(" ".join(t for t in texts if t))
    return bool(_MINOR.search(text) and _SEXUAL.search(text))


_SYSTEM = {
    "image": (
        "You write prompts for a photorealistic image generator. The user's idea may be in any language, "
        "often colloquial: understand every word literally and keep its exact meaning - never "
        "replace or reinterpret it. The prompt MUST be written in English. "
        "Reply with ONE prompt: a single paragraph of 60-120 words, no preamble, no title, no "
        "quotes around it, no markdown, no notes. Describe the subject and their appearance, clothing, pose and "
        "expression, the setting, lighting, camera (lens, angle, framing), mood and fine details. "
        "Follow the user's idea exactly and describe every person as an adult with a concrete "
        "age of 21 or more (e.g. 'a 27-year-old woman'). Markers like [1], [2] point to the "
        "user's reference photos: copy "
        "each one right after the thing it names and never replace that thing with something "
        "else; do not invent its color, print or details - the photo defines them. "
        "Example: idea 'man [1] in hoodie [2]' -> 'a man [1] wearing the hoodie [2], "
        "...'."),
    "video": (
        "You write prompts for a realistic text-to-video model that makes 5-second shots. The "
        "user's idea may be in any language, often colloquial: understand its meaning and keep "
        "its exact meaning - never replace or reinterpret it. The prompt MUST "
        "be written in English. Preserve spoken lines exactly in the language supplied by the user. Reply with the prompt only, no preamble, no markdown, no notes. Per shot: "
        "one main "
        "action, the subject large in frame, their appearance and clothing, setting, lighting, "
        "camera movement and framing, and end with what is heard (e.g. 'sound of rain, distant "
        "traffic'). If a person speaks, preserve the spoken line in its original language inside „…” right after "
        "'says' - use „…” ONLY for human speech, never for animal or other sounds (write 'a dog "
        "barking', not „Woof!”). Use a single shot (one paragraph, 50-110 words) unless the idea "
        "clearly describes a sequence (e.g. 'then', 'next'); only then write each "
        "scene on its own line starting with 'Shot 1:', 'Shot 2:' and so on. Follow the "
        "user's idea exactly and describe every person as an adult with a concrete age of 21 "
        "or more. Markers like [1], "
        "[2] point to the user's reference photos: copy each one right after the thing it names "
        "and never replace that thing with something else; do not invent its color, print or "
        "details - the photo defines them. Example: idea 'man in hoodie [1] "
        "jumps onto building [2]' -> 'Shot 1: an adult man wearing the hoodie [1] ...\\nShot 2: "
        "he jumps onto the building [2] ...'."),
}

_REMIND = ("\nIMPORTANT: answer in English only (quoted spoken lines in their original language are the only "
           "exception) and keep every [n] image marker.")


_ORDINALS = {"pierwsz": 1, "drug": 2, "trzec": 3, "czwart": 4, "piąt": 5, "szóst": 6,
             "siódm": 7, "ósm": 8, "dziewiąt": 9, "dziesiąt": 10}
_PHOTO = r"(?:zdjęci\w*|załącznik\w*|obraz\w*|obrazk\w*|fotk\w*|fotografi\w*|foto)"


def mark_refs(text: str, n: int) -> str:
    """Polskie odwolania do zdjec -> numery, zanim tekst trafi do modelu (9B mylil "z
    pierwszego zdjecia" / "z drugiego zdjecia"): "z drugiego zdjęcia" -> [2], "ze zdjęcia 3"
    -> [3], przy jednym zdjeciu "ze zdjęcia", "na zdjęciu", "załączona" -> [1]."""
    if not n:
        return text

    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
    text = re.sub(r"\b(?:from |in |on )?(?:the )?(first|second|third|fourth|fifth|sixth|"
                  r"seventh|eighth|ninth|tenth) (?:photo|image|picture|attachment)\b",
                  lambda m: f"[{ordinals[m.group(1).lower()]}]"
                  if ordinals[m.group(1).lower()] <= n else m.group(0), text, flags=re.I)
    text = re.sub(r"\b(?:photo|image|picture|attachment)\s*(?:#|no\.?\s*)?(\d{1,2})\b",
                  lambda m: f"[{m.group(1)}]" if 1 <= int(m.group(1)) <= n else m.group(0),
                  text, flags=re.I)

    def ordinal(m: re.Match) -> str:
        k = next((v for s, v in _ORDINALS.items() if m.group(1).lower().startswith(s)), 0)
        return f" [{k}]" if 1 <= k <= n else m.group(0)

    text = re.sub(rf"\s*\b(?:z|ze|na|w)?\s*(\w+)\s+{_PHOTO}\b", lambda m: ordinal(m)
                  if any(m.group(1).lower().startswith(s) for s in _ORDINALS) else m.group(0),
                  text, flags=re.I)
    text = re.sub(rf"\s*\b(?:z|ze|na|w)?\s*{_PHOTO}\s*(?:nr\.?|numer)?\s*(\d{{1,2}})\b",
                  lambda m: f" [{m.group(1)}]" if 1 <= int(m.group(1)) <= n else m.group(0),
                  text, flags=re.I)
    if n == 1:
        text = re.sub(rf"\s*\b(?:z|ze|na)\s+{_PHOTO}\b", " [1]", text, flags=re.I)
        text = re.sub(r"\b(?:załączon|dołączon)\w*\s*", "", text, flags=re.I)
        if "[1]" not in text:
            text = re.sub(r"\s*\b(?:z|ze)\s+załącznik\w*\b", " [1]", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def refs_rule(n: int) -> str:
    """Wskazowka dla modelu, gdy uzytkownik dolaczyl n zdjec-referencji."""
    one = (" With one photo, the person (or item) the idea talks about is the one in [1], "
           "unless the idea clearly describes someone else." if n == 1 else "")
    return (f" The user attached {n} reference photo{'s' if n > 1 else ''} ([1]"
            f"{f' to [{n}]' if n > 1 else ''}). When the idea mentions the photo or what is in "
            "it ('from the photo', 'in the attachment', 'this woman', 'that outfit'), refer to it with its marker right after it, e.g. 'the woman [1]', 'the "
            f"dress [2]'.{one} Never describe the face, hair, skin, body shape or look of a "
            "marked person, or the color, print or details of a marked item - the photo defines "
            "them; describe only pose, action, expression, clothing changes the user asks for, "
            "setting, lighting and camera.")
_GPU_LOCK = threading.Lock()  # zajety, gdy model liczy na karcie - kolejka czeka (release_gpu)


def ensure(progress_cb: Optional[ProgressCb] = None) -> Optional[str]:
    """Install the catalog model and checksum-pinned official llama.cpp runtime."""
    err = model_manager.ensure([MODEL_ID], progress_cb)
    if err:
        return err
    with _INSTALL_LOCK:
        if _EXE.exists() and _VISION_EXE.exists():
            return None
        staged = []
        try:
            if progress_cb:
                progress_cb(0.0, "Downloading the llama.cpp engine...")
            chunks, size = [], 0
            with requests.get(_ZIP_URL, timeout=(15, 120), stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > _RUNTIME_MAX_DOWNLOAD or size > _RUNTIME_SIZE:
                        raise ValueError("The llama.cpp archive exceeds its pinned size.")
                    chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != _RUNTIME_SIZE or hashlib.sha256(data).hexdigest() != _RUNTIME_SHA256:
                raise ValueError("The llama.cpp archive failed its pinned size or SHA-256 check.")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 1000 or sum(item.file_size for item in entries) > _RUNTIME_MAX_UNPACKED:
                    raise ValueError("The llama.cpp archive exceeds its extraction limits.")
                selected = [(item, Path(item.filename).name) for item in entries
                            if not item.is_dir() and (Path(item.filename).name in (_EXE.name, _VISION_EXE.name)
                            or Path(item.filename).suffix.lower() in (".dll", ".txt")
                            or Path(item.filename).name.startswith("LICENSE"))]
                names = [name.casefold() for _, name in selected]
                if len(set(names)) != len(names):
                    raise ValueError("The llama.cpp archive contains duplicate runtime filenames.")
                if not {_EXE.name.casefold(), _VISION_EXE.name.casefold()} <= set(names):
                    raise ValueError("The llama.cpp archive is missing required executables.")
                # Nothing executable is written until the entire archive has been verified.
                BIN_DIR.mkdir(parents=True, exist_ok=True)
                for item, name in selected:
                    fd, temporary = tempfile.mkstemp(prefix="runtime_", suffix=".tmp", dir=BIN_DIR)
                    staged.append((Path(temporary), BIN_DIR / name))
                    with os.fdopen(fd, "wb") as output:
                        output.write(archive.read(item))
            for temporary, destination in staged:
                temporary.replace(destination)
        except (requests.RequestException, zipfile.BadZipFile, OSError, ValueError) as exc:
            return f"Could not install the verified llama.cpp engine: {exc}"
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
        return None if _EXE.exists() and _VISION_EXE.exists() else (
            "A llama.cpp executable is missing. Check whether your antivirus quarantined it "
            "(vendor\\llama.cpp).")


def _catalog_file(projector: bool = False) -> Optional[str]:
    """Select only the configured model bundle, preserving older custom files on disk."""
    model = config.get_model(model_manager.tiered(MODEL_ID)) or {}
    for spec in model.get("files", []):
        name = spec.get("save_as") or Path(spec["filename"]).name
        if ("mmproj" in name.lower()) == projector and spec.get("dest") == "llm":
            if (config.destination_dir("llm") / name).is_file():
                return name
    return None


def _model_file() -> Optional[str]:
    return _catalog_file()


_LOOK_PROMPT = (
    "Describe only this person's physical appearance for an image generator, in one line of at "
    "most 45 words: exact skin tone (e.g. very pale porcelain, fair, olive, tanned, dark), hair "
    "color, length and style, eye color, face shape and distinctive facial features, body type. "
    "No clothing, no background, no pose. If there is no person, answer NONE.")
_LOOKS_FILE = config.destination_dir("llm") / "looks.json"


def _vision(image_path: str, prompt: str, max_tokens: int, schema: Optional[dict] = None,
            size: int = 512) -> Optional[str]:
    """Jedno pytanie do modelu o zdjecie (llama-mtmd-cli + mmproj na karcie). Wolane z zadania
    kolejki - karta nalezy wtedy do niego, wiec najpierw zwalniamy modele ComfyUI. Zdjecie
    zmniejszone do `size` px: kilka razy szybciej, odpowiedz ta sama. None - blad."""
    from PIL import Image, ImageOps

    if ensure():
        return None
    mmproj = _catalog_file(projector=True)
    if not mmproj or not _VISION_EXE.exists() or not _model_file():
        return None
    fd, small = tempfile.mkstemp(suffix=".jpg", prefix="aistudio_look_")
    os.close(fd)
    schema_file = None
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            im.save(small, quality=92)
        args = [str(_VISION_EXE), "-m", str(config.destination_dir("llm") / _model_file()),
                "--mmproj", str(config.destination_dir("llm") / mmproj), "--image", small,
                "-p", prompt, "-n", str(max_tokens), "--temp", "0.2", "-ngl", GPU_LAYERS, "-c", "4096"]
        if schema:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="utf-8") as f:
                json.dump(schema, f)
                schema_file = f.name
            args += ["-jf", schema_file]
        with _GPU_LOCK:
            from .comfy_client import ComfyClient

            cfg = config.SETTINGS["comfyui"]
            client = ComfyClient(cfg["host"], cfg["port"])
            if client.system_stats():
                client.free_memory()
            r = subprocess.run(args, cwd=str(BIN_DIR), stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=600,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        Path(small).unlink(missing_ok=True)
        if schema_file:
            Path(schema_file).unlink(missing_ok=True)
    if r.returncode != 0:
        return None
    return re.sub(r"<think>.*?</think>", "", r.stdout, flags=re.S).replace("[end of text]", "")


def describe_person(image_path: str, progress_cb: Optional[ProgressCb] = None) -> str:
    """Wyglad osoby ze zdjecia pomocniczego (skora, wlosy, oczy, rysy) - jedna linia po
    angielsku albo "" (brak osoby / blad). Wstawiany do promptow, zeby tekst wspieral zdjecie:
    sama "plaza w Dubaju" opalala blada modelke ze zdjecia. Wynik zapamietany wg zawartosci
    pliku (seria zdjec z ta sama referencja liczy go raz); liczenie ok. 25-45 s."""
    import hashlib

    try:
        key = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
        cache = json.loads(_LOOKS_FILE.read_text(encoding="utf-8")) if _LOOKS_FILE.exists() else {}
    except (OSError, ValueError):
        key, cache = "", {}
    if key and key in cache:
        return cache[key]
    if progress_cb:
        progress_cb(0.0, "describing the person in the image...")
    text = _vision(image_path, _LOOK_PROMPT, 120)
    if text is None:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    look = lines[-1].rstrip(".") if lines else ""
    if look.upper().startswith("NONE") or blocked(look):
        look = ""
    if key:
        cache[key] = look
        try:
            _LOOKS_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        except OSError:
            pass
    return look


_PHOTO_PROMPT = (
    "This photo is training data for a character LoRA. Return JSON with: person - true if one "
    "real-looking human is clearly visible; age - your best estimate of that person's age in "
    "years; gender - woman or man; caption - one English sentence of at most 30 words "
    "describing ONLY the clothing, pose, facial expression, framing (close-up, half body or "
    "full body), background and lighting. Do not describe the face, hair, skin, eyes, body "
    "shape or age in the caption.")
_PHOTO_SCHEMA = {
    "type": "object",
    "properties": {"person": {"type": "boolean"}, "age": {"type": "integer"},
                   "gender": {"type": "string", "enum": ["woman", "man"]},
                   "caption": {"type": "string"}},
    "required": ["person", "age", "gender", "caption"],
}


def check_photo(image_path: str) -> Optional[dict]:
    """Zdjecie do treningu postaci: {"person", "age", "gender", "caption"} albo None (blad).
    Wiek sluzy do twardej blokady - postac z osoba niepelnoletnia nie zostanie wytrenowana."""
    text = _vision(image_path, _PHOTO_PROMPT, 160, schema=_PHOTO_SCHEMA, size=448)
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return None
    age, person, gender, caption = (data.get(k) for k in ("age", "person", "gender", "caption"))
    if (type(age) is not int or not 0 <= age <= 130 or type(person) is not bool
            or gender not in ("woman", "man") or not isinstance(caption, str)
            or not caption.strip()):
        return None
    caption = caption.strip().rstrip(".")
    return {"person": person, "age": age, "gender": gender,
            "caption": "" if blocked(caption) else refs.neutral_look(caption)}


# Bez przykladow z nagoscia rozpisana na lewa/prawa piers - model 9B przepisywal je doslownie
# do opisu swetra i bikini (test 31.10)
_COVERAGE_PROMPT = (
    "Inspect only what is actually visible in the image. Return JSON. First classify people: "
    "'none' if there is no human or human body anywhere in the image, 'present' if any person "
    "or part of a human body is visible, 'uncertain' if you cannot tell. Products, objects, "
    "landscapes and empty rooms do not have human anatomy; never invent a person. "
    "If people is 'none', set chest, hips and hands to 'not applicable'. Otherwise describe "
    "all visible people: chest - the actual clothing covering each person's chest, or the "
    "specific uncovered anatomy you can see; hips - the actual clothing covering the crotch "
    "and buttocks, or specific uncovered anatomy you can see; hands - their visible condition. "
    "For a region outside the image or hidden from view say 'not visible'. Do not guess what "
    "is under clothing. Do not assume exposed anatomy when a region is outside the frame. "
    "Describe the observed image, without imagining examples or additional details.")
_COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {"people": {"type": "string", "enum": ["none", "present", "uncertain"]},
                   "chest": {"type": "string"}, "hips": {"type": "string"},
                   "hands": {"type": "string"}},
    "required": ["people", "chest", "hips", "hands"],
    "additionalProperties": False,
}


def coverage(image_path: str) -> Optional[dict]:
    """Person-aware visual coverage. Missing, uncertain or malformed responses fail closed."""
    text = _vision(image_path, _COVERAGE_PROMPT, 220, schema=_COVERAGE_SCHEMA, size=1024)
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict) or data.get("people") not in ("none", "present"):
        return None
    if any(not isinstance(data.get(k), str) or not data[k].strip()
           for k in ("chest", "hips", "hands")):
        return None
    result = {k: data[k].strip() for k in ("people", "chest", "hips", "hands")}
    if result["people"] == "none" and any(
            result[k].casefold() != "not applicable" for k in ("chest", "hips", "hands")):
        return None
    return result


def release_gpu(timeout: float = 120.0) -> None:
    """Przed zadaniem z kolejki: poczekaj, az pomocnik skonczy liczyc na karcie."""
    if _GPU_LOCK.acquire(timeout=timeout):
        _GPU_LOCK.release()


def _clean(text: str) -> str:
    text = text.replace("[end of text]", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^(here('s| is)[^:\n]*:|prompt:)\s*", "", text, flags=re.I).strip()
    text = text.replace("\\n", "\n")
    # kazde "Ujęcie N:" od nowej linii - tylko tak aplikacja dzieli film na ujecia
    text = re.sub(r"\s*\b((?:Uj[ęe]cie|Shot) \d+\s*:)", r"\n\1", text).strip()
    return text.strip("`\"' \n")


def _generate(system: str, user: str, gpu: bool, schema: Optional[dict] = None,
              max_tokens: int = 700, temp: float = 0.8) -> Tuple[bool, str]:
    """Jeden proces llama-completion. schema - wymusza odpowiedz zgodna ze schematem JSON.
    temp - nizsza = spojniej, mniej pomyslow (persona), wyzsza = rozmaitosc (prompty)."""
    model = _model_file()
    if not model:
        return False, "The prompt assistant model is missing."
    # szablon czatu Qwen (ChatML) z pustym blokiem <think> = odpowiedz bez "myslenia"
    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    args = [str(_EXE), "-m", str(config.destination_dir("llm") / model), "-p", prompt,
            "-no-cnv", "--no-escape", "--no-display-prompt", "--simple-io", "-c", "8192",
            "-n", str(max_tokens), "--temp", f"{temp:.2f}", "--top-p", "0.95",
            "-ngl", GPU_LAYERS if gpu else "0"]
    schema_file = None
    if schema:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump(schema, f)
            schema_file = f.name
        args += ["-jf", schema_file]
    try:
        r = subprocess.run(args, cwd=str(BIN_DIR), stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=900,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Prompt assistant could not start: {e}"
    finally:
        if schema_file:
            Path(schema_file).unlink(missing_ok=True)
    if r.returncode != 0:
        tail = " | ".join(r.stderr.strip().splitlines()[-2:])
        return False, f"Prompt assistant engine error ({r.returncode & 0xFFFFFFFF:#x}) {tail[:200]}"
    return True, r.stdout.replace("[end of text]", "").strip()


def ask(system: str, user: str, schema: Optional[dict] = None, max_tokens: int = 700,
        check: Optional[Callable[[str], bool]] = None, retry_suffix: str = "",
        progress_cb: Optional[ProgressCb] = None, desc: str = "Writing the prompt...",
        frac: float = 0.3, temp: float = 0.8) -> Tuple[bool, str]:
    """Odpowiedz modelu (surowy tekst). Na karcie, gdy kolejka nic nie generuje, inaczej na
    procesorze. check - gdy odpowiedz go nie spelnia, druga proba z retry_suffix."""
    from . import queue_manager  # karta tylko wtedy, gdy kolejka nic nie generuje

    def once(gpu: bool) -> Tuple[bool, str]:
        ok, text = _generate(system, user, gpu, schema, max_tokens, temp)
        if ok and check is not None and not check(text):
            ok, text = _generate(system, user + retry_suffix, gpu, schema, max_tokens, temp)
        return ok, text

    if queue_manager.paused():  # pauza karty (np. gra) - pomocnik nie zabiera jej VRAM
        if progress_cb:
            progress_cb(frac, desc + " (CPU — GPU paused)")
        return once(gpu=False)
    if any(j.status == queue_manager.RUNNING for j in queue_manager.snapshot()):
        if progress_cb:
            progress_cb(frac, desc + " (CPU — GPU busy)")
        return once(gpu=False)
    with _GPU_LOCK:
        from .comfy_client import ComfyClient

        cfg = config.SETTINGS["comfyui"]
        client = ComfyClient(cfg["host"], cfg["port"])
        if client.system_stats():
            client.free_memory()  # modele ComfyUI oddaja karte na kilkanascie sekund
        if progress_cb:
            progress_cb(frac, desc)
        return once(gpu=True)


def write_prompt(idea: str, kind: str, current: str = "",
                 progress_cb: Optional[ProgressCb] = None,
                 n_refs: int = 0) -> Tuple[bool, str]:
    """Prompt z pomyslu (albo poprawiony obecny prompt). kind: 'image' / 'video'.
    n_refs - ile zdjec-referencji jest dolaczonych w zakladce (model wskaze je [1], [2]...).
    Tylko SFW: regula dla modelu i sprawdzenie gotowego promptu (app/content.py).
    Zwraca (ok, prompt_albo_komunikat)."""
    idea, current = mark_refs((idea or "").strip(), n_refs), (current or "").strip()
    if not idea and not current:
        return False, "Enter an idea in any language, or a prompt to improve."
    if blocked(idea, current):
        return False, BLOCK_MSG
    refusal = content.check(idea, current)
    if refusal:
        return False, refusal
    err = ensure(progress_cb)
    if err:
        return False, err
    if idea and current:
        user = (f"My idea: {idea}\n\nRewrite and improve this prompt according to "
                f"the idea, keep what still fits:\n{current}")
    elif idea:
        user = f"My idea: {idea}"
    else:
        user = f"Improve this prompt, make it more detailed and precise:\n{current}"
    marks = refs.numbers(f"{idea} {current}")
    if marks:
        user += "\nKeep these image markers in the prompt: " + ", ".join(f"[{n}]" for n in marks)
    user += "\nWrite the prompt in English."

    def fine(raw: str) -> bool:
        """Po angielsku (poza kwestiami w cudzyslowie), ze wszystkimi numerami obrazow,
        a gdy sa dolaczone zdjecia - z co najmniej jednym numerem."""
        text = _clean(raw)
        got = refs.numbers(text)
        return (english(text) and set(marks) <= set(got) and (bool(got) or not n_refs)
                and content.enforce(text) is not None)

    system = _SYSTEM["video" if kind == "video" else "image"] + (refs_rule(n_refs) if n_refs
                                                                 else "") + content.RULE
    ok, text = ask(system, user, check=fine, retry_suffix=_REMIND + content.RULE,
                   progress_cb=progress_cb)
    if not ok:
        return False, text
    text = _clean(text)
    # nadal bez numeru przy jednym zdjeciu - osoba/rzecz z opisu to ta ze zdjecia
    text = re.sub(r"\s*\[(\d{1,2})\]", lambda m: m.group(0) if int(m.group(1)) <= n_refs else "",
                  text)
    if n_refs == 1 and not refs.numbers(text):
        text = f"The subject is the one from the reference photo [1]. {text}"
    if text:
        fixed = content.enforce(text)
        if fixed is None:
            return False, "The model returned content that failed the SFW check. Try again."
        text = fixed
    if not text:
        return False, "The model returned no prompt. Try again."
    if blocked(text):
        return False, BLOCK_MSG
    return True, text


def english(text: str) -> bool:
    """Tekst po angielsku - poza kwestiami w cudzyslowie prawie bez polskich liter."""
    outside = re.sub(r'„[^”]*”|“[^”]*”|«[^»]*»|"[^"]*"', "", text)
    return len(re.findall(r"[ąćęłńóśźż]", outside, re.I)) / max(len(outside), 1) < 0.004
