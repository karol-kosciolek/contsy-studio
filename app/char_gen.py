# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Generator postaci - zestaw zdjec NOWEJ osoby do treningu LoRA (strona Trening).

Z opisu (albo z Pomocnika promptu) powstaje najpierw portret wzorcowy (Z-Image), a z niego
- Qwen-Image-Edit, "ta sama osoba co na obrazie 1" - zdjecia w roznych planach, strojach,
minach, tlach i swietle. Jedna twarz w calym zestawie (osobne generowanie z samego opisu
dawalo za kazdym razem inna osobe). Wynik: folder ze zdjeciami i ZIP gotowy do wgrania
w "Nowa modelka". LoRA nie powstaje automatycznie - zestaw najpierw ogladasz.
Stroje celowo zwykle (jedno zdjecie w stroju kapielowym) - postac przejmuje styl ubran
ze zdjec treningowych.
"""
from __future__ import annotations

import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import List

from . import characters, config, content, history, image_gen, llm, metadata, model_manager, prefs, queue_manager
from . import qc, refs
from . import workflows
from .comfy_client import ComfyClient

KIND = "zestaw"
MIN_COUNT, MAX_COUNT, DEFAULT_COUNT = 12, 30, 20
SIZE = (832, 1216)  # 2:3 ok. 1 MP - jak zdjecia treningowe w tescie (1024 px w treningu)
_ANCHOR = ("Portrait photo, head and shoulders, looking straight at the camera, neutral relaxed "
           "expression, plain light gray studio background, soft even daylight")
SCENES = [
    "close-up portrait, gentle smile, sitting in a cozy cafe, warm window light, knit sweater",
    "half body, white t-shirt and blue jeans, standing on a city street, overcast daylight",
    "full body, walking in an autumn park, long beige coat, soft afternoon light",
    "close-up, laughing and looking slightly to the side, outdoors at golden hour",
    "half body, elegant black evening outfit, rooftop bar at night, city lights bokeh",
    "sitting on a sofa in a bright living room, gray hoodie, relaxed pose",
    "three-quarter view, looking at the camera over the shoulder, plain gray studio background",
    "full body, in a gym, sportswear, holding a water bottle",
    "close-up portrait, serious expression, soft studio lighting, dark background",
    "half body, in a kitchen, striped shirt, holding a coffee mug, morning light",
    "on a beach at sunset, light summer clothes, wind in the hair",
    "selfie-style close-up, smiling, sitting in a car, daylight",
    "full body, winter city street with snow, white puffer jacket, knitted beanie, scarf",
    "half body, navy business suit, modern office, soft window light",
    "full body, standing in a flower field, light linen summer outfit, bright afternoon",
    "half body, black leather jacket, neon-lit street at night, rain",
    "close-up portrait, surprised expression, colorful background",
    "full body, sitting on stone steps, oversized denim jacket, overcast",
    "half body, round glasses and a cardigan, reading a book in a library",
    "profile view close-up, looking into the distance, soft backlight at sunset",
    "half body, swimwear, at a sunny swimming pool",
    "full body, hiking outfit, mountain trail, clear blue sky",
    "close-up portrait, eyes closed, calm expression, soft window light",
    "half body, chef apron, cooking in a modern kitchen",
    "full body, dancing in an empty studio, casual clothes, dynamic pose",
    "close-up portrait, wet hair after rain, dramatic cloudy light",
    "half body, cozy pajamas, sitting on a bed, warm lamp light",
    "full body, riding a bicycle on a quiet street, casual outfit, sunny day",
    "half body, holding a bouquet of flowers, spring garden, soft light",
    "close-up portrait, three-quarter view, rim light, black background",
]


def check(description: str) -> str:
    """Pusty napis = mozna generowac; inaczej komunikat dla uzytkownika."""
    if not (description or "").strip():
        return "Describe the character: gender, age, hair, eyes, skin tone, and style."
    if llm.blocked(description) or llm._MINOR.search(description):
        return "⛔ Only adult characters are supported. This description appears to refer to a minor."
    return content.check(description)


def add(name: str, description: str, count: int) -> queue_manager.Job:
    refusal = check(description)
    if refusal:
        raise ValueError(refusal)
    count = int(min(max(int(count), MIN_COUNT), MAX_COUNT))
    return queue_manager.add(KIND, f"Character image set: {name or 'character'} ({count})",
                             {"name": name or "character", "description": description.strip(),
                              "count": count})


def _english(description: str, cb) -> str:
    """Z-Image i Qwen-Edit lepiej rozumieja angielski - polski opis tlumaczy pomocnik."""
    if llm.english(description):
        return description
    ok, text = llm.write_prompt(f"Character appearance for photos: {description}", "image",
                                progress_cb=cb)
    return text if ok and text and not check(text) else description


def run_job(p: dict, cb) -> tuple:
    """Portret wzorcowy -> zestaw zdjec tej samej osoby -> ZIP."""
    refusal = check(str(p.get("description") or ""))
    if refusal:
        return False, [], [], refusal
    err = model_manager.ensure(["z-image-turbo", refs.MODEL_ID], cb)
    if err:
        return False, [], [], err
    cfg = config.SETTINGS["comfyui"]
    client = ComfyClient(cfg["host"], cfg["port"])
    if not client.system_stats():
        return False, [], [], "ComfyUI is not responding. Run start.bat."
    count = min(max(int(p.get("count") or DEFAULT_COUNT), MIN_COUNT), MAX_COUNT)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = config.OUTPUTS_DIR / f"zestaw_{characters.slugify(p['name'])}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    cb(0.01, "Preparing the character description...")
    look = _english(p["description"], lambda f, d: cb(0.01, d))
    seed = random.randint(0, 2 ** 31 - 1)

    # 1. Portret wzorcowy - z niego bierze sie twarz calego zestawu
    cb(0.03, "Generating the reference portrait...")
    models = image_gen.available_models()
    model = next((m for m in models if workflows.detect_family(m) == "zimage"), None)
    if not model:
        return False, [], [], "Z-Image-Turbo is not available after installation."
    spec = image_gen.resolve_spec(model)
    wf = workflows.txt2img(spec, prefs.styled(f"{look}. {_ANCHOR}", "image"), "", *SIZE, seed,
                           prefs.IMAGE_STEPS, 1.0)
    wf["9"]["inputs"]["filename_prefix"] = "aistudio_set"
    try:
        client.prepare_pipeline("image")
        result = client.run_and_wait(wf)
    except Exception as e:  # noqa: BLE001
        return False, [], [], f"Reference portrait generation failed: {e}"
    if not result.images:
        return False, [], [], "The engine returned no reference portrait."
    anchor = folder / "01.png"
    anchor.write_bytes(result.images[0])
    verdict = qc.check(str(anchor), None, quality=False)
    if not verdict.safe:
        qc.reject(str(anchor), verdict, "character reference portrait")
        return False, [], [], "Reference portrait failed the safety check: " + verdict.summary()

    # 2. Ta sama osoba w roznych ujeciach (Qwen-Image-Edit z portretu)
    scenes = SCENES[:count - 1]
    files: List[str] = [str(anchor)]
    notes: List[str] = []
    for i, scene in enumerate(scenes, 2):
        cb(0.05 + 0.9 * (i - 2) / len(scenes), f"Image {i}/{count}...")
        prompt = ("The same person as in image 1: identical face, facial features, hairstyle, "
                  "hair color, skin tone and body. "
                  + scene + ". Photorealistic photo, natural skin texture.")
        out = folder / f"{i:02d}.png"
        ok, msg = refs.compose(client, [str(anchor)], prompt, *SIZE, out, seed=seed + i)
        if ok:
            files.append(str(out))
        else:
            notes.append(f"Image {i}: {msg}")

    if len(files) < MIN_COUNT:
        return False, [], notes, (f"Only {len(files)} safe images were generated; "
                                  f"at least {MIN_COUNT} are needed. Generate the set again.")

    # 3. ZIP do wgrania w "Nowa modelka" (bez rozpakowywania)
    info = {"mode": "character_set", "model": "Z-Image-Turbo + Qwen-Image-Edit-2511",
            "positive": p["description"], "name": p.get("name", "character"), "seed": seed,
            "files": files, "qc": {"ok": True, "safe": True,
                                   "source": "checked anchor and reference-composition outputs"}}
    for image_path in files:
        metadata.embed_metadata(image_path, info)
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, Path(f).name)
    history.record(info)
    notes.insert(0, f"Set: {len(files)} images. Review them, then upload the ZIP under "
                    f"“New character” and select “Train character (LoRA)”.")
    saved = prefs.autosave([str(zip_path)])
    if saved:
        notes.append(saved)
    cb(1.0, "Done")
    return True, [str(zip_path)] + files, notes, ""


def photos_from_zip(zip_path: str, dest: Path) -> List[str]:
    """Zdjecia z ZIP-a do folderu dest (tylko obrazy, bez sciezek z archiwum, z limitem)."""
    import io
    import warnings
    from PIL import Image

    max_file, max_total = 40 * 1024 * 1024, 250 * 1024 * 1024
    out: List[str] = []
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            if len(z.infolist()) > 1000:
                raise ValueError("The ZIP archive contains too many entries.")
            for info in z.infolist():
                ext = Path(info.filename).suffix.lower()
                if info.is_dir() or ext not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                if len(out) >= 60:
                    break
                if info.file_size > max_file or total + info.file_size > max_total:
                    raise ValueError("The ZIP image set exceeds the extraction size limit.")
                # Read bounded bytes even if the archive's size metadata is incorrect.
                with z.open(info) as src:
                    data = src.read(max_file + 1)
                if len(data) > max_file:
                    raise ValueError("An image in the ZIP exceeds the size limit.")
                total += len(data)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(data)) as im:
                        if im.width * im.height > 40_000_000:
                            raise ValueError("An image in the ZIP is too large.")
                        im.verify()
                target = dest / f"zip_{len(out):03d}{ext}"
                target.write_bytes(data)
                out.append(str(target))
    except Exception:
        for path in out:
            Path(path).unlink(missing_ok=True)
        raise
    return out


queue_manager.register(KIND, run_job)
