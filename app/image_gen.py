# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Serwis generowania obrazow - orkiestracja pipeline'u dla Opcji 3.

Dla kazdego trybu: rozwiazujemy pliki modelu, budujemy workflow, uruchamiamy ComfyUI,
zapisujemy wyniki do outputs/ ze znacznikami AI i dopisujemy wpis do historii.
Bledy braku VRAM tlumaczymy na czytelna podpowiedz.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from . import config, content, history, llm, metadata, model_manager, qc, workflows
from .comfy_client import ComfyClient
from .vram import low_vram_hint

ProgressCb = Callable[[float, str], None]

_OOM_HINTS = ("out of memory", "allocation on device", "oom")
ZIMAGE_ID = "z-image-turbo"  # catalog entry; the model set picks full or Lite files
# Checkpointy, ktore nie sa modelami obrazu (wideo/audio) - nie pokazujemy ich w generatorze
_NOT_IMAGE = ("ltx", "audio", "wan2")


@dataclass
class GenOutput:
    ok: bool
    files: List[str] = field(default_factory=list)   # absolutne sciezki zapisanych plikow
    seed: int = -1
    error: str = ""


def _client() -> ComfyClient:
    cfg = config.SETTINGS["comfyui"]
    return ComfyClient(cfg["host"], cfg["port"])


def available_models() -> List[str]:
    """Modele obrazu do wyboru: Z-Image z aktywnego zestawu modeli + checkpointy obrazu."""
    unet = model_manager.model_file(ZIMAGE_ID, "diffusion_models")
    ckpts = [n for n in model_manager.list_custom("checkpoints")
             if not any(k in n.lower() for k in _NOT_IMAGE)]
    return ([unet] if unet else []) + ckpts


def resolve_spec(model_name: str) -> dict:
    """Zamienia nazwe modelu z listy na opis plikow dla workflows.py."""
    family = workflows.detect_family(model_name)
    if family != "zimage":
        return {"family": family, "ckpt": model_name}
    clip = model_manager.model_file(ZIMAGE_ID, "text_encoders")
    vae = model_manager.model_file(ZIMAGE_ID, "vae")
    if not clip or not vae:
        raise FileNotFoundError("Z-Image-Turbo files are missing. Generate an image to download "
                                "them, or run: venv\\Scripts\\python.exe -m app.download_models images")
    return {"family": "zimage", "unet": model_name, "clip": clip, "vae": vae}


def _resolve_seed(seed: int) -> int:
    return random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)


def _save_images(images: List[bytes], prefix: str, info: dict) -> List[str]:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved: List[str] = []
    for i, data in enumerate(images):
        path = config.OUTPUTS_DIR / f"{prefix}_{stamp}_{i}.png"
        path.write_bytes(data)
        metadata.embed_metadata(path, info)
        saved.append(str(path))
    return saved


def _friendly_error(exc: Exception, required_gb: float = 14.0) -> str:
    msg = str(exc)
    if any(h in msg.lower() for h in _OOM_HINTS):
        return "Not enough GPU memory.\n" + low_vram_hint(required_gb)
    return f"Generation failed: {msg}"


def _run(workflow: dict, prefix: str, info: dict, progress_cb: Optional[ProgressCb],
         required_gb: float = 14.0) -> GenOutput:
    client = _client()
    if not client.system_stats():
        return GenOutput(ok=False, error=(
            "ComfyUI is not responding. Start the app with start.bat "
            "(the engine starts automatically), or check comfyui.log."))
    try:
        client.prepare_pipeline("image")
        if progress_cb:
            progress_cb(0.02, "Sending the job to the engine...")
        result = client.run_and_wait(workflow, progress_cb=progress_cb)
    except Exception as e:  # noqa: BLE001 - komunikat trafia do UI
        return GenOutput(ok=False, error=_friendly_error(e, required_gb))

    if not result.images:
        return GenOutput(ok=False, error="The engine returned no image. Check comfyui.log.")
    saved = _save_images(result.images, prefix, info)
    safety_checks = {}
    for path in saved:
        verdict = qc.check(path, None, quality=False)
        if not verdict.safe:
            for candidate in saved:
                if Path(candidate).exists():
                    qc.reject(candidate, verdict, f"image: {info.get('positive', '')[:80]}")
            return GenOutput(ok=False, error="Output failed the safety check: " + verdict.summary())
        safety_checks[path] = verdict.record()
    history.record(dict(info, files=saved, qc={**verdict.record(), "files": safety_checks}))
    return GenOutput(ok=True, files=saved, seed=info.get("seed", -1))


def _spec_or_error(model_name: str):
    try:
        return resolve_spec(model_name), None
    except FileNotFoundError as e:
        return None, GenOutput(ok=False, error=str(e))


# ----------------------------- Tryby -----------------------------

def txt2img(model_name: str, positive: str, negative: str, width: int, height: int,
            steps: int, cfg: float, seed: int, loras: Optional[list] = None,
            batch_size: int = 1, progress_cb: Optional[ProgressCb] = None) -> GenOutput:
    refusal = content.check(positive)
    if refusal or llm.blocked(positive):
        return GenOutput(ok=False, error=refusal or llm.BLOCK_MSG)
    spec, err = _spec_or_error(model_name)
    if err:
        return err
    seed = _resolve_seed(seed)
    wf = workflows.txt2img(spec, positive, negative, width, height, seed, steps, cfg,
                           loras=loras, batch_size=batch_size)
    info = {"mode": "txt2img", "model": model_name, "positive": positive,
            "negative": negative, "width": width, "height": height,
            "steps": steps, "cfg": cfg, "seed": seed, "loras": loras or []}
    return _run(wf, "img", info, progress_cb)


def crop_to(path: str, width: int, height: int, info: dict) -> None:
    """Przycina obraz do dokladnego rozmiaru (srodek) i odnawia metadane AI."""
    from PIL import Image

    with Image.open(path) as im:
        if (im.width, im.height) == (width, height):
            return
        left, top = (im.width - width) // 2, (im.height - height) // 2
        im.crop((left, top, left + width, top + height)).save(path)
    metadata.embed_metadata(path, dict(info, width=width, height=height))


def edit_with_refs(ref_list: List[tuple], positive: str, width: int, height: int,
                   progress_cb: Optional[ProgressCb] = None,
                   character: Optional[dict] = None) -> GenOutput:
    """Obraz z obrazow pomocniczych [(numer, sciezka)] - do 10, skladanych w przebiegach
    (Qwen-Image-Edit-2511) w ok. 1 MP, potem 1080p (SeedVR2) w rozmiarze width x height.
    positive - opis z numerami [n]. character - postac z LoRA: najpierw jej obraz (Z-Image),
    ktory jest baza skladania (glowna osoba), zdjecia pomocnicze dokladaja rzeczy i miejsca."""
    from . import refs, upscale
    from .video_gen import frame_size

    refusal = content.check(positive)
    if refusal or llm.blocked(positive):
        return GenOutput(ok=False, error=refusal or llm.BLOCK_MSG)
    if not ref_list:
        return GenOutput(ok=False, error="No reference images were provided.")
    err = model_manager.ensure([refs.MODEL_ID, upscale.MODEL_ID], progress_cb)
    if err:
        return GenOutput(ok=False, error=err)
    client = _client()
    if not client.system_stats():
        return GenOutput(ok=False, error="ComfyUI is not responding. Run start.bat.")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    draft = config.OUTPUTS_DIR / f"edit_{stamp}_draft.png"
    out = config.OUTPUTS_DIR / f"img_{stamp}_0.png"
    seed = _resolve_seed(-1)
    gw, gh = frame_size(width / height, 1024 * 1024, 16)

    def ecb(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(0.1 + 0.6 * frac, f"Composing references: {desc}")

    base = config.OUTPUTS_DIR / f"edit_{stamp}_character.png"
    try:
        looks = {n: llm.describe_person(path, ecb) for n, path in ref_list}
        start = None
        if character:
            from . import characters

            ok, msg = characters.render(client, character, positive, gw, gh, base, seed,
                                        lambda f, d: ecb(0.0, f"character {character['name']}..."))
            if not ok:
                return GenOutput(ok=False, error=msg)
            start = msg
        ok, msg = refs.compose_refs(client, start, ref_list, positive, gw, gh, draft, seed,
                                    progress_cb=ecb, looks=looks,
                                    base_lead=refs.CHARACTER if start else "")
        if not ok:
            return GenOutput(ok=False, error=msg)
        if character:  # skladanie upraszczalo rysy - twarz z LoRA postaci
            from . import face_fix

            ecb(1.0, "restoring character identity...")
            fixed = draft.with_name(draft.stem + "_face.png")
            if face_fix.fix(client, character, str(draft), fixed, seed=seed)[0]:
                fixed.replace(draft)
        if progress_cb:
            progress_cb(0.75, "1080p (SeedVR2)...")
        ok, note = upscale.image(client, str(draft), out, (width, height))
        if not ok or not out.is_file():
            out.unlink(missing_ok=True)
            return GenOutput(ok=False, error=note or "Upscaling returned no output image.")
    finally:
        draft.unlink(missing_ok=True)
        base.unlink(missing_ok=True)
    info = {"mode": "edit", "model": "Qwen-Image-Edit-2511 + SeedVR2", "positive": positive,
            "width": width, "height": height, "seed": seed, "ref_images": len(ref_list)}
    if character:
        info["character"] = character["name"]
    verdict = qc.check(str(out), character, quality=False)
    if not verdict.safe:
        qc.reject(str(out), verdict, f"image: {positive[:80]}")
        return GenOutput(ok=False, error="Output failed the safety check: " + verdict.summary())
    metadata.embed_metadata(out, info)
    history.record(dict(info, files=[str(out)], qc=verdict.record()))
    if progress_cb:
        progress_cb(1.0, "Done")
    return GenOutput(ok=True, files=[str(out)], seed=seed, error=note)


def img2img(model_name: str, image_path: str, positive: str, negative: str, denoise: float,
            steps: int, cfg: float, seed: int, loras: Optional[list] = None,
            progress_cb: Optional[ProgressCb] = None) -> GenOutput:
    refusal = content.check(positive)
    if refusal or llm.blocked(positive):
        return GenOutput(ok=False, error=refusal or llm.BLOCK_MSG)
    spec, err = _spec_or_error(model_name)
    if err:
        return err
    seed = _resolve_seed(seed)
    try:
        image_name = _client().upload_image(image_path)
    except Exception as e:  # noqa: BLE001
        return GenOutput(ok=False, error=f"Could not upload the image: {e}")
    wf = workflows.img2img(spec, image_name, positive, negative, denoise, seed, steps, cfg,
                           loras=loras)
    info = {"mode": "img2img", "model": model_name, "positive": positive,
            "negative": negative, "denoise": denoise, "steps": steps, "cfg": cfg,
            "seed": seed, "source": image_path, "loras": loras or []}
    return _run(wf, "img2img", info, progress_cb)


def inpaint(model_name: str, image_path: str, mask_path: str, positive: str, negative: str,
            denoise: float, steps: int, cfg: float, seed: int,
            loras: Optional[list] = None,
            progress_cb: Optional[ProgressCb] = None) -> GenOutput:
    refusal = content.check(positive)
    if refusal or llm.blocked(positive):
        return GenOutput(ok=False, error=refusal or llm.BLOCK_MSG)
    spec, err = _spec_or_error(model_name)
    if err:
        return err
    seed = _resolve_seed(seed)
    client = _client()
    try:
        image_name = client.upload_image(image_path)
        mask_name = client.upload_image(mask_path)
    except Exception as e:  # noqa: BLE001
        return GenOutput(ok=False, error=f"Could not upload the image or mask: {e}")
    wf = workflows.inpaint(spec, image_name, mask_name, positive, negative, denoise, seed,
                           steps, cfg, loras=loras)
    info = {"mode": "inpaint", "model": model_name, "positive": positive,
            "negative": negative, "denoise": denoise, "steps": steps, "cfg": cfg,
            "seed": seed, "source": image_path, "loras": loras or []}
    return _run(wf, "inpaint", info, progress_cb)


def upscale(image_path: str, upscale_model: str,
            progress_cb: Optional[ProgressCb] = None) -> GenOutput:
    try:
        image_name = _client().upload_image(image_path)
    except Exception as e:  # noqa: BLE001
        return GenOutput(ok=False, error=f"Could not upload the image: {e}")
    wf = workflows.upscale(image_name, upscale_model)
    info = {"mode": "upscale", "upscale_model": upscale_model, "source": image_path}
    return _run(wf, "upscaled", info, progress_cb, required_gb=4.0)
