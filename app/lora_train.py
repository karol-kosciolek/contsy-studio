# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Trening LoRA postaci dla Z-Image-Turbo - musubi-tuner (kohya-ss, Apache 2.0).

Jak w poradniku musubi (docs/zimage.md) plus adapter treningowy ostrisa
(zimage_turbo_training_adapter_v2, Apache 2.0) jako --base_weights: Turbo jest modelem
destylowanym, a adapter na czas treningu "odkreca" destylacje - bez niego LoRA psula szybkie
8-krokowe generowanie. Przy generowaniu adaptera juz nie ma - tylko LoRA postaci. Adapter
jest raz konwertowany do formatu musubi (_adapter) - w oryginalnym nie bylby wczytany.

Zmierzone na RTX 4070 Ti Super 16 GB: 1024 px - 5,5 s/krok i 9,1 GB VRAM (fp8 + 10 blokow
w RAM; bez wymiany blokow 11,2 GB i ta sama predkosc - liczenie, nie przesyl, jest waskim
gardlem), 768 px - ok. 3 s/krok. Dwa zestawy (768 i 1024 px) na przemian: LoRA dziala w
roznych rozmiarach, a sredni krok to ok. 4,2 s.

Trzy procesy po kolei: latenty (VAE), wyjscia enkodera tekstu, trening. Potem konwersja do
formatu, ktory ComfyUI wczytuje bez ostrzezen "lora key not loaded" (convert_lora --target
other). Procesy sa sledzone przez cancel - "Przerwij" w Kolejce zabija biezacy.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import cancel, config, model_manager

ProgressCb = Callable[[float, str], None]

MUSUBI = config.ROOT / "vendor" / "musubi-tuner"
_SRC = MUSUBI / "src" / "musubi_tuner"
ADAPTER_ID = "zimage-train-adapter"
RESOLUTIONS = (768, 1024)
SEC_PER_STEP = 4.2  # srednio dla 768 + 1024 px (do szacowania czasu kolejki)
# Transformer blocks (of 30) kept in RAM: 10 needs 9.1 GB of VRAM at 1024 px; the Lite model set
# (8-12 GB GPUs) keeps 22 there, about 2.5 GB less, at a similar speed (compute-bound).
BLOCKS_TO_SWAP = {"full": 10, "lite": 22}
_STEP_RE = re.compile(r"steps:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
_LOSS_RE = re.compile(r"avr_loss=([\d.]+)")


def steps_for(n_photos: int) -> int:
    """Liczba krokow: wiecej zdjec - dluzej (ok. 2-3 h dla 15-20 zdjec)."""
    return int(min(max(100 * n_photos + 500, 1500), 2400))


def ready() -> Optional[str]:
    """Brak musubi-tuner albo bibliotek -> komunikat; None = mozna trenowac."""
    if not (_SRC / "zimage_train_network.py").exists():
        return ("musubi-tuner is missing (vendor\\musubi-tuner). Run install.bat again to "
                "install character training.")
    try:
        import bitsandbytes  # noqa: F401
        import easydict  # noqa: F401
        import ftfy  # noqa: F401
        import voluptuous  # noqa: F401
    except ImportError as e:
        return f"Training dependency missing ({e.name}). Run install.bat again."
    return None


def _models() -> Tuple[Path, Path, Path]:
    """Full Z-Image weights in every model set: musubi cannot train on the Lite int8 files."""
    files = [model_manager.model_file("z-image-turbo", dest, tier_name="full")
             for dest in ("diffusion_models", "vae", "text_encoders")]
    return tuple(config.destination_dir(dest) / (name or "")
                 for dest, name in zip(("diffusion_models", "vae", "text_encoders"), files))


def _adapter(log: Path) -> Optional[Path]:
    """Adapter ostrisa w formacie musubi (lora_unet_*). Oryginal ma klucze ai-toolkit
    (diffusion_model.*), ktorych musubi przy --base_weights nie dopasowuje - log pokazywal
    "create LoRA for U-Net/DiT: 0 modules", czyli trening szedl bez adaptera. Konwersja raz,
    przy pierwszym treningu."""
    src = config.destination_dir("train") / "zimage_turbo_training_adapter_v2.safetensors"
    dst = src.with_name(src.stem + "_musubi.safetensors")
    if not dst.exists():
        tmp = dst.with_suffix(".tmp")
        if not _run([str(_SRC / "convert_lora.py"), "--input", str(src), "--output", str(tmp),
                     "--target", "default"], log) or not tmp.exists():
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(dst)
    return dst


def _dataset(work: Path, images: Path) -> Path:
    blocks = [f"""[[datasets]]
image_directory = "{images.as_posix()}"
cache_directory = "{(work / f'cache_{res}').as_posix()}"
resolution = [{res}, {res}]
num_repeats = 1
""" for res in RESOLUTIONS]
    toml = work / "dataset.toml"
    toml.write_text("""[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

""" + "\n".join(blocks), encoding="utf-8")
    return toml


def _run(args: List[str], log: Path, on_line: Optional[Callable[[str], None]] = None) -> bool:
    """Proces musubi; wyjscie do logu, linie (takze odswiezenia paska tqdm po \\r) do on_line."""
    env = {**os.environ, "PYTHONPATH": str(MUSUBI / "src"), "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1"}
    with open(log, "a", encoding="utf-8", buffering=1) as f:  # od razu na dysk - podglad
        f.write("\n$ " + " ".join(Path(a).name if i < 2 else a for i, a in enumerate(args)) + "\n")
        proc = subprocess.Popen([sys.executable, *args], cwd=str(MUSUBI), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        cancel.track(proc)
        try:
            buf = b""
            while True:
                chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") \
                    else proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                for raw in parts:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if not _STEP_RE.search(line):  # paskow postepu nie zapisujemy - tysiace linii
                        f.write(line + "\n")
                    if on_line:
                        on_line(line)
            proc.wait()
        finally:
            cancel.untrack(proc)
    cancel.check()
    return proc.returncode == 0


def _tail(log: Path) -> str:
    try:
        lines = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip()]
    except OSError:
        return ""
    errors = [ln for ln in lines if "Error" in ln or "error" in ln]
    return (errors[-1] if errors else (lines[-1] if lines else ""))[:300]


def train(work: Path, images: Path, steps: int, out_file: Path,
          progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """Trening na zdjeciach z `images` (pary obraz + .txt z opisem). Wynik (gotowy dla
    ComfyUI) -> out_file. Zwraca (ok, blad)."""
    err = ready() or model_manager.ensure(["z-image-turbo", ADAPTER_ID], progress_cb, exact=True)
    if err:
        return False, err
    dit, vae, te = _models()
    work.mkdir(parents=True, exist_ok=True)
    log = work / "train.log"
    log.write_text("", encoding="utf-8")
    toml = _dataset(work, images)
    adapter = _adapter(log)
    if adapter is None:
        return False, "Training: could not prepare the training adapter. " + _tail(log)

    def cb(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(frac, desc)

    cb(0.0, "Training: preparing photos (VAE)...")
    # musubi encodes in fp32 without tiling: 8.3 GB of VRAM at 1024 px (measured), so GPUs
    # under 15 GB encode the photos on the CPU instead (a few minutes for a whole set)
    device = ["--device", "cpu"] if model_manager.small_gpu() else []
    if not _run([str(_SRC / "zimage_cache_latents.py"), "--dataset_config", str(toml),
                 "--vae", str(vae), *device], log):
        return False, "Training: photo preparation failed. " + _tail(log)
    cb(0.01, "Training: preparing captions (text encoder)...")
    if not _run([str(_SRC / "zimage_cache_text_encoder_outputs.py"), "--dataset_config",
                 str(toml), "--text_encoder", str(te), "--batch_size", "4", "--fp8_llm"], log):
        return False, "Training: caption preparation failed. " + _tail(log)

    raw = work / "lora_musubi.safetensors"
    raw.unlink(missing_ok=True)

    def on_line(line: str) -> None:
        m = _STEP_RE.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            loss = _LOSS_RE.search(line)
            cb(0.02 + 0.96 * done / max(total, 1),
               f"LoRA training: step {done}/{total}" + (f" · loss {loss.group(1)}" if loss else ""))

    cb(0.02, "LoRA training: loading model...")
    ok = _run([str(_SRC / "zimage_train_network.py"), "--dit", str(dit), "--vae", str(vae),
               "--text_encoder", str(te), "--dataset_config", str(toml), "--sdpa",
               "--mixed_precision", "bf16", "--timestep_sampling", "shift",
               "--weighting_scheme", "none", "--discrete_flow_shift", "2.0",
               "--optimizer_type", "adamw8bit", "--learning_rate", "1e-4",
               "--gradient_checkpointing", "--max_data_loader_n_workers", "0",
               "--network_module", "networks.lora_zimage", "--network_dim", "32",
               "--network_alpha", "32", "--max_train_steps", str(int(steps)), "--seed", "42",
               "--output_dir", str(work), "--output_name", raw.stem,
               "--fp8_base", "--fp8_scaled",
               "--blocks_to_swap", str(BLOCKS_TO_SWAP[model_manager.tier()]),
               "--base_weights", str(adapter)], log, on_line)
    if not ok or not raw.exists():
        return False, "LoRA training failed. " + _tail(log)
    merged = re.search(r"merging module.*?create LoRA for U-Net/DiT: (\d+) modules",
                       log.read_text(encoding="utf-8", errors="replace"), flags=re.S)
    if merged and merged.group(1) == "0":  # adapter nie zadzialal - LoRA psulaby Turbo
        raw.unlink(missing_ok=True)
        return False, "LoRA training: the training adapter was not loaded (0 modules)."
    cb(0.99, "Training: saving LoRA for ComfyUI...")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    staged = work / "lora_comfy.safetensors"
    ok = _run([str(_SRC / "convert_lora.py"), "--input", str(raw), "--output", str(staged),
               "--target", "other"], log)
    raw.unlink(missing_ok=True)
    if not ok or not staged.exists():
        return False, "LoRA conversion failed. " + _tail(log)
    if out_file.exists():
        import shutil
        import time
        shutil.copy2(out_file, out_file.with_name(f"{out_file.stem}_{time.time_ns()}.safetensors"))
    staged.replace(out_file)
    return True, ""
