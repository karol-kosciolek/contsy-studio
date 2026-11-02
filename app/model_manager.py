# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Menedzer modeli: stan instalacji, pobieranie, usuwanie, zestawy, modele wlasne.

Katalog modeli pochodzi z config/models.json. Model moze byc paczka kilku plikow
(np. model + enkoder + VAE), a plik moze pochodzic z innego repo niz reszta.
Stan "zainstalowany" = wszystkie pliki modelu leza w katalogach docelowych.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

from . import config, prefs, vram
from .downloader import download_file, get_hf_token, _safe_name

ProgressCb = Callable[[float, str], None]
_VERIFIED = {}
_VERIFY_LOCK = threading.Lock()


def _verified_file(target: Path, file_spec: dict) -> bool:
    """Verify catalog bytes once per file identity; never trust size alone."""
    try:
        if target.is_symlink() or not target.is_file():
            return False
        stat = target.stat()
        if not stat.st_size or (file_spec.get("size_bytes") and stat.st_size != file_spec["size_bytes"]):
            return False
        expected = file_spec.get("sha256")
        if not expected:
            return True
        key = (str(target.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
        cacheable = stat.st_size >= 64 * 1024 * 1024
        with _VERIFY_LOCK:
            if cacheable and key in _VERIFIED:
                return _VERIFIED[key]
        with target.open("rb") as stream:
            valid = hashlib.file_digest(stream, "sha256").hexdigest().lower() == expected.lower()
        if cacheable:
            with _VERIFY_LOCK:
                _VERIFIED[key] = valid
        return valid
    except OSError:
        return False


@dataclass
class ModelStatus:
    model: dict
    installed: bool
    size_on_disk_gb: float = 0.0
    missing_files: List[str] = field(default_factory=list)


def _file_target(file_spec: dict) -> Path:
    name = file_spec.get("save_as") or Path(file_spec["filename"]).name
    return config.destination_dir(file_spec["dest"]) / _safe_name(name)


def model_status(model: dict) -> ModelStatus:
    """Sprawdza, czy wszystkie pliki modelu sa na dysku."""
    missing: List[str] = []
    size = 0
    for fs in model["files"]:
        target = _file_target(fs)
        if _verified_file(target, fs):
            size += target.stat().st_size
        else:
            missing.append(target.name)
    return ModelStatus(model=model, installed=not missing,
                       size_on_disk_gb=size / (1024 ** 3), missing_files=missing)


def all_statuses() -> List[ModelStatus]:
    return [model_status(m) for m in config.CATALOG["models"]]


# ----------------------------- Model sets -----------------------------
# "full" uses the catalog's base entries. "lite" swaps in each "<id>-lite" entry: smaller
# quantizations of the same models for GPUs with less VRAM, RAM and disk. Character training
# always uses the full Z-Image weights (exact ids).
TIERS = ("full", "lite")
LITE_BELOW_GB = 15.0


@lru_cache(maxsize=1)
def _gpu_total_gb() -> float:
    return vram.get_gpu_info().total_gb


def small_gpu() -> bool:
    """A GPU with less than 15 GB of VRAM (unknown counts as large)."""
    return 0 < _gpu_total_gb() < LITE_BELOW_GB


def default_tier() -> str:
    """Lite on a GPU with less than 15 GB of VRAM; full otherwise or when it is unknown."""
    return "lite" if small_gpu() else "full"


def tier() -> str:
    """The model set chosen in Settings, or the one suited to this GPU."""
    chosen = prefs.load().get("tier")
    return chosen if chosen in TIERS else default_tier()


def tiered(model_id: str, tier_name: Optional[str] = None) -> str:
    """Catalog entry that serves `model_id` in the given (default: current) model set."""
    lite = model_id + "-lite"
    return lite if (tier_name or tier()) == "lite" and config.get_model(lite) else model_id


def model_file(model_id: str, dest: str, *subs: str, tier_name: Optional[str] = None) -> Optional[str]:
    """Installed file of the model set's entry in `dest` whose name contains every fragment.

    Selection follows the catalog, so full and Lite files can sit side by side."""
    model = config.get_model(tiered(model_id, tier_name)) or {}
    for spec in model.get("files", []):
        target = _file_target(spec)
        if spec["dest"] == dest and all(s in target.name.lower() for s in subs) and target.is_file():
            return target.name
    return None


def is_installed(model_id: str, exact: bool = False) -> bool:
    """exact=False checks the entry of the current model set (see tiered)."""
    model = config.get_model(model_id if exact else tiered(model_id))
    return model is not None and model_status(model).installed


def ensure(model_ids: List[str], progress_cb: Optional[ProgressCb] = None,
           exact: bool = False) -> Optional[str]:
    """Plug and play: pobiera brakujace modele przed pierwszym uzyciem.

    Ids are mapped to the current model set unless exact=True.
    Zwraca komunikat bledu albo None, gdy wszystko jest na dysku.
    """
    if not exact:
        model_ids = [tiered(m) for m in model_ids]
    missing = [m for m in dict.fromkeys(model_ids) if not is_installed(m, exact=True)]
    for i, mid in enumerate(missing):
        model = config.get_model(mid)
        if model is None:
            return f"Unknown model: {mid}. Check the installed model catalog."
        name = model["name"].split(" — ")[0]

        def cb(frac: float, desc: str, _i=i, _name=name) -> None:
            if progress_cb:
                progress_cb((_i + frac) / len(missing),
                            f"First use — downloading {_name}: {desc}")

        try:
            download_model(mid, progress_cb=cb)
        except Exception as e:  # noqa: BLE001 - komunikat trafia do UI
            return f"Could not download {name}: {e}"
    return None


def download_model(model_id: str, progress_cb: Optional[ProgressCb] = None,
                   cancel_flag: Optional[Callable[[], bool]] = None) -> None:
    """Pobiera wszystkie pliki modelu (istniejace pomija). progress_cb(0-1, opis)."""
    model = config.get_model(model_id)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")

    token = get_hf_token()
    files = model["files"]
    n = len(files)
    for i, fs in enumerate(files):
        name = fs.get("save_as") or Path(fs["filename"]).name

        def file_cb(done: int, total: int, _i=i, _name=name) -> None:
            if progress_cb is None:
                return
            frac = (done / total) if total else 0.0
            progress_cb((_i + frac) / n,
                        f"{_name}: {done / 2**20:.0f}/{total / 2**20:.0f} MB")

        download_file(repo_id=fs.get("repo_id", model["repo_id"]), filename=fs["filename"],
                      dest_dir=config.destination_dir(fs["dest"]), token=token,
                      progress_cb=file_cb, cancel_flag=cancel_flag, save_as=fs.get("save_as"),
                      revision=fs.get("revision", "main"), expected_size=fs.get("size_bytes", 0),
                      sha256=fs.get("sha256"))
    if progress_cb:
        progress_cb(1.0, "Ready")


def download_set(set_key: str, progress_cb: Optional[ProgressCb] = None) -> List[str]:
    """Pobiera caly zestaw modeli (np. 'obrazy'). Zwraca liste pobranych id."""
    ids = config.CATALOG.get("sets", {}).get(set_key, {}).get("models", [])
    n = len(ids)
    for i, mid in enumerate(ids):
        name = config.get_model(mid)["name"]

        def cb(frac: float, desc: str, _i=i, _name=name) -> None:
            if progress_cb:
                progress_cb((_i + frac) / n, f"[{_i + 1}/{n}] {_name} — {desc}")

        download_model(mid, progress_cb=cb)
    return ids


def delete_model(model_id: str) -> List[str]:
    """Usuwa pliki modelu. Pomija pliki wspoldzielone z innymi zainstalowanymi modelami."""
    model = config.get_model(model_id)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")
    shared = {
        _file_target(fs)
        for other in config.CATALOG["models"] if other["id"] != model_id
        and model_status(other).installed
        for fs in other["files"]
    }
    removed: List[str] = []
    for fs in model["files"]:
        target = _file_target(fs)
        if target.exists() and target not in shared:
            target.unlink()
            removed.append(target.name)
    return removed


# --- Modele wlasne (wgrywane przez uzytkownika) ---

def add_custom_file(src_path: str, kind: str) -> Path:
    """Kopiuje wlasny checkpoint lub LoRA do odpowiedniego katalogu ComfyUI."""
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src_path}")
    if kind not in ("checkpoints", "loras"):
        raise ValueError("Model type must be 'checkpoints' or 'loras'.")
    if src.suffix.lower() != ".safetensors":
        raise ValueError("Import custom checkpoints and LoRAs as .safetensors files. Pickle model formats are not accepted.")
    dest = config.destination_dir(kind)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / src.name
    if target.exists():
        raise FileExistsError(f"A model named {src.name} already exists. Rename the imported file first.")
    shutil.copy2(src, target)
    return target


def list_custom(kind: str) -> List[str]:
    """Pliki modeli w danym katalogu docelowym."""
    dest = config.destination_dir(kind)
    if not dest.exists():
        return []
    exts = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".onnx"}
    return sorted(p.name for p in dest.iterdir() if p.suffix.lower() in exts)


def find_file(kind: str, *subs: str) -> Optional[str]:
    """Pierwszy plik w katalogu `kind`, ktorego nazwa zawiera wszystkie fragmenty."""
    for name in list_custom(kind):
        low = name.lower()
        if all(s in low for s in subs):
            return name
    return None
