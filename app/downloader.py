# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Verified, resumable Hugging Face downloads using the system TLS trust store."""
from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests
from huggingface_hub import hf_hub_url

ProgressCb = Callable[[int, int], None]
CHUNK = 1024 * 1024
_locks: dict[str, threading.Lock] = {}
_lock_guard = threading.Lock()


class DownloadError(Exception):
    pass


def _auth_headers(token: Optional[str]) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)


def _safe_name(name: str) -> str:
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                *[f"LPT{i}" for i in range(1, 10)]}
    if (not name or name in (".", "..") or name.endswith((".", " "))
            or name.split(".", 1)[0].upper() in reserved
            or re.search(r'[\\/:<>"|?*\x00-\x1f]', name)):
        raise DownloadError("Invalid model destination filename.")
    return name


def _verify(path: Path, size: int, sha256: Optional[str]) -> None:
    if not path.is_file() or not path.stat().st_size or (size and path.stat().st_size != size):
        raise DownloadError(f"Incomplete download: {path.name}. Expected {size} bytes; the partial file has been preserved.")
    if sha256:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest.lower() != sha256.lower():
            raise DownloadError(f"Integrity check failed for {path.name}. The file has been preserved; do not use it.")


def _preserve_invalid(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DownloadError("Refusing to preserve a redirected or non-file model path.")
    target = path.with_name(path.name + ".invalid_" + uuid.uuid4().hex)
    path.rename(target)
    print(f"[download] Preserved invalid file: {target.name}", flush=True)
    return target


def download_file(repo_id: str, filename: str, dest_dir: Path,
                  token: Optional[str] = None, progress_cb: Optional[ProgressCb] = None,
                  cancel_flag: Optional[Callable[[], bool]] = None,
                  save_as: Optional[str] = None, revision: str = "main",
                  expected_size: int = 0, sha256: Optional[str] = None) -> Path:
    """Resume into a .part file; publish it only after size and hash verification."""
    name = _safe_name(save_as or filename.replace("\\", "/").rsplit("/", 1)[-1])
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / name
    if final.is_symlink() or final.resolve().parent != dest_dir.resolve():
        raise DownloadError("Model destination must stay inside its configured folder.")
    with _lock_guard:
        lock = _locks.setdefault(str(final.resolve()), threading.Lock())
    with lock:
        if final.exists():
            if not final.is_file():
                raise DownloadError("The model destination is not a regular file.")
            try:
                _verify(final, expected_size, sha256)
            except DownloadError:
                _preserve_invalid(final)
            else:
                if progress_cb:
                    progress_cb(final.stat().st_size, final.stat().st_size)
                return final
        part = final.with_suffix(final.suffix + ".part")
        if part.is_symlink() or (part.exists() and not part.is_file()):
            raise DownloadError("The partial model destination must be a regular file, not a link or directory.")
        existing = part.stat().st_size if part.exists() else 0
        if expected_size and existing > expected_size:
            _preserve_invalid(part)
            existing = 0
        if existing and expected_size and existing == expected_size:
            try:
                _verify(part, expected_size, sha256)
            except DownloadError:
                _preserve_invalid(part)
                existing = 0
            else:
                part.replace(final)
                return final
        headers = _auth_headers(token)
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            with requests.get(_resolve_url(repo_id, filename, revision), headers=headers,
                              stream=True, timeout=(30, 120)) as response:
                if response.status_code in (401, 403):
                    raise DownloadError(f"Access denied for {repo_id}. Review its model card and accept its license on Hugging Face, then set HF_TOKEN if required.")
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != existing:
                        raise DownloadError("The server returned an invalid resume range. The partial file has been preserved.")
                    total = int(match[3])
                    mode = "ab"
                else:
                    existing = 0
                    total = int(response.headers.get("Content-Length", 0))
                    mode = "wb"
                if expected_size and total and total != expected_size:
                    raise DownloadError("The remote model size does not match the verified catalog.")
                total = expected_size or total
                downloaded = existing
                with part.open(mode) as stream:
                    for chunk in response.iter_content(CHUNK):
                        if cancel_flag and cancel_flag():
                            raise DownloadError("Download cancelled. Run it again to resume.")
                        if chunk:
                            stream.write(chunk)
                            downloaded += len(chunk)
                            if total and downloaded > total:
                                raise DownloadError("Download exceeded its expected size; the partial file has been preserved.")
                            if progress_cb:
                                progress_cb(downloaded, total or downloaded)
                    stream.flush()
                    os.fsync(stream.fileno())
                _verify(part, total, sha256)
                part.replace(final)
                if progress_cb:
                    progress_cb(downloaded, total or downloaded)
                return final
        except requests.RequestException as exc:
            raise DownloadError(f"Could not download {name} ({type(exc).__name__}). Check your connection and retry; partial progress is preserved.") from None


def get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
