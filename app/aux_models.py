# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pinned, verified auxiliary models used by face tracking and lip sync."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import requests

from .downloader import DownloadError, _preserve_invalid, _verify, download_file

ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((ROOT / "config/auxiliary-models.json").read_text(encoding="utf-8"))


def _verified_url(target: Path, spec: dict) -> Path:
    if target.is_symlink():
        raise DownloadError("Refusing to replace a redirected auxiliary model.")
    if target.exists():
        try:
            _verify(target, spec["size_bytes"], spec["sha256"])
            return target
        except DownloadError:
            _preserve_invalid(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".",
                                         suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            with requests.get(spec["url"], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > spec["size_bytes"]:
                        raise DownloadError("Auxiliary model exceeded its pinned size.")
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        _verify(temporary, spec["size_bytes"], spec["sha256"])
        temporary.replace(target)
        return target
    except requests.RequestException:
        raise DownloadError("Could not download the auxiliary model. Check your connection and retry.") from None
    finally:
        if temporary and temporary.exists():
            # This temporary belongs to this download, never an existing user file.
            temporary.unlink()


def ensure_landmarker() -> Path:
    return _verified_url(ROOT / "models/mediapipe/face_landmarker.task", CATALOG["landmarker"])


def ensure_clip() -> Path:
    spec = CATALOG["clip"]
    folder = ROOT / "models/clip-vit-base-patch32"
    for item in spec["files"]:
        download_file(spec["repo_id"], item["filename"], folder,
                      revision=spec["revision"], expected_size=item["size_bytes"],
                      sha256=item["sha256"])
    return folder
