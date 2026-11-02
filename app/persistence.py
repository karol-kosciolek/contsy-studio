# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Atomic UTF-8 persistence shared by local application stores."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()
