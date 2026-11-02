# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Kosz Windows - usuwanie, ktore da sie cofnac (Historia, paczka dnia zlozona ponownie)."""
from __future__ import annotations

from pathlib import Path


def move(path) -> bool:
    """Plik albo folder do Kosza Windows (da sie przywrocic) zamiast trwalego kasowania.
    True, gdy zniknal ze swojego miejsca."""
    import ctypes
    from ctypes import wintypes

    class _Op(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    # FO_DELETE + FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    op = _Op(None, 3, str(Path(path).resolve()) + "\0", None, 0x40 | 0x10 | 0x4 | 0x400,
             False, None, None)
    try:
        return ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op)) == 0 \
            and not Path(path).exists()
    except (AttributeError, OSError):
        return False
