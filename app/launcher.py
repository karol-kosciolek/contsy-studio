# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""One-click local startup without package changes or terminating other processes."""
from __future__ import annotations

import socket
import sys
import webbrowser
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

# Browser titles of a running instance; the old product name keeps detecting an instance
# started before an update. The mutex below keeps its original identifier for the same reason.
TITLES = ("Contsy Studio", "Contsy Media Studio")


@contextmanager
def _single_instance(identity: str | None = None):
    """Serialize double-click startup before a second queue worker can be created."""
    if os.name != 'nt':
        yield True
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    root = identity or os.path.normcase(str(Path(__file__).resolve().parent.parent))
    name = 'Local\\ContsyMediaStudio_' + hashlib.sha256(root.encode()).hexdigest()[:24]
    ctypes.set_last_error(0)
    handle = kernel.CreateMutexW(None, True, name)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Could not protect app startup.')
    owned = ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS
    try:
        yield owned
    finally:
        if owned:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)

def main():
    with _single_instance() as owned:
        if not owned:
            from . import config
            url = f"http://127.0.0.1:{config.SETTINGS['app']['port']}"
            print('Contsy Studio is already starting or running. Opening the local app.')
            webbrowser.open(url)
            return 0
        return _main()


def _main():
    from . import config
    cfg = config.SETTINGS['app']
    url = f"http://127.0.0.1:{cfg['port']}"
    with socket.socket() as probe:
        busy = probe.connect_ex(('127.0.0.1', cfg['port'])) == 0
    if busy:
        import requests
        try:
            response = requests.get(url + '/config', timeout=5)
            if response.ok and response.json().get('title') in TITLES:
                print('Contsy Studio is already running. Opening the local app.')
                webbrowser.open(url)
                return 0
        except (requests.RequestException, ValueError):
            pass
        print(f"[ERROR] Port {cfg['port']} is used by another application. Close it or choose another app port in config/settings.local.json.")
        return 1
    from .main import main as start
    start()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print(f'[ERROR] Contsy Studio could not start: {exc}')
        print('Run install.bat to repair dependencies. Review config/settings.local.json for configuration errors.')
        raise SystemExit(1)
