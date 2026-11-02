# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Uruchamianie i zatrzymywanie ComfyUI jako procesu w tle.

Aplikacja sama startuje silnik, czeka az odpowie, a przy zamknieciu go konczy -
dzieki temu start.bat sprowadza sie do jednej komendy (plug and play).
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

from . import config, model_manager

# Start ComfyUI przez systemowy magazyn certyfikatow (truststore), zeby wezly pobierajace
# modele (Manager) dzialaly tez za antywirusem skanujacym HTTPS.
_BOOTSTRAP = (
    "import runpy, sys\n"
    "try:\n"
    "    import truststore\n"
    "    truststore.inject_into_ssl()\n"
    "except ImportError:\n"
    "    pass\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _startup_error(log_path: Path) -> str:
    """Pokazuje przyczyne problemu CUDA bez importowania torch do procesu UI."""
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(max(0, log_path.stat().st_size - 16384))
            tail = log_file.read().decode("utf-8", errors="replace")
    except OSError:
        tail = ""
    if "Torch not compiled with CUDA enabled" in tail:
        reason = (
            "The installed PyTorch build is CPU-only, without CUDA. "
            "Close Contsy Studio and run install.bat to restore the NVIDIA build."
        )
    elif "Found no NVIDIA driver" in tail or "driver on your system is too old" in tail:
        reason = "PyTorch cannot use the NVIDIA driver. Check your driver installation."
    else:
        reason = "ComfyUI exited unexpectedly."
    return f"{reason} Check the log: {log_path}"


def small_gpu_args() -> List[str]:
    """Below 15 GB of VRAM, keep 2 GB free for working memory.

    Measured with Wan at 480p: the engine leaves about 2.3 GB for activations, which need
    about 3.8 GB. On a smaller card the excess would spill into shared system memory; with
    the reserve, more weights stream from RAM instead."""
    return ["--reserve-vram", "2"] if model_manager.small_gpu() else []


class ComfyProcess:
    def __init__(self) -> None:
        cfg = config.SETTINGS["comfyui"]
        self.host: str = cfg["host"]
        self.port: int = cfg["port"]
        self.extra_args: List[str] = cfg.get("extra_args", [])
        self.timeout_s: int = cfg.get("startup_timeout_s", 180)
        self.proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_up(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def start(self, log_cb=None) -> None:
        """Startuje ComfyUI, jesli jeszcze nie dziala, i czeka na gotowosc."""
        def log(msg: str) -> None:
            if log_cb:
                log_cb(msg)
            print(f"[ComfyUI] {msg}", flush=True)

        if self.is_up():
            log("The engine is already running; connecting to it.")
            return

        main_py = config.COMFYUI_DIR / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(
                f"ComfyUI was not found ({main_py}). Run install.bat first."
            )

        args = [
            sys.executable,
            "-c", _BOOTSTRAP,
            str(main_py),
            "--listen", self.host,
            "--port", str(self.port),
            *small_gpu_args(),
            *self.extra_args,  # last: a user's own --reserve-vram wins
        ]
        log("Starting the engine...")
        # Logi silnika do pliku, zeby konsola aplikacji byla czytelna
        log_path = config.ROOT / "comfyui.log"
        self._log_file = open(log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        constraints = config.ROOT / "venv" / "torch-constraints.txt"
        if constraints.exists():
            # Manager uruchamia pip podczas startu. Chronimy build CUDA rowniez
            # przy bezposrednim python -m app.main, bez start.bat.
            env.setdefault("PIP_CONSTRAINT", str(constraints))
        self.proc = subprocess.Popen(
            args,
            cwd=str(config.COMFYUI_DIR),
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        atexit.register(self.stop)

        # Czekanie na gotowosc
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.stop()
                raise RuntimeError(_startup_error(log_path))
            if self.is_up():
                log("The engine is ready.")
                return
            time.sleep(1.5)
        self.stop()
        raise TimeoutError(
            f"ComfyUI did not start within {self.timeout_s}s. Check the log: {log_path}"
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None
        try:
            self._log_file.close()
        except Exception:
            pass
