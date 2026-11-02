# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Klient API ComfyUI: kolejkowanie workflow, sledzenie postepu, odbior wynikow.

Workflow przekazujemy w formacie API (slownik wezlow). Postep i bledy sledzimy po
WebSocket (ws://host:port/ws), a gotowe pliki pobieramy przez /history i /view.

VRAM: ComfyUI sam trzyma i wymienia modele w pamieci. Pelne zwolnienie robimy tylko
przy zmianie typu potoku (obraz <-> wideo <-> motion), dzieki czemu kolejne
generacje tym samym modelem nie wczytuja go od nowa z dysku.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests

try:
    import websocket  # websocket-client
except Exception:  # pragma: no cover
    websocket = None  # type: ignore

# Ostatni typ potoku uruchomiony w silniku (wspolny dla calego procesu aplikacji)
_last_pipeline: Optional[str] = None


@dataclass
class GenResult:
    prompt_id: str
    images: List[bytes]
    files: List[dict]  # surowe wpisy z /history (nazwa, subfolder, typ)


class ComfyClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())

    @property
    def http(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"

    # --- Kolejkowanie ---
    def queue_prompt(self, workflow: Dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = requests.post(f"{self.http}/prompt", json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(_validation_message(r))
        return r.json()["prompt_id"]

    def interrupt(self) -> None:
        try:
            requests.post(f"{self.http}/interrupt", timeout=10)
        except requests.RequestException:
            pass

    def upload_image(self, file_path: str) -> str:
        """Wgrywa obraz do ComfyUI (folder input) i zwraca nazwe dla LoadImage."""
        with open(file_path, "rb") as fh:
            files = {"image": (os.path.basename(file_path), fh, "application/octet-stream")}
            r = requests.post(f"{self.http}/upload/image", files=files,
                              data={"overwrite": "true"}, timeout=120)
        r.raise_for_status()
        info = r.json()
        name = info.get("name", os.path.basename(file_path))
        sub = info.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    # --- VRAM ---
    def free_memory(self) -> None:
        """Zwalnia wszystkie modele z VRAM i czeka, az silnik faktycznie odda pamiec.

        ComfyUI obsluguje /free w watku roboczym dopiero po wybudzeniu z kolejki - zadanie
        wyslane zaraz po /free wykonaloby sie PRZED zwolnieniem pamieci. Dlatego czekamy,
        az wolny VRAM karty wyraznie wzrosnie (modele w "dynamic VRAM" ComfyUI nie sa
        widoczne w liczniku PyTorcha); bezczynny silnik robi to od razu, wiec gdy nie bylo
        czego zwalniac, konczymy po 3 s.
        """
        global _last_pipeline
        before = self._vram_free()
        try:
            requests.post(f"{self.http}/free",
                          json={"unload_models": True, "free_memory": True}, timeout=15)
        except requests.RequestException:
            return
        deadline = time.time() + 3
        while time.time() < deadline:
            time.sleep(0.2)
            if self._vram_free() > before + 256 * 1024 ** 2:
                break
        _last_pipeline = None

    def _vram_free(self) -> int:
        devices = self.system_stats().get("devices") or [{}]
        return devices[0].get("vram_free", 0)

    def prepare_pipeline(self, kind: str) -> None:
        """Zwalnia VRAM tylko przy przejsciu na inny typ potoku."""
        global _last_pipeline
        if _last_pipeline is not None and _last_pipeline != kind:
            self.free_memory()
        _last_pipeline = kind

    # --- Wykonanie ---
    def run_and_wait(self, workflow: Dict,
                     progress_cb: Optional[Callable[[float, str], None]] = None,
                     timeout_s: int = 3 * 3600,
                     node_progress_cb: Optional[Callable[[str, int, int], None]] = None
                     ) -> GenResult:
        """Kolejkuje workflow i czeka na koniec. Bledy silnika zamienia na wyjatki.

        node_progress_cb(wezel, wartosc, maks) dostaje surowy postep kazdego wezla - dla
        grafow z kilkoma samplerami (np. dwaj eksperci Wan A14B), gdzie wspolny licznik
        "Krok 1/2" bylby mylacy. Gdy jest podany, zastepuje progress_cb dla postepu.
        """
        if websocket is None:
            raise RuntimeError("websocket-client is missing. Run install.bat.")

        ws = websocket.WebSocket()
        try:
            ws.connect(self.ws_url, timeout=30)
            prompt_id = self.queue_prompt(workflow)
            ws.settimeout(5)
            deadline = time.time() + timeout_s
            peak = 0.0
            while True:
                from . import cancel
                if cancel.requested():
                    self.interrupt()
                    cancel.check()
                if time.time() > deadline:
                    self.interrupt()
                    raise TimeoutError("Generation timed out. Try a shorter clip or close other GPU applications.")
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    done, error = self._history_status(prompt_id)
                    if error:
                        raise RuntimeError(error)
                    if done:
                        break
                    continue
                if isinstance(msg, bytes):
                    continue  # binarne podglady pomijamy
                data = json.loads(msg)
                mtype = data.get("type")
                d = data.get("data", {})
                if d.get("prompt_id") not in (None, prompt_id):
                    continue
                if mtype == "progress" and node_progress_cb:
                    node_progress_cb(str(d.get("node")), d.get("value", 0), d.get("max", 1) or 1)
                elif mtype == "progress" and progress_cb:
                    value, maximum = d.get("value", 0), d.get("max", 1) or 1
                    # kolejne wezly (probkowanie, dekodowanie VAE) licza od zera - pasek sie nie cofa
                    peak = max(peak, value / maximum)
                    progress_cb(peak, f"Step {value}/{maximum}")
                elif mtype == "execution_error":
                    raise RuntimeError(f"{d.get('node_type', 'node')}: "
                                       f"{d.get('exception_message', 'unknown error').strip()}")
                elif mtype == "execution_interrupted":
                    raise RuntimeError("Generation cancelled.")
                elif mtype == "execution_success":
                    break
                elif mtype == "executing" and d.get("node") is None:
                    break
        finally:
            ws.close()

        return self._collect_result(prompt_id)

    def _history_status(self, prompt_id: str) -> Tuple[bool, str]:
        """Zapasowe sprawdzenie stanu przez /history: (zakonczone, komunikat_bledu)."""
        try:
            r = requests.get(f"{self.http}/history/{prompt_id}", timeout=10)
            entry = r.json().get(prompt_id) if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return False, ""
        if not entry:
            return False, ""
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            for kind, info in status.get("messages", []):
                if kind == "execution_error":
                    return True, f"{info.get('node_type')}: {info.get('exception_message', '').strip()}"
            return True, "The engine reported an execution error. See comfyui.log."
        return bool(status.get("completed")), ""

    def _collect_result(self, prompt_id: str) -> GenResult:
        # ComfyUI wysyla execution_success, zanim zapisze wynik w /history (przy duzych
        # wynikach, np. wideo 720p po interpolacji, to kilka sekund) - czekamy na wpis.
        deadline = time.time() + 120
        while True:
            r = requests.get(f"{self.http}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            entry = r.json().get(prompt_id) or {}
            if entry.get("outputs") or entry.get("status", {}).get("completed") \
                    or time.time() > deadline:
                break
            time.sleep(0.5)
        outputs = entry.get("outputs", {})

        images: List[bytes] = []
        files: List[dict] = []
        for node_out in outputs.values():
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_out.get(key, []):
                    files.append(item)
                    if key == "images":
                        images.append(self._view(item))
        return GenResult(prompt_id=prompt_id, images=images, files=files)

    def download_output(self, item: dict) -> bytes:
        """Pobiera dowolny plik wyjsciowy (wideo/audio) wskazany wpisem z /history."""
        return self._view(item)

    def _view(self, item: dict) -> bytes:
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        r = requests.get(f"{self.http}/view", params=params, timeout=120)
        r.raise_for_status()
        return r.content

    def system_stats(self) -> dict:
        try:
            r = requests.get(f"{self.http}/system_stats", timeout=5)
            return r.json() if r.status_code == 200 else {}
        except requests.RequestException:
            return {}


def _validation_message(r: requests.Response) -> str:
    """Czytelny komunikat z odrzuconego workflow (np. brak pliku modelu)."""
    try:
        body = r.json()
    except ValueError:
        return f"ComfyUI rejected the workflow ({r.status_code}): {r.text[:300]}"
    parts = []
    for node in (body.get("node_errors") or {}).values():
        for err in node.get("errors", []):
            parts.append(f"{node.get('class_type')}: {err.get('message')} — {err.get('details')}")
    if parts:
        return "ComfyUI rejected the workflow: " + "; ".join(parts[:3])
    return f"ComfyUI rejected the workflow: {(body.get('error') or {}).get('message', r.text[:300])}"
