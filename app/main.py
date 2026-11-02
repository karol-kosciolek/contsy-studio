# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Contsy Studio - punkt wejscia aplikacji (Gradio).

Uklad jak w aplikacjach SaaS: panel boczny (logo, wybor modelki, menu w grupach, stan karty
i pauza) i strony - Dashboard; Create: Video, Image, AI Series; Characters: Persona
i autopilot, Content Packs, Training; Library: Gallery, GPU Queue; Settings. Wybrana modelka to
tez profil: Dashboard, GPU Queue i Gallery pokazuja wtedy tylko ja. Stan karty odswieza sie sam.
"""
from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse

from . import autopilot, characters, config, prefs, queue_manager, ui_characters, ui_images
from . import studio, ui_history, ui_plan, ui_queue, ui_settings, ui_studio, ui_theme, ui_video
from .comfy_process import ComfyProcess
from .vram import get_gpu_info

# Menu: (etykieta, strona, grupa nad pozycja; "-" = sama kreska). Ikony i naglowki grup: ui_theme.nav_css.
NAV = [("Dashboard", "home", ""),
       ("Video", "video", "Create"), ("Image", "image", ""), ("AI Series", "plan", ""),
       ("Persona & Autopilot", "persona", "Characters"), ("Content Packs", "packs", ""),
       ("Training", "train", ""),
       ("Gallery", "gallery", "Library"), ("GPU Queue", "queue", ""),
       ("Settings", "settings", "-")]
STYLE = {**ui_theme.STYLE, "css": ui_theme.CSS + ui_theme.nav_css(NAV)}

# Globalny uchwyt silnika (ustawiany w main())
ENGINE: ComfyProcess | None = None
ENGINE_STATUS: str = ""


def _loopback_host(host: str) -> bool:
    try:
        return host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class LocalRequestGuard:
    """Reject foreign hosts and cross-origin writes before a UI action can execute."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        host = headers.get("host", "")
        try:
            allowed = _loopback_host(urlsplit("//" + host).hostname or "")
            route_path = scope.get("path", "")
            file_marker = next((m for m in ("/file=", "/file/") if m in route_path), None)
            if file_marker:
                requested = Path(route_path.split(file_marker, 1)[1]).resolve()
                if requested.is_relative_to(config.OUTPUTS_DIR.resolve()) \
                        and requested.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".mp4"):
                    allowed = allowed and studio.is_approved(str(requested))
            if scope["method"] not in ("GET", "HEAD", "OPTIONS"):
                origin = headers.get("origin")
                if origin:
                    parsed = urlsplit(origin)
                    allowed = allowed and parsed.scheme == scope.get("scheme", "http") \
                        and parsed.netloc.lower() == host.lower()
                allowed = allowed and headers.get("sec-fetch-site") != "cross-site"
        except (ValueError, OSError):
            allowed = False
        if not allowed:
            return await PlainTextResponse("Local requests only.", status_code=403)(scope, receive, send)
        await self.app(scope, receive, send)


def local_launch_options(app_cfg: dict) -> dict:
    """Keep the local studio private and expose only completed output files."""
    host = str(app_cfg.get("host", "127.0.0.1")).strip()
    if not _loopback_host(host):
        raise ValueError("The studio host must be a loopback address (127.0.0.1, ::1 or localhost).")
    private = [config.CONFIG_DIR, config.MODELS_DIR, config.COMFYUI_DIR, config.HISTORY_DIR,
               config.ROOT / ".git", config.ROOT / "vendor", config.ROOT / "queue_files",
               config.OUTPUTS_DIR / "odrzucone"]
    approved = studio.approved_files()
    private += [p for p in config.OUTPUTS_DIR.rglob("*") if p.is_file()
                and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".mp4")
                and str(p.resolve()).casefold() not in approved]
    return {"server_name": host, "share": False, "footer_links": [],
            "app_kwargs": {"middleware": [Middleware(LocalRequestGuard)]},
            "allowed_paths": [str(config.OUTPUTS_DIR.resolve())],
            "blocked_paths": [str(p.resolve()) for p in private]}


def status_html() -> str:
    """Stan w dole panelu bocznego: silnik, karta, kolejka."""
    up = bool(ENGINE and ENGINE.is_up())
    paused = queue_manager.paused()
    dot = "pause" if paused else ("" if up else "off")
    state = "Paused — GPU released" if paused else ("Engine ready" if up else
                                                  "Engine unavailable")
    lines = [f"<span class='dot {dot}'></span><b>{state}</b>"]
    gpu = get_gpu_info()
    if gpu.available:
        name = gpu.name.replace("NVIDIA ", "").replace("GeForce ", "")
        free = f"{gpu.free_gb:.1f}"
        lines.append(f"{ui_theme.esc(name)} · {free} of {gpu.total_gb:.0f} GB free")
    else:
        lines.append("No NVIDIA GPU detected")
    snap = queue_manager.snapshot()
    busy = sum(1 for j in snap if j.status in (queue_manager.PENDING, queue_manager.RUNNING))
    lines.append(f"Queue: {busy} · about {queue_manager.fmt_time(queue_manager.remaining_seconds())}"
                 if busy else "Queue empty")
    return "<div class='sb-status'>" + "<br>".join(lines) + "</div>"


def pause_label() -> str:
    return "Resume generation" if queue_manager.paused() else "Pause and release GPU"


# ----------------------------- Budowa UI -----------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Contsy Studio") as demo:
        with gr.Sidebar(width=264, elem_id="sidebar"):
            gr.HTML(ui_theme.BRAND, padding=False)
            # modelka z wlasna LoRA (strona Training) - dotyczy Video, Zdjecia i Serii, i jest
            # profilem: Dashboard, GPU Queue i Gallery pokazuja wtedy tylko ja
            character = gr.Dropdown(characters.choices(), value="", label="Character",
                                    elem_id="character", interactive=True,
                                    allow_custom_value=True)
            nav = gr.Radio([(label, page) for label, page, _ in NAV], value="home",
                           show_label=False, container=False, elem_id="nav")
            with gr.Column(elem_id="sb-foot"):
                status = gr.HTML(status_html(), padding=False)
                # tylko na klikniecie (np. przed gra): przerywa biezace zadanie - wroci do
                # kolejki - i zwalnia pamiec karty; drugie klikniecie wznawia
                gpu_pause = gr.Button(pause_label(), size="sm", elem_id="gpu-pause")
        nav.change(None, inputs=nav, js=ui_theme.NAV_JS)
        gr.Timer(10.0).tick(lambda: (status_html(), pause_label()), outputs=[status, gpu_pause])

        def _toggle_gpu():
            if queue_manager.paused():
                queue_manager.set_paused(False)
                gr.Info("Generation resumed.")
            else:
                stopped = queue_manager.pause_now()
                gr.Info("Paused. Generation has stopped and GPU memory is being released." + (
                    f" Job #{stopped} will return to the queue and restart when generation resumes."
                    if stopped else ""))
            return pause_label(), status_html()
        gpu_pause.click(_toggle_gpu, outputs=[gpu_pause, status])

        with gr.Tabs(selected="home", elem_id="pages"):
            with gr.Tab("Dashboard", id="home") as home_tab:
                dash_fn, _, dash_outs = ui_studio.build_dashboard(home_tab, character)
            with gr.Tab("Video", id="video"):
                ui_video.build_video_tab()
            with gr.Tab("Image", id="image"):
                ui_images.build_images_tab()
            with gr.Tab("AI Series", id="plan"):
                ui_plan.build_plan_tab()
            with gr.Tab("Persona & Autopilot", id="persona") as persona_tab:
                ui_studio.build_persona(persona_tab, character)
            with gr.Tab("Content Packs", id="packs") as packs_tab:
                ui_studio.build_packs(packs_tab, character)
            with gr.Tab("Training", id="train") as train_tab:
                ui_characters.build_characters_tab(train_tab, character)
            with gr.Tab("Gallery", id="gallery") as hist_tab:
                ui_history.build_history_tab(hist_tab)
            with gr.Tab("GPU Queue", id="queue") as queue_tab:
                ui_queue.build_queue_tab(queue_tab)
            with gr.Tab("Settings", id="settings"):
                settings_fn, settings_outs = ui_settings.build_settings_tab()

        # Settings czytane przy kazdym otwarciu strony (nie z chwili startu serwera) -
        # inaczej odswiezona karta pokazywala stare wartosci
        def _characters():
            ch = characters.choices()
            sel = prefs.load()["character"]
            return gr.update(choices=ch, value=sel if any(v == sel for _, v in ch) else "")

        demo.load(_characters, outputs=character)
        demo.load(settings_fn, outputs=settings_outs)
        demo.load(lambda: dash_fn(prefs.load()["character"]), outputs=dash_outs)
        character.input(lambda v: prefs.save(character=v or "") and None, inputs=character)

        def _sync_characters(sig):
            """Nowa wytrenowana / usunieta modelka -> odswiezony wybor w panelu bocznym."""
            new_sig = repr(characters.choices())
            return (gr.skip(), sig) if new_sig == sig else (_characters(), new_sig)
        char_sig = gr.State(repr(characters.choices()))
        gr.Timer(6.0).tick(_sync_characters, inputs=char_sig, outputs=[character, char_sig])

    return demo


def main() -> None:
    global ENGINE, ENGINE_STATUS
    config.ensure_dirs()
    launch_options = local_launch_options(config.SETTINGS["app"])

    ENGINE = ComfyProcess()
    try:
        ENGINE.start()
        ENGINE_STATUS = "ComfyUI engine is running."
    except Exception as e:  # noqa: BLE001 - UI ma dzialac nawet bez silnika
        ENGINE_STATUS = f"ComfyUI engine unavailable: {e}"
        print(f"[WARNING] {ENGINE_STATUS}", flush=True)

    app_cfg = config.SETTINGS["app"]
    queue_manager.start()  # kolejka z dysku + watek roboczy (po kolei, w tle)
    autopilot.start()  # watek autopilota - dziala dopiero po wlaczeniu na Pulpicie
    demo = build_ui()
    try:
        demo.launch(
            **launch_options,
            server_port=app_cfg["port"],
            inbrowser=app_cfg.get("open_browser", True),
            **STYLE,  # Gradio 6: motyw i CSS przekazujemy do launch()
            quiet=True,
        )
    finally:
        if ENGINE:
            ENGINE.stop()


if __name__ == "__main__":
    main()
