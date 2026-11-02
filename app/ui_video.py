# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'Video' - jeden prompt (obraz, dzwiek, dialogi, ujecia), obrazy pomocnicze,
zdjecie startowe, dlugosc, format, dzwiek, tryb.

Result ultra realistyczne 1080p; tryb Szybki liczy film w 480p zamiast 720p, Podglad dodatkowo
pomija 1080p (app/video_gen.py MODES). "Generuj" tylko dodaje zadanie do kolejki
(app/queue_manager.py), gotowy film pojawia sie sam.
"""
from __future__ import annotations

from typing import List, Optional

import gradio as gr

from . import characters, content, history, llm, prefs, qc, queue_manager, refs, ui_assist
from . import ui_queue, video_gen
from .ui_theme import note, page_head, section

ASPECTS = list(video_gen.ASPECTS)
MODE_CHOICES = [(video_gen.MODE_LABELS[m], m) for m in video_gen.MODES]
MODE_INFO = ("Fast renders at 480p rather than 720p, roughly twice as fast (a 5-second clip takes about 6.5 minutes "
             "rather than 14). Skin is smoother and distant faces are less detailed; output is 1080p. "
             "Preview uses Fast without upscaling: about 3 minutes for a 5-second clip at 480p, "
             "for trying ideas (excluded from content packs). Timings depend on hardware and safety checks.")


def gallery_paths(items) -> List[str]:
    """Sciezki z galerii Gradio (lista (sciezka, podpis) albo sciezek), w kolejnosci."""
    out = []
    for it in items or []:
        path = it[0] if isinstance(it, (list, tuple)) else it
        if isinstance(path, dict):
            path = path.get("path") or path.get("name")
        if path:
            out.append(str(path))
    return out


def build_video_tab():
    """Buduje strone. Zwraca (funkcja_odswiezania, wejscia, wyjscia) dla auto-odswiezania."""
    page_head("Video", "Describe a scene to queue a video with audio. You can immediately "
                       "add more jobs.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["card"]):
            positive = gr.Textbox(
                label="Scene prompt", lines=5, max_lines=14,
                info="Describe the subject, action, location, lighting, camera and sounds in any language. Put "
                     "spoken dialogue in quotation marks, preserving your preferred language. Reference images as [1], [2]; "
                     "start the next shot with “Shot 2:”.",
                placeholder="A young man wearing the hoodie [1] walks across a rooftop at sunset, "
                            "wind in his hair, sound of city traffic.\n"
                            "Shot 2: he jumps onto the glass building [2], slow motion")
            wire_assistant = ui_assist.build_assistant("video", positive)
            with gr.Row(equal_height=True):
                seconds = gr.Slider(1, video_gen.MAX_SECONDS, value=5, step=1, label="Duration (s)",
                                    scale=3)
                sound_on = gr.Checkbox(value=True, label="Audio", scale=1, min_width=120,
                                       elem_classes=["switch"])
            aspect = gr.Radio(ASPECTS, value=ASPECTS[0], label="Aspect ratio (1080p · 30 FPS)")
            speed = gr.Radio(MODE_CHOICES, value=video_gen.QUALITY, label="Mode",
                             info=MODE_INFO)
            with gr.Accordion("Reference images and starting image (optional)", open=False):
                ref_gallery = gr.Gallery(
                    label="Reference images (up to 10): [1], [2], [3]… in upload order (for example, "
                          "a hoodie, a person or a building)", type="filepath", interactive=True, columns=5,
                    height=150, file_types=["image"], object_fit="contain")
                wire_assistant(ref_gallery)
                img_input = gr.Image(type="filepath", label="Starting image (the exact first "
                                                            "frame of the video)", height=170)
            generate = gr.Button("Generate video", variant="primary", size="lg",
                                 elem_classes=["cta"])
        with gr.Column(scale=4, elem_classes=["card", "result-card"]):
            section("Result", "Your latest completed video. Browse all results in Gallery.")
            out_video = gr.Video(show_label=False, height=380)
            status = gr.Markdown(elem_classes=["status-box"])

    def add(*values):
        return note(_add(*values))

    def _add(pos_v, refs_v, img_v, seconds_v, aspect_v, sound_v, speed_v):
        """Tylko dodaje zadanie do kolejki - przycisk od razu jest gotowy na kolejne."""
        if not (pos_v or "").strip():
            return "❌ Enter a prompt."
        if llm.blocked(pos_v):
            return llm.BLOCK_MSG
        refusal = content.check(pos_v)
        if refusal:
            return refusal
        ref_paths = gallery_paths(refs_v)
        if len(ref_paths) > refs.MAX_REFS:
            return f"❌ Up to {refs.MAX_REFS} reference images are supported ({len(ref_paths)} provided)."
        missing = [n for n in refs.numbers(pos_v) if n > len(ref_paths)]
        if missing:
            return (f"❌ The prompt refers to image [{missing[0]}], but only "
                    f"{len(ref_paths)} reference images were provided.")
        speed_v = speed_v if speed_v in video_gen.MODES else video_gen.QUALITY
        params = {"prompt": pos_v, "seconds": int(seconds_v), "aspect": aspect_v,
                  "sound": bool(sound_v), "speed": speed_v}
        who = characters.tag(params)
        job = queue_manager.add(
            "wideo", f"Video {int(seconds_v)} s{video_gen.mode_tag(speed_v)}{who}: {pos_v}",
            params, files={"refs": ref_paths, "image": img_v})
        return ui_queue.added_message(job)

    generate.click(add, inputs=[positive, ref_gallery, img_input, seconds, aspect, sound_on,
                                speed],
                   outputs=status, trigger_mode="multiple")  # kazde klikniecie = zadanie
    last = gr.State(getattr(queue_manager.latest_finished("wideo"), "id", None))
    gr.Timer(3.0).tick(ui_queue.result_poller("wideo"), inputs=last,
                       outputs=[out_video, status, last])
    return None, [], []


def run_job(p: dict, cb) -> tuple:
    """Wykonanie zadania z kolejki (watek roboczy) -> (ok, pliki, uwagi, blad)."""
    refusal = content.check(p["prompt"])  # takze zadania zapisane przed usunieciem trybu 18+
    if refusal:
        return False, [], [], refusal
    pre_notes: List[str] = []
    if p.get("image"):  # zdjecie startowe Wan ozywia doslownie - straz SFW przed liczeniem filmu
        cb(0.0, "SFW check: starting image...")
        start = qc.check(p["image"], None, quality=False)
        if start.nude:
            return False, [], [], ("⛔ The starting image contains nudity. Contsy Studio "
                                   "creates SFW content only.")
        if start.unchecked:
            return False, [], [], ("⛔ The SFW check could not verify the starting image. "
                                   "Try again when the vision model is available.")
    ch, ch_note = characters.for_job(p)
    out = video_gen.generate(
        p["prompt"], p["seconds"], p["aspect"], image_path=p.get("image"),
        ref_images=p.get("refs") or [], sound_on=p.get("sound", True),
        style=prefs.STYLE["video"], progress_cb=cb, character=ch,
        speed=p.get("speed", video_gen.QUALITY))
    notes = [n for n in [ch_note, *pre_notes] if n] + out.notes
    # The video engine performs the mandatory final SFW/quality scan before returning success.
    if out.ok:
        notes += [n for n in [prefs.autosave(out.files)] if n]
    return out.ok, out.files, notes, out.error


queue_manager.register("wideo", run_job)
