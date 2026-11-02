# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'Zdjecie' - prompt, format 1080p, opcjonalne obrazy pomocnicze.

Bez obrazow: Z-Image-Turbo (8 krokow) od razu w 1080p. Z obrazami pomocniczymi (np. osoba
[1] w bluzie [2]): Qwen-Image-Edit-2511 sklada je wedlug opisu, SeedVR2 podnosi do 1080p.
Wybrana postac (gorny pasek): Z-Image z jej LoRA; z obrazami pomocniczymi postac jest baza.
Modele pobieraja sie same przy pierwszym uzyciu, styl jest staly - ultra realistyczny.
"""
from __future__ import annotations

import gradio as gr

from . import characters, content, image_gen, llm, model_manager, prefs, qc, queue_manager, refs
from . import ui_assist, ui_queue
from . import workflows
from .ui_theme import note, page_head, section
from .ui_video import gallery_paths

# Dokladny rozmiar wyniku (krotszy bok 1080 px)
SIZES = {"Square 1:1": (1080, 1080), "Landscape 16:9": (1920, 1080), "Portrait 9:16": (1080, 1920),
         "Portrait 3:4": (1080, 1440)}
FORMAT_ALIASES = {"Kwadrat 1:1": "Square 1:1", "Poziomo 16:9": "Landscape 16:9",
                  "Pionowo 9:16": "Portrait 9:16", "Portret 3:4": "Portrait 3:4"}


def image_size(format_name: str | None) -> tuple[int, int]:
    return SIZES.get(FORMAT_ALIASES.get(format_name, format_name), SIZES["Square 1:1"])


def _up16(v: int) -> int:
    return (v + 15) // 16 * 16


def build_images_tab():
    """Buduje strone. Zwraca (funkcja_odswiezania, wejscia, wyjscia) dla auto-odswiezania."""
    page_head("Image", "Describe a scene to create a 1080p image in the queue. With reference images, "
                         "AI combines people, clothing and locations according to your prompt.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["card"]):
            positive = gr.Textbox(
                label="Image prompt", lines=4, max_lines=12,
                info="Write in any language; the AI assistant can refine your idea into English. Refer to images as [1], [2], [3], for example: the man "
                     "[1] wearing the hoodie [2] in front of the building [3].",
                placeholder="A cozy coffee shop at night with a neon sign, rain on the window")
            wire_assistant = ui_assist.build_assistant("image", positive)
            fmt = gr.Radio(list(SIZES), value=next(iter(SIZES)), label="Aspect ratio (1080p)")
            with gr.Accordion("Reference images (optional)", open=False):
                ref_gallery = gr.Gallery(
                    label="Reference images (up to 10): [1], [2], [3]… in upload order",
                    type="filepath", interactive=True, columns=5, height=150,
                    file_types=["image"], object_fit="contain")
                wire_assistant(ref_gallery)
            generate = gr.Button("Generate image", variant="primary", size="lg",
                                 elem_classes=["cta"])
        with gr.Column(scale=4, elem_classes=["card", "result-card"]):
            section("Result", "Your latest completed image. Browse all results in Gallery.")
            gallery = gr.Gallery(show_label=False, columns=1, height=440, preview=True,
                                 buttons=["download", "fullscreen"])
            status = gr.Markdown(elem_classes=["status-box"])

    def add(*values):
        return note(_add(*values))

    def _add(pos_v, refs_v, fmt_v):
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
        params = {"prompt": pos_v, "format": fmt_v}
        who = characters.tag(params)
        job = queue_manager.add("obraz", f"Image {fmt_v}{who}: {pos_v}", params,
                                files={"refs": ref_paths})
        return ui_queue.added_message(job)

    generate.click(add, inputs=[positive, ref_gallery, fmt], outputs=status,
                   trigger_mode="multiple")  # kazde klikniecie = zadanie
    last = gr.State(getattr(queue_manager.latest_finished("obraz"), "id", None))
    gr.Timer(3.0).tick(ui_queue.result_poller("obraz"), inputs=last,
                       outputs=[gallery, status, last])
    return None, [], []


def run_job(p: dict, cb) -> tuple:
    """Wykonanie zadania z kolejki (watek roboczy) -> (ok, pliki, uwagi, blad)."""
    pos_v, ref_paths = p["prompt"], p.get("refs") or []
    refusal = content.check(pos_v)  # takze zadania zapisane przed usunieciem trybu 18+
    if refusal:
        return False, [], [], refusal
    width, height = image_size(p.get("format"))
    ch, ch_note = characters.for_job(p)
    prompt = prefs.styled(pos_v, "image")
    if ref_paths:
        used = [n for n in refs.numbers(pos_v) if n <= len(ref_paths)] or \
            list(range(1, len(ref_paths) + 1))
        used = used[:refs.MAX_REFS]

        def make():
            out = image_gen.edit_with_refs([(n, ref_paths[n - 1]) for n in used], prompt,
                                           width, height, cb, character=ch)
            return out.ok, list(out.files), out.error
    else:
        err = model_manager.ensure(["z-image-turbo"], cb)
        models = image_gen.available_models()
        if err or not models:
            return False, [], [], err or "No image model is available."
        rec = workflows.recommended(workflows.detect_family(models[0]), "jakosc")
        if ch:  # postac: slowo-klucz LoRA, wyglad z LoRA zamiast z opisu
            prompt = prefs.styled(characters.positive(ch, pos_v), "image")

        def make():
            # LoRA uczona tez na aktach rozbierala postac - slabsza LoRA (content.py)
            out = image_gen.txt2img(models[0], prompt, "", _up16(width), _up16(height),
                                    prefs.IMAGE_STEPS, rec["cfg"], -1,
                                    characters.loras(ch, content.LORA_STRENGTH) if ch else None,
                                    1, cb)
            if out.ok:
                for f in out.files:
                    image_gen.crop_to(f, width, height, {"positive": prompt})
            return out.ok, list(out.files), "" if out.ok else out.error
    # straz SFW (postac albo obrazy pomocnicze - przez nie tez mogla wejsc nagosc) i kontrola
    # jakosci postaci: zly wynik do odrzuconych i od nowa (app/qc.py)
    qc_notes: list = []
    ok, files, msg = qc.generate_checked(make, ch, qc_notes, cb, label=f"image: {pos_v[:80]}",
                                         guard=True)
    if not ok:
        return False, [], [], msg
    notes = [n for n in [ch_note, msg, *qc_notes, prefs.autosave(files)] if n]
    return True, files, notes, ""


queue_manager.register("obraz", run_job)
