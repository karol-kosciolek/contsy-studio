# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pomocnik promptu na stronach Video i Zdjecie (app/llm.py).

Pomysl po polsku -> nowy prompt po angielsku wpisany w pole opisu (poprzedni opis nie jest
doklejany - wczesniej stary prompt "przeciekal" do nowego). Puste pole pomyslu = popraw obecny
opis. Pomocnik wie, ile zdjec pomocniczych dolaczono w zakladce, i wskazuje je [1], [2].
Model dziala lokalnie i pisze tylko tresci SFW (app/content.py).
"""
from __future__ import annotations

from typing import Callable

import gradio as gr

from . import llm
from .ui_theme import note


def build_assistant(kind: str, prompt_box: gr.Textbox) -> Callable[..., None]:
    """kind: 'video' / 'image'; prompt_box - pole opisu, do ktorego trafia wynik. Zwraca
    funkcje, ktora podpina galerie zdjec pomocniczych (tworzona nizej na stronie)."""
    with gr.Accordion("AI prompt assistant — describe your idea in any language", open=False):
        idea = gr.Textbox(show_label=False, lines=2, max_lines=6,
                          placeholder="For example, an adult in a green hoodie jumps between rooftops at "
                                      "sunset, then says “Let’s go!” to the camera. With reference "
                                      "images, refer to “the person in image [1]”. "
                                      "Leave blank to improve the current prompt.")
        with gr.Row(elem_classes=["toolbar"]):
            write = gr.Button("Write prompt", size="sm")
            status = gr.Markdown(elem_classes=["status-box", "grow"])

    def wire(ref_gallery: gr.Gallery) -> None:
        from .ui_video import gallery_paths

        def run(idea_v, current_v, refs_v, progress=gr.Progress()):
            def cb(frac, desc):
                progress(min(max(frac, 0.0), 1.0), desc=desc)

            idea_v = (idea_v or "").strip()
            ok, text = llm.write_prompt(idea_v, kind, "" if idea_v else current_v, cb,
                                        n_refs=len(gallery_paths(refs_v)))
            if not ok:
                return gr.update(), note(f"❌ {text}" if not text.startswith("⛔") else text)
            return text, note("✅ The prompt has been added above. Review or edit it, then click "
                              "Generate.")

        write.click(run, inputs=[idea, prompt_box, ref_gallery], outputs=[prompt_box, status])

    return wire
