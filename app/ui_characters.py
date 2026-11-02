# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'Trening' - wlasne modelki (postacie z LoRA, plug and play).

Nazwa + zdjecia jednej osoby + zgoda -> "Trenuj modelke": zadanie w kolejce (ok. 2-3 h), potem
modelka pojawia sie w wyborze w panelu bocznym. Wybrana dziala w Wideo, Zdjeciu i Serii
z polecenia; "Wszystkie (bez postaci)" generuje bez LoRA. Generator zestawu zdjec (u gory,
zwiniety) robi zdjecia NOWEJ osoby z opisu - ZIP do treningu.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import gradio as gr

from . import char_gen, characters, lora_train, prefs, queue_manager, ui_assist
from .ui_theme import badge, esc, note, page_head, section, table
from .ui_video import gallery_paths

TIPS = (
    "**Choosing photos:** Use 15–25 clear photos of the same adult, with varied views "
    "(portrait, half-length and full-length), expressions, clothing, backgrounds and lighting. "
    "Avoid other people, filters and text. Better photos produce a more consistent character.\n\n"
    "Training runs in the queue (about 2–3 hours of GPU time). Photos identified as depicting "
    "minors are blocked and removed from the training set.")
CONSENT = ("I confirm that this is a fictional AI character or an adult who has consented "
           "to training on their likeness.")


def _cover(ch: dict) -> str:
    if ch.get("preview_safe") is not True:
        return ""
    folder = characters.DIR / ch["slug"]
    preview = folder / "preview.png"
    if preview.exists():
        return str(preview)
    first = sorted((folder / "photos").glob("*.png"))
    return str(first[0]) if first else ""


_STATE_BADGE = {characters.READY: "ok", characters.FAILED: "err", characters.STOPPED: "warn"}


def _state_badge(state: str) -> str:
    kind = _STATE_BADGE.get(state, "info" if state.lower().startswith(("training", "trening"))
                            else {"Ready": "ok", "Failed": "err", "Stopped": "warn"}.get(state, ""))
    return badge(state[:1].upper() + state[1:], kind)


def _rows():
    """(galeria, lista, wybor) dla strony - z aktualnymi statusami z kolejki."""
    chars = characters.all_characters()
    cards = [(c, _cover(c)) for c in chars]
    gallery = [(path, f"{c['name']} · {characters.state(c)}") for c, path in cards if path]
    rows = []
    for c in chars:
        when = (time.strftime("%d.%m.%Y", time.localtime(c["trained"])) if c.get("trained")
                else "—")
        problem = c.get("error") if c.get("status") != characters.READY else c.get("warning")
        name = f"<b>{esc(c['name'])}</b>" + (f"<br><span class='muted'>{esc(problem)}</span>"
                                             if problem else "")
        rows.append([name, _state_badge(characters.state(c)),
                     f"<span class='num'>{c.get('photos', 0)}</span>",
                     f"<span class='muted'>{esc(when)}</span>"])
    listing = table(["Character", "Status", "Photos", "Trained"], rows,
                    empty="<b>No characters yet</b>Add photos on the left and click "
                          "“Train character”.")
    return gallery, listing, [(c["name"], c["slug"]) for c in chars]


def build_characters_tab(tab: gr.Tab, top_picker: gr.Dropdown):
    """tab - strona Trening (odswiezenie przy kazdym wejsciu); top_picker - wybor modelki
    w panelu bocznym (odswiezany po zmianach)."""
    page_head("Training", "Train a character LoRA from 15–25 photos of one adult. Training runs "
                          "in the queue and takes about 2–3 hours. Then select the character in the sidebar.")
    with gr.Accordion("Photo set generator — a new character from a description (training ZIP)",
                      open=False, elem_classes=["accordion-card"]):
        gr.Markdown("Describe a new character’s appearance, or use the AI prompt assistant. A reference "
                    "portrait guides a set of photos with varied views, clothing and locations, "
                    "packaged as a ZIP. Review the photos before training a LoRA; "
                    "generate another set if needed.", elem_classes=["hint"])
        with gr.Row():
            gen_name = gr.Textbox(label="Name", placeholder="For example, Mia", max_lines=1, scale=1)
            gen_count = gr.Slider(char_gen.MIN_COUNT, char_gen.MAX_COUNT,
                                  value=char_gen.DEFAULT_COUNT, step=1, label="Number of photos",
                                  scale=1)
        gen_desc = gr.Textbox(label="Character appearance", lines=3, max_lines=8,
                              placeholder="e.g. a 25-year-old woman with long wavy auburn hair, "
                                          "green eyes, freckles, fair skin, slim build")
        ui_assist.build_assistant("image", gen_desc)(gr.Gallery(visible=False, type="filepath"))
        gen_btn = gr.Button("Generate photo set", variant="primary")
        gen_status = gr.Markdown(elem_classes=["status-box"])
        gen_gallery = gr.Gallery(show_label=False, columns=6, height=260, object_fit="cover")
        gen_zip = gr.File(label="Photo set ZIP — upload below under “New character”", interactive=False)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["card"]):
            section("New character")
            name = gr.Textbox(label="Name", placeholder="For example, Mia", max_lines=1)
            photos = gr.Gallery(
                label=f"Photos of the same adult ({characters.MIN_PHOTOS}–{characters.MAX_PHOTOS}, "
                      "ideally 15–25)", type="filepath", interactive=True, columns=6,
                height=250, file_types=["image"], object_fit="cover")
            photos_zip = gr.File(label="…or upload a ZIP of photos (no extraction needed)",
                                 file_types=[".zip"], type="filepath")
            gr.Markdown(TIPS, elem_classes=["hint"])
            consent = gr.Checkbox(label=CONSENT)
            train = gr.Button("Train character (LoRA)", variant="primary", size="lg",
                              elem_classes=["cta"])
            status = gr.Markdown(elem_classes=["status-box"])
        with gr.Column(scale=4, elem_classes=["card"]):
            section("Your characters")
            gallery0, listing0, names0 = _rows()
            cards = gr.Gallery(value=gallery0, show_label=False, columns=3, height=300,
                               allow_preview=False, object_fit="cover", elem_id="char-cards",
                               buttons=[])
            listing = gr.HTML(listing0)
            # allow_custom_value: lista odswieza sie co kilka sekund, a modelka dodana po starcie
            # aplikacji bylaby do tego czasu odrzucana przez Gradio (slug sprawdza i tak kod)
            pick = gr.Dropdown(names0, value=names0[0][1] if names0 else None,
                               label="Selected character", interactive=True, allow_custom_value=True)
            with gr.Row(elem_classes=["toolbar"]):
                use = gr.Button("Use this character", size="sm", variant="primary")
                retrain = gr.Button("Retrain", size="sm")
                delete = gr.Button("Delete", size="sm", variant="stop")
            info = gr.Markdown(elem_classes=["status-box"])

    def estimate(photos_v):
        n = len(gallery_paths(photos_v))
        if not n:
            return ""
        secs = queue_manager._RATE.get(characters.KIND, 4.5) * lora_train.steps_for(n)
        return (f"{n} photos · estimated training time {queue_manager.fmt_time(secs)}"
                + ("" if n >= characters.MIN_PHOTOS else
                   f" · add at least {characters.MIN_PHOTOS - n} more"))

    photos.upload(estimate, inputs=photos, outputs=status)

    def start(name_v, photos_v, zip_v, consent_v):
        tmp = Path(tempfile.mkdtemp(prefix="aistudio_zip_"))
        try:
            paths = gallery_paths(photos_v)
            if zip_v:
                try:
                    paths += char_gen.photos_from_zip(zip_v, tmp)
                except (OSError, zipfile.BadZipFile) as e:
                    return (note(f"❌ Could not open the ZIP file ({e})."), gr.skip(),
                            gr.skip(), gr.skip())
            ch, msg = characters.create(name_v, paths, bool(consent_v))  # kopiuje zdjecia
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not ch:
            return note(f"❌ {msg}"), gr.skip(), gr.skip(), gr.skip()
        return note(f"✅ {msg}"), "", None, None

    train.click(start, inputs=[name, photos, photos_zip, consent],
                outputs=[status, name, photos, photos_zip])

    def generate(name_v, desc_v, count_v):
        err = char_gen.check(desc_v)
        if err:
            return note(err if err.startswith("⛔") else f"❌ {err}")
        job = char_gen.add((name_v or "").strip(), desc_v, count_v)
        return note(f"➕ Photo set queued (job #{job.id}, about "
                    f"{queue_manager.fmt_time(job.estimate())}). Photos and the ZIP will appear here.")

    gen_btn.click(generate, inputs=[gen_name, gen_desc, gen_count],
                  outputs=gen_status)

    def gen_result(last_id):
        job = queue_manager.latest_finished(char_gen.KIND)
        if job is None or job.id == last_id:
            return gr.skip(), gr.skip(), gr.skip(), last_id
        files = [f for f in job.files if Path(f).exists()]
        if job.status != queue_manager.DONE or not files:
            return gr.skip(), gr.skip(), note(f"❌ {job.error or 'Stopped.'}"), job.id
        return (files[1:], files[0], note("✅ " + " ".join(job.notes)), job.id)

    gen_last = gr.State(getattr(queue_manager.latest_finished(char_gen.KIND), "id", None))
    gr.Timer(4.0).tick(gen_result, inputs=gen_last,
                       outputs=[gen_gallery, gen_zip, gen_status, gen_last])

    def refresh(sig, current):
        """Co kilka sekund: statusy treningu; bez zmian - nic nie przerysowujemy."""
        gallery, text, names = _rows()
        new_sig = json.dumps([text, [g[1] for g in gallery]], ensure_ascii=False)
        if new_sig == sig:
            return gr.skip(), gr.skip(), gr.skip(), sig
        slugs = [s for _, s in names]
        value = current if current in slugs else (slugs[0] if slugs else None)
        return gallery, text, gr.update(choices=names, value=value), new_sig

    sig = gr.State("")
    gr.Timer(4.0).tick(refresh, inputs=[sig, pick], outputs=[cards, listing, pick, sig])
    tab.select(refresh, inputs=[sig, pick], outputs=[cards, listing, pick, sig])

    def on_card(evt: gr.SelectData):
        chars = [c for c in characters.all_characters() if _cover(c)]
        return chars[evt.index]["slug"] if 0 <= evt.index < len(chars) else gr.skip()

    cards.select(on_card, outputs=pick)

    def use_it(slug):
        ch = characters.usable(slug or "")
        if not ch:
            return (note("❌ This character is not ready yet. Wait for training to finish."),
                    gr.skip())
        prefs.save(character=slug)
        return (note(f"✅ “{ch['name']}” is now selected for Video, Image and AI Series."),
                gr.update(choices=characters.choices(), value=slug))

    use.click(use_it, inputs=pick, outputs=[info, top_picker])
    retrain.click(lambda s: note(("✅ " if s else "❌ ") + (characters.retrain(s) if s else
                                                           "Select a character.")),
                  inputs=pick, outputs=info)

    def remove(slug, typed):
        """Usuwa tylko po wpisaniu frazy USUŃ - samo klikniecie moglo byc przypadkowe."""
        phrase = (typed or "").strip().upper()
        if not slug:
            return note("❌ Select a character to delete."), gr.skip()
        if phrase != "DELETE":
            return "Nothing deleted. The confirmation was cancelled or did not match DELETE.", gr.skip()
        msg = characters.delete(slug)
        sel = prefs.load().get("character", "")
        return note(msg), gr.update(choices=characters.choices(), value=sel)

    # okno przegladarki z polem tekstowym: Anuluj albo wpisz USUN i OK -> ukryte pole -> usuwanie
    typed = gr.Textbox(visible=False)
    delete.click(None, outputs=typed,
                 js="() => prompt('Delete the selected character and its LoRA? This cannot be "
                    "undone.\\nTo confirm, type: DELETE') || ''").then(
        remove, inputs=[pick, typed], outputs=[info, top_picker])
