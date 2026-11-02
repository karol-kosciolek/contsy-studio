# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'Seria z polecenia' (dawniej Auto-plan) - polecenie po polsku (+ do 10 zdjec
pomocniczych), AI rozpisuje serie zdjec i filmow i dodaje je do kolejki.

Format, dlugosc i dzwiek dobiera AI z polecenia - osobno dla kazdej grupy. Planowanie dziala
w tle (app/planner.py): przycisk od razu pokazuje pasek postepu, a timer odswieza go, az plan
bedzie gotowy. "Preview plan" pokazuje edytowalna tabele i dodaje ja drugim przyciskiem.
"""
from __future__ import annotations

import html

import gradio as gr

from . import planner, queue_manager, video_gen
from .ui_theme import bar, note, page_head
from .ui_video import MODE_CHOICES, MODE_INFO, gallery_paths

_HEADERS = ["Type", "Aspect ratio", "Seconds", "Audio", "Prompt"]


def _rows(table) -> list:
    """Wiersze z tabeli Gradio (DataFrame albo lista list)."""
    if table is None:
        return []
    if hasattr(table, "values"):
        table = table.values.tolist()
    return [list(r) for r in table if r and str(r[-1] or "").strip()]


def _summary(rows: list, speed: str) -> str:
    photos = sum(1 for r in rows if r[0] == planner.KINDS["image"])
    films = len(rows) - photos
    mode = f" ({video_gen.MODE_LABELS[speed]} mode)" if films else ""
    return (f"**{photos} images, {films} videos**{mode} · estimated generation time "
            f"**{queue_manager.fmt_time(planner.estimate(rows, speed))}**")


def _progress_html(st: dict) -> str:
    pct = int(st["progress"] * 100)
    return (f"<div class='job'><div class='j-title'>AI is preparing your plan…</div>"
            f"{bar(max(st['progress'], 0.03))}"
            f"<div class='j-meta'>{pct}% · {html.escape(st['desc'] or 'start…')} · you can use other "
            f"pages while planning continues</div></div>")


def build_plan_tab():
    page_head("AI Series", "Describe what you want in any language. AI creates a prompt for each "
                            "image and video, chooses aspect ratios, durations and audio, and queues the series.")
    with gr.Column(elem_classes=["card"]):
        order = gr.Textbox(label="Instructions for AI", lines=3, max_lines=10,
                           placeholder="For example, seven portrait videos by the sea at sunset, eight "
                                       "seconds each, and 19 images of the adult in [1] wearing a "
                                       "summer outfit in a hotel room, five in landscape format")
        with gr.Accordion("Reference images (optional, up to 10)", open=False):
            ref_gallery = gr.Gallery(
                label="[1], [2], [3]… in upload order. Refer to them by number, for example, "
                      "“the adult in [1] wearing the dress in [2]”", type="filepath", interactive=True,
                columns=5, height=150, file_types=["image"], object_fit="contain")
        speed = gr.Radio(MODE_CHOICES, value=video_gen.QUALITY, label="Video mode",
                         info=MODE_INFO)
        with gr.Row(elem_classes=["toolbar"]):
            go = gr.Button("Plan and queue series", variant="primary", size="lg",
                           elem_classes=["cta"])
            preview = gr.Button("Preview plan", size="lg", elem_classes=["cta"])
    progress_box = gr.HTML(visible=False, padding=False)
    status = gr.Markdown(elem_classes=["status-box"])
    table = gr.Dataframe(headers=_HEADERS, datatype=["str", "str", "number", "str", "str"],
                         column_count=(5, "fixed"), interactive=True, wrap=True, visible=False,
                         label="Plan — edit prompts, aspect ratios, seconds and audio "
                               "(yes/no). Clear a prompt to skip that row",
                         column_widths=["8%", "11%", "7%", "7%", "67%"])
    add = gr.Button("Queue plan", variant="primary", visible=False,
                    elem_classes=["compact-btn"])
    shown = gr.State(planner.status()["id"])

    def launch(order_v, refs_v, auto_add, speed_v):
        ok, msg = planner.start(order_v, gallery_paths(refs_v), auto_add, speed_v)
        if not ok:
            return (gr.update(visible=False), note(msg if msg.startswith("⛔") else f"❌ {msg}"),
                    gr.update(), gr.update())
        return (gr.update(value=_progress_html(planner.status()), visible=True), "",
                gr.update(visible=False), gr.update(visible=False))

    def tick(shown_id):
        st = planner.status()
        if st["running"]:
            return gr.update(value=_progress_html(st), visible=True), gr.update(), gr.update(), \
                gr.update(), shown_id
        if st["id"] == shown_id or st["ok"] is None:
            return gr.update(visible=False), gr.update(), gr.update(), gr.update(), shown_id
        if not st["ok"]:
            msg = st["msg"] if st["msg"].startswith("⛔") else f"❌ {st['msg']}"
            return gr.update(visible=False), note(msg), gr.update(), gr.update(), st["id"]
        rows = st["rows"]
        if st["auto_add"]:
            msg = note(f"✅ Queued: **{st['added']}** — {_summary(rows, st['speed'])}. "
                       "Track progress in **GPU Queue**.")
            return (gr.update(visible=False), msg, gr.update(value=rows, visible=True),
                    gr.update(visible=False), st["id"])
        msg = (f"Plan ready: {_summary(rows, st['speed'])}. Review it below and click “Queue "
               "plan”.")
        return (gr.update(visible=False), msg, gr.update(value=rows, visible=True),
                gr.update(visible=True), st["id"])

    def add_table(table_v, speed_v):
        rows = _rows(table_v)
        if not rows:
            return note("❌ The plan is empty."), gr.update()
        st = planner.status()
        added, skipped = planner.enqueue(rows, st["refs"], st["order"], speed=speed_v)
        msg = f"✅ Queued: **{added}**. Track progress in **GPU Queue**."
        if skipped:
            msg += f" Skipped empty or disallowed prompts (SFW content only): {skipped}."
        return note(msg), gr.update(visible=False)

    outs = [progress_box, status, table, add]
    go.click(lambda o, r, sp: launch(o, r, True, sp),
             inputs=[order, ref_gallery, speed], outputs=outs)
    preview.click(lambda o, r, sp: launch(o, r, False, sp),
                  inputs=[order, ref_gallery, speed], outputs=outs)
    gr.Timer(1.5).tick(tick, inputs=shown, outputs=outs + [shown])
    add.click(add_table, inputs=[table, speed], outputs=[status, add])
