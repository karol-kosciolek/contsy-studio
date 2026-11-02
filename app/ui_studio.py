# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pulpit oraz strony modelki: 'Persona i autopilot' i 'Paczki'.

Pulpit: wskazniki (kolejka, biezace zadanie, autopilot, wyniki), wlacznik autopilota, kafelki
modelek (klik = jej profil), tabela - stan, zadania w kolejce, wyniki dzis i razem, zapas dni
autopilota, ostatni wynik, uwagi - i najnowsze wyniki. Profil = modelka wybrana w panelu
bocznym: wtedy tylko ona. 'Persona i autopilot': karta persony i rytm (app/autopilot.py).
'Paczki': paczka dnia do publikacji i lista gotowych paczek (app/packs.py).
Liczby z app/studio.py; odswiezanie przy wejsciu na strone, po zmianie modelki i przyciskiem.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr

from . import autopilot, characters, packs, prefs, queue_manager, studio, ui_queue, video_gen
from .ui_theme import LEGACY_NOTICE, badge, esc, head_html, kpis, note, section, table

_STATUS = {characters.NEW: ("Training", "info"), characters.READY: ("Ready", "ok"),
           characters.FAILED: ("Training failed", "err"),
           characters.STOPPED: ("Training stopped", "warn")}
PICK_FIRST = ("<div class='empty'><b>Select a character</b>Choose a character in the sidebar "
              "or click a character card on Dashboard.</div>")


def _ago(ts: float) -> str:
    if not ts:
        return "—"
    secs = time.time() - ts
    if secs < 3600:
        return f"{max(int(secs // 60), 1)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return time.strftime("%d.%m %H:%M", time.localtime(ts))


def _fmt(secs) -> str:
    return queue_manager.fmt_time(secs)


def _buffer(slug: str) -> str:
    """Autopilot buffer: gotowe dni / cel (albo — bez autopilota)."""
    if not slug:
        return "—"
    p = autopilot.persona(slug)
    if not p["enabled"]:
        return "—"
    ready = autopilot.buffer_days(p, autopilot.day_counts(slug, queued=False), dt.date.today())
    return f"{ready} of {p['buffer_days']} days"


def _profile(profile: str) -> str:
    return profile if profile and characters.usable(profile) else ""


# ----------------------------- Pulpit -----------------------------

def _dash_head(profile: str) -> str:
    sub = (f"Character <b>{esc(studio.name(profile))}</b>. Gallery and GPU Queue also show only "
           "this character. Select “All characters” in the sidebar to see everything." if profile else
           "All characters. Click a card to open a character profile.")
    return head_html("Dashboard", sub + ("<br>" + LEGACY_NOTICE if studio.has_legacy_results() else ""))


def _kpis(profile: str, rows: List[dict]) -> str:
    snap = queue_manager.snapshot()
    run = next((j for j in snap if j.status == queue_manager.RUNNING), None)
    pending = [j for j in snap if j.status == queue_manager.PENDING]
    if profile:
        mine = [j for j in studio.jobs(profile, snap)
                if j.status in (queue_manager.PENDING, queue_manager.RUNNING)]
        eta = sum((j.eta() or 0.0) if j.status == queue_manager.RUNNING else j.estimate()
                  for j in mine)
        queue = {"label": "Character jobs", "value": str(len(mine)),
                 "sub": f"about {_fmt(eta)} · entire queue: {len(pending) + bool(run)}"}
    else:
        total = len(pending) + bool(run)
        queue = {"label": "Queued", "value": str(total),
                 "sub": f"estimated time remaining {_fmt(queue_manager.remaining_seconds())}" if total
                 else "queue empty"}
    if queue_manager.paused():
        now = {"label": "Now", "value": "Paused", "sub": "GPU released — no generation running"}
    elif run:
        now = {"label": f"Now · #{run.id}",
               "value": esc(ui_queue.short(ui_queue.label_text(run.label), 80)), "small": True,
               "sub": f"{int(run.progress * 100)}% · estimated time remaining {_fmt(run.eta())}",
               "progress": run.progress}
    else:
        now = {"label": "Now", "value": "Idle", "sub": "GPU waiting for jobs"}
    hours, fits = autopilot.capacity()
    auto = {"label": "Autopilot", "value": "On" if autopilot.enabled() else "Off",
            "sub": f"{_fmt(hours * 3600)} of GPU time per day"
                   + ("" if fits else f" — exceeds {autopilot.HOURS_PER_DAY:.0f} h")}
    today, total = sum(r["today"] for r in rows), sum(r["total"] for r in rows)
    if profile:
        last = {"label": "Autopilot buffer", "value": _buffer(profile),
                "sub": f"results today: {today} · total: {total}"}
    else:
        last = {"label": "Results today", "value": str(today), "sub": f"total: {total}"}
    return kpis([queue, now, auto, last])


def _table(rows: List[dict]) -> str:
    out = []
    for r in rows:
        if r["slug"]:
            text, kind = _STATUS.get(r["status"], (r["status"] or "—", ""))
            state = badge(text, kind)
            name = f"<b>{esc(r['name'])}</b>"
        else:
            state, name = "<span class='muted'>—</span>", esc(r["name"])
        queued = (f"{r['queued']}" + (" · running" if r["running"] else "")
                  + (f" <span class='muted'>(about {_fmt(r['eta'])})</span>" if r["queued"] else ""))
        out.append([name, state, queued, f"<span class='num'>{r['today']}</span>",
                    f"<span class='num'>{r['total']}</span>", esc(_buffer(r["slug"])),
                    f"<span class='muted'>{esc(_ago(r['last']))}</span>",
                    f"<span class='muted'>{esc((r['note'] or '—')[:90])}</span>"])
    return table(["Character", "Status", "Queued", "Today", "Total", "Buffer", "Latest result",
                  "Notes"], out, empty="<b>No characters</b>Add a character in Training.")


def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _auto_status() -> str:
    last = autopilot.STATUS["last"]
    when = time.strftime(" (%H:%M)", time.localtime(autopilot.STATUS["when"])) if last else ""
    hours, fits = autopilot.capacity()
    load = (f"Autopilot characters need about **{_fmt(hours * 3600)}** of GPU time per day out of "
            f"{autopilot.HOURS_PER_DAY:.0f} h" + (" — within capacity." if fits else
                                                   " — **over capacity**. Reduce the daily schedule."))
    return (f"Plans character sessions and generates them in the background when the queue is idle. {load}"
            + (f" Last update{when}: {last}" if last else ""))


def build_dashboard(tab: gr.Tab, character: gr.Dropdown):
    """character - wybor modelki w panelu bocznym (to on jest profilem). Zwraca (funkcja,
    wejscia, wyjscia) do odswiezenia przy otwarciu strony."""
    head = gr.HTML(_dash_head(""), padding=False)
    stats = gr.HTML(padding=False)
    with gr.Row(elem_classes=["card", "tight", "toolbar"]):
        auto_on = gr.Checkbox(label="Autopilot", value=autopilot.enabled(), container=False,
                              elem_classes=["switch"])
        auto_status = gr.Markdown(_auto_status(), elem_classes=["hint", "grow"])
    with gr.Column(elem_classes=["card"]):
        with gr.Row(elem_classes=["toolbar"]):
            with gr.Column(elem_classes=["grow"]):
                section("Characters", "Click a card to open a character profile.")
            show_all = gr.Button("Show all", size="sm", visible=False)
            refresh_btn = gr.Button("Refresh", size="sm")
        tiles = gr.Gallery(show_label=False, columns=6, height="auto", object_fit="cover",
                           allow_preview=False, elem_id="studio-tiles", buttons=[])
        rows_html = gr.HTML()
    with gr.Column(elem_classes=["card"]):
        section("Latest results", "Browse all results in Gallery.")
        recent = gr.Gallery(show_label=False, columns=6, height=360, object_fit="cover",
                            allow_preview=True, elem_id="studio-recent", buttons=["download"])
    slugs = gr.State([])
    rows_st = gr.State([])  # wiersze z ostatniego odswiezenia - dla timera wskaznikow
    outs = [head, stats, tiles, rows_html, recent, slugs, show_all, auto_status, rows_st, auto_on]

    def refresh(profile):
        profile = _profile(profile)
        rows = studio.overview()
        if profile:
            rows = [r for r in rows if r["slug"] == profile]
        else:  # "bez postaci" tylko, gdy cos ma
            rows = [r for r in rows if r["slug"] or r["queued"] or r["total"]]
        tile_rows = [r for r in rows if r["slug"] and r["preview"]]
        cards = [(r["preview"], r["name"]) for r in tile_rows]
        recent_v = [(f, time.strftime("%d.%m %H:%M", time.localtime(_mtime(f))))
                    for f in studio.results(profile, limit=12)]
        return (_dash_head(profile), _kpis(profile, rows), cards, _table(rows), recent_v,
                [r["slug"] for r in tile_rows], gr.update(visible=bool(profile)), _auto_status(),
                rows, autopilot.enabled())

    def pick(slugs_v, evt: gr.SelectData):
        if not (0 <= evt.index < len(slugs_v)) or not characters.usable(slugs_v[evt.index]):
            return (gr.skip(),) * (len(outs) + 1)
        slug = slugs_v[evt.index]
        prefs.save(character=slug)  # profil = modelka do generowania (jak wybor w panelu)
        return (gr.update(value=slug), *refresh(slug))

    def everyone():
        prefs.save(character="")
        return (gr.update(value=""), *refresh(""))

    def toggle(on):
        prefs.save(autopilot=bool(on))
        return _auto_status()

    tab.select(refresh, inputs=character, outputs=outs)
    refresh_btn.click(refresh, inputs=character, outputs=outs)
    character.change(refresh, inputs=character, outputs=outs)
    tiles.select(pick, inputs=slugs, outputs=[character] + outs)
    show_all.click(everyone, outputs=[character] + outs)
    auto_on.input(toggle, inputs=auto_on, outputs=auto_status)
    # biezace zadanie i kolejka co kilka sekund; wyniki (historia) - przy odswiezeniu strony
    gr.Timer(5.0).tick(lambda p, rows: (_kpis(_profile(p), rows or []), _auto_status()),
                       inputs=[character, rows_st], outputs=[stats, auto_status])
    return refresh, [character], outs


# ----------------------------- Persona i autopilot -----------------------------

FIELDS = ["age", "origin", "personality", "style", "wardrobe", "places", "pillars",
          "photos_per_day", "videos_per_day", "video_seconds", "buffer_days",
          "enabled", "language", "video_speed"]


def _values(p: dict) -> list:
    return [("\n".join(p[k]) if k in autopilot._LISTS else p[k]) for k in FIELDS]


def _persona_note(slug: str, p: dict) -> str:
    per_day = autopilot.daily_seconds(p)
    mode = " (Fast mode)" if p["video_speed"] == video_gen.FAST else ""
    return (f"Daily content: **{p['photos_per_day']} images + {p['videos_per_day']} "
            f"videos of {p['video_seconds']} s each**{mode} ≈ "
            f"**{_fmt(per_day)}** of GPU time. Buffer: **{_buffer(slug)}**.")


def _persona_head(profile: str) -> str:
    sub = (f"Character <b>{esc(studio.name(profile))}</b>. Autopilot uses the persona to plan varied "
           "session themes and the schedule to determine how much content to prepare." if profile else
           "Define a character’s personality, style and content schedule for autopilot sessions.")
    return head_html("Persona & Autopilot", sub)


def build_persona(tab: gr.Tab, character: gr.Dropdown) -> None:
    head = gr.HTML(_persona_head(""), padding=False)
    empty = gr.HTML(PICK_FIRST, padding=False)
    with gr.Column(visible=False) as form:
        with gr.Column(elem_classes=["card"]):
            section("Persona", "Enter one item per line in lists. “Suggest” fills in the fields; "
                               "click “Save” below to keep your changes.")
            with gr.Row(equal_height=True):
                hint = gr.Textbox(label="Persona idea (optional, in any language)", scale=4,
                                  placeholder="For example, fitness and healthy food, based in Lisbon")
                suggest_btn = gr.Button("Suggest (AI)", scale=1, min_width=160)
            with gr.Row():
                age = gr.Number(label="Age", precision=0, minimum=21, maximum=60, scale=1)
                origin = gr.Textbox(label="Origin / current location", scale=3)
            with gr.Row():
                personality = gr.Textbox(label="Personality")
                style = gr.Textbox(label="Style")
            with gr.Row():
                wardrobe = gr.Textbox(label="Wardrobe", lines=5)
                places = gr.Textbox(label="Locations", lines=5)
                pillars = gr.Textbox(label="Content themes", lines=5)
        with gr.Column(elem_classes=["card"]):
            section("Autopilot schedule", "Set daily content targets and how many days to prepare ahead.")
            with gr.Row():
                photos = gr.Number(label="Images per day", precision=0, minimum=0, maximum=20)
                videos = gr.Number(label="Videos per day", precision=0, minimum=0, maximum=5)
                seconds = gr.Number(label="Video duration (s)", precision=0, minimum=2, maximum=15)
                buffer = gr.Number(label="Buffer (days ahead)", precision=0, minimum=1,
                                   maximum=30)
            with gr.Row():
                language = gr.Radio([("Polish", "pl"), ("English", "en")], label="Caption language")
                # tylko tryby z gotowym 1080p - Podglad (480p) nie nadaje sie do publikacji
                video_speed = gr.Radio([(video_gen.MODE_LABELS[m], m)
                                        for m in video_gen.PUBLISH_MODES], label="Video mode")
            enabled = gr.Checkbox(label="Enable autopilot for this character", elem_classes=["switch"])
            with gr.Row(elem_classes=["toolbar"]):
                save_btn = gr.Button("Save", variant="primary", min_width=140)
                note_md = gr.Markdown(elem_classes=["status-box", "grow"])
    fields = [age, origin, personality, style, wardrobe, places, pillars, photos, videos,
              seconds, buffer, enabled, language, video_speed]

    def refresh(profile):
        profile = _profile(profile)
        if not profile:
            return (_persona_head(""), gr.update(visible=True), gr.update(visible=False),
                    *[gr.skip()] * len(fields), "")
        p = autopilot.persona(profile)
        return (_persona_head(profile), gr.update(visible=False), gr.update(visible=True),
                *_values(p), _persona_note(profile, p))

    def save(profile, *values):
        if not _profile(profile):
            return note("❌ Select a character first.")
        p = autopilot.save_persona(profile, dict(zip(FIELDS, values)))
        hours, fits = autopilot.capacity()
        warn = "" if fits else (f"\n\nAll autopilot characters need about {hours:.1f} hours "
                                f"per day — exceeding {autopilot.HOURS_PER_DAY:.0f} hours.")
        return note(f"✅ Saved. {_persona_note(profile, p)}{warn}")

    def suggest(profile, hint_v, progress=gr.Progress()):
        ch = characters.usable(profile) if profile else None
        if not ch:
            return [gr.skip()] * len(fields) + [note("❌ Select a character first.")]
        ok, data = autopilot.suggest(ch, hint_v or "",
                                     lambda f, d: progress(min(max(f, 0.0), 1.0), desc=d))
        if not ok:
            return [gr.skip()] * len(fields) + [note(f"❌ {data}")]
        return _values(data) + [note("✨ Suggestion added. Edit it as needed and click "
                                     "“Save”. Your schedule and autopilot settings are unchanged.")]

    outs = [head, empty, form, *fields, note_md]
    tab.select(refresh, inputs=character, outputs=outs)
    character.change(refresh, inputs=character, outputs=outs)
    save_btn.click(save, inputs=[character, *fields], outputs=note_md)
    suggest_btn.click(suggest, inputs=[character, hint], outputs=[*fields, note_md])


# ----------------------------- Paczki -----------------------------

def _pack_days(slug: str) -> List[str]:
    """Dni z gotowymi tresciami autopilota (od dzisiaj i 7 dni wstecz) - do paczki."""
    counts = autopilot.day_counts(slug, queued=False)
    since = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    return sorted(d for d in counts if d >= since)


def _count(folder: Path) -> int:
    return sum(1 for f in folder.rglob("*") if f.is_file()) if folder.is_dir() else 0


def _pack_list(profile: str, limit: int = 20) -> str:
    """Ready packs (paczki/<modelka>/<dzien albo data_godzina>), najnowsze na gorze."""
    roots = [packs.PACKS_DIR / profile] if profile else \
        [d for d in packs.PACKS_DIR.glob("*") if d.is_dir()]
    found: List[Tuple[float, Path]] = [(f.stat().st_mtime, f) for root in roots if root.is_dir()
                                       for f in root.iterdir() if f.is_dir()]
    rows = []
    for mtime, folder in sorted(found, reverse=True)[:limit]:
        who = "General" if folder.parent.name == "ogolne" else studio.name(folder.parent.name)
        rows.append(([f"<b>{esc(folder.name)}</b>"] + ([esc(who)] if not profile else [])
                     + [f"<span class='num'>{_count(folder / n)}</span>"
                        for n in ("instagram", "tiktok")]
                     + [f"<span class='muted'>{time.strftime('%d.%m %H:%M', time.localtime(mtime))}"
                        "</span>"]))
    headers = ["Pack"] + ([] if profile else ["Character"]) + ["Instagram", "TikTok",
                                                               "Created"]
    return table(headers, rows, empty="<b>No packs yet</b>Autopilot creates a pack when a day’s content is "
                                      "complete. You can also build one manually.")


def _packs_head(profile: str) -> str:
    sub = ("Instagram carousels and reels, plus TikTok videos, with captions in the character’s "
           "voice, AI disclosure and suggested posting times.")
    if profile:
        sub = f"Character <b>{esc(studio.name(profile))}</b>. " + sub
    return head_html("Content Packs", sub)


def build_packs(tab: gr.Tab, character: gr.Dropdown) -> None:
    head = gr.HTML(_packs_head(""), padding=False)
    with gr.Column(elem_classes=["card"]):
        section("Daily pack", "Autopilot builds a pack when the day’s content is complete. Rebuilding "
                               "replaces the previous pack, which is moved to the Recycle Bin.")
        pick_hint = gr.Markdown("Select a character in the sidebar to build a daily content pack.",
                                elem_classes=["hint"])
        with gr.Row(elem_classes=["toolbar"], visible=False) as day_row:
            pack_day = gr.Dropdown(label="Day", choices=[], min_width=220)
            pack_btn = gr.Button("Build daily pack", variant="primary")
        pack_note = gr.Markdown(elem_classes=["status-box"])
    with gr.Column(elem_classes=["card"]):
        with gr.Row(elem_classes=["toolbar"]):
            with gr.Column(elem_classes=["grow"]):
                section("Ready packs", "The latest 20 packs. Review the images before publishing "
                                         "and read the included README.txt.")
            open_btn = gr.Button("Open packs folder", size="sm")
        pack_list = gr.HTML()

    def refresh(profile):
        profile = _profile(profile)
        days = _pack_days(profile) if profile else []
        return (_packs_head(profile), gr.update(visible=not profile),
                gr.update(visible=bool(profile)),
                gr.update(choices=days, value=days[-1] if days else None), _pack_list(profile))

    def make_pack(profile, day, progress=gr.Progress()):
        if not (_profile(profile) and day):
            return note("❌ Select a character and a day."), gr.skip()
        progress(0.1, desc="Building the pack and writing captions...")
        ok, msg = packs.build_day(profile, day)
        return note(msg if ok else f"❌ {msg}"), _pack_list(_profile(profile))

    def open_folder(profile):
        folder = packs.PACKS_DIR / (_profile(profile) or "")
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # aplikacja dziala lokalnie - okno Eksploratora u Ciebie
        except (AttributeError, OSError) as e:
            return note(f"❌ Folder: {folder} ({e})")
        return note(f"📂 Opened: {folder}")

    outs = [head, pick_hint, day_row, pack_day, pack_list]
    tab.select(refresh, inputs=character, outputs=outs)
    character.change(refresh, inputs=character, outputs=outs)
    pack_btn.click(make_pack, inputs=[character, pack_day], outputs=[pack_note, pack_list])
    open_btn.click(open_folder, inputs=character, outputs=pack_note)
