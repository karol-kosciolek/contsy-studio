# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'Kolejka' i wspolne elementy kolejki dla stron Wideo / Zdjecie.

Biezace zadanie z paskiem postepu i przyciskami Przerwij / Wstrzymaj, lista oczekujacych
z zaznaczaniem (usun zaznaczone, wyczysc kolejke) i lista zakonczonych. Odswiezanie co 2 s;
liste oczekujacych przebudowujemy tylko, gdy sie zmienila - inaczej znikalyby zaznaczenia.
Przy wybranej modelce (panel boczny) lista, zakonczone i "Wyczysc" dotycza tylko jej zadan
(app/studio.py); biezace zadanie widac zawsze - karta jest jedna.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import gradio as gr

from . import queue_manager as qm
from . import studio
from .ui_theme import badge, bar, esc, note, page_head, section, table

KINDS = {"wideo": "Video", "obraz": "Image", "narracja": "Narration", "trening": "Training",
         "zestaw": "Photo set"}
STATUS = {qm.DONE: ("Complete", "ok"), qm.ERROR: ("Failed", "err"), qm.STOPPED: ("Stopped", "warn"),
          qm.PENDING: ("Queued", ""), qm.RUNNING: ("Running", "info")}
_LEAD = re.compile(r"^[\U0001F300-\U0001FAFF☀-➿]️?\s*")
_LEGACY_LABEL = re.compile(
    r"^(?P<series>plan\s*·\s*)?(?P<kind>obraz|wideo)\s*"
    r"(?P<spec>(?:(?:Kwadrat|Poziomo|Pionowo|Portret)\s*)?\d+\s*:\s*\d+|\d+\s*s)"
    r"(?P<tags>(?:\s*·\s*[^:\n]+)*):")
_FORMAT_NAMES = {"Kwadrat": "Square", "Poziomo": "Landscape", "Pionowo": "Portrait",
                 "Portret": "Portrait"}
_SET_NOTE = re.compile(
    r"Zestaw: (\d+) zdjęć\. Obejrzyj je - jeśli postać pasuje, wgraj plik ZIP w "
    r"„Nowa postać” i kliknij „Trenuj postać”\.")
_PREVIEW_NOTE = re.compile(
    r"Podgląd (\d+×\d+) — bez dorysowania do 1080p \(SeedVR2\)\. "
    r"Do publikacji zrób film w trybie Jakość albo Szybki\.")


def short(text: str, n: int = 70) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def label_text(label: str) -> str:
    """Display known legacy system labels in English without changing prompt text."""
    label = label or ""
    if label.startswith("🤖"):
        label = "Autopilot · " + _LEAD.sub("", label)
    autopilot = re.match(r"^(Autopilot\s*·\s*\d{4}-\d{2}-\d{2}\s*·\s*)(.*)$", label,
                         flags=re.S)
    if autopilot:
        return autopilot[1] + label_text(autopilot[2])
    if label.startswith("Zestaw zdjęć postaci:"):
        return "Character photo set:" + label[len("Zestaw zdjęć postaci:"):]
    match = _LEGACY_LABEL.match(label)
    label_only = False
    if not match:
        match = _LEGACY_LABEL.fullmatch(label + ":")
        label_only = bool(match)
    if not match:
        return label.removeprefix("👤 ")
    spec = re.sub(r"^(Kwadrat|Poziomo|Pionowo|Portret)\s*",
                  lambda m: _FORMAT_NAMES[m[1]] + " ", match["spec"])
    spec = re.sub(r"(\d)\s*s$", r"\1 s", spec)
    tags = re.sub(r"\s*·\s*(podgląd|szybki)(?=\s*(?:·|$))",
                  lambda m: " · " + {"podgląd": "Preview", "szybki": "Fast"}[m[1]],
                  match["tags"]).replace("👤 ", "")
    prefix = (("AI Series · " if match["series"] else "")
              + {"obraz": "Image", "wideo": "Video"}[match["kind"]] + " " + spec + tags
              + ("" if label_only else ":"))
    return prefix + ("" if label_only else label[match.end():])


def note_text(text: str) -> str:
    """Translate only recognized persisted system notes; leave other notes intact."""
    match = _SET_NOTE.fullmatch(text)
    if match:
        return (f"Set: {match[1]} photos. Review them — if the character looks right, upload "
                "the ZIP under “New character” and click “Train character”.")
    match = _PREVIEW_NOTE.fullmatch(text)
    if match:
        return (f"Preview {match[1]} — without 1080p upscaling (SeedVR2). For publishing, "
                "generate a video in Quality or Fast mode.")
    return text


def added_message(job: qm.Job) -> str:
    ahead = qm.position(job.id)
    where = "starting now" if ahead == 0 else f"{ahead} jobs ahead"
    return note(f"➕ Queued as **#{job.id}** ({where}). You can add more jobs immediately. "
                "The result will appear here; track progress in **GPU Queue**.")


def finished_message(job: qm.Job) -> str:
    if job.status == qm.DONE:
        head = f"✅ #{job.id} complete ({qm.fmt_time(job.elapsed())})."
    elif job.status == qm.STOPPED:
        head = f"⏹ #{job.id} stopped."
    else:
        head = f"❌ #{job.id}: {job.error}"
    return note(head + ("\n\n" + "\n".join(f"- {note_text(n)}" for n in job.notes)
                        if job.notes else ""))


def result_poller(kind: str):
    """Funkcja dla gr.Timer: najnowszy zakonczony wynik danego rodzaju (tylko gdy nowy)."""
    def tick(last_id):
        job = qm.latest_finished(kind)
        if job is None or job.id == last_id:
            return gr.update(), gr.update(), last_id
        files = [f for f in job.files if Path(f).exists() and studio.is_approved(f)]
        if job.status == qm.DONE and job.files and not files:
            return gr.update(), note("⚠ This output is hidden until its SFW review."), job.id
        if job.status != qm.DONE or not files:  # blad / przerwane - poprzedni wynik zostaje
            return gr.update(), finished_message(job), job.id
        return (files if kind == "obraz" else files[0]), finished_message(job), job.id
    return tick


def _current_html() -> str:
    run = next((j for j in qm.snapshot() if j.status == qm.RUNNING), None)
    if run is None:
        text = "Queue paused — no generation is running." if qm.paused() else \
            "No generation is running."
        return f"<div class='job idle'>{text}</div>"
    pct = int(run.progress * 100)
    return (f"<div class='job'><div class='j-title'>{badge(KINDS.get(run.kind, run.kind), 'info')} "
            f"#{run.id} · {esc(short(label_text(run.label), 90))}</div>{bar(run.progress)}"
            f"<div class='j-meta'>{pct}% · {esc(run.desc or 'start…')} · elapsed "
            f"{qm.fmt_time(run.elapsed())} · about {qm.fmt_time(run.eta())} remaining</div></div>")


def _pending_choices():
    jobs = [j for j in studio.jobs(studio.current()) if j.status in (qm.RUNNING, qm.PENDING)]
    # nazwa zadania mowi juz, co to jest ("wideo 6 s · SFW: ...") - bez dopisku rodzaju
    return [(f"#{j.id} · {short(label_text(j.label), 90)}"
             + (" — running (removing will stop it)" if j.status == qm.RUNNING else ""), str(j.id))
            for j in jobs]


def _summary() -> str:
    snap = qm.snapshot()
    pending = [j for j in snap if j.status == qm.PENDING]
    paused = " · **paused**" if qm.paused() else ""
    profile = studio.current()
    if profile:
        mine = [j for j in studio.jobs(profile, snap) if j.status in (qm.PENDING, qm.RUNNING)]
        eta = sum((j.eta() or 0.0) if j.status == qm.RUNNING else j.estimate() for j in mine)
        return (f"Character **{studio.name(profile)}** — **{len(mine)}** jobs, about "
                f"**{qm.fmt_time(eta)}** · entire queue: {len(pending)}, about "
                f"{qm.fmt_time(qm.remaining_seconds())}{paused}")
    if not pending and not any(j.status == qm.RUNNING for j in snap):
        return "The queue is empty. Each click of Generate adds a job here."
    return (f"Queued: **{len(pending)}** · estimated total time remaining "
            f"**{qm.fmt_time(qm.remaining_seconds())}**{paused}")


def _finished_html() -> str:
    done = [j for j in studio.jobs(studio.current())
            if j.status in (qm.DONE, qm.ERROR, qm.STOPPED)]
    rows = []
    for j in sorted(done, key=lambda j: -(j.finished or 0))[:30]:
        text, kind = STATUS[j.status]
        state = badge(text, kind) + (f" <span class='muted'>{esc(short(j.error, 70))}</span>"
                                     if j.status == qm.ERROR else "")
        file = esc(Path(j.files[0]).name) if j.files else "—"
        rows.append([f"<span class='muted num'>{j.id}</span>",
                     esc(short(label_text(j.label), 80)),
                     state, f"<span class='num'>{qm.fmt_time(j.elapsed())}</span>",
                     f"<span class='muted'>{file}</span>"])
    return table(["#", "Job", "Status", "Time", "File"], rows,
                 empty="Completed jobs will appear here.")


def _pause_label() -> str:
    return "Resume queue" if qm.paused() else "Pause after current job"


def build_queue_tab(tab: gr.Tab) -> None:
    """tab - strona Kolejka (odswiezenie przy kazdym wejsciu)."""
    page_head("GPU Queue", "Jobs run one at a time on one GPU. Manual jobs take priority over "
                            "autopilot jobs.")
    current = gr.HTML(_current_html(), padding=False)
    with gr.Row(elem_classes=["toolbar"]):
        stop = gr.Button("Stop current job", variant="stop", size="sm")
        pause = gr.Button(_pause_label(), size="sm")
        summary = gr.Markdown(_summary(), elem_classes=["hint", "grow"])
    with gr.Column(elem_classes=["card"]):
        section("Pending jobs", "Select jobs to remove them.")
        pending = gr.CheckboxGroup(_pending_choices(), show_label=False, elem_id="queue-list")
        with gr.Row(elem_classes=["toolbar"]):
            delete = gr.Button("Remove selected", size="sm")
            select_all = gr.Button("Select all", size="sm")
            clear = gr.Button("Clear pending jobs", size="sm", variant="stop")
    with gr.Column(elem_classes=["card"]):
        with gr.Row(elem_classes=["toolbar"]):
            with gr.Column(elem_classes=["grow"]):
                section("Finished jobs", "The latest 30 jobs.")
            clear_done = gr.Button("Clear finished jobs", size="sm")
        done = gr.HTML(_finished_html())
    shown = gr.State([label for label, _ in _pending_choices()])

    def refresh(selected: Optional[List[str]], shown_labels: Optional[List[str]]):
        """shown_labels=None wymusza przebudowe listy (po usunieciu / wyczyszczeniu)."""
        choices = _pending_choices()
        labels = [label for label, _ in choices]
        if labels == shown_labels:  # bez zmian - nie ruszamy listy (zaznaczenia zostaja)
            pend = gr.update()
        else:
            ids = [v for _, v in choices]
            pend = gr.update(choices=choices, value=[v for v in (selected or []) if v in ids])
        run = next((j for j in qm.snapshot() if j.status == qm.RUNNING), None)
        return (_current_html(), _summary(), pend, _finished_html(), _pause_label(), labels,
                run.id if run else None)

    running = gr.State(None)  # zadanie widoczne na karcie - tylko je zatrzymuje "Przerwij"
    outs = [current, summary, pending, done, pause, shown, running]
    gr.Timer(2.0).tick(refresh, inputs=[pending, shown], outputs=outs)
    # Timer rusza dopiero z pierwszym otwarciem strony - bez tego przez 2 s widac stan ze startu
    tab.select(refresh, inputs=[pending, shown], outputs=outs)

    def _stop(selected, shown_ids, running_id):
        # tylko zadanie, ktore uzytkownik widzial: drugie klikniecie (gdy wideo jeszcze sie
        # zatrzymywalo) przerywalo nastepne zadanie, ktore wlasnie ruszylo
        if running_id is not None:
            qm.stop_current(running_id)
        return refresh(selected, shown_ids)

    def _delete(selected, shown_ids):
        qm.remove([int(v) for v in selected or []])
        return refresh([], None)

    def _clear(selected, shown_ids):
        # tylko to, co widac - przy wybranej modelce jej zadania, inne zostaja
        qm.remove([j.id for j in studio.jobs(studio.current()) if j.status == qm.PENDING])
        return refresh([], None)

    def _toggle(selected, shown_ids):
        qm.set_paused(not qm.paused())
        return refresh(selected, shown_ids)

    def _clear_done(selected, shown_ids):
        qm.clear_finished()
        return refresh(selected, shown_ids)

    stop.click(_stop, inputs=[pending, shown, running], outputs=outs)
    for btn, fn in ((delete, _delete), (clear, _clear), (pause, _toggle),
                    (clear_done, _clear_done)):
        btn.click(fn, inputs=[pending, shown], outputs=outs)
    select_all.click(lambda: [v for _, v in _pending_choices()], outputs=pending)
