# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'Galeria' - same wyniki (zdjecia i filmy) z outputs/.

Filtr: Wszystko / Zdjecia / Filmy. Klik w miniature = podglad i wybor pliku; "Zaznaczanie
wielu" - kolejne kliki zaznaczaja (ramka i znacznik na miniaturze, pasek "Selected" pod
galeria). Pobierz daje plik albo ZIP, Usun przenosi zaznaczone do Kosza Windows
(z potwierdzeniem i liczba plikow). Usuwamy tylko pliki z outputs/ (kopie w folderze
autozapisu zostaja).
"""
from __future__ import annotations

import tempfile
import time
import zipfile
from pathlib import Path
from typing import List

import gradio as gr

from . import config, history, packs, studio, trash, video_gen
from .ui_theme import LEGACY_NOTICE, esc, head_html, note

LIMIT = 150
FILTERS = ["All", "Images", "Videos"]
_IMAGES = (".png", ".jpg", ".jpeg", ".webp")
_VIDEOS = (".mp4",)


def _list(exts) -> List[str]:
    """Najnowsze wyniki; przy wybranej modelce (panel boczny) - tylko jej (app/studio.py)."""
    return studio.results(studio.current(), exts, LIMIT)


def _head() -> str:
    profile = studio.current()
    sub = (f"Results for <b>{esc(studio.name(profile))}</b>. Select “All characters” in the "
           "sidebar to see everything." if profile else f"All results, showing the latest {LIMIT}.")
    return head_html("Gallery", sub + ("<br>" + LEGACY_NOTICE if studio.has_legacy_results() else ""))


def _media(filt: str) -> List[str]:
    return _list({"Images": _IMAGES, "Videos": _VIDEOS}.get(filt, _IMAGES + _VIDEOS))


def _when(f: str) -> str:
    try:
        return time.strftime("%d.%m %H:%M", time.localtime(Path(f).stat().st_mtime))
    except OSError:
        return ""


def _items(files: List[str]):
    return [(f, _when(f)) for f in files]  # film poznac po znaczku na miniaturze (CSS)


def _picked(selected: List[str]):
    """Pasek "Selected" pod galeria."""
    return gr.update(value=list(selected or []), visible=bool(selected),
                     label=f"Selected: {len(selected or [])}")


def _safe(paths: List[str]) -> List[str]:
    """Tylko istniejace pliki bezposrednio w outputs/ - nic spoza folderu wynikow."""
    root = config.OUTPUTS_DIR.resolve()
    return list(dict.fromkeys(p for p in paths or []
                              if Path(p).is_file() and Path(p).resolve().parent == root
                              and studio.is_approved(p)))


def recycle_selected(paths: List[str]) -> tuple[list[str], list[str]]:
    """Forget only files successfully moved to the Recycle Bin; retain failures."""
    moved, failed = [], []
    for path in _safe(paths):
        try:
            recycled = trash.move(path)
        except OSError:
            recycled = False
        (moved if recycled else failed).append(path)
    if moved:
        history.forget(moved)
    return moved, failed


def _info(selected: List[str], multi: bool) -> str:
    if not selected:
        return ("Click a thumbnail to preview and select it. Enable **Select multiple** to choose "
                "several files at once.")
    if len(selected) == 1 and not multi:
        msg = f"Selected file: **{Path(selected[0]).name}**"
        entry = history.entry_for(selected[0]) or {}
        if entry.get("speed") in (video_gen.FAST, video_gen.PREVIEW):
            msg += f" · Mode: **{video_gen.MODE_LABELS[entry['speed']]}**" + (
                " (480p, no upscaling — excluded from packs)"
                if entry["speed"] == video_gen.PREVIEW else "")
        if entry.get("idea"):
            idea = entry["idea"] if len(entry["idea"]) <= 300 else entry["idea"][:300] + "…"
            msg += f"\n\nYour instructions: “{idea}”"
        return msg
    return f"Selected: **{len(selected)}** — Download as ZIP, create a pack or delete."


def build_history_tab(tab: gr.Tab) -> None:
    """tab - strona Galeria (odswiezenie przy kazdym wejsciu)."""
    files0 = _media("All")
    head = gr.HTML(_head(), padding=False)
    with gr.Column(elem_classes=["card"]):
        with gr.Row(elem_classes=["toolbar"]):
            filt = gr.Radio(FILTERS, value="All", show_label=False, container=False,
                            elem_id="hist-filter")
            multi = gr.Checkbox(label="Select multiple", value=False, container=False,
                                elem_id="hist-multi", elem_classes=["switch"])
            gr.HTML("", padding=False, elem_classes=["grow"])
            all_btn = gr.Button("Select all", size="sm", elem_id="hist-all")
            none_btn = gr.Button("Clear selection", size="sm", elem_id="hist-none")
            refresh_btn = gr.Button("Refresh", size="sm", elem_id="hist-refresh")
        info = gr.Markdown(_info([], False), elem_classes=["status-box"])
        with gr.Row(elem_classes=["toolbar"]):
            dl_btn = gr.Button("Download", size="sm", variant="primary")
            pack_btn = gr.Button("Pack from selection", size="sm")
            rm_btn = gr.Button("Delete", size="sm", variant="stop", elem_id="hist-del")
        gallery = gr.Gallery(value=_items(files0), show_label=False, columns=6, height=680,
                             object_fit="cover", allow_preview=True, elem_id="hist-gallery",
                             buttons=["download", "fullscreen"])
        picked = gr.Gallery(visible=False, columns=12, height=110, object_fit="cover",
                            allow_preview=False, label="Selected", buttons=[])
        out_file = gr.File(label="Download", visible=False, interactive=False)
    files_st, sel_st = gr.State(files0), gr.State([])

    def refresh(filt_v, multi_v, selected):
        files = _media(filt_v)
        selected = [f for f in selected or [] if f in files]
        return (gr.update(value=_items(files), allow_preview=not multi_v), files, selected,
                _info(selected, multi_v), _picked(selected), _head())

    outs = [gallery, files_st, sel_st, info, picked, head]
    tab.select(refresh, inputs=[filt, multi, sel_st], outputs=outs)
    refresh_btn.click(refresh, inputs=[filt, multi, sel_st], outputs=outs)
    filt.change(lambda f, m: refresh(f, m, []), inputs=[filt, multi], outputs=outs)

    def on_select(multi_v, files, selected, evt: gr.SelectData):
        if not (0 <= evt.index < len(files)):
            return gr.skip(), selected, gr.skip()
        f = files[evt.index]
        if not multi_v:  # pojedynczy wybor - podglad zostaje otwarty
            return _picked([]), [f], _info([f], False)
        selected = [x for x in selected if x != f] if f in selected else selected + [f]
        return _picked(selected), selected, _info(selected, True)

    gallery.select(on_select, inputs=[multi, files_st, sel_st], outputs=[picked, sel_st, info])

    multi.change(lambda m: (gr.update(allow_preview=not m), [], _info([], m), _picked([])),
                 inputs=multi, outputs=[gallery, sel_st, info, picked])
    all_btn.click(lambda files: (True, gr.update(allow_preview=False), list(files),
                                 _info(files, True), _picked(files)),
                  inputs=files_st, outputs=[multi, gallery, sel_st, info, picked])
    none_btn.click(lambda m: ([], _info([], m), _picked([])), inputs=multi,
                   outputs=[sel_st, info, picked])

    def download(selected):
        paths = _safe(selected)
        if not paths:
            return gr.update(visible=False), note("❌ Select a file first.")
        if len(paths) == 1:
            return gr.update(value=paths[0], visible=True), note(
                f"✅ Ready: **{Path(paths[0]).name}**")
        zip_path = Path(tempfile.gettempdir()) / f"contsy_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for p in paths:
                z.write(p, Path(p).name)
        return gr.update(value=str(zip_path), visible=True), note(
            f"✅ Ready: ZIP containing {len(paths)} files.")

    dl_btn.click(download, inputs=sel_st, outputs=[out_file, info])

    def pack(selected, progress=gr.Progress()):
        """Paczka do publikacji z zaznaczonych (app/packs.py): modelka z wybranego profilu,
        a bez profilu - ta, do ktorej naleza wszystkie zaznaczone pliki."""
        paths = _safe(selected)
        if not paths:
            return note("❌ Select images or videos first.")
        owners = {studio.file_owners().get(Path(p).name, "") for p in paths}
        slug = studio.current() or (owners.pop() if len(owners) == 1 else "")
        progress(0.1, desc="Checking files, building the pack and writing captions...")
        ok, msg = packs.build_selection(paths, slug)
        return note(msg if ok else f"❌ {msg}")

    pack_btn.click(pack, inputs=sel_st, outputs=info)

    def remove(confirmed, filt_v, multi_v, selected):
        paths = _safe(selected)
        if confirmed != "yes" or not paths:
            return (*[gr.skip()] * 3, "Nothing deleted." if paths else
                    note("❌ Select a file first."), gr.skip(), gr.skip(),
                    gr.update(visible=False))
        moved, failed = recycle_selected(paths)
        res = list(refresh(filt_v, multi_v, failed))
        res[3] = note(f"🗑️ Moved to Recycle Bin: {len(moved)}. You can restore these files.")
        if failed:
            res[3] += note(f"\n\n⚠ Could not move {len(failed)} file(s) to the Recycle Bin. "
                           "They remain selected and have not been deleted.")
        return (*res, gr.update(visible=False))

    # potwierdzenie w przegladarce -> ukryte pole, potem usuwanie (stan wyboru zostaje na
    # serwerze - nie przechodzi przez JavaScript)
    confirm = gr.Textbox(visible=False)
    rm_btn.click(None, outputs=confirm,
                 js="() => { const t = [...document.querySelectorAll('.status-box')]"
                    ".map(e => e.innerText).find(s => /Selected/.test(s)) || ''; "
                    "return confirm('Delete? ' + t.replace(/ —.*/, '').replace(/ · .*/, '') + "
                    "'\\nFiles will be moved to the Recycle Bin.') ? 'yes' : 'no'; }").then(
        remove, inputs=[confirm, filt, multi, sel_st], outputs=outs + [out_file])
