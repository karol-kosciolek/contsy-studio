# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Interfejs: menu w panelu bocznym ma strone dla kazdej pozycji, komunikaty i nazwy zadan
bez emoji (spokojny wyglad), cala aplikacja buduje sie bez serwera."""
import tempfile
import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
from fastapi import FastAPI
from starlette.testclient import TestClient

from app import characters, main, prefs, qc, studio, ui_history, ui_images, ui_queue, ui_theme, ui_video
from app.vram import GpuInfo


class NoteTests(unittest.TestCase):
    def test_leading_marks_become_colored_dots(self):
        self.assertIn("st-ok", ui_theme.note("✅ Zapisano."))
        self.assertIn("st-err", ui_theme.note("❌ Wpisz prompt."))
        self.assertIn("st-err", ui_theme.note("⛔ Zablokowane."))
        self.assertIn("st-warn", ui_theme.note("⚠ Kontrola jakości: 3 próby."))
        self.assertTrue(ui_theme.note("❌ Wpisz prompt.").endswith("Wpisz prompt."))
        self.assertEqual(ui_theme.note("Zwykły tekst"), "Zwykły tekst")
        self.assertEqual(ui_theme.note(""), "")
        self.assertEqual(ui_theme.note(ui_theme.note("✅ x")), ui_theme.note("✅ x"))  # raz

    def test_queue_labels_drop_old_emoji(self):
        self.assertEqual(ui_queue.label_text("🤖 2026-11-01 · wideo 6 s · SFW"),
                         "Autopilot · 2026-11-01 · Video 6 s · SFW")
        self.assertEqual(ui_queue.label_text("obraz 1:1 · SFW · 👤 Mia: plaża"),
                         "Image 1:1 · SFW · Mia: plaża")

    def test_legacy_queue_labels_translate_only_system_prefix(self):
        suffix = "  polski obraz: podgląd · 👤 Ola\n日本語  "
        for old, expected in [
                ("obraz Pionowo 9:16", "Image Portrait 9:16"),
                ("obraz Poziomo 16:9", "Image Landscape 16:9"),
                ("obraz Kwadrat 1:1", "Image Square 1:1"),
                ("obraz Portret 3:4", "Image Portrait 3:4"),
                ("wideo 5 s · podgląd", "Video 5 s · Preview"),
                ("wideo 5 s · szybki", "Video 5 s · Fast"),
                ("plan · obrazPionowo9:16", "AI Series · Image Portrait 9:16"),
                ("plan · wideo5s · podgląd", "AI Series · Video 5 s · Preview"),
                ("Zestaw zdjęć postaci", "Character photo set")]:
            with self.subTest(prefix=old):
                rendered = ui_queue.label_text(old + ":" + suffix)
                self.assertEqual(rendered, expected + ":" + suffix)
                self.assertEqual(ui_queue.label_text(rendered), rendered)
        for custom in ("Własna nazwa: obraz Pionowo 9:16", "Image 1:1: 👤 prompt", "opis"):
            self.assertEqual(ui_queue.label_text(custom), custom)

    def test_legacy_queue_notes_are_display_only_and_conservative(self):
        notes = [
            "Zestaw: 20 zdjęć. Obejrzyj je - jeśli postać pasuje, wgraj plik ZIP w "
            "„Nowa postać” i kliknij „Trenuj postać”.",
            "Podgląd 480×848 — bez dorysowania do 1080p (SeedVR2). Do publikacji "
            "zrób film w trybie Jakość albo Szybki."]
        job = ui_queue.qm.Job(6, "zestaw", "Zestaw zdjęć postaci: Ola (20)", {},
                              status=ui_queue.qm.DONE, notes=list(notes))
        rendered = ui_queue.finished_message(job)
        self.assertIn("Set: 20 photos.", rendered)
        self.assertIn("Preview 480×848", rendered)
        self.assertIn("Quality or Fast mode", rendered)
        self.assertEqual(job.notes, notes)
        self.assertEqual(job.label, "Zestaw zdjęć postaci: Ola (20)")
        for custom in ("Zestaw: mój własny tekst", "Podgląd 480×848: komentarz", "用户注释"):
            self.assertEqual(ui_queue.note_text(custom), custom)

    def test_queue_views_render_legacy_labels_without_mutation(self):
        labels = ["obraz Pionowo 9:16: scena", "wideo 5 s · podgląd: scena",
                  "plan · obraz Pionowo 9:16: scena", "Zestaw zdjęć postaci: Ola (20)"]
        jobs = [ui_queue.qm.Job(i, "obraz", label, {}, status=ui_queue.qm.DONE)
                for i, label in enumerate(labels, 1)]
        with patch.object(ui_queue.studio, "jobs", return_value=jobs):
            rendered = ui_queue._finished_html()
        for expected in ("Image Portrait 9:16: scena", "Video 5 s · Preview: scena",
                         "AI Series · Image Portrait 9:16: scena", "Character photo set: Ola (20)"):
            self.assertIn(expected, rendered)
        jobs[0].status = ui_queue.qm.RUNNING
        with patch.object(ui_queue.qm, "snapshot", return_value=jobs), \
                patch.object(ui_queue.studio, "jobs", return_value=jobs):
            self.assertIn("Image Portrait 9:16: scena", ui_queue._current_html())
            self.assertIn("Image Portrait 9:16: scena", ui_queue._pending_choices()[0][0])
        self.assertEqual([job.label for job in jobs], labels)

    def test_model_picker_and_job_names_have_no_emoji(self):
        with patch.object(characters, "all_characters", return_value=[
                {"name": "Mia", "slug": "mia"}]), \
                patch.object(characters, "usable", return_value={"name": "Mia", "slug": "mia"}):
            self.assertEqual(characters.choices(), [("All characters (no LoRA)", ""),
                                                    ("Mia", "mia")])
        with patch.object(characters, "selected", return_value={"name": "Mia", "slug": "mia"}):
            params = {}
            self.assertEqual(characters.tag(params), " · Mia")
            self.assertEqual(params["character"], "mia")


class ShellTests(unittest.TestCase):
    def test_output_media_http_guard_checks_current_approval(self):
        app = FastAPI()

        @app.get("/gradio_api/file/{path:path}")
        @app.get("/gradio_api/file={path:path}")
        def file(path: str):
            return {"ok": True}

        app.add_middleware(main.LocalRequestGuard)
        with TestClient(app, base_url="http://127.0.0.1:7881") as client, \
                patch.object(main.studio, "is_approved", return_value=False) as approved:
            for marker in ("/gradio_api/file=", "/gradio_api/file/"):
                url = marker + (main.config.OUTPUTS_DIR / "new_unverified.png").as_posix()
                approved.return_value = False
                self.assertEqual(client.get(url).status_code, 403)
                approved.return_value = True
                self.assertEqual(client.get(url).status_code, 200)

    def test_foreign_origins_and_hosts_cannot_execute_ui_actions(self):
        app = FastAPI()
        calls = []

        @app.post("/action")
        def action():
            calls.append(True)
            return {"ok": True}

        app.add_middleware(main.LocalRequestGuard)
        with TestClient(app, base_url="http://127.0.0.1:7881") as client:
            for headers in ({"Origin": "https://example.com"}, {"Origin": "null"},
                            {"Host": "example.com", "Origin": "http://example.com"},
                            {"Sec-Fetch-Site": "cross-site"}):
                self.assertEqual(client.post("/action", headers=headers).status_code, 403)
            self.assertEqual(calls, [])
            self.assertEqual(client.post("/action", headers={"Origin": "http://127.0.0.1:7881"})
                             .status_code, 200)
            self.assertEqual(client.post("/action").status_code, 200)
            self.assertEqual(len(calls), 2)

    def test_launch_is_local_and_excludes_private_data_and_rejected_outputs(self):
        for host in ("127.0.0.1", "::1", "localhost"):
            options = main.local_launch_options({"host": host})
            self.assertEqual(options["server_name"], host)
            self.assertFalse(options["share"])
            self.assertEqual(options["allowed_paths"], [str(main.config.OUTPUTS_DIR.resolve())])
            self.assertIn(str(main.config.CONFIG_DIR.resolve()), options["blocked_paths"])
            self.assertIn(str(qc.REJECTED_DIR.resolve()), options["blocked_paths"])
            self.assertTrue(set(options).issubset(inspect.signature(gr.Blocks.launch).parameters))
        for host in ("0.0.0.0", "::", "192.168.1.10", "example.com"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                main.local_launch_options({"host": host})

    def test_existing_unverified_media_is_blocked_without_changing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            approved = Path(tmp) / "approved.png"
            legacy = Path(tmp) / "legacy.png"
            approved.write_bytes(b"approved")
            legacy.write_bytes(b"preserve")
            with patch.object(main.config, "OUTPUTS_DIR", Path(tmp)), \
                    patch.object(main.studio, "approved_files", return_value={str(approved.resolve()).casefold()}):
                blocked = main.local_launch_options({"host": "127.0.0.1"})["blocked_paths"]
            self.assertIn(str(legacy.resolve()), blocked)
            self.assertNotIn(str(approved.resolve()), blocked)
            self.assertEqual(legacy.read_bytes(), b"preserve")

    def test_every_menu_item_has_a_page_and_menu_css(self):
        css = ui_theme.nav_css(main.NAV)
        for label, page, group in main.NAV:
            self.assertIn(f'data-testid="{label}-radio-label"', css)
            self.assertIn(page, ui_theme.ICONS)
            if group and group != "-":
                self.assertIn(f'content: "{group}"', css)
        self.assertEqual(len({p for _, p, _ in main.NAV}), len(main.NAV))

    def test_whole_app_builds_and_tabs_match_menu(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"), \
                patch.object(characters, "all_characters", return_value=[]), \
                patch.object(studio, "results", return_value=[]), \
                patch.object(main, "get_gpu_info",
                             return_value=GpuInfo(True, "NVIDIA GeForce RTX 4070", total_gb=16.0,
                                                  free_gb=9.5)):
            demo = main.build_ui()
        tabs = {b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)}
        self.assertEqual(tabs, {page for _, page, _ in main.NAV})
        nav = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "nav")
        self.assertEqual([v for _, v in nav.choices], [page for _, page, _ in main.NAV])
        self.assertTrue(any(isinstance(b, gr.Sidebar) for b in demo.blocks.values()))
        self.assertEqual(demo.title, "Contsy Studio")  # karta przegladarki
        self.assertIn(">Contsy Studio<", ui_theme.BRAND)  # logo w panelu bocznym
        self.assertEqual([label for label, _, _ in main.NAV], [
            "Dashboard", "Video", "Image", "AI Series", "Persona & Autopilot", "Content Packs",
            "Training", "Gallery", "GPU Queue", "Settings"])
        text = str(demo.get_config_file())
        for unwanted in ("Fanvue", "Wirtualne modelki", "Zaznaczanie wielu", "Trening modelek",
                         "Seria z polecenia", "Polecenie dla AI", "Przerwij bieżące"):
            self.assertNotIn(unwanted, text)


class GallerySafetyTests(unittest.TestCase):
    def test_recycle_failure_preserves_file_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"keep me")
            with patch.object(ui_history.config, "OUTPUTS_DIR", Path(tmp)), \
                    patch.object(ui_history.studio, "is_approved", return_value=True), \
                    patch.object(ui_history.trash, "move", return_value=False), \
                    patch.object(ui_history.history, "forget") as forget:
                self.assertEqual(ui_history.recycle_selected([str(path)]), ([], [str(path)]))
                self.assertEqual(path.read_bytes(), b"keep me")
                forget.assert_not_called()

    def test_recycle_partial_failure_forgets_only_successful_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("ok.png", "keep.png")]
            for path in paths:
                path.touch()
            with patch.object(ui_history.config, "OUTPUTS_DIR", Path(tmp)), \
                    patch.object(ui_history.studio, "is_approved", return_value=True), \
                    patch.object(ui_history.trash, "move", side_effect=[True, False]), \
                    patch.object(ui_history.history, "forget") as forget:
                self.assertEqual(ui_history.recycle_selected([str(p) for p in paths]),
                                 ([str(paths[0])], [str(paths[1])]))
                forget.assert_called_once_with([str(paths[0])])

    def test_gallery_rejects_outside_files_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            outside = Path(tmp) / "private.txt"
            outside.touch()
            self.assertEqual(ui_history._safe([str(root), str(outside)]), [])

    def test_gallery_rejects_unverified_output_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unverified.png"
            path.touch()
            with patch.object(ui_history.config, "OUTPUTS_DIR", Path(tmp)), \
                    patch.object(ui_history.studio, "is_approved", return_value=False):
                self.assertEqual(ui_history._safe([str(path)]), [])


class GenerationSafetyTests(unittest.TestCase):
    def test_queue_poller_hides_unverified_completed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.png"
            path.touch()
            job = ui_queue.qm.Job(123, "obraz", "Legacy image", {}, status=ui_queue.qm.DONE,
                                  files=[str(path)])
            with patch.object(ui_queue.qm, "latest_finished", return_value=job), \
                    patch.object(ui_queue.studio, "is_approved", return_value=False):
                result, message, last_id = ui_queue.result_poller("obraz")(None)
            self.assertEqual(result, gr.update())
            self.assertIn("hidden until its SFW review", message)
            self.assertEqual(last_id, 123)

    def test_legacy_image_formats_keep_their_dimensions(self):
        for legacy, expected in [("Kwadrat 1:1", (1080, 1080)),
                                 ("Poziomo 16:9", (1920, 1080)),
                                 ("Pionowo 9:16", (1080, 1920)),
                                 ("Portret 3:4", (1080, 1440))]:
            self.assertEqual(ui_images.image_size(legacy), expected)

    def test_unverified_start_image_never_starts_video_generation(self):
        with patch.object(ui_video.content, "check", return_value=None), \
                patch.object(ui_video.qc, "check", return_value=qc.Verdict(False, [qc.UNCHECKED])), \
                patch.object(ui_video.video_gen, "generate") as generate:
            ok, files, _, error = ui_video.run_job({"prompt": "an adult walking", "image": "start.png"},
                                                  lambda *_: None)
        self.assertFalse(ok)
        self.assertEqual(files, [])
        self.assertIn("could not verify", error)
        generate.assert_not_called()

    def test_failed_engine_video_is_not_autosaved(self):
        out = SimpleNamespace(ok=False, files=[], notes=[], error="Output failed the safety check.")
        with patch.object(ui_video.content, "check", return_value=None), \
                patch.object(ui_video.characters, "for_job", return_value=(None, "")), \
                patch.object(ui_video.video_gen, "generate", return_value=out), \
                patch.object(ui_video.prefs, "autosave") as autosave:
            ok, files, _, error = ui_video.run_job({"prompt": "an adult walking", "seconds": 5,
                                                   "aspect": "Portrait 9:16"}, lambda *_: None)
        self.assertFalse(ok)
        self.assertEqual(files, [])
        self.assertIn("safety check", error)
        autosave.assert_not_called()


if __name__ == "__main__":
    unittest.main()
