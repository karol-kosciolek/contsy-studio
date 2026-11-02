# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pauza karty (przycisk na gornym pasku) i profile modelek (widoki Kolejki, Historii, Studia)."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import cancel, history, llm, studio
from app import queue_manager as qm


class _IsolatedQueue(unittest.TestCase):
    """Kolejka na plikach tymczasowych - prawdziwa config/queue.json zostaje nietknieta."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.files_dir = root / "queue_files"
        self.patches = [
            patch.object(qm, "_STATE", root / "queue.json"),
            patch.object(qm, "FILES_DIR", self.files_dir),
            patch.object(qm, "_JOBS", []),
            patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}),
            patch.object(qm, "_free_gpu", lambda *a, **k: None),  # bez silnika ComfyUI
            patch.object(cancel, "request", cancel._FLAG.set),
        ]
        for p in self.patches:
            p.start()
        cancel.reset()
        qm._REQUEUE.clear()
        self.ref = root / "ref.png"
        self.ref.write_bytes(b"png")

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        cancel.reset()
        qm._REQUEUE.clear()
        self.tmp.cleanup()

    def start_job(self, runner):
        qm.register("test", runner)
        job = qm.add("test", "test", {}, files={"refs": [str(self.ref)]})
        job.status, job.started = qm.RUNNING, time.time()
        worker = threading.Thread(target=qm._run, args=(job,))
        worker.start()
        return job, worker


class PauseTests(_IsolatedQueue):
    def test_pause_puts_the_running_job_back_with_its_files(self):
        started = threading.Event()

        def runner(params, cb):
            started.set()
            while True:  # liczy, az pauza przerwie (cb -> cancel.check)
                cb(0.5, "liczy")
                time.sleep(0.01)

        job, worker = self.start_job(runner)
        self.assertTrue(started.wait(2))
        self.assertEqual(qm.pause_now(), job.id)
        worker.join(5)
        self.assertEqual((job.status, job.progress, job.started), (qm.PENDING, 0.0, None))
        self.assertTrue((self.files_dir / str(job.id)).exists())  # zdjecia do ponownego liczenia
        self.assertTrue(qm.paused())
        self.assertIsNone(qm._next())  # pauza - nic nie rusza
        qm.set_paused(False)
        self.assertIs(qm._next(), job)  # po wznowieniu wraca na swoje miejsce

    def test_job_finished_at_the_moment_of_pause_stays_done(self):
        started, go = threading.Event(), threading.Event()

        def runner(params, cb):
            started.set()
            go.wait(2)
            return True, ["wynik.png"], [], ""

        job, worker = self.start_job(runner)
        self.assertTrue(started.wait(2))
        qm.pause_now()
        go.set()
        worker.join(5)
        self.assertEqual(job.status, qm.DONE)

    def test_stop_button_still_stops_for_good(self):
        started = threading.Event()

        def runner(params, cb):
            started.set()
            while True:
                cb(0.5, "liczy")
                time.sleep(0.01)

        job, worker = self.start_job(runner)
        self.assertTrue(started.wait(2))
        self.assertTrue(qm.stop_current(job.id))
        worker.join(5)
        self.assertEqual(job.status, qm.STOPPED)
        self.assertFalse((self.files_dir / str(job.id)).exists())

    def test_prompt_helper_stays_off_the_gpu_while_paused(self):
        calls = []
        with patch.object(qm, "paused", return_value=True), \
                patch.object(llm, "_generate",
                             side_effect=lambda s, u, gpu, *a: calls.append(gpu) or (True, "ok")):
            ok, _ = llm.ask("system", "user")
        self.assertTrue(ok)
        self.assertEqual(calls, [False])


CHARS = [{"name": "Virtual Mia", "slug": "mia", "status": "gotowa", "created": 1},
         {"name": "Nova", "slug": "nova", "status": "gotowa", "created": 2}]


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(studio.characters, "all_characters", return_value=CHARS)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_results_belong_to_a_profile_by_slug_name_or_lora(self):
        entries = [{"files": ["C:/o/d.png"]},
                   {"loras": [{"name": "aistudio_char_mia.safetensors"}], "files": ["C:/o/c.png"]},
                   {"character": "Nova", "files": ["C:/o/b.mp4"]},
                   {"profile": "mia", "character": "Virtual Mia", "files": ["C:/o/a.png"]}]
        self.assertEqual(studio.file_owners(entries),
                         {"a.png": "mia", "b.mp4": "nova", "c.png": "mia", "d.png": ""})

    def test_queue_view_shows_only_the_selected_models_jobs(self):
        jobs = [SimpleNamespace(id=1, params={"character": "mia"}),
                SimpleNamespace(id=2, params={"character": "nova"}),
                SimpleNamespace(id=3, params={})]
        self.assertEqual([j.id for j in studio.jobs("mia", jobs)], [1])
        self.assertEqual([j.id for j in studio.jobs("", jobs)], [1, 2, 3])

    def test_history_view_lists_only_the_profiles_files(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(studio.config, "OUTPUTS_DIR", Path(tmp)):
            for name in ("a.png", "b.mp4", "c.png", "x_raw.png"):
                (Path(tmp) / name).write_bytes(b"x")
            owners = {"a.png": "mia", "b.mp4": "nova", "c.png": "mia", "x_raw.png": "mia"}
            entries = [{"files": [str(Path(tmp) / name)], "qc": {"safe": True}} for name in owners]
            with patch.object(studio.history, "recent", return_value=entries):
                mine = studio.results("mia", owners=owners)
                everyone = studio.results("", owners=owners)
        self.assertEqual(sorted(Path(p).name for p in mine), ["a.png", "c.png"])
        self.assertEqual(sorted(Path(p).name for p in everyone), ["a.png", "b.mp4", "c.png"])

    def test_gallery_hides_legacy_and_unsafe_results_without_removing_them(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(studio.config, "OUTPUTS_DIR", Path(tmp)):
            paths = {name: Path(tmp) / name for name in ("safe.png", "legacy.png", "unsafe.png")}
            for path in paths.values():
                path.write_bytes(b"preserved")
            entries = [{"files": [str(paths["safe.png"])], "qc": {"safe": True}},
                       {"files": [str(paths["legacy.png"])], "qc": {"ok": True}},
                       {"files": [str(paths["unsafe.png"])], "qc": {"safe": False}}]
            with patch.object(studio.history, "recent", return_value=entries):
                self.assertEqual(studio.results(), [str(paths["safe.png"])])
            self.assertTrue(all(path.read_bytes() == b"preserved" for path in paths.values()))

    def test_legacy_notice_only_when_history_has_unverified_records(self):
        checked = [{"files": ["a.png"], "qc": {"ok": True, "safe": True}},
                   {"files": ["b.png"], "qc": {"ok": False, "safe": False}}, {"mode": "no files"}]
        self.assertFalse(studio.has_legacy_results(checked))
        self.assertFalse(studio.has_legacy_results([]))
        for legacy in ({"files": ["c.png"], "qc": {"ok": True}}, {"files": ["d.png"]},
                       {"files": ["e.png"], "qc": {"safe": True}, "level": "nsfw"}):
            self.assertTrue(studio.has_legacy_results(checked + [legacy]), legacy)

    def test_overview_counts_queue_and_results_per_model(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(studio.characters, "DIR", Path(tmp)), \
                patch.object(studio.characters, "usable", return_value={"slug": "x"}):
            files = []
            for name in ("a.png", "b.png", "c.png"):
                (Path(tmp) / name).write_bytes(b"x")
                files.append(str(Path(tmp) / name))
            entries = [{"profile": "mia", "ts": now, "files": [files[0]]},
                       {"profile": "mia", "ts": now - 3 * 86400, "files": [files[1]]},
                       {"profile": "nova", "ts": now, "files": [files[2], str(Path(tmp) / "gone.png")]}]
            for entry in entries:
                entry["qc"] = {"safe": True}
            snapshot = [qm.Job(id=1, kind="obraz", label="a", params={"character": "mia"}),
                        qm.Job(id=2, kind="obraz", label="b", params={"character": "mia"},
                               status=qm.DONE),
                        qm.Job(id=3, kind="obraz", label="c", params={})]
            rows = {r["slug"]: r for r in studio.overview(snapshot, entries, now)}
        self.assertEqual((rows["mia"]["queued"], rows["mia"]["today"], rows["mia"]["total"]),
                         (1, 1, 2))
        self.assertEqual((rows["nova"]["queued"], rows["nova"]["total"]), (0, 1))  # bez znikłego
        self.assertEqual(rows[""]["queued"], 1)  # zadanie bez postaci

    def test_queue_jobs_record_their_profile_in_history(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), \
                patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            history.job_context({"character": "mia", "level": "sfw"})
            entry = history.record({"files": ["a.png"]})
            history.job_context(None)
        self.assertEqual(entry["profile"], "mia")


if __name__ == "__main__":
    unittest.main()
