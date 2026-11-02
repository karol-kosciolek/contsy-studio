# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Regression checks for mandatory SFW validation and preservation of local data."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app import characters, content, history, llm, prefs, profiles, qc, queue_manager as qm


class ValidationTests(unittest.TestCase):
    def test_missing_second_vision_response_does_not_pass(self):
        covered = {"chest": "shirt", "hips": "jeans"}
        for second in (None, {}, {"chest": "shirt"}):
            with self.subTest(second=second), patch.object(qc.llm, "coverage", side_effect=[covered, second]):
                self.assertIsNone(qc.coverage("test.png"))

    def test_malformed_coverage_fails_closed(self):
        with patch.object(qc, "coverage", return_value={"hips": "jeans"}):
            verdict = qc.check("test.png", None, quality=False)
        self.assertFalse(verdict.ok)
        self.assertFalse(verdict.safe)

    def test_every_file_in_a_batch_is_checked_and_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / "one.png", root / "two.png"]
            for path in files:
                path.write_bytes(b"image")
            verdicts = [qc.Verdict(ok=True), qc.Verdict(ok=False, reasons=[qc.UNCHECKED])]
            with patch.object(qc, "REJECTED_DIR", root / "quarantine"), patch.object(qc, "check", side_effect=verdicts) as check, patch.object(history, "forget"), patch.object(history, "annotate"):
                ok, output, _ = qc.generate_checked(lambda: (True, list(map(str, files)), ""), None, [], guard=False)
            self.assertFalse(ok)
            self.assertEqual(output, [])
            self.assertEqual(check.call_count, 2)
            self.assertTrue(all(not p.exists() for p in files))
            self.assertEqual(len(list((root / "quarantine").glob("*.png"))), 2)

    def test_quarantine_never_overwrites_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected = root / "quarantine"
            rejected.mkdir()
            (rejected / "same.png").write_bytes(b"old")
            source = root / "same.png"
            source.write_bytes(b"new")
            with patch.object(qc, "REJECTED_DIR", rejected), patch.object(history, "forget"):
                target = qc.reject(str(source), qc.Verdict(ok=False, reasons=[qc.UNCHECKED]))
            self.assertEqual((rejected / "same.png").read_bytes(), b"old")
            self.assertEqual(Path(target).read_bytes(), b"new")

    def test_video_checks_middle_of_clip_when_endpoints_are_safe(self):
        sampled = {}
        def extract(args, **kwargs):
            moment = float(args[args.index("-ss") + 1])
            frame = Path(args[-1])
            frame.write_bytes(b"frame")
            sampled[str(frame)] = moment
        def check(path, *args):
            if sampled[path] == 4.0:
                return qc.Verdict(ok=False, reasons=[qc.NUDITY])
            return qc.Verdict(ok=True)
        with patch.object(qc, "_duration", return_value=8.0), patch.object(qc.subprocess, "run", side_effect=extract), patch.object(qc, "check", side_effect=check):
            verdict = qc.check_video("clip.mp4", None, quality=False)
        self.assertFalse(verdict.safe)
        self.assertTrue(verdict.nude)
        self.assertEqual(sorted(sampled.values()), [0.0, 2.0, 4.0, 6.0, 7.9])

    def test_video_missing_any_required_frame_fails_closed(self):
        def extract(args, **kwargs):
            moment = float(args[args.index("-ss") + 1])
            if moment != 2.0:
                Path(args[-1]).write_bytes(b"frame")
        with patch.object(qc, "_duration", return_value=4.0), patch.object(qc.subprocess, "run", side_effect=extract), patch.object(qc, "check", return_value=qc.Verdict(ok=True)):
            verdict = qc.check_video("clip.mp4", None, quality=False)
        self.assertFalse(verdict.safe)
        self.assertTrue(verdict.unchecked)
        self.assertIn("2.00s", verdict.summary())

    def test_short_video_still_checks_three_frames(self):
        moments = []
        def extract(args, **kwargs):
            moments.append(float(args[args.index("-ss") + 1]))
            Path(args[-1]).write_bytes(b"frame")
        with patch.object(qc, "_duration", return_value=1.0), patch.object(qc.subprocess, "run", side_effect=extract), patch.object(qc, "check", return_value=qc.Verdict(ok=True)):
            verdict = qc.check_video("clip.mp4", None, quality=False)
        self.assertTrue(verdict.ok)
        self.assertEqual(moments, [0.0, 0.5, 0.9])

    def test_minor_flagged_training_preserves_source_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = root / "test" / "photos"
            photos.mkdir(parents=True)
            photo = photos / "01.png"
            photo.write_bytes(b"original")
            ch = {"slug": "test", "name": "Test", "steps": 1}
            with patch.object(characters, "DIR", root), patch.object(characters, "get", return_value=ch), patch.object(characters, "_save"), patch.object(characters.llm, "check_photo", return_value={"person": True, "age": 16}):
                ok, files, notes, error = characters.run_job({"slug": "test"}, lambda *a: None)
            self.assertFalse(ok)
            self.assertIn("preserved", error)
            self.assertEqual(photo.read_bytes(), b"original")


class MultilingualPreflightTests(unittest.TestCase):
    def test_common_explicit_terms_are_refused(self):
        for prompt in ("una mujer desnuda", "sin ropa", "une femme nue", "nudité",
                       "une personne sans vêtements", "eine nackte Frau", "Geschlechtsverkehr",
                       "una donna nuda", "senza vestiti", "uma mulher nua", "sem roupa",
                       "pornografía", "contenido sexual", "contenu sexuel", "sesso esplicito"):
            with self.subTest(prompt=prompt):
                self.assertEqual(content.check(prompt), content.REFUSAL)

    def test_safe_multilingual_prompts_and_ambiguous_words_are_allowed(self):
        for prompt in ("Una mujer adulta con un vestido rojo", "Une femme adulte en robe rouge",
                       "Eine erwachsene Frau mit rotem Kleid", "Una donna adulta con un vestito rosso",
                       "Uma mulher adulta com vestido vermelho", "Un nudo en la cuerda",
                       "a nu metal concert", "Nudeln in einer Schüssel", "batom nude", "rossetto nude",
                       "labial nude", "rouge à lèvres nude", "nude Lippenstift", "vestido nude"):
            with self.subTest(prompt=prompt):
                self.assertEqual(content.check(prompt), "")

    def test_comparison_normalization_blocks_invisible_and_compatibility_bypasses(self):
        for prompt in ("nu\u200bde portrait", "ｎｕｄｅ portrait", "desnu\u200bda", "nack\u200dt"):
            with self.subTest(prompt=prompt):
                self.assertEqual(content.check(prompt), content.REFUSAL)
        self.assertEqual(content.enforce("ａ clothed adult in a red dress"), "ａ clothed adult in a red dress")

    def test_known_saved_prompt_bypasses_are_refused_before_queue_execution(self):
        prompts = ("a n4ked woman standing by a window in morning light",
                   "kobieta opalająca się na plaży bez stroju kąpielowego",
                   "an adult with no clothes", "an adult without clothing",
                   "an adult without any clothes", "an adult wearing nothing")
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(content.check(prompt), content.REFUSAL)
                self.assertIsNone(content.enforce(prompt))
                self.assertTrue(qm._prompt_refusal({"prompt": prompt}))

    def test_naked_eye_and_nude_cosmetics_remain_safe(self):
        for prompt in ("stars visible to the naked eye", "a naked-eye view of the moon",
                       "nude lipstick and nude heels", "a dress in nude colors"):
            with self.subTest(prompt=prompt):
                self.assertEqual(content.check(prompt), "")

    def test_known_legacy_level_is_casefolded_without_interpreting_unrelated_flags(self):
        self.assertTrue(content.legacy_adult({"level": " NSFW "}))
        self.assertTrue(content.legacy_adult({"level": "ｎｓｆｗ"}))
        self.assertFalse(content.legacy_adult({"level": "sfw", "adult": True}))


class QueueTests(unittest.TestCase):
    def test_restored_known_bypasses_are_blocked_without_changing_finished_history(self):
        prompts = ("a n4ked woman standing by a window in morning light",
                   "kobieta opalająca się na plaży bez stroju kąpielowego")
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            saved = []
            for prompt in prompts:
                for status in (qm.PENDING, qm.RUNNING, qm.DONE):
                    saved.append({"id": len(saved) + 1, "kind": "test", "label": "legacy",
                                  "params": {"prompt": prompt}, "status": status,
                                  "files": ["preserved-output.png"] if status == qm.DONE else []})
            state.write_text(json.dumps({"jobs": saved}, ensure_ascii=False), encoding="utf-8")
            original = state.read_bytes()
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                jobs = qm.snapshot()
                self.assertEqual(len(jobs), len(saved))
                for job, entry in zip(jobs, saved):
                    with self.subTest(prompt=entry["params"]["prompt"], status=entry["status"]):
                        self.assertEqual(job.params, entry["params"])
                        if entry["status"] == qm.DONE:
                            self.assertEqual(job.status, qm.DONE)
                            self.assertEqual(job.files, entry["files"])
                        else:
                            self.assertEqual(job.status, qm.ERROR)
                            self.assertEqual(job.error, content.REFUSAL)
                self.assertEqual(state.read_bytes(), original)

    def test_restored_queue_skips_malformed_entries_and_blocks_unsafe_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            state.write_text(json.dumps({"next_id": "broken", "rates": {"obraz": "bad"}, "jobs": [None, {"id": 1}, {"id": 2, "kind": "obraz", "label": "unsafe", "params": {"positive": "nude portrait"}}, {"id": 3, "kind": "obraz", "label": "safe", "params": {"positive": "a landscape"}, "status": qm.RUNNING}]}), encoding="utf-8")
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                jobs = {j.id: j for j in qm.snapshot()}
                self.assertEqual(set(jobs), {2, 3})
                self.assertEqual(jobs[2].status, qm.ERROR)
                self.assertEqual(jobs[3].status, qm.PENDING)
                self.assertEqual(qm._STATE_INFO["next_id"], 4)

    def test_restored_malformed_status_and_timestamps_do_not_break_other_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            base = {"kind": "obraz", "label": "image", "params": {}}
            jobs = [{**base, "id": 1, "status": []},
                    {**base, "id": 2, "created": "invalid"},
                    {**base, "id": 3, "files": "bad"},
                    {**base, "id": 4}]
            state.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                self.assertEqual([j.id for j in qm.snapshot()], [4])

    def test_replay_prompt_is_checked_before_runner(self):
        for prompt in ("naked portrait", "a n4ked woman standing by a window in morning light",
                       "kobieta opalająca się na plaży bez stroju kąpielowego"):
            job = qm.Job(id=1, kind="test", label="unsafe", params={"positive": prompt})
            with self.subTest(prompt=prompt), patch.dict(qm._RUNNERS, {"test": unittest.mock.Mock()}), patch.object(qm, "_save"), patch.object(qm, "FILES_DIR", Path(tempfile.gettempdir()) / "nonexistent-qm-security-test"):
                qm._run(job)
                qm._RUNNERS["test"].assert_not_called()
                self.assertEqual(job.status, qm.ERROR)

    def test_minor_in_intimate_clothing_is_blocked_for_every_queue_kind(self):
        for kind in ("obraz", "wideo", "narracja", "test"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "minors"):
                qm.add(kind, "blocked", {"text": "a 15-year-old in a bikini"})

    def test_restored_minor_job_never_reaches_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            state.write_text(json.dumps({"jobs": [{"id": 1, "kind": "narracja", "label": "legacy",
                                                  "params": {"text": "a １５ｙｏ in a bi\u200bkini"}}]}), encoding="utf-8")
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                self.assertEqual(qm.snapshot()[0].status, qm.ERROR)
                self.assertIn("minors", qm.snapshot()[0].error)

    def test_runner_preflight_combines_age_and_clothing_from_separate_fields(self):
        job = qm.Job(id=1, kind="test", label="blocked", params={"idea": "a 15-year-old", "text": "in a bikini"})
        with patch.dict(qm._RUNNERS, {"test": unittest.mock.Mock()}), patch.object(qm, "_save"), patch.object(qm, "FILES_DIR", Path(tempfile.gettempdir()) / "nonexistent-qm-security-test"):
            qm._run(job)
            qm._RUNNERS["test"].assert_not_called()
        self.assertEqual(job.status, qm.ERROR)
        self.assertIn("minors", job.error)

    def test_attachment_field_cannot_escape_queue_folder(self):
        with self.assertRaisesRegex(ValueError, "attachment"):
            qm.add("test", "test", {}, files={"../outside": "anything"})


class PersistenceTests(unittest.TestCase):
    def test_concurrent_preference_updates_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"):
            barrier = threading.Barrier(2)
            def save(key, value):
                barrier.wait()
                prefs.save(**{key: value})
            threads = [threading.Thread(target=save, args=("folder", "my-folder")), threading.Thread(target=save, args=("character", "test"))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            result = prefs.load()
            self.assertEqual(result["folder"], "my-folder")
            self.assertEqual(result["character"], "test")

    def test_autosave_preserves_an_existing_file_with_the_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "image.png"
            source.write_bytes(b"new")
            destination = root / "saved"
            destination.mkdir()
            (destination / "image.png").write_bytes(b"old")
            with patch.object(prefs, "load", return_value={"autosave": True, "folder": str(destination)}):
                prefs.autosave([str(source)])
            self.assertEqual((destination / "image.png").read_bytes(), b"old")
            self.assertEqual((destination / "image_2.png").read_bytes(), b"new")

    def test_history_forget_does_not_match_prompt_or_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            history.record({"idea": "a.png", "files": ["C:/out/b.png"]})
            history.record({"files": ["C:/out/a.png", "C:/out/c.png"]})
            history.forget(["C:/out/a.png"])
            records = history.recent()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["files"], ["C:/out/c.png"])
            self.assertEqual(records[1]["idea"], "a.png")

    def test_history_annotation_cannot_approve_a_different_file_with_the_same_name(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            first = str(Path(tmp) / "one" / "image.png")
            second = str(Path(tmp) / "two" / "image.png")
            history.record({"files": [first]})
            history.record({"files": [second]})
            history.annotate(first, qc={"safe": True})
            self.assertEqual(history.entry_for(first)["qc"], {"safe": True})
            self.assertNotIn("qc", history.entry_for(second))

    def test_character_slug_cannot_escape_library(self):
        for slug in ("../escape", "C:/escape", "a/b", "a\\b", ".", ".."):
            with self.subTest(slug=slug):
                self.assertIsNone(characters.get(slug))
                with self.assertRaises(ValueError):
                    characters._file(slug)

    def test_profile_load_has_no_raw_path_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outside.json").write_text(json.dumps({"name": "Outside"}), encoding="utf-8")
            with patch.object(profiles, "PROFILES_DIR", root / "profiles"):
                self.assertIsNone(profiles.load("../outside"))


if __name__ == "__main__":
    unittest.main()
