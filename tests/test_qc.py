# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Straz SFW i kontrola jakosci: werdykt z pomiarow i liczenie od nowa zlych wynikow (bez karty)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import history, qc

COVERED = {"chest": "red bikini top", "hips": "red bikini bottoms", "hands": "normal"}
NUDE = f"{qc.NUDITY} (klatka: nothing; biodra: nothing)"


class JudgeTests(unittest.TestCase):
    def test_visible_nudity_is_always_rejected_but_not_bare_shoulders(self):
        bare = {"chest": "white sheet draped over left breast, right breast and nipple bare",
                "hips": "white sheet", "hands": "normal"}
        self.assertTrue(qc.judge(0.6, bare).nude)
        self.assertTrue(qc.judge(None, bare, quality=False).nude)  # straz bez kontroli jakosci
        self.assertFalse(qc.judge(0.6, {**COVERED, "hips": "nothing, bare buttocks visible"}).ok)
        shoulders = {"chest": "white sheet, bare shoulders", "hips": "white sheet",
                     "hands": "normal"}
        self.assertTrue(qc.judge(0.6, shoulders).ok)
        self.assertTrue(qc.judge(0.6, COVERED).ok)

    def test_face_below_threshold_is_another_person(self):
        verdict = qc.judge(0.35, COVERED)
        self.assertFalse(verdict.ok)
        self.assertIn("different face", verdict.summary())
        self.assertTrue(qc.judge(None, COVERED).ok)  # twarz niewidoczna
        self.assertTrue(qc.judge(0.35, COVERED, quality=False).ok)  # kontrola jakosci wylaczona

    def test_malformed_hands(self):
        hands = {**COVERED, "hands": "six fingers on left hand"}
        self.assertFalse(qc.judge(0.6, hands).ok)
        self.assertTrue(qc.judge(0.6, hands, quality=False).ok)

    def test_silent_vision_model_means_unchecked_not_a_pass(self):
        with patch.object(qc, "coverage", return_value=None):
            verdict = qc.check("x.png", None, quality=False)
        self.assertFalse(verdict.ok)
        self.assertTrue(verdict.unchecked and not verdict.nude)
        self.assertFalse(verdict.record()["ok"])  # liczy sie do zapasu, paczka sprawdzi znowu

    def test_video_without_frames_is_unchecked(self):
        with tempfile.TemporaryDirectory() as tmp:
            verdict = qc.check_video(str(Path(tmp) / "brak.mp4"), None, quality=False)
        self.assertTrue(verdict.unchecked)


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.n = 0
        self.patches = [patch.object(qc, "REJECTED_DIR", self.root / "odrzucone"),
                        patch.object(qc, "enabled", return_value=True),
                        patch.object(history, "forget"), patch.object(history, "annotate")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def make(self):
        self.n += 1
        f = self.root / f"img_{self.n}.png"
        f.write_bytes(b"png")
        return True, [str(f)], ""

    def verdicts(self, *kinds):
        reasons = {"ok": [], "nude": [NUDE], "face": ["inna twarz (podobieństwo 0.30 < 0.40)"],
                   "unchecked": [f"{qc.UNCHECKED} (model wizyjny nie opisał stroju)"]}
        return [qc.Verdict(ok=k == "ok", reasons=reasons[k]) for k in kinds]

    def test_bad_results_are_set_aside_and_generated_again(self):
        notes = []
        with patch.object(qc, "check", side_effect=self.verdicts("nude", "face", "ok")):
            ok, files, _ = qc.generate_checked(self.make, {"slug": "mia"}, notes)
        self.assertTrue(ok)
        self.assertEqual(Path(files[0]).name, "img_3.png")
        rejected = sorted(p.name for p in (self.root / "odrzucone").glob("*.png"))
        self.assertEqual(rejected, ["img_1.png", "img_2.png"])
        log = (self.root / "odrzucone" / "qc.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log), 2)
        self.assertEqual(json.loads(log[0])["reasons"], [NUDE])
        self.assertIn("attempt 3", notes[-1])

    def test_nude_result_never_stays(self):
        with patch.object(qc, "check", side_effect=self.verdicts("nude", "nude", "nude")):
            ok, files, msg = qc.generate_checked(self.make, {"slug": "mia"}, [])
        self.assertFalse(ok)
        self.assertEqual(files, [])
        self.assertTrue(msg.startswith("⛔"))
        self.assertEqual(sorted(p.name for p in self.root.glob("*.png")), [])  # nic nie zostalo
        self.assertEqual(len(list((self.root / "odrzucone").glob("*.png"))), qc.MAX_TRIES)

    def test_after_max_tries_a_quality_miss_stays_with_a_warning(self):
        notes = []
        with patch.object(qc, "check", side_effect=self.verdicts("face", "face", "face")):
            ok, files, _ = qc.generate_checked(self.make, {"slug": "mia"}, notes)
        self.assertTrue(ok and Path(files[0]).exists())
        self.assertEqual(self.n, qc.MAX_TRIES)
        self.assertTrue(notes[-1].startswith("⚠"))

    def test_unchecked_result_is_quarantined(self):
        notes = []
        with patch.object(qc, "check", side_effect=self.verdicts("unchecked")):
            ok, files, _ = qc.generate_checked(self.make, {"slug": "mia"}, notes)
        self.assertFalse(ok)
        self.assertEqual(files, [])
        self.assertEqual(self.n, 1)
        self.assertTrue((self.root / "odrzucone" / "img_1.png").exists())

    def test_without_a_character_the_guard_is_mandatory(self):
        with patch.object(qc, "check", return_value=qc.Verdict(ok=True)) as check:
            ok, files, _ = qc.generate_checked(self.make, None, [], guard=False)
        check.assert_called_once()
        self.assertEqual(self.n, 1)

    def test_reference_images_turn_the_guard_on_without_a_character(self):
        with patch.object(qc, "check", side_effect=self.verdicts("ok")) as check:
            ok, files, _ = qc.generate_checked(self.make, None, [], guard=True)
        self.assertTrue(ok)
        self.assertIs(check.call_args.args[2], False)  # bez postaci - tylko straz SFW

    def test_guard_runs_even_with_quality_control_switched_off(self):
        with patch.object(qc, "enabled", return_value=False), \
                patch.object(qc, "check", side_effect=self.verdicts("ok")) as check:
            qc.generate_checked(self.make, {"slug": "mia"}, [])
        check.assert_called_once()
        self.assertIs(check.call_args.args[2], False)

    def render(self, client, ch, text, w, h, out, seed, cb=None):
        self.seeds.append(seed)
        Path(out).write_bytes(b"png")
        return True, str(out)

    def test_bad_start_frame_is_rendered_again_with_another_seed(self):
        self.seeds, notes = [], []
        frame = self.root / "video_char1.png"
        with patch.object(qc.characters, "render", side_effect=self.render), \
                patch.object(qc, "check", side_effect=self.verdicts("nude", "ok")):
            ok, path = qc.render_checked(None, {"slug": "mia"}, "beach", 720, 1280, frame, 7,
                                         notes)
        self.assertTrue(ok and Path(path).exists())
        self.assertEqual(self.seeds, [7, 1007])
        self.assertTrue((self.root / "odrzucone" / "video_char1.png").exists())

    def test_nude_start_frame_three_times_stops_the_video(self):
        self.seeds = []
        with patch.object(qc.characters, "render", side_effect=self.render), \
                patch.object(qc, "check", side_effect=self.verdicts("nude", "nude", "nude")):
            ok, msg = qc.render_checked(None, {"slug": "mia"}, "beach", 720, 1280,
                                        self.root / "video_char1.png", 7, [])
        self.assertFalse(ok)
        self.assertTrue(msg.startswith("⛔"))


class AnnotateTests(unittest.TestCase):
    def test_qc_verdict_lands_in_the_files_history_entry(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), \
                patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            history.record({"files": ["C:/o/a.png"]})
            history.record({"files": ["C:/o/b.png"]})
            history.annotate("C:/o/b.png", qc={"ok": False})
            a, b = history.entry_for("a.png"), history.entry_for("b.png")
        self.assertNotIn("qc", a)
        self.assertEqual(b["qc"], {"ok": False})


if __name__ == "__main__":
    unittest.main()
