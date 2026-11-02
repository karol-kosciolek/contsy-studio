# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Paczki do publikacji: pliki dla platform, straz SFW, opisy z oznaczeniem AI, napisy."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import content, packs, qc

NUDE = qc.Verdict(ok=False, reasons=[f"{qc.NUDITY} (klatka: nothing)"])

HAS_FFMPEG = shutil.which("ffmpeg") is not None
CAPTIONS = {"instagram": "IG opis\n\n🤖 Postać wygenerowana przez AI.\n\n#plaza #aiinfluencer",
            "tiktok": "TT opis", "fanvue": "FV opis"}


def image(path: Path, size=(108, 144)) -> str:
    Image.new("RGB", size, (200, 120, 90)).save(path)
    return str(path)


def video(path: Path) -> str:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=108x192:rate=10:duration=1", "-pix_fmt", "yuv420p", str(path)],
                   check=True)
    return str(path)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, items):
        with patch.object(packs, "captions", return_value=CAPTIONS):
            return packs.build(items, {"name": "Mia", "slug": "mia"}, {"language": "pl"},
                               self.root / "paczka", "2026-11-01")

    def test_session_goes_to_every_platform(self):
        a = image(self.root / "a.png")
        b = image(self.root / "b.png")
        items = [packs.Item(a, "image", "s1", ratio="3:4"),
                 packs.Item(b, "image", "s1", ratio="3:4")]
        plan = self.build(items)
        out = self.root / "paczka"
        self.assertEqual(sorted(p.name for p in (out / "instagram" / "carousel_1").iterdir()),
                         ["01.jpg", "02.jpg"])
        self.assertFalse((out / "fanvue").exists())
        posts = {(p["platform"], p["type"]) for p in plan["posts"]}
        self.assertEqual(posts, {("instagram", "carousel")})
        self.assertNotIn("18+", (out / "README.txt").read_text(encoding="utf-8"))
        saved = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["day"], "2026-11-01")
        self.assertTrue((out / "captions.txt").exists() and (out / "README.txt").exists())
        self.assertFalse((out / "opisy.txt").exists() or (out / "CZYTAJ.txt").exists())
        self.assertIn("captions.txt", (out / "README.txt").read_text(encoding="utf-8"))
        with Image.open(out / "instagram" / "carousel_1" / "01.jpg") as im:
            self.assertIn(b"AI-generated", im.info.get("exif", b""))

    def test_one_carousel_per_aspect_ratio(self):
        items = [packs.Item(image(self.root / "p.png"), "image", "g", ratio="3:4"),
                 packs.Item(image(self.root / "s.png", (100, 100)), "image", "g", ratio="1:1")]
        plan = self.build(items)
        self.assertEqual(sorted(p["ratio"] for p in plan["posts"] if p["type"] == "carousel"),
                         ["1:1", "3:4"])

    @unittest.skipUnless(HAS_FFMPEG, "brak ffmpeg")
    def test_reel_and_tiktok_get_burned_subtitles(self):
        v = video(self.root / "v.mp4")
        items = [packs.Item(v, "video", "s1", dialogue=["Cześć, miło Cię widzieć!"])]
        plan = self.build(items)
        out = self.root / "paczka"
        self.assertEqual(plan["notes"], [])  # napisy sie udaly
        self.assertEqual({post["type"] for post in plan["posts"]}, {"reel", "video"})
        self.assertTrue(all(Path(path).name.startswith(("reel_", "video_"))
                            for post in plan["posts"] for path in post["files"]))
        reel = out / "instagram" / "reel_1.mp4"
        self.assertTrue(reel.exists() and (out / "tiktok" / "video_1.mp4").exists())
        self.assertNotEqual(reel.read_bytes(), Path(v).read_bytes())  # przeliczony z napisami
        self.assertFalse((out / "fanvue").exists())


class TextTests(unittest.TestCase):
    def test_srt_spreads_lines_over_the_video(self):
        text = packs.srt(["Cześć!", "Lecimy na plażę."], 4.0)
        self.assertIn("00:00:00,000 --> 00:00:01,950\nCześć!", text)
        self.assertIn("00:00:02,000 --> 00:00:03,950\nLecimy na plażę.", text)

    def test_captions_always_carry_the_ai_label(self):
        reply = json.dumps({"instagram": "Złota godzina w Rzymie ☀️ Gdzie wy dziś?",
                            "tiktok": "POV: Rzym o zachodzie", "fanvue": "Tylko dla was 💋",
                            "hashtags": ["rzym", "#travel", "golden hour"]})
        with patch.object(packs.llm, "ask", return_value=(True, reply)) as ask:
            cap = packs.captions({}, "Rzym, taras", "pl")
        self.assertIn(content.RULE, ask.call_args.args[0])  # opisy tez tylko SFW
        self.assertIn("🤖 AI-generated character.", cap["instagram"])
        self.assertIn("#rzym #travel #goldenhour #aiinfluencer", cap["instagram"])
        with patch.object(packs.llm, "ask", return_value=(False, "błąd")):
            fallback = packs.captions({}, "", "en")
        self.assertIn("🤖 AI-generated character.", fallback["tiktok"])

    def test_character_name_never_reaches_the_theme_or_hashtags(self):
        reply = json.dumps({"instagram": "x", "tiktok": "y", "fanvue": "z",
                            "hashtags": ["wirtualnamią", "mią", "rzym", "virtual_mia_aist"]})
        with patch.object(packs.llm, "ask", return_value=(True, reply)) as ask:
            cap = packs.captions({}, "woman with Virtual Mią's glowing skin by the pool",
                                 "pl", "Virtual Mia")
        self.assertNotIn("Mią", ask.call_args.args[1])
        self.assertIn("#rzym #aiinfluencer", cap["instagram"])
        self.assertNotIn("mią", cap["instagram"].lower())

    def test_scene_drops_trigger_word_look_and_style(self):
        prompt = ("Photo of virtual_mia_aist woman. She walks on a beach in a red bikini. The "
                  "woman has tanned skin, long hair, ultra realistic photo, sharp focus")
        self.assertEqual(packs._scene(prompt), "She walks on a beach in a red bikini.")


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def item(self, name="x.png", **kw):
        return packs.Item(image(self.root / name), "image", "g", **kw)

    def test_old_result_with_nudity_is_left_out(self):
        with patch.object(packs.qc, "check", return_value=NUDE):
            kept, notes = packs.safety_check([self.item()], entries=[])
        self.assertEqual(kept, [])
        self.assertIn("nudity", notes[0])

    def test_risky_scene_in_the_prompt_is_left_out(self):
        # prawdziwy prompt (30.10): spod koldry z doklejonym "fully clothed" - wyszlo nago
        prompt = ("adult woman perched on the edge of a large bed, the white duvet pooled around "
                  "her hips and chest. The woman is fully clothed in an outfit that fits the scene.")
        with patch.object(packs.qc, "check") as check:
            kept, notes = packs.safety_check([self.item(prompt=prompt)], entries=[])
        check.assert_not_called()
        self.assertEqual(kept, [])
        self.assertIn("duvet", notes[0])

    def test_result_from_the_old_18_plus_mode_is_left_out(self):
        with patch.object(packs.qc, "check") as check:
            kept, notes = packs.safety_check([self.item(legacy_adult=True)], entries=[])
        check.assert_not_called()
        self.assertEqual(kept, [])
        self.assertIn("18+", notes[0])
        self.assertTrue(packs._item(str(self.root / "x.png"), {"level": "nsfw"}, "g").legacy_adult)

    def test_result_with_a_passing_verdict_is_not_checked_again(self):
        it = self.item("y.png")
        entries = [{"files": [it.path], "qc": {"ok": True, "safe": True}}]
        with patch.object(packs.qc, "check") as check:
            kept, _ = packs.safety_check([it], entries=entries)
        check.assert_not_called()
        self.assertEqual(kept, [it])

    def test_result_with_a_nudity_verdict_is_left_out(self):
        it = self.item("w.png")
        entries = [{"files": [it.path], "qc": NUDE.record()}]
        with patch.object(packs.qc, "check") as check:
            kept, _ = packs.safety_check([it], entries=entries)
        check.assert_not_called()
        self.assertEqual(kept, [])

    def test_unchecked_result_is_checked_now_and_left_out_if_still_unchecked(self):
        it = self.item("u.png")
        entries = [{"files": [it.path], "qc": {"ok": None, "reasons": ["niesprawdzone"]}}]
        unchecked = qc.Verdict(ok=False, reasons=[f"{qc.UNCHECKED} (model wizyjny)"])
        with patch.object(packs.qc, "check", return_value=unchecked) as check:
            kept, notes = packs.safety_check([it], entries=entries)
        check.assert_called_once()
        self.assertEqual(kept, [])
        self.assertIn("validation unavailable", notes[0])
        with patch.object(packs.qc, "check", return_value=qc.Verdict(ok=True)):
            kept, _ = packs.safety_check([it], entries=entries)
        self.assertEqual(kept, [it])


class FolderTests(unittest.TestCase):
    """Paczka nigdy nie kasuje na stale poprzedniej - i paczka dnia przechodzi kontrole."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.day = self.root / "paczki" / "mia" / "2026-11-01"

    def tearDown(self):
        self.tmp.cleanup()

    def build_day(self, *prompts, moved=True):
        """packs.build_day na podstawionych danych; moved - czy Kosz przyjal stara paczke."""
        prompts = prompts or ("She walks on a beach in a red dress.",)
        items = [packs.Item(image(self.root / f"{i}.png"), "image", "s1", ratio="3:4", prompt=p)
                 for i, p in enumerate(prompts)]

        def to_trash(path):
            if moved:
                shutil.rmtree(path)
            return moved

        with patch.object(packs, "PACKS_DIR", self.root / "paczki"), \
                patch.object(packs.characters, "usable",
                             return_value={"name": "Mia", "slug": "mia"}), \
                patch.object(packs, "items_for_day", return_value=items), \
                patch.object(packs.history, "recent", return_value=[]), \
                patch.object(packs.qc, "check", return_value=qc.Verdict(ok=True)), \
                patch.object(packs, "captions", return_value=CAPTIONS), \
                patch("app.autopilot.persona", return_value={"language": "pl"}), \
                patch.object(packs.trash, "move", side_effect=to_trash) as move:
            return packs.build_day("mia", "2026-11-01"), move

    def old_pack(self):
        self.day.mkdir(parents=True)
        (self.day / "plan.json").write_text("{}", encoding="utf-8")
        (self.day / "opisy.txt").write_text("poprawione ręcznie", encoding="utf-8")

    def test_day_pack_goes_through_the_sfw_guard(self):
        risky = "adult woman on a bed, the white duvet pooled around her hips"
        (ok, _), _ = self.build_day("She walks on a beach in a red dress.", risky)
        self.assertTrue(ok)
        plan = json.loads((self.day / "plan.json").read_text(encoding="utf-8"))
        carousel = next(p for p in plan["posts"] if p["platform"] == "instagram")
        self.assertEqual(len(carousel["files"]), 1)  # ryzykowne zdjecie nie trafia nigdzie
        self.assertIn("duvet", " ".join(plan["notes"]))

    def test_day_with_nothing_safe_makes_no_pack(self):
        (ok, msg), _ = self.build_day("adult woman on a bed, the white duvet pooled around her")
        self.assertFalse(ok)
        self.assertIn("SFW check", msg)
        self.assertFalse(self.day.exists())

    def test_rebuilt_day_pack_sends_the_old_one_to_the_recycle_bin(self):
        self.old_pack()
        (ok, _), move = self.build_day()
        self.assertTrue(ok)
        move.assert_called_once_with(self.day)
        self.assertIn("instagram", (self.day / "plan.json").read_text(encoding="utf-8"))

    def test_old_day_pack_stays_when_the_recycle_bin_refuses_it(self):
        self.old_pack()
        (ok, msg), _ = self.build_day(moved=False)
        self.assertFalse(ok)
        self.assertIn("Recycle Bin", msg)
        self.assertEqual((self.day / "opisy.txt").read_text(encoding="utf-8"),
                         "poprawione ręcznie")

    def test_pack_from_history_never_replaces_an_earlier_one(self):
        first = self.root / "2026-10-31_1206"
        self.assertEqual(packs._fresh(first), first)
        first.mkdir()
        self.assertEqual(packs._fresh(first).name, "2026-10-31_1206_2")
        (self.root / "2026-10-31_1206_2").mkdir()
        self.assertEqual(packs._fresh(first).name, "2026-10-31_1206_3")

    def test_failed_build_leaves_no_half_pack(self):
        items = [packs.Item(image(self.root / "a.png"), "image", "s1")]
        with patch.object(packs, "captions", side_effect=RuntimeError("model padł")), \
                self.assertRaises(RuntimeError):
            packs.build(items, None, {}, self.day)
        self.assertFalse(self.day.exists())


if __name__ == "__main__":
    unittest.main()
