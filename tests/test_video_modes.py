# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Tryby wideo: Jakosc (720p, 6 krokow), Szybki (480p, 4 kroki) i Podglad (Szybki bez 1080p) -
rozmiary, kroki, przebieg (SeedVR2 albo nie), kolejka, seria z polecenia, persona autopilota
i paczki."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import autopilot, config, ffmpeg_utils, history, model_manager, packs, planner
from app import qc, queue_manager, templates, upscale, video_gen

RATES = {"wideo": 170.0, "wideo-szybkie": 60.0, "wideo-podglad": 30.0, "obraz": 30.0}


def _run(speed: str):
    """Caly _generate z atrapami silnika i FFmpeg: (wynik, info z historii, upscale.video,
    ensure, combine_video_audio)."""
    client = MagicMock()
    client.run_and_wait.return_value = SimpleNamespace(files=[{"filename": "wan.mp4"}],
                                                       images=[])
    client.download_output.return_value = b"mp4"
    recorded = {}
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
            patch.object(video_gen, "_client", return_value=client), \
            patch.object(video_gen, "_models", return_value={}), \
            patch.object(model_manager, "ensure", return_value=None) as ensure, \
            patch.object(templates, "build", return_value={}), \
            patch.object(ffmpeg_utils, "convert_fps", return_value=(True, "")), \
            patch.object(ffmpeg_utils, "combine_video_audio", return_value=(True, "")) as combine, \
            patch.object(ffmpeg_utils, "video_fps", return_value=30.0), \
            patch.object(qc, "check_video", return_value=qc.Verdict(True)), \
            patch.object(upscale, "video", return_value=(True, "")) as up, \
            patch.object(history, "record", side_effect=recorded.update):
        out = video_gen._generate("A woman walks on a beach.", 5, "Poziomo 16:9",
                                  sound_on=False, speed=speed)
    return out, recorded, up, ensure, combine


class ModeTests(unittest.TestCase):
    def test_fast_mode_renders_480p_in_4_steps_and_quality_720p_in_6(self):
        fast = video_gen.sampling_plan(video_gen.MODES[video_gen.FAST]["steps"])
        quality = video_gen.sampling_plan(video_gen.MODES[video_gen.QUALITY]["steps"])
        self.assertEqual((fast["steps"], fast["cfg"], fast["lora"]), (4, 1.0, 1.0))
        self.assertEqual((quality["steps"], quality["cfg"], quality["lora"]), (6, 1.0, 1.0))
        self.assertEqual(video_gen.sizes(9 / 16, video_gen.FAST), ((480, 848), (720, 1280)))
        self.assertEqual(video_gen.sizes(9 / 16, video_gen.QUALITY), ((720, 1280), (720, 1280)))

    def test_both_modes_end_in_the_same_1080p_frame(self):
        for name, ratio in video_gen.ASPECTS.items():
            (w, h), full = video_gen.sizes(ratio, video_gen.FAST)
            self.assertEqual((w % 16, h % 16), (0, 0), name)
            self.assertLess(abs(w / h - ratio), 0.015, name)  # Wan liczy prawie te same proporcje
            self.assertEqual(full, video_gen.sizes(ratio, video_gen.QUALITY)[0], name)
        # 848x480 nie jest 16:9 co do piksela - 1080p liczone z klatki trybu Jakosc
        self.assertEqual(upscale.target_size(*video_gen.sizes(16 / 9, video_gen.FAST)[1]),
                         (1920, 1080))

    def test_preview_is_fast_without_1080p(self):
        self.assertEqual(video_gen.sizes(9 / 16, video_gen.PREVIEW),
                         video_gen.sizes(9 / 16, video_gen.FAST))
        self.assertEqual(video_gen.MODES[video_gen.PREVIEW]["steps"], 4)
        self.assertFalse(video_gen.upscales(video_gen.PREVIEW))
        self.assertTrue(video_gen.upscales(video_gen.FAST) and video_gen.upscales(video_gen.QUALITY))
        self.assertEqual(video_gen.mode_tag(video_gen.PREVIEW), " · preview")
        self.assertNotIn(video_gen.PREVIEW, video_gen.PUBLISH_MODES)


class PipelineTests(unittest.TestCase):
    def test_preview_skips_seedvr2_and_stays_480p(self):
        out, info, up, ensure, combine = _run(video_gen.PREVIEW)
        self.assertTrue(out.ok, out.error)
        up.assert_not_called()
        self.assertTrue(combine.call_args.kwargs["compress"])  # nie bezstratny plik posredni
        self.assertEqual((info["width"], info["height"], info["speed"]),
                         (848, 480, video_gen.PREVIEW))
        self.assertEqual(info["engine"], "Wan 2.2 A14B")
        self.assertNotIn("seedvr2-3b", ensure.call_args_list[0].args[0])
        self.assertTrue(any("Preview 848×480" in n for n in out.notes))

    def test_fast_still_ends_in_1080p(self):
        out, info, up, ensure, combine = _run(video_gen.FAST)
        self.assertTrue(out.ok, out.error)
        up.assert_called_once()
        self.assertFalse(combine.call_args.kwargs["compress"])  # SeedVR2 oddaje juz zwykle H.264
        self.assertEqual(up.call_args.args[3:5], (1280, 720))  # 1080p z klatki trybu Jakosc
        self.assertEqual((info["width"], info["height"]), (1920, 1080))
        self.assertEqual(info["engine"], "Wan 2.2 A14B + SeedVR2")
        self.assertIn("seedvr2-3b", ensure.call_args_list[0].args[0])


class QueueTests(unittest.TestCase):
    def test_preview_has_its_own_time_average(self):
        with patch.dict(queue_manager._RATE, RATES):
            job = queue_manager.Job(1, "wideo", "a", {"seconds": 5, "speed": video_gen.PREVIEW})
            self.assertEqual(job.estimate(), 150.0)
        self.assertEqual(queue_manager._rate_key("wideo", {"speed": "preview"}), "wideo-podglad")

    def test_fast_video_keeps_its_own_time_average(self):
        self.assertEqual(video_gen.FAST, "fast")  # tę nazwę zna queue_manager (bez importu)
        with patch.dict(queue_manager._RATE, RATES):
            fast = queue_manager.Job(1, "wideo", "a", {"seconds": 5, "speed": video_gen.FAST})
            quality = queue_manager.Job(2, "wideo", "b", {"seconds": 5})
            self.assertEqual((fast.estimate(), quality.estimate()), (300.0, 850.0))
        self.assertEqual(queue_manager._rate_key("wideo", {"speed": "fast"}), "wideo-szybkie")
        self.assertEqual(queue_manager._rate_key("obraz", {"speed": "fast"}), "obraz")


class PlanTests(unittest.TestCase):
    ROWS = [[planner.KINDS["video"], "Pionowo 9:16", 5, "tak", "SFW", "A woman walks on a beach."],
            [planner.KINDS["image"], "Portret 3:4", 0, "—", "SFW", "A woman on a sunny terrace."]]

    def test_auto_plan_gives_the_mode_to_videos_only(self):
        with patch.object(planner.queue_manager, "add") as add:
            added, _ = planner.enqueue(self.ROWS, [], "plaża", speed=video_gen.FAST,
                                       character={"slug": "mia", "name": "Mia"})
        self.assertEqual(added, 2)
        (kind, label, params), _ = add.call_args_list[0]
        self.assertEqual((kind, params["speed"]), ("wideo", video_gen.FAST))
        self.assertIn("fast", label)
        kind, _, params = add.call_args_list[1].args
        self.assertEqual(kind, "obraz")
        self.assertNotIn("speed", params)

    def test_plan_estimate_follows_the_mode(self):
        with patch.dict(queue_manager._RATE, RATES):
            self.assertEqual(planner.estimate(self.ROWS, video_gen.QUALITY), 170.0 * 5 + 30.0)
            self.assertEqual(planner.estimate(self.ROWS, video_gen.FAST), 60.0 * 5 + 30.0)


class PersonaTests(unittest.TestCase):
    def test_persona_keeps_a_valid_video_mode(self):
        self.assertEqual(autopilot.clean({})["video_speed"], video_gen.QUALITY)
        self.assertEqual(autopilot.clean({"video_speed": "fast"})["video_speed"], video_gen.FAST)
        self.assertEqual(autopilot.clean({"video_speed": "turbo"})["video_speed"],
                         video_gen.QUALITY)
        # autopilot robi tresci do publikacji - podglad 480p sie do tego nie nadaje
        self.assertEqual(autopilot.clean({"video_speed": "preview"})["video_speed"],
                         video_gen.QUALITY)

    def test_fast_persona_needs_less_card_time_per_day(self):
        with patch.dict(queue_manager._RATE, RATES):
            quality = autopilot.daily_seconds(autopilot.clean({"video_seconds": 6}))
            fast = autopilot.daily_seconds(autopilot.clean({"video_seconds": 6,
                                                            "video_speed": video_gen.FAST}))
        self.assertAlmostEqual(quality - fast, (170.0 - 60.0) * 6)


class PackTests(unittest.TestCase):
    def test_selection_pack_leaves_out_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            photo, preview = Path(tmp) / "img_1.png", Path(tmp) / "video_2.mp4"
            photo.write_bytes(b"png")
            preview.write_bytes(b"mp4")
            entries = [{"files": [str(photo)], "positive": "a terrace"},
                       {"files": [str(preview)], "speed": video_gen.PREVIEW,
                        "positive": "a beach"}]
            plan = {"posts": [{"platform": "instagram"}], "notes": []}
            with patch.object(history, "recent", return_value=entries), \
                    patch.object(packs, "safety_check",
                                 side_effect=lambda items, entries=None: (items, [])), \
                    patch.object(packs, "_fresh", return_value=Path(tmp) / "paczka"), \
                    patch.object(packs, "build", return_value=plan) as build:
                ok, msg = packs.build_selection([str(photo), str(preview)])
                self.assertTrue(ok)
                self.assertEqual([i.path for i in build.call_args.args[0]], [str(photo)])
                self.assertIn("Skipped 480p previews: 1", msg)
                ok, msg = packs.build_selection([str(preview)])
                self.assertFalse(ok)
                self.assertIn("480p previews", msg)


if __name__ == "__main__":
    unittest.main()
