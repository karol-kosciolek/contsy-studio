# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Release regressions at direct generation and archive entry points."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from app import autopilot, char_gen, config, history, image_gen, llm, model_manager
from app import planner, qc, refs, tts, video_gen


class GenerationGuards(unittest.TestCase):
    def test_direct_image_video_and_character_jobs_reject_before_download(self):
        with patch.object(model_manager, "ensure") as ensure:
            self.assertFalse(image_gen.txt2img("missing", "nude adult woman", "", 512, 512,
                                               8, 1, 1).ok)
            self.assertFalse(video_gen.generate("nude adult woman", 5, "Portrait 9:16").ok)
            self.assertFalse(char_gen.run_job({"description": "nude adult woman"}, None)[0])
        ensure.assert_not_called()

    def test_unchecked_image_never_reaches_history(self):
        client = MagicMock()
        data = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(data, format="PNG")
        client.run_and_wait.return_value = SimpleNamespace(images=[data.getvalue()])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
                patch.object(image_gen, "_client", return_value=client), \
                patch.object(qc, "check", return_value=qc.Verdict(False, [qc.UNCHECKED])), \
                patch.object(qc, "reject") as reject, \
                patch.object(history, "record") as record:
            out = image_gen._run({}, "test", {"positive": "landscape"}, None)
        self.assertFalse(out.ok)
        self.assertEqual(out.files, [])
        reject.assert_called_once()
        record.assert_not_called()

    def test_failed_upscale_cannot_publish_a_missing_reference_output(self):
        from app import upscale
        client = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
                patch.object(model_manager, "ensure", return_value=None), \
                patch.object(image_gen, "_client", return_value=client), \
                patch.object(llm, "describe_person", return_value=""), \
                patch.object(refs, "compose_refs", return_value=(True, "draft.png")), \
                patch.object(upscale, "image", return_value=(False, "upscale failed")), \
                patch.object(history, "record") as record:
            out = image_gen.edit_with_refs([(1, "reference.png")], "A city scene", 1080, 1080)
        self.assertFalse(out.ok)
        self.assertEqual(out.error, "upscale failed")
        self.assertEqual(out.files, [])
        record.assert_not_called()

    def test_reference_composition_rejects_excess_images_instead_of_truncating(self):
        client = MagicMock()
        ok, error = refs.compose(client, ["one"] * 4, "landscape", 512, 512, "unused.png")
        self.assertFalse(ok)
        self.assertIn("at most 3", error)
        client.upload_image.assert_not_called()

    def test_no_jobs_is_not_a_successful_autopilot_session(self):
        with patch.object(autopilot, "session_theme", return_value=(True, "city walk")), \
                patch.object(planner, "write_series", return_value=["city walk"]), \
                patch.object(planner, "enqueue", return_value=(0, 1)):
            ok, error = autopilot.plan_session({"slug": "test"}, autopilot.clean({}),
                                               "2026-11-01", 1, 0)
        self.assertFalse(ok)
        self.assertIn("No jobs", error)


class ModelAndLanguage(unittest.TestCase):
    def test_mixed_media_formats_survive_an_incorrect_model_format(self):
        order = ("1 square product photo of a red mug on a wooden table and "
                 "1 silent landscape video of the same mug, 5 seconds")
        groups = [{"kind": kind, "count": 1, "theme": "A red mug on a wooden table",
                   "format": "Square 1:1", "seconds": 5, "sound": False}
                  for kind in ("image", "video")]
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(planner, "parse_order", return_value=(True, groups)), \
                patch.object(planner, "write_series", return_value=["A red mug on a wooden table"]), \
                patch.object(planner.queue_manager, "add") as add:
            ok, rows = planner.make_plan(order)
            self.assertTrue(ok)
            self.assertEqual([row[1] for row in rows], ["Square 1:1", "Landscape 16:9"])
            self.assertEqual(planner.enqueue(rows, character={"slug": "test", "name": "Test"}), (2, 0))
        video = add.call_args_list[1].args[2]
        self.assertEqual(video["aspect"], "Landscape 16:9")
        self.assertFalse(video["sound"])

    def test_multiple_photo_group_formats_are_not_overridden(self):
        groups = [{"kind": "image", "count": 1, "theme": "A mug", "format": fmt}
                  for fmt in ("Square 1:1", "Portrait 9:16")]
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(planner, "parse_order", return_value=(True, groups)), \
                patch.object(planner, "write_series", return_value=["A mug on a table"]):
            ok, rows = planner.make_plan("1 square photo and 1 photo with vertical format")
        self.assertTrue(ok)
        self.assertEqual([row[1] for row in rows], ["Square 1:1", "Portrait 9:16"])

    def test_explicit_polish_media_formats_stay_scoped_to_their_kind(self):
        self.assertEqual(planner._explicit_media_formats("1 kwadratowe zdjęcie i 1 poziomy film"),
                         {"image": "Square 1:1", "video": "Landscape 16:9"})
        self.assertEqual(planner._explicit_media_formats("1 photo in 1:1 format; 1 video in 16:9 format"),
                         {"image": "Square 1:1", "video": "Landscape 16:9"})
        self.assertEqual(planner._auto_format("Landscape 16:9", "A woman", "1 square photo and 1 landscape video"),
                         "Landscape 16:9")

    def test_runtime_hash_mismatch_never_writes_executables(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "runtime"
            response = MagicMock()
            response.__enter__.return_value = response
            payload = b"invalid official archive"
            response.iter_content.return_value = [payload]
            with patch.object(model_manager, "ensure", return_value=None), \
                    patch.object(llm.requests, "get", return_value=response), \
                    patch.object(llm, "_RUNTIME_SIZE", len(payload)), \
                    patch.object(llm, "BIN_DIR", folder), \
                    patch.object(llm, "_EXE", folder / "llama-completion.exe"), \
                    patch.object(llm, "_VISION_EXE", folder / "llama-mtmd-cli.exe"):
                error = llm.ensure()
            self.assertIn("SHA-256", error)
            self.assertFalse(folder.exists())

    def test_complete_installed_runtime_is_preserved_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exe, vision = folder / "llama-completion.exe", folder / "llama-mtmd-cli.exe"
            exe.write_bytes(b"existing runtime")
            vision.write_bytes(b"existing vision runtime")
            with patch.object(model_manager, "ensure", return_value=None), \
                    patch.object(llm.requests, "get") as get, \
                    patch.object(llm, "_EXE", exe), patch.object(llm, "_VISION_EXE", vision):
                self.assertIsNone(llm.ensure())
            get.assert_not_called()
            self.assertEqual(exe.read_bytes(), b"existing runtime")
            self.assertEqual(vision.read_bytes(), b"existing vision runtime")

    def test_coverage_distinguishes_objects_from_people_and_fails_closed(self):
        import json
        no_person = {"people": "none", "chest": "not applicable", "hips": "not applicable",
                     "hands": "not applicable"}
        with patch.object(llm, "_vision", return_value=json.dumps(no_person)):
            self.assertEqual(llm.coverage("mug.png"), no_person)
        for invalid in ({**no_person, "people": "uncertain"},
                        {**no_person, "people": True},
                        {**no_person, "chest": []},
                        {**no_person, "hips": None},
                        {**no_person, "hands": ""},
                        {**no_person, "chest": "bare anatomy"},
                        {k: v for k, v in no_person.items() if k != "people"}):
            with self.subTest(invalid=invalid), \
                    patch.object(llm, "_vision", return_value=json.dumps(invalid)):
                self.assertIsNone(llm.coverage("mug.png"))

    def test_first_vision_request_sees_projector_installed_by_ensure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            photo = folder / "photo.png"
            Image.new("RGB", (8, 8), "red").save(photo)
            bundle = {"files": [{"filename": n, "dest": "llm"}
                                 for n in ("aligned.gguf", "aligned-mmproj.gguf")]}

            def install():
                for name in ("aligned.gguf", "aligned-mmproj.gguf", "vision.exe"):
                    (folder / name).touch()

            with patch.object(llm, "ensure", side_effect=install), \
                    patch.object(config, "get_model", return_value=bundle), \
                    patch.object(config, "destination_dir", return_value=folder), \
                    patch.object(llm, "BIN_DIR", folder), \
                    patch.object(llm, "_VISION_EXE", folder / "vision.exe"), \
                    patch("app.comfy_client.ComfyClient") as client, \
                    patch.object(llm.subprocess, "run", return_value=SimpleNamespace(
                        returncode=0, stdout="fresh vision response")) as run:
                client.return_value.system_stats.return_value = None
                self.assertEqual(llm._vision(str(photo), "Describe it", 10), "fresh vision response")
                self.assertIn(str(folder / "aligned-mmproj.gguf"), run.call_args.args[0])

    def test_training_photo_requires_strict_model_response_types(self):
        import json
        valid = {"person": True, "age": 25, "gender": "woman", "caption": "Wearing a coat."}
        for field, value in (("person", "false"), ("age", True), ("age", -1),
                             ("caption", []), ("caption", ""), ("gender", "unknown")):
            with self.subTest(field=field, value=value), \
                    patch.object(llm, "_vision", return_value=json.dumps({**valid, field: value})):
                self.assertIsNone(llm.check_photo("unused.png"))
        with patch.object(llm, "_vision", return_value=json.dumps({**valid, "age": 16})):
            self.assertEqual(llm.check_photo("unused.png")["age"], 16)

    def test_empty_speech_batch_is_rejected_before_loading_model(self):
        with patch.object(tts, "available") as available:
            self.assertFalse(tts.synthesize_many([])[0])
        available.assert_not_called()

    def test_english_reference_ordinals_are_normalized_without_changing_speech(self):
        self.assertEqual(llm.mark_refs("the woman from the second photo holds image 1", 2),
                         "the woman [2] holds [1]")

    def test_catalog_selection_never_chooses_old_custom_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("old-abliterated.gguf", "aligned.gguf", "aligned-mmproj.gguf"):
                (Path(tmp) / name).write_bytes(b"test")
            bundle = {"files": [{"filename": n, "dest": "llm"}
                                 for n in ("aligned.gguf", "aligned-mmproj.gguf")]}
            with patch.object(config, "get_model", return_value=bundle), \
                    patch.object(config, "destination_dir", return_value=Path(tmp)):
                self.assertEqual(llm._model_file(), "aligned.gguf")
                self.assertEqual(llm._catalog_file(True), "aligned-mmproj.gguf")

    def test_speech_language_is_preserved_by_prompt_instruction(self):
        prompt = 'A woman says „Dzień dobry, świecie!” in a cafe.'
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(llm, "ask", return_value=(True, prompt)) as ask:
            ok, text = llm.write_prompt('She says „Dzień dobry, świecie!”', "video")
        self.assertTrue(ok)
        self.assertIn("Dzień dobry, świecie!", text)
        self.assertIn("language supplied by the user", ask.call_args.args[0])
        self.assertNotIn("in Polish", ask.call_args.args[1])
        self.assertTrue(llm.english(prompt))

    def test_legacy_plan_video_kind_and_aspect_still_dispatch_as_video(self):
        with patch.object(planner.queue_manager, "add") as add, \
                patch.object(planner.characters, "tag", return_value=""):
            added, skipped = planner.enqueue([["film", "Pionowo 9:16", 5, "nie",
                                               "A landscape with gentle clouds"]])
        self.assertEqual((added, skipped), (1, 0))
        self.assertEqual(add.call_args.args[0], "wideo")
        self.assertEqual(add.call_args.args[2]["aspect"], "Portrait 9:16")
        self.assertFalse(add.call_args.args[2]["sound"])


class ArchiveImports(unittest.TestCase):
    def test_zip_paths_cannot_escape_and_valid_images_are_preserved(self):
        data = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(data, format="PNG")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "photos.zip"
            dest = Path(tmp) / "photos"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../../escape.png", data.getvalue())
            files = char_gen.photos_from_zip(str(archive), dest)
            self.assertEqual(files, [str(dest / "zip_000.png")])
            self.assertFalse((Path(tmp) / "escape.png").exists())

    def test_invalid_later_image_rolls_back_extracted_images(self):
        data = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(data, format="PNG")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "photos.zip"
            dest = Path(tmp) / "photos"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("good.png", data.getvalue())
                z.writestr("bad.png", b"not an image")
            with self.assertRaises(Exception):
                char_gen.photos_from_zip(str(archive), dest)
            self.assertEqual(list(dest.iterdir()), [])
