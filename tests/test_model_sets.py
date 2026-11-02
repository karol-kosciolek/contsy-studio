# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Full and Lite model sets: catalog pins, file selection, workflows and downloads."""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config, download_models, image_gen, llm, lora_train, model_manager, refs, templates, video_gen


def _install(root: Path, model_ids):
    """Create empty files for every catalog file of the given entries."""
    for model_id in model_ids:
        for spec in config.get_model(model_id)["files"]:
            target = root / spec["dest"] / Path(spec.get("save_as") or spec["filename"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")


class _BothSetsInstalled(unittest.TestCase):
    """Full and Lite files side by side, as after switching sets on one PC."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        _install(self.root, [m["id"] for m in config.CATALOG["models"]])
        for target in (patch.object(config, "destination_dir", lambda dest: self.root / dest),
                       patch.object(model_manager.prefs, "load", lambda: {"tier": self.tier})):
            target.start()
            self.addCleanup(target.stop)
        self.tier = "full"


class CatalogTests(unittest.TestCase):
    def test_every_lite_entry_is_a_pinned_variant_of_a_full_entry(self):
        ids = {m["id"] for m in config.CATALOG["models"]}
        lite = [m for m in config.CATALOG["models"] if m["id"].endswith("-lite")]
        self.assertEqual({m["id"] for m in lite}, {"z-image-turbo-lite", "wan22-a14b-t2v-lite",
                                                    "wan22-a14b-i2v-lite", "qwen-image-edit-2511-lite",
                                                    "prompt-llm-lite"})
        for model in lite:
            self.assertIn(model["id"][:-5], ids)
            for spec in model["files"]:
                self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$", spec["filename"])
                self.assertGreater(spec["size_bytes"], 0)
                if spec["size_bytes"] > 10_000_000:
                    self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$", spec["filename"])

    def test_lite_entries_are_smaller_than_the_full_ones(self):
        for model in config.CATALOG["models"]:
            if model["id"].endswith("-lite"):
                full = config.get_model(model["id"][:-5])
                self.assertLess(model["size_gb"], full["size_gb"], model["id"])


class TierTests(unittest.TestCase):
    def tearDown(self):
        model_manager._gpu_total_gb.cache_clear()

    def test_automatic_choice_follows_vram(self):
        with patch.object(model_manager.prefs, "load", return_value={"tier": ""}):
            for total, expected in ((8.0, "lite"), (12.0, "lite"), (16.0, "full"), (24.0, "full"), (0.0, "full")):
                model_manager._gpu_total_gb.cache_clear()
                with patch.object(model_manager.vram, "get_gpu_info",
                                  return_value=model_manager.vram.GpuInfo(available=total > 0, total_gb=total)):
                    self.assertEqual(model_manager.tier(), expected, total)

    def test_settings_override_the_automatic_choice(self):
        with patch.object(model_manager, "_gpu_total_gb", return_value=24.0), \
                patch.object(model_manager.prefs, "load", return_value={"tier": "lite"}):
            self.assertEqual(model_manager.tier(), "lite")
            self.assertEqual(model_manager.tiered("wan22-a14b-i2v"), "wan22-a14b-i2v-lite")
            self.assertEqual(model_manager.tiered("voxcpm2"), "voxcpm2")  # no Lite variant
            self.assertEqual(model_manager.tiered("wan22-a14b-i2v", "full"), "wan22-a14b-i2v")


class SelectionTests(_BothSetsInstalled):
    def test_video_uses_the_files_of_the_active_set(self):
        full = video_gen._models("i2v")
        self.assertIn("Q6_K", full["%UNET_HIGH%"])
        self.assertIn("umt5_xxl_fp8", full["%CLIP%"])
        self.tier = "lite"
        lite = video_gen._models("i2v")
        self.assertIn("HighNoise-Q4_K_M", lite["%UNET_HIGH%"])
        self.assertIn("LowNoise-Q4_K_M", lite["%UNET_LOW%"])
        self.assertTrue(lite["%CLIP%"].endswith(".gguf"))
        self.assertEqual(lite["%LORA_HIGH%"], full["%LORA_HIGH%"])
        self.assertTrue(all(lite.values()))

    def test_lite_video_workflow_loads_the_text_encoder_through_gguf(self):
        self.tier = "lite"
        graph = templates.build("wan14b_i2v", video_gen._models("i2v"))
        loaders = {n["class_type"] for n in graph.values()}
        self.assertIn("CLIPLoaderGGUF", loaders)
        self.assertNotIn("CLIPLoader", loaders)

    def test_image_model_and_encoder_follow_the_set(self):
        self.assertIn("bf16", image_gen.available_models()[0])
        self.tier = "lite"
        model = image_gen.available_models()[0]
        self.assertIn("int8", model)
        self.assertIn("fp8_mixed", image_gen.resolve_spec(model)["clip"])

    def test_reference_edit_uses_the_gguf_loader_in_lite(self):
        self.assertEqual(refs._workflow(["a.png"], "p", 64, 64, 1)["1"]["class_type"], "UNETLoader")
        self.tier = "lite"
        node = refs._workflow(["a.png"], "p", 64, 64, 1)["1"]
        self.assertEqual(node, {"class_type": "UnetLoaderGGUF",
                                "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"}})

    def test_assistant_uses_the_lighter_quantization(self):
        self.tier = "lite"
        self.assertEqual(llm._model_file(), "Qwen3.5-9B-Q4_K_M.gguf")
        self.assertEqual(llm._catalog_file(projector=True), "Qwen3.5-9B-mmproj-F16.gguf")

    def test_training_always_uses_the_full_image_weights(self):
        self.tier = "lite"
        dit, vae, te = lora_train._models()
        self.assertEqual((dit.name, vae.name, te.name),
                         ("z_image_turbo_bf16.safetensors", "ae.safetensors", "qwen_3_4b.safetensors"))

    def test_lite_training_keeps_more_blocks_in_ram(self):
        self.assertLess(lora_train.BLOCKS_TO_SWAP["full"], lora_train.BLOCKS_TO_SWAP["lite"])
        self.assertLessEqual(lora_train.BLOCKS_TO_SWAP["lite"], 28)  # musubi limit: 30 blocks - 2

    def test_ensure_maps_to_the_set_unless_exact(self):
        self.tier = "lite"
        with patch.object(model_manager, "is_installed", return_value=True) as installed:
            self.assertIsNone(model_manager.ensure(["z-image-turbo", "voxcpm2"]))
            self.assertEqual([c.args[0] for c in installed.call_args_list], ["z-image-turbo-lite", "voxcpm2"])
            installed.reset_mock()
            model_manager.ensure(["z-image-turbo"], exact=True)
            self.assertEqual(installed.call_args.args[0], "z-image-turbo")


class SmallGpuTests(unittest.TestCase):
    def test_small_gpus_keep_working_memory_and_upscale_in_shorter_chunks(self):
        from app import comfy_process, upscale
        for total, args in ((8.0, ["--reserve-vram", "2"]), (12.0, ["--reserve-vram", "2"]),
                            (16.0, []), (0.0, [])):
            with patch.object(model_manager, "_gpu_total_gb", return_value=total):
                self.assertEqual(comfy_process.small_gpu_args(), args, total)
                self.assertEqual(model_manager.small_gpu(), bool(args), total)
        self.assertLess(upscale._SMALL_CHUNK, upscale._CHUNK)
        self.assertEqual((upscale._SMALL_CHUNK - 1) % 4, 0)  # SeedVR2 chunks are 4n+1 frames


class SmallGpuTrainingTests(unittest.TestCase):
    def _first_command(self, total_gb):
        with tempfile.TemporaryDirectory() as tmp,                 patch.object(model_manager, "_gpu_total_gb", return_value=total_gb),                 patch.object(lora_train, "ready", return_value=None),                 patch.object(model_manager, "ensure", return_value=None),                 patch.object(lora_train, "_adapter", return_value=Path(tmp) / "adapter.safetensors"),                 patch.object(lora_train, "_run", return_value=False) as run:
            lora_train.train(Path(tmp) / "work", Path(tmp), 8, Path(tmp) / "out.safetensors")
        return run.call_args_list[0].args[0]

    def test_small_gpus_prepare_training_photos_on_the_cpu(self):
        self.assertEqual(self._first_command(8.0)[-2:], ["--device", "cpu"])
        self.assertNotIn("--device", self._first_command(16.0))


class DownloadBundleTests(unittest.TestCase):
    def test_lite_bundles_replace_entries_and_count_shared_files_once(self):
        full, lite = download_models.model_ids("all", "full"), download_models.model_ids("all", "lite")
        self.assertFalse([i for i in full if i.endswith("-lite")])
        self.assertIn("wan22-a14b-i2v-lite", lite)
        self.assertNotIn("wan22-a14b-i2v", lite)
        self.assertEqual(len(full), len(lite))
        self.assertLess(download_models.size_gb(lite), download_models.size_gb(full))
        both_wan = download_models.size_gb(["wan22-a14b-i2v", "wan22-a14b-t2v"])
        self.assertLess(both_wan, config.get_model("wan22-a14b-i2v")["size_gb"]
                        + config.get_model("wan22-a14b-t2v")["size_gb"])


if __name__ == "__main__":
    unittest.main()
