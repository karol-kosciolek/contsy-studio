# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Regresje awarii startu po przypadkowej instalacji PyTorcha CPU."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import comfy_process


class StartupTests(unittest.TestCase):
    def test_cpu_build_has_actionable_diagnosis_at_end_of_large_log(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "comfyui.log"
            log.write_text("wczesne logi\n" * 3000 +
                           "AssertionError: Torch not compiled with CUDA enabled\n",
                           encoding="utf-8")
            error = comfy_process._startup_error(log)
            self.assertIn("CPU-only, without CUDA", error)
            self.assertIn("install.bat", error)
            self.assertIn(str(log), error)

    def test_missing_log_preserves_generic_error_and_log_location(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "missing.log"
            error = comfy_process._startup_error(log)
            self.assertIn("exited unexpectedly", error)
            self.assertIn(str(log), error)

    def test_driver_error_is_not_misdiagnosed_as_cpu_build(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "comfyui.log"
            log.write_text("RuntimeError: Found no NVIDIA driver on your system.",
                           encoding="utf-8")
            error = comfy_process._startup_error(log)
            self.assertIn("NVIDIA driver", error)
            self.assertNotIn("CPU-only", error)

    def test_child_gets_constraints_and_failed_start_closes_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ComfyUI").mkdir()
            (root / "ComfyUI" / "main.py").touch()
            (root / "venv").mkdir()
            constraints = root / "venv" / "torch-constraints.txt"
            constraints.write_text("torch==2.14.1+cu130\n", encoding="utf-8")
            child = Mock()
            child.poll.return_value = 1
            with patch.object(comfy_process.config, "ROOT", root), \
                 patch.object(comfy_process.config, "COMFYUI_DIR", root / "ComfyUI"), \
                 patch.object(comfy_process.ComfyProcess, "is_up", return_value=False), \
                 patch.object(comfy_process.subprocess, "Popen", return_value=child) as popen, \
                 patch.object(comfy_process.atexit, "register"), \
                 patch.dict(os.environ, {}, clear=True):
                engine = comfy_process.ComfyProcess()
                with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                    engine.start()
                self.assertEqual(popen.call_args.kwargs["env"]["PIP_CONSTRAINT"],
                                 str(constraints))
                self.assertTrue(engine._log_file.closed)
                self.assertIsNone(engine.proc)
                self.assertNotIn("PIP_CONSTRAINT", os.environ)


if __name__ == "__main__":
    unittest.main()
