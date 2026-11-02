# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""install.bat asks Lite or Full once and downloads every model of that set."""
import builtins
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app import download_models, model_manager, prefs, setup_windows


class ChooseTierTests(unittest.TestCase):
    def ask(self, reply, current="", recommended="full"):
        answers = iter([reply]) if reply is not None else iter(())

        def fake_input(prompt=""):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError

        with patch.object(builtins, "input", fake_input), \
                patch.object(prefs, "load", return_value={"tier": current}), \
                patch.object(model_manager, "default_tier", return_value=recommended), \
                redirect_stdout(io.StringIO()) as out:
            return setup_windows.choose_tier(), out.getvalue()

    def test_numbers_pick_a_set_and_enter_takes_the_default(self):
        self.assertEqual(self.ask("1")[0], "lite")
        self.assertEqual(self.ask("2")[0], "full")
        self.assertEqual(self.ask("", recommended="lite")[0], "lite")
        self.assertEqual(self.ask("x", recommended="full")[0], "full")

    def test_rerun_keeps_the_saved_set_and_no_console_uses_the_default(self):
        self.assertEqual(self.ask("", current="lite", recommended="full")[0], "lite")
        self.assertEqual(self.ask(None, recommended="lite")[0], "lite")

    def test_menu_names_both_sets_and_the_recommendation(self):
        _, out = self.ask("", recommended="lite")
        self.assertIn("1) Lite", out)
        self.assertIn("2) Full quality", out)
        self.assertIn("Recommended: Lite", out)

    def test_installer_argument_skips_the_question(self):
        with patch.object(builtins, "input", side_effect=AssertionError("asked")):
            self.assertEqual(setup_windows.choose_tier("full"), "full")


class InstallModelsTests(unittest.TestCase):
    def run_install(self, answer, missing_gb=50.0, free_gb=500.0, error=None):
        with patch.object(setup_windows, "_missing_gb", return_value=missing_gb), \
                patch.object(setup_windows.shutil, "disk_usage",
                             return_value=type("U", (), {"free": free_gb * 1e9})()), \
                patch.object(prefs, "save") as save, \
                patch.object(model_manager, "ensure", return_value=error) as ensure, \
                redirect_stdout(io.StringIO()):
            return setup_windows.install_models(answer), save, ensure

    def test_downloads_every_model_of_the_chosen_set_and_remembers_it(self):
        ok, save, ensure = self.run_install("lite")
        self.assertTrue(ok)
        save.assert_called_once_with(tier="lite")
        self.assertEqual(ensure.call_args.args[0], download_models.model_ids("all", "lite"))
        self.assertTrue(ensure.call_args.kwargs.get("exact"))

    def test_not_enough_disk_space_stops_before_downloading(self):
        ok, save, ensure = self.run_install("full", missing_gb=138, free_gb=100)
        self.assertFalse(ok)
        save.assert_not_called()
        ensure.assert_not_called()

    def test_download_errors_fail_the_installer(self):
        self.assertFalse(self.run_install("lite", error="Could not download X")[0])

    def test_skip_leaves_downloads_for_first_use(self):
        ok, save, ensure = self.run_install("skip")
        self.assertTrue(ok)
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
