# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The bundle pre-download command must cover the catalog and report failures."""
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from app import config, download_models


class DownloadModelsTests(unittest.TestCase):
    def setUp(self):
        tier = patch.object(download_models.model_manager, "tier", return_value="full")
        tier.start()
        self.addCleanup(tier.stop)

    def test_bundles_resolve_to_catalog_models(self):
        catalog = {m["id"] for m in config.CATALOG["models"]}
        full = {i for i in catalog if not i.endswith("-lite")}
        for bundle in download_models.BUNDLES:
            for tier_name in ("full", "lite"):
                ids = download_models.model_ids(bundle, tier_name)
                self.assertTrue(ids, bundle)
                self.assertLessEqual(set(ids), catalog)
        self.assertEqual(set(download_models.model_ids("all")), full)

    def test_listing_downloads_nothing(self):
        with patch.object(download_models.model_manager, "ensure") as ensure, \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(download_models.main([]), 0)
        ensure.assert_not_called()
        self.assertIn("images", out.getvalue())
        self.assertIn("lite", out.getvalue())

    def test_download_reports_errors(self):
        with patch.object(download_models.model_manager, "ensure", return_value="Could not download X") as ensure, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(download_models.main(["images"]), 1)
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[0], download_models.model_ids("images"))
        self.assertIn("Could not download X", err.getvalue())

    def test_lite_flag_downloads_the_lite_entries(self):
        with patch.object(download_models.model_manager, "ensure", return_value=None) as ensure, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(download_models.main(["video", "--lite"]), 0)
        ids = ensure.call_args.args[0]
        self.assertIn("wan22-a14b-i2v-lite", ids)
        self.assertTrue(ensure.call_args.kwargs.get("exact"))


if __name__ == "__main__":
    unittest.main()
