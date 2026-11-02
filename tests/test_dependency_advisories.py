# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Known dependency advisories must stay unreachable until upstream lifts its version bounds.

See SECURITY.md. Remove each check once the patched version can be installed.
"""
import re
import unittest
from pathlib import Path

from PIL import Image

APP = Path(__file__).resolve().parents[1] / "app"


class DependencyAdvisoryTests(unittest.TestCase):
    def test_pillow_mic_decoder_is_unavailable(self):
        # GHSA-22hg-r758-rrw9 (Pillow < 13): infinite loop in the MIC decoder, which needs olefile.
        # The installer does not install olefile; Gradio still requires pillow<13.
        Image.init()
        self.assertNotIn("MIC", Image.OPEN)

    def test_app_never_imports_datasets_or_fsspec(self):
        # GHSA-379c-qx7v-6h59 (datasets < 5.0.1) and GHSA-27vj-qcqg-25rc (fsspec < 2026.6.0):
        # both packages are present only as VoxCPM/Hugging Face dependencies.
        pattern = re.compile(r"^\s*(?:import|from)\s+(?:datasets|fsspec)\b", re.M)
        for path in APP.glob("*.py"):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)


if __name__ == "__main__":
    unittest.main()
