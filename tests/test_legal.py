# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""License notices stay consistent: GPL-3.0-only for own code, SPDX headers, legal notices in the UI."""
import html
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from app import config, ui_settings

ROOT = Path(__file__).resolve().parents[1]
OWN_CODE = [*ROOT.glob("app/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("tools/*.py"), *ROOT.glob("*.bat")]
DOCS = [*(ROOT / name for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "MODEL_LICENSES.md",
                                    "THIRD_PARTY_NOTICES.md", "NOTICE")), *ROOT.glob("docs/*.md")]


class LicenseFileTests(unittest.TestCase):
    def test_license_is_the_unmodified_gpl_text(self):
        text = (ROOT / "LICENSE").read_text("utf-8")
        self.assertTrue(text.lstrip().startswith("GNU GENERAL PUBLIC LICENSE"))
        self.assertIn("Version 3, 29 June 2007", text)
        self.assertEqual(text, (ROOT / "LICENSES" / "GPL-3.0-only.txt").read_text("utf-8"))
        self.assertEqual(app.__license__, "GPL-3.0-only")

    def test_own_files_carry_spdx_headers(self):
        self.assertTrue(OWN_CODE)
        for path in OWN_CODE:
            head = path.read_text("utf-8", errors="replace")[:600]
            with self.subTest(path=path.name):
                self.assertIn("SPDX-License-Identifier: GPL-3.0-only", head)
                self.assertIn("SPDX-FileCopyrightText:", head)

    def test_docs_use_the_current_name_and_license(self):
        for path in DOCS:
            text = path.read_text("utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("Contsy Media Studio", text)
                if path.name != "LICENSING.md":  # explains why earlier licenses were replaced
                    self.assertNotIn("Elastic", text)
                    self.assertNotIn("AGPL", text)


class AboutCardTests(unittest.TestCase):
    def about(self, url):
        with patch.dict(config.SETTINGS["app"], {"source_url": url}):
            return ui_settings.about_html()

    def test_shows_appropriate_legal_notices(self):
        text = self.about("")
        for needed in ("Contsy Studio", app.__version__, "© 2026 Karol Kosciolek", "(GPL-3.0-only)",
                       "no warranty", "LICENSE", "THIRD_PARTY_NOTICES.md", "Powered by Stability AI"):
            with self.subTest(needed=needed):
                self.assertIn(needed, text)
        self.assertNotIn("Affero", text)
        self.assertIn(html.escape(str(config.ROOT)), text)  # without a URL: the local source folder

    def test_links_the_configured_source_and_escapes_it(self):
        text = self.about("https://example.org/contsy?v=1&x=2")
        self.assertIn("href='https://example.org/contsy?v=1&amp;x=2'", text)
        self.assertNotIn(html.escape(str(config.ROOT)), text)


if __name__ == "__main__":
    unittest.main()
