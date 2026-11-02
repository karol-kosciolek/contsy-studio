# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Reject settings that expose the local engine or override protected paths."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config


class LocalConfigTests(unittest.TestCase):
    def load(self, override):
        defaults = json.loads((config.CONFIG_DIR / 'settings.json').read_text('utf-8'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'settings.json').write_text(json.dumps(defaults), encoding='utf-8')
            (root/'settings.local.json').write_text(json.dumps(override), encoding='utf-8')
            with patch.object(config, 'CONFIG_DIR', root):
                return config.load_settings()

    def test_remote_bind_is_refused_for_both_services(self):
        for section in ('app', 'comfyui'):
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'local-only'):
                self.load({section:{'host':'0.0.0.0'}})

    def test_port_overrides_are_validated(self):
        for value in (True, 80, 65536, '7861'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'port'):
                self.load({'app':{'port':value}})

    def test_engine_args_cannot_bypass_local_file_policy(self):
        for args in (['--listen=0.0.0.0'], ['--output-directory','C:/'], ['--enable-cors-header','*']):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'override'):
                self.load({'comfyui':{'extra_args':args}})

    def test_valid_nondefault_ports_keep_other_settings(self):
        data = self.load({'app':{'port':7882}, 'comfyui':{'extra_args':['--preview-method','none']}})
        self.assertEqual(data['app']['port'], 7882)
        self.assertEqual(data['comfyui']['host'], '127.0.0.1')

    def test_source_url_is_empty_or_an_http_address(self):
        self.assertEqual(self.load({})['app']['source_url'], '')
        url = 'https://example.org/contsy/tree/v0.2.0'
        self.assertEqual(self.load({'app':{'source_url':url}})['app']['source_url'], url)
        for value in ('javascript:alert(1)', 'ftp://example.org/src', 'https://a b', "https://x'y", 42):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'source_url'):
                self.load({'app':{'source_url':value}})

    def test_services_cannot_share_the_same_port(self):
        with self.assertRaisesRegex(ValueError, 'different ports'):
            self.load({'app':{'port':8188}})


if __name__ == '__main__':
    unittest.main()
