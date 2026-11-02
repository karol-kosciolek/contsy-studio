# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Contsy's changes to pinned extensions must reach fresh installs without clobbering local edits."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import setup_windows

FOLDER = 'ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation'


class OverlayTests(unittest.TestCase):
    def test_every_overlay_targets_a_pinned_repository_and_ships_its_files(self):
        pinned = {folder for folder, _, _ in setup_windows.REPOS}
        for folder, files in setup_windows.OVERLAYS.items():
            self.assertIn(folder, pinned)
            for relative in files:
                self.assertTrue((setup_windows.PATCHES / Path(folder).name / relative).is_file(), relative)

    def test_rife_overlay_adds_the_frame_rate_inputs_the_workflows_use(self):
        init = (setup_windows.PATCHES / 'ComfyUI-Frame-Interpolation/vfi_models/rife/__init__.py').read_text(encoding='utf-8')
        self.assertIn('from .frame_rate import frame_rate_plan', init)
        self.assertIn('"target_fps": ("INT"', init)
        workflow = (setup_windows.ROOT / 'workflows/wan14b_i2v.json').read_text(encoding='utf-8')
        self.assertIn('"source_fps": 16', workflow)

    def _fixture(self, tmp: Path):
        upstream = b'line one\nline two\n'
        patches = tmp / 'patches' / 'Ext'
        (patches / 'pkg').mkdir(parents=True)
        (patches / 'pkg/a.py').write_bytes(b'line one\npatched\n')
        (patches / 'pkg/new.py').write_bytes(b'new file\n')
        destination = tmp / 'root' / 'nodes' / 'Ext'
        (destination / 'pkg').mkdir(parents=True)
        (destination / 'pkg/a.py').write_bytes(upstream.replace(b'\n', b'\r\n'))  # CRLF checkout
        overlays = {'nodes/Ext': {'pkg/a.py': hashlib.sha256(upstream).hexdigest(), 'pkg/new.py': None}}
        return patches.parent, tmp / 'root', destination, overlays

    def test_applies_to_a_pinned_checkout_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches, root, destination, overlays = self._fixture(Path(tmp))
            with patch.object(setup_windows, 'PATCHES', patches), patch.object(setup_windows, 'ROOT', root), \
                    patch.dict(setup_windows.OVERLAYS, overlays, clear=True), patch('builtins.print'):
                self.assertEqual(setup_windows.overlay_state('nodes/Ext'), ['pkg/a.py', 'pkg/new.py'])
                setup_windows.apply_overlays('nodes/Ext', destination)
                self.assertEqual((destination / 'pkg/a.py').read_bytes(), b'line one\npatched\n')
                self.assertEqual((destination / 'pkg/new.py').read_bytes(), b'new file\n')
                (destination / 'pkg/a.py').write_bytes(b'line one\r\npatched\r\n')  # same change, CRLF
                setup_windows.apply_overlays('nodes/Ext', destination)
                self.assertEqual(setup_windows.overlay_state('nodes/Ext'), [])

    def test_other_local_edits_are_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches, root, destination, overlays = self._fixture(Path(tmp))
            (destination / 'pkg/a.py').write_bytes(b'someone else changed this\n')
            with patch.object(setup_windows, 'PATCHES', patches), patch.object(setup_windows, 'ROOT', root), \
                    patch.dict(setup_windows.OVERLAYS, overlays, clear=True):
                with self.assertRaises(RuntimeError):
                    setup_windows.apply_overlays('nodes/Ext', destination)
            self.assertEqual((destination / 'pkg/a.py').read_bytes(), b'someone else changed this\n')


if __name__ == '__main__':
    unittest.main()
