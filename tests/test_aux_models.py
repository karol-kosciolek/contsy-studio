# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Auxiliary downloads must verify bytes before exposing model files to workers."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app import aux_models
from app.downloader import DownloadError


class AuxiliaryModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.target = Path(self.tmp.name) / 'face.task'
        self.data = b'verified face model'
        self.spec = {'url':'https://storage.googleapis.com/pinned/model.task',
                     'size_bytes':len(self.data), 'sha256':hashlib.sha256(self.data).hexdigest()}

    def tearDown(self):
        self.tmp.cleanup()

    def response(self, data):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [data]
        return response

    def test_existing_valid_model_requires_no_network(self):
        self.target.write_bytes(self.data)
        with patch.object(aux_models.requests, 'get') as get:
            self.assertEqual(aux_models._verified_url(self.target, self.spec), self.target)
        get.assert_not_called()

    def test_existing_corruption_is_preserved_before_repair(self):
        self.target.write_bytes(b'x'*len(self.data))
        with patch.object(aux_models.requests, 'get', return_value=self.response(self.data)):
            aux_models._verified_url(self.target, self.spec)
        self.assertEqual(self.target.read_bytes(), self.data)
        saved = list(self.target.parent.glob('face.task.invalid_*'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), b'x'*len(self.data))

    def test_bad_remote_hash_never_publishes(self):
        with patch.object(aux_models.requests, 'get', return_value=self.response(b'x'*len(self.data))):
            with self.assertRaisesRegex(DownloadError, 'Integrity'):
                aux_models._verified_url(self.target, self.spec)
        self.assertFalse(self.target.exists())

    def test_excess_download_is_stopped_before_publication(self):
        with patch.object(aux_models.requests, 'get', return_value=self.response(self.data+b'extra')):
            with self.assertRaisesRegex(DownloadError, 'exceeded'):
                aux_models._verified_url(self.target, self.spec)
        self.assertFalse(self.target.exists())

    def test_clip_downloads_use_pinned_revision_and_integrity(self):
        with patch.object(aux_models, 'download_file') as download:
            aux_models.ensure_clip()
        self.assertEqual(download.call_count, 8)
        for call in download.call_args_list:
            self.assertEqual(call.kwargs['revision'], aux_models.CATALOG['clip']['revision'])
            self.assertEqual(len(call.kwargs['sha256']), 64)
            self.assertGreater(call.kwargs['expected_size'], 0)


if __name__ == '__main__':
    unittest.main()
