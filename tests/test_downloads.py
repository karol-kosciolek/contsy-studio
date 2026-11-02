# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Real local HTTP tests for resumable, verified downloads and safe installation."""
import hashlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from app import config, downloader, model_manager, setup_windows

DATA = b"verified model payload" * 100


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                start = int(self.headers.get('Range', 'bytes=0-')[6:-1])
                if self.path == '/bad-range':
                    self.send_response(206)
                    self.send_header('Content-Range', f'bytes 0-{len(DATA)-1}/{len(DATA)}')
                elif start:
                    self.send_response(206)
                    self.send_header('Content-Range', f'bytes {start}-{len(DATA)-1}/{len(DATA)}')
                else:
                    self.send_response(200)
                self.send_header('Content-Length', str(len(DATA)-start))
                self.end_headers()
                self.wfile.write(DATA[start:])

            def log_message(self, *_):
                pass

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.tmp.cleanup()

    def download(self, **kwargs):
        with patch.object(downloader, '_resolve_url', return_value=self.url):
            return downloader.download_file('official/model', 'model.gguf', self.folder,
                                            expected_size=len(DATA),
                                            sha256=hashlib.sha256(DATA).hexdigest(), **kwargs)

    def test_download_verified_before_atomic_publication(self):
        result = self.download()
        self.assertEqual(result.read_bytes(), DATA)
        self.assertFalse((self.folder/'model.gguf.part').exists())

    def test_interrupted_download_resumes_from_exact_offset(self):
        (self.folder/'model.gguf.part').write_bytes(DATA[:123])
        self.assertEqual(self.download().read_bytes(), DATA)

    def test_bad_resume_header_preserves_original_partial(self):
        part = self.folder/'model.gguf.part'
        part.write_bytes(DATA[:123])
        with patch.object(downloader, '_resolve_url', return_value=self.url+'/bad-range'):
            with self.assertRaisesRegex(downloader.DownloadError, 'invalid resume range'):
                downloader.download_file('r/m', 'model.gguf', self.folder, expected_size=len(DATA))
        self.assertEqual(part.read_bytes(), DATA[:123])

    def test_corrupt_complete_partial_is_preserved_and_repaired(self):
        part = self.folder/'model.gguf.part'
        invalid = b'x'*len(DATA)
        part.write_bytes(invalid)
        self.assertEqual(self.download().read_bytes(), DATA)
        preserved = list(self.folder.glob('model.gguf.part.invalid_*'))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), invalid)

    def test_corrupt_existing_model_is_preserved_and_repaired(self):
        model = self.folder/'model.gguf'
        model.write_bytes(b'x'*len(DATA))
        self.assertEqual(self.download().read_bytes(), DATA)
        self.assertEqual(next(self.folder.glob('model.gguf.invalid_*')).read_bytes(), b'x'*len(DATA))

    def test_bad_network_hash_never_becomes_installed_model(self):
        with patch.object(downloader, '_resolve_url', return_value=self.url), self.assertRaisesRegex(downloader.DownloadError, 'Integrity'):
            downloader.download_file('official/model', 'model.gguf', self.folder,
                                     expected_size=len(DATA), sha256='0'*64)
        self.assertFalse((self.folder/'model.gguf').exists())
        self.assertEqual((self.folder/'model.gguf.part').read_bytes(), DATA)

    def test_oversized_partial_is_preserved_and_replaced_by_verified_bytes(self):
        part = self.folder/'model.gguf.part'
        part.write_bytes(DATA+b'extra')
        self.assertEqual(self.download().read_bytes(), DATA)
        self.assertEqual(next(self.folder.glob('model.gguf.part.invalid_*')).read_bytes(), DATA+b'extra')

    def test_path_traversal_and_windows_streams_refused(self):
        for name in ('../outside', '..\\outside', 'file:stream', 'C:\\bad', '',
                     'CON', 'NUL.gguf', 'bad.', 'bad '):
            with self.subTest(name=name), self.assertRaises(downloader.DownloadError):
                downloader._safe_name(name)

    def test_cancel_keeps_partial_without_publishing(self):
        with self.assertRaisesRegex(downloader.DownloadError, 'cancelled'):
            self.download(cancel_flag=lambda: True)
        self.assertFalse((self.folder/'model.gguf').exists())

    def test_partial_directory_is_preserved_without_writing_or_renaming(self):
        part = self.folder/'model.gguf.part'
        part.mkdir()
        (part/'user.txt').write_text('keep', encoding='utf-8')
        with self.assertRaisesRegex(downloader.DownloadError, 'regular file'):
            self.download()
        self.assertEqual((part/'user.txt').read_text(), 'keep')

    def test_unknown_model_is_actionable(self):
        self.assertIn('Unknown model', model_manager.ensure(['no-such-model']))

    def test_empty_model_is_not_installed(self):
        p = self.folder/'model.gguf'
        p.touch()
        model = {'files': [{'filename':'model.gguf','dest':'llm'}]}
        with patch.object(config, 'destination_dir', return_value=self.folder):
            self.assertFalse(model_manager.model_status(model).installed)

    def test_same_size_corrupt_model_is_not_installed(self):
        path = self.folder/'model.gguf'
        spec = {'filename': 'model.gguf', 'dest': 'llm', 'size_bytes': len(DATA),
                'sha256': hashlib.sha256(DATA).hexdigest()}
        with patch.object(config, 'destination_dir', return_value=self.folder):
            path.write_bytes(DATA)
            self.assertTrue(model_manager.model_status({'files': [spec]}).installed)
            path.write_bytes(b'x'*len(DATA))
            self.assertFalse(model_manager.model_status({'files': [spec]}).installed)
            self.assertEqual(path.read_bytes(), b'x'*len(DATA))

    def test_broken_environment_is_preserved_before_recreation(self):
        environment = self.folder/'venv'
        environment.mkdir()
        (environment/'keep.txt').write_text('existing data', encoding='utf-8')
        with patch.object(setup_windows, 'ROOT', self.folder), patch.object(setup_windows, 'run') as run:
            setup_windows.prepare_venv()
        run.assert_called_once()
        self.assertEqual(next(self.folder.glob('venv.preserved_*/keep.txt')).read_text(), 'existing data')

    def test_failed_new_checkout_never_occupies_final_installation_path(self):
        import subprocess
        with patch.object(setup_windows, 'ROOT', self.folder), patch.object(setup_windows, 'run', side_effect=subprocess.CalledProcessError(1, ['git'])):
            with self.assertRaises(subprocess.CalledProcessError):
                setup_windows.prepare_repository('engine', 'https://example.invalid/repo.git', 'revision')
        self.assertFalse((self.folder/'engine').exists())
        self.assertEqual(len(list(self.folder.glob('.engine.install_*'))), 1)

    def test_only_empty_matching_origin_checkout_is_resumed(self):
        from types import SimpleNamespace
        destination = self.folder/'engine'
        (destination/'.git').mkdir(parents=True)
        origin = 'https://example.invalid/repo.git'
        with patch.object(setup_windows, 'ROOT', self.folder), patch.object(setup_windows, 'run', return_value=SimpleNamespace(stdout=origin)) as run:
            setup_windows.prepare_repository('engine', origin, 'revision')
        self.assertEqual(run.call_count, 3)
        self.assertIn('fetch', run.call_args_list[1].args[0])
        self.assertIn('checkout', run.call_args_list[2].args[0])

    def test_existing_working_checkout_is_never_reset(self):
        destination = self.folder/'engine'
        destination.mkdir()
        (destination/'user.py').write_text('user changes', encoding='utf-8')
        with patch.object(setup_windows, 'ROOT', self.folder), patch.object(setup_windows, 'run') as run:
            setup_windows.prepare_repository('engine', 'https://example.invalid/repo.git', 'revision')
        run.assert_not_called()
        self.assertEqual((destination/'user.py').read_text(), 'user changes')

    def test_installer_stops_on_dependency_failure(self):
        import subprocess
        with patch.object(setup_windows, 'run', side_effect=subprocess.CalledProcessError(1, ['pip'])):
            with self.assertRaises(subprocess.CalledProcessError):
                setup_windows.install()

    def test_voxcpm_is_installed_without_funasr(self):
        from types import SimpleNamespace
        calls = []

        def fake_run(args, **kwargs):
            calls.append([str(a) for a in args])
            return SimpleNamespace(stdout='torch>=2.5.0\nfunasr\nlibrosa\npytest>=6.0; extra == "dev"\n')

        with patch.object(setup_windows, 'ROOT', self.folder), patch.object(setup_windows, 'run', side_effect=fake_run), \
                patch.object(setup_windows, 'prepare_repository', side_effect=lambda folder, url, rev: self.folder/folder), \
                patch.object(setup_windows, 'apply_overlays'), patch.object(setup_windows, 'check_environment', return_value=True), \
                patch.object(setup_windows.importlib.metadata, 'version', return_value='0'), \
                patch.dict(setup_windows.os.environ), patch.object(config, 'ensure_dirs'):
            setup_windows.install()
        self.assertIn(['--no-deps', 'voxcpm==2.0.3'], [c[-2:] for c in calls])
        deps = next(c for c in calls if 'requirements.txt' in c)
        self.assertIn('librosa', deps)
        self.assertIn('torch>=2.5.0', deps)
        self.assertFalse(any('funasr' in a or 'pytest' in a for a in deps))
        self.assertIn(['uninstall', '-y', 'funasr', 'kaldiio'], [c[-4:] for c in calls])

    def test_dependency_check_accepts_only_the_unused_funasr(self):
        from types import SimpleNamespace
        unused = 'voxcpm 2.0.3 requires funasr, which is not installed.\n'
        conflict = 'gradio 6.30.0 has requirement pillow<13, but you have pillow 13.0.0.'
        for output, expected in (('No broken requirements found.\n', []), (unused, []),
                                 (unused + conflict + '\n', [conflict])):
            with self.subTest(output=output), \
                    patch.object(setup_windows.subprocess, 'run', return_value=SimpleNamespace(stdout=output)):
                self.assertEqual(setup_windows.broken_requirements(), expected)


if __name__ == '__main__':
    unittest.main()
