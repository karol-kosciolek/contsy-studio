# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Windows process-start serialization must prevent duplicate queue workers."""
import os
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch

from app import launcher


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows named mutex test')
    def test_other_process_cannot_acquire_startup_mutex(self):
        identity = 'test_' + uuid.uuid4().hex
        code = f'from app.launcher import _single_instance\nwith _single_instance({identity!r}) as owned: print(owned)'
        with launcher._single_instance(identity) as owned:
            self.assertTrue(owned)
            child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                   timeout=20, check=True)
            self.assertEqual(child.stdout.strip(), 'False')
        with launcher._single_instance(identity) as owned:
            self.assertTrue(owned)

    def test_second_start_never_creates_worker(self):
        from contextlib import nullcontext
        with patch.object(launcher, '_single_instance', return_value=nullcontext(False)), \
                patch.object(launcher, '_main') as start, \
                patch.object(launcher.webbrowser, 'open'):
            self.assertEqual(launcher.main(), 0)
        start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
