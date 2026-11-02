# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Small-GPU memory savers must not change results, and only small GPUs may use them."""
import types
import unittest
from unittest.mock import patch

from app import lipsync, model_manager

try:
    import torch
    from diffusers.models.attention import FeedForward
except ImportError:  # CI installs neither; the helpers only run inside the lip-sync worker
    torch = None


@unittest.skipIf(torch is None, "torch and diffusers are not installed")
class ExactnessTests(unittest.TestCase):
    def test_chunked_feed_forward_matches_the_original(self):
        from app import lowvram
        torch.manual_seed(0)
        model = torch.nn.Sequential(FeedForward(16, activation_fn="geglu"))
        x = torch.randn(3, 10, 16)
        expected = model(x)
        lowvram.chunk_feed_forward(model, chunks=4)
        self.assertTrue(torch.allclose(model(x), expected, atol=1e-6))

    def test_split_cfg_matches_one_batched_pass(self):
        from app import lowvram

        class Net(torch.nn.Module):
            def forward(self, sample, timestep, encoder_hidden_states=None):
                mixed = sample * timestep + encoder_hidden_states.mean() * sample.shape[0]
                return types.SimpleNamespace(sample=mixed + sample.sum(dim=0, keepdim=True) * 0)

        net = Net()
        sample, audio = torch.randn(2, 4, 3), torch.randn(6, 2)
        expected = torch.cat([net(s, 2.0, encoder_hidden_states=a).sample
                              for s, a in zip(sample.chunk(2), audio.chunk(2))])
        lowvram.split_cfg(net)
        self.assertTrue(torch.equal(net(sample, 2.0, encoder_hidden_states=audio).sample, expected))
        single = torch.randn(1, 4, 3)  # no guidance pair: passes straight through
        self.assertEqual(net(single, 2.0, encoder_hidden_states=audio[:3]).sample.shape, single.shape)


class WorkerFlagTests(unittest.TestCase):
    def _args(self, small):
        with patch.object(model_manager, "_gpu_total_gb", return_value=8.0 if small else 16.0), \
                patch.object(lipsync.subprocess, "Popen", side_effect=OSError("stop")) as popen:
            with self.assertRaises(OSError):
                lipsync.run("video.mp4", "voice.wav", "out.mp4")
        return popen.call_args.args[0]

    def test_only_small_gpus_run_lip_sync_in_low_vram_mode(self):
        self.assertIn("--low-vram", self._args(small=True))
        self.assertNotIn("--low-vram", self._args(small=False))


if __name__ == "__main__":
    unittest.main()
