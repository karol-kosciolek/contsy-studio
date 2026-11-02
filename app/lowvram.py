# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Exact memory savers for running LatentSync on GPUs under 15 GB (used by lipsync_worker).

Measured with a 7 GB cap: the default run (DeepCache, 2 x 16 frames per pass) ran out of
memory; with both helpers below and VAE slicing it peaked at 5.1 GB and took about as long.
"""
from __future__ import annotations

import types


def chunk_feed_forward(unet, chunks: int = 4) -> None:
    """Run every position-wise feed-forward in slices of the token axis.

    The output is identical; the GEGLU intermediate (4x the hidden size) never peaks at once."""
    import torch
    from diffusers.models.attention import FeedForward

    for module in unet.modules():
        if isinstance(module, FeedForward):
            def chunked(hidden_states, *rest, _forward=module.forward, **kwargs):
                if hidden_states.dim() < 3 or hidden_states.shape[1] < chunks:
                    return _forward(hidden_states, *rest, **kwargs)
                return torch.cat([_forward(part, *rest, **kwargs)
                                  for part in hidden_states.chunk(chunks, dim=1)], dim=1)
            module.forward = chunked


def split_cfg(unet) -> None:
    """Run the classifier-free-guidance pair (silent audio, real audio) as two passes.

    The result is the same with half the activation memory. Do not combine with DeepCache:
    it would reuse the features of one pass for the other."""
    import torch

    forward = unet.forward

    def split(sample, timestep, encoder_hidden_states=None, **kwargs):
        if sample.shape[0] != 2 or encoder_hidden_states is None:
            return forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
        outs = [forward(s, timestep, encoder_hidden_states=e, **kwargs).sample
                for s, e in zip(sample.chunk(2), encoder_hidden_states.chunk(2))]
        return types.SimpleNamespace(sample=torch.cat(outs))

    unet.forward = split
