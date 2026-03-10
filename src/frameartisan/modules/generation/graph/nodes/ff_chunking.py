"""Feed-forward chunking for LTX2VideoTransformer3DModel.

Diffusers' LTX2 transformer blocks do not support chunked feed-forward out of
the box.  This module monkey-patches each FF module's ``forward`` so that the
inner ``net`` (Sequential of GELU → Dropout → Linear) is called in chunks along
the sequence dimension, trading a small amount of speed for lower peak VRAM.

The key is **in-place** write-back: each chunk's result is written directly into
the input tensor, avoiding any extra output allocation.

Usage::

    from frameartisan.modules.generation.graph.nodes.ff_chunking import (
        enable_forward_chunking,
        disable_forward_chunking,
    )

    enable_forward_chunking(transformer, num_chunks=2)
    # ... run inference ...
    disable_forward_chunking(transformer)
"""

from __future__ import annotations

import torch.nn as nn


def _apply_net(net: nn.ModuleList, x):
    """Run all layers in a FeedForward's ``net`` ModuleList sequentially."""
    for module in net:
        x = module(x)
    return x


def _chunked_ff_forward(self: nn.Module, x):
    """Replacement ``FeedForward.forward`` that processes ``self.net`` in chunks.

    Writes results back into *x* in-place so no extra output tensor is needed.
    """
    num_chunks: int = getattr(self, "_ff_num_chunks", 1)
    seq_len = x.shape[1]

    if num_chunks <= 1 or seq_len <= 1:
        return _apply_net(self.net, x)

    chunk_size = seq_len // num_chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else seq_len
        x[:, start:end] = _apply_net(self.net, x[:, start:end])
    return x


def enable_forward_chunking(
    transformer,
    num_chunks: int = 2,
) -> None:
    """Enable chunked feed-forward on all ``LTX2VideoTransformerBlock`` instances.

    Args:
        transformer: An ``LTX2VideoTransformer3DModel`` (or any model with
            ``transformer_blocks`` containing blocks that have ``.ff`` and
            ``.audio_ff`` attributes).
        num_chunks: Number of chunks to split the sequence into (default 2).
            ``1`` or less disables chunking.
    """
    blocks = getattr(transformer, "transformer_blocks", [])
    for block in blocks:
        for ff_attr in ("ff", "audio_ff"):
            ff = getattr(block, ff_attr, None)
            if ff is None:
                continue

            ff._ff_num_chunks = num_chunks

            if not hasattr(ff, "_original_forward"):
                ff._original_forward = ff.forward
                ff.forward = _chunked_ff_forward.__get__(ff, type(ff))


def disable_forward_chunking(transformer) -> None:
    """Remove chunked feed-forward from all transformer blocks."""
    blocks = getattr(transformer, "transformer_blocks", [])
    for block in blocks:
        for ff_attr in ("ff", "audio_ff"):
            ff = getattr(block, ff_attr, None)
            if ff is None:
                continue

            ff._ff_num_chunks = 1
            if hasattr(ff, "_original_forward"):
                ff.forward = ff._original_forward
                del ff._original_forward
