"""Feed-forward chunking for LTX2VideoTransformer3DModel.

Diffusers' LTX2 transformer blocks do not support chunked feed-forward out of
the box.  This module monkey-patches the block's ``forward`` method so that the
video and audio FF layers can be processed in chunks along the sequence
dimension, trading a small amount of speed for significantly lower peak VRAM.

Usage::

    from frameartisan.modules.generation.graph.nodes.ff_chunking import (
        enable_forward_chunking,
        disable_forward_chunking,
    )

    enable_forward_chunking(transformer, chunk_size=256)
    # ... run inference ...
    disable_forward_chunking(transformer)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _chunked_feed_forward(
    ff: nn.Module,
    hidden_states: torch.Tensor,
    chunk_dim: int,
    chunk_size: int,
) -> torch.Tensor:
    """Apply *ff* in chunks of *chunk_size* along *chunk_dim*."""
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    if num_chunks <= 1:
        return ff(hidden_states)
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


def _make_patched_forward(original_forward, block):
    """Create a replacement ``forward`` that conditionally chunks the FF calls."""

    def patched_forward(*args, **kwargs):
        chunk_size = getattr(block, "_ff_chunk_size", None)
        if chunk_size is None:
            return original_forward(*args, **kwargs)

        chunk_dim = getattr(block, "_ff_chunk_dim", 1)

        # Run the original forward but intercept the FF calls.
        # We temporarily replace self.ff and self.audio_ff with chunked wrappers.
        orig_ff = block.ff
        orig_audio_ff = block.audio_ff

        class _ChunkedFF(nn.Module):
            def __init__(self, ff_module, c_dim, c_size):
                super().__init__()
                self._ff = ff_module
                self._c_dim = c_dim
                self._c_size = c_size

            def forward(self, x):
                return _chunked_feed_forward(self._ff, x, self._c_dim, self._c_size)

        block.ff = _ChunkedFF(orig_ff, chunk_dim, chunk_size)
        block.audio_ff = _ChunkedFF(orig_audio_ff, chunk_dim, chunk_size)
        try:
            result = original_forward(*args, **kwargs)
        finally:
            block.ff = orig_ff
            block.audio_ff = orig_audio_ff

        return result

    return patched_forward


def enable_forward_chunking(
    transformer,
    chunk_size: int | None = None,
    dim: int = 1,
) -> None:
    """Enable chunked feed-forward on all ``LTX2VideoTransformerBlock`` instances.

    Args:
        transformer: An ``LTX2VideoTransformer3DModel`` (or any model with
            ``transformer_blocks`` containing blocks that have ``.ff`` and
            ``.audio_ff`` attributes).
        chunk_size: Number of tokens per chunk along *dim*.  ``None`` or ``0``
            disables chunking (same as calling :func:`disable_forward_chunking`).
        dim: The dimension to chunk along (default ``1`` = sequence dim for
            shape ``[B, seq_len, hidden_dim]``).
    """
    chunk_size = chunk_size or 0
    blocks = getattr(transformer, "transformer_blocks", [])
    for block in blocks:
        if not hasattr(block, "ff") or not hasattr(block, "audio_ff"):
            continue

        block._ff_chunk_size = chunk_size if chunk_size > 0 else None
        block._ff_chunk_dim = dim

        # Only patch once — store the original forward so we can restore it.
        if not hasattr(block, "_original_forward"):
            block._original_forward = block.forward
            block.forward = _make_patched_forward(block._original_forward, block)


def disable_forward_chunking(transformer) -> None:
    """Remove chunked feed-forward from all transformer blocks."""
    blocks = getattr(transformer, "transformer_blocks", [])
    for block in blocks:
        block._ff_chunk_size = None
        if hasattr(block, "_original_forward"):
            block.forward = block._original_forward
            del block._original_forward
