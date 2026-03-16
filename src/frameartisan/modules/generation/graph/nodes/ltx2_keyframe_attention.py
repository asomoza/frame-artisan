"""Monkey-patch utilities for keyframe self-attention masking.

The diffusers ``LTX2VideoTransformerBlock`` does not pass an
``attention_mask`` to its video self-attention layer (``attn1``).
The original LTX-2 transformer **does** — it converts a ``[0, 1]``
mask into log-space additive bias and feeds it through SDPA.

These helpers install / remove a lightweight forward-wrapper on each
block's ``attn1`` so that a pre-built mask is injected into the
self-attention call without modifying the installed diffusers package.
"""

from __future__ import annotations

import torch


def install_keyframe_mask(transformer: torch.nn.Module, mask: torch.Tensor) -> list[tuple]:
    """Install self-attention mask hooks on all transformer blocks.

    Args:
        transformer: The ``LTX2VideoTransformer3DModel`` instance.
        mask: Additive-bias mask of shape ``(B, T, T)`` where ``B``
              matches the inference batch size (e.g. 2 for CFG).

    Returns:
        A list of ``(attn1_module, original_forward)`` pairs that must
        be passed to :func:`remove_keyframe_mask` for cleanup.
    """
    hooks: list[tuple] = []
    for block in transformer.transformer_blocks:
        orig = block.attn1.forward

        def _forward_with_mask(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            *,
            _orig=orig,
            _mask=mask,
            **kwargs,
        ):
            # Inject mask only for self-attention (no encoder_hidden_states)
            if encoder_hidden_states is None and _mask is not None:
                attention_mask = _mask
            return _orig(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        block.attn1.forward = _forward_with_mask
        hooks.append((block.attn1, orig))
    return hooks


def remove_keyframe_mask(hooks: list[tuple]) -> None:
    """Restore original forwards removed by :func:`install_keyframe_mask`."""
    for attn, orig in hooks:
        attn.forward = orig
