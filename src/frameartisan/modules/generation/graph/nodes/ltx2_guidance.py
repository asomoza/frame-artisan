"""MultiModal Guidance helpers for LTX2 denoising.

Provides:
- ``calculate_guidance`` — combined CFG + STG + modality isolation + variance rescaling
- ``stg_perturbation_hooks`` — context manager that zeros self-attention in specified blocks
- ``modality_isolation_hooks`` — context manager that zeros cross-modal attention in all blocks

The hook-based approach uses ``register_forward_hook`` to zero out outputs
*without* replacing modules.  This is critical for compatibility with:
- ``torch.compile`` (Dynamo caches the module graph; swapping modules triggers
  re-tracing which fails with sdnq quantised weights on CPU)
- sequential group offload (offload hooks live on the original submodules;
  replacing them orphans those hooks)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


def _zero_output_hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> torch.Tensor:
    """Forward hook that replaces a module's output with zeros of the same shape."""
    return torch.zeros_like(output)


def calculate_guidance(
    cond_video: torch.Tensor,
    cond_audio: torch.Tensor,
    uncond_video: torch.Tensor | float,
    uncond_audio: torch.Tensor | float,
    perturbed_video: torch.Tensor | float,
    perturbed_audio: torch.Tensor | float,
    isolated_video: torch.Tensor | float,
    isolated_audio: torch.Tensor | float,
    cfg_scale: float,
    stg_scale: float,
    modality_scale: float,
    rescale_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply combined guidance formula per modality.

    Formula per modality::

        pred = cond + (cfg_scale - 1) * (cond - uncond)
                    + stg_scale * (cond - perturbed)
                    + (modality_scale - 1) * (cond - isolated)

    When a component is unused, pass ``0.0`` and the corresponding scale
    should be 0 (for stg) or 1.0 (for modality) so the term zeroes out.

    If *rescale_scale* > 0, variance rescaling is applied to bring the
    guided prediction's variance closer to the conditional prediction.
    """

    def _apply(
        cond: torch.Tensor,
        uncond: torch.Tensor | float,
        perturbed: torch.Tensor | float,
        isolated: torch.Tensor | float,
    ) -> torch.Tensor:
        pred = cond.clone()
        if cfg_scale != 1.0:
            pred = pred + (cfg_scale - 1.0) * (cond - uncond)
        if stg_scale != 0.0:
            pred = pred + stg_scale * (cond - perturbed)
        if modality_scale != 1.0:
            pred = pred + (modality_scale - 1.0) * (cond - isolated)

        if rescale_scale > 0.0:
            std_cond = cond.std(dim=list(range(1, cond.ndim)), keepdim=True)
            std_pred = pred.std(dim=list(range(1, pred.ndim)), keepdim=True)
            factor = std_cond / (std_pred + 1e-8)
            pred = rescale_scale * (pred * factor) + (1.0 - rescale_scale) * pred

        return pred

    video = _apply(cond_video, uncond_video, perturbed_video, isolated_video)
    audio = _apply(cond_audio, uncond_audio, perturbed_audio, isolated_audio)
    return video, audio


def parse_stg_blocks(stg_blocks_str: str) -> list[int]:
    """Parse a comma-separated string of block indices, e.g. ``"28,29"`` → ``[28, 29]``."""
    result: list[int] = []
    for part in stg_blocks_str.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result


@contextmanager
def stg_perturbation_hooks(transformer: nn.Module, stg_block_indices: list[int]) -> Generator[None, None, None]:
    """Temporarily zero self-attention output in specified transformer blocks.

    Registers forward hooks on ``block.attn1`` and ``block.audio_attn1``
    that replace their output with zeros.  The original modules are NOT
    replaced, preserving the module graph for torch.compile and offload hooks.
    """
    handles: list[RemovableHandle] = []

    try:
        for idx in stg_block_indices:
            block = transformer.transformer_blocks[idx]
            if hasattr(block, "attn1"):
                handles.append(block.attn1.register_forward_hook(_zero_output_hook))
            if hasattr(block, "audio_attn1"):
                handles.append(block.audio_attn1.register_forward_hook(_zero_output_hook))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def modality_isolation_hooks(transformer: nn.Module) -> Generator[None, None, None]:
    """Temporarily zero cross-modal attention output in ALL transformer blocks.

    Registers forward hooks on ``block.audio_to_video_attn`` and
    ``block.video_to_audio_attn`` that replace their output with zeros.
    """
    handles: list[RemovableHandle] = []

    try:
        for block in transformer.transformer_blocks:
            if hasattr(block, "audio_to_video_attn"):
                handles.append(block.audio_to_video_attn.register_forward_hook(_zero_output_hook))
            if hasattr(block, "video_to_audio_attn"):
                handles.append(block.video_to_audio_attn.register_forward_hook(_zero_output_hook))
        yield
    finally:
        for h in handles:
            h.remove()
