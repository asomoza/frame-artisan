"""Spatiotemporal masking for LoRA adapters in video generation.

Ported from image-artisan-z's lora_spatial_mask.py, extended with temporal
control for video. Patches PEFT LoRA layer forwards to multiply outputs by
a combined spatial + temporal mask.

Mask semantics (internal values after loading):
- 1.0: LoRA applies fully
- 0.0: LoRA is blocked (base model only)
- Intermediate values: Partial LoRA application

For RGBA mask images, the alpha channel determines the mask value
(painted pixels have alpha=255 → mask=1.0, transparent → mask=0.0).
The paint color itself is irrelevant.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# Global registry to track patched layers and their original forward methods
_PATCHED_LAYERS: dict[int, tuple[nn.Module, callable]] = {}


def build_temporal_weights(
    num_frames: int,
    start_frame: int,
    end_frame: int,
    fade_in_frames: int = 0,
    fade_out_frames: int = 0,
) -> torch.Tensor:
    """Build a per-frame weight vector for temporal LoRA masking.

    Args:
        num_frames: Total number of latent frames.
        start_frame: First frame where LoRA is active (inclusive).
        end_frame: Last frame where LoRA is active (inclusive). -1 means last frame.
        fade_in_frames: Number of frames to linearly ramp from 0 → 1 at start.
        fade_out_frames: Number of frames to linearly ramp from 1 → 0 at end.

    Returns:
        Tensor of shape [num_frames] with values in [0, 1].
    """
    if end_frame < 0:
        end_frame = num_frames - 1
    end_frame = min(end_frame, num_frames - 1)
    start_frame = max(0, min(start_frame, num_frames - 1))

    weights = torch.zeros(num_frames)

    if start_frame > end_frame:
        return weights

    # Fill active region with 1.0
    weights[start_frame : end_frame + 1] = 1.0

    # Apply fade-in ramp
    if fade_in_frames > 0:
        fade_end = min(start_frame + fade_in_frames, end_frame + 1)
        for i in range(start_frame, fade_end):
            weights[i] = (i - start_frame) / fade_in_frames

    # Apply fade-out ramp
    if fade_out_frames > 0:
        fade_start = max(end_frame - fade_out_frames + 1, start_frame)
        for i in range(fade_start, end_frame + 1):
            weights[i] = min(weights[i], (end_frame - i) / fade_out_frames)

    return weights


def build_combined_mask(
    spatial_mask: torch.Tensor | None,
    temporal_weights: torch.Tensor | None,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor | None:
    """Combine spatial and temporal masks into a packed token mask.

    Args:
        spatial_mask: [1, 1, H, W] spatial mask (any resolution, will be resized).
        temporal_weights: [F] per-frame weight vector.
        latent_num_frames: Number of latent frames.
        latent_height: Latent spatial height.
        latent_width: Latent spatial width.

    Returns:
        [1, F*H*W, 1] mask tensor, or None if both inputs are None.
    """
    if spatial_mask is None and temporal_weights is None:
        return None

    F_dim, H, W = latent_num_frames, latent_height, latent_width

    if spatial_mask is not None:
        # Resize spatial mask to latent dims: [1, 1, H, W]
        spatial = F.interpolate(
            spatial_mask.float(), size=(H, W), mode="bilinear", align_corners=False
        )  # [1, 1, H, W]
    else:
        spatial = torch.ones(1, 1, H, W)

    if temporal_weights is not None:
        # temporal_weights: [F] → [1, F, 1, 1]
        temporal = temporal_weights.float().view(1, F_dim, 1, 1)
    else:
        temporal = torch.ones(1, F_dim, 1, 1)

    # Combine: spatial [1, 1, H, W] → [1, 1, 1, H, W]; temporal [1, F, 1, 1] → [1, F, 1, 1]
    # Broadcast: [1, F, H, W]
    combined = spatial.squeeze(0) * temporal  # [1, F, H, W] (spatial is [1, H, W] after squeeze)
    combined = combined.view(1, F_dim * H * W, 1)  # [1, seq_len, 1]

    return combined


def load_spatial_mask(path: str) -> torch.Tensor | None:
    """Load a spatial mask image from file.

    Args:
        path: Path to a PNG/JPG mask image.

    Returns:
        Mask tensor [1, 1, H, W] with values in [0.0, 1.0], or None on failure.
    """
    if not path or not os.path.exists(path):
        return None

    try:
        from PIL import Image

        mask_img = Image.open(path)

        if mask_img.mode == "RGBA":
            mask_array = np.array(mask_img)[:, :, 3]
        elif mask_img.mode != "L":
            mask_img = mask_img.convert("L")
            mask_array = np.array(mask_img)
        else:
            mask_array = np.array(mask_img)

        mask_array = mask_array.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        logger.info("Loaded spatial mask from %s (shape=%s)", path, mask_tensor.shape)
        return mask_tensor

    except Exception as e:
        logger.error("Failed to load spatial mask from %s: %s", path, e)
        return None


def patch_lora_layer_with_mask(
    layer: nn.Module,
    combined_mask: torch.Tensor,
    latent_dims: tuple[int, int, int],
) -> None:
    """Patch a LoRA layer to apply a spatiotemporal mask during forward pass.

    Args:
        layer: The LoRA layer to patch.
        combined_mask: Pre-built mask tensor [1, seq_len, 1] where seq_len = F*H*W.
        latent_dims: (latent_num_frames, latent_height, latent_width).
    """
    layer_id = id(layer)
    if layer_id not in _PATCHED_LAYERS:
        original_forward = layer.forward
        _PATCHED_LAYERS[layer_id] = (layer, original_forward)
    else:
        _, original_forward = _PATCHED_LAYERS[layer_id]

    stored_mask = combined_mask
    stored_num_video_tokens = latent_dims[0] * latent_dims[1] * latent_dims[2]

    def masked_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        lora_output = original_forward(hidden_states)

        # Only mask 3D tensors [B, N, D] (spatial/temporal hidden states).
        # 2D tensors (adaLN conditioning) pass through unchanged.
        if hidden_states.ndim != 3:
            return lora_output

        B, N, D = hidden_states.shape

        mask = stored_mask.to(device=lora_output.device, dtype=lora_output.dtype)

        # Handle batch size mismatch (e.g., CFG doubles the batch)
        if B != mask.shape[0]:
            mask = mask.expand(B, -1, -1)

        # If N > num_video_tokens, this is joint attention (video + text tokens).
        # Only mask video tokens; text tokens get 1.0.
        if N > stored_num_video_tokens:
            full_mask = torch.ones(B, N, 1, device=lora_output.device, dtype=lora_output.dtype)
            full_mask[:, :stored_num_video_tokens, :] = mask
            return lora_output * full_mask
        elif N == stored_num_video_tokens:
            return lora_output * mask
        else:
            # N < expected video tokens — unknown layout, skip masking
            return lora_output

    layer.forward = masked_forward


def patch_lora_adapter_with_mask(
    transformer: nn.Module,
    adapter_name: str,
    combined_mask: torch.Tensor,
    latent_dims: tuple[int, int, int],
) -> int:
    """Patch all LoRA layers for a specific adapter with a spatiotemporal mask.

    Args:
        transformer: The transformer model with loaded LoRA adapters.
        adapter_name: Name of the LoRA adapter to patch.
        combined_mask: Pre-built mask [1, seq_len, 1].
        latent_dims: (latent_num_frames, latent_height, latent_width).

    Returns:
        Number of layers that were patched.
    """
    lora_layers = _get_lora_layers_for_adapter(transformer, adapter_name)

    if not lora_layers:
        logger.warning("No LoRA layers found for adapter '%s'. Skipping masking.", adapter_name)
        return 0

    patched_count = 0
    for layer_name, layer in lora_layers.items():
        try:
            patch_lora_layer_with_mask(layer, combined_mask, latent_dims)
            patched_count += 1
        except Exception as e:
            logger.error("Failed to patch layer %s: %s", layer_name, e)

    logger.info("Patched %d LoRA layers for adapter '%s' with mask", patched_count, adapter_name)
    return patched_count


def unpatch_all_lora_layers() -> int:
    """Restore all patched LoRA layers to their original state.

    Returns:
        Number of layers that were unpatched.
    """
    layer_ids = list(_PATCHED_LAYERS.keys())

    for layer_id in layer_ids:
        layer, original_forward = _PATCHED_LAYERS[layer_id]
        layer.forward = original_forward

    count = len(_PATCHED_LAYERS)
    _PATCHED_LAYERS.clear()

    if count > 0:
        logger.debug("Unpatched %d LoRA layers", count)

    return count


def get_patched_layer_count() -> int:
    """Get the number of currently patched layers."""
    return len(_PATCHED_LAYERS)


def _get_lora_layers_for_adapter(transformer: nn.Module, adapter_name: str) -> dict[str, nn.Module]:
    """Get all LoRA layers belonging to a specific adapter.

    Handles both test fixtures (with get_lora_layers method) and real PEFT models.
    """
    lora_layers = {}

    if hasattr(transformer, "get_lora_layers"):
        lora_layers = transformer.get_lora_layers(adapter_name)
    else:
        for name, module in transformer.named_modules():
            name_lower = name.lower()
            if "lora" in name_lower:
                if adapter_name.lower() in name_lower or "default" in name_lower:
                    if hasattr(module, "weight") or hasattr(module, "lora_A"):
                        lora_layers[name] = module

    return lora_layers
