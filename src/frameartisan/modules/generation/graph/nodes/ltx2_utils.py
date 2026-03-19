from __future__ import annotations

import torch


def pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Pack video latents from [B, C, F, H, W] to [B, seq_len, features]."""
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    """Unpack video latents from [B, seq_len, features] to [B, C, F, H, W]."""
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack audio latents from [B, C, L, M] to [B, L, C*M]."""
    return latents.transpose(1, 2).flatten(2, 3)


def unpack_audio_latents(latents: torch.Tensor, num_frames: int, num_mel_bins: int) -> torch.Tensor:
    """Unpack audio latents from [B, L, D] to [B, C, L, M]."""
    return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


def normalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Normalize video latents across channel dimension [B, C, F, H, W]."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - latents_mean) * scaling_factor / latents_std


def denormalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Denormalize video latents across channel dimension [B, C, F, H, W]."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * latents_std / scaling_factor + latents_mean


def normalize_audio_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    """Normalize audio latents."""
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


def denormalize_audio_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    """Denormalize audio latents."""
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


def pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-layer masked normalization and flatten layers into channels.

    Args:
        text_hidden_states: [B, T, hidden_dim, num_layers]
        sequence_lengths: [B] number of non-padded tokens per sample
        padding_side: "left" or "right"
        scale_factor: normalization scale
        eps: epsilon for numerical stability

    Returns:
        [B, T, hidden_dim * num_layers] normalized and flattened embeddings
    """
    original_dtype = text_hidden_states.dtype

    batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape

    # Build per-token mask [B, T, 1, 1]
    token_indices = torch.arange(seq_len, device=text_hidden_states.device).unsqueeze(0)  # [1, T]
    if padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]

    # Masked mean over (T, hidden_dim) per sample per layer
    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

    # Masked min/max
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    # Normalize: (x - mean) / (max - min) * scale_factor
    normalized = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized = normalized * scale_factor

    # Flatten layers into feature dimension: [B, T, hidden_dim * num_layers]
    normalized = normalized.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
    normalized = normalized.masked_fill(~mask_flat, 0.0)

    return normalized.to(dtype=original_dtype)


def get_pixel_coords(
    latent_height: int,
    latent_width: int,
    latent_num_frames: int,
    frame_idx: int,
    fps: float,
    vae_spatial_scale: int = 32,
    vae_temporal_scale: int = 8,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate pixel coordinates for concatenated/keyframe conditioning tokens.

    Returns shape ``[1, 3, num_tokens, 2]`` with ``[start, end)`` bounds
    per ``(t, y, x)`` dimension, matching the format of
    ``prepare_video_coords``.  The RoPE takes the midpoint of each bound
    pair, so providing proper ``[start, end)`` ensures concat/keyframe
    tokens have the same positional alignment as base video tokens.
    """
    y_coords = torch.arange(latent_height, device=device)
    x_coords = torch.arange(latent_width, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    spatial_tokens = latent_height * latent_width
    yy_flat = yy.flatten()
    xx_flat = xx.flatten()

    frame_indices = torch.arange(latent_num_frames, device=device)
    frame_indices = frame_indices.repeat_interleave(spatial_tokens)
    frame_indices = frame_indices + frame_idx

    # Match prepare_video_coords temporal convention:
    # pixel_start = (latent_frame * scale + causal_offset - scale).clamp(0)
    # pixel_end   = ((latent_frame + 1) * scale + causal_offset - scale).clamp(0)
    causal_offset = 1
    t_start = (frame_indices.float() * vae_temporal_scale + causal_offset - vae_temporal_scale).clamp(min=0) / fps
    t_end = ((frame_indices.float() + 1) * vae_temporal_scale + causal_offset - vae_temporal_scale).clamp(min=0) / fps

    # Spatial: [start, start + scale) per latent pixel
    y_start = yy_flat.repeat(latent_num_frames).float() * vae_spatial_scale
    y_end = y_start + vae_spatial_scale
    x_start = xx_flat.repeat(latent_num_frames).float() * vae_spatial_scale
    x_end = x_start + vae_spatial_scale

    # Stack into [3, N, 2] then unsqueeze batch → [1, 3, N, 2]
    starts = torch.stack([t_start, y_start, x_start], dim=0)  # [3, N]
    ends = torch.stack([t_end, y_end, x_end], dim=0)  # [3, N]
    positions = torch.stack([starts, ends], dim=-1)  # [3, N, 2]
    positions = positions.unsqueeze(0)  # [1, 3, N, 2]
    return positions


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent sigma shift."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def build_keyframe_attention_mask(
    num_base_tokens: int,
    keyframe_group_sizes: list[int],
    attention_scales: list[float] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Build self-attention mask for keyframe conditioning.

    Follows the original LTX-2 ``build_attention_mask`` structure:
    * Base tokens (noisy + any concat tokens) fully attend to everything.
    * Each keyframe group fully attends to base tokens and itself.
    * Different keyframe groups do NOT attend to each other.

    Args:
        num_base_tokens: Number of noisy + concat tokens (attend to everything).
        keyframe_group_sizes: Number of tokens per keyframe group.
        attention_scales: Per-group attention scale in ``[0, 1]``.  Controls how
            strongly base tokens attend to each keyframe group (and vice versa).
            ``1.0`` = full attention, ``0.0`` = masked.  Converted to log-space
            additive bias: ``log(scale)`` so that SDPA softmax sees the correct
            weight.  ``None`` defaults to ``1.0`` for every group.
        dtype: Output dtype (must match model compute dtype for SDPA).
        device: Output device.

    Returns:
        ``(1, T, T)`` additive-bias tensor (0.0 = full attend, large
        negative = masked) ready for ``Attention.prepare_attention_mask``,
        or ``None`` when there are no keyframe groups.
    """
    if not keyframe_group_sizes:
        return None

    total = num_base_tokens + sum(keyframe_group_sizes)
    # Start with full attention (0.0 additive bias)
    mask = torch.zeros(1, total, total, dtype=dtype, device=device)

    neg_inf = -10000.0

    offset_i = num_base_tokens
    for i, size_i in enumerate(keyframe_group_sizes):
        # Per-group attention scale: log-space bias for base↔keyframe cross-attention
        if attention_scales is not None and attention_scales[i] < 1.0:
            scale = max(attention_scales[i], 1e-6)  # clamp to avoid log(0)
            import math

            bias = math.log(scale)
            # Base tokens attending to this keyframe group
            mask[:, :num_base_tokens, offset_i : offset_i + size_i] = bias
            # This keyframe group attending to base tokens
            mask[:, offset_i : offset_i + size_i, :num_base_tokens] = bias

        # Block out cross-attention between different keyframe groups
        offset_j = num_base_tokens
        for j, size_j in enumerate(keyframe_group_sizes):
            if i != j:
                mask[:, offset_i : offset_i + size_i, offset_j : offset_j + size_j] = neg_inf
            offset_j += size_j
        offset_i += size_i

    return mask


def vae_temporal_decode_streaming(
    vae,
    latents_cpu: torch.Tensor,
    *,
    device: torch.device,
    temb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode long video latents with lower peak VRAM by streaming temporal tiles to *device*.

    Mirrors the VAE's internal temporal tiled decode logic but avoids moving the
    full latent tensor to GPU at once.  Each tile is decoded independently and
    moved back to CPU before the next tile is processed, so peak GPU memory is
    proportional to a single tile rather than the entire video.

    Args:
        vae: An ``AutoencoderKLLTX2Video`` with tiling enabled.
        latents_cpu: Latents in ``[B, C, F, H, W]`` on **CPU**.
        device: The GPU device to decode on.
        temb: Optional timestep embedding passed to ``vae.decode``.

    Returns:
        Decoded video tensor on CPU.
    """
    tile_latent_min_num_frames = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio
    if latents_cpu.shape[2] <= tile_latent_min_num_frames:
        latents = latents_cpu.to(device=device, dtype=vae.dtype, non_blocking=True)
        return vae.decode(latents, temb=temb, return_dict=False)[0].cpu()

    num_frames = latents_cpu.shape[2]
    num_sample_frames = (num_frames - 1) * vae.temporal_compression_ratio + 1

    tile_latent_stride_num_frames = vae.tile_sample_stride_num_frames // vae.temporal_compression_ratio
    blend_num_frames = vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames

    result_tiles: list[torch.Tensor] = []
    prev_row_tile: torch.Tensor | None = None

    for i in range(0, num_frames, tile_latent_stride_num_frames):
        tile_cpu = latents_cpu[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
        tile = tile_cpu.to(device=device, dtype=vae.dtype, non_blocking=True)

        # Temporarily disable framewise decoding to prevent the VAE from
        # re-entering _temporal_tiled_decode on the already-sliced tile.
        saved_framewise = vae.use_framewise_decoding
        vae.use_framewise_decoding = False
        decoded = vae.decode(tile, temb=temb, return_dict=False)[0]
        vae.use_framewise_decoding = saved_framewise
        row_tile = decoded.cpu()

        if i > 0:
            row_tile = row_tile[:, :, :-1, :, :]

        if prev_row_tile is None:
            result_tiles.append(row_tile[:, :, : vae.tile_sample_stride_num_frames + 1, :, :])
        else:
            stitched = vae.blend_t(prev_row_tile, row_tile, blend_num_frames)
            stitched = stitched[:, :, : vae.tile_sample_stride_num_frames, :, :]
            result_tiles.append(stitched)

        prev_row_tile = row_tile
        del tile, decoded

    return torch.cat(result_tiles, dim=2)[:, :, :num_sample_frames]
