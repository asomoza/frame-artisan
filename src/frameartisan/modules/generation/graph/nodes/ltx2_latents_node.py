from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2LatentsNode(Node):
    PRIORITY = 0
    REQUIRED_INPUTS: ClassVar[list[str]] = [
        "transformer",
        "vae",
        "audio_vae",
        "num_frames",
        "height",
        "width",
        "frame_rate",
        "seed",
    ]
    OPTIONAL_INPUTS: ClassVar[list[str]] = [
        "image_latents",
        "conditioning_mask",
        "clean_latents",
        "clean_conditioning_mask",
        "clean_audio_latents",
        "audio_conditioning_mask",
        "concat_latents",
        "concat_positions",
        "concat_conditioning_mask",
        "keyframe_latents",
        "keyframe_positions",
        "keyframe_strengths",
        "keyframe_group_sizes",
        "keyframe_attention_scales",
    ]
    OUTPUTS: ClassVar[list[str]] = [
        "video_latents",
        "audio_latents",
        "video_coords",
        "audio_coords",
        "latent_num_frames",
        "latent_height",
        "latent_width",
        "audio_num_frames",
        "conditioning_mask",
        "clean_latents",
        "clean_audio_latents",
        "audio_conditioning_mask",
        "base_num_tokens",
        "keyframe_group_sizes",
        "keyframe_attention_scales",
    ]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    @staticmethod
    def _build_keyframe_strength_mask(
        strengths: list[float],
        group_sizes: list[int],
        device,
    ):
        """Build a [1, M, 1] mask with per-token strength values."""
        import torch

        parts = []
        for strength, size in zip(strengths, group_sizes):
            parts.append(torch.full((1, size, 1), strength, device=device, dtype=torch.float32))
        return torch.cat(parts, dim=1)

    def __call__(self):
        import torch
        from diffusers.utils.torch_utils import randn_tensor

        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_audio_latents, pack_latents

        transformer = self.transformer
        vae = self.vae
        audio_vae = self.audio_vae

        if transformer is None or vae is None or audio_vae is None:
            raise NodeError(
                "transformer, vae, and audio_vae must be connected",
                self.__class__.__name__,
            )

        num_frames = int(self.num_frames)
        height = int(self.height)
        width = int(self.width)
        frame_rate = float(self.frame_rate)
        seed = int(self.seed)

        # LTX2 constraint: num_frames must satisfy 8n+1
        num_frames = 8 * ((num_frames - 1) // 8) + 1

        device = self.device

        # VAE compression ratios
        vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 32)
        vae_temporal_ratio = getattr(vae, "temporal_compression_ratio", 8)

        # Audio VAE properties
        audio_sample_rate = getattr(audio_vae.config, "sample_rate", 16000)
        audio_hop_length = getattr(audio_vae.config, "mel_hop_length", 160)
        audio_temporal_ratio = getattr(audio_vae, "temporal_compression_ratio", 4)
        audio_mel_ratio = getattr(audio_vae, "mel_compression_ratio", 4)
        num_mel_bins = getattr(audio_vae.config, "mel_bins", 64)
        num_channels_audio = getattr(audio_vae.config, "latent_channels", 8)

        # Latent dimensions
        latent_num_frames = (num_frames - 1) // vae_temporal_ratio + 1
        latent_height = height // vae_spatial_ratio
        latent_width = width // vae_spatial_ratio

        # Transformer config
        num_channels = transformer.config.in_channels
        patch_size = getattr(transformer.config, "patch_size", 1)
        patch_size_t = getattr(transformer.config, "patch_size_t", 1)

        generator = torch.Generator(str(device)).manual_seed(seed)

        # Generate and pack video latents
        video_shape = (1, num_channels, latent_num_frames, latent_height, latent_width)
        video_latents = randn_tensor(video_shape, generator=generator, device=device, dtype=torch.float32)
        video_latents = pack_latents(video_latents, patch_size, patch_size_t)

        # --- Generalized visual conditioning (new path) ---
        clean_latents_5d = self.clean_latents  # 5D [B, C, F, H, W] or None
        clean_conditioning_mask_5d = self.clean_conditioning_mask  # 5D [B, 1, F, H, W] or None

        # --- Legacy I2V conditioning (backward compat) ---
        image_latents = self.image_latents  # packed [B, seq, feat] or None
        legacy_conditioning_mask = self.conditioning_mask  # packed [B, seq] or None

        # Packed outputs for downstream nodes
        packed_clean_latents = None
        conditioning_mask = None

        if clean_latents_5d is not None and clean_conditioning_mask_5d is not None:
            # New generalized path: pack 5D → 3D and blend
            packed_clean = pack_latents(
                clean_latents_5d.to(device=device, dtype=video_latents.dtype), patch_size, patch_size_t
            )
            packed_mask = pack_latents(
                clean_conditioning_mask_5d.to(device=device, dtype=video_latents.dtype), patch_size, patch_size_t
            )
            conditioning_mask = packed_mask.squeeze(-1)  # [B, seq]
            packed_clean_latents = packed_clean

            mask_3d = conditioning_mask.unsqueeze(-1)  # [B, seq, 1]
            video_latents = packed_clean * mask_3d + video_latents * (1 - mask_3d)
        elif image_latents is not None and legacy_conditioning_mask is not None:
            # Legacy path: pre-packed image_latents from old LTX2ImageEncodeNode
            conditioning_mask = legacy_conditioning_mask
            conditioning_mask_expanded = legacy_conditioning_mask.unsqueeze(-1)  # [B, seq_len, 1]
            image_latents = image_latents.to(device=device, dtype=video_latents.dtype)
            conditioning_mask_expanded = conditioning_mask_expanded.to(device=device, dtype=video_latents.dtype)
            video_latents = image_latents * conditioning_mask_expanded + video_latents * (
                1 - conditioning_mask_expanded
            )
            packed_clean_latents = image_latents

        # --- Concat conditioning (IC-LoRA style) ---
        concat_latents_input = self.concat_latents
        concat_positions_input = self.concat_positions
        concat_cond_mask_input = self.concat_conditioning_mask
        base_num_tokens = None

        if concat_latents_input is not None:
            base_num_tokens = video_latents.shape[1]

            concat_lat = concat_latents_input.to(device=device, dtype=video_latents.dtype)
            video_latents = torch.cat([video_latents, concat_lat], dim=1)

            # Extend conditioning_mask with concat denoise mask values
            concat_mask_vals = concat_cond_mask_input.to(device=device, dtype=torch.float32).squeeze(-1)
            if conditioning_mask is None:
                conditioning_mask = torch.zeros(1, base_num_tokens, device=device, dtype=torch.float32)
            conditioning_mask = torch.cat([conditioning_mask, concat_mask_vals], dim=1)

            # Extend clean_latents with concat tokens (they are the clean reference)
            if packed_clean_latents is None:
                packed_clean_latents = torch.zeros(
                    1, base_num_tokens, concat_lat.shape[-1], device=device, dtype=video_latents.dtype
                )
            packed_clean_latents = torch.cat([packed_clean_latents, concat_lat], dim=1)

        # --- Keyframe conditioning (original VideoConditionByKeyframeIndex) ---
        keyframe_latents_input = self.keyframe_latents
        keyframe_positions_input = self.keyframe_positions
        keyframe_strengths_input = self.keyframe_strengths
        keyframe_group_sizes = self.keyframe_group_sizes

        if keyframe_latents_input is not None:
            if base_num_tokens is None:
                base_num_tokens = video_latents.shape[1]

            kf_lat = keyframe_latents_input.to(device=device, dtype=video_latents.dtype)

            # Blend keyframe tokens with noise based on per-group strength.
            # strength=1.0 → pure clean; strength=0.0 → pure noise.
            kf_noise = randn_tensor(kf_lat.shape, generator=generator, device=device, dtype=torch.float32)
            kf_mask = self._build_keyframe_strength_mask(
                keyframe_strengths_input, keyframe_group_sizes, device
            )  # [1, M, 1]
            kf_initial = kf_lat * kf_mask + kf_noise * (1 - kf_mask)
            video_latents = torch.cat([video_latents, kf_initial], dim=1)

            # Extend conditioning_mask: strength per token
            kf_cond_mask = kf_mask.squeeze(-1)  # [1, M]
            if conditioning_mask is None:
                conditioning_mask = torch.zeros(
                    1, base_num_tokens, device=device, dtype=torch.float32
                )
            conditioning_mask = torch.cat([conditioning_mask, kf_cond_mask], dim=1)

            # Extend clean_latents with keyframe tokens (clean reference for x0 blending)
            if packed_clean_latents is None:
                packed_clean_latents = torch.zeros(
                    1, base_num_tokens, kf_lat.shape[-1], device=device, dtype=video_latents.dtype
                )
            packed_clean_latents = torch.cat([packed_clean_latents, kf_lat], dim=1)

        # Audio latent dimensions
        duration_s = num_frames / frame_rate
        audio_latents_per_second = audio_sample_rate / audio_hop_length / float(audio_temporal_ratio)
        audio_num_frames = round(duration_s * audio_latents_per_second)
        latent_mel_bins = num_mel_bins // audio_mel_ratio

        # Audio conditioning: use clean audio latents if provided, else generate noise
        clean_audio_latents = self.clean_audio_latents
        audio_conditioning_mask = self.audio_conditioning_mask
        if clean_audio_latents is not None:
            clean_audio_latents = clean_audio_latents.to(device=device, dtype=torch.float32)
            if audio_conditioning_mask is not None:
                # Partial audio conditioning: blend clean with noise for unconditioned tokens
                audio_conditioning_mask = audio_conditioning_mask.to(device=device, dtype=torch.float32)
                audio_shape = (1, num_channels_audio, audio_num_frames, latent_mel_bins)
                audio_noise = randn_tensor(audio_shape, generator=generator, device=device, dtype=torch.float32)
                audio_noise = pack_audio_latents(audio_noise)
                acm_3d = audio_conditioning_mask.unsqueeze(-1)  # [1, seq, 1]
                audio_latents = clean_audio_latents * acm_3d + audio_noise * (1 - acm_3d)
            else:
                # Full audio conditioning: use clean latents directly
                audio_latents = clean_audio_latents
        else:
            # Generate and pack audio latents
            audio_shape = (1, num_channels_audio, audio_num_frames, latent_mel_bins)
            audio_latents = randn_tensor(audio_shape, generator=generator, device=device, dtype=torch.float32)
            audio_latents = pack_audio_latents(audio_latents)

        # RoPE coordinates
        video_coords = transformer.rope.prepare_video_coords(
            video_latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            video_latents.device,
            fps=frame_rate,
        )

        # Extend video_coords with concat position embeddings
        if concat_positions_input is not None:
            concat_pos = concat_positions_input.to(device=device, dtype=video_coords.dtype)
            # concat_pos is already [1, 3, N, 2] with [start, end) bounds
            video_coords = torch.cat([video_coords, concat_pos], dim=2)

        # Extend video_coords with keyframe position embeddings
        if keyframe_positions_input is not None:
            kf_pos = keyframe_positions_input.to(device=device, dtype=video_coords.dtype)
            # kf_pos is already [1, 3, N, 2] with [start, end) bounds
            video_coords = torch.cat([video_coords, kf_pos], dim=2)

        audio_coords = transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0],
            audio_num_frames,
            audio_latents.device,
        )

        logger.info(
            "Latents prepared: video %s, audio %s, coords video %s audio %s, base_tokens=%s, keyframe_groups=%s",
            video_latents.shape,
            audio_latents.shape,
            video_coords.shape,
            audio_coords.shape,
            base_num_tokens,
            keyframe_group_sizes,
        )

        # Move tensor outputs to CPU to free VRAM between stages.  The
        # downstream denoise node will .to(device) each tensor as needed.
        def _cpu(t):
            return t.cpu() if hasattr(t, "cpu") else t

        self.values = {
            "video_latents": _cpu(video_latents),
            "audio_latents": _cpu(audio_latents),
            "video_coords": _cpu(video_coords),
            "audio_coords": _cpu(audio_coords),
            "latent_num_frames": latent_num_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "audio_num_frames": audio_num_frames,
            "conditioning_mask": _cpu(conditioning_mask),
            "clean_latents": _cpu(packed_clean_latents),
            "clean_audio_latents": _cpu(clean_audio_latents),
            "audio_conditioning_mask": _cpu(audio_conditioning_mask),
            "base_num_tokens": base_num_tokens,
            "keyframe_group_sizes": keyframe_group_sizes,
            "keyframe_attention_scales": self.keyframe_attention_scales,
        }
        return self.values
