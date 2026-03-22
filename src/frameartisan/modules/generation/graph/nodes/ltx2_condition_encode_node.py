from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2ConditionEncodeNode(Node):
    """VAE-encode source images and video clips into clean latents with a conditioning mask.

    Each condition specifies a pixel frame index, a strength, and a type
    (``"image"`` or ``"video"``).  The node produces 5-D unpacked
    ``clean_latents`` and ``conditioning_mask`` tensors that are packed
    downstream by the latents node.
    """

    PRIORITY = 0
    REQUIRED_INPUTS: ClassVar[list[str]] = ["vae", "num_frames", "height", "width", "frame_rate"]
    OPTIONAL_INPUTS: ClassVar[list[str]] = ["images", "videos", "reference_downscale_factor"]
    OUTPUTS: ClassVar[list[str]] = [
        "clean_latents",
        "conditioning_mask",
        "concat_latents",
        "concat_positions",
        "concat_conditioning_mask",
        "keyframe_latents",
        "keyframe_positions",
        "keyframe_strengths",
        "keyframe_group_sizes",
        "keyframe_attention_scales",
    ]
    SERIALIZE_INCLUDE: ClassVar[set[str]] = {
        "conditions", "keyframe_spatial_downscale", "keyframe_temporal_stride", "keyframe_speed",
    }

    def __init__(self):
        super().__init__()
        # Each entry: {"type": "image"|"video", "pixel_frame_index": int, "strength": float}
        self.conditions: list[dict] = []
        self.keyframe_spatial_downscale: int = 1
        self.keyframe_temporal_stride: int = 1
        self.keyframe_speed: int = 1

    def update_conditions(self, conditions: list[dict]) -> None:
        self.conditions = conditions
        self.set_updated()

    def __call__(self):
        import torch

        from frameartisan.app.model_manager import get_model_manager

        vae = self.vae
        num_frames = int(self.num_frames)
        height = int(self.height)
        width = int(self.width)

        if vae is None:
            raise NodeError("vae input is not connected", self.__class__.__name__)

        device = self.device
        mm = get_model_manager()

        # VAE compression ratios
        vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 32)
        vae_temporal_ratio = getattr(vae, "temporal_compression_ratio", 8)

        # Latent dimensions
        num_frames_snapped = 8 * ((num_frames - 1) // 8) + 1
        latent_num_frames = (num_frames_snapped - 1) // vae_temporal_ratio + 1
        latent_height = height // vae_spatial_ratio
        latent_width = width // vae_spatial_ratio

        # Latent channels from VAE config
        latent_channels = vae.config.latent_channels

        # Initialize outputs
        clean_latents = torch.zeros(
            1,
            latent_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=torch.float32,
        )
        conditioning_mask = torch.zeros(
            1,
            1,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=torch.float32,
        )

        # Resolve images input — may be a single image or a list of images
        raw_images = self.images
        if raw_images is None:
            raw_images = []
        elif not isinstance(raw_images, list):
            raw_images = [raw_images]

        # Resolve videos input — may be a single video or a list of videos
        raw_videos = self.videos
        if raw_videos is None:
            raw_videos = []
        elif not isinstance(raw_videos, list):
            raw_videos = [raw_videos]

        has_inputs = raw_images or raw_videos
        if not has_inputs or not self.conditions:
            logger.info(
                "No conditions/inputs provided, outputting empty clean_latents %s",
                clean_latents.shape,
            )
            self.values = {
                "clean_latents": clean_latents,
                "conditioning_mask": conditioning_mask,
                "concat_latents": None,
                "concat_positions": None,
                "concat_conditioning_mask": None,
                "keyframe_latents": None,
                "keyframe_positions": None,
                "keyframe_strengths": None,
                "keyframe_group_sizes": None,
                "keyframe_attention_scales": None,
            }
            return self.values

        latents_mean = vae.latents_mean
        latents_std = vae.latents_std
        frame_rate = float(self.frame_rate)
        ref_downscale = int(self.reference_downscale_factor) if self.reference_downscale_factor is not None else 1

        # First pass: replace and keyframe conditions.
        # Each condition has a "method" (images) or "mode" (videos) that controls routing:
        #   "replace"  → in-place replacement into clean_latents (hard constraint)
        #   "keyframe" → append as attention reference tokens (soft guidance)
        #   "auto"     → frame 0 uses replace, other frames use keyframe
        #   "concat"   → handled in second pass (IC-LoRA style)
        keyframe_tokens_list: list[torch.Tensor] = []
        keyframe_positions_list: list[torch.Tensor] = []
        keyframe_strengths_list: list[float] = []
        keyframe_attention_scales_list: list[float] = []
        keyframe_group_sizes: list[int] = []

        image_idx = 0
        video_idx = 0
        for i, condition in enumerate(self.conditions):
            cond_type = condition.get("type", "image")
            pixel_frame_index = int(condition.get("pixel_frame_index", 1))
            strength = float(condition.get("strength", 1.0))
            attention_scale = float(condition.get("attention_scale", 1.0))

            # Resolve the effective method for this condition
            if cond_type == "video":
                # Video conditions use "mode" field
                raw_method = condition.get("mode", "replace")
            else:
                # Image conditions use "method" field
                raw_method = condition.get("method", "auto")

            if raw_method == "concat":
                # Concat conditions handled in second pass
                if cond_type == "video":
                    video_idx += 1
                else:
                    image_idx += 1
                continue

            # Resolve "auto": frame 0 → replace, others → keyframe
            if raw_method == "auto":
                resolved_pfi = pixel_frame_index
                if resolved_pfi < 0:
                    resolved_pfi = num_frames_snapped + resolved_pfi + 1
                latent_idx = (resolved_pfi - 1) // vae_temporal_ratio
                latent_idx = max(0, min(latent_idx, latent_num_frames - 1))
                effective_method = "replace" if latent_idx == 0 else "keyframe"
            else:
                effective_method = raw_method  # "replace" or "keyframe"

            if cond_type == "video":
                if video_idx >= len(raw_videos):
                    video_idx += 1
                    continue
                source_start = int(condition.get("source_frame_start", 1))
                source_end = int(condition.get("source_frame_end", 0))
                video_clip = raw_videos[video_idx]
                if source_start >= 1 and source_end >= source_start:
                    video_clip = video_clip[source_start - 1 : source_end]

                if effective_method == "replace":
                    self._encode_video_condition(
                        video_clip,
                        pixel_frame_index,
                        strength,
                        vae,
                        mm,
                        device,
                        height,
                        width,
                        num_frames_snapped,
                        vae_temporal_ratio,
                        vae_spatial_ratio,
                        latent_num_frames,
                        latents_mean,
                        latents_std,
                        clean_latents,
                        conditioning_mask,
                        i,
                    )
                else:
                    result = self._encode_video_keyframe(
                        video_clip,
                        pixel_frame_index,
                        strength,
                        vae,
                        mm,
                        device,
                        height,
                        width,
                        num_frames_snapped,
                        vae_temporal_ratio,
                        vae_spatial_ratio,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        latents_mean,
                        latents_std,
                        frame_rate,
                        i,
                        spatial_downscale=self.keyframe_spatial_downscale,
                        temporal_stride=self.keyframe_temporal_stride,
                        speed=self.keyframe_speed,
                    )
                    if result is not None:
                        tokens, positions, num_tokens = result
                        keyframe_tokens_list.append(tokens)
                        keyframe_positions_list.append(positions)
                        keyframe_strengths_list.append(strength)
                        keyframe_attention_scales_list.append(attention_scale)
                        keyframe_group_sizes.append(num_tokens)
                video_idx += 1
            else:
                if image_idx >= len(raw_images):
                    image_idx += 1
                    continue

                if effective_method == "replace":
                    self._encode_image_condition(
                        raw_images[image_idx],
                        pixel_frame_index,
                        strength,
                        vae,
                        mm,
                        device,
                        height,
                        width,
                        num_frames_snapped,
                        vae_temporal_ratio,
                        latent_num_frames,
                        latents_mean,
                        latents_std,
                        clean_latents,
                        conditioning_mask,
                        i,
                    )
                else:
                    result = self._encode_image_keyframe(
                        raw_images[image_idx],
                        pixel_frame_index,
                        strength,
                        vae,
                        mm,
                        device,
                        height,
                        width,
                        num_frames_snapped,
                        vae_temporal_ratio,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        latents_mean,
                        latents_std,
                        frame_rate,
                        i,
                        spatial_downscale=self.keyframe_spatial_downscale,
                    )
                    if result is not None:
                        tokens, positions, num_tokens = result
                        keyframe_tokens_list.append(tokens)
                        keyframe_positions_list.append(positions)
                        keyframe_strengths_list.append(strength)
                        keyframe_attention_scales_list.append(attention_scale)
                        keyframe_group_sizes.append(num_tokens)
                image_idx += 1

        # Keyframe outputs
        keyframe_latents = None
        keyframe_positions = None
        keyframe_strengths = None
        keyframe_attention_scales = None
        if keyframe_tokens_list:
            keyframe_latents = torch.cat(keyframe_tokens_list, dim=1)
            keyframe_positions = torch.cat(keyframe_positions_list, dim=2)
            keyframe_strengths = keyframe_strengths_list
            keyframe_attention_scales = keyframe_attention_scales_list

        # Second pass: concat conditions (collect into lists, then cat)
        concat_tokens_list: list[torch.Tensor] = []
        concat_positions_list: list[torch.Tensor] = []
        concat_masks_list: list[torch.Tensor] = []

        image_idx = 0
        video_idx = 0
        for i, condition in enumerate(self.conditions):
            cond_type = condition.get("type", "image")
            mode = condition.get("mode", "replace")

            if cond_type == "video":
                if mode == "concat":
                    if video_idx < len(raw_videos):
                        pixel_frame_index = int(condition.get("pixel_frame_index", 1))
                        strength = float(condition.get("strength", 1.0))
                        downscale_factor = int(condition.get("downscale_factor", 0))
                        if downscale_factor <= 0:
                            downscale_factor = ref_downscale

                        # Source frame range: 1-based inclusive
                        source_start = int(condition.get("source_frame_start", 1))
                        source_end = int(condition.get("source_frame_end", 0))
                        video_clip = raw_videos[video_idx]
                        if source_start >= 1 and source_end >= source_start:
                            video_clip = video_clip[source_start - 1 : source_end]

                        result = self._encode_concat_video_condition(
                            video_clip,
                            pixel_frame_index,
                            strength,
                            downscale_factor,
                            vae,
                            mm,
                            device,
                            height,
                            width,
                            num_frames_snapped,
                            vae_temporal_ratio,
                            vae_spatial_ratio,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            latents_mean,
                            latents_std,
                            frame_rate,
                            i,
                        )
                        if result is not None:
                            tokens, positions, mask = result
                            concat_tokens_list.append(tokens)
                            concat_positions_list.append(positions)
                            concat_masks_list.append(mask)
                video_idx += 1
            else:
                image_idx += 1

        concat_latents = None
        concat_positions = None
        concat_conditioning_mask = None
        if concat_tokens_list:
            concat_latents = torch.cat(concat_tokens_list, dim=1)
            concat_positions = torch.cat(concat_positions_list, dim=2)
            concat_conditioning_mask = torch.cat(concat_masks_list, dim=1)

        logger.info(
            "Conditions encoded: clean_latents %s, conditioning_mask %s, concat_tokens=%s, keyframe_tokens=%s (%s groups)",
            clean_latents.shape,
            conditioning_mask.shape,
            concat_latents.shape if concat_latents is not None else None,
            keyframe_latents.shape if keyframe_latents is not None else None,
            len(keyframe_group_sizes) if keyframe_group_sizes else 0,
        )

        self.values = {
            "clean_latents": clean_latents,
            "conditioning_mask": conditioning_mask,
            "concat_latents": concat_latents,
            "concat_positions": concat_positions,
            "concat_conditioning_mask": concat_conditioning_mask,
            "keyframe_latents": keyframe_latents,
            "keyframe_positions": keyframe_positions,
            "keyframe_strengths": keyframe_strengths,
            "keyframe_group_sizes": keyframe_group_sizes if keyframe_group_sizes else None,
            "keyframe_attention_scales": keyframe_attention_scales,
        }
        return self.values

    @staticmethod
    def _encode_image_condition(
        image,
        pixel_frame_index: int,
        strength: float,
        vae,
        mm,
        device,
        height: int,
        width: int,
        num_frames_snapped: int,
        vae_temporal_ratio: int,
        latent_num_frames: int,
        latents_mean,
        latents_std,
        clean_latents,
        conditioning_mask,
        condition_index: int,
    ) -> None:
        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_utils import normalize_latents

        # Resolve pixel frame index → latent frame index
        if pixel_frame_index < 0:
            pixel_frame_index = num_frames_snapped + pixel_frame_index + 1
        latent_idx = (pixel_frame_index - 1) // vae_temporal_ratio
        latent_idx = max(0, min(latent_idx, latent_num_frames - 1))

        # Preprocess image: numpy HWC uint8 → torch [1, C, H, W] float32 in [-1, 1]
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor * 2.0 - 1.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        # Resize to target dimensions
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(height, width), mode="bilinear", align_corners=False
        )

        # Add temporal dimension: [1, C, 1, H, W]
        image_tensor = image_tensor.unsqueeze(2)

        # VAE encode
        with mm.use_components("vae", device=device, strategy_override="model_offload"):
            image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
            init_latents = vae.encode(image_tensor).latent_dist.mode()

        # Normalize latents (keep float32 per MEMORY.md pitfall)
        if latents_mean is not None and latents_std is not None:
            init_latents = normalize_latents(init_latents.float(), latents_mean, latents_std)

        # Place in clean_latents at the resolved frame index
        clean_latents[:, :, latent_idx : latent_idx + 1] = init_latents.cpu()
        conditioning_mask[:, :, latent_idx] = strength

        logger.info(
            "Condition %d (image): pixel_frame=%d → latent_idx=%d, strength=%.2f",
            condition_index,
            pixel_frame_index,
            latent_idx,
            strength,
        )

    @staticmethod
    def _encode_video_condition(
        video_frames,
        pixel_frame_index: int,
        strength: float,
        vae,
        mm,
        device,
        height: int,
        width: int,
        num_frames_snapped: int,
        vae_temporal_ratio: int,
        vae_spatial_ratio: int,
        latent_num_frames: int,
        latents_mean,
        latents_std,
        clean_latents,
        conditioning_mask,
        condition_index: int,
    ) -> None:
        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_utils import normalize_latents

        # video_frames: numpy [F, H, W, 3] uint8
        total_video_frames = video_frames.shape[0]

        # Resolve pixel frame index
        if pixel_frame_index < 0:
            pixel_frame_index = num_frames_snapped + pixel_frame_index + 1

        # Trim to fit within generation length
        available = num_frames_snapped - (pixel_frame_index - 1)
        available = max(available, 0)
        use_frames = min(total_video_frames, available)

        if use_frames <= 0:
            logger.warning(
                "Condition %d (video): no frames fit (start=%d, available=%d)",
                condition_index,
                pixel_frame_index,
                available,
            )
            return

        # Align to VAE temporal grid: 8n+1 — pad up by repeating last frame
        import math

        aligned_frames = (math.ceil((use_frames - 1) / vae_temporal_ratio)) * vae_temporal_ratio + 1
        if aligned_frames <= 0:
            aligned_frames = 1  # single frame fallback

        video_clip = video_frames[:use_frames]
        if aligned_frames > use_frames:
            pad_count = aligned_frames - use_frames
            last_frame = video_clip[-1:]
            video_clip = np.concatenate(
                [video_clip] + [last_frame] * pad_count,
                axis=0,
            )

        # Convert to tensor: [1, C, F, H, W] float32 in [-1, 1]
        video_tensor = torch.from_numpy(video_clip).float() / 255.0
        video_tensor = video_tensor * 2.0 - 1.0
        # [F, H, W, C] → [F, C, H, W]
        video_tensor = video_tensor.permute(0, 3, 1, 2)

        # Resize each frame to target dimensions
        video_tensor = torch.nn.functional.interpolate(
            video_tensor, size=(height, width), mode="bilinear", align_corners=False
        )

        # [F, C, H, W] → [1, C, F, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        # VAE encode
        with mm.use_components("vae", device=device, strategy_override="model_offload"):
            video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
            encoded = vae.encode(video_tensor).latent_dist.mode()

        # Normalize latents (keep float32)
        if latents_mean is not None and latents_std is not None:
            encoded = normalize_latents(encoded.float(), latents_mean, latents_std)

        encoded = encoded.cpu()

        # encoded shape: [1, latent_channels, F_latent, H_latent, W_latent]
        f_latent = encoded.shape[2]

        # Resolve start latent position
        latent_start = (pixel_frame_index - 1) // vae_temporal_ratio
        latent_start = max(0, min(latent_start, latent_num_frames - 1))

        # Clamp to not exceed latent dimensions
        latent_end = min(latent_start + f_latent, latent_num_frames)
        actual_latent_frames = latent_end - latent_start

        clean_latents[:, :, latent_start:latent_end] = encoded[:, :, :actual_latent_frames]
        conditioning_mask[:, :, latent_start:latent_end] = strength

        logger.info(
            "Condition %d (video): %d pixel frames → %d latent frames at latent[%d:%d], strength=%.2f",
            condition_index,
            aligned_frames,
            actual_latent_frames,
            latent_start,
            latent_end,
            strength,
        )

    @staticmethod
    def _encode_concat_video_condition(
        video_frames,
        pixel_frame_index: int,
        strength: float,
        downscale_factor: int,
        vae,
        mm,
        device,
        height: int,
        width: int,
        num_frames_snapped: int,
        vae_temporal_ratio: int,
        vae_spatial_ratio: int,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        latents_mean,
        latents_std,
        frame_rate: float,
        condition_index: int,
    ) -> tuple | None:
        """Encode a video condition for concat (IC-LoRA) mode.

        Returns ``(concat_tokens, positions, denoise_mask)`` or ``None`` if
        no frames fit.
        """
        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            get_pixel_coords,
            normalize_latents,
            pack_latents,
        )

        total_video_frames = video_frames.shape[0]

        if pixel_frame_index < 0:
            pixel_frame_index = num_frames_snapped + pixel_frame_index + 1

        available = num_frames_snapped - (pixel_frame_index - 1)
        available = max(available, 0)
        use_frames = min(total_video_frames, available)

        if use_frames <= 0:
            logger.warning(
                "Condition %d (concat video): no frames fit (start=%d, available=%d)",
                condition_index,
                pixel_frame_index,
                available,
            )
            return None

        # Align to VAE temporal grid: 8n+1 — pad up by repeating last frame
        import math

        aligned_frames = (math.ceil((use_frames - 1) / vae_temporal_ratio)) * vae_temporal_ratio + 1
        if aligned_frames <= 0:
            aligned_frames = 1

        video_clip = video_frames[:use_frames]
        if aligned_frames > use_frames:
            pad_count = aligned_frames - use_frames
            last_frame = video_clip[-1:]
            video_clip = np.concatenate(
                [video_clip] + [last_frame] * pad_count,
                axis=0,
            )

        # Downscale spatial dims for IC-LoRA: encode at lower resolution
        encode_height = height // downscale_factor
        encode_width = width // downscale_factor
        cond_latent_height = latent_height // downscale_factor
        cond_latent_width = latent_width // downscale_factor

        # Convert to tensor: [1, C, F, H, W] float32 in [-1, 1]
        video_tensor = torch.from_numpy(video_clip).float() / 255.0
        video_tensor = video_tensor * 2.0 - 1.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = torch.nn.functional.interpolate(
            video_tensor, size=(encode_height, encode_width), mode="bilinear", align_corners=False
        )
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        # VAE encode
        with mm.use_components("vae", device=device, strategy_override="model_offload"):
            video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
            encoded = vae.encode(video_tensor).latent_dist.mode()

        if latents_mean is not None and latents_std is not None:
            encoded = normalize_latents(encoded.float(), latents_mean, latents_std)

        encoded = encoded.cpu()

        # encoded: [1, C, F_latent, H_latent_down, W_latent_down]
        cond_latent_frames = encoded.shape[2]

        # Pack 5D → 3D: [1, num_tokens, features]
        concat_tokens = pack_latents(encoded, patch_size=1, patch_size_t=1)

        # Resolve latent start position
        latent_start = (pixel_frame_index - 1) // vae_temporal_ratio
        latent_start = max(0, min(latent_start, latent_num_frames - 1))

        # Generate position embeddings using downscaled latent dims
        positions = get_pixel_coords(
            latent_height=cond_latent_height,
            latent_width=cond_latent_width,
            latent_num_frames=cond_latent_frames,
            frame_idx=latent_start,
            fps=frame_rate,
            vae_spatial_scale=vae_spatial_ratio,
            vae_temporal_scale=vae_temporal_ratio,
        )

        # Scale spatial positions by downscale_factor so they map to the
        # correct pixel-space locations in the full-resolution output
        if downscale_factor > 1:
            positions[:, 1, :] = positions[:, 1, :] * downscale_factor
            positions[:, 2, :] = positions[:, 2, :] * downscale_factor

        # Create conditioning mask: strength per token, matching the reference
        # pipeline convention where 1.0 = fully conditioned (timestep=0, clean)
        # and 0.0 = unconditioned (full timestep, noisy).  The concat tokens
        # are clean VAE-encoded references so they must see timestep=0.
        num_tokens = concat_tokens.shape[1]
        conditioning_mask = torch.full(
            (1, num_tokens, 1),
            fill_value=strength,
            dtype=concat_tokens.dtype,
        )

        logger.info(
            "Condition %d (concat video): %d pixel frames → %d latent frames, %d tokens, strength=%.2f, downscale=%d",
            condition_index,
            aligned_frames,
            cond_latent_frames,
            num_tokens,
            strength,
            downscale_factor,
        )

        return concat_tokens, positions, conditioning_mask

    @staticmethod
    def _encode_image_keyframe(
        image,
        pixel_frame_index: int,
        strength: float,
        vae,
        mm,
        device,
        height: int,
        width: int,
        num_frames_snapped: int,
        vae_temporal_ratio: int,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        latents_mean,
        latents_std,
        frame_rate: float,
        condition_index: int,
        spatial_downscale: int = 1,
    ) -> tuple | None:
        """Encode a single-frame image as keyframe tokens (appended to sequence).

        Args:
            spatial_downscale: Spatial downscale factor (1=full, 2=half, 4=quarter).

        Returns ``(packed_tokens, positions, num_tokens)`` or ``None``.
        """
        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            get_pixel_coords,
            normalize_latents,
            pack_latents,
        )

        # Resolve pixel frame index → latent frame index
        if pixel_frame_index < 0:
            pixel_frame_index = num_frames_snapped + pixel_frame_index + 1
        latent_idx = (pixel_frame_index - 1) // vae_temporal_ratio
        latent_idx = max(0, min(latent_idx, latent_num_frames - 1))

        # Spatial downscale — snap to VAE spatial ratio
        vae_sp = getattr(vae, "spatial_compression_ratio", 32)
        encode_height = (height // spatial_downscale // vae_sp) * vae_sp
        encode_width = (width // spatial_downscale // vae_sp) * vae_sp
        encode_height = max(encode_height, vae_sp)
        encode_width = max(encode_width, vae_sp)
        kf_latent_height = encode_height // vae_sp
        kf_latent_width = encode_width // vae_sp

        # Preprocess image: numpy HWC uint8 → torch [1, C, H, W] float32 in [-1, 1]
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor * 2.0 - 1.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        # Resize to downscaled dimensions
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(encode_height, encode_width), mode="bilinear", align_corners=False
        )

        # Add temporal dimension: [1, C, 1, H, W]
        image_tensor = image_tensor.unsqueeze(2)

        # VAE encode
        with mm.use_components("vae", device=device, strategy_override="model_offload"):
            image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
            init_latents = vae.encode(image_tensor).latent_dist.mode()

        # Normalize latents (keep float32)
        if latents_mean is not None and latents_std is not None:
            init_latents = normalize_latents(init_latents.float(), latents_mean, latents_std)

        init_latents = init_latents.cpu()

        # Pack 5D → 3D: [1, num_tokens, features]
        packed_tokens = pack_latents(init_latents, patch_size=1, patch_size_t=1)

        # Generate position embeddings at downscaled dims and scale back
        positions = get_pixel_coords(
            latent_height=kf_latent_height,
            latent_width=kf_latent_width,
            latent_num_frames=1,  # single frame
            frame_idx=latent_idx,
            fps=frame_rate,
            vae_spatial_scale=getattr(vae, "spatial_compression_ratio", 32),
            vae_temporal_scale=vae_temporal_ratio,
        )
        if spatial_downscale > 1:
            positions[:, 1, :] = positions[:, 1, :] * spatial_downscale
            positions[:, 2, :] = positions[:, 2, :] * spatial_downscale

        num_tokens = packed_tokens.shape[1]

        logger.info(
            "Condition %d (keyframe image): pixel_frame=%d → latent_idx=%d, %d tokens, strength=%.2f, spatial_ds=%d",
            condition_index,
            pixel_frame_index,
            latent_idx,
            num_tokens,
            strength,
            spatial_downscale,
        )

        return packed_tokens, positions, num_tokens

    @staticmethod
    def _encode_video_keyframe(
        video_frames,
        pixel_frame_index: int,
        strength: float,
        vae,
        mm,
        device,
        height: int,
        width: int,
        num_frames_snapped: int,
        vae_temporal_ratio: int,
        vae_spatial_ratio: int,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        latents_mean,
        latents_std,
        frame_rate: float,
        condition_index: int,
        spatial_downscale: int = 1,
        temporal_stride: int = 1,
        speed: int = 1,
    ) -> tuple | None:
        """Encode a video clip as keyframe tokens (appended to sequence).

        Args:
            spatial_downscale: Spatial downscale factor (1=full, 2=half, 4=quarter).
            temporal_stride: Take every Nth frame for VRAM reduction (positions scaled
                to preserve motion speed).
            speed: Speed multiplier — skip frames to compress time (positions NOT
                scaled, so motion appears faster).

        Returns ``(packed_tokens, positions, num_tokens)`` or ``None``.
        """
        import math

        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            get_pixel_coords,
            normalize_latents,
            pack_latents,
        )

        total_video_frames = video_frames.shape[0]

        if pixel_frame_index < 0:
            pixel_frame_index = num_frames_snapped + pixel_frame_index + 1

        available = num_frames_snapped - (pixel_frame_index - 1)
        available = max(available, 0)
        use_frames = min(total_video_frames, available)

        if use_frames <= 0:
            logger.warning(
                "Condition %d (keyframe video): no frames fit (start=%d, available=%d)",
                condition_index,
                pixel_frame_index,
                available,
            )
            return None

        video_clip = video_frames[:use_frames]

        # Speed: skip frames to compress time (motion speeds up)
        if speed > 1:
            video_clip = video_clip[::speed]

        # Temporal downsample: skip frames to reduce tokens (motion preserved via position scaling)
        if temporal_stride > 1:
            video_clip = video_clip[::temporal_stride]

        use_frames_after_stride = video_clip.shape[0]

        # Align to VAE temporal grid: 8n+1
        aligned_frames = (math.ceil((use_frames_after_stride - 1) / vae_temporal_ratio)) * vae_temporal_ratio + 1
        if aligned_frames <= 0:
            aligned_frames = 1

        if aligned_frames > use_frames_after_stride:
            import numpy as np

            pad_count = aligned_frames - use_frames_after_stride
            last_frame = video_clip[-1:]
            video_clip = np.concatenate(
                [video_clip] + [last_frame] * pad_count,
                axis=0,
            )

        # Spatial downscale for keyframe encoding — snap to VAE spatial ratio
        # so encoded dimensions are cleanly divisible by the VAE's internal layers
        encode_height = (height // spatial_downscale // vae_spatial_ratio) * vae_spatial_ratio
        encode_width = (width // spatial_downscale // vae_spatial_ratio) * vae_spatial_ratio
        encode_height = max(encode_height, vae_spatial_ratio)
        encode_width = max(encode_width, vae_spatial_ratio)
        kf_latent_height = encode_height // vae_spatial_ratio
        kf_latent_width = encode_width // vae_spatial_ratio

        # Convert to tensor: [1, C, F, H, W] float32 in [-1, 1]
        video_tensor = torch.from_numpy(video_clip).float() / 255.0
        video_tensor = video_tensor * 2.0 - 1.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = torch.nn.functional.interpolate(
            video_tensor, size=(encode_height, encode_width), mode="bilinear", align_corners=False
        )
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        # VAE encode
        with mm.use_components("vae", device=device, strategy_override="model_offload"):
            video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
            encoded = vae.encode(video_tensor).latent_dist.mode()

        if latents_mean is not None and latents_std is not None:
            encoded = normalize_latents(encoded.float(), latents_mean, latents_std)

        encoded = encoded.cpu()

        # Pack 5D → 3D
        packed_tokens = pack_latents(encoded, patch_size=1, patch_size_t=1)

        # Resolve latent start position
        latent_start = (pixel_frame_index - 1) // vae_temporal_ratio
        latent_start = max(0, min(latent_start, latent_num_frames - 1))

        cond_latent_frames = encoded.shape[2]

        # Generate position embeddings at downscaled dims and scale back
        positions = get_pixel_coords(
            latent_height=kf_latent_height,
            latent_width=kf_latent_width,
            latent_num_frames=cond_latent_frames,
            frame_idx=latent_start,
            fps=frame_rate,
            vae_spatial_scale=vae_spatial_ratio,
            vae_temporal_scale=vae_temporal_ratio,
        )
        # Scale positions so they map to full-resolution coords
        if spatial_downscale > 1:
            positions[:, 1, :] = positions[:, 1, :] * spatial_downscale
            positions[:, 2, :] = positions[:, 2, :] * spatial_downscale
        # Temporal downsample: scale temporal positions so motion speed is preserved
        # (each encoded frame actually represents `temporal_stride` original frames)
        if temporal_stride > 1:
            positions[:, 0, :] = positions[:, 0, :] * temporal_stride

        num_tokens = packed_tokens.shape[1]

        logger.info(
            "Condition %d (keyframe video): %d pixel frames (speed=%d, stride=%d) → %d latent frames, %d tokens, strength=%.2f, spatial_ds=%d",
            condition_index,
            aligned_frames,
            speed,
            temporal_stride,
            cond_latent_frames,
            num_tokens,
            strength,
            spatial_downscale,
        )

        return packed_tokens, positions, num_tokens
