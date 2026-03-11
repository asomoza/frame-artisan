from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2SecondPassLatentsNode(Node):
    PRIORITY = 0
    REQUIRED_INPUTS: ClassVar[list[str]] = [
        "transformer",
        "vae",
        "audio_vae",
        "video_latents",
        "audio_latents",
        "audio_coords",
        "latent_num_frames",
        "latent_height",
        "latent_width",
        "audio_num_frames",
        "frame_rate",
        "seed",
        "model_type",
    ]
    OPTIONAL_INPUTS: ClassVar[list[str]] = [
        "conditioning_mask",
        "clean_latents",
        "clean_audio_latents",
        "audio_conditioning_mask",
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
    ]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    def __call__(self):
        import torch
        from diffusers.utils.torch_utils import randn_tensor

        from frameartisan.modules.generation.constants import LTX2_STAGE2_DISTILLED_SIGMAS
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            normalize_audio_latents,
            normalize_latents,
            pack_latents,
            unpack_latents,
        )

        transformer = self.transformer
        vae = self.vae
        audio_vae = self.audio_vae
        video_latents = self.video_latents
        audio_latents = self.audio_latents
        audio_coords = self.audio_coords
        latent_num_frames = int(self.latent_num_frames)
        latent_height = int(self.latent_height)
        latent_width = int(self.latent_width)
        audio_num_frames = int(self.audio_num_frames)
        frame_rate = float(self.frame_rate)
        seed = int(self.seed)
        model_type = int(self.model_type) if self.model_type is not None else None
        conditioning_mask = self.conditioning_mask  # None for T2V, present for I2V
        clean_latents_packed = self.clean_latents  # packed [B, seq, feat] or None
        clean_audio_latents = self.clean_audio_latents  # packed audio or None
        audio_conditioning_mask = self.audio_conditioning_mask  # [B, seq_len] or None

        if transformer is None:
            raise NodeError("transformer must be connected", self.__class__.__name__)

        device = self.device
        has_conditioning = conditioning_mask is not None
        is_i2v = has_conditioning

        # Normalize video latents — the upsample node outputs denormalized latents
        # (the upsampler model operates in denormalized space), and the denoiser
        # expects normalized latents.  This matches the pipeline's prepare_latents.
        video_latents = unpack_latents(
            video_latents,
            latent_num_frames,
            latent_height,
            latent_width,
            patch_size=1,
            patch_size_t=1,
        )
        video_latents = normalize_latents(
            video_latents,
            vae.latents_mean,
            vae.latents_std,
            vae.config.scaling_factor,
        )
        video_latents = pack_latents(video_latents, patch_size=1, patch_size_t=1)

        # Upsample clean_latents if present (for x0 blending in 2nd pass denoise)
        packed_clean_for_output = None
        if clean_latents_packed is not None and has_conditioning:
            # Unpack → spatial upsample 2x → repack
            # The upsampler only doubles spatial dims, not temporal — unpack
            # clean_latents at the original (pre-upsample) dimensions.
            clean_5d = unpack_latents(
                clean_latents_packed,
                latent_num_frames,
                latent_height // 2,
                latent_width // 2,
                patch_size=1,
                patch_size_t=1,
            )
            # Upsample spatial dims 2x (keep temporal unchanged)
            b, c, f, h, w = clean_5d.shape
            clean_5d = torch.nn.functional.interpolate(
                clean_5d, size=(f, h * 2, w * 2), mode="trilinear", align_corners=False
            )
            packed_clean_for_output = pack_latents(clean_5d, patch_size=1, patch_size_t=1)

        if model_type == 2:
            noise_scale = LTX2_STAGE2_DISTILLED_SIGMAS[0]

            generator = torch.Generator(str(device)).manual_seed(seed)

            if is_i2v:
                # I2V: add noise using conditioning_mask to preserve conditioned frames.
                # Unpack → noise → repack in 5D for correct random sequence.
                # Rebuild the 5D mask from the packed conditioning_mask
                cond_mask_packed = conditioning_mask.unsqueeze(-1)  # [B, seq, 1]
                cond_mask_5d_raw = unpack_latents(
                    cond_mask_packed,
                    latent_num_frames,
                    latent_height // 2,
                    latent_width // 2,
                    patch_size=1,
                    patch_size_t=1,
                )
                # Upsample mask to new 2x spatial dims
                bm, cm, fm, hm, wm = cond_mask_5d_raw.shape
                cond_mask_5d = torch.nn.functional.interpolate(
                    cond_mask_5d_raw, size=(fm, hm * 2, wm * 2), mode="nearest"
                )

                video_5d = unpack_latents(
                    video_latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    patch_size=1,
                    patch_size_t=1,
                )
                video_noise_5d = randn_tensor(video_5d.shape, generator=generator, device=device, dtype=video_5d.dtype)
                masked_scale = noise_scale * (1 - cond_mask_5d)
                video_5d = masked_scale * video_noise_5d + (1 - masked_scale) * video_5d
                video_latents = pack_latents(video_5d, patch_size=1, patch_size_t=1)
            else:
                # T2V: pipeline adds noise to packed 3D video latents.
                video_noise = randn_tensor(
                    video_latents.shape, generator=generator, device=device, dtype=video_latents.dtype
                )
                video_latents = noise_scale * video_noise + (1 - noise_scale) * video_latents

            # Normalize audio latents, then add noise (matching pipeline order for both T2V and I2V)
            # Skip audio noise if full audio conditioning is active (mask=None, clean present)
            if audio_conditioning_mask is not None and clean_audio_latents is not None:
                # Partial audio conditioning: add noise only to unconditioned tokens
                audio_latents = normalize_audio_latents(audio_latents, audio_vae.latents_mean, audio_vae.latents_std)
                audio_noise = randn_tensor(
                    audio_latents.shape, generator=generator, device=device, dtype=audio_latents.dtype
                )
                acm_3d = audio_conditioning_mask.unsqueeze(-1).to(device=device, dtype=audio_latents.dtype)
                masked_audio_scale = noise_scale * (1 - acm_3d)
                audio_latents = masked_audio_scale * audio_noise + (1 - masked_audio_scale) * audio_latents
            elif clean_audio_latents is None:
                audio_latents = normalize_audio_latents(audio_latents, audio_vae.latents_mean, audio_vae.latents_std)
                audio_noise = randn_tensor(
                    audio_latents.shape, generator=generator, device=device, dtype=audio_latents.dtype
                )
                audio_latents = noise_scale * audio_noise + (1 - noise_scale) * audio_latents
            # else: full audio conditioning (mask=None, clean present) — skip noise

        # For conditioning, create a fresh conditioning_mask at stage 2 latent dimensions
        if has_conditioning:
            if clean_latents_packed is not None and packed_clean_for_output is not None:
                # Generalized path: rebuild mask at 2x spatial resolution from the
                # upsampled clean latents (any non-zero mask position persists)
                cond_mask_packed_src = conditioning_mask.unsqueeze(-1)
                cond_mask_5d_src = unpack_latents(
                    cond_mask_packed_src,
                    latent_num_frames,
                    latent_height // 2,
                    latent_width // 2,
                    patch_size=1,
                    patch_size_t=1,
                )
                bm2, cm2, fm2, hm2, wm2 = cond_mask_5d_src.shape
                cond_mask_5d_new = torch.nn.functional.interpolate(
                    cond_mask_5d_src, size=(fm2, hm2 * 2, wm2 * 2), mode="nearest"
                )
                conditioning_mask = pack_latents(cond_mask_5d_new, patch_size=1, patch_size_t=1).squeeze(-1)
            else:
                # Legacy path: simple frame-0 mask
                mask_shape = (1, 1, latent_num_frames, latent_height, latent_width)
                conditioning_mask = torch.zeros(mask_shape, device=device, dtype=video_latents.dtype)
                conditioning_mask[:, :, 0] = 1.0
                conditioning_mask = pack_latents(conditioning_mask, patch_size=1, patch_size_t=1).squeeze(-1)

        # Generate new RoPE video_coords for the 2x latent dimensions
        video_coords = transformer.rope.prepare_video_coords(
            video_latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            video_latents.device,
            fps=frame_rate,
        )

        logger.info(
            "Second pass latents prepared: latent dims %dx%dx%d, model_type=%s, i2v=%s",
            latent_num_frames,
            latent_height,
            latent_width,
            model_type,
            is_i2v,
        )

        self.values = {
            "video_latents": video_latents,
            "audio_latents": audio_latents,
            "video_coords": video_coords,
            "audio_coords": audio_coords,
            "latent_num_frames": latent_num_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "audio_num_frames": audio_num_frames,
            "conditioning_mask": conditioning_mask,
            "clean_latents": packed_clean_for_output,
            "clean_audio_latents": clean_audio_latents,
            "audio_conditioning_mask": audio_conditioning_mask,
            "base_num_tokens": None,
        }
        return self.values
