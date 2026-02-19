from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2DecodeNode(Node):
    PRIORITY = -1
    REQUIRED_INPUTS: ClassVar[list[str]] = [
        "vae",
        "audio_vae",
        "vocoder",
        "video_latents",
        "audio_latents",
        "latent_num_frames",
        "latent_height",
        "latent_width",
        "audio_num_frames",
        "num_frames",
        "frame_rate",
    ]
    OPTIONAL_INPUTS: ClassVar[list[str]] = ["clean_audio_latents", "audio_conditioning_mask"]
    OUTPUTS: ClassVar[list[str]] = ["video", "audio", "frame_rate_out"]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    def __call__(self):
        import torch

        from frameartisan.app.model_manager import get_model_manager
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            denormalize_audio_latents,
            denormalize_latents,
            unpack_audio_latents,
            unpack_latents,
            vae_temporal_decode_streaming,
        )

        vae = self.vae
        audio_vae = self.audio_vae
        vocoder = self.vocoder
        video_latents = self.video_latents
        audio_latents = self.audio_latents
        latent_num_frames = int(self.latent_num_frames)
        latent_height = int(self.latent_height)
        latent_width = int(self.latent_width)
        audio_num_frames = int(self.audio_num_frames)
        num_frames = int(self.num_frames)  # noqa: F841
        frame_rate = float(self.frame_rate)

        if vae is None or audio_vae is None or vocoder is None:
            raise NodeError("vae, audio_vae, and vocoder must be connected", self.__class__.__name__)

        device = self.device
        mm = get_model_manager()

        # Transformer patch sizes (from config stored on VAE side or defaults)
        patch_size = 1
        patch_size_t = 1

        # --- Video decode ---
        logger.info("Unpacking and decoding video latents")
        video_latents = unpack_latents(
            video_latents,
            latent_num_frames,
            latent_height,
            latent_width,
            patch_size,
            patch_size_t,
        )

        # Determine whether streaming decode is active.
        use_streaming = getattr(vae, "_use_streaming_decode", False)
        tile_latent_min = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio

        # When streaming decode is enabled, let the base strategy (e.g.
        # sequential_group_offload) handle VAE layer placement so only a few
        # layers are on GPU at a time.  Without streaming, bulk model_offload
        # is faster for small decode workloads.
        decode_override = None if use_streaming else "model_offload"

        with mm.use_components("vae", "audio_vae", "vocoder", device=device, strategy_override=decode_override):
            video_latents = denormalize_latents(
                video_latents,
                vae.latents_mean,
                vae.latents_std,
                vae.config.scaling_factor,
            )

            # Timestep-conditioned VAE decoding
            timestep = None
            if getattr(vae.config, "timestep_conditioning", False):
                decode_timestep = torch.tensor([0.0], device=device, dtype=video_latents.dtype)
                timestep = decode_timestep

            # When streaming decode is enabled, tile temporal chunks from CPU
            # to GPU one at a time — same logic as the VAE's internal
            # _temporal_tiled_decode but with much lower peak VRAM.
            if use_streaming and video_latents.shape[2] > tile_latent_min:
                logger.info("Using streaming temporal decode")
                video_latents_cpu = video_latents.to("cpu", dtype=vae.dtype)
                video = vae_temporal_decode_streaming(vae, video_latents_cpu, device=device, temb=timestep)
                del video_latents_cpu
            else:
                video_latents = video_latents.to(device=device, dtype=vae.dtype)
                video = vae.decode(video_latents, timestep, return_dict=False)[0]

            # Post-process video: [B, C, F, H, W] -> [F, H, W, C] uint8
            # When streaming decode returned a CPU tensor, keep it on CPU to
            # preserve the VRAM savings.  Otherwise use GPU for speed.
            if video.device.type != "cpu":
                video = (video.float() * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(1, 0, 2, 3)  # [F, C, H, W]
                video = video.permute(0, 2, 3, 1).contiguous()  # [F, H, W, C]
                video = (video * 255).round().to(torch.uint8)
                video_uint8 = video.cpu().numpy()
                del video
            else:
                video = (video.float() * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(1, 0, 2, 3)
                video = video.permute(0, 2, 3, 1)
                video_uint8 = (video * 255).round().to(torch.uint8).numpy()
                del video

            # --- Audio decode ---
            logger.info("Unpacking and decoding audio latents")
            audio_mel_ratio = getattr(audio_vae, "mel_compression_ratio", 4)
            num_mel_bins = getattr(audio_vae.config, "mel_bins", 64)
            latent_mel_bins = num_mel_bins // audio_mel_ratio

            # Use clean audio latents only for full conditioning (mask=None).
            # When mask is present (partial conditioning), use denoised audio_latents.
            clean_audio = self.clean_audio_latents
            audio_cond_mask = self.audio_conditioning_mask
            if clean_audio is not None and audio_cond_mask is None:
                audio_to_decode = clean_audio
            else:
                audio_to_decode = audio_latents

            audio_to_decode = denormalize_audio_latents(
                audio_to_decode,
                audio_vae.latents_mean,
                audio_vae.latents_std,
            )
            audio_to_decode = unpack_audio_latents(audio_to_decode, audio_num_frames, num_mel_bins=latent_mel_bins)

            audio_to_decode = audio_to_decode.to(device=device, dtype=audio_vae.dtype)
            mel_spectrograms = audio_vae.decode(audio_to_decode, return_dict=False)[0]
            audio = vocoder(mel_spectrograms)

        audio_out = None
        try:
            audio_out = audio[0].float().cpu() if hasattr(audio[0], "cpu") else audio[0]
        except Exception:
            audio_out = None

        self.values = {
            "video": video_uint8,
            "audio": audio_out,
            "frame_rate_out": float(frame_rate),
        }
        return self.values
