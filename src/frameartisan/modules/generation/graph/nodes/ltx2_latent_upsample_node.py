from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2LatentUpsampleNode(Node):
    PRIORITY = 0
    REQUIRED_INPUTS: ClassVar[list[str]] = [
        "video_latents",
        "vae",
        "latent_num_frames",
        "latent_height",
        "latent_width",
    ]
    OPTIONAL_INPUTS: ClassVar[list[str]] = []
    OUTPUTS: ClassVar[list[str]] = [
        "video_latents",
        "latent_num_frames",
        "latent_height",
        "latent_width",
    ]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = {"upsampler_model_path"}

    def __init__(self, upsampler_model_path: str = ""):
        super().__init__()
        self.upsampler_model_path = upsampler_model_path

    def __call__(self):
        import torch
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        from frameartisan.app.model_manager import get_model_manager
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            denormalize_latents,
            pack_latents,
            unpack_latents,
        )

        video_latents = self.video_latents
        vae = self.vae
        latent_num_frames = int(self.latent_num_frames)
        latent_height = int(self.latent_height)
        latent_width = int(self.latent_width)

        if video_latents is None or vae is None:
            raise NodeError("video_latents and vae must be connected", self.__class__.__name__)

        if not self.upsampler_model_path:
            raise NodeError("upsampler_model_path is empty", self.__class__.__name__)

        device = self.device
        mm = get_model_manager()

        # Unpack from [B, seq, feat] to [B, C, F, H, W] (patch_size=1)
        video_latents = unpack_latents(
            video_latents,
            latent_num_frames,
            latent_height,
            latent_width,
            patch_size=1,
            patch_size_t=1,
        )

        # Load upsampler model
        upsampler_hash = mm.component_hash(self.upsampler_model_path)
        latent_upsampler = mm.get_cached(upsampler_hash)
        if latent_upsampler is None:
            logger.info("Loading latent upsampler from %s", self.upsampler_model_path)
            try:
                latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                    self.upsampler_model_path,
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                raise NodeError(f"Failed to load latent upsampler: {e}", self.__class__.__name__) from e
            mm.set_cached(upsampler_hash, latent_upsampler)
        else:
            logger.info("Latent upsampler loaded from cache")

        # The latent upsampler operates on unnormalized (denormalized) latents.
        # Pipeline comment: "NOTE: latent upsampler operates on the unnormalized latents"
        video_latents = denormalize_latents(
            video_latents,
            vae.latents_mean,
            vae.latents_std,
            vae.config.scaling_factor,
        )

        # The upsampler is a small one-shot model — manage GPU placement
        # directly rather than via the managed component system.  This avoids
        # leaving it on GPU after use (with group_offload, use_components is a
        # no-op so the explicit .to(device) would never be undone).
        latent_upsampler.to(device=device, dtype=torch.bfloat16)
        video_latents = video_latents.to(device=device, dtype=torch.bfloat16)
        upsampled = latent_upsampler(video_latents)
        latent_upsampler.to("cpu")
        del video_latents

        # Output dimensions are 2x spatial
        new_latent_height = latent_height * 2
        new_latent_width = latent_width * 2

        # Pack back to [B, seq, feat] and move to CPU to free VRAM between stages
        upsampled = pack_latents(upsampled, patch_size=1, patch_size_t=1)

        logger.info(
            "Latents upsampled: %dx%d -> %dx%d",
            latent_height,
            latent_width,
            new_latent_height,
            new_latent_width,
        )

        # Evict upsampler from cache — it's only used once per generation
        mm.evict_cached(upsampler_hash)

        self.values = {
            "video_latents": upsampled.cpu(),
            "latent_num_frames": latent_num_frames,
            "latent_height": new_latent_height,
            "latent_width": new_latent_width,
        }
        return self.values
