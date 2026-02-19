from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2ImageEncodeNode(Node):
    """VAE-encode a source image into latents and create a conditioning mask for image-to-video."""

    PRIORITY = 0
    REQUIRED_INPUTS: ClassVar[list[str]] = ["vae", "image", "num_frames", "height", "width"]
    OPTIONAL_INPUTS: ClassVar[list[str]] = []
    OUTPUTS: ClassVar[list[str]] = ["image_latents", "conditioning_mask"]
    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    def __call__(self):
        import torch

        from frameartisan.app.model_manager import get_model_manager
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import normalize_latents, pack_latents

        vae = self.vae
        image = self.image  # numpy HWC uint8
        num_frames = int(self.num_frames)
        height = int(self.height)
        width = int(self.width)

        if vae is None:
            raise NodeError("vae input is not connected", self.__class__.__name__)
        if image is None:
            raise NodeError("image input is not connected", self.__class__.__name__)

        device = self.device
        mm = get_model_manager()

        # Preprocess image: numpy HWC uint8 → torch [1, C, H, W] float32 in [-1, 1]
        # (matches diffusers VaeImageProcessor normalization)
        image_tensor = torch.from_numpy(image).float() / 255.0  # [H, W, C] in [0, 1]
        image_tensor = image_tensor * 2.0 - 1.0  # scale to [-1, 1]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        # Resize to target dimensions
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(height, width), mode="bilinear", align_corners=False
        )

        # Add temporal dimension: [1, C, 1, H, W]
        image_tensor = image_tensor.unsqueeze(2)

        # VAE compression ratios
        vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 32)
        vae_temporal_ratio = getattr(vae, "temporal_compression_ratio", 8)

        # Latent dimensions
        num_frames_snapped = 8 * ((num_frames - 1) // 8) + 1
        latent_num_frames = (num_frames_snapped - 1) // vae_temporal_ratio + 1
        latent_height = height // vae_spatial_ratio
        latent_width = width // vae_spatial_ratio

        # VAE encode
        with mm.use_components("vae", device=device, strategy_override="model_offload"):
            image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
            init_latents = vae.encode(image_tensor).latent_dist.mode()

        # Normalize latents (match pipeline: cast to float32 first, keep float32)
        latents_mean = vae.latents_mean
        latents_std = vae.latents_std
        if latents_mean is not None and latents_std is not None:
            init_latents = normalize_latents(init_latents.float(), latents_mean, latents_std)

        # Repeat across frames: [1, C, 1, lH, lW] → [1, C, latent_num_frames, lH, lW]
        init_latents = init_latents.repeat(1, 1, latent_num_frames, 1, 1)

        # Get patch sizes from transformer config (will be available on device at graph exec time)
        patch_size = 1
        patch_size_t = 1

        # Pack latents to [B, seq_len, features]
        packed_latents = pack_latents(init_latents, patch_size, patch_size_t)

        # Create conditioning mask: [1, C, latent_num_frames, lH, lW]
        # Frame 0 is conditioned (mask=1), rest are not (mask=0)
        mask = torch.zeros(1, 1, latent_num_frames, latent_height, latent_width, device=init_latents.device)
        mask[:, :, 0] = 1.0

        # Pack mask and squeeze channel dim → [B, seq_len]
        packed_mask = pack_latents(mask, patch_size, patch_size_t).squeeze(-1)

        logger.info(
            "Image encoded: latents %s, conditioning_mask %s",
            packed_latents.shape,
            packed_mask.shape,
        )

        self.values = {
            "image_latents": packed_latents,
            "conditioning_mask": packed_mask,
        }
        return self.values
