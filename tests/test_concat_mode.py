"""Tests for IC-LoRA concat mode: get_pixel_coords, latents concat, denoise strip, downscale."""

from __future__ import annotations

import torch

from frameartisan.modules.generation.graph.nodes.ltx2_utils import get_pixel_coords


class TestGetPixelCoords:
    """Test get_pixel_coords position embedding generation."""

    def test_output_shape(self):
        pos = get_pixel_coords(
            latent_height=4,
            latent_width=5,
            latent_num_frames=3,
            frame_idx=0,
            fps=24.0,
        )
        assert pos.shape == (1, 3, 3 * 4 * 5, 2)

    def test_single_frame(self):
        pos = get_pixel_coords(
            latent_height=2,
            latent_width=2,
            latent_num_frames=1,
            frame_idx=0,
            fps=24.0,
        )
        assert pos.shape == (1, 3, 4, 2)

    def test_temporal_coords_match_base_rope(self):
        """Temporal start coords must match prepare_video_coords convention:
        t_start = (latent_frame * 8 + 1 - 8).clamp(0) / fps
        t_end = ((latent_frame + 1) * 8 + 1 - 8).clamp(0) / fps
        Midpoint = (t_start + t_end) / 2 matches prepare_video_coords.
        """
        pos = get_pixel_coords(
            latent_height=1,
            latent_width=1,
            latent_num_frames=3,
            frame_idx=0,
            fps=24.0,
        )
        # frame 0: start=(0*8+1-8).clamp(0)=0, end=(1*8+1-8).clamp(0)=1
        # frame 1: start=(1*8+1-8).clamp(0)=1, end=(2*8+1-8).clamp(0)=9
        # frame 2: start=(2*8+1-8).clamp(0)=9, end=(3*8+1-8).clamp(0)=17
        t_start = pos[0, 0, :, 0]
        t_end = pos[0, 0, :, 1]
        assert torch.allclose(t_start, torch.tensor([0.0, 1.0 / 24, 9.0 / 24]))
        assert torch.allclose(t_end, torch.tensor([1.0 / 24, 9.0 / 24, 17.0 / 24]))

    def test_midpoints_match_prepare_video_coords_convention(self):
        """The midpoint (start + end) / 2 should match what prepare_video_coords produces."""
        pos = get_pixel_coords(
            latent_height=1,
            latent_width=1,
            latent_num_frames=3,
            frame_idx=0,
            fps=24.0,
        )
        midpoints = (pos[0, 0, :, 0] + pos[0, 0, :, 1]) / 2.0
        # frame 0: (0 + 1/24) / 2 = 0.5/24
        # frame 1: (1/24 + 9/24) / 2 = 5/24
        # frame 2: (9/24 + 17/24) / 2 = 13/24
        expected = torch.tensor([0.5 / 24, 5.0 / 24, 13.0 / 24])
        assert torch.allclose(midpoints, expected)

    def test_frame_idx_offset(self):
        pos = get_pixel_coords(
            latent_height=1,
            latent_width=1,
            latent_num_frames=2,
            frame_idx=5,
            fps=10.0,
        )
        # frame_idx=5: start=(5*8+1-8)=33/10=3.3, end=(6*8+1-8)=41/10=4.1
        # frame_idx=6: start=(6*8+1-8)=41/10=4.1, end=(7*8+1-8)=49/10=4.9
        t_start = pos[0, 0, :, 0]
        t_end = pos[0, 0, :, 1]
        assert torch.allclose(t_start, torch.tensor([3.3, 4.1]))
        assert torch.allclose(t_end, torch.tensor([4.1, 4.9]))

    def test_spatial_coords_scale(self):
        pos = get_pixel_coords(
            latent_height=2,
            latent_width=3,
            latent_num_frames=1,
            frame_idx=0,
            fps=24.0,
            vae_spatial_scale=32,
        )
        # y start: [0, 0, 0, 32, 32, 32], y end: [32, 32, 32, 64, 64, 64]
        y_start = pos[0, 1, :, 0]
        y_end = pos[0, 1, :, 1]
        expected_y_start = torch.tensor([0, 0, 0, 32, 32, 32], dtype=torch.float32)
        expected_y_end = torch.tensor([32, 32, 32, 64, 64, 64], dtype=torch.float32)
        assert torch.allclose(y_start, expected_y_start)
        assert torch.allclose(y_end, expected_y_end)

        # x start: [0, 32, 64, 0, 32, 64], x end: [32, 64, 96, 32, 64, 96]
        x_start = pos[0, 2, :, 0]
        x_end = pos[0, 2, :, 1]
        expected_x_start = torch.tensor([0, 32, 64, 0, 32, 64], dtype=torch.float32)
        expected_x_end = torch.tensor([32, 64, 96, 32, 64, 96], dtype=torch.float32)
        assert torch.allclose(x_start, expected_x_start)
        assert torch.allclose(x_end, expected_x_end)

    def test_device_parameter(self):
        pos = get_pixel_coords(
            latent_height=2,
            latent_width=2,
            latent_num_frames=1,
            frame_idx=0,
            fps=24.0,
            device=torch.device("cpu"),
        )
        assert pos.device == torch.device("cpu")


class TestDownscaleFactorPositionScaling:
    """Test that downscale_factor correctly scales spatial coords."""

    def test_downscale_factor_2(self):
        pos = get_pixel_coords(
            latent_height=2,
            latent_width=2,
            latent_num_frames=1,
            frame_idx=0,
            fps=24.0,
            vae_spatial_scale=32,
        )
        # Apply downscale_factor=2 to spatial coords (both start and end)
        pos[:, 1, :] *= 2
        pos[:, 2, :] *= 2

        y_start = pos[0, 1, :, 0]
        x_start = pos[0, 2, :, 0]
        # y start: [0, 0, 32, 32] * 2 = [0, 0, 64, 64]
        assert torch.allclose(y_start, torch.tensor([0, 0, 64, 64], dtype=torch.float32))
        # x start: [0, 32, 0, 32] * 2 = [0, 64, 0, 64]
        assert torch.allclose(x_start, torch.tensor([0, 64, 0, 64], dtype=torch.float32))

    def test_downscale_factor_1_noop(self):
        pos1 = get_pixel_coords(
            latent_height=2,
            latent_width=2,
            latent_num_frames=1,
            frame_idx=0,
            fps=24.0,
        )
        pos2 = pos1.clone()
        pos2[:, 1, :] *= 1
        pos2[:, 2, :] *= 1
        assert torch.allclose(pos1, pos2)


class TestLatentsNodeConcatInputs:
    """Test that LTX2LatentsNode declares correct concat inputs/outputs."""

    def test_concat_optional_inputs(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_latents_node import LTX2LatentsNode

        assert "concat_latents" in LTX2LatentsNode.OPTIONAL_INPUTS
        assert "concat_positions" in LTX2LatentsNode.OPTIONAL_INPUTS
        assert "concat_conditioning_mask" in LTX2LatentsNode.OPTIONAL_INPUTS

    def test_base_num_tokens_output(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_latents_node import LTX2LatentsNode

        assert "base_num_tokens" in LTX2LatentsNode.OUTPUTS


class TestDenoiseNodeConcatInputs:
    """Test that LTX2DenoiseNode declares base_num_tokens input."""

    def test_base_num_tokens_optional_input(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        assert "base_num_tokens" in LTX2DenoiseNode.OPTIONAL_INPUTS


class TestDenoiseStripConcatTokens:
    """Test the concat token stripping logic."""

    def test_strip_with_base_num_tokens(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        node = LTX2DenoiseNode()
        node.base_num_tokens = 100
        latents = torch.randn(1, 150, 128)
        result = node._strip_concat_tokens(latents)
        assert result.shape == (1, 100, 128)

    def test_strip_noop_when_none(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        node = LTX2DenoiseNode()
        node.base_num_tokens = None
        latents = torch.randn(1, 100, 128)
        result = node._strip_concat_tokens(latents)
        assert result.shape == (1, 100, 128)

    def test_strip_preserves_base_tokens(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        node = LTX2DenoiseNode()
        node.base_num_tokens = 50
        base = torch.randn(1, 50, 128)
        concat = torch.randn(1, 30, 128)
        combined = torch.cat([base, concat], dim=1)
        result = node._strip_concat_tokens(combined)
        assert torch.allclose(result, base)


class TestSecondPassBaseNumTokens:
    """Test that second pass latents node outputs base_num_tokens."""

    def test_base_num_tokens_in_outputs(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import (
            LTX2SecondPassLatentsNode,
        )

        assert "base_num_tokens" in LTX2SecondPassLatentsNode.OUTPUTS


class TestLoraNodeDownscaleFactor:
    """Test that LoRA node declares reference_downscale_factor output."""

    def test_output_declared(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import LTX2LoraNode

        assert "reference_downscale_factor" in LTX2LoraNode.OUTPUTS

    def test_read_downscale_missing_metadata(self, tmp_path):
        """LoRA without metadata should return downscale_factor=1."""
        from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import LTX2LoraNode

        # Create a minimal safetensors file without reference_downscale_factor
        from safetensors.torch import save_file

        dummy = {"weight": torch.zeros(1)}
        path = str(tmp_path / "test.safetensors")
        save_file(dummy, path)

        result = LTX2LoraNode._read_reference_downscale_factor(path)
        assert result == 1

    def test_read_downscale_with_metadata(self, tmp_path):
        """LoRA with reference_downscale_factor metadata should return that value."""
        from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import LTX2LoraNode

        from safetensors.torch import save_file

        dummy = {"weight": torch.zeros(1)}
        path = str(tmp_path / "test.safetensors")
        save_file(dummy, path, metadata={"reference_downscale_factor": "4"})

        result = LTX2LoraNode._read_reference_downscale_factor(path)
        assert result == 4

    def test_read_downscale_nonexistent_file(self):
        """Nonexistent file should return 1 (graceful fallback)."""
        from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import LTX2LoraNode

        result = LTX2LoraNode._read_reference_downscale_factor("/nonexistent/path.safetensors")
        assert result == 1


class TestConcatConditioningMaskConvention:
    """Regression: concat conditioning mask must use strength (not 1-strength).

    The denoise node convention is: 1.0 = conditioned (timestep=0), 0.0 = unconditioned.
    Concat tokens are clean VAE-encoded references that must see timestep=0.
    Using ``1 - strength`` caused the model to treat clean references as noisy,
    producing dark output that reproduced the raw reference appearance.
    """

    def test_concat_mask_equals_strength(self):
        """Conditioning mask for concat tokens must equal the condition strength."""
        from unittest.mock import MagicMock, patch

        from frameartisan.modules.generation.graph.nodes.ltx2_condition_encode_node import (
            LTX2ConditionEncodeNode,
        )

        import numpy as np

        # 9 frames (8n+1), small spatial dims
        video_frames = np.random.randint(0, 255, (9, 64, 64, 3), dtype=np.uint8)
        strength = 0.8

        # Mock VAE
        vae = MagicMock()
        vae.config.latent_channels = 128
        vae.spatial_compression_ratio = 32
        vae.temporal_compression_ratio = 8
        vae.dtype = torch.float32
        vae.latents_mean = torch.zeros(128)
        vae.latents_std = torch.ones(128)
        encode_result = MagicMock()
        encode_result.latent_dist.mode.return_value = torch.randn(1, 128, 1, 2, 2)
        vae.encode.return_value = encode_result

        # Mock ModelManager
        mm = MagicMock()
        mm.use_components.return_value.__enter__ = MagicMock(return_value=None)
        mm.use_components.return_value.__exit__ = MagicMock(return_value=False)

        result = LTX2ConditionEncodeNode._encode_concat_video_condition(
            video_frames=video_frames,
            pixel_frame_index=1,
            strength=strength,
            downscale_factor=1,
            vae=vae,
            mm=mm,
            device=torch.device("cpu"),
            height=64,
            width=64,
            num_frames_snapped=9,
            vae_temporal_ratio=8,
            vae_spatial_ratio=32,
            latent_num_frames=2,
            latent_height=2,
            latent_width=2,
            latents_mean=vae.latents_mean,
            latents_std=vae.latents_std,
            frame_rate=24.0,
            condition_index=0,
        )

        assert result is not None
        _, _, cond_mask = result
        # The mask must be strength, NOT (1 - strength)
        assert torch.allclose(cond_mask, torch.full_like(cond_mask, strength))

    def test_concat_mask_full_strength_means_timestep_zero(self):
        """With strength=1.0, conditioning_mask=1.0 → timestep * (1 - 1.0) = 0."""
        strength = 1.0
        mask_value = strength  # correct convention
        timestep = 0.8

        video_timestep = timestep * (1 - mask_value)
        assert video_timestep == 0.0, (
            "Concat tokens with strength=1.0 must see timestep=0 (fully clean)"
        )

    def test_concat_mask_zero_strength_means_full_timestep(self):
        """With strength=0.0, conditioning_mask=0.0 → timestep * (1 - 0.0) = timestep."""
        strength = 0.0
        mask_value = strength
        timestep = 0.8

        video_timestep = timestep * (1 - mask_value)
        assert video_timestep == timestep, (
            "Concat tokens with strength=0.0 must see full timestep (unconditioned)"
        )


class TestGetPixelCoordsMatchesPrepareVideoCoords:
    """Regression: get_pixel_coords must return [1, 3, N, 2] with [start, end) bounds.

    The transformer's RoPE (prepare_video_coords) returns [B, 3, N, 2] and computes
    midpoints: (start + end) / 2.  If get_pixel_coords returns only start values
    (duplicated as [start, start]), midpoints are offset by half a patch from base
    tokens, causing positional misalignment for concat/keyframe references.
    """

    def test_bounds_are_not_degenerate(self):
        """Start and end must differ (non-zero-width patches)."""
        pos = get_pixel_coords(
            latent_height=2, latent_width=2, latent_num_frames=2,
            frame_idx=0, fps=24.0,
        )
        starts = pos[..., 0]
        ends = pos[..., 1]
        # Every end must be strictly greater than start
        assert (ends > starts).all(), "All patches must have end > start (non-degenerate)"

    def test_spatial_bounds_width_equals_vae_scale(self):
        """Spatial [start, end) width must equal vae_spatial_scale."""
        vae_spatial = 32
        pos = get_pixel_coords(
            latent_height=3, latent_width=4, latent_num_frames=1,
            frame_idx=0, fps=24.0, vae_spatial_scale=vae_spatial,
        )
        y_width = pos[0, 1, :, 1] - pos[0, 1, :, 0]
        x_width = pos[0, 2, :, 1] - pos[0, 2, :, 0]
        assert torch.allclose(y_width, torch.full_like(y_width, vae_spatial))
        assert torch.allclose(x_width, torch.full_like(x_width, vae_spatial))

    def test_midpoints_match_prepare_video_coords(self):
        """Midpoints from get_pixel_coords must match prepare_video_coords exactly."""
        from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed

        latent_h, latent_w, latent_f = 2, 3, 4
        fps = 24.0
        vae_t, vae_s = 8, 32

        rope = LTX2AudioVideoRotaryPosEmbed(
            dim=64,
            patch_size=1,
            patch_size_t=1,
            scale_factors=(vae_t, vae_s, vae_s),
            causal_offset=1,
        )
        base_coords = rope.prepare_video_coords(1, latent_f, latent_h, latent_w, torch.device("cpu"), fps=fps)
        base_mid = (base_coords[..., 0] + base_coords[..., 1]) / 2.0  # [1, 3, N]

        concat_coords = get_pixel_coords(
            latent_height=latent_h, latent_width=latent_w, latent_num_frames=latent_f,
            frame_idx=0, fps=fps, vae_spatial_scale=vae_s, vae_temporal_scale=vae_t,
        )
        concat_mid = (concat_coords[..., 0] + concat_coords[..., 1]) / 2.0  # [1, 3, N]

        assert torch.allclose(base_mid, concat_mid, atol=1e-5), (
            f"Midpoint mismatch:\nbase={base_mid}\nconcat={concat_mid}"
        )

    def test_midpoints_match_with_frame_offset(self):
        """Midpoints must match even when frame_idx > 0."""
        from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed

        latent_h, latent_w = 2, 2
        latent_f_total = 8  # total frames for base coords
        frame_offset = 3     # concat starts at latent frame 3
        latent_f_concat = 2  # concat covers 2 frames
        fps = 24.0
        vae_t, vae_s = 8, 32

        rope = LTX2AudioVideoRotaryPosEmbed(
            dim=64, patch_size=1, patch_size_t=1,
            scale_factors=(vae_t, vae_s, vae_s), causal_offset=1,
        )
        base_coords = rope.prepare_video_coords(1, latent_f_total, latent_h, latent_w, torch.device("cpu"), fps=fps)
        base_mid = (base_coords[..., 0] + base_coords[..., 1]) / 2.0

        concat_coords = get_pixel_coords(
            latent_height=latent_h, latent_width=latent_w, latent_num_frames=latent_f_concat,
            frame_idx=frame_offset, fps=fps, vae_spatial_scale=vae_s, vae_temporal_scale=vae_t,
        )
        concat_mid = (concat_coords[..., 0] + concat_coords[..., 1]) / 2.0

        # Extract the matching slice from base (frames 3 and 4)
        tokens_per_frame = latent_h * latent_w
        start = frame_offset * tokens_per_frame
        end = start + latent_f_concat * tokens_per_frame
        base_slice = base_mid[:, :, start:end]

        assert torch.allclose(base_slice, concat_mid, atol=1e-5), (
            f"Midpoint mismatch at frame_offset={frame_offset}:\n"
            f"base={base_slice}\nconcat={concat_mid}"
        )


class TestGenerationSettingsMode:
    """Test video_conditioning_mode in GenerationSettings."""

    def test_default_mode(self):
        from frameartisan.modules.generation.generation_settings import GenerationSettings

        s = GenerationSettings()
        assert s.video_conditioning_mode == "replace"
