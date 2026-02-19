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
        assert pos.shape == (1, 3, 3 * 4 * 5)

    def test_single_frame(self):
        pos = get_pixel_coords(
            latent_height=2,
            latent_width=2,
            latent_num_frames=1,
            frame_idx=0,
            fps=24.0,
        )
        assert pos.shape == (1, 3, 4)

    def test_temporal_coords_match_base_rope(self):
        """Temporal coords must match prepare_video_coords convention:
        t_pixel = (latent_frame * 8 + 1 - 8).clamp(0) / fps
        """
        pos = get_pixel_coords(
            latent_height=1,
            latent_width=1,
            latent_num_frames=3,
            frame_idx=0,
            fps=24.0,
        )
        # frame 0: (0*8+1-8).clamp(0)=0 → 0/24=0
        # frame 1: (1*8+1-8).clamp(0)=1 → 1/24
        # frame 2: (2*8+1-8).clamp(0)=9 → 9/24
        t = pos[0, 0, :]
        assert torch.allclose(t, torch.tensor([0.0, 1.0 / 24, 9.0 / 24]))

    def test_frame_idx_offset(self):
        pos = get_pixel_coords(
            latent_height=1,
            latent_width=1,
            latent_num_frames=2,
            frame_idx=5,
            fps=10.0,
        )
        # frame_idx=5: (5*8+1-8)=33 → 33/10=3.3
        # frame_idx=6: (6*8+1-8)=41 → 41/10=4.1
        t = pos[0, 0, :]
        assert torch.allclose(t, torch.tensor([3.3, 4.1]))

    def test_spatial_coords_scale(self):
        pos = get_pixel_coords(
            latent_height=2,
            latent_width=3,
            latent_num_frames=1,
            frame_idx=0,
            fps=24.0,
            vae_spatial_scale=32,
        )
        # y_coords: [0, 0, 0, 1, 1, 1] * 32
        y = pos[0, 1, :]
        expected_y = torch.tensor([0, 0, 0, 32, 32, 32], dtype=torch.float32)
        assert torch.allclose(y, expected_y)

        # x_coords: [0, 1, 2, 0, 1, 2] * 32
        x = pos[0, 2, :]
        expected_x = torch.tensor([0, 32, 64, 0, 32, 64], dtype=torch.float32)
        assert torch.allclose(x, expected_x)

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
        # Apply downscale_factor=2 to spatial coords
        pos[:, 1, :] *= 2
        pos[:, 2, :] *= 2

        y = pos[0, 1, :]
        x = pos[0, 2, :]
        # y: [0, 0, 32, 32] * 2 = [0, 0, 64, 64]
        assert torch.allclose(y, torch.tensor([0, 0, 64, 64], dtype=torch.float32))
        # x: [0, 32, 0, 32] * 2 = [0, 64, 0, 64]
        assert torch.allclose(x, torch.tensor([0, 64, 0, 64], dtype=torch.float32))

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


class TestGenerationSettingsMode:
    """Test video_conditioning_mode in GenerationSettings."""

    def test_default_mode(self):
        from frameartisan.modules.generation.generation_settings import GenerationSettings

        s = GenerationSettings()
        assert s.video_conditioning_mode == "replace"
