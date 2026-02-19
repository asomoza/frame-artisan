"""Tests for the denoise node's x0 blending and audio conditioning logic."""

from __future__ import annotations

import torch
import pytest

from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode


class TestDenoiseNodeMetadata:
    def test_optional_inputs_include_clean_latents(self):
        assert "clean_latents" in LTX2DenoiseNode.OPTIONAL_INPUTS

    def test_optional_inputs_include_clean_audio_latents(self):
        assert "clean_audio_latents" in LTX2DenoiseNode.OPTIONAL_INPUTS

    def test_optional_inputs_include_conditioning_mask(self):
        assert "conditioning_mask" in LTX2DenoiseNode.OPTIONAL_INPUTS


class TestX0Blending:
    """Test the x0-space blending math used in generalized conditioning."""

    def test_x0_blending_no_mask(self):
        """With zero mask, x0 blended = x0 original → no change to velocity."""
        latents = torch.randn(1, 10, 16)
        noise_pred = torch.randn(1, 10, 16)
        clean = torch.randn(1, 10, 16)
        sigma = torch.tensor(0.5)

        mask = torch.zeros(1, 10)  # no conditioning

        x0 = latents - sigma * noise_pred
        mask_3d = mask.unsqueeze(-1)
        x0_blended = x0 * (1 - mask_3d) + clean * mask_3d
        noise_pred_blended = (latents - x0_blended) / sigma

        # With zero mask, blended velocity should equal original
        torch.testing.assert_close(noise_pred_blended, noise_pred)

    def test_x0_blending_full_mask(self):
        """With full mask, x0 blended = clean latents."""
        latents = torch.randn(1, 10, 16)
        noise_pred = torch.randn(1, 10, 16)
        clean = torch.randn(1, 10, 16)
        sigma = torch.tensor(0.5)

        mask = torch.ones(1, 10)  # full conditioning

        x0 = latents - sigma * noise_pred
        mask_3d = mask.unsqueeze(-1)
        x0_blended = x0 * (1 - mask_3d) + clean * mask_3d

        # With full mask, x0_blended should equal clean
        torch.testing.assert_close(x0_blended, clean)

    def test_x0_blending_partial_mask(self):
        """With partial mask (0.5 strength), x0 blended = average of x0 and clean."""
        latents = torch.randn(1, 10, 16)
        noise_pred = torch.randn(1, 10, 16)
        clean = torch.randn(1, 10, 16)
        sigma = torch.tensor(0.5)

        mask = torch.full((1, 10), 0.5)

        x0 = latents - sigma * noise_pred
        mask_3d = mask.unsqueeze(-1)
        x0_blended = x0 * (1 - mask_3d) + clean * mask_3d

        expected = 0.5 * x0 + 0.5 * clean
        torch.testing.assert_close(x0_blended, expected)

    def test_x0_blending_per_token_mask(self):
        """Different tokens can have different mask values."""
        latents = torch.randn(1, 4, 8)
        noise_pred = torch.randn(1, 4, 8)
        clean = torch.randn(1, 4, 8)
        sigma = torch.tensor(0.5)

        mask = torch.tensor([[1.0, 0.0, 0.5, 0.0]])  # [1, 4]

        x0 = latents - sigma * noise_pred
        mask_3d = mask.unsqueeze(-1)
        x0_blended = x0 * (1 - mask_3d) + clean * mask_3d

        # Token 0: fully clean
        torch.testing.assert_close(x0_blended[:, 0], clean[:, 0])
        # Token 1: fully x0
        torch.testing.assert_close(x0_blended[:, 1], x0[:, 1])
        # Token 2: 50/50
        expected_2 = 0.5 * x0[:, 2] + 0.5 * clean[:, 2]
        torch.testing.assert_close(x0_blended[:, 2], expected_2)


class TestTimestepMasking:
    """Test per-token timestep masking for conditioning."""

    def test_conditioned_tokens_see_zero_timestep(self):
        timestep = torch.tensor([0.8])
        mask = torch.tensor([[1.0, 0.0, 0.5, 0.0]])  # [1, 4]

        video_timestep = timestep.unsqueeze(-1) * (1 - mask)

        # Token 0 (mask=1.0): sees timestep 0
        assert video_timestep[0, 0].item() == pytest.approx(0.0)
        # Token 1 (mask=0.0): sees full timestep
        assert video_timestep[0, 1].item() == pytest.approx(0.8)
        # Token 2 (mask=0.5): sees half timestep
        assert video_timestep[0, 2].item() == pytest.approx(0.4)
        # Token 3 (mask=0.0): sees full timestep
        assert video_timestep[0, 3].item() == pytest.approx(0.8)

    def test_audio_timestep_zero_for_clean_audio(self):
        """When clean audio is provided, audio_timestep should be zeros."""
        timestep = torch.tensor([0.8])
        audio_timestep = torch.zeros_like(timestep)
        assert audio_timestep.item() == 0.0
