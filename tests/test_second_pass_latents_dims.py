"""Regression tests for LTX2SecondPassLatentsNode dimension handling.

The latent upsampler only doubles spatial dimensions (H, W), NOT temporal (F).
The second_pass_latents node must use latent_num_frames (not latent_num_frames // 2)
when unpacking clean_latents and conditioning_mask at pre-upsample dimensions.

See: shape '[1, 30, 24, 24, ...]' invalid error when enabling 2nd stage with I2V.
"""

from __future__ import annotations

import pytest
import torch

from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import (
    LTX2SecondPassLatentsNode,
)
from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents


class _ValueNode:
    """Minimal stand-in for a graph node with .values and .enabled."""

    def __init__(self, **kwargs):
        self.values = kwargs
        self.enabled = True
        self.dependents = []


def _connect_node(node, input_name, source_node, output_name):
    node.connections[input_name] = [(source_node, output_name)]
    if source_node not in node.dependencies:
        node.dependencies.append(source_node)


def _make_mock_transformer():
    """Create a mock transformer with rope.prepare_video_coords."""
    from unittest.mock import MagicMock

    transformer = MagicMock()

    def fake_prepare_video_coords(batch, f, h, w, device, fps=None):
        return torch.zeros(batch, 3, f * h * w, 2)

    transformer.rope.prepare_video_coords = fake_prepare_video_coords
    return transformer


def _make_mock_vae():
    """Create a mock VAE with latents_mean/std and config."""
    from unittest.mock import MagicMock

    vae = MagicMock()
    vae.latents_mean = torch.zeros(128)
    vae.latents_std = torch.ones(128)
    vae.config.scaling_factor = 1.0
    return vae


def _make_mock_audio_vae():
    from unittest.mock import MagicMock

    audio_vae = MagicMock()
    audio_vae.latents_mean = torch.zeros(128)
    audio_vae.latents_std = torch.ones(128)
    return audio_vae


class TestSecondPassLatentsDimensionsI2V:
    """Verify correct dimension handling when upsampler only doubles spatial dims."""

    # First-pass dimensions
    LATENT_FRAMES = 61
    LATENT_H_ORIG = 24
    LATENT_W_ORIG = 24
    LATENT_CHANNELS = 128
    AUDIO_NUM_FRAMES = 501

    # Post-upsample dimensions (2x spatial only)
    LATENT_H_UP = 48
    LATENT_W_UP = 48

    def _build_node(self, *, has_conditioning=True, has_clean_latents=True, has_audio=False):
        """Build a connected SecondPassLatentsNode with mock inputs."""
        node = LTX2SecondPassLatentsNode()
        node.device = "cpu"
        node.dtype = torch.bfloat16
        node.updated = True
        node.enabled = True

        transformer = _make_mock_transformer()
        vae = _make_mock_vae()
        audio_vae = _make_mock_audio_vae()

        # Video latents: packed at POST-upsample dimensions
        video_5d = torch.randn(1, self.LATENT_CHANNELS, self.LATENT_FRAMES, self.LATENT_H_UP, self.LATENT_W_UP)
        video_packed = pack_latents(video_5d)

        # Audio latents: packed
        audio_packed = torch.randn(1, self.AUDIO_NUM_FRAMES, self.LATENT_CHANNELS)
        audio_coords = torch.zeros(1, 1, self.AUDIO_NUM_FRAMES, 2)

        sources = {
            "transformer": _ValueNode(transformer=transformer),
            "vae": _ValueNode(vae=vae),
            "audio_vae": _ValueNode(audio_vae=audio_vae),
            "video_latents": _ValueNode(video_latents=video_packed),
            "audio_latents": _ValueNode(audio_latents=audio_packed),
            "audio_coords": _ValueNode(audio_coords=audio_coords),
            "latent_num_frames": _ValueNode(latent_num_frames=self.LATENT_FRAMES),
            "latent_height": _ValueNode(latent_height=self.LATENT_H_UP),
            "latent_width": _ValueNode(latent_width=self.LATENT_W_UP),
            "audio_num_frames": _ValueNode(audio_num_frames=self.AUDIO_NUM_FRAMES),
            "frame_rate": _ValueNode(frame_rate=24),
            "seed": _ValueNode(seed=42),
            "model_type": _ValueNode(model_type=2),
        }

        for input_name, src in sources.items():
            _connect_node(node, input_name, src, input_name)

        # Conditioning mask: packed at ORIGINAL (pre-upsample) dimensions
        if has_conditioning:
            mask_5d = torch.zeros(1, 1, self.LATENT_FRAMES, self.LATENT_H_ORIG, self.LATENT_W_ORIG)
            mask_5d[:, :, 0] = 1.0  # first frame conditioned
            mask_packed = pack_latents(mask_5d).squeeze(-1)
            _connect_node(node, "conditioning_mask", _ValueNode(conditioning_mask=mask_packed), "conditioning_mask")
        else:
            _connect_node(node, "conditioning_mask", _ValueNode(conditioning_mask=None), "conditioning_mask")

        # Clean latents: packed at ORIGINAL dimensions
        if has_clean_latents and has_conditioning:
            clean_5d = torch.randn(1, self.LATENT_CHANNELS, self.LATENT_FRAMES, self.LATENT_H_ORIG, self.LATENT_W_ORIG)
            clean_packed = pack_latents(clean_5d)
            _connect_node(node, "clean_latents", _ValueNode(clean_latents=clean_packed), "clean_latents")
        else:
            _connect_node(node, "clean_latents", _ValueNode(clean_latents=None), "clean_latents")

        if has_audio:
            clean_audio = torch.randn(1, self.AUDIO_NUM_FRAMES, self.LATENT_CHANNELS)
            audio_mask = torch.zeros(1, self.AUDIO_NUM_FRAMES)
            _connect_node(node, "clean_audio_latents", _ValueNode(clean_audio_latents=clean_audio), "clean_audio_latents")
            _connect_node(node, "audio_conditioning_mask", _ValueNode(audio_conditioning_mask=audio_mask), "audio_conditioning_mask")
        else:
            _connect_node(node, "clean_audio_latents", _ValueNode(clean_audio_latents=None), "clean_audio_latents")
            _connect_node(node, "audio_conditioning_mask", _ValueNode(audio_conditioning_mask=None), "audio_conditioning_mask")

        return node

    def test_i2v_with_clean_latents_no_reshape_error(self):
        """I2V 2nd pass must not fail on reshape when unpacking clean_latents.

        This is the exact regression: latent_num_frames // 2 was wrong because
        the upsampler only doubles spatial dims, not temporal.
        """
        node = self._build_node(has_conditioning=True, has_clean_latents=True)
        # This would raise "shape '[1, 30, 24, 24, ...]' is invalid" before the fix
        result = node()
        assert result is not None

    def test_i2v_output_dimensions_correct(self):
        """Output latents must have post-upsample spatial dimensions."""
        node = self._build_node(has_conditioning=True, has_clean_latents=True)
        result = node()

        expected_seq_len = self.LATENT_FRAMES * self.LATENT_H_UP * self.LATENT_W_UP
        assert result["video_latents"].shape == (1, expected_seq_len, self.LATENT_CHANNELS)
        assert result["latent_num_frames"] == self.LATENT_FRAMES
        assert result["latent_height"] == self.LATENT_H_UP
        assert result["latent_width"] == self.LATENT_W_UP

    def test_i2v_clean_latents_upsampled_to_new_spatial(self):
        """clean_latents output should be at post-upsample dimensions."""
        node = self._build_node(has_conditioning=True, has_clean_latents=True)
        result = node()

        expected_seq_len = self.LATENT_FRAMES * self.LATENT_H_UP * self.LATENT_W_UP
        assert result["clean_latents"] is not None
        assert result["clean_latents"].shape == (1, expected_seq_len, self.LATENT_CHANNELS)

    def test_i2v_conditioning_mask_upsampled(self):
        """conditioning_mask output should be at post-upsample dimensions."""
        node = self._build_node(has_conditioning=True, has_clean_latents=True)
        result = node()

        expected_seq_len = self.LATENT_FRAMES * self.LATENT_H_UP * self.LATENT_W_UP
        assert result["conditioning_mask"] is not None
        assert result["conditioning_mask"].shape == (1, expected_seq_len)

    def test_t2v_no_conditioning_still_works(self):
        """T2V path (no conditioning) should not be affected."""
        node = self._build_node(has_conditioning=False, has_clean_latents=False)
        result = node()

        expected_seq_len = self.LATENT_FRAMES * self.LATENT_H_UP * self.LATENT_W_UP
        assert result["video_latents"].shape == (1, expected_seq_len, self.LATENT_CHANNELS)
        assert result["conditioning_mask"] is None
        assert result["clean_latents"] is None

    def test_i2v_conditioning_mask_without_clean_latents(self):
        """Legacy path: conditioning present but no clean_latents."""
        node = self._build_node(has_conditioning=True, has_clean_latents=False)
        result = node()

        expected_seq_len = self.LATENT_FRAMES * self.LATENT_H_UP * self.LATENT_W_UP
        assert result["conditioning_mask"] is not None
        assert result["conditioning_mask"].shape == (1, expected_seq_len)
