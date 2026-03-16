"""Tests for LoRA spatiotemporal masking."""

from __future__ import annotations

import torch
import torch.nn as nn

from frameartisan.modules.generation.graph.nodes.lora_mask import (
    _PATCHED_LAYERS,
    _get_lora_layers_for_adapter,
    build_combined_mask,
    build_temporal_weights,
    get_patched_layer_count,
    patch_lora_adapter_with_mask,
    patch_lora_layer_with_mask,
    unpatch_all_lora_layers,
)


# ---------------------------------------------------------------------------
# build_temporal_weights
# ---------------------------------------------------------------------------


class TestBuildTemporalWeights:
    def test_full_range_no_fade(self):
        w = build_temporal_weights(10, start_frame=0, end_frame=9)
        assert w.shape == (10,)
        assert torch.allclose(w, torch.ones(10))

    def test_partial_range(self):
        w = build_temporal_weights(10, start_frame=3, end_frame=6)
        assert w[:3].sum() == 0.0
        assert w[7:].sum() == 0.0
        assert torch.allclose(w[3:7], torch.ones(4))

    def test_end_frame_minus_one_means_last(self):
        w = build_temporal_weights(8, start_frame=0, end_frame=-1)
        assert w.shape == (8,)
        assert torch.allclose(w, torch.ones(8))

    def test_fade_in(self):
        w = build_temporal_weights(10, start_frame=0, end_frame=9, fade_in_frames=4)
        assert w[0].item() == 0.0
        assert 0.0 < w[1].item() < 1.0
        assert 0.0 < w[2].item() < 1.0
        assert 0.0 < w[3].item() < 1.0
        assert w[4].item() == 1.0
        assert w[9].item() == 1.0

    def test_fade_out(self):
        w = build_temporal_weights(10, start_frame=0, end_frame=9, fade_out_frames=3)
        assert w[0].item() == 1.0
        assert w[6].item() == 1.0
        assert 0.0 < w[7].item() < 1.0
        assert 0.0 < w[8].item() < 1.0
        assert w[9].item() == 0.0

    def test_fade_in_and_out(self):
        w = build_temporal_weights(10, start_frame=0, end_frame=9, fade_in_frames=3, fade_out_frames=3)
        assert w[0].item() == 0.0
        assert w[9].item() == 0.0
        assert w[4].item() == 1.0  # middle fully active

    def test_outside_range_is_zero(self):
        w = build_temporal_weights(20, start_frame=5, end_frame=10)
        assert w[:5].sum() == 0.0
        assert w[11:].sum() == 0.0

    def test_start_after_end_all_zero(self):
        w = build_temporal_weights(10, start_frame=7, end_frame=3)
        assert w.sum() == 0.0

    def test_clamps_to_valid_range(self):
        w = build_temporal_weights(5, start_frame=-5, end_frame=100)
        assert w.shape == (5,)
        assert torch.allclose(w, torch.ones(5))


# ---------------------------------------------------------------------------
# build_combined_mask
# ---------------------------------------------------------------------------


class TestBuildCombinedMask:
    def test_returns_none_when_both_none(self):
        result = build_combined_mask(None, None, 4, 8, 8)
        assert result is None

    def test_spatial_only(self):
        spatial = torch.ones(1, 1, 16, 16)
        spatial[:, :, :, 8:] = 0.0  # right half masked

        mask = build_combined_mask(spatial, None, 4, 8, 8)
        assert mask.shape == (1, 4 * 8 * 8, 1)

        # Reshape to [F, H, W] to verify spatial pattern tiled across frames
        reshaped = mask.view(4, 8, 8)
        for f in range(4):
            assert reshaped[f, :, :4].min() > 0.0  # left half active
            assert reshaped[f, :, 4:].max() == 0.0  # right half masked

    def test_temporal_only(self):
        temporal = torch.tensor([0.0, 0.5, 1.0, 1.0])
        mask = build_combined_mask(None, temporal, 4, 8, 8)
        assert mask.shape == (1, 4 * 8 * 8, 1)

        reshaped = mask.view(4, 8, 8)
        assert reshaped[0].sum() == 0.0
        assert reshaped[2].min() == 1.0
        assert reshaped[3].min() == 1.0
        # Frame 1 should be 0.5 everywhere
        assert torch.allclose(reshaped[1], torch.full((8, 8), 0.5))

    def test_combined_spatial_and_temporal(self):
        spatial = torch.zeros(1, 1, 8, 8)
        spatial[:, :, :4, :] = 1.0  # top half active

        temporal = torch.tensor([0.0, 1.0, 1.0, 0.0])

        mask = build_combined_mask(spatial, temporal, 4, 8, 8)
        reshaped = mask.view(4, 8, 8)

        # Frame 0 and 3: temporal=0, so all zero
        assert reshaped[0].sum() == 0.0
        assert reshaped[3].sum() == 0.0

        # Frame 1: temporal=1, so top half=1, bottom half=0
        assert reshaped[1, :4, :].min() == 1.0
        assert reshaped[1, 4:, :].max() == 0.0

    def test_spatial_resize(self):
        """Spatial mask at different resolution gets resized."""
        spatial = torch.ones(1, 1, 32, 32)  # larger than latent
        mask = build_combined_mask(spatial, None, 2, 4, 4)
        assert mask.shape == (1, 2 * 4 * 4, 1)
        assert torch.allclose(mask, torch.ones_like(mask))


# ---------------------------------------------------------------------------
# Layer patching
# ---------------------------------------------------------------------------


class _MockLoraLayer(nn.Module):
    """Simple linear layer to simulate a LoRA layer."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0  # simple transformation for testing


class _MockTransformer(nn.Module):
    """Transformer with fake LoRA layers for testing."""

    def __init__(self):
        super().__init__()
        self.lora_a_test_adapter = _MockLoraLayer()
        self.lora_b_test_adapter = _MockLoraLayer()

    def get_lora_layers(self, adapter_name: str) -> dict[str, nn.Module]:
        if adapter_name == "test_adapter":
            return {
                "lora_a": self.lora_a_test_adapter,
                "lora_b": self.lora_b_test_adapter,
            }
        return {}


class TestPatchLoraLayer:
    def setup_method(self):
        _PATCHED_LAYERS.clear()

    def teardown_method(self):
        unpatch_all_lora_layers()

    def test_patch_masks_output(self):
        layer = _MockLoraLayer()
        F, H, W = 2, 4, 4

        # Mask that zeros out the right half
        mask = torch.zeros(1, F * H * W, 1)
        mask[:, :F * H * (W // 2), :] = 1.0  # only first half of each row

        patch_lora_layer_with_mask(layer, mask, (F, H, W))
        assert get_patched_layer_count() == 1

        x = torch.ones(1, F * H * W, 8)
        out = layer(x)

        # Original forward doubles: x*2. Mask zeros right half.
        assert out.shape == (1, F * H * W, 8)
        # Verify some masked positions are zero
        assert out[0, 0, 0].item() == 2.0  # unmasked
        # Check a position in the second half (masked)
        assert out[0, F * H * W - 1, 0].item() == 0.0

    def test_skips_2d_tensors(self):
        layer = _MockLoraLayer()
        mask = torch.ones(1, 32, 1)
        patch_lora_layer_with_mask(layer, mask, (2, 4, 4))

        x = torch.ones(4, 8)  # 2D — should pass through unmasked
        out = layer(x)
        assert torch.allclose(out, x * 2.0)

    def test_joint_attention_text_tokens_unmasked(self):
        layer = _MockLoraLayer()
        F, H, W = 2, 4, 4
        num_video = F * H * W

        # Mask that zeros all video tokens
        mask = torch.zeros(1, num_video, 1)
        patch_lora_layer_with_mask(layer, mask, (F, H, W))

        # Input with extra text tokens
        text_tokens = 10
        x = torch.ones(1, num_video + text_tokens, 8)
        out = layer(x)

        # Video tokens should be zeroed
        assert out[0, :num_video, 0].max() == 0.0
        # Text tokens should be unmasked (multiplied by 1.0)
        assert out[0, num_video:, 0].min() == 2.0

    def test_unknown_sequence_length_passes_through(self):
        """When N < expected video tokens, output is unchanged."""
        layer = _MockLoraLayer()
        mask = torch.ones(1, 100, 1)
        patch_lora_layer_with_mask(layer, mask, (10, 10, 10))  # expects 1000 tokens

        x = torch.ones(1, 50, 8)  # 50 < 1000
        out = layer(x)
        assert torch.allclose(out, x * 2.0)  # passes through

    def test_batch_expansion(self):
        """Mask with batch=1 expands to match input batch."""
        layer = _MockLoraLayer()
        F, H, W = 1, 2, 2
        mask = torch.ones(1, F * H * W, 1) * 0.5
        patch_lora_layer_with_mask(layer, mask, (F, H, W))

        x = torch.ones(3, F * H * W, 8)  # batch=3
        out = layer(x)
        assert torch.allclose(out, x * 2.0 * 0.5)


class TestPatchAdapter:
    def setup_method(self):
        _PATCHED_LAYERS.clear()

    def teardown_method(self):
        unpatch_all_lora_layers()

    def test_patches_all_layers_for_adapter(self):
        transformer = _MockTransformer()
        mask = torch.ones(1, 32, 1)
        latent_dims = (2, 4, 4)

        count = patch_lora_adapter_with_mask(transformer, "test_adapter", mask, latent_dims)
        assert count == 2
        assert get_patched_layer_count() == 2

    def test_no_layers_for_unknown_adapter(self):
        transformer = _MockTransformer()
        mask = torch.ones(1, 32, 1)
        count = patch_lora_adapter_with_mask(transformer, "nonexistent", mask, (2, 4, 4))
        assert count == 0


class TestUnpatch:
    def setup_method(self):
        _PATCHED_LAYERS.clear()

    def test_unpatch_restores_original(self):
        layer = _MockLoraLayer()
        original_output = layer(torch.ones(1, 8, 8))

        mask = torch.zeros(1, 8, 1)  # zero mask
        patch_lora_layer_with_mask(layer, mask, (1, 2, 4))
        assert get_patched_layer_count() == 1

        count = unpatch_all_lora_layers()
        assert count == 1
        assert get_patched_layer_count() == 0

        restored_output = layer(torch.ones(1, 8, 8))
        assert torch.allclose(original_output, restored_output)

    def test_unpatch_empty_is_noop(self):
        count = unpatch_all_lora_layers()
        assert count == 0


class TestGetLoraLayers:
    def test_fixture_path(self):
        transformer = _MockTransformer()
        layers = _get_lora_layers_for_adapter(transformer, "test_adapter")
        assert len(layers) == 2

    def test_fixture_unknown_adapter(self):
        transformer = _MockTransformer()
        layers = _get_lora_layers_for_adapter(transformer, "unknown")
        assert len(layers) == 0
