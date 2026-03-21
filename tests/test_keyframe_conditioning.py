"""Tests for keyframe conditioning (VideoConditionByKeyframeIndex approach)."""

from __future__ import annotations

import torch

from frameartisan.modules.generation.graph.nodes.ltx2_utils import build_keyframe_attention_mask


class TestBuildKeyframeAttentionMask:
    def test_returns_none_for_empty_groups(self):
        result = build_keyframe_attention_mask(num_base_tokens=100, keyframe_group_sizes=[])
        assert result is None

    def test_single_keyframe_group_full_scale_returns_none(self):
        """Single group at full attention scale → mask is all zeros → skip (preserves flash attention)."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=10, keyframe_group_sizes=[5]
        )
        assert mask is None

    def test_single_keyframe_group_with_reduced_scale(self):
        """Single group with attention scale < 1.0 → mask IS needed."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=10, keyframe_group_sizes=[5], attention_scales=[0.5]
        )
        assert mask is not None
        assert mask.shape == (1, 15, 15)
        # Keyframe↔base cross-attention has log(0.5) bias
        import math
        bias = math.log(0.5)
        assert torch.allclose(mask[:, :10, 10:], torch.tensor(bias))
        assert torch.allclose(mask[:, 10:, :10], torch.tensor(bias))

    def test_two_keyframe_groups_full_scale_returns_none(self):
        """Multiple groups at full attention scale → None (preserves flash attention)."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=10, keyframe_group_sizes=[3, 4]
        )
        assert mask is None

    def test_two_keyframe_groups_with_reduced_scale(self):
        """Multiple groups with reduced scale → mask is created with cross-group blocking."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=10, keyframe_group_sizes=[3, 4], attention_scales=[0.8, 0.8]
        )
        assert mask.shape == (1, 17, 17)
        # Group 1 does NOT attend to group 2
        assert (mask[:, 10:13, 13:17] == -10000.0).all()
        # Group 2 does NOT attend to group 1
        assert (mask[:, 13:17, 10:13] == -10000.0).all()

    def test_three_groups_isolation(self):
        """Three groups with reduced scale → cross-group blocking."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=8, keyframe_group_sizes=[2, 3, 2], attention_scales=[0.5, 0.5, 0.5]
        )
        assert mask.shape == (1, 15, 15)
        # Groups 0 (8:10), 1 (10:13), 2 (13:15)
        # Group 0 ↔ Group 1: masked
        assert (mask[:, 8:10, 10:13] == -10000.0).all()
        assert (mask[:, 10:13, 8:10] == -10000.0).all()
        # Group 0 ↔ Group 2: masked
        assert (mask[:, 8:10, 13:15] == -10000.0).all()
        assert (mask[:, 13:15, 8:10] == -10000.0).all()
        # Group 1 ↔ Group 2: masked
        assert (mask[:, 10:13, 13:15] == -10000.0).all()
        assert (mask[:, 13:15, 10:13] == -10000.0).all()
        # Each group self-attends
        assert (mask[:, 8:10, 8:10] == 0.0).all()
        assert (mask[:, 10:13, 10:13] == 0.0).all()
        assert (mask[:, 13:15, 13:15] == 0.0).all()

    def test_symmetry(self):
        """Mask should be symmetric: if A attends to B, B attends to A."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=10, keyframe_group_sizes=[3, 4], attention_scales=[0.5, 0.5]
        )
        assert torch.equal(mask, mask.transpose(1, 2))

    def test_attention_scale_reduces_cross_attention(self):
        """attention_scales < 1.0 should produce log-space bias for base↔keyframe."""
        import math

        mask = build_keyframe_attention_mask(
            num_base_tokens=5, keyframe_group_sizes=[3], attention_scales=[0.5]
        )
        expected_bias = math.log(0.5)
        # Base → keyframe cross-attention has log(0.5) bias
        assert torch.allclose(mask[:, :5, 5:], torch.tensor(expected_bias, dtype=torch.float32))
        # Keyframe → base cross-attention has log(0.5) bias (symmetric)
        assert torch.allclose(mask[:, 5:, :5], torch.tensor(expected_bias, dtype=torch.float32))
        # Keyframe self-attention is still full (0.0)
        assert (mask[:, 5:, 5:] == 0.0).all()
        # Base self-attention is still full (0.0)
        assert (mask[:, :5, :5] == 0.0).all()

    def test_attention_scale_1_0_is_noop(self):
        """attention_scales=1.0 with single group → None (same as no scales)."""
        mask_with = build_keyframe_attention_mask(
            num_base_tokens=5, keyframe_group_sizes=[3], attention_scales=[1.0]
        )
        mask_without = build_keyframe_attention_mask(
            num_base_tokens=5, keyframe_group_sizes=[3], attention_scales=None
        )
        # Both return None for single group at full scale
        assert mask_with is None
        assert mask_without is None

    def test_attention_scale_1_0_multi_group_returns_none(self):
        """Multiple groups at full scale → None (flash attention preserved)."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=5, keyframe_group_sizes=[3, 2], attention_scales=[1.0, 1.0]
        )
        assert mask is None

    def test_per_group_attention_scales(self):
        """Different groups can have different attention scales."""
        import math

        mask = build_keyframe_attention_mask(
            num_base_tokens=5, keyframe_group_sizes=[2, 2], attention_scales=[1.0, 0.3]
        )
        # Group 0 (tokens 5:7): full attention to base (0.0)
        assert (mask[:, :5, 5:7] == 0.0).all()
        # Group 1 (tokens 7:9): reduced attention to base (log(0.3))
        expected = math.log(0.3)
        assert torch.allclose(mask[:, :5, 7:9], torch.tensor(expected, dtype=torch.float32))


class TestKeyframeIsolation:
    """Tests for the force_cross_group_isolation parameter (Isolate Keyframes checkbox)."""

    def test_no_isolation_single_group_returns_none(self):
        """Single group without isolation → None (flash attention)."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=100, keyframe_group_sizes=[50],
            force_cross_group_isolation=False,
        )
        assert mask is None

    def test_no_isolation_multi_group_returns_none(self):
        """Multiple groups at full scale without isolation → None (flash attention)."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=100, keyframe_group_sizes=[50, 50],
            force_cross_group_isolation=False,
        )
        assert mask is None

    def test_isolation_single_group_returns_none(self):
        """Single group with isolation → still None (nothing to isolate from)."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=100, keyframe_group_sizes=[50],
            force_cross_group_isolation=True,
        )
        assert mask is None

    def test_isolation_multi_group_creates_mask(self):
        """Multiple groups with isolation → mask with cross-group blocking."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=10, keyframe_group_sizes=[3, 4],
            force_cross_group_isolation=True,
        )
        assert mask is not None
        assert mask.shape == (1, 17, 17)
        # Base tokens attend to everything (0.0)
        assert (mask[:, :10, :] == 0.0).all()
        # Group 1 (10:13) does NOT attend to group 2 (13:17)
        assert (mask[:, 10:13, 13:17] == -10000.0).all()
        # Group 2 (13:17) does NOT attend to group 1 (10:13)
        assert (mask[:, 13:17, 10:13] == -10000.0).all()
        # Each group self-attends
        assert (mask[:, 10:13, 10:13] == 0.0).all()
        assert (mask[:, 13:17, 13:17] == 0.0).all()

    def test_isolation_three_groups(self):
        """Three groups with isolation → all cross-group pairs blocked."""
        mask = build_keyframe_attention_mask(
            num_base_tokens=8, keyframe_group_sizes=[2, 3, 2],
            force_cross_group_isolation=True,
        )
        assert mask is not None
        assert mask.shape == (1, 15, 15)
        # All cross-group pairs blocked
        assert (mask[:, 8:10, 10:13] == -10000.0).all()
        assert (mask[:, 10:13, 8:10] == -10000.0).all()
        assert (mask[:, 8:10, 13:15] == -10000.0).all()
        assert (mask[:, 13:15, 8:10] == -10000.0).all()
        assert (mask[:, 10:13, 13:15] == -10000.0).all()
        assert (mask[:, 13:15, 10:13] == -10000.0).all()

    def test_isolation_with_reduced_scale(self):
        """Isolation + reduced scale → both cross-group blocking and attention bias."""
        import math

        mask = build_keyframe_attention_mask(
            num_base_tokens=5, keyframe_group_sizes=[3, 2],
            attention_scales=[0.5, 1.0],
            force_cross_group_isolation=True,
        )
        assert mask is not None
        # Group 0 has log(0.5) bias with base
        bias = math.log(0.5)
        assert torch.allclose(mask[:, :5, 5:8], torch.tensor(bias))
        # Group 1 has full attention to base (0.0)
        assert (mask[:, :5, 8:10] == 0.0).all()
        # Cross-group still blocked
        assert (mask[:, 5:8, 8:10] == -10000.0).all()
        assert (mask[:, 8:10, 5:8] == -10000.0).all()

    def test_install_keyframe_attention_no_isolation(self):
        """_install_keyframe_attention with force_isolate=False and full scale → no hooks."""
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        class MockAttn(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states

        class MockBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MockAttn()

        class MockTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
                self.transformer_blocks = torch.nn.ModuleList([MockBlock()])

        transformer = MockTransformer()
        video_latents = torch.zeros(1, 13, 8)

        hooks = LTX2DenoiseNode._install_keyframe_attention(
            transformer, video_latents, [3, 3], [1.0, 1.0],
            do_cfg=False, device="cpu", force_isolate=False,
        )
        # No mask needed → no hooks installed
        assert hooks is None

    def test_install_keyframe_attention_with_isolation(self):
        """_install_keyframe_attention with force_isolate=True → hooks installed."""
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode
        from frameartisan.modules.generation.graph.nodes.ltx2_keyframe_attention import (
            remove_keyframe_mask,
        )

        class MockAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4))

            def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                self._last_mask = attention_mask
                return hidden_states

        class MockBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MockAttn()

        class MockTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
                self.transformer_blocks = torch.nn.ModuleList([MockBlock()])

        transformer = MockTransformer()
        video_latents = torch.zeros(1, 13, 8)

        hooks = LTX2DenoiseNode._install_keyframe_attention(
            transformer, video_latents, [3, 3], [1.0, 1.0],
            do_cfg=False, device="cpu", force_isolate=True,
        )
        assert hooks is not None

        # Verify mask is injected
        transformer.transformer_blocks[0].attn1(torch.zeros(1, 13, 4))
        injected_mask = transformer.transformer_blocks[0].attn1._last_mask
        assert injected_mask is not None
        assert injected_mask.shape == (1, 13, 13)

        remove_keyframe_mask(hooks)


class TestKeyframeIsolateSerialization:
    """Ensure keyframe_isolate survives graph round-trip."""

    def test_denoise_node_default(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        node = LTX2DenoiseNode()
        assert node.keyframe_isolate is False

    def test_round_trip_false(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        node = LTX2DenoiseNode()
        node.keyframe_isolate = False
        state = node.get_state()
        assert state["keyframe_isolate"] is False

        node2 = LTX2DenoiseNode()
        node2.apply_state(state)
        assert node2.keyframe_isolate is False

    def test_round_trip_true(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        node = LTX2DenoiseNode()
        node.keyframe_isolate = True
        state = node.get_state()
        assert state["keyframe_isolate"] is True

        node2 = LTX2DenoiseNode()
        node2.apply_state(state)
        assert node2.keyframe_isolate is True

    def test_change_detected_by_dict_comparison(self):
        """Changing keyframe_isolate produces a different dict for update_from_json."""
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        node = LTX2DenoiseNode()
        node.id = 0
        node.name = "denoise"
        d1 = node.to_dict()

        node.keyframe_isolate = True
        d2 = node.to_dict()

        assert d1 != d2


class TestKeyframeAttentionHooks:
    """Test install/remove of self-attention mask hooks."""

    def test_install_and_remove(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_keyframe_attention import (
            install_keyframe_mask,
            remove_keyframe_mask,
        )

        # Create a minimal mock transformer with blocks that have attn1
        class MockAttn(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states

        class MockBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MockAttn()

        class MockTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([MockBlock(), MockBlock()])

        transformer = MockTransformer()
        mask = torch.zeros(1, 5, 5)

        hooks = install_keyframe_mask(transformer, mask)
        assert len(hooks) == 2

        # Forward should be patched — the function name changes
        for block in transformer.transformer_blocks:
            assert block.attn1.forward.__name__ == "_forward_with_mask"

        remove_keyframe_mask(hooks)

        # After removal, forward name should be restored to the original
        for block in transformer.transformer_blocks:
            assert block.attn1.forward.__name__ == "forward"
            result = block.attn1(torch.zeros(1, 3, 4))
            assert result.shape == (1, 3, 4)

    def test_mask_injected_for_self_attention(self):
        """Verify that the mask is passed to self-attention (no encoder_hidden_states)."""
        from frameartisan.modules.generation.graph.nodes.ltx2_keyframe_attention import (
            install_keyframe_mask,
            remove_keyframe_mask,
        )

        received_masks = []

        class MockAttn(torch.nn.Module):
            def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                received_masks.append(attention_mask)
                return hidden_states

        class MockBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MockAttn()

        class MockTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([MockBlock()])

        transformer = MockTransformer()
        mask = torch.ones(1, 5, 5)

        hooks = install_keyframe_mask(transformer, mask)

        # Self-attention call (no encoder_hidden_states) should get the mask
        transformer.transformer_blocks[0].attn1(torch.zeros(1, 5, 32))
        assert received_masks[-1] is mask

        # Cross-attention call (with encoder_hidden_states) should NOT override
        received_masks.clear()
        transformer.transformer_blocks[0].attn1(
            torch.zeros(1, 5, 32),
            encoder_hidden_states=torch.zeros(1, 3, 32),
            attention_mask=None,
        )
        assert received_masks[-1] is None

        remove_keyframe_mask(hooks)


class TestKeyframeMaskDtype:
    """Regression: mask dtype must match model compute dtype for SDPA."""

    def test_mask_dtype_matches_model_dtype(self):
        """build_keyframe_attention_mask should respect the dtype argument."""
        # Use reduced attention scale to ensure mask is actually created
        mask_f32 = build_keyframe_attention_mask(
            num_base_tokens=4, keyframe_group_sizes=[2], attention_scales=[0.5], dtype=torch.float32
        )
        assert mask_f32.dtype == torch.float32

        mask_bf16 = build_keyframe_attention_mask(
            num_base_tokens=4, keyframe_group_sizes=[2], attention_scales=[0.5], dtype=torch.bfloat16
        )
        assert mask_bf16.dtype == torch.bfloat16

    def test_install_keyframe_attention_uses_model_dtype(self):
        """_install_keyframe_attention must create the mask in the transformer's
        parameter dtype so SDPA doesn't raise 'invalid dtype for bias'."""
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode
        from frameartisan.modules.generation.graph.nodes.ltx2_keyframe_attention import (
            remove_keyframe_mask,
        )

        # Mock transformer with bfloat16 parameters (like a real model)
        class MockAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

            def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                self._last_mask = attention_mask
                return hidden_states

        class MockBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MockAttn()

        class MockTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Need a parameter so next(transformer.parameters()) works
                self.proj = torch.nn.Linear(4, 4)
                self.proj.to(torch.bfloat16)
                self.transformer_blocks = torch.nn.ModuleList([MockBlock()])

        transformer = MockTransformer()
        video_latents = torch.zeros(1, 13, 8)  # 13 total tokens
        # Use reduced attention scale so mask is actually created
        keyframe_group_sizes = [3, 3]  # 6 keyframe tokens, 7 base

        hooks = LTX2DenoiseNode._install_keyframe_attention(
            transformer, video_latents, keyframe_group_sizes, [0.8, 0.8], do_cfg=False, device="cpu"
        )

        # Trigger the hooked forward to capture the mask
        transformer.transformer_blocks[0].attn1(torch.zeros(1, 13, 4))
        injected_mask = transformer.transformer_blocks[0].attn1._last_mask

        assert injected_mask is not None
        assert injected_mask.dtype == torch.bfloat16, (
            f"Mask dtype {injected_mask.dtype} != model dtype bfloat16 — "
            "SDPA will raise 'invalid dtype for bias'"
        )

        remove_keyframe_mask(hooks)
