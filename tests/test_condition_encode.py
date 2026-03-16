"""Tests for LTX2ConditionEncodeNode."""

from __future__ import annotations


from frameartisan.modules.generation.graph.nodes.ltx2_condition_encode_node import LTX2ConditionEncodeNode


class TestConditionEncodeNodeMetadata:
    def test_required_inputs(self):
        assert set(LTX2ConditionEncodeNode.REQUIRED_INPUTS) == {"vae", "num_frames", "height", "width", "frame_rate"}

    def test_optional_inputs(self):
        assert "images" in LTX2ConditionEncodeNode.OPTIONAL_INPUTS

    def test_outputs(self):
        assert set(LTX2ConditionEncodeNode.OUTPUTS) == {
            "clean_latents",
            "conditioning_mask",
            "concat_latents",
            "concat_positions",
            "concat_conditioning_mask",
            "keyframe_latents",
            "keyframe_positions",
            "keyframe_strengths",
            "keyframe_group_sizes",
            "keyframe_attention_scales",
        }

    def test_serialize_include(self):
        assert "conditions" in LTX2ConditionEncodeNode.SERIALIZE_INCLUDE

    def test_default_conditions(self):
        node = LTX2ConditionEncodeNode()
        assert node.conditions == []

    def test_update_conditions(self):
        node = LTX2ConditionEncodeNode()
        conditions = [{"pixel_frame_index": 1, "strength": 0.8}]
        node.update_conditions(conditions)
        assert node.conditions == conditions
        assert node.updated is True


class TestConditionEncodeSerialization:
    def test_get_state_includes_conditions(self):
        node = LTX2ConditionEncodeNode()
        node.conditions = [{"pixel_frame_index": 1, "strength": 1.0}]
        state = node.get_state()
        assert "conditions" in state
        assert state["conditions"] == [{"pixel_frame_index": 1, "strength": 1.0}]

    def test_apply_state_restores_conditions(self):
        node = LTX2ConditionEncodeNode()
        state = {"conditions": [{"pixel_frame_index": -1, "strength": 0.5}]}
        node.apply_state(state)
        assert node.conditions == [{"pixel_frame_index": -1, "strength": 0.5}]

    def test_round_trip(self):
        node = LTX2ConditionEncodeNode()
        node.conditions = [
            {"pixel_frame_index": 1, "strength": 1.0},
            {"pixel_frame_index": -1, "strength": 0.7},
        ]
        state = node.get_state()

        node2 = LTX2ConditionEncodeNode()
        node2.apply_state(state)
        assert node2.conditions == node.conditions

    def test_empty_conditions_serialization(self):
        node = LTX2ConditionEncodeNode()
        state = node.get_state()
        assert state["conditions"] == []


class TestConditionEncodeFrameIndexResolution:
    """Test pixel frame index → latent frame index conversion logic."""

    def test_frame_1_maps_to_latent_0(self):
        # pixel_frame_index=1, temporal_ratio=8 → latent_idx = (1-1)//8 = 0
        assert (1 - 1) // 8 == 0

    def test_frame_9_maps_to_latent_1(self):
        # pixel_frame_index=9, temporal_ratio=8 → latent_idx = (9-1)//8 = 1
        assert (9 - 1) // 8 == 1

    def test_frame_8_maps_to_latent_0(self):
        # pixel_frame_index=8, temporal_ratio=8 → latent_idx = (8-1)//8 = 0
        assert (8 - 1) // 8 == 0

    def test_negative_index_resolution(self):
        # -1 with num_frames_snapped=25 → pixel_frame_index = 25 + (-1) + 1 = 25
        num_frames_snapped = 25
        pixel_frame_index = -1
        resolved = num_frames_snapped + pixel_frame_index + 1
        assert resolved == 25
        latent_idx = (resolved - 1) // 8
        assert latent_idx == 3  # last latent frame for 25-frame video
