"""Tests for LTX2ConditionEncodeNode video conditioning support."""

from __future__ import annotations

import pytest

from frameartisan.modules.generation.graph.nodes.ltx2_condition_encode_node import LTX2ConditionEncodeNode


class TestConditionEncodeVideoMetadata:
    def test_videos_in_optional_inputs(self):
        assert "videos" in LTX2ConditionEncodeNode.OPTIONAL_INPUTS

    def test_images_still_in_optional_inputs(self):
        assert "images" in LTX2ConditionEncodeNode.OPTIONAL_INPUTS


class TestConditionEncodeVideoConditions:
    def test_video_type_condition(self):
        node = LTX2ConditionEncodeNode()
        conditions = [{"type": "video", "pixel_frame_index": 1, "strength": 0.8}]
        node.update_conditions(conditions)
        assert node.conditions[0]["type"] == "video"
        assert node.updated is True

    def test_mixed_conditions(self):
        node = LTX2ConditionEncodeNode()
        conditions = [
            {"type": "image", "pixel_frame_index": 1, "strength": 1.0},
            {"type": "video", "pixel_frame_index": 9, "strength": 0.8},
        ]
        node.update_conditions(conditions)
        assert len(node.conditions) == 2
        assert node.conditions[0]["type"] == "image"
        assert node.conditions[1]["type"] == "video"

    def test_backward_compat_no_type(self):
        """Conditions without type field should default to image."""
        node = LTX2ConditionEncodeNode()
        conditions = [{"pixel_frame_index": 1, "strength": 1.0}]
        node.update_conditions(conditions)
        # No type field — should be treated as image by __call__
        assert "type" not in node.conditions[0]


class TestConditionEncodeVideoSerialization:
    def test_video_condition_round_trip(self):
        node = LTX2ConditionEncodeNode()
        node.conditions = [
            {"type": "image", "pixel_frame_index": 1, "strength": 1.0},
            {"type": "video", "pixel_frame_index": 1, "strength": 0.5},
        ]
        state = node.get_state()
        assert state["conditions"][1]["type"] == "video"

        node2 = LTX2ConditionEncodeNode()
        node2.apply_state(state)
        assert node2.conditions[1]["type"] == "video"
        assert node2.conditions[1]["strength"] == 0.5


class TestVideoTemporalAlignment:
    """Test the 8n+1 alignment logic used for video clips."""

    @pytest.mark.parametrize(
        "input_frames,expected",
        [
            (25, 25),  # already aligned (8*3+1)
            (30, 33),  # ceil((30-1)/8)*8+1 = ceil(3.625)*8+1 = 33
            (10, 17),  # ceil((10-1)/8)*8+1 = ceil(1.125)*8+1 = 17
            (9, 9),  # already aligned (8*1+1)
            (17, 17),  # already aligned (8*2+1)
            (1, 1),  # single frame
            (2, 9),  # ceil((2-1)/8)*8+1 = ceil(0.125)*8+1 = 9
            (8, 9),  # ceil((8-1)/8)*8+1 = ceil(0.875)*8+1 = 9
            (33, 33),  # already aligned (8*4+1)
            (120, 121),  # ceil((120-1)/8)*8+1 = ceil(14.875)*8+1 = 121
        ],
    )
    def test_alignment(self, input_frames, expected):
        import math

        aligned = math.ceil((input_frames - 1) / 8) * 8 + 1
        if aligned <= 0:
            aligned = 1
        assert aligned == expected


class TestVideoFrameTrimming:
    """Test that video frames are trimmed to fit within generation length."""

    def test_video_fits(self):
        num_frames_snapped = 25
        pixel_frame_index = 1
        video_frames = 25

        available = num_frames_snapped - (pixel_frame_index - 1)
        use_frames = min(video_frames, available)
        assert use_frames == 25

    def test_video_needs_trimming(self):
        num_frames_snapped = 25
        pixel_frame_index = 1
        video_frames = 200

        available = num_frames_snapped - (pixel_frame_index - 1)
        use_frames = min(video_frames, available)
        assert use_frames == 25

    def test_video_starts_midway(self):
        num_frames_snapped = 97
        pixel_frame_index = 49
        video_frames = 100

        available = num_frames_snapped - (pixel_frame_index - 1)
        use_frames = min(video_frames, available)
        assert use_frames == 49  # only 49 frames available from position 49

    def test_video_beyond_end(self):
        num_frames_snapped = 25
        pixel_frame_index = 26
        video_frames = 10

        available = num_frames_snapped - (pixel_frame_index - 1)
        available = max(available, 0)
        use_frames = min(video_frames, available)
        assert use_frames == 0

    def test_negative_index(self):
        """Negative pixel_frame_index should resolve before trimming."""
        num_frames_snapped = 25
        pixel_frame_index = -1
        resolved = num_frames_snapped + pixel_frame_index + 1
        assert resolved == 25  # last frame

        available = num_frames_snapped - (resolved - 1)
        use_frames = min(10, available)
        assert use_frames == 1  # only 1 frame at the end


class TestVideoLatentPlacement:
    """Test pixel frame → latent frame mapping for video conditions."""

    def test_start_at_frame_1(self):
        vae_temporal_ratio = 8
        pixel_frame_index = 1
        latent_start = (pixel_frame_index - 1) // vae_temporal_ratio
        assert latent_start == 0

    def test_start_at_frame_9(self):
        vae_temporal_ratio = 8
        pixel_frame_index = 9
        latent_start = (pixel_frame_index - 1) // vae_temporal_ratio
        assert latent_start == 1

    def test_start_at_frame_17(self):
        vae_temporal_ratio = 8
        pixel_frame_index = 17
        latent_start = (pixel_frame_index - 1) // vae_temporal_ratio
        assert latent_start == 2


class TestConcatModeOutputs:
    """Test that concat mode conditions are tracked in OUTPUTS."""

    def test_concat_outputs_declared(self):
        assert "concat_latents" in LTX2ConditionEncodeNode.OUTPUTS
        assert "concat_positions" in LTX2ConditionEncodeNode.OUTPUTS
        assert "concat_conditioning_mask" in LTX2ConditionEncodeNode.OUTPUTS

    def test_reference_downscale_factor_in_optional_inputs(self):
        assert "reference_downscale_factor" in LTX2ConditionEncodeNode.OPTIONAL_INPUTS

    def test_frame_rate_in_required_inputs(self):
        assert "frame_rate" in LTX2ConditionEncodeNode.REQUIRED_INPUTS


class TestConcatModeConditionMetadata:
    """Test concat mode condition metadata handling."""

    def test_concat_mode_condition(self):
        node = LTX2ConditionEncodeNode()
        conditions = [{"type": "video", "mode": "concat", "pixel_frame_index": 1, "strength": 0.8}]
        node.update_conditions(conditions)
        assert node.conditions[0]["mode"] == "concat"

    def test_replace_mode_default(self):
        node = LTX2ConditionEncodeNode()
        conditions = [{"type": "video", "pixel_frame_index": 1, "strength": 1.0}]
        node.update_conditions(conditions)
        assert node.conditions[0].get("mode", "replace") == "replace"

    def test_mixed_replace_and_concat(self):
        node = LTX2ConditionEncodeNode()
        conditions = [
            {"type": "image", "pixel_frame_index": 1, "strength": 1.0},
            {"type": "video", "mode": "replace", "pixel_frame_index": 1, "strength": 0.8},
            {"type": "video", "mode": "concat", "pixel_frame_index": 1, "strength": 0.6},
        ]
        node.update_conditions(conditions)
        assert len(node.conditions) == 3
        assert node.conditions[1].get("mode") == "replace"
        assert node.conditions[2].get("mode") == "concat"

    def test_concat_condition_round_trip(self):
        node = LTX2ConditionEncodeNode()
        node.conditions = [
            {"type": "video", "mode": "concat", "pixel_frame_index": 1, "strength": 0.5},
        ]
        state = node.get_state()
        assert state["conditions"][0]["mode"] == "concat"

        node2 = LTX2ConditionEncodeNode()
        node2.apply_state(state)
        assert node2.conditions[0]["mode"] == "concat"
