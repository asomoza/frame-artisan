"""Tests for LTX2AudioEncodeNode."""

from __future__ import annotations


from frameartisan.modules.generation.graph.nodes.ltx2_audio_encode_node import LTX2AudioEncodeNode


class TestAudioEncodeNodeMetadata:
    def test_required_inputs(self):
        assert set(LTX2AudioEncodeNode.REQUIRED_INPUTS) == {"audio_vae", "num_frames", "frame_rate"}

    def test_outputs(self):
        assert LTX2AudioEncodeNode.OUTPUTS == ["clean_audio_latents", "audio_conditioning_mask", "audio_path"]

    def test_serialize_include(self):
        assert "audio_path" in LTX2AudioEncodeNode.SERIALIZE_INCLUDE

    def test_default_audio_path(self):
        node = LTX2AudioEncodeNode()
        assert node.audio_path is None

    def test_update_path(self):
        node = LTX2AudioEncodeNode()
        node.update_path("/some/audio.wav")
        assert node.audio_path == "/some/audio.wav"
        assert node.updated is True


class TestAudioEncodeSerialization:
    def test_get_state_includes_audio_path(self):
        node = LTX2AudioEncodeNode()
        node.audio_path = "/path/to/audio.wav"
        state = node.get_state()
        assert "audio_path" in state
        assert state["audio_path"] == "/path/to/audio.wav"

    def test_apply_state_restores_audio_path(self):
        node = LTX2AudioEncodeNode()
        state = {"audio_path": "/restored/path.mp3"}
        node.apply_state(state)
        assert node.audio_path == "/restored/path.mp3"

    def test_round_trip(self):
        node = LTX2AudioEncodeNode()
        node.audio_path = "/test/audio.flac"
        state = node.get_state()

        node2 = LTX2AudioEncodeNode()
        node2.apply_state(state)
        assert node2.audio_path == node.audio_path

    def test_none_path_serialization(self):
        node = LTX2AudioEncodeNode()
        state = node.get_state()
        assert state["audio_path"] is None
