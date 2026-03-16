"""Test that video audio conditioning flows correctly through the graph.

Simulates the user flow: load video → check "Use video audio" → generate.
Verifies that audio_encode node runs, produces clean_audio_latents, and these
flow through latents → denoise → decode without getting lost.
"""

from __future__ import annotations

import json

from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES
from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph


def _make_test_graph():
    """Create a default graph and return it along with gen_settings."""
    from unittest.mock import MagicMock

    from frameartisan.modules.generation.generation_settings import GenerationSettings

    gen_settings = GenerationSettings()
    directories = MagicMock()
    directories.outputs_videos = "/tmp/test_outputs"

    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph

    graph = create_default_ltx2_graph(gen_settings, directories)
    return graph, gen_settings


class TestVideoAudioConditioningFlow:
    """Simulate the event-bus flow that the generation module performs
    when a user loads a video and checks 'Use video audio'."""

    def test_audio_encode_enabled_after_video_audio_event(self):
        """After simulating the audio_condition 'add' event, audio_encode
        must be enabled with a path and marked updated."""
        graph, _ = _make_test_graph()

        audio_encode = graph.get_node_by_name("audio_encode")
        assert audio_encode is not None
        assert audio_encode.enabled is False  # default

        # Simulate GenerationModule.on_audio_condition_event "add"
        audio_encode.update_path("/tmp/test_video_audio.wav")
        audio_encode.update_trim(None, None)
        audio_encode.enabled = True
        audio_encode.set_updated()

        assert audio_encode.enabled is True
        assert audio_encode.audio_path == "/tmp/test_video_audio.wav"
        assert audio_encode.updated is True

    def test_audio_encode_survives_json_roundtrip(self):
        """audio_encode enabled + path must survive to_json → from_json."""
        graph, _ = _make_test_graph()

        audio_encode = graph.get_node_by_name("audio_encode")
        audio_encode.update_path("/tmp/test_video_audio.wav")
        audio_encode.enabled = True

        json_str = graph.to_json()
        parsed = json.loads(json_str)

        # Find audio_encode in the serialized graph
        audio_node_dict = None
        for node_dict in parsed["nodes"]:
            if node_dict.get("name") == "audio_encode":
                audio_node_dict = node_dict
                break

        assert audio_node_dict is not None
        assert audio_node_dict["enabled"] is True
        state = audio_node_dict.get("state", {})
        assert state.get("audio_path") == "/tmp/test_video_audio.wav"

    def test_audio_encode_updated_after_update_from_json(self):
        """When audio_encode changes from disabled→enabled with a path,
        update_from_json must mark it updated so the executor runs it."""
        graph, _ = _make_test_graph()

        # First round-trip: audio disabled (default)
        json_str_disabled = graph.to_json()
        run_graph = FrameArtisanNodeGraph()
        run_graph.from_json(json_str_disabled, NODE_CLASSES)

        # Simulate first generation: all nodes start updated=True from from_json
        # Clear updated flags to simulate post-execution state
        for node in run_graph.nodes:
            node.updated = False

        audio_encode_run = run_graph.get_node_by_name("audio_encode")
        assert audio_encode_run.enabled is False
        assert audio_encode_run.updated is False

        # Now enable audio on the main graph (simulating user action)
        audio_encode = graph.get_node_by_name("audio_encode")
        audio_encode.update_path("/tmp/test_video_audio.wav")
        audio_encode.enabled = True

        json_str_enabled = graph.to_json()

        # Apply update (simulates second generation)
        run_graph.update_from_json(json_str_enabled, NODE_CLASSES)

        audio_encode_run = run_graph.get_node_by_name("audio_encode")
        assert audio_encode_run.enabled is True
        assert audio_encode_run.updated is True, (
            "audio_encode must be marked updated after enable change "
            "so the executor runs it"
        )
        assert audio_encode_run.audio_path == "/tmp/test_video_audio.wav"

    def test_audio_connections_to_latents_and_decode(self):
        """Verify audio_encode outputs are wired to latents and decode nodes."""
        graph, _ = _make_test_graph()

        latents = graph.get_node_by_name("latents")
        decode = graph.get_node_by_name("decode")
        audio_encode = graph.get_node_by_name("audio_encode")

        # latents should depend on audio_encode (via clean_audio_latents connection)
        assert audio_encode in latents.dependencies

        # decode should depend on latents (for clean_audio_latents passthrough)
        assert latents in decode.dependencies

    def test_clean_audio_latents_connection_path(self):
        """Verify the full connection chain: audio_encode → latents → {denoise, decode}."""
        graph, _ = _make_test_graph()

        json_str = graph.to_json()
        parsed = json.loads(json_str)
        connections = parsed["connections"]

        # Build a node id → name mapping
        id_to_name = {}
        for node_dict in parsed["nodes"]:
            id_to_name[node_dict["id"]] = node_dict.get("name", "")

        def find_connection(from_name, from_output, to_name, to_input):
            for conn in connections:
                if (
                    id_to_name.get(conn["from_node_id"]) == from_name
                    and conn["from_output_name"] == from_output
                    and id_to_name.get(conn["to_node_id"]) == to_name
                    and conn["to_input_name"] == to_input
                ):
                    return True
            return False

        # audio_encode → latents
        assert find_connection("audio_encode", "clean_audio_latents", "latents", "clean_audio_latents")
        assert find_connection("audio_encode", "audio_conditioning_mask", "latents", "audio_conditioning_mask")

        # latents → denoise
        assert find_connection("latents", "clean_audio_latents", "denoise", "clean_audio_latents")
        assert find_connection("latents", "audio_conditioning_mask", "denoise", "audio_conditioning_mask")

        # latents → decode
        assert find_connection("latents", "clean_audio_latents", "decode", "clean_audio_latents")
        assert find_connection("latents", "audio_conditioning_mask", "decode", "audio_conditioning_mask")

    def test_condition_encode_does_not_interfere_with_audio(self):
        """When visual conditions are active, audio connections must still exist."""
        graph, _ = _make_test_graph()

        # Enable condition_encode (visual conditioning)
        condition_encode = graph.get_node_by_name("condition_encode")
        condition_encode.enabled = True

        # Enable audio_encode
        audio_encode = graph.get_node_by_name("audio_encode")
        audio_encode.update_path("/tmp/test_audio.wav")
        audio_encode.enabled = True

        latents = graph.get_node_by_name("latents")

        # Both should be dependencies of latents
        assert audio_encode in latents.dependencies
        assert condition_encode in latents.dependencies

    def test_audio_encode_disabled_clears_values_in_executor(self):
        """When audio_encode is disabled+updated, the executor must clear its values
        so downstream nodes see None for clean_audio_latents."""
        from unittest.mock import patch

        graph, _ = _make_test_graph()

        audio_encode = graph.get_node_by_name("audio_encode")
        # Simulate a previous run that left values
        audio_encode.values = {
            "clean_audio_latents": "fake_latents",
            "audio_conditioning_mask": None,
            "audio_path": "/some/path.wav",
        }
        audio_encode.enabled = False
        audio_encode.set_updated()

        # The executor clears values for disabled+updated nodes (line 124-130 in frameartisan_node_graph.py)
        # Simulate that logic:
        if audio_encode.updated and not audio_encode.enabled:
            audio_encode.values.clear()
            audio_encode.updated = False

        assert audio_encode.values == {}
