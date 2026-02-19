"""Tests for LTX2 node classes."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from frameartisan.modules.generation.graph.nodes.ltx2_decode_node import LTX2DecodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode
from frameartisan.modules.generation.graph.nodes.ltx2_latent_upsample_node import LTX2LatentUpsampleNode
from frameartisan.modules.generation.graph.nodes.ltx2_latents_node import LTX2LatentsNode
from frameartisan.modules.generation.graph.nodes.ltx2_model_node import LTX2ModelNode
from frameartisan.modules.generation.graph.nodes.ltx2_prompt_encode_node import LTX2PromptEncodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import LTX2SecondPassLatentsNode
from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import LTX2VideoSendNode


# ---------------------------------------------------------------------------
# LTX2ModelNode
# ---------------------------------------------------------------------------


class TestLTX2ModelNodeMetadata:
    def test_priority(self):
        assert LTX2ModelNode.PRIORITY == 2

    def test_outputs(self):
        expected = {
            "tokenizer",
            "text_encoder",
            "transformer",
            "vae",
            "audio_vae",
            "connectors",
            "vocoder",
            "scheduler_config",
            "transformer_component_name",
        }
        assert set(LTX2ModelNode.OUTPUTS) == expected

    def test_required_inputs_empty(self):
        assert LTX2ModelNode.REQUIRED_INPUTS == []

    def test_optional_inputs(self):
        assert LTX2ModelNode.OPTIONAL_INPUTS == ["use_torch_compile", "torch_compile_max_autotune"]

    def test_serialize_include_contains_model_path(self):
        assert "model_path" in LTX2ModelNode.SERIALIZE_INCLUDE

    def test_default_model_path(self):
        node = LTX2ModelNode()
        assert node.model_path == ""

    def test_custom_model_path(self):
        node = LTX2ModelNode(model_path="/some/path")
        assert node.model_path == "/some/path"


class TestLTX2ModelNodeQuantizationDetection:
    @pytest.fixture()
    def model_dir(self):
        """Temp directory representing a component dir with config.json."""
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp, os.path.join(tmp, "config.json")

    def _write_config(self, config_path, data):
        with open(config_path, "w") as f:
            json.dump(data, f)

    def test_sdnq_detection(self, model_dir):
        tmpdir, cfg = model_dir
        self._write_config(cfg, {"quantization_config": {"quant_type": "sdnq-fp8"}})
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) == "sdnq"

    def test_sdnq_uppercase(self, model_dir):
        tmpdir, cfg = model_dir
        self._write_config(cfg, {"quantization_config": {"quant_type": "SDNQ"}})
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) == "sdnq"

    def test_bnb_load_in_4bit(self, model_dir):
        tmpdir, cfg = model_dir
        self._write_config(cfg, {"quantization_config": {"load_in_4bit": True}})
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) == "bnb"

    def test_bnb_load_in_8bit(self, model_dir):
        tmpdir, cfg = model_dir
        self._write_config(cfg, {"quantization_config": {"load_in_8bit": True}})
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) == "bnb"

    def test_bnb_quant_type_string(self, model_dir):
        tmpdir, cfg = model_dir
        self._write_config(cfg, {"quantization_config": {"quant_type": "bitsandbytes-nf4"}})
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) == "bnb"

    def test_full_precision_no_quantization_config(self, model_dir):
        tmpdir, cfg = model_dir
        self._write_config(cfg, {"model_type": "ltx_video_transformer3d"})
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) is None

    def test_full_precision_null_quantization_config(self, model_dir):
        tmpdir, cfg = model_dir
        self._write_config(cfg, {"quantization_config": None})
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) is None

    def test_missing_config_file(self, model_dir):
        tmpdir, _ = model_dir
        # Don't write config.json
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) is None

    def test_missing_model_dir(self):
        assert LTX2ModelNode._detect_quantization_from_path("/nonexistent/path") is None

    def test_invalid_json(self, model_dir):
        tmpdir, cfg = model_dir
        with open(cfg, "w") as f:
            f.write("not valid json {{{")
        assert LTX2ModelNode._detect_quantization_from_path(tmpdir) is None


class TestLTX2ModelNodeSerialization:
    def test_to_dict_contains_model_path(self):
        node = LTX2ModelNode(model_path="/models/ltx2")
        node.id = 0
        node.name = "model"
        d = node.to_dict()
        assert d["state"]["model_path"] == "/models/ltx2"

    def test_from_dict_restores_model_path(self):
        node = LTX2ModelNode(model_path="/original")
        node.id = 0
        node.name = "model"
        d = node.to_dict()

        restored = LTX2ModelNode.from_dict(d)
        assert restored.model_path == "/original"

    def test_offload_strategy_serialised(self):
        node = LTX2ModelNode(model_path="/m", offload_strategy="group_offload")
        node.id = 0
        node.name = "model"
        d = node.to_dict()
        assert d["state"]["offload_strategy"] == "group_offload"

    def test_offload_strategy_round_trip(self):
        node = LTX2ModelNode(model_path="/m", offload_strategy="model_offload")
        node.id = 0
        node.name = "model"
        restored = LTX2ModelNode.from_dict(node.to_dict())
        assert restored.offload_strategy == "model_offload"

    def test_offload_strategy_change_detected_by_dict_comparison(self):
        """Changing offload_strategy on the staged node produces a different dict,
        which update_from_json uses to detect the change."""
        node = LTX2ModelNode(model_path="/m", offload_strategy="no_offload")
        node.id = 0
        node.name = "model"
        d1 = node.to_dict()

        node.offload_strategy = "group_offload"
        d2 = node.to_dict()

        assert d1 != d2

    def test_group_offload_use_stream_serialised(self):
        node = LTX2ModelNode(model_path="/m", group_offload_use_stream=True)
        node.id = 0
        node.name = "model"
        d = node.to_dict()
        assert d["state"]["group_offload_use_stream"] is True

    def test_group_offload_low_cpu_mem_serialised(self):
        node = LTX2ModelNode(model_path="/m", group_offload_low_cpu_mem=True)
        node.id = 0
        node.name = "model"
        d = node.to_dict()
        assert d["state"]["group_offload_low_cpu_mem"] is True

    def test_group_offload_options_round_trip(self):
        node = LTX2ModelNode(
            model_path="/m",
            group_offload_use_stream=True,
            group_offload_low_cpu_mem=True,
        )
        node.id = 0
        node.name = "model"
        restored = LTX2ModelNode.from_dict(node.to_dict())
        assert restored.group_offload_use_stream is True
        assert restored.group_offload_low_cpu_mem is True

    def test_group_offload_options_default_false(self):
        node = LTX2ModelNode(model_path="/m")
        assert node.group_offload_use_stream is False
        assert node.group_offload_low_cpu_mem is False

    def test_component_overrides_change_detected_by_dict_comparison(self):
        """Changing component_overrides must produce a different to_dict()
        so that update_from_json detects the change and re-runs the node."""
        node = LTX2ModelNode(model_path="/m", model_id=1)
        node.id = 0
        node.name = "model"
        d1 = node.to_dict()

        node.component_overrides = {"transformer": 42}
        d2 = node.to_dict()

        assert d1 != d2

    def test_component_overrides_triggers_update_from_json(self):
        """Simulate the generation flow: UI graph has new overrides,
        thread graph has old overrides — update_from_json must detect
        the change and mark the node as updated."""
        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES

        # Build a minimal graph with just the model node.
        graph = FrameArtisanNodeGraph()
        model_node = LTX2ModelNode(model_path="/m", model_id=1)
        graph.add_node(model_node, name="model")

        # First "generation" — serialise, update, run sets updated=False.
        json1 = graph.to_json()
        thread_graph = FrameArtisanNodeGraph()
        thread_graph.from_json(json1, NODE_CLASSES)
        thread_node = thread_graph.get_node_by_name("model")
        thread_node.updated = False  # simulate post-run

        # User changes variant → update component_overrides on UI node.
        model_node.component_overrides = {"transformer": 99}
        json2 = graph.to_json()

        # Thread receives new JSON — must detect the override change.
        thread_graph.update_from_json(json2, NODE_CLASSES)
        assert thread_node.updated is True

    def test_component_overrides_triggers_update_with_db(self):
        """Same as above but with a real DB — proves get_state() doesn't
        read from DB (which would make both sides identical, hiding the change)."""
        from frameartisan.app.app import get_app_database_path, set_app_database_path
        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES
        from frameartisan.utils.database import Database

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            db = Database(db_path)
            db.execute("""
                CREATE TABLE IF NOT EXISTS model_component_override (
                    model_id       INTEGER NOT NULL,
                    component_type TEXT NOT NULL,
                    component_id   INTEGER NOT NULL,
                    UNIQUE(model_id, component_type)
                )
            """)

            old_db_path = get_app_database_path()
            set_app_database_path(db_path)
            try:
                # Build graph, first generation with no overrides.
                graph = FrameArtisanNodeGraph()
                model_node = LTX2ModelNode(model_path="/m", model_id=1)
                graph.add_node(model_node, name="model")

                json1 = graph.to_json()
                thread_graph = FrameArtisanNodeGraph()
                thread_graph.from_json(json1, NODE_CLASSES)
                thread_node = thread_graph.get_node_by_name("model")
                thread_node.updated = False

                # User changes variant: DB override written, node updated.
                db.execute(
                    "INSERT OR REPLACE INTO model_component_override VALUES (?, ?, ?)",
                    (1, "transformer", 42),
                )
                model_node.component_overrides = {"transformer": 42}
                json2 = graph.to_json()

                thread_graph.update_from_json(json2, NODE_CLASSES)
                assert thread_node.updated is True, (
                    "update_from_json must detect component_overrides change "
                    "even when DB has the new override on both sides"
                )
            finally:
                set_app_database_path(old_db_path)


# ---------------------------------------------------------------------------
# LTX2PromptEncodeNode
# ---------------------------------------------------------------------------


class TestLTX2PromptEncodeNodeMetadata:
    def test_priority(self):
        assert LTX2PromptEncodeNode.PRIORITY == 1

    def test_required_inputs(self):
        expected = {"tokenizer", "text_encoder", "connectors", "prompt"}
        assert set(LTX2PromptEncodeNode.REQUIRED_INPUTS) == expected

    def test_optional_inputs(self):
        assert "negative_prompt" in LTX2PromptEncodeNode.OPTIONAL_INPUTS

    def test_outputs(self):
        assert set(LTX2PromptEncodeNode.OUTPUTS) == {"prompt_embeds", "audio_prompt_embeds", "attention_mask"}

    def test_serialize_include_is_empty(self):
        assert LTX2PromptEncodeNode.SERIALIZE_INCLUDE == set()


# ---------------------------------------------------------------------------
# LTX2LatentsNode
# ---------------------------------------------------------------------------


class TestLTX2LatentsNodeMetadata:
    def test_priority(self):
        assert LTX2LatentsNode.PRIORITY == 0

    def test_required_inputs(self):
        expected = {"transformer", "vae", "audio_vae", "num_frames", "height", "width", "frame_rate", "seed"}
        assert set(LTX2LatentsNode.REQUIRED_INPUTS) == expected

    def test_outputs(self):
        expected = {
            "video_latents",
            "audio_latents",
            "video_coords",
            "audio_coords",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "audio_num_frames",
            "conditioning_mask",
            "clean_latents",
            "clean_audio_latents",
            "audio_conditioning_mask",
            "base_num_tokens",
        }
        assert set(LTX2LatentsNode.OUTPUTS) == expected

    def test_serialize_include_is_empty(self):
        assert LTX2LatentsNode.SERIALIZE_INCLUDE == set()


# ---------------------------------------------------------------------------
# LTX2DenoiseNode
# ---------------------------------------------------------------------------


class TestLTX2DenoiseNodeMetadata:
    def test_priority(self):
        assert LTX2DenoiseNode.PRIORITY == 0

    def test_required_inputs(self):
        expected = {
            "transformer",
            "scheduler_config",
            "prompt_embeds",
            "audio_prompt_embeds",
            "attention_mask",
            "video_latents",
            "audio_latents",
            "video_coords",
            "audio_coords",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "audio_num_frames",
            "num_inference_steps",
            "guidance_scale",
            "frame_rate",
        }
        assert set(LTX2DenoiseNode.REQUIRED_INPUTS) == expected

    def test_optional_inputs(self):
        assert "model_type" in LTX2DenoiseNode.OPTIONAL_INPUTS

    def test_outputs(self):
        assert set(LTX2DenoiseNode.OUTPUTS) == {"video_latents", "audio_latents"}

    def test_serialize_include_is_empty(self):
        assert LTX2DenoiseNode.SERIALIZE_INCLUDE == set()

    def test_callback_defaults_to_none(self):
        node = LTX2DenoiseNode()
        assert node.callback is None


class TestLTX2DenoiseNodeNumFramesConstraint:
    """The 8n+1 frame count constraint is enforced in the latents node.

    We verify the formula independently here.
    """

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (1, 1),
            (9, 9),
            (17, 17),
            (25, 25),
            (121, 121),
            # Non-conforming values snap down to nearest 8n+1
            (10, 9),
            (16, 9),
            (20, 17),
            (100, 97),
            (481, 481),
        ],
    )
    def test_snapping_formula(self, raw, expected):
        snapped = 8 * ((raw - 1) // 8) + 1
        assert snapped == expected


class TestLTX2DenoiseNodeSerialization:
    def test_to_dict_has_no_state(self):
        node = LTX2DenoiseNode()
        node.id = 1
        node.name = "denoise"
        d = node.to_dict()
        # SERIALIZE_INCLUDE = set() means state should be empty or absent
        assert d.get("state", {}) == {}


# ---------------------------------------------------------------------------
# LTX2DecodeNode
# ---------------------------------------------------------------------------


class TestLTX2DecodeNodeMetadata:
    def test_priority(self):
        assert LTX2DecodeNode.PRIORITY == -1

    def test_required_inputs(self):
        expected = {
            "vae",
            "audio_vae",
            "vocoder",
            "video_latents",
            "audio_latents",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "audio_num_frames",
            "num_frames",
            "frame_rate",
        }
        assert set(LTX2DecodeNode.REQUIRED_INPUTS) == expected

    def test_outputs(self):
        assert set(LTX2DecodeNode.OUTPUTS) == {"video", "audio", "frame_rate_out"}

    def test_serialize_include_is_empty(self):
        assert LTX2DecodeNode.SERIALIZE_INCLUDE == set()


# ---------------------------------------------------------------------------
# LTX2VideoSendNode
# ---------------------------------------------------------------------------


class TestLTX2VideoSendNodeMetadata:
    def test_priority(self):
        assert LTX2VideoSendNode.PRIORITY == -1

    def test_required_inputs(self):
        assert set(LTX2VideoSendNode.REQUIRED_INPUTS) == {"video", "audio", "frame_rate_out"}

    def test_outputs_empty(self):
        assert LTX2VideoSendNode.OUTPUTS == []

    def test_serialize_include_empty(self):
        assert LTX2VideoSendNode.SERIALIZE_INCLUDE == set()

    def test_defaults(self):
        node = LTX2VideoSendNode()
        assert node.video_callback is None
        assert node.output_dir == "."


class TestLTX2VideoSendNodeCallback:
    def test_video_callback_is_called(self, tmp_path):
        """video_callback receives the output path after encoding."""
        import numpy as np

        node = LTX2VideoSendNode()
        node.output_dir = str(tmp_path)

        received_paths = []
        node.video_callback = received_paths.append

        # Minimal 1-frame, 4x4 video
        dummy_video = np.zeros((1, 4, 4, 3), dtype="uint8")

        # Patch _encode_video so we don't need ffmpeg in CI
        import frameartisan.modules.generation.graph.nodes.ltx2_video_send_node as mod

        written = []

        def fake_encode(video, fps, output_path, **kwargs):
            written.append(output_path)
            # Write a real (empty) file so the callback path exists
            open(output_path, "wb").close()

        original = mod._encode_video_pyav
        mod._encode_video_pyav = fake_encode
        try:
            # Wire up dummy inputs via values dict directly
            node.values = {}
            # Bypass normal input resolution by setting attrs that __getattr__ would return
            node.__dict__["_video"] = dummy_video
            node.__dict__["_audio"] = None
            node.__dict__["_frame_rate_out"] = 24.0

            # Directly call the body with monkeypatching the input properties

            original_getattr = LTX2VideoSendNode.__getattr__

            def patched_getattr(self, name):
                mapping = {
                    "video": dummy_video,
                    "audio": None,
                    "frame_rate_out": 24.0,
                }
                if name in mapping:
                    return mapping[name]
                return original_getattr(self, name)

            LTX2VideoSendNode.__getattr__ = patched_getattr
            try:
                node()
            finally:
                LTX2VideoSendNode.__getattr__ = original_getattr
        finally:
            mod._encode_video_pyav = original

        assert len(received_paths) == 1
        assert received_paths[0].endswith(".mp4")
        assert "ltx2_" in received_paths[0]

    def test_output_path_filename_pattern(self, tmp_path):
        """Output filename follows ltx2_YYYYMMDD_HHMMSS.mp4 pattern."""
        import re

        import numpy as np
        import frameartisan.modules.generation.graph.nodes.ltx2_video_send_node as mod

        node = LTX2VideoSendNode()
        node.output_dir = str(tmp_path)
        node.video_callback = lambda p: None

        received = []
        original = mod._encode_video_pyav

        def fake_encode(video, fps, output_path, **kwargs):
            received.append(output_path)
            open(output_path, "wb").close()

        mod._encode_video_pyav = fake_encode
        try:
            original_getattr = LTX2VideoSendNode.__getattr__

            def patched_getattr(self, name):
                mapping = {"video": np.zeros((1, 4, 4, 3), dtype="uint8"), "audio": None, "frame_rate_out": 24.0}
                if name in mapping:
                    return mapping[name]
                return original_getattr(self, name)

            LTX2VideoSendNode.__getattr__ = patched_getattr
            try:
                node()
            finally:
                LTX2VideoSendNode.__getattr__ = original_getattr
        finally:
            mod._encode_video_pyav = original

        assert received
        filename = os.path.basename(received[0])
        assert re.match(r"ltx2_\d{8}_\d{6}\.mp4", filename), f"Unexpected filename: {filename}"


class TestLTX2VideoSendNodeGraphMetadata:
    """Tests that graph JSON metadata is embedded in saved MP4 files."""

    def test_metadata_written_to_mp4(self, tmp_path):
        """When json_graph_metadata is set, the MP4 comment field contains the graph JSON."""
        import json

        import av
        import numpy as np
        from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import _encode_video_pyav

        video = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        graph_json = json.dumps({"nodes": [{"class": "TextNode", "name": "prompt"}]})
        out = str(tmp_path / "test_meta.mp4")

        _encode_video_pyav(video, fps=24, output_path=out, metadata=graph_json)

        with av.open(out) as container:
            comment = container.metadata.get("comment")

        assert comment is not None, "MP4 should have a 'comment' metadata field"
        parsed = json.loads(comment)
        assert "nodes" in parsed
        assert parsed["nodes"][0]["class"] == "TextNode"

    def test_no_metadata_when_none(self, tmp_path):
        """When json_graph_metadata is None, no comment metadata is written."""
        import av
        import numpy as np
        from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import _encode_video_pyav

        video = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        out = str(tmp_path / "test_no_meta.mp4")

        _encode_video_pyav(video, fps=24, output_path=out, metadata=None)

        with av.open(out) as container:
            comment = container.metadata.get("comment")

        assert comment is None, "MP4 should not have a 'comment' field when metadata is None"

    def test_node_passes_metadata_to_encoder(self, tmp_path):
        """LTX2VideoSendNode passes json_graph_metadata through to _encode_video_pyav."""
        import json

        import av
        import numpy as np

        node = LTX2VideoSendNode()
        node.output_dir = str(tmp_path)
        node.video_callback = lambda p: None
        node.json_graph_metadata = json.dumps({"nodes": [], "connections": []})

        dummy_video = np.random.randint(0, 255, (2, 8, 8, 3), dtype=np.uint8)

        original_getattr = LTX2VideoSendNode.__getattr__

        def patched_getattr(self, name):
            mapping = {"video": dummy_video, "audio": None, "frame_rate_out": 24.0}
            if name in mapping:
                return mapping[name]
            return original_getattr(self, name)

        LTX2VideoSendNode.__getattr__ = patched_getattr
        try:
            node()
        finally:
            LTX2VideoSendNode.__getattr__ = original_getattr

        # Find the written file
        mp4_files = list(tmp_path.glob("*.mp4"))
        assert len(mp4_files) == 1

        with av.open(str(mp4_files[0])) as container:
            comment = container.metadata.get("comment")

        assert comment is not None
        parsed = json.loads(comment)
        assert parsed == {"nodes": [], "connections": []}

    def test_node_no_metadata_when_not_set(self, tmp_path):
        """LTX2VideoSendNode writes no comment when json_graph_metadata is None (default)."""
        import av
        import numpy as np

        node = LTX2VideoSendNode()
        node.output_dir = str(tmp_path)
        node.video_callback = lambda p: None
        # json_graph_metadata defaults to None — do NOT set it

        dummy_video = np.random.randint(0, 255, (2, 8, 8, 3), dtype=np.uint8)

        original_getattr = LTX2VideoSendNode.__getattr__

        def patched_getattr(self, name):
            mapping = {"video": dummy_video, "audio": None, "frame_rate_out": 24.0}
            if name in mapping:
                return mapping[name]
            return original_getattr(self, name)

        LTX2VideoSendNode.__getattr__ = patched_getattr
        try:
            node()
        finally:
            LTX2VideoSendNode.__getattr__ = original_getattr

        mp4_files = list(tmp_path.glob("*.mp4"))
        assert len(mp4_files) == 1

        with av.open(str(mp4_files[0])) as container:
            comment = container.metadata.get("comment")

        assert comment is None, "No comment metadata should be written when json_graph_metadata is None"

    def test_large_graph_metadata_roundtrip(self, tmp_path):
        """A realistically sized graph JSON survives the MP4 write/read cycle."""
        import json

        import av
        import numpy as np
        from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import _encode_video_pyav

        graph = {
            "nodes": [
                {"id": i, "class": "TextNode", "name": f"node_{i}", "state": {"text": f"value_{i}"}} for i in range(20)
            ],
            "connections": [
                {"from_node_id": i, "to_node_id": i + 1, "from_output_name": "out", "to_input_name": "in"}
                for i in range(19)
            ],
        }
        graph_json = json.dumps(graph)
        video = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        out = str(tmp_path / "test_large.mp4")

        _encode_video_pyav(video, fps=24, output_path=out, metadata=graph_json)

        with av.open(out) as container:
            comment = container.metadata.get("comment")

        assert comment is not None
        restored = json.loads(comment)
        assert len(restored["nodes"]) == 20
        assert len(restored["connections"]) == 19


class TestSaveVideoMetadataPreference:
    """Tests that the save_video_metadata preference gates metadata embedding."""

    def test_wire_callbacks_sets_metadata_when_enabled(self, tmp_path):
        """When save_video_metadata is True, wire_callbacks sets json_graph_metadata on video_send."""
        from unittest.mock import MagicMock

        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread

        dirs = MagicMock()
        dirs.outputs_videos = tmp_path
        node_graph = FrameArtisanNodeGraph()
        thread = NodeGraphThread(
            directories=dirs,
            node_graph=node_graph,
            dtype=None,
            device=None,
            graph_factory=FrameArtisanNodeGraph,
        )
        thread.save_video_metadata = True
        thread._job_json_graph = '{"nodes": [], "connections": []}'

        # Create a graph with a video_send node
        video_send = LTX2VideoSendNode()
        video_send.name = "video_send"
        run_graph = FrameArtisanNodeGraph()
        run_graph.nodes.append(video_send)

        thread.wire_callbacks(run_graph)

        assert video_send.json_graph_metadata == '{"nodes": [], "connections": []}'

    def test_wire_callbacks_no_metadata_when_disabled(self, tmp_path):
        """When save_video_metadata is False, wire_callbacks sets json_graph_metadata to None."""
        from unittest.mock import MagicMock

        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread

        dirs = MagicMock()
        dirs.outputs_videos = tmp_path
        node_graph = FrameArtisanNodeGraph()
        thread = NodeGraphThread(
            directories=dirs,
            node_graph=node_graph,
            dtype=None,
            device=None,
            graph_factory=FrameArtisanNodeGraph,
        )
        thread.save_video_metadata = False
        thread._job_json_graph = '{"nodes": [], "connections": []}'

        video_send = LTX2VideoSendNode()
        video_send.name = "video_send"
        run_graph = FrameArtisanNodeGraph()
        run_graph.nodes.append(video_send)

        thread.wire_callbacks(run_graph)

        assert video_send.json_graph_metadata is None

    def test_default_save_video_metadata_is_false(self, tmp_path):
        """NodeGraphThread defaults save_video_metadata to False."""
        from unittest.mock import MagicMock

        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread

        dirs = MagicMock()
        dirs.outputs_videos = tmp_path
        thread = NodeGraphThread(
            directories=dirs,
            node_graph=FrameArtisanNodeGraph(),
            dtype=None,
            device=None,
            graph_factory=FrameArtisanNodeGraph,
        )

        assert thread.save_video_metadata is False

    def test_end_to_end_metadata_disabled(self, tmp_path):
        """Full encode with save_video_metadata=False produces MP4 with no comment."""
        import av
        import numpy as np
        from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import _encode_video_pyav

        video = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        out = str(tmp_path / "no_meta.mp4")

        # metadata=None simulates save_video_metadata=False
        _encode_video_pyav(video, fps=24, output_path=out, metadata=None)

        with av.open(out) as container:
            assert container.metadata.get("comment") is None

    def test_end_to_end_metadata_enabled(self, tmp_path):
        """Full encode with save_video_metadata=True produces MP4 with graph in comment."""
        import json

        import av
        import numpy as np
        from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import _encode_video_pyav

        video = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        graph_json = json.dumps({"nodes": [{"class": "TestNode"}], "connections": []})
        out = str(tmp_path / "with_meta.mp4")

        _encode_video_pyav(video, fps=24, output_path=out, metadata=graph_json)

        with av.open(out) as container:
            comment = container.metadata.get("comment")

        assert comment is not None
        assert json.loads(comment)["nodes"][0]["class"] == "TestNode"

    def test_preference_default_is_false(self):
        """PreferencesObject defaults save_video_metadata to False."""
        from frameartisan.app.preferences import PreferencesObject

        prefs = PreferencesObject()
        assert prefs.save_video_metadata is False


# ---------------------------------------------------------------------------
# LTX2LatentUpsampleNode
# ---------------------------------------------------------------------------


class TestLTX2LatentUpsampleNodeMetadata:
    def test_priority(self):
        assert LTX2LatentUpsampleNode.PRIORITY == 0

    def test_required_inputs(self):
        expected = {"video_latents", "vae", "latent_num_frames", "latent_height", "latent_width"}
        assert set(LTX2LatentUpsampleNode.REQUIRED_INPUTS) == expected

    def test_outputs(self):
        expected = {"video_latents", "latent_num_frames", "latent_height", "latent_width"}
        assert set(LTX2LatentUpsampleNode.OUTPUTS) == expected

    def test_serialize_include(self):
        assert "upsampler_model_path" in LTX2LatentUpsampleNode.SERIALIZE_INCLUDE

    def test_default_upsampler_model_path(self):
        node = LTX2LatentUpsampleNode()
        assert node.upsampler_model_path == ""

    def test_custom_upsampler_model_path(self):
        node = LTX2LatentUpsampleNode(upsampler_model_path="/models/upsampler")
        assert node.upsampler_model_path == "/models/upsampler"


class TestLTX2LatentUpsampleNodeSerialization:
    def test_to_dict_contains_upsampler_model_path(self):
        node = LTX2LatentUpsampleNode(upsampler_model_path="/models/upsampler")
        node.id = 0
        node.name = "upsample"
        d = node.to_dict()
        assert d["state"]["upsampler_model_path"] == "/models/upsampler"

    def test_from_dict_restores_upsampler_model_path(self):
        node = LTX2LatentUpsampleNode(upsampler_model_path="/original")
        node.id = 0
        node.name = "upsample"
        d = node.to_dict()
        restored = LTX2LatentUpsampleNode.from_dict(d)
        assert restored.upsampler_model_path == "/original"

    def test_round_trip(self):
        node = LTX2LatentUpsampleNode(upsampler_model_path="/path/to/upsampler")
        node.id = 0
        node.name = "upsample"
        d = node.to_dict()
        restored = LTX2LatentUpsampleNode.from_dict(d)
        assert restored.upsampler_model_path == "/path/to/upsampler"
        assert restored.name == "upsample"
        assert restored.id == 0


# ---------------------------------------------------------------------------
# LTX2SecondPassLatentsNode
# ---------------------------------------------------------------------------


class TestLTX2SecondPassLatentsNodeMetadata:
    def test_priority(self):
        assert LTX2SecondPassLatentsNode.PRIORITY == 0

    def test_required_inputs(self):
        expected = {
            "transformer",
            "vae",
            "audio_vae",
            "video_latents",
            "audio_latents",
            "audio_coords",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "audio_num_frames",
            "frame_rate",
            "seed",
            "model_type",
        }
        assert set(LTX2SecondPassLatentsNode.REQUIRED_INPUTS) == expected

    def test_optional_inputs(self):
        assert "conditioning_mask" in LTX2SecondPassLatentsNode.OPTIONAL_INPUTS

    def test_outputs(self):
        expected = {
            "video_latents",
            "audio_latents",
            "video_coords",
            "audio_coords",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "audio_num_frames",
            "conditioning_mask",
            "clean_latents",
            "clean_audio_latents",
            "audio_conditioning_mask",
            "base_num_tokens",
        }
        assert set(LTX2SecondPassLatentsNode.OUTPUTS) == expected

    def test_serialize_include_is_empty(self):
        assert LTX2SecondPassLatentsNode.SERIALIZE_INCLUDE == set()


class TestLTX2SecondPassLatentsNodeSerialization:
    def test_to_dict_has_no_state(self):
        node = LTX2SecondPassLatentsNode()
        node.id = 1
        node.name = "second_pass_latents"
        d = node.to_dict()
        assert d.get("state", {}) == {}


# ---------------------------------------------------------------------------
# LTX2DenoiseNode stage 2
# ---------------------------------------------------------------------------


class TestLTX2DenoiseNodeStage2:
    def test_stage_in_optional_inputs(self):
        assert "stage" in LTX2DenoiseNode.OPTIONAL_INPUTS


# ---------------------------------------------------------------------------
# ltx2_utils
# ---------------------------------------------------------------------------


class TestLTX2Utils:
    def test_pack_unpack_latents_roundtrip(self):
        import torch
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents, unpack_latents

        latents = torch.randn(1, 4, 2, 4, 4)
        packed = pack_latents(latents, patch_size=1, patch_size_t=1)
        assert packed.ndim == 3
        unpacked = unpack_latents(packed, num_frames=2, height=4, width=4, patch_size=1, patch_size_t=1)
        assert torch.allclose(latents, unpacked)

    def test_pack_unpack_audio_roundtrip(self):
        import torch
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_audio_latents, unpack_audio_latents

        latents = torch.randn(1, 8, 10, 16)
        packed = pack_audio_latents(latents)
        assert packed.shape == (1, 10, 128)
        unpacked = unpack_audio_latents(packed, num_frames=10, num_mel_bins=16)
        assert torch.allclose(latents, unpacked)

    def test_normalize_denormalize_roundtrip(self):
        import torch
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import denormalize_latents, normalize_latents

        latents = torch.randn(1, 4, 2, 4, 4)
        mean = torch.zeros(4)
        std = torch.ones(4)
        normalized = normalize_latents(latents, mean, std, scaling_factor=1.0)
        recovered = denormalize_latents(normalized, mean, std, scaling_factor=1.0)
        assert torch.allclose(latents, recovered, atol=1e-6)

    def test_calculate_shift(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import calculate_shift

        mu = calculate_shift(1024, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
        assert isinstance(mu, float)
        # At base_seq_len=256, mu should be approximately base_shift=0.5
        mu_base = calculate_shift(256, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
        assert abs(mu_base - 0.5) < 1e-6

    def test_pack_text_embeds_shape(self):
        import torch
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_text_embeds

        hidden_states = torch.randn(1, 10, 32, 4)
        seq_lengths = torch.tensor([8])
        result = pack_text_embeds(hidden_states, seq_lengths)
        assert result.shape == (1, 10, 128)  # 32 * 4 = 128
