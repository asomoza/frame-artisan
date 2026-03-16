"""Tests for the default LTX2 graph factory and node graph serialization."""

from __future__ import annotations

import json
from unittest.mock import patch

import attr
import pytest

from frameartisan.modules.generation.generation_settings import GenerationSettings, compute_num_frames
from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph
from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.ltx2_decode_node import LTX2DecodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode
from frameartisan.modules.generation.graph.nodes.ltx2_latent_upsample_node import LTX2LatentUpsampleNode
from frameartisan.modules.generation.graph.nodes.ltx2_latents_node import LTX2LatentsNode
from frameartisan.modules.generation.graph.nodes.ltx2_model_node import LTX2ModelNode
from frameartisan.modules.generation.graph.nodes.ltx2_prompt_encode_node import LTX2PromptEncodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import LTX2SecondPassLatentsNode
from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import LTX2VideoSendNode
from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES
from frameartisan.modules.generation.graph.nodes.number_node import NumberNode
from frameartisan.modules.generation.graph.nodes.text_node import TextNode


@attr.s
class _FakeDirs:
    outputs_videos: str = attr.ib(default="/tmp/outputs_videos")


@pytest.fixture()
def gen_settings():
    return GenerationSettings()


@pytest.fixture()
def directories():
    return _FakeDirs()


@pytest.fixture()
def graph(gen_settings, directories):
    return create_default_ltx2_graph(gen_settings, directories)


# ---------------------------------------------------------------------------
# Node presence
# ---------------------------------------------------------------------------


class TestGraphNodePresence:
    def test_model_node_exists(self, graph):
        assert graph.get_node_by_name("model") is not None

    def test_prompt_node_exists(self, graph):
        assert graph.get_node_by_name("prompt") is not None

    def test_negative_prompt_node_exists(self, graph):
        assert graph.get_node_by_name("negative_prompt") is not None

    def test_seed_node_exists(self, graph):
        assert graph.get_node_by_name("seed") is not None

    def test_num_inference_steps_node_exists(self, graph):
        assert graph.get_node_by_name("num_inference_steps") is not None

    def test_guidance_scale_node_exists(self, graph):
        assert graph.get_node_by_name("guidance_scale") is not None

    def test_width_node_exists(self, graph):
        assert graph.get_node_by_name("width") is not None

    def test_height_node_exists(self, graph):
        assert graph.get_node_by_name("height") is not None

    def test_num_frames_node_exists(self, graph):
        assert graph.get_node_by_name("num_frames") is not None

    def test_frame_rate_node_exists(self, graph):
        assert graph.get_node_by_name("frame_rate") is not None

    def test_denoise_node_exists(self, graph):
        assert graph.get_node_by_name("denoise") is not None

    def test_video_send_node_exists(self, graph):
        assert graph.get_node_by_name("video_send") is not None

    def test_model_type_node_exists(self, graph):
        assert graph.get_node_by_name("model_type") is not None

    def test_prompt_encode_node_exists(self, graph):
        assert graph.get_node_by_name("prompt_encode") is not None

    def test_latents_node_exists(self, graph):
        assert graph.get_node_by_name("latents") is not None

    def test_decode_node_exists(self, graph):
        assert graph.get_node_by_name("decode") is not None

    def test_total_node_count(self, graph):
        # 13 source nodes + 6 advanced guidance nodes + 1 lora node + 5 processing nodes
        # + 2 image conditioning nodes (legacy) + 1 condition_encode + 1 audio_encode
        # + 9 second pass nodes (model, lora, model_type, steps, guidance, stage, upsample, latents, denoise)
        assert len(graph.nodes) == 38


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class TestGraphNodeTypes:
    def test_model_is_ltx2_model_node(self, graph):
        assert isinstance(graph.get_node_by_name("model"), LTX2ModelNode)

    def test_prompt_is_text_node(self, graph):
        assert isinstance(graph.get_node_by_name("prompt"), TextNode)

    def test_negative_prompt_is_text_node(self, graph):
        assert isinstance(graph.get_node_by_name("negative_prompt"), TextNode)

    def test_seed_is_number_node(self, graph):
        assert isinstance(graph.get_node_by_name("seed"), NumberNode)

    def test_denoise_is_ltx2_denoise_node(self, graph):
        assert isinstance(graph.get_node_by_name("denoise"), LTX2DenoiseNode)

    def test_video_send_is_ltx2_video_send_node(self, graph):
        assert isinstance(graph.get_node_by_name("video_send"), LTX2VideoSendNode)

    def test_prompt_encode_is_ltx2_prompt_encode_node(self, graph):
        assert isinstance(graph.get_node_by_name("prompt_encode"), LTX2PromptEncodeNode)

    def test_latents_is_ltx2_latents_node(self, graph):
        assert isinstance(graph.get_node_by_name("latents"), LTX2LatentsNode)

    def test_decode_is_ltx2_decode_node(self, graph):
        assert isinstance(graph.get_node_by_name("decode"), LTX2DecodeNode)


# ---------------------------------------------------------------------------
# Initial values
# ---------------------------------------------------------------------------


class TestGraphInitialValues:
    def test_model_path_from_gen_settings(self, graph, gen_settings):
        model_node = graph.get_node_by_name("model")
        assert model_node.model_path == gen_settings.model.filepath

    def test_num_frames_from_gen_settings(self, graph, gen_settings):
        node = graph.get_node_by_name("num_frames")
        expected = compute_num_frames(gen_settings.video_duration, gen_settings.frame_rate)
        assert node.number == expected

    def test_frame_rate_from_gen_settings(self, graph, gen_settings):
        node = graph.get_node_by_name("frame_rate")
        assert node.number == gen_settings.frame_rate

    def test_num_inference_steps_from_gen_settings(self, graph, gen_settings):
        node = graph.get_node_by_name("num_inference_steps")
        assert node.number == gen_settings.num_inference_steps

    def test_guidance_scale_from_gen_settings(self, graph, gen_settings):
        node = graph.get_node_by_name("guidance_scale")
        assert node.number == gen_settings.guidance_scale

    def test_video_send_output_dir(self, graph, directories):
        node = graph.get_node_by_name("video_send")
        assert node.output_dir == directories.outputs_videos

    def test_negative_prompt_has_default_text(self, graph):
        from frameartisan.modules.generation.constants import DEFAULT_NEGATIVE_PROMPT

        node = graph.get_node_by_name("negative_prompt")
        assert node.text == DEFAULT_NEGATIVE_PROMPT

    def test_group_offload_use_stream_from_gen_settings(self, directories):
        settings = GenerationSettings(group_offload_use_stream=True)
        g = create_default_ltx2_graph(settings, directories)
        model_node = g.get_node_by_name("model")
        assert model_node.group_offload_use_stream is True

    def test_group_offload_low_cpu_mem_from_gen_settings(self, directories):
        settings = GenerationSettings(group_offload_low_cpu_mem=True)
        g = create_default_ltx2_graph(settings, directories)
        model_node = g.get_node_by_name("model")
        assert model_node.group_offload_low_cpu_mem is True


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def _source_names(node, input_name):
    return {src_node.name for src_node, _ in node.connections.get(input_name, [])}


class TestGraphConnections:
    # prompt_encode connections
    def test_prompt_encode_tokenizer(self, graph):
        pe = graph.get_node_by_name("prompt_encode")
        assert _source_names(pe, "tokenizer") == {"model"}

    def test_prompt_encode_text_encoder(self, graph):
        pe = graph.get_node_by_name("prompt_encode")
        assert _source_names(pe, "text_encoder") == {"model"}

    def test_prompt_encode_connectors(self, graph):
        pe = graph.get_node_by_name("prompt_encode")
        assert _source_names(pe, "connectors") == {"model"}

    def test_prompt_encode_prompt(self, graph):
        pe = graph.get_node_by_name("prompt_encode")
        assert _source_names(pe, "prompt") == {"prompt"}

    def test_prompt_encode_negative_prompt(self, graph):
        pe = graph.get_node_by_name("prompt_encode")
        assert _source_names(pe, "negative_prompt") == {"negative_prompt"}

    # latents connections
    def test_latents_transformer(self, graph):
        lat = graph.get_node_by_name("latents")
        assert _source_names(lat, "transformer") == {"lora"}

    def test_latents_seed(self, graph):
        lat = graph.get_node_by_name("latents")
        assert _source_names(lat, "seed") == {"seed"}

    def test_latents_dimensions(self, graph):
        lat = graph.get_node_by_name("latents")
        assert _source_names(lat, "width") == {"width"}
        assert _source_names(lat, "height") == {"height"}
        assert _source_names(lat, "num_frames") == {"num_frames"}

    # denoise connections
    def test_denoise_transformer(self, graph):
        dn = graph.get_node_by_name("denoise")
        assert _source_names(dn, "transformer") == {"lora"}

    def test_denoise_scheduler_config(self, graph):
        dn = graph.get_node_by_name("denoise")
        assert _source_names(dn, "scheduler_config") == {"model"}

    def test_denoise_prompt_embeds(self, graph):
        dn = graph.get_node_by_name("denoise")
        assert _source_names(dn, "prompt_embeds") == {"prompt_encode"}

    def test_denoise_video_latents(self, graph):
        dn = graph.get_node_by_name("denoise")
        assert _source_names(dn, "video_latents") == {"latents"}

    def test_denoise_inference_params(self, graph):
        dn = graph.get_node_by_name("denoise")
        assert _source_names(dn, "num_inference_steps") == {"num_inference_steps"}
        assert _source_names(dn, "guidance_scale") == {"guidance_scale"}
        assert _source_names(dn, "frame_rate") == {"frame_rate"}
        assert _source_names(dn, "model_type") == {"model_type"}

    # decode connections
    def test_decode_vae(self, graph):
        dec = graph.get_node_by_name("decode")
        assert _source_names(dec, "vae") == {"model"}

    def test_decode_video_latents(self, graph):
        dec = graph.get_node_by_name("decode")
        assert _source_names(dec, "video_latents") == {"denoise"}

    def test_decode_audio_latents(self, graph):
        dec = graph.get_node_by_name("decode")
        assert _source_names(dec, "audio_latents") == {"denoise"}

    # video_send connections
    def test_video_connected_to_video_send(self, graph):
        vs = graph.get_node_by_name("video_send")
        assert _source_names(vs, "video") == {"decode"}

    def test_audio_connected_to_video_send(self, graph):
        vs = graph.get_node_by_name("video_send")
        assert _source_names(vs, "audio") == {"decode"}

    def test_frame_rate_out_connected_to_video_send(self, graph):
        vs = graph.get_node_by_name("video_send")
        assert _source_names(vs, "frame_rate_out") == {"decode"}


# ---------------------------------------------------------------------------
# JSON serialization round-trip
# ---------------------------------------------------------------------------


class TestGraphSerialization:
    def test_to_json_is_valid_json(self, graph):
        json_str = graph.to_json()
        parsed = json.loads(json_str)
        assert "nodes" in parsed
        assert "connections" in parsed

    def test_all_node_classes_in_json(self, graph):
        parsed = json.loads(graph.to_json())
        classes = {n["class"] for n in parsed["nodes"]}
        assert "LTX2ModelNode" in classes
        assert "LTX2LoraNode" in classes
        assert "LTX2DenoiseNode" in classes
        assert "LTX2DecodeNode" in classes
        assert "LTX2PromptEncodeNode" in classes
        assert "LTX2LatentsNode" in classes
        assert "LTX2VideoSendNode" in classes
        assert "TextNode" in classes
        assert "NumberNode" in classes

    def test_connection_count_in_json(self, graph):
        parsed = json.loads(graph.to_json())
        # First pass: 84 + audio_conditioning_mask(latents+denoise+decode=3) = 87 + keyframe(5+2) = 94 + lora_masks(1) = 95
        # Second pass: 53 + audio_conditioning_mask(sp_latents+sp_denoise=2) = 55 + lora_masks(1) = 56
        # Total: 95 + 56 = 151
        assert len(parsed["connections"]) == 151

    def test_round_trip_node_names(self, graph):
        json_str = graph.to_json()
        g2 = FrameArtisanNodeGraph()
        g2.from_json(json_str, NODE_CLASSES)
        names = {n.name for n in g2.nodes}
        expected = {
            "model",
            "lora",
            "prompt",
            "negative_prompt",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "width",
            "height",
            "num_frames",
            "frame_rate",
            "model_type",
            "use_torch_compile",
            "torch_compile_max_autotune",
            "prompt_encode",
            "latents",
            "denoise",
            "decode",
            "video_send",
            "source_image",
            "image_encode",
            "condition_encode",
            "audio_encode",
            "second_pass_model",
            "second_pass_lora",
            "second_pass_model_type",
            "second_pass_steps",
            "second_pass_guidance",
            "stage",
            "upsample",
            "second_pass_latents",
            "second_pass_denoise",
            "advanced_guidance",
            "stg_scale",
            "stg_blocks",
            "rescale_scale",
            "modality_scale",
            "guidance_skip_step",
        }
        assert names == expected

    def test_round_trip_model_path(self, graph):
        graph.get_node_by_name("model").model_path = "/test/model"
        json_str = graph.to_json()
        g2 = FrameArtisanNodeGraph()
        g2.from_json(json_str, NODE_CLASSES)
        assert g2.get_node_by_name("model").model_path == "/test/model"

    def test_round_trip_preserves_connections(self, graph):
        json_str = graph.to_json()
        g2 = FrameArtisanNodeGraph()
        g2.from_json(json_str, NODE_CLASSES)
        denoise = g2.get_node_by_name("denoise")
        assert _source_names(denoise, "transformer") == {"lora"}
        assert _source_names(denoise, "prompt_embeds") == {"prompt_encode"}
        assert _source_names(denoise, "video_latents") == {"latents"}


# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------


class TestNodeRegistry:
    def test_ltx2_model_node_registered(self):
        assert "LTX2ModelNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2ModelNode"] is LTX2ModelNode

    def test_ltx2_denoise_node_registered(self):
        assert "LTX2DenoiseNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2DenoiseNode"] is LTX2DenoiseNode

    def test_ltx2_video_send_node_registered(self):
        assert "LTX2VideoSendNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2VideoSendNode"] is LTX2VideoSendNode

    def test_ltx2_prompt_encode_node_registered(self):
        assert "LTX2PromptEncodeNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2PromptEncodeNode"] is LTX2PromptEncodeNode

    def test_ltx2_latents_node_registered(self):
        assert "LTX2LatentsNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2LatentsNode"] is LTX2LatentsNode

    def test_ltx2_decode_node_registered(self):
        assert "LTX2DecodeNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2DecodeNode"] is LTX2DecodeNode

    def test_existing_nodes_still_registered(self):
        assert "NumberNode" in NODE_CLASSES
        assert "TextNode" in NODE_CLASSES
        assert "BooleanNode" in NODE_CLASSES
        assert "ImageLoadNode" in NODE_CLASSES
        assert "VideoLoadNode" in NODE_CLASSES


# ---------------------------------------------------------------------------
# Graph executor error wrapping & timing
# ---------------------------------------------------------------------------


class _OkNode:
    """Minimal node-like object that succeeds."""

    PRIORITY = 0

    def __init__(self, name="ok"):
        self.id = 0
        self.name = name
        self.enabled = True
        self.updated = True
        self.abort = False
        self.dependencies = []
        self.dependents = []
        self.device = None
        self.dtype = None
        self.elapsed_time = None
        self.abort_callable = lambda: None

    def __call__(self):
        pass


class _FailNode(_OkNode):
    """Node that raises a bare RuntimeError (simulates OOM)."""

    def __init__(self, name="fail"):
        super().__init__(name)

    def __call__(self):
        raise RuntimeError("CUDA out of memory")


class TestGraphExecutorErrorWrapping:
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_unexpected_exception_wrapped_as_node_error(self, mock_mm):
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        g = FrameArtisanNodeGraph()
        g.nodes = [_FailNode("boom")]

        with pytest.raises(NodeError, match="CUDA out of memory") as exc_info:
            g()

        assert exc_info.value.node_name == "boom"
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_node_error_passes_through_unchanged(self, mock_mm):
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        class _NodeErrorNode(_OkNode):
            def __call__(self):
                raise NodeError("bad config", "my_node")

        g = FrameArtisanNodeGraph()
        g.nodes = [_NodeErrorNode("ne")]

        with pytest.raises(NodeError, match="bad config") as exc_info:
            g()

        assert exc_info.value.node_name == "my_node"


# ---------------------------------------------------------------------------
# Second pass graph creation
# ---------------------------------------------------------------------------


class TestSecondPassGraphCreation:
    def test_second_pass_nodes_exist(self, graph):
        assert graph.get_node_by_name("second_pass_model") is not None
        assert graph.get_node_by_name("second_pass_lora") is not None
        assert graph.get_node_by_name("second_pass_model_type") is not None
        assert graph.get_node_by_name("second_pass_steps") is not None
        assert graph.get_node_by_name("second_pass_guidance") is not None
        assert graph.get_node_by_name("stage") is not None
        assert graph.get_node_by_name("upsample") is not None
        assert graph.get_node_by_name("second_pass_latents") is not None
        assert graph.get_node_by_name("second_pass_denoise") is not None

    def test_second_pass_processing_nodes_disabled_by_default(self, graph):
        for name in (
            "second_pass_model",
            "second_pass_lora",
            "upsample",
            "second_pass_latents",
            "second_pass_denoise",
        ):
            node = graph.get_node_by_name(name)
            assert not node.enabled, f"{name} should be disabled by default"

    def test_second_pass_source_nodes_enabled(self, graph):
        """Source number nodes (steps, guidance, model_type, stage) are always enabled."""
        for name in ("second_pass_steps", "second_pass_guidance", "second_pass_model_type", "stage"):
            node = graph.get_node_by_name(name)
            assert node.enabled, f"{name} should be enabled"

    def test_stage_node_value_is_2(self, graph):
        stage = graph.get_node_by_name("stage")
        assert stage.number == 2

    def test_second_pass_model_node_type(self, graph):
        assert isinstance(graph.get_node_by_name("second_pass_model"), LTX2ModelNode)

    def test_upsample_node_type(self, graph):
        assert isinstance(graph.get_node_by_name("upsample"), LTX2LatentUpsampleNode)

    def test_second_pass_latents_node_type(self, graph):
        assert isinstance(graph.get_node_by_name("second_pass_latents"), LTX2SecondPassLatentsNode)

    def test_upsample_connections(self, graph):
        up = graph.get_node_by_name("upsample")
        assert _source_names(up, "video_latents") == {"denoise"}
        assert _source_names(up, "vae") == {"model"}
        assert _source_names(up, "latent_num_frames") == {"latents"}

    def test_second_pass_denoise_connections(self, graph):
        sp_dn = graph.get_node_by_name("second_pass_denoise")
        assert _source_names(sp_dn, "transformer") == {"second_pass_lora"}
        assert _source_names(sp_dn, "scheduler_config") == {"second_pass_model"}
        assert _source_names(sp_dn, "prompt_embeds") == {"prompt_encode"}
        assert _source_names(sp_dn, "video_latents") == {"second_pass_latents"}
        assert _source_names(sp_dn, "stage") == {"stage"}

    def test_second_pass_graph_serialization_roundtrip(self, graph):
        """JSON round-trip preserves second pass nodes and connections."""
        json_str = graph.to_json()
        g2 = FrameArtisanNodeGraph()
        g2.from_json(json_str, NODE_CLASSES)

        sp_denoise = g2.get_node_by_name("second_pass_denoise")
        assert sp_denoise is not None
        assert not sp_denoise.enabled
        assert _source_names(sp_denoise, "stage") == {"stage"}

        upsample = g2.get_node_by_name("upsample")
        assert upsample is not None
        assert not upsample.enabled


# ---------------------------------------------------------------------------
# Node registry for new nodes
# ---------------------------------------------------------------------------


class TestNewNodeRegistry:
    def test_ltx2_latent_upsample_node_registered(self):
        assert "LTX2LatentUpsampleNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2LatentUpsampleNode"] is LTX2LatentUpsampleNode

    def test_ltx2_second_pass_latents_node_registered(self):
        assert "LTX2SecondPassLatentsNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2SecondPassLatentsNode"] is LTX2SecondPassLatentsNode


class TestGraphTotalElapsedTime:
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_total_elapsed_time_set_after_execution(self, mock_mm):
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        g = FrameArtisanNodeGraph()
        g.nodes = [_OkNode("a"), _OkNode("b")]
        # Give them unique ids so topo sort works
        g.nodes[0].id = 0
        g.nodes[1].id = 1

        assert g.total_elapsed_time is None
        g()
        assert g.total_elapsed_time is not None
        assert g.total_elapsed_time >= 0

    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_total_elapsed_time_none_on_error(self, mock_mm):
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        g = FrameArtisanNodeGraph()
        g.nodes = [_FailNode("x")]

        with pytest.raises(NodeError):
            g()

        # total_elapsed_time should still be None since we errored before setting it
        assert g.total_elapsed_time is None


# ---------------------------------------------------------------------------
# Node values cache clearing
# ---------------------------------------------------------------------------


class _ProducerNode(_OkNode):
    """Node that stores a large value in self.values on each call."""

    def __init__(self, name="producer"):
        super().__init__(name)
        self.values = {}
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        self.values = {"output": [self.call_count] * 1000}


class _ConsumerNode(_OkNode):
    """Node that reads from a producer via connections."""

    def __init__(self, name="consumer"):
        super().__init__(name)
        self.values = {}
        self.received = None

    def __call__(self):
        # Simulate reading from producer via node.values
        if self.dependencies:
            dep = self.dependencies[0]
            self.received = dep.values.get("output")
        self.values = {"done": True}


class TestNodeValuesCacheClearing:
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_values_cleared_before_reexecution(self, mock_mm):
        """When a node is updated, its old values should be cleared before __call__."""
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        producer = _ProducerNode("p")
        producer.id = 0
        g = FrameArtisanNodeGraph()
        g.nodes = [producer]

        # First run — populates values
        g()
        assert producer.values == {"output": [1] * 1000}

        # Mark updated for second run
        producer.updated = True
        # Stuff old sentinel into values to verify it gets cleared
        producer.values["stale_key"] = "should_be_gone"

        g()
        assert "stale_key" not in producer.values
        assert producer.values == {"output": [2] * 1000}

    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_values_preserved_when_not_updated(self, mock_mm):
        """When a node is NOT updated, its cached values must be preserved."""
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        producer = _ProducerNode("p")
        producer.id = 0
        consumer = _ConsumerNode("c")
        consumer.id = 1
        consumer.dependencies = [producer]

        g = FrameArtisanNodeGraph()
        g.nodes = [producer, consumer]

        # First run — both execute
        g()
        assert producer.call_count == 1
        assert consumer.received == [1] * 1000

        # Second run — only consumer is updated; producer values must persist
        producer.updated = False
        consumer.updated = True

        g()
        assert producer.call_count == 1  # not re-executed
        assert producer.values == {"output": [1] * 1000}  # preserved
        assert consumer.received == [1] * 1000  # read from cached values

    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_disabled_node_values_cleared(self, mock_mm):
        """When a node is disabled and updated, its stale values must be cleared."""
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        producer = _ProducerNode("p")
        producer.id = 0
        g = FrameArtisanNodeGraph()
        g.nodes = [producer]

        # First run — populates values
        g()
        assert producer.values == {"output": [1] * 1000}

        # Disable and mark updated (simulates disabling a node between generations)
        producer.enabled = False
        producer.updated = True

        g()
        assert producer.values == {}  # stale values must be cleared
        assert producer.call_count == 1  # must NOT re-execute

    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_disabled_node_updated_flag_cleared(self, mock_mm):
        """Disabled node's updated flag should be cleared to prevent repeated propagation."""
        mock_mm.return_value.device_scope.return_value.__enter__ = lambda s: None
        mock_mm.return_value.device_scope.return_value.__exit__ = lambda s, *a: None

        producer = _ProducerNode("p")
        producer.id = 0
        g = FrameArtisanNodeGraph()
        g.nodes = [producer]

        producer.enabled = False
        producer.updated = True

        g()
        assert producer.updated is False
