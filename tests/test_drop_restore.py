"""Tests for drag-and-drop video metadata restore in GenerationModule.

Instead of instantiating the full GenerationModule (which needs GemmaTokenizer,
model managers, etc.), we build a lightweight stand-in that has only the
attributes _apply_loaded_graph_subset actually reads/writes, then call
the real method on it.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication, QProgressBar

from frameartisan.app.event_bus import EventBus
from frameartisan.modules.generation.generation_settings import GenerationSettings, compute_num_frames
from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_graph_json(
    *,
    prompt: str = "a dancing cat",
    negative_prompt: str = "blurry, bad quality",
    seed: int = 12345,
    width: int = 768,
    height: int = 480,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    frame_rate: int = 30,
    video_duration: int = 4,
    model_id: int = 0,
    model_path: str = "",
    lora_configs: list[dict] | None = None,
) -> dict:
    """Build a realistic graph JSON dict matching the LTX2 graph format."""
    gs = GenerationSettings()
    gs.video_width = width
    gs.video_height = height
    gs.num_inference_steps = num_inference_steps
    gs.guidance_scale = guidance_scale
    gs.frame_rate = frame_rate
    gs.video_duration = video_duration

    dirs = MagicMock()
    dirs.outputs_videos = "/tmp"

    graph = create_default_ltx2_graph(gs, dirs)

    prompt_node = graph.get_node_by_name("prompt")
    if prompt_node:
        prompt_node.update_value(prompt)

    neg_node = graph.get_node_by_name("negative_prompt")
    if neg_node:
        neg_node.update_value(negative_prompt)

    seed_node = graph.get_node_by_name("seed")
    if seed_node:
        seed_node.update_value(seed)

    model_node = graph.get_node_by_name("model")
    if model_node and (model_id or model_path):
        model_node.update_value(model_path, model_id)

    lora_node = graph.get_node_by_name("lora")
    if lora_node and lora_configs:
        lora_node.update_loras(lora_configs)

    return json.loads(graph.to_json())


class _FakeModule:
    """Lightweight stand-in for GenerationModule with only what _apply_loaded_graph_subset needs."""

    def __init__(self, qapp):
        from frameartisan.modules.generation.generation_module import GenerationModule

        # Real graph and settings
        self.gen_settings = GenerationSettings()
        dirs = MagicMock()
        dirs.outputs_videos = "/tmp"
        self.node_graph = create_default_ltx2_graph(self.gen_settings, dirs)

        # Real progress bar (light Qt widget)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.gen_settings.num_inference_steps)

        # Mock prompt bar — just needs the attrs _apply_loaded_graph_subset touches
        self.prompt_bar = MagicMock()
        self.prompt_bar.previous_positive_prompt = None
        self.prompt_bar.previous_negative_prompt = None
        self.prompt_bar.previous_seed = None
        self.prompt_bar.use_random_seed = True

        # Mock right_menu (for panel refresh)
        self.right_menu = MagicMock()

        # Real event bus + mock settings for save
        self.event_bus = EventBus()
        self.settings = MagicMock()

        # Visual conditions state
        self._visual_conditions: dict[str, dict] = {}
        self._source_image_path = None
        self.directories = dirs

        # Bind the real methods
        self._apply_loaded_graph_subset = GenerationModule._apply_loaded_graph_subset.__get__(self)
        self._resolve_image_path = GenerationModule._resolve_image_path.__get__(self)
        self.on_model_event = GenerationModule.on_model_event.__get__(self)


@pytest.fixture()
def module(qapp):
    return _FakeModule(qapp)


# ---------------------------------------------------------------------------
# Tests: _apply_loaded_graph_subset
# ---------------------------------------------------------------------------


class TestApplyLoadedGraphSubset:
    """Tests that _apply_loaded_graph_subset correctly restores all parameters."""

    def test_restores_prompts(self, module):
        graph_data = _build_graph_json(
            prompt="a beautiful sunset over the ocean",
            negative_prompt="ugly, dark, blurry",
        )

        module._apply_loaded_graph_subset(graph_data)

        module.prompt_bar.positive_prompt.setPlainText.assert_called_with("a beautiful sunset over the ocean")
        module.prompt_bar.negative_prompt.setPlainText.assert_called_with("ugly, dark, blurry")
        assert module.prompt_bar.previous_positive_prompt == "a beautiful sunset over the ocean"
        assert module.prompt_bar.previous_negative_prompt == "ugly, dark, blurry"

    def test_restores_seed(self, module):
        graph_data = _build_graph_json(seed=42)

        module._apply_loaded_graph_subset(graph_data)

        module.prompt_bar.seed_text.setText.assert_called_with("42")
        module.prompt_bar.seed_text.setDisabled.assert_called_with(False)
        module.prompt_bar.random_checkbox.setChecked.assert_called_with(False)
        assert module.prompt_bar.use_random_seed is False
        assert module.prompt_bar.previous_seed == 42

    def test_restores_dimensions(self, module):
        graph_data = _build_graph_json(width=768, height=480)

        module._apply_loaded_graph_subset(graph_data)

        assert module.gen_settings.video_width == 768
        assert module.gen_settings.video_height == 480
        assert module.node_graph.get_node_by_name("width").number == 768
        assert module.node_graph.get_node_by_name("height").number == 480

    def test_restores_steps(self, module):
        graph_data = _build_graph_json(num_inference_steps=30)

        module._apply_loaded_graph_subset(graph_data)

        assert module.gen_settings.num_inference_steps == 30
        assert module.progress_bar.maximum() == 30
        assert module.node_graph.get_node_by_name("num_inference_steps").number == 30

    def test_restores_guidance(self, module):
        graph_data = _build_graph_json(guidance_scale=5.0)

        module._apply_loaded_graph_subset(graph_data)

        assert module.gen_settings.guidance_scale == 5.0
        assert module.node_graph.get_node_by_name("guidance_scale").number == 5.0

    def test_restores_frame_rate(self, module):
        graph_data = _build_graph_json(frame_rate=30)

        module._apply_loaded_graph_subset(graph_data)

        assert module.gen_settings.frame_rate == 30
        assert module.node_graph.get_node_by_name("frame_rate").number == 30

    def test_restores_duration_from_num_frames(self, module):
        graph_data = _build_graph_json(frame_rate=30, video_duration=4)

        # Find num_frames from the graph
        num_frames = None
        for node in graph_data["nodes"]:
            if node.get("name") == "num_frames":
                num_frames = node["state"]["number"]
                break
        assert num_frames is not None

        module._apply_loaded_graph_subset(graph_data)

        expected_duration = max(1, math.ceil((num_frames - 1) / 30))
        assert module.gen_settings.video_duration == expected_duration
        assert module.node_graph.get_node_by_name("num_frames").number == num_frames

    def test_restores_seed_on_graph_node(self, module):
        graph_data = _build_graph_json(seed=99999)

        module._apply_loaded_graph_subset(graph_data)

        assert module.node_graph.get_node_by_name("seed").number == 99999

    def test_restores_prompt_on_graph_node(self, module):
        graph_data = _build_graph_json(
            prompt="hello world",
            negative_prompt="goodbye world",
        )

        module._apply_loaded_graph_subset(graph_data)

        assert module.node_graph.get_node_by_name("prompt").text == "hello world"
        assert module.node_graph.get_node_by_name("negative_prompt").text == "goodbye world"

    def test_graph_nodes_marked_updated(self, module):
        graph_data = _build_graph_json(width=1024, height=576)

        module._apply_loaded_graph_subset(graph_data)

        assert module.node_graph.get_node_by_name("width").updated is True
        assert module.node_graph.get_node_by_name("height").updated is True

    def test_restores_all_params_together(self, module):
        """End-to-end: all params are restored from a single graph JSON."""
        graph_data = _build_graph_json(
            prompt="a cat dancing in the rain",
            negative_prompt="low quality",
            seed=99999,
            width=512,
            height=320,
            num_inference_steps=20,
            guidance_scale=3.5,
            frame_rate=24,
            video_duration=5,
        )

        module._apply_loaded_graph_subset(graph_data)

        # Prompts
        module.prompt_bar.positive_prompt.setPlainText.assert_called_with("a cat dancing in the rain")
        module.prompt_bar.negative_prompt.setPlainText.assert_called_with("low quality")

        # Seed
        module.prompt_bar.seed_text.setText.assert_called_with("99999")
        assert module.prompt_bar.previous_seed == 99999
        assert module.prompt_bar.use_random_seed is False

        # Gen settings
        assert module.gen_settings.video_width == 512
        assert module.gen_settings.video_height == 320
        assert module.gen_settings.num_inference_steps == 20
        assert module.gen_settings.guidance_scale == 3.5
        assert module.gen_settings.frame_rate == 24

        # Graph nodes
        assert module.node_graph.get_node_by_name("width").number == 512
        assert module.node_graph.get_node_by_name("height").number == 320
        assert module.node_graph.get_node_by_name("num_inference_steps").number == 20
        assert module.node_graph.get_node_by_name("guidance_scale").number == 3.5
        assert module.node_graph.get_node_by_name("frame_rate").number == 24
        assert module.node_graph.get_node_by_name("seed").number == 99999

        expected_nf = compute_num_frames(5, 24)
        assert module.node_graph.get_node_by_name("num_frames").number == expected_nf

        # Progress bar
        assert module.progress_bar.maximum() == 20


class TestDropRestoreRoundTrip:
    """Tests the full write-metadata → read-metadata → restore cycle."""

    def test_encode_then_restore_matches(self, module, tmp_path):
        """Parameters survive MP4 encode → read → _apply_loaded_graph_subset."""
        import av
        import numpy as np
        from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import _encode_video_pyav

        graph_data = _build_graph_json(
            prompt="round trip test",
            negative_prompt="bad quality",
            seed=77777,
            width=640,
            height=352,
            num_inference_steps=24,
            guidance_scale=4.0,
            frame_rate=24,
            video_duration=3,
        )

        # Write video with metadata
        video = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        out = str(tmp_path / "roundtrip.mp4")
        _encode_video_pyav(video, fps=24, output_path=out, metadata=json.dumps(graph_data))

        # Read it back
        with av.open(out) as container:
            comment = container.metadata.get("comment")

        assert comment is not None
        restored_data = json.loads(comment)

        # Apply to module
        module._apply_loaded_graph_subset(restored_data)

        # Verify gen_settings
        assert module.gen_settings.video_width == 640
        assert module.gen_settings.video_height == 352
        assert module.gen_settings.num_inference_steps == 24
        assert module.gen_settings.guidance_scale == 4.0
        assert module.gen_settings.frame_rate == 24

        # Verify prompts
        module.prompt_bar.positive_prompt.setPlainText.assert_called_with("round trip test")
        module.prompt_bar.negative_prompt.setPlainText.assert_called_with("bad quality")

        # Verify seed
        module.prompt_bar.seed_text.setText.assert_called_with("77777")
        assert module.prompt_bar.previous_seed == 77777

        # Verify graph nodes
        assert module.node_graph.get_node_by_name("seed").number == 77777
        assert module.node_graph.get_node_by_name("width").number == 640
        assert module.node_graph.get_node_by_name("height").number == 352
        assert module.node_graph.get_node_by_name("num_inference_steps").number == 24
        assert module.node_graph.get_node_by_name("guidance_scale").number == 4.0
        assert module.node_graph.get_node_by_name("frame_rate").number == 24

        expected_nf = compute_num_frames(3, 24)
        assert module.node_graph.get_node_by_name("num_frames").number == expected_nf
        assert module.node_graph.get_node_by_name("prompt").text == "round trip test"
        assert module.node_graph.get_node_by_name("negative_prompt").text == "bad quality"
