"""Tests for LoRA support: data object, graph node, settings, panel, and graph wiring."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import attr
import pytest
import torch

from frameartisan.modules.generation.constants import get_default_granular_weights
from frameartisan.modules.generation.data_objects.lora_data_object import LoraDataObject
from frameartisan.modules.generation.generation_settings import GenerationSettings
from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph
from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import (
    LTX2LoraNode,
    _apply_strength_filtering,
    convert_non_diffusers_lora,
)
from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES


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


def _source_names(node, input_name):
    return {src_node.name for src_node, _ in node.connections.get(input_name, [])}


# ---------------------------------------------------------------------------
# LoraDataObject
# ---------------------------------------------------------------------------


class TestLoraDataObject:
    def test_default_values(self):
        ldo = LoraDataObject()
        assert ldo.id == 0
        assert ldo.name == ""
        assert ldo.filepath == ""
        assert ldo.weight == 1.0
        assert ldo.enabled is True
        assert ldo.hash == ""

    def test_to_dict(self):
        ldo = LoraDataObject(id=1, name="test_lora", hash="abc123")
        d = ldo.to_dict()
        assert d["id"] == 1
        assert d["name"] == "test_lora"
        assert d["hash"] == "abc123"
        assert d["weight"] == 1.0

    def test_from_dict(self):
        d = {"id": 2, "name": "my_lora", "filepath": "/loras/test.safetensors", "weight": 0.8, "hash": "def456"}
        ldo = LoraDataObject.from_dict(d)
        assert ldo.id == 2
        assert ldo.name == "my_lora"
        assert ldo.weight == 0.8

    def test_from_dict_ignores_extra_keys(self):
        d = {"id": 1, "name": "x", "extra_field": "ignored"}
        ldo = LoraDataObject.from_dict(d)
        assert ldo.name == "x"
        assert not hasattr(ldo, "extra_field")

    def test_new_fields_defaults(self):
        ldo = LoraDataObject()
        assert ldo.granular_transformer_weights_enabled is False
        assert ldo.granular_transformer_weights == {}
        assert ldo.is_slider is False
        assert ldo.video_strength == 1.0
        assert ldo.audio_strength == 1.0

    def test_new_fields_round_trip(self):
        granular = {f"transformer_blocks.{i}": 0.5 for i in range(3)}
        ldo = LoraDataObject(
            granular_transformer_weights_enabled=True,
            granular_transformer_weights=granular,
            is_slider=True,
            video_strength=0.8,
            audio_strength=0.0,
        )
        d = ldo.to_dict()
        restored = LoraDataObject.from_dict(d)
        assert restored.granular_transformer_weights_enabled is True
        assert restored.granular_transformer_weights == granular
        assert restored.is_slider is True
        assert restored.video_strength == 0.8
        assert restored.audio_strength == 0.0


# ---------------------------------------------------------------------------
# get_default_granular_weights
# ---------------------------------------------------------------------------


class TestGetDefaultGranularWeights:
    def test_returns_48_keys_model_type_1(self):
        weights = get_default_granular_weights(1)
        assert len(weights) == 48
        assert all(v == 1.0 for v in weights.values())
        assert "transformer_blocks.0" in weights
        assert "transformer_blocks.47" in weights

    def test_returns_48_keys_model_type_2(self):
        weights = get_default_granular_weights(2)
        assert len(weights) == 48


# ---------------------------------------------------------------------------
# Audio/video strength filtering
# ---------------------------------------------------------------------------


class TestStrengthFiltering:
    def test_audio_keys_removed_at_zero(self):
        state = {
            "transformer.blocks.0.attn.lora_A.weight": MagicMock(),
            "transformer.blocks.0.audio_attn.lora_A.weight": MagicMock(),
            "transformer.blocks.0.audio_ff.net.lora_A.weight": MagicMock(),
        }
        filtered = _apply_strength_filtering(state, video_strength=1.0, audio_strength=0.0)
        assert "transformer.blocks.0.attn.lora_A.weight" in filtered
        assert "transformer.blocks.0.audio_attn.lora_A.weight" not in filtered
        assert "transformer.blocks.0.audio_ff.net.lora_A.weight" not in filtered

    def test_video_keys_removed_at_zero(self):
        state = {
            "transformer.blocks.0.attn.lora_A.weight": MagicMock(),
            "transformer.blocks.0.audio_attn.lora_A.weight": MagicMock(),
        }
        filtered = _apply_strength_filtering(state, video_strength=0.0, audio_strength=1.0)
        assert "transformer.blocks.0.attn.lora_A.weight" not in filtered
        assert "transformer.blocks.0.audio_attn.lora_A.weight" in filtered

    def test_cross_attention_keys_are_audio(self):
        state = {
            "transformer.blocks.0.video_to_audio_attn.lora_A.weight": MagicMock(),
            "transformer.blocks.0.audio_to_video_attn.lora_A.weight": MagicMock(),
        }
        filtered = _apply_strength_filtering(state, video_strength=1.0, audio_strength=0.0)
        assert len(filtered) == 0

    def test_all_strength_one_passthrough(self):
        state = {
            "transformer.blocks.0.attn.lora_A.weight": MagicMock(),
            "transformer.blocks.0.audio_attn.lora_A.weight": MagicMock(),
        }
        filtered = _apply_strength_filtering(state, video_strength=1.0, audio_strength=1.0)
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# LTX2LoraNode
# ---------------------------------------------------------------------------


class TestLTX2LoraNode:
    def test_default_lora_configs_empty(self):
        node = LTX2LoraNode()
        assert node.lora_configs == []

    def test_lora_configs_init(self):
        configs = [{"hash": "a", "name": "test", "weight": 0.5, "enabled": True}]
        node = LTX2LoraNode(lora_configs=configs)
        assert node.lora_configs == configs

    def test_priority(self):
        assert LTX2LoraNode.PRIORITY == 1

    def test_outputs(self):
        assert "transformer" in LTX2LoraNode.OUTPUTS

    def test_required_inputs(self):
        assert "transformer" in LTX2LoraNode.REQUIRED_INPUTS

    def test_serialize_include(self):
        assert "lora_configs" in LTX2LoraNode.SERIALIZE_INCLUDE

    def test_serialization_round_trip(self):
        configs = [
            {"hash": "abc", "name": "lora1", "weight": 0.7, "enabled": True, "filepath": "/loras/a.safetensors"},
            {"hash": "def", "name": "lora2", "weight": 1.2, "enabled": False, "filepath": "/loras/b.safetensors"},
        ]
        node = LTX2LoraNode(lora_configs=configs)
        node.id = 99
        node.name = "lora"

        d = node.to_dict()
        assert d["state"]["lora_configs"] == configs

        restored = LTX2LoraNode.from_dict(d)
        assert restored.lora_configs == configs
        assert restored.id == 99
        assert restored.name == "lora"

    def test_update_loras_sets_updated(self):
        node = LTX2LoraNode()
        node.updated = False
        node.update_loras([{"hash": "x", "name": "y", "weight": 1.0}])
        assert node.updated is True
        assert len(node.lora_configs) == 1

    def test_update_loras_no_change_does_not_set_updated(self):
        configs = [{"hash": "x", "name": "y", "weight": 1.0}]
        node = LTX2LoraNode(lora_configs=configs)
        node.updated = False
        node.update_loras(configs)
        assert node.updated is False

    def test_empty_configs_passthrough(self):
        """Empty lora_configs should not error when node has no real transformer."""
        node = LTX2LoraNode(lora_configs=[])
        node.id = 0
        node.name = "lora"
        assert node.lora_configs == []

    def test_serialization_with_granular_and_strength(self):
        granular = {f"transformer_blocks.{i}": 0.5 for i in range(48)}
        configs = [
            {
                "hash": "abc",
                "name": "lora1",
                "weight": 0.7,
                "enabled": True,
                "filepath": "/loras/a.safetensors",
                "video_strength": 0.5,
                "audio_strength": 0.0,
                "granular_transformer_weights_enabled": True,
                "granular_transformer_weights": granular,
                "is_slider": True,
            },
        ]
        node = LTX2LoraNode(lora_configs=configs)
        node.id = 99
        node.name = "lora"

        d = node.to_dict()
        restored = LTX2LoraNode.from_dict(d)
        cfg = restored.lora_configs[0]
        assert cfg["video_strength"] == 0.5
        assert cfg["audio_strength"] == 0.0
        assert cfg["granular_transformer_weights_enabled"] is True
        assert len(cfg["granular_transformer_weights"]) == 48
        assert cfg["is_slider"] is True


# ---------------------------------------------------------------------------
# LoraAdvancedDialog instantiation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QApplication exists for widget tests."""
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    return app


class TestLoraAdvancedDialog:
    def test_basic_creation(self, qapp):
        from frameartisan.modules.generation.lora.lora_advanced_dialog import LoraAdvancedDialog

        config = {
            "hash": "abc123",
            "name": "test_lora",
            "weight": 0.8,
            "video_strength": 1.0,
            "audio_strength": 0.5,
        }
        dialog = LoraAdvancedDialog("lora_advanced_abc123", config)
        assert dialog.lora_hash == "abc123"
        assert dialog.weight == 0.8
        assert dialog.audio_strength == 0.5
        assert len(dialog.layer_sliders) == 48
        assert len(dialog.layer_linked_buttons) == 48

    def test_get_config_returns_all_fields(self, qapp):
        from frameartisan.modules.generation.lora.lora_advanced_dialog import LoraAdvancedDialog

        config = {"hash": "x", "name": "y", "weight": 1.0}
        dialog = LoraAdvancedDialog("key", config)
        cfg = dialog._get_config()
        assert "weight" in cfg
        assert "video_strength" in cfg
        assert "audio_strength" in cfg
        assert "granular_transformer_weights_enabled" in cfg
        assert "granular_transformer_weights" in cfg
        assert "is_slider" in cfg


# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------


class TestLoraNodeRegistry:
    def test_lora_node_registered(self):
        assert "LTX2LoraNode" in NODE_CLASSES
        assert NODE_CLASSES["LTX2LoraNode"] is LTX2LoraNode


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


class TestLoraGraphWiring:
    def test_lora_node_exists(self, graph):
        assert graph.get_node_by_name("lora") is not None

    def test_lora_node_is_correct_type(self, graph):
        assert isinstance(graph.get_node_by_name("lora"), LTX2LoraNode)

    def test_lora_node_receives_transformer_from_model(self, graph):
        lora = graph.get_node_by_name("lora")
        assert _source_names(lora, "transformer") == {"model"}

    def test_latents_receives_transformer_from_lora(self, graph):
        latents = graph.get_node_by_name("latents")
        assert _source_names(latents, "transformer") == {"lora"}

    def test_denoise_receives_transformer_from_lora(self, graph):
        denoise = graph.get_node_by_name("denoise")
        assert _source_names(denoise, "transformer") == {"lora"}

    def test_total_node_count(self, graph):
        # 13 source nodes + 6 advanced guidance nodes + 1 lora node + 5 processing nodes
        # + 2 image conditioning nodes (legacy) + 1 condition_encode + 1 audio_encode
        # + 9 second pass nodes (model, lora, model_type, steps, guidance, stage, upsample, latents, denoise)
        assert len(graph.nodes) == 38

    def test_lora_configs_from_gen_settings(self, directories):
        configs = [{"hash": "abc", "name": "test", "weight": 0.5}]
        settings = GenerationSettings(active_loras=configs)
        g = create_default_ltx2_graph(settings, directories)
        lora_node = g.get_node_by_name("lora")
        assert lora_node.lora_configs == configs


# ---------------------------------------------------------------------------
# Graph serialization with lora node
# ---------------------------------------------------------------------------


class TestLoraGraphSerialization:
    def test_lora_node_in_json(self, graph):
        parsed = json.loads(graph.to_json())
        classes = {n["class"] for n in parsed["nodes"]}
        assert "LTX2LoraNode" in classes

    def test_round_trip_lora_configs(self, graph):
        lora = graph.get_node_by_name("lora")
        lora.lora_configs = [{"hash": "h1", "name": "lr", "weight": 0.9}]
        json_str = graph.to_json()

        g2 = FrameArtisanNodeGraph()
        g2.from_json(json_str, NODE_CLASSES)
        lora2 = g2.get_node_by_name("lora")
        assert lora2.lora_configs == [{"hash": "h1", "name": "lr", "weight": 0.9}]

    def test_round_trip_preserves_lora_connections(self, graph):
        json_str = graph.to_json()
        g2 = FrameArtisanNodeGraph()
        g2.from_json(json_str, NODE_CLASSES)

        lora = g2.get_node_by_name("lora")
        assert _source_names(lora, "transformer") == {"model"}

        denoise = g2.get_node_by_name("denoise")
        assert _source_names(denoise, "transformer") == {"lora"}

        latents = g2.get_node_by_name("latents")
        assert _source_names(latents, "transformer") == {"lora"}

    def test_round_trip_node_names_include_lora(self, graph):
        json_str = graph.to_json()
        g2 = FrameArtisanNodeGraph()
        g2.from_json(json_str, NODE_CLASSES)
        names = {n.name for n in g2.nodes}
        assert "lora" in names


# ---------------------------------------------------------------------------
# GenerationSettings.active_loras
# ---------------------------------------------------------------------------


class TestGenerationSettingsActiveLoras:
    def test_default_empty(self):
        settings = GenerationSettings()
        assert settings.active_loras == []

    def test_active_loras_not_persisted(self, tmp_path):
        """active_loras should always start empty — not restored from QSettings."""
        from PyQt6.QtCore import QSettings

        settings_path = str(tmp_path / "test_settings.ini")
        qsettings = QSettings(settings_path, QSettings.Format.IniFormat)

        gs = GenerationSettings()
        gs.active_loras = [
            {"hash": "abc", "name": "lora1", "weight": 0.8, "filepath": "/loras/a.safetensors", "enabled": True},
        ]
        gs.save(qsettings)
        qsettings.sync()

        loaded = GenerationSettings.load(qsettings)
        assert loaded.active_loras == []


# ---------------------------------------------------------------------------
# LoRA node execution (mocked transformer)
# ---------------------------------------------------------------------------


def _make_lora_node_with_mock(configs):
    """Create a LoRA node wired to a fake model node that provides a mock transformer."""
    from frameartisan.modules.generation.graph.nodes.node import Node

    class FakeModelNode(Node):
        OUTPUTS = ["transformer"]

        def __init__(self, transformer):
            super().__init__()
            self.values = {"transformer": transformer}

    mock_transformer = MagicMock()
    model_node = FakeModelNode(mock_transformer)
    model_node.id = 0
    model_node.name = "model"

    lora_node = LTX2LoraNode(lora_configs=configs)
    lora_node.id = 1
    lora_node.name = "lora"
    lora_node.device = "cpu"
    lora_node.connect("transformer", model_node, "transformer")

    return lora_node, mock_transformer


class TestLoraNodeExecution:
    """Verify the node calls the correct transformer APIs with the right args."""

    @patch("frameartisan.modules.generation.graph.nodes.ltx2_lora_node._load_lora_state_dict", return_value={})
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_load_lora_adapter_called_with_state_dict_and_name(self, mock_mm, mock_load):
        """load_lora_adapter is called with the converted state dict and adapter_name."""
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None

        configs = [
            {"hash": "abc", "name": "my_lora", "weight": 1.0, "enabled": True, "filepath": "/loras/a.safetensors"}
        ]
        node, mock_transformer = _make_lora_node_with_mock(configs)

        node()

        mock_load.assert_called_once_with("/loras/a.safetensors", 1.0, 1.0)
        mock_transformer.load_lora_adapter.assert_called_once_with(
            {},
            adapter_name="abc",
        )
        mock_transformer.set_adapters.assert_called_once_with(["abc"], [1.0])

    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_empty_configs_passthrough_returns_transformer(self, mock_mm):
        """With no configs, node returns the input transformer unchanged."""
        node, mock_transformer = _make_lora_node_with_mock([])
        node()

        assert node.values["transformer"] is mock_transformer
        mock_transformer.load_lora_adapter.assert_not_called()
        mock_transformer.set_adapters.assert_not_called()

    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_disabled_lora_skipped(self, mock_mm):
        """Disabled LoRA configs are ignored."""
        configs = [
            {"hash": "abc", "name": "disabled", "weight": 1.0, "enabled": False, "filepath": "/loras/a.safetensors"}
        ]
        node, mock_transformer = _make_lora_node_with_mock(configs)
        node()

        assert node.values["transformer"] is mock_transformer
        mock_transformer.load_lora_adapter.assert_not_called()

    @patch("frameartisan.modules.generation.graph.nodes.ltx2_lora_node._load_lora_state_dict", return_value={})
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_multiple_loras_set_adapters_with_weights(self, mock_mm, mock_load):
        """Multiple enabled LoRAs are all loaded and set_adapters receives all names+weights."""
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None

        configs = [
            {"hash": "a", "name": "lora_a", "weight": 0.7, "enabled": True, "filepath": "/a.safetensors"},
            {"hash": "b", "name": "lora_b", "weight": 1.3, "enabled": True, "filepath": "/b.safetensors"},
        ]
        node, mock_transformer = _make_lora_node_with_mock(configs)
        node()

        assert mock_transformer.load_lora_adapter.call_count == 2
        names, weights = mock_transformer.set_adapters.call_args[0]
        assert set(names) == {"a", "b"}
        assert len(weights) == 2

    @patch("frameartisan.modules.generation.graph.nodes.ltx2_lora_node._load_lora_state_dict", return_value={})
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_adapter_not_reloaded_on_second_call(self, mock_mm, mock_load):
        """A previously loaded adapter is not re-loaded if the filepath hasn't changed."""
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None

        configs = [{"hash": "a", "name": "lora_a", "weight": 1.0, "enabled": True, "filepath": "/a.safetensors"}]
        node, mock_transformer = _make_lora_node_with_mock(configs)

        node.updated = True
        node()
        assert mock_transformer.load_lora_adapter.call_count == 1

        # Second call — adapter is already loaded, should not reload
        node.updated = True
        mock_transformer.load_lora_adapter.reset_mock()
        node()
        mock_transformer.load_lora_adapter.assert_not_called()
        # But set_adapters is still called to ensure weights are current
        mock_transformer.set_adapters.assert_called()

    @patch("frameartisan.modules.generation.graph.nodes.ltx2_lora_node._load_lora_state_dict", return_value={})
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_removed_adapter_is_deleted(self, mock_mm, mock_load):
        """When a LoRA is removed from configs, delete_adapters is called."""
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None

        configs = [{"hash": "a", "name": "lora_a", "weight": 1.0, "enabled": True, "filepath": "/a.safetensors"}]
        node, mock_transformer = _make_lora_node_with_mock(configs)

        node()
        assert "a" in node._loaded_adapters

        # Remove the LoRA — adapter should be deleted
        node.lora_configs = []
        node.updated = True
        node()
        mock_transformer.delete_adapters.assert_called_with("a")
        assert "a" not in node._loaded_adapters

    @patch("frameartisan.modules.generation.graph.nodes.ltx2_lora_node._load_lora_state_dict", return_value={})
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_output_transformer_is_same_object(self, mock_mm, mock_load):
        """The output transformer is the same object as the input (not a copy)."""
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None

        configs = [{"hash": "a", "name": "lora_a", "weight": 1.0, "enabled": True, "filepath": "/a.safetensors"}]
        node, mock_transformer = _make_lora_node_with_mock(configs)
        node()

        assert node.values["transformer"] is mock_transformer

    @patch("frameartisan.modules.generation.graph.nodes.ltx2_lora_node._load_lora_state_dict", return_value={})
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_strength_change_triggers_reload(self, mock_mm, mock_load):
        """Changing video/audio strength should reload the adapter (delete + re-load)."""
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None

        configs = [
            {
                "hash": "a",
                "name": "lora_a",
                "weight": 1.0,
                "enabled": True,
                "filepath": "/a.safetensors",
                "video_strength": 1.0,
                "audio_strength": 1.0,
            }
        ]
        node, mock_transformer = _make_lora_node_with_mock(configs)

        # First call — initial load
        node()
        assert mock_transformer.load_lora_adapter.call_count == 1
        mock_load.assert_called_with("/a.safetensors", 1.0, 1.0)

        # Change audio_strength — should trigger reload
        node.lora_configs = [
            {
                "hash": "a",
                "name": "lora_a",
                "weight": 1.0,
                "enabled": True,
                "filepath": "/a.safetensors",
                "video_strength": 1.0,
                "audio_strength": 0.0,
            }
        ]
        node.updated = True
        mock_transformer.load_lora_adapter.reset_mock()
        mock_load.reset_mock()
        node()

        # Should have deleted and reloaded with new strengths
        mock_transformer.delete_adapters.assert_called_with("a")
        mock_transformer.load_lora_adapter.assert_called_once()
        mock_load.assert_called_with("/a.safetensors", 1.0, 0.0)

    @patch("frameartisan.modules.generation.graph.nodes.ltx2_lora_node._load_lora_state_dict", return_value={})
    @patch("frameartisan.app.model_manager.get_model_manager")
    def test_same_strength_no_reload(self, mock_mm, mock_load):
        """Unchanged strengths should NOT trigger a reload."""
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None

        configs = [
            {
                "hash": "a",
                "name": "lora_a",
                "weight": 1.0,
                "enabled": True,
                "filepath": "/a.safetensors",
                "video_strength": 0.5,
                "audio_strength": 0.0,
            }
        ]
        node, mock_transformer = _make_lora_node_with_mock(configs)

        node()
        assert mock_transformer.load_lora_adapter.call_count == 1

        # Second call with same config — no reload
        node.updated = True
        mock_transformer.load_lora_adapter.reset_mock()
        node()
        mock_transformer.load_lora_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# GPU test: fake LoRA actually changes transformer output
# ---------------------------------------------------------------------------

TINY_MODEL_REPO = "OzzyGT/tiny_LTX2"


_fake_lora_counter = 0


def _create_fake_lora(transformer, tmp_path, *, scale=1.0, target_filter="to_q"):
    """Create a fake LoRA safetensors file with random weights for one attention layer."""
    global _fake_lora_counter
    import torch
    from safetensors.torch import save_file

    lora_state = {}
    rank = 4
    for name, param in transformer.named_parameters():
        if param.ndim < 2 or target_filter not in name:
            continue
        # load_lora_adapter with prefix="transformer" expects: transformer.<name>.lora_A/B.weight
        base_key = f"transformer.{name}".replace(".weight", "")
        out_dim, in_dim = param.shape[0], param.shape[1]
        lora_state[f"{base_key}.lora_A.weight"] = torch.randn(rank, in_dim, dtype=torch.bfloat16) * scale
        lora_state[f"{base_key}.lora_B.weight"] = torch.randn(out_dim, rank, dtype=torch.bfloat16) * scale
        break  # one layer is enough to prove the point

    assert lora_state, f"No parameters matched filter '{target_filter}'"
    _fake_lora_counter += 1
    filepath = str(tmp_path / f"fake_lora_{_fake_lora_counter}.safetensors")
    save_file(lora_state, filepath)
    return filepath


@pytest.fixture(scope="session")
def tiny_transformer():
    """Load the tiny transformer once per session."""
    import torch
    from diffusers import LTX2VideoTransformer3DModel
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(TINY_MODEL_REPO)
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    return transformer


@pytest.mark.gpu_light
class TestLoraActualEffect:
    """Verify a LoRA adapter actually changes the transformer output (uses tiny model)."""

    def test_fake_lora_changes_output(self, tiny_transformer, tmp_path):
        """Apply a fake LoRA via load_lora_adapter and verify a linear layer's output changes."""
        import copy

        import torch

        transformer = copy.deepcopy(tiny_transformer).to("cpu")

        # Pick the first attention layer to target
        target_name = None
        target_module = None
        for name, mod in transformer.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                target_module = mod
                break

        assert target_name is not None, "Could not find an attn.to_q linear layer"

        # Baseline: feed random input through this linear layer
        x = torch.randn(1, target_module.in_features, dtype=torch.bfloat16)
        with torch.no_grad():
            out_baseline = target_module(x).clone()

        # Create and apply fake LoRA targeting this specific layer
        filepath = _create_fake_lora(transformer, tmp_path, scale=5.0)
        transformer.load_lora_adapter(filepath, adapter_name="test_lora")
        transformer.set_adapters(["test_lora"], [1.0])

        # After LoRA, the module should be wrapped by PEFT
        # Re-resolve the module (PEFT replaces modules in-place)
        parts = target_name.split(".")
        mod = transformer
        for p in parts:
            mod = getattr(mod, p)

        with torch.no_grad():
            out_lora = mod(x).clone()

        # Clean up
        transformer.delete_adapters("test_lora")

        diff = (out_lora - out_baseline).abs().max().item()
        assert diff > 0, "LoRA had no effect on the linear layer output!"
        print(f"\n  LoRA effect confirmed on {target_name}: max_diff={diff:.6f}")

    def test_disable_adapters_restores_baseline(self, tiny_transformer, tmp_path):
        """disable_adapters() should make output match the no-LoRA baseline (uncheck scenario)."""
        import copy

        import torch

        transformer = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        target_module = None
        for name, mod in transformer.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                target_module = mod
                break

        assert target_name is not None

        x = torch.randn(1, target_module.in_features, dtype=torch.bfloat16)
        with torch.no_grad():
            out_baseline = target_module(x).clone()

        # Load and activate LoRA
        filepath = _create_fake_lora(transformer, tmp_path, scale=5.0)
        transformer.load_lora_adapter(filepath, adapter_name="test_lora")
        transformer.set_adapters(["test_lora"], [1.0])

        # Disable adapters (simulates unchecking the LoRA)
        transformer.disable_adapters()

        # Re-resolve module after PEFT wrapping
        mod = transformer
        for p in target_name.split("."):
            mod = getattr(mod, p)

        with torch.no_grad():
            out_disabled = mod(x).clone()

        # Clean up
        transformer.delete_adapters("test_lora")

        diff = (out_disabled - out_baseline).abs().max().item()
        assert diff == 0, f"disable_adapters did not restore baseline! max_diff={diff:.6f}"
        print(f"\n  disable_adapters restored baseline: max_diff={diff:.6f}")

    def test_delete_adapter_restores_baseline(self, tiny_transformer, tmp_path):
        """delete_adapters() should fully remove the LoRA and restore baseline (remove scenario)."""
        import copy

        import torch

        transformer = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        target_module = None
        for name, mod in transformer.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                target_module = mod
                break

        assert target_name is not None

        x = torch.randn(1, target_module.in_features, dtype=torch.bfloat16)
        with torch.no_grad():
            out_baseline = target_module(x).clone()

        # Load, activate, then delete
        filepath = _create_fake_lora(transformer, tmp_path, scale=5.0)
        transformer.load_lora_adapter(filepath, adapter_name="test_lora")
        transformer.set_adapters(["test_lora"], [1.0])
        transformer.delete_adapters("test_lora")

        # Re-resolve module after PEFT unwrapping
        mod = transformer
        for p in target_name.split("."):
            mod = getattr(mod, p)

        with torch.no_grad():
            out_after_delete = mod(x).clone()

        diff = (out_after_delete - out_baseline).abs().max().item()
        assert diff == 0, f"delete_adapters did not restore baseline! max_diff={diff:.6f}"
        print(f"\n  delete_adapters restored baseline: max_diff={diff:.6f}")


def _create_fake_lora_video_and_audio(transformer, tmp_path, *, scale=5.0):
    """Create a fake LoRA with keys targeting both video AND audio attention layers.

    Returns (filepath, video_target_name, audio_target_name) where target names are
    the module paths (e.g. 'transformer_blocks.0.attn1.to_q' for video,
    'transformer_blocks.0.audio_attn2.to_q' for audio).
    """
    global _fake_lora_counter
    import torch
    from safetensors.torch import save_file

    lora_state = {}
    rank = 4
    video_target = None
    audio_target = None

    for name, param in transformer.named_parameters():
        if param.ndim < 2 or "to_q" not in name:
            continue

        is_audio = any(pat in name for pat in ("audio_attn", "audio_to_video_attn", "video_to_audio_attn"))

        if is_audio and audio_target is None:
            audio_target = name.replace(".weight", "")
        elif not is_audio and "attn" in name and video_target is None:
            video_target = name.replace(".weight", "")

        if video_target and audio_target:
            break

    assert video_target is not None, "Could not find a video attn.to_q parameter"
    assert audio_target is not None, "Could not find an audio attn.to_q parameter"

    for target in (video_target, audio_target):
        # Resolve the module (e.g. transformer_blocks.0.attn1.to_q) and get weight shape
        mod = transformer
        for part in target.split("."):
            mod = getattr(mod, part)
        out_dim, in_dim = mod.weight.shape[0], mod.weight.shape[1]
        base_key = f"transformer.{target}"
        lora_state[f"{base_key}.lora_A.weight"] = torch.randn(rank, in_dim, dtype=torch.bfloat16) * scale
        lora_state[f"{base_key}.lora_B.weight"] = torch.randn(out_dim, rank, dtype=torch.bfloat16) * scale

    _fake_lora_counter += 1
    filepath = str(tmp_path / f"fake_lora_va_{_fake_lora_counter}.safetensors")
    save_file(lora_state, filepath)
    return filepath, video_target, audio_target


@pytest.mark.gpu_light
class TestLoraStrengthEffect:
    """Verify video/audio strength filtering actually changes which layers are affected."""

    def test_video_strength_zero_only_affects_audio(self, tiny_transformer, tmp_path):
        """With video_strength=0, audio_strength=1: video layers unchanged, audio layers changed."""
        import copy

        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import _load_lora_state_dict

        transformer = copy.deepcopy(tiny_transformer).to("cpu")
        filepath, video_target, audio_target = _create_fake_lora_video_and_audio(transformer, tmp_path)

        # Resolve modules and get baselines
        def resolve_module(name):
            mod = transformer
            for p in name.split("."):
                mod = getattr(mod, p)
            return mod

        video_mod = resolve_module(video_target)
        audio_mod = resolve_module(audio_target)

        x_video = torch.randn(1, video_mod.in_features, dtype=torch.bfloat16)
        x_audio = torch.randn(1, audio_mod.in_features, dtype=torch.bfloat16)

        with torch.no_grad():
            video_baseline = video_mod(x_video).clone()
            audio_baseline = audio_mod(x_audio).clone()

        # Load with video_strength=0, audio_strength=1
        state_dict = _load_lora_state_dict(filepath, video_strength=0.0, audio_strength=1.0)

        # Verify video keys were removed from state dict
        video_keys = [k for k in state_dict if video_target in k]
        audio_keys = [k for k in state_dict if audio_target in k]
        assert len(video_keys) == 0, f"Video keys should be removed: {video_keys}"
        assert len(audio_keys) > 0, "Audio keys should be present"

        # Apply the filtered LoRA
        transformer.load_lora_adapter(state_dict, adapter_name="test_strength")
        transformer.set_adapters(["test_strength"], [1.0])

        # Re-resolve after PEFT wrapping
        video_mod = resolve_module(video_target)
        audio_mod = resolve_module(audio_target)

        with torch.no_grad():
            video_after = video_mod(x_video).clone()
            audio_after = audio_mod(x_audio).clone()

        video_diff = (video_after - video_baseline).abs().max().item()
        audio_diff = (audio_after - audio_baseline).abs().max().item()

        assert video_diff == 0, f"Video layer should be unchanged (video_strength=0) but max_diff={video_diff:.6f}"
        assert audio_diff > 0, f"Audio layer should be changed (audio_strength=1) but max_diff={audio_diff:.6f}"
        print(f"\n  video_strength=0: video_diff={video_diff:.6f}, audio_diff={audio_diff:.6f}")

        transformer.delete_adapters("test_strength")

    def test_audio_strength_zero_only_affects_video(self, tiny_transformer, tmp_path):
        """With video_strength=1, audio_strength=0: video layers changed, audio layers unchanged."""
        import copy

        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import _load_lora_state_dict

        transformer = copy.deepcopy(tiny_transformer).to("cpu")
        filepath, video_target, audio_target = _create_fake_lora_video_and_audio(transformer, tmp_path)

        def resolve_module(name):
            mod = transformer
            for p in name.split("."):
                mod = getattr(mod, p)
            return mod

        video_mod = resolve_module(video_target)
        audio_mod = resolve_module(audio_target)

        x_video = torch.randn(1, video_mod.in_features, dtype=torch.bfloat16)
        x_audio = torch.randn(1, audio_mod.in_features, dtype=torch.bfloat16)

        with torch.no_grad():
            video_baseline = video_mod(x_video).clone()
            audio_baseline = audio_mod(x_audio).clone()

        # Load with video_strength=1, audio_strength=0
        state_dict = _load_lora_state_dict(filepath, video_strength=1.0, audio_strength=0.0)

        # Verify audio keys were removed from state dict
        video_keys = [k for k in state_dict if video_target in k]
        audio_keys = [k for k in state_dict if audio_target in k]
        assert len(video_keys) > 0, "Video keys should be present"
        assert len(audio_keys) == 0, f"Audio keys should be removed: {audio_keys}"

        # Apply the filtered LoRA
        transformer.load_lora_adapter(state_dict, adapter_name="test_strength")
        transformer.set_adapters(["test_strength"], [1.0])

        video_mod = resolve_module(video_target)
        audio_mod = resolve_module(audio_target)

        with torch.no_grad():
            video_after = video_mod(x_video).clone()
            audio_after = audio_mod(x_audio).clone()

        video_diff = (video_after - video_baseline).abs().max().item()
        audio_diff = (audio_after - audio_baseline).abs().max().item()

        assert video_diff > 0, f"Video layer should be changed (video_strength=1) but max_diff={video_diff:.6f}"
        assert audio_diff == 0, f"Audio layer should be unchanged (audio_strength=0) but max_diff={audio_diff:.6f}"
        print(f"\n  audio_strength=0: video_diff={video_diff:.6f}, audio_diff={audio_diff:.6f}")

        transformer.delete_adapters("test_strength")

    def test_partial_strength_scales_effect(self, tiny_transformer, tmp_path):
        """With video_strength=0.5, the LoRA effect should be smaller than full strength."""
        import copy

        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import _load_lora_state_dict

        transformer_full = copy.deepcopy(tiny_transformer).to("cpu")
        transformer_half = copy.deepcopy(tiny_transformer).to("cpu")

        filepath, video_target, _ = _create_fake_lora_video_and_audio(transformer_full, tmp_path)

        def resolve_module(model, name):
            mod = model
            for p in name.split("."):
                mod = getattr(mod, p)
            return mod

        video_mod = resolve_module(transformer_full, video_target)
        x = torch.randn(1, video_mod.in_features, dtype=torch.bfloat16)
        with torch.no_grad():
            baseline = video_mod(x).clone()

        # Full strength
        state_full = _load_lora_state_dict(filepath, video_strength=1.0, audio_strength=1.0)
        transformer_full.load_lora_adapter(state_full, adapter_name="full")
        transformer_full.set_adapters(["full"], [1.0])

        video_mod_full = resolve_module(transformer_full, video_target)
        with torch.no_grad():
            out_full = video_mod_full(x).clone()

        # Half strength
        state_half = _load_lora_state_dict(filepath, video_strength=0.5, audio_strength=1.0)
        transformer_half.load_lora_adapter(state_half, adapter_name="half_strength")
        transformer_half.set_adapters(["half_strength"], [1.0])

        video_mod_half = resolve_module(transformer_half, video_target)
        with torch.no_grad():
            out_half = video_mod_half(x).clone()

        diff_full = (out_full - baseline).abs().max().item()
        diff_half = (out_half - baseline).abs().max().item()

        assert diff_full > 0, "Full strength should change output"
        assert diff_half > 0, "Half strength should change output"
        assert diff_half < diff_full, (
            f"Half strength effect ({diff_half:.6f}) should be smaller than full ({diff_full:.6f})"
        )
        print(f"\n  Partial strength: full_diff={diff_full:.6f}, half_diff={diff_half:.6f}")

        transformer_full.delete_adapters("full")
        transformer_half.delete_adapters("half_strength")


# ---------------------------------------------------------------------------
# GPU tests: two LoRA nodes sharing one transformer (stage 1 + stage 2)
# ---------------------------------------------------------------------------


def _make_two_stage_lora_nodes(transformer, stage1_configs, stage2_configs):
    """Create two LTX2LoraNode instances wired to the same transformer.

    Returns (stage1_node, stage2_node).
    """
    from frameartisan.modules.generation.graph.nodes.node import Node

    class FakeModelNode(Node):
        OUTPUTS = ["transformer"]

        def __init__(self, t):
            super().__init__()
            self.values = {"transformer": t}

    model_node = FakeModelNode(transformer)
    model_node.id = 0
    model_node.name = "model"

    stage1 = LTX2LoraNode(lora_configs=stage1_configs)
    stage1.id = 1
    stage1.name = "lora"
    stage1.device = "cpu"
    stage1.connect("transformer", model_node, "transformer")

    stage2 = LTX2LoraNode(lora_configs=stage2_configs)
    stage2.id = 2
    stage2.name = "second_pass_lora"
    stage2.device = "cpu"
    stage2.connect("transformer", model_node, "transformer")

    return stage1, stage2


def _run_stage(node):
    """Execute a LoRA node with mocked ModelManager."""
    with patch("frameartisan.app.model_manager.get_model_manager") as mock_mm:
        mock_mm.return_value.use_components.return_value.__enter__ = lambda s: None
        mock_mm.return_value.use_components.return_value.__exit__ = lambda s, *a: None
        node.updated = True
        node()


def _get_output(transformer, target_name, x):
    """Forward x through a named submodule of transformer."""
    import torch

    mod = transformer
    for p in target_name.split("."):
        mod = getattr(mod, p)
    with torch.no_grad():
        return mod(x).clone()


@pytest.mark.gpu_light
class TestTwoStageSharedTransformerLora:
    """Two LoRA nodes sharing a single transformer — mirrors the stage 1 / stage 2 graph."""

    def test_add_second_lora_keeps_first_active(self, tiny_transformer, tmp_path):
        """Adding a second LoRA must not break the first one on the shared transformer."""
        import copy

        import torch

        transformer = copy.deepcopy(tiny_transformer).to("cpu")

        # Find a target layer for output comparison
        target_name = None
        for name, mod in transformer.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                break
        assert target_name is not None

        target_mod = transformer
        for p in target_name.split("."):
            target_mod = getattr(target_mod, p)
        x = torch.randn(1, target_mod.in_features, dtype=torch.bfloat16)
        baseline = _get_output(transformer, target_name, x)

        # Create two distinct fake LoRAs targeting different layers
        filepath_a = _create_fake_lora(transformer, tmp_path, scale=5.0, target_filter="to_q")
        filepath_b = _create_fake_lora(transformer, tmp_path, scale=5.0, target_filter="to_k")

        cfg_a = {"hash": "lora_a", "name": "LoRA_A", "weight": 1.0, "enabled": True, "filepath": filepath_a}
        cfg_b = {"hash": "lora_b", "name": "LoRA_B", "weight": 1.0, "enabled": True, "filepath": filepath_b}

        # --- Generation 1: only LoRA A on both stages ---
        stage1, stage2 = _make_two_stage_lora_nodes(transformer, [cfg_a], [cfg_a])
        _run_stage(stage1)
        _run_stage(stage2)

        out_after_a = _get_output(transformer, target_name, x)
        assert (out_after_a - baseline).abs().max().item() > 0, "LoRA A should change output"

        # --- Generation 2: add LoRA B to both stages ---
        stage1.update_loras([cfg_a, cfg_b])
        stage2.update_loras([cfg_a, cfg_b])
        _run_stage(stage1)
        out_after_ab_stage1 = _get_output(transformer, target_name, x)

        # LoRA A must still be active (output differs from baseline)
        assert (out_after_ab_stage1 - baseline).abs().max().item() > 0, "LoRA A stopped working after adding LoRA B"
        # Both LoRAs should produce a different result than A alone
        # (to_k LoRA may not affect to_q output, so we just check A is still present)

        _run_stage(stage2)
        out_after_ab_stage2 = _get_output(transformer, target_name, x)
        assert (out_after_ab_stage2 - baseline).abs().max().item() > 0, (
            "LoRA A stopped working on stage 2 after adding LoRA B"
        )

        # Clean up
        transformer.delete_adapters("lora_a")
        transformer.delete_adapters("lora_b")

    def test_disable_lora_for_second_stage_then_reenable(self, tiny_transformer, tmp_path):
        """Disable a LoRA for stage 2, then re-enable it — must not raise 'already in use'."""
        import copy

        import torch

        transformer = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        for name, mod in transformer.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                break
        assert target_name is not None

        target_mod = transformer
        for p in target_name.split("."):
            target_mod = getattr(target_mod, p)
        x = torch.randn(1, target_mod.in_features, dtype=torch.bfloat16)
        baseline = _get_output(transformer, target_name, x)

        filepath_a = _create_fake_lora(transformer, tmp_path, scale=5.0, target_filter="to_q")
        cfg_a = {"hash": "lora_a", "name": "LoRA_A", "weight": 1.0, "enabled": True, "filepath": filepath_a}

        stage1, stage2 = _make_two_stage_lora_nodes(transformer, [cfg_a], [cfg_a])

        # --- Generation 1: LoRA A on both stages ---
        _run_stage(stage1)
        _run_stage(stage2)
        out_both = _get_output(transformer, target_name, x)
        assert (out_both - baseline).abs().max().item() > 0

        # --- Generation 2: disable LoRA A for stage 2 ---
        stage2.update_loras([])  # filtered: no LoRAs for stage 2
        _run_stage(stage1)
        out_stage1_only = _get_output(transformer, target_name, x)
        assert (out_stage1_only - baseline).abs().max().item() > 0, "LoRA A should still work on stage 1"

        _run_stage(stage2)
        # Stage 2 removed the adapter — output should match baseline
        out_stage2_disabled = _get_output(transformer, target_name, x)
        assert (out_stage2_disabled - baseline).abs().max().item() == 0, (
            "Stage 2 should have no LoRA effect after disabling"
        )

        # --- Generation 3: re-enable LoRA A for stage 2 ---
        stage2.update_loras([cfg_a])
        _run_stage(stage1)  # stage 1 must reload A (stage 2 deleted it)
        out_reenabled_s1 = _get_output(transformer, target_name, x)
        assert (out_reenabled_s1 - baseline).abs().max().item() > 0, "LoRA A should work on stage 1 after re-enable"

        # This is the call that previously raised "adapter name already in use"
        _run_stage(stage2)
        out_reenabled_s2 = _get_output(transformer, target_name, x)
        assert (out_reenabled_s2 - baseline).abs().max().item() > 0, "LoRA A should work on stage 2 after re-enable"

        transformer.delete_adapters("lora_a")

    def test_stage2_delete_does_not_break_next_stage1(self, tiny_transformer, tmp_path):
        """Stage 2 filtering out a LoRA must not permanently break stage 1 across generations."""
        import copy

        import torch

        transformer = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        for name, mod in transformer.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                break
        assert target_name is not None

        target_mod = transformer
        for p in target_name.split("."):
            target_mod = getattr(target_mod, p)
        x = torch.randn(1, target_mod.in_features, dtype=torch.bfloat16)
        baseline = _get_output(transformer, target_name, x)

        filepath_a = _create_fake_lora(transformer, tmp_path, scale=5.0, target_filter="to_q")
        cfg_a = {"hash": "lora_a", "name": "LoRA_A", "weight": 1.0, "enabled": True, "filepath": filepath_a}

        # Stage 1 always has LoRA A; stage 2 never has it
        stage1, stage2 = _make_two_stage_lora_nodes(transformer, [cfg_a], [])

        # --- Generation 1 ---
        _run_stage(stage1)
        out_s1_gen1 = _get_output(transformer, target_name, x)
        assert (out_s1_gen1 - baseline).abs().max().item() > 0, "Stage 1 gen 1: LoRA A should be active"

        _run_stage(stage2)  # stage 2 has no LoRAs — does nothing (empty desired)

        # --- Generation 2: stage 1 must still work (adapter intact since stage 2 had nothing to delete) ---
        _run_stage(stage1)
        out_s1_gen2 = _get_output(transformer, target_name, x)
        assert (out_s1_gen2 - baseline).abs().max().item() > 0, "Stage 1 gen 2: LoRA A should be active"

        _run_stage(stage2)

        # --- Generation 3: verify consistency ---
        _run_stage(stage1)
        out_s1_gen3 = _get_output(transformer, target_name, x)
        assert (out_s1_gen3 - baseline).abs().max().item() > 0, "Stage 1 gen 3: LoRA A should be active"

        # All stage 1 outputs should be identical (deterministic, no reloading surprises)
        assert torch.equal(out_s1_gen1, out_s1_gen2), "Stage 1 output changed between gen 1 and 2"
        assert torch.equal(out_s1_gen2, out_s1_gen3), "Stage 1 output changed between gen 2 and 3"

        transformer.delete_adapters("lora_a")

    def test_two_loras_stage2_filters_one(self, tiny_transformer, tmp_path):
        """Two LoRAs on stage 1, only one on stage 2 — both must remain correct across generations."""
        import copy

        import torch

        transformer = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        for name, mod in transformer.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                break
        assert target_name is not None

        target_mod = transformer
        for p in target_name.split("."):
            target_mod = getattr(target_mod, p)
        x = torch.randn(1, target_mod.in_features, dtype=torch.bfloat16)
        baseline = _get_output(transformer, target_name, x)

        filepath_a = _create_fake_lora(transformer, tmp_path, scale=5.0, target_filter="to_q")
        filepath_b = _create_fake_lora(transformer, tmp_path, scale=5.0, target_filter="to_k")
        cfg_a = {"hash": "lora_a", "name": "LoRA_A", "weight": 1.0, "enabled": True, "filepath": filepath_a}
        cfg_b = {"hash": "lora_b", "name": "LoRA_B", "weight": 1.0, "enabled": True, "filepath": filepath_b}

        # Stage 1 has both; stage 2 only has B
        stage1, stage2 = _make_two_stage_lora_nodes(transformer, [cfg_a, cfg_b], [cfg_b])

        # --- Generation 1 ---
        _run_stage(stage1)
        out_s1_gen1 = _get_output(transformer, target_name, x)
        assert (out_s1_gen1 - baseline).abs().max().item() > 0, "Stage 1: LoRA A should affect to_q output"

        _run_stage(stage2)  # stage 2 deletes lora_a, keeps lora_b
        # lora_b targets to_k, so to_q output should revert to baseline after stage 2
        out_s2_gen1 = _get_output(transformer, target_name, x)
        assert torch.equal(out_s2_gen1, baseline), (
            "Stage 2 should not affect to_q output (only lora_b on to_k is active)"
        )

        # --- Generation 2: stage 1 must detect lora_a was deleted and reload ---
        _run_stage(stage1)
        out_s1_gen2 = _get_output(transformer, target_name, x)
        assert (out_s1_gen2 - baseline).abs().max().item() > 0, "Stage 1 gen 2: LoRA A should be reloaded"
        assert torch.equal(out_s1_gen1, out_s1_gen2), "Stage 1 output should be identical across generations"

        _run_stage(stage2)
        out_s2_gen2 = _get_output(transformer, target_name, x)
        assert torch.equal(out_s2_gen2, baseline), "Stage 2 gen 2: filtered LoRA A should still not affect to_q"

        # --- Generation 3: verify stable ---
        _run_stage(stage1)
        out_s1_gen3 = _get_output(transformer, target_name, x)
        assert torch.equal(out_s1_gen1, out_s1_gen3), "Stage 1 output drifted on gen 3"

        transformer.delete_adapters("lora_a")
        transformer.delete_adapters("lora_b")


# ---------------------------------------------------------------------------
# GPU tests: two LoRA nodes with separate transformers (stage 1 + stage 2)
# ---------------------------------------------------------------------------


def _make_two_stage_lora_nodes_separate(transformer1, transformer2, stage1_configs, stage2_configs):
    """Create two LTX2LoraNode instances each wired to its own transformer.

    Returns (stage1_node, stage2_node).
    """
    from frameartisan.modules.generation.graph.nodes.node import Node

    class FakeModelNode(Node):
        OUTPUTS = ["transformer"]

        def __init__(self, t):
            super().__init__()
            self.values = {"transformer": t}

    model_node1 = FakeModelNode(transformer1)
    model_node1.id = 0
    model_node1.name = "model"

    model_node2 = FakeModelNode(transformer2)
    model_node2.id = 10
    model_node2.name = "second_pass_model"

    stage1 = LTX2LoraNode(lora_configs=stage1_configs)
    stage1.id = 1
    stage1.name = "lora"
    stage1.device = "cpu"
    stage1.connect("transformer", model_node1, "transformer")

    stage2 = LTX2LoraNode(lora_configs=stage2_configs)
    stage2.id = 2
    stage2.name = "second_pass_lora"
    stage2.device = "cpu"
    stage2.connect("transformer", model_node2, "transformer")

    return stage1, stage2


@pytest.mark.gpu_light
class TestTwoStageSeparateTransformerLora:
    """Two LoRA nodes with independent transformers — mirrors using different models per stage."""

    def test_separate_models_independent_lora(self, tiny_transformer, tmp_path):
        """LoRA on stage 1 must not affect stage 2's transformer and vice-versa."""
        import copy

        import torch

        t1 = copy.deepcopy(tiny_transformer).to("cpu")
        t2 = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        for name, mod in t1.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                break
        assert target_name is not None

        target_mod = t1
        for p in target_name.split("."):
            target_mod = getattr(target_mod, p)
        x = torch.randn(1, target_mod.in_features, dtype=torch.bfloat16)
        baseline_t1 = _get_output(t1, target_name, x)
        baseline_t2 = _get_output(t2, target_name, x)

        filepath_a = _create_fake_lora(t1, tmp_path, scale=5.0, target_filter="to_q")
        cfg_a = {"hash": "lora_a", "name": "LoRA_A", "weight": 1.0, "enabled": True, "filepath": filepath_a}

        # Stage 1 has LoRA A; stage 2 has nothing
        stage1, stage2 = _make_two_stage_lora_nodes_separate(t1, t2, [cfg_a], [])

        _run_stage(stage1)
        _run_stage(stage2)

        out_t1 = _get_output(t1, target_name, x)
        out_t2 = _get_output(t2, target_name, x)

        assert (out_t1 - baseline_t1).abs().max().item() > 0, "Stage 1 LoRA should change t1 output"
        assert torch.equal(out_t2, baseline_t2), "Stage 2 transformer should be unaffected"

        t1.delete_adapters("lora_a")

    def test_separate_models_disable_reenable(self, tiny_transformer, tmp_path):
        """Disable/re-enable a LoRA for stage 2 on separate transformers — no cross-contamination."""
        import copy

        import torch

        t1 = copy.deepcopy(tiny_transformer).to("cpu")
        t2 = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        for name, mod in t1.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                break
        assert target_name is not None

        target_mod = t1
        for p in target_name.split("."):
            target_mod = getattr(target_mod, p)
        x = torch.randn(1, target_mod.in_features, dtype=torch.bfloat16)
        baseline_t1 = _get_output(t1, target_name, x)
        baseline_t2 = _get_output(t2, target_name, x)

        filepath_a = _create_fake_lora(t1, tmp_path, scale=5.0, target_filter="to_q")
        cfg_a = {"hash": "lora_a", "name": "LoRA_A", "weight": 1.0, "enabled": True, "filepath": filepath_a}

        # Both stages start with LoRA A
        stage1, stage2 = _make_two_stage_lora_nodes_separate(t1, t2, [cfg_a], [cfg_a])

        # --- Generation 1: both have LoRA A ---
        _run_stage(stage1)
        _run_stage(stage2)
        out_t1_gen1 = _get_output(t1, target_name, x)
        out_t2_gen1 = _get_output(t2, target_name, x)
        assert (out_t1_gen1 - baseline_t1).abs().max().item() > 0
        assert (out_t2_gen1 - baseline_t2).abs().max().item() > 0

        # --- Generation 2: disable LoRA A on stage 2 ---
        stage2.update_loras([])
        _run_stage(stage1)
        _run_stage(stage2)

        out_t1_gen2 = _get_output(t1, target_name, x)
        out_t2_gen2 = _get_output(t2, target_name, x)
        assert (out_t1_gen2 - baseline_t1).abs().max().item() > 0, "Stage 1 LoRA should still be active"
        assert torch.equal(out_t1_gen1, out_t1_gen2), "Stage 1 output should be stable"
        assert torch.equal(out_t2_gen2, baseline_t2), "Stage 2 should revert to baseline"

        # --- Generation 3: re-enable LoRA A on stage 2 ---
        stage2.update_loras([cfg_a])
        _run_stage(stage1)
        _run_stage(stage2)

        out_t1_gen3 = _get_output(t1, target_name, x)
        out_t2_gen3 = _get_output(t2, target_name, x)
        assert torch.equal(out_t1_gen1, out_t1_gen3), "Stage 1 output should be stable"
        assert (out_t2_gen3 - baseline_t2).abs().max().item() > 0, "Stage 2 LoRA should be active again"

        t1.delete_adapters("lora_a")
        t2.delete_adapters("lora_a")

    def test_separate_models_different_loras_per_stage(self, tiny_transformer, tmp_path):
        """Each stage gets a different LoRA — they must be fully independent."""
        import copy

        import torch

        t1 = copy.deepcopy(tiny_transformer).to("cpu")
        t2 = copy.deepcopy(tiny_transformer).to("cpu")

        target_name = None
        for name, mod in t1.named_modules():
            if isinstance(mod, torch.nn.Linear) and "attn" in name and "to_q" in name:
                target_name = name
                break
        assert target_name is not None

        target_mod = t1
        for p in target_name.split("."):
            target_mod = getattr(target_mod, p)
        x = torch.randn(1, target_mod.in_features, dtype=torch.bfloat16)
        baseline = _get_output(t1, target_name, x)

        filepath_a = _create_fake_lora(t1, tmp_path, scale=5.0, target_filter="to_q")
        filepath_b = _create_fake_lora(t1, tmp_path, scale=3.0, target_filter="to_q")
        cfg_a = {"hash": "lora_a", "name": "LoRA_A", "weight": 1.0, "enabled": True, "filepath": filepath_a}
        cfg_b = {"hash": "lora_b", "name": "LoRA_B", "weight": 1.0, "enabled": True, "filepath": filepath_b}

        # Stage 1 gets A, stage 2 gets B
        stage1, stage2 = _make_two_stage_lora_nodes_separate(t1, t2, [cfg_a], [cfg_b])

        _run_stage(stage1)
        _run_stage(stage2)

        out_t1 = _get_output(t1, target_name, x)
        out_t2 = _get_output(t2, target_name, x)

        # Both should differ from baseline
        assert (out_t1 - baseline).abs().max().item() > 0, "Stage 1 LoRA A should be active"
        assert (out_t2 - baseline).abs().max().item() > 0, "Stage 2 LoRA B should be active"
        # And differ from each other (different LoRA weights)
        assert not torch.equal(out_t1, out_t2), "Different LoRAs should produce different outputs"

        # --- Generation 2: swap — remove A from stage 1, add A to stage 2 ---
        stage1.update_loras([])
        stage2.update_loras([cfg_a])
        _run_stage(stage1)
        _run_stage(stage2)

        out_t1_gen2 = _get_output(t1, target_name, x)
        out_t2_gen2 = _get_output(t2, target_name, x)
        assert torch.equal(out_t1_gen2, baseline), "Stage 1 should revert to baseline after removing LoRA"
        assert (out_t2_gen2 - baseline).abs().max().item() > 0, "Stage 2 should have LoRA A active"

        t2.delete_adapters("lora_a")


# ---------------------------------------------------------------------------
# LoRA key conversion (convert_non_diffusers_lora)
# ---------------------------------------------------------------------------


class TestConvertNonDiffusersLora:
    """Regression tests for our local ComfyUI→diffusers LoRA key conversion."""

    @staticmethod
    def _make_state_dict(keys: list[str]) -> dict:
        return {k: torch.zeros(1) for k in keys}

    def test_basic_prefix_replacement(self):
        sd = self._make_state_dict(
            [
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd)
        assert set(converted.keys()) == {
            "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "transformer.transformer_blocks.0.attn1.to_q.lora_B.weight",
        }

    def test_patchify_proj_renamed(self):
        sd = self._make_state_dict(["diffusion_model.patchify_proj.lora_A.weight"])
        converted = convert_non_diffusers_lora(sd)
        assert "transformer.proj_in.lora_A.weight" in converted

    def test_audio_patchify_proj_renamed(self):
        sd = self._make_state_dict(["diffusion_model.audio_patchify_proj.lora_A.weight"])
        converted = convert_non_diffusers_lora(sd)
        assert "transformer.audio_proj_in.lora_A.weight" in converted

    def test_qk_norm_renamed(self):
        sd = self._make_state_dict(
            [
                "diffusion_model.transformer_blocks.5.attn1.q_norm.lora_A.weight",
                "diffusion_model.transformer_blocks.5.attn1.k_norm.lora_A.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd)
        assert "transformer.transformer_blocks.5.attn1.norm_q.lora_A.weight" in converted
        assert "transformer.transformer_blocks.5.attn1.norm_k.lora_A.weight" in converted

    def test_adaln_single_to_time_embed(self):
        sd = self._make_state_dict(
            [
                "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_A.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd)
        assert "transformer.time_embed.emb.timestep_embedder.linear_1.lora_A.weight" in converted

    def test_audio_adaln_single_to_audio_time_embed(self):
        sd = self._make_state_dict(
            [
                "diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_1.lora_A.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd)
        assert "transformer.audio_time_embed.emb.timestep_embedder.linear_1.lora_A.weight" in converted

    def test_av_cross_attn_renames(self):
        sd = self._make_state_dict(
            [
                "diffusion_model.av_ca_video_scale_shift_adaln_single.lora_A.weight",
                "diffusion_model.av_ca_a2v_gate_adaln_single.lora_A.weight",
                "diffusion_model.av_ca_audio_scale_shift_adaln_single.lora_A.weight",
                "diffusion_model.av_ca_v2a_gate_adaln_single.lora_A.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd)
        assert "transformer.av_cross_attn_video_scale_shift.lora_A.weight" in converted
        assert "transformer.av_cross_attn_video_a2v_gate.lora_A.weight" in converted
        assert "transformer.av_cross_attn_audio_scale_shift.lora_A.weight" in converted
        assert "transformer.av_cross_attn_audio_v2a_gate.lora_A.weight" in converted

    def test_scale_shift_table_renames(self):
        sd = self._make_state_dict(
            [
                "diffusion_model.scale_shift_table_a2v_ca_video.lora_A.weight",
                "diffusion_model.scale_shift_table_a2v_ca_audio.lora_A.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd)
        assert "transformer.video_a2v_cross_attn_scale_shift_table.lora_A.weight" in converted
        assert "transformer.audio_a2v_cross_attn_scale_shift_table.lora_A.weight" in converted

    def test_connector_prefix(self):
        sd = self._make_state_dict(
            [
                "text_embedding_projection.aggregate_embed.lora_A.weight",
                "text_embedding_projection.linear.lora_A.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd, "text_embedding_projection")
        assert "connectors.text_proj_in.lora_A.weight" in converted
        assert "connectors.linear.lora_A.weight" in converted

    def test_ignores_keys_without_prefix(self):
        sd = self._make_state_dict(
            [
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
                "some_other_prefix.foo.lora_A.weight",
            ]
        )
        converted = convert_non_diffusers_lora(sd)
        assert len(converted) == 1
        assert "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight" in converted

    def test_preserves_tensor_values(self):
        t = torch.randn(4, 8)
        sd = {"diffusion_model.transformer_blocks.0.ff.net.0.proj.lora_A.weight": t}
        converted = convert_non_diffusers_lora(sd)
        out_t = converted["transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight"]
        assert torch.equal(t, out_t)

    def test_all_48_blocks_converted(self):
        keys = [f"diffusion_model.transformer_blocks.{i}.attn1.to_q.lora_A.weight" for i in range(48)]
        sd = self._make_state_dict(keys)
        converted = convert_non_diffusers_lora(sd)
        for i in range(48):
            assert f"transformer.transformer_blocks.{i}.attn1.to_q.lora_A.weight" in converted

    def test_matches_diffusers_on_real_lora(self):
        """Regression: our conversion must produce identical keys to diffusers."""
        import os

        lora_path = "/home/ozzy/Desktop/failing_loras/lora.safetensors"
        if not os.path.exists(lora_path):
            pytest.skip("Real LoRA file not available")

        from safetensors.torch import load_file

        sd = load_file(lora_path)

        from diffusers.loaders.lora_pipeline import _convert_non_diffusers_ltx2_lora_to_diffusers

        diffusers_result = _convert_non_diffusers_ltx2_lora_to_diffusers(sd)
        our_result = convert_non_diffusers_lora(sd)

        assert set(our_result.keys()) == set(diffusers_result.keys()), (
            f"Key mismatch.\n"
            f"Only in ours: {set(our_result.keys()) - set(diffusers_result.keys())}\n"
            f"Only in diffusers: {set(diffusers_result.keys()) - set(our_result.keys())}"
        )
        for k in our_result:
            assert torch.equal(our_result[k], diffusers_result[k]), f"Tensor mismatch at {k}"

    def test_matches_diffusers_connectors_on_real_lora(self):
        """Regression: connector conversion must match diffusers too."""
        import os

        lora_path = "/home/ozzy/Desktop/failing_loras/lora.safetensors"
        if not os.path.exists(lora_path):
            pytest.skip("Real LoRA file not available")

        from safetensors.torch import load_file

        sd = load_file(lora_path)

        has_connector = any(k.startswith("text_embedding_projection.") for k in sd)
        if not has_connector:
            pytest.skip("LoRA has no connector keys")

        from diffusers.loaders.lora_pipeline import _convert_non_diffusers_ltx2_lora_to_diffusers

        diffusers_result = _convert_non_diffusers_ltx2_lora_to_diffusers(sd, "text_embedding_projection")
        our_result = convert_non_diffusers_lora(sd, "text_embedding_projection")

        assert set(our_result.keys()) == set(diffusers_result.keys())
        for k in our_result:
            assert torch.equal(our_result[k], diffusers_result[k])
