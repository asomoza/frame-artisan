"""Regression tests for LTX2ModelNode second-pass (component_prefix) behaviour.

Verifies that:
1. A prefixed model node only loads the transformer + scheduler (not text_encoder,
   vae, audio_vae, connectors, vocoder, tokenizer).
2. Two model nodes (first pass and second pass) can resolve different transformer
   variants independently via their own component_overrides.
3. The second-pass model node registers only the prefixed transformer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from frameartisan.app.model_manager import ModelManager
from frameartisan.modules.generation.graph.nodes.ltx2_model_node import LTX2ModelNode


@pytest.fixture()
def mm():
    """Provide a fresh ModelManager and patch get_model_manager to return it."""
    manager = ModelManager()
    with patch("frameartisan.app.model_manager.get_model_manager", return_value=manager):
        yield manager


def _make_fake_transformer():
    """Create a minimal nn.Module pretending to be a transformer."""
    mod = nn.Linear(4, 4)
    mod.config = MagicMock()
    mod.config.quantization_config = None
    # compile_repeated_blocks is called when use_torch_compile is set
    mod.compile_repeated_blocks = MagicMock()
    return mod


def _make_fake_text_encoder():
    mod = nn.Linear(4, 4)
    mod.config = MagicMock()
    mod.config.quantization_config = None
    return mod


def _make_fake_scheduler():
    sched = MagicMock()
    sched.config = {"num_train_timesteps": 1000}
    return sched


def _make_fake_light_component():
    mod = nn.Linear(4, 4)
    mod._use_streaming_decode = False
    return mod


class TestSecondPassOnlyLoadsTransformer:
    """When component_prefix is set, only transformer + scheduler should be loaded."""

    def test_prefixed_node_skips_non_transformer_components(self, mm):
        node = LTX2ModelNode(model_path="/fake/model", component_prefix="sp_")
        node.device = "cpu"
        node.dtype = torch.bfloat16
        node.updated = True
        node.enabled = True

        fake_transformer = _make_fake_transformer()
        fake_scheduler = _make_fake_scheduler()

        loaded_components = []

        def mock_load_component(cls, path, label, **kwargs):
            loaded_components.append(label)
            if label == "transformer":
                return fake_transformer
            if label == "scheduler":
                return fake_scheduler
            # Should NOT be reached for prefixed node
            raise AssertionError(f"Unexpected component load: {label}")

        with (
            patch.object(node, "_load_component", side_effect=mock_load_component),
            patch.object(node, "_resolve_component_paths", return_value={
                "transformer": "/fake/model/transformer",
                "scheduler": "/fake/model/scheduler",
            }),
            patch(
                "diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",
                return_value=_make_fake_scheduler(),
            ),
            patch("diffusers.hooks.group_offloading.apply_group_offloading"),
        ):
            result = node()

        # Only transformer should have been loaded via _load_component
        assert "transformer" in loaded_components
        assert "text_encoder" not in loaded_components
        assert "vae" not in loaded_components
        assert "audio_vae" not in loaded_components
        assert "connectors" not in loaded_components
        assert "vocoder" not in loaded_components
        assert "tokenizer" not in loaded_components

    def test_prefixed_node_outputs_only_transformer_and_scheduler(self, mm):
        node = LTX2ModelNode(model_path="/fake/model", component_prefix="sp_")
        node.device = "cpu"
        node.dtype = torch.bfloat16
        node.updated = True
        node.enabled = True

        fake_transformer = _make_fake_transformer()

        with (
            patch.object(node, "_load_component", return_value=fake_transformer),
            patch.object(node, "_resolve_component_paths", return_value={
                "transformer": "/fake/model/transformer",
                "scheduler": "/fake/model/scheduler",
            }),
            patch(
                "diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",
                return_value=_make_fake_scheduler(),
            ),
            patch("diffusers.hooks.group_offloading.apply_group_offloading"),
        ):
            result = node()

        assert "transformer" in result
        assert "scheduler_config" in result
        assert "transformer_component_name" in result
        assert result["transformer_component_name"] == "sp_transformer"

        # Should NOT contain shared components
        assert "tokenizer" not in result
        assert "text_encoder" not in result
        assert "vae" not in result
        assert "audio_vae" not in result
        assert "connectors" not in result
        assert "vocoder" not in result

    def test_prefixed_node_registers_only_prefixed_transformer(self, mm):
        node = LTX2ModelNode(model_path="/fake/model", component_prefix="sp_")
        node.device = "cpu"
        node.dtype = torch.bfloat16
        node.updated = True
        node.enabled = True

        fake_transformer = _make_fake_transformer()

        with (
            patch.object(node, "_load_component", return_value=fake_transformer),
            patch.object(node, "_resolve_component_paths", return_value={
                "transformer": "/fake/model/transformer",
                "scheduler": "/fake/model/scheduler",
            }),
            patch(
                "diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",
                return_value=_make_fake_scheduler(),
            ),
            patch("diffusers.hooks.group_offloading.apply_group_offloading"),
        ):
            node()

        assert mm.get_component("sp_transformer") is fake_transformer
        # Should not have overwritten primary components
        assert mm.get_component("transformer") is None
        assert mm.get_component("vae") is None
        assert mm.get_component("text_encoder") is None

    def test_non_prefixed_node_loads_all_components(self, mm):
        """Sanity check: a normal (non-prefixed) node loads everything."""
        node = LTX2ModelNode(model_path="/fake/model")
        node.device = "cpu"
        node.dtype = torch.bfloat16
        node.updated = True
        node.enabled = True

        loaded_components = []
        fake_transformer = _make_fake_transformer()
        fake_te = _make_fake_text_encoder()
        fake_light = _make_fake_light_component()

        def mock_load_component(cls, path, label, **kwargs):
            loaded_components.append(label)
            if label == "transformer":
                return fake_transformer
            if label == "text_encoder":
                return fake_te
            return fake_light

        with (
            patch.object(node, "_load_component", side_effect=mock_load_component),
            patch.object(node, "_resolve_component_paths", return_value={
                "text_encoder": "/fake/model/text_encoder",
                "transformer": "/fake/model/transformer",
                "tokenizer": "/fake/model/tokenizer",
                "scheduler": "/fake/model/scheduler",
                "vae": "/fake/model/vae",
                "audio_vae": "/fake/model/audio_vae",
                "connectors": "/fake/model/connectors",
                "vocoder": "/fake/model/vocoder",
            }),
            patch(
                "diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",
                return_value=_make_fake_scheduler(),
            ),
            patch("diffusers.hooks.group_offloading.apply_group_offloading"),
        ):
            result = node()

        assert "text_encoder" in loaded_components
        assert "transformer" in loaded_components
        assert "vae" in loaded_components
        assert "audio_vae" in loaded_components
        assert "connectors" in loaded_components
        assert "vocoder" in loaded_components
        assert "tokenizer" in loaded_components

        # All components present in output
        for key in ("tokenizer", "text_encoder", "transformer", "vae", "audio_vae",
                     "connectors", "vocoder", "scheduler_config"):
            assert key in result


class TestIndependentComponentOverrides:
    """First-pass and second-pass nodes can use different transformer variants."""

    def test_different_overrides_resolve_different_paths(self):
        """component_overrides should be used for path resolution, not global DB."""
        node_1st = LTX2ModelNode(
            model_path="/fake/model",
            model_id=1,
            component_overrides={"transformer": 100},  # 8-bit variant
        )
        node_2nd = LTX2ModelNode(
            model_path="/fake/model",
            model_id=1,
            component_prefix="sp_",
            component_overrides={"transformer": 200},  # 4-bit variant
        )

        # Mock the registry to verify which overrides are passed
        mock_registry = MagicMock()

        def fake_resolve_with_overrides(model_id, overrides):
            # Return different paths based on the override
            tr_id = overrides.get("transformer", 0)
            return {
                "transformer": MagicMock(storage_path=f"/components/transformer_{tr_id}"),
                "scheduler": MagicMock(storage_path="/fake/model/scheduler"),
                "text_encoder": MagicMock(storage_path="/fake/model/text_encoder"),
                "tokenizer": MagicMock(storage_path="/fake/model/tokenizer"),
                "vae": MagicMock(storage_path="/fake/model/vae"),
                "audio_vae": MagicMock(storage_path="/fake/model/audio_vae"),
                "connectors": MagicMock(storage_path="/fake/model/connectors"),
                "vocoder": MagicMock(storage_path="/fake/model/vocoder"),
            }

        mock_registry.resolve_model_components_with_overrides = MagicMock(
            side_effect=fake_resolve_with_overrides
        )
        # DB override should NOT be called when component_overrides is set
        mock_registry.resolve_model_components = MagicMock(
            side_effect=AssertionError("Should not read DB overrides when component_overrides is set")
        )

        with (
            patch("frameartisan.app.app.get_app_database_path", return_value="/fake/db.sqlite"),
            patch(
                "frameartisan.app.component_registry.ComponentRegistry",
                return_value=mock_registry,
            ),
        ):
            paths_1st = node_1st._resolve_component_paths("/fake/model")
            paths_2nd = node_2nd._resolve_component_paths("/fake/model")

        # Both should use resolve_model_components_with_overrides
        assert mock_registry.resolve_model_components_with_overrides.call_count == 2

        # They should resolve to different transformer paths
        assert paths_1st["transformer"] == "/components/transformer_100"
        assert paths_2nd["transformer"] == "/components/transformer_200"

    def test_no_overrides_falls_back_to_db(self):
        """When component_overrides is None, should read from DB."""
        node = LTX2ModelNode(model_path="/fake/model", model_id=1, component_overrides=None)

        mock_registry = MagicMock()
        mock_registry.resolve_model_components.return_value = {
            "transformer": MagicMock(storage_path="/db/transformer"),
        }

        with (
            patch("frameartisan.app.app.get_app_database_path", return_value="/fake/db.sqlite"),
            patch(
                "frameartisan.app.component_registry.ComponentRegistry",
                return_value=mock_registry,
            ),
        ):
            paths = node._resolve_component_paths("/fake/model")

        mock_registry.resolve_model_components.assert_called_once_with(1)
        mock_registry.resolve_model_components_with_overrides.assert_not_called()
        assert paths["transformer"] == "/db/transformer"

    def test_empty_overrides_falls_back_to_db(self):
        """Empty dict {} is falsy — should also fall back to DB."""
        node = LTX2ModelNode(model_path="/fake/model", model_id=1, component_overrides={})

        mock_registry = MagicMock()
        mock_registry.resolve_model_components.return_value = {
            "transformer": MagicMock(storage_path="/db/transformer"),
        }

        with (
            patch("frameartisan.app.app.get_app_database_path", return_value="/fake/db.sqlite"),
            patch(
                "frameartisan.app.component_registry.ComponentRegistry",
                return_value=mock_registry,
            ),
        ):
            paths = node._resolve_component_paths("/fake/model")

        mock_registry.resolve_model_components.assert_called_once_with(1)
        assert paths["transformer"] == "/db/transformer"
