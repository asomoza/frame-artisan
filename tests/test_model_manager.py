"""Tests for ModelManager additions: component cache, device_scope, component_hash, offload."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from frameartisan.app.model_manager import (
    OFFLOAD_STRATEGIES,
    ModelManager,
    _SCOPED_DEVICE,
    _SCOPED_DTYPE,
)


@pytest.fixture()
def mm() -> ModelManager:
    """Fresh ModelManager for each test (bypasses the singleton)."""
    return ModelManager()


# ---------------------------------------------------------------------------
# component_hash
# ---------------------------------------------------------------------------


class TestComponentHash:
    def test_returns_16_char_hex_string(self, mm):
        h = mm.component_hash("some/path")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, mm):
        assert mm.component_hash("foo") == mm.component_hash("foo")

    def test_different_inputs_produce_different_hashes(self, mm):
        assert mm.component_hash("foo/te") != mm.component_hash("foo/tr")

    def test_static_method(self):
        """Can be called without an instance."""
        assert ModelManager.component_hash("x") == ModelManager.component_hash("x")


# ---------------------------------------------------------------------------
# get_cached / set_cached
# ---------------------------------------------------------------------------


class TestComponentCache:
    def test_cache_miss_returns_none(self, mm):
        assert mm.get_cached("nonexistent") is None

    def test_cache_hit_returns_stored_object(self, mm):
        obj = object()
        mm.set_cached("k", obj)
        assert mm.get_cached("k") is obj

    def test_overwrite_replaces_value(self, mm):
        mm.set_cached("k", "first")
        mm.set_cached("k", "second")
        assert mm.get_cached("k") == "second"

    def test_different_keys_are_independent(self, mm):
        mm.set_cached("a", 1)
        mm.set_cached("b", 2)
        assert mm.get_cached("a") == 1
        assert mm.get_cached("b") == 2

    def test_clear_empties_component_cache(self, mm):
        mm.set_cached("k", "val")
        mm.clear()
        assert mm.get_cached("k") is None
        assert mm._component_cache == {}

    def test_thread_safety(self, mm):
        errors = []

        def worker(i):
            try:
                key = f"key_{i}"
                mm.set_cached(key, i)
                assert mm.get_cached(key) == i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# device_scope
# ---------------------------------------------------------------------------


class TestDeviceScope:
    def test_sets_and_resets_device_context_var(self, mm):
        import torch

        device = torch.device("cpu")
        assert _SCOPED_DEVICE.get() is None

        with mm.device_scope(device=device):
            assert _SCOPED_DEVICE.get() is device

        assert _SCOPED_DEVICE.get() is None

    def test_sets_and_resets_dtype_context_var(self, mm):
        import torch

        with mm.device_scope(device="cpu", dtype=torch.float16):
            assert _SCOPED_DTYPE.get() is torch.float16

        assert _SCOPED_DTYPE.get() is None

    def test_dtype_defaults_to_none(self, mm):
        with mm.device_scope(device="cpu"):
            assert _SCOPED_DTYPE.get() is None

    def test_resets_on_exception(self, mm):
        with pytest.raises(ValueError):
            with mm.device_scope(device="cpu"):
                raise ValueError("boom")

        assert _SCOPED_DEVICE.get() is None
        assert _SCOPED_DTYPE.get() is None

    def test_nested_scopes(self, mm):
        import torch

        with mm.device_scope(device="cpu"):
            with mm.device_scope(device=torch.device("cuda") if False else "cpu"):
                assert _SCOPED_DEVICE.get() == "cpu"
            assert _SCOPED_DEVICE.get() == "cpu"
        assert _SCOPED_DEVICE.get() is None


# ---------------------------------------------------------------------------
# offload_strategy property / setter
# ---------------------------------------------------------------------------


class TestOffloadStrategy:
    def test_default_is_auto(self, mm):
        assert mm.offload_strategy == "auto"

    def test_set_valid_strategies(self, mm):
        for strategy in OFFLOAD_STRATEGIES:
            mm.offload_strategy = strategy
            assert mm.offload_strategy == strategy

    def test_set_invalid_strategy_raises(self, mm):
        with pytest.raises(ValueError, match="Unknown offload strategy"):
            mm.offload_strategy = "invalid_strategy"

    def test_applied_strategy_initially_none(self, mm):
        assert mm.applied_strategy is None


class TestGroupOffloadOptions:
    def test_use_stream_default_false(self, mm):
        assert mm.group_offload_use_stream is False

    def test_low_cpu_mem_default_false(self, mm):
        assert mm.group_offload_low_cpu_mem is False

    def test_set_use_stream(self, mm):
        mm.group_offload_use_stream = True
        assert mm.group_offload_use_stream is True

    def test_set_low_cpu_mem(self, mm):
        mm.group_offload_low_cpu_mem = True
        assert mm.group_offload_low_cpu_mem is True

    def test_clear_resets_stream_and_low_cpu_mem(self, mm):
        mm.group_offload_use_stream = True
        mm.group_offload_low_cpu_mem = True
        mm.clear()
        assert mm.group_offload_use_stream is False
        assert mm.group_offload_low_cpu_mem is False


# ---------------------------------------------------------------------------
# resolve_offload_strategy
# ---------------------------------------------------------------------------


class TestResolveOffloadStrategy:
    def test_explicit_strategy_returned_directly(self, mm):
        for strategy in ("no_offload", "model_offload", "sequential_group_offload", "group_offload"):
            mm.offload_strategy = strategy
            assert mm.resolve_offload_strategy("cpu") == strategy

    def test_auto_cpu_device_returns_group_offload(self, mm):
        mm.offload_strategy = "auto"
        assert mm.resolve_offload_strategy("cpu") == "group_offload"

    @patch("torch.cuda.get_device_properties")
    def test_auto_high_vram_returns_no_offload(self, mock_props, mm):
        mock_props.return_value = MagicMock(total_mem=24 * 1024**3)
        mm.offload_strategy = "auto"
        assert mm.resolve_offload_strategy(torch.device("cuda:0")) == "no_offload"

    @patch("torch.cuda.get_device_properties")
    def test_auto_mid_vram_returns_model_offload(self, mock_props, mm):
        mock_props.return_value = MagicMock(total_mem=16 * 1024**3)
        mm.offload_strategy = "auto"
        assert mm.resolve_offload_strategy(torch.device("cuda:0")) == "model_offload"

    @patch("torch.cuda.get_device_properties")
    def test_auto_8gb_returns_sequential_group_offload(self, mock_props, mm):
        mock_props.return_value = MagicMock(total_mem=8 * 1024**3)
        mm.offload_strategy = "auto"
        assert mm.resolve_offload_strategy(torch.device("cuda:0")) == "sequential_group_offload"

    @patch("torch.cuda.get_device_properties")
    def test_auto_low_vram_returns_group_offload(self, mock_props, mm):
        mock_props.return_value = MagicMock(total_mem=6 * 1024**3)
        mm.offload_strategy = "auto"
        assert mm.resolve_offload_strategy(torch.device("cuda:0")) == "group_offload"


# ---------------------------------------------------------------------------
# register_component / get_component
# ---------------------------------------------------------------------------


class TestManagedComponents:
    def test_register_and_get(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        assert mm.get_component("transformer") is module

    def test_get_missing_returns_none(self, mm):
        assert mm.get_component("nonexistent") is None

    def test_clear_resets_managed_components(self, mm):
        mm.register_component("vae", nn.Linear(4, 4))
        mm.clear()
        assert mm.get_component("vae") is None
        assert mm.applied_strategy is None


# ---------------------------------------------------------------------------
# use_components
# ---------------------------------------------------------------------------


class TestUseComponents:
    def test_noop_for_no_offload(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm._applied_strategy = "no_offload"

        with mm.use_components("transformer", device="cpu"):
            pass  # should not raise or move anything

    def test_noop_for_group_offload(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm._applied_strategy = "group_offload"

        with mm.use_components("transformer", device="cpu"):
            pass

    def test_model_offload_moves_to_device_and_back(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm._applied_strategy = "model_offload"

        # Module starts on CPU
        assert next(module.parameters()).device.type == "cpu"

        with mm.use_components("transformer", device="cpu"):
            # During context, still on CPU (since device="cpu")
            assert next(module.parameters()).device.type == "cpu"

        # After context, still on CPU
        assert next(module.parameters()).device.type == "cpu"

    def test_model_offload_handles_missing_component(self, mm):
        mm._applied_strategy = "model_offload"
        # Should not raise for unknown component names
        with mm.use_components("nonexistent", device="cpu"):
            pass

    def test_noop_for_sequential_group_offload_outside_use(self, mm):
        """sequential_group_offload's apply_offload_strategy does NOT install hooks."""
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "sequential_group_offload"

        result = mm.apply_offload_strategy("cpu")
        assert result == "sequential_group_offload"
        # No hooks installed
        assert not hasattr(module, "_diffusers_hook")
        # Module on CPU
        assert next(module.parameters()).device.type == "cpu"

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_sequential_group_offload_applies_hooks_in_use_components(self, mock_apply, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm._applied_strategy = "sequential_group_offload"

        with mm.use_components("transformer", device="cpu"):
            assert mock_apply.call_count == 1
            mock_apply.assert_called_once_with(
                module,
                onload_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=False,
            )

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_sequential_group_offload_removes_hooks_after_use(self, mock_apply, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm._applied_strategy = "sequential_group_offload"

        with mm.use_components("transformer", device="cpu"):
            pass

        # After exit, module should be on CPU with hooks removed
        assert next(module.parameters()).device.type == "cpu"

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_sequential_group_offload_respects_stream_options(self, mock_apply, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm._applied_strategy = "sequential_group_offload"
        mm.group_offload_use_stream = True
        mm.group_offload_low_cpu_mem = True

        with mm.use_components("transformer", device="cpu"):
            mock_apply.assert_called_once_with(
                module,
                onload_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
                low_cpu_mem_usage=True,
            )


# ---------------------------------------------------------------------------
# prepare_strategy_transition
# ---------------------------------------------------------------------------


class TestPrepareStrategyTransition:
    def test_noop_when_same_strategy(self, mm):
        mm._applied_strategy = "no_offload"
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)

        mm.prepare_strategy_transition("no_offload", "cpu")
        assert mm.applied_strategy == "no_offload"

    def test_leaving_no_offload_moves_to_cpu(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm._applied_strategy = "no_offload"

        mm.prepare_strategy_transition("model_offload", "cpu")
        assert mm.applied_strategy == "model_offload"
        assert next(module.parameters()).device.type == "cpu"

    def test_leaving_group_offload_calls_remove_hooks(self, mm):
        module = nn.Linear(4, 4)
        mock_registry = MagicMock()
        module._diffusers_hook = mock_registry
        mm.register_component("transformer", module)
        mm._applied_strategy = "group_offload"

        mm.prepare_strategy_transition("no_offload", "cpu")
        assert mm.applied_strategy == "no_offload"
        assert mock_registry.remove_hook.call_count == 3

    def test_leaving_sequential_group_offload_removes_hooks_and_moves_cpu(self, mm):
        module = nn.Linear(4, 4)
        mock_registry = MagicMock()
        module._diffusers_hook = mock_registry
        mm.register_component("transformer", module)
        mm._applied_strategy = "sequential_group_offload"

        mm.prepare_strategy_transition("no_offload", "cpu")
        assert mm.applied_strategy == "no_offload"
        assert mock_registry.remove_hook.call_count == 3
        assert next(module.parameters()).device.type == "cpu"

    def test_first_run_none_to_strategy(self, mm):
        mm.prepare_strategy_transition("group_offload", "cpu")
        assert mm.applied_strategy == "group_offload"


# ---------------------------------------------------------------------------
# remove_offload_hooks
# ---------------------------------------------------------------------------


class TestRemoveOffloadHooks:
    def test_removes_hooks_when_present(self):
        module = nn.Linear(4, 4)
        mock_registry = MagicMock()
        module._diffusers_hook = mock_registry

        ModelManager.remove_offload_hooks(module)

        expected_hooks = [
            "group_offloading",
            "layer_execution_tracker",
            "lazy_prefetch_group_offloading",
        ]
        for hook_name in expected_hooks:
            mock_registry.remove_hook.assert_any_call(hook_name, recurse=False)

    def test_noop_when_no_hooks(self):
        module = nn.Linear(4, 4)
        # Should not raise
        ModelManager.remove_offload_hooks(module)


# ---------------------------------------------------------------------------
# apply_offload_strategy
# ---------------------------------------------------------------------------


class TestApplyOffloadStrategy:
    def test_no_offload_moves_to_device(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "no_offload"

        result = mm.apply_offload_strategy("cpu")
        assert result == "no_offload"
        assert mm.applied_strategy == "no_offload"
        assert next(module.parameters()).device.type == "cpu"

    def test_model_offload_keeps_on_cpu(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "model_offload"

        result = mm.apply_offload_strategy("cpu")
        assert result == "model_offload"
        assert mm.applied_strategy == "model_offload"
        assert next(module.parameters()).device.type == "cpu"

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_group_offload_calls_apply(self, mock_apply, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "group_offload"

        result = mm.apply_offload_strategy("cpu")
        assert result == "group_offload"
        assert mock_apply.call_count == 1
        mock_apply.assert_called_once_with(
            module,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_group_offload_with_stream(self, mock_apply, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "group_offload"
        mm.group_offload_use_stream = True

        mm.apply_offload_strategy("cpu")
        mock_apply.assert_called_once_with(
            module,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
        )

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_group_offload_with_stream_and_low_cpu_mem(self, mock_apply, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "group_offload"
        mm.group_offload_use_stream = True
        mm.group_offload_low_cpu_mem = True

        mm.apply_offload_strategy("cpu")
        mock_apply.assert_called_once_with(
            module,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_group_offload_low_cpu_mem_without_stream_ignored(self, mock_apply, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "group_offload"
        mm.group_offload_use_stream = False
        mm.group_offload_low_cpu_mem = True

        mm.apply_offload_strategy("cpu")
        # low_cpu_mem_usage should NOT be passed when use_stream is False
        mock_apply.assert_called_once_with(
            module,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )

    def test_noop_when_strategy_unchanged(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "no_offload"

        mm.apply_offload_strategy("cpu")
        # Second call — strategy is already applied, should be a true no-op
        # (early return before prepare_strategy_transition is reached)
        with patch.object(mm, "prepare_strategy_transition", wraps=mm.prepare_strategy_transition) as wrapped:
            mm.apply_offload_strategy("cpu")
            wrapped.assert_not_called()

        assert mm.applied_strategy == "no_offload"

    def test_noop_when_no_managed_components(self, mm):
        mm.offload_strategy = "no_offload"

        result = mm.apply_offload_strategy("cpu")
        assert result == "no_offload"
        assert mm.applied_strategy == "no_offload"

    def test_transition_no_offload_to_model_offload(self, mm):
        module = nn.Linear(4, 4)
        mm.register_component("transformer", module)
        mm.offload_strategy = "no_offload"

        # First apply: no_offload
        mm.apply_offload_strategy("cpu")
        assert mm.applied_strategy == "no_offload"

        # Switch to model_offload
        mm.offload_strategy = "model_offload"
        result = mm.apply_offload_strategy("cpu")
        assert result == "model_offload"
        assert mm.applied_strategy == "model_offload"
        # prepare_strategy_transition moves components to CPU
        assert next(module.parameters()).device.type == "cpu"

    def test_returns_resolved_auto_strategy(self, mm):
        mm.offload_strategy = "auto"
        # On CPU device, auto resolves to group_offload
        with patch("diffusers.hooks.group_offloading.apply_group_offloading"):
            result = mm.apply_offload_strategy("cpu")
        assert result == "group_offload"
