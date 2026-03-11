"""Regression tests for use_components restoring group_offload hooks after model_offload override.

When the applied strategy is group_offload but a node uses strategy_override="model_offload"
(e.g. condition_encode for the VAE), the model_offload branch strips the group_offload hooks.
After the context manager exits, the hooks must be restored so subsequent nodes (e.g. streaming
decode) that rely on group_offload hooks still work.

See: https://github.com/... (CUDABFloat16Type vs CPUBFloat16Type error with streaming decode + I2V)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn as nn

from frameartisan.app.model_manager import ModelManager


@pytest.fixture()
def mm() -> ModelManager:
    return ModelManager()


class TestModelOffloadOverrideRestoresGroupOffloadHooks:
    """use_components with strategy_override='model_offload' must restore group_offload hooks."""

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_hooks_restored_after_model_offload_override(self, mock_apply, mm):
        """When applied strategy is group_offload and override is model_offload,
        hooks must be re-applied after the context manager exits."""
        vae = nn.Linear(4, 4)
        mm.register_component("vae", vae)
        mm._applied_strategy = "group_offload"

        with mm.use_components("vae", device="cpu", strategy_override="model_offload"):
            # Inside context: hooks were stripped, module moved to device
            pass

        # After exit: apply_group_offloading should have been called to restore hooks
        assert mock_apply.call_count == 1
        mock_apply.assert_called_once_with(
            vae,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_hooks_restored_with_stream_options(self, mock_apply, mm):
        """Restored hooks must respect use_stream and low_cpu_mem_usage settings."""
        vae = nn.Linear(4, 4)
        mm.register_component("vae", vae)
        mm._applied_strategy = "group_offload"
        mm.group_offload_use_stream = True
        mm.group_offload_low_cpu_mem = True

        with mm.use_components("vae", device="cpu", strategy_override="model_offload"):
            pass

        mock_apply.assert_called_once_with(
            vae,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_hooks_restored_for_multiple_components(self, mock_apply, mm):
        """All components passed to use_components get their hooks restored."""
        vae = nn.Linear(4, 4)
        audio_vae = nn.Linear(4, 4)
        vocoder = nn.Linear(4, 4)
        mm.register_component("vae", vae)
        mm.register_component("audio_vae", audio_vae)
        mm.register_component("vocoder", vocoder)
        mm._applied_strategy = "group_offload"

        with mm.use_components("vae", "audio_vae", "vocoder", device="cpu", strategy_override="model_offload"):
            pass

        assert mock_apply.call_count == 3
        mock_apply.assert_any_call(
            vae,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )
        mock_apply.assert_any_call(
            audio_vae,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )
        mock_apply.assert_any_call(
            vocoder,
            onload_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_no_restore_when_strategy_is_model_offload(self, mock_apply, mm):
        """When the actual strategy IS model_offload (not an override), no hooks to restore."""
        vae = nn.Linear(4, 4)
        mm.register_component("vae", vae)
        mm._applied_strategy = "model_offload"

        with mm.use_components("vae", device="cpu"):
            pass

        mock_apply.assert_not_called()

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_no_restore_when_strategy_is_sequential(self, mock_apply, mm):
        """sequential_group_offload has its own cleanup — no extra restore needed."""
        vae = nn.Linear(4, 4)
        mm.register_component("vae", vae)
        mm._applied_strategy = "sequential_group_offload"

        with mm.use_components("vae", device="cpu", strategy_override="model_offload"):
            pass

        # sequential doesn't need restore — hooks are applied/removed on demand
        mock_apply.assert_not_called()

    @patch("diffusers.hooks.group_offloading.apply_group_offloading")
    def test_hooks_restored_even_on_exception(self, mock_apply, mm):
        """Hooks must be restored even if the wrapped code raises."""
        vae = nn.Linear(4, 4)
        mm.register_component("vae", vae)
        mm._applied_strategy = "group_offload"

        with pytest.raises(RuntimeError, match="boom"):
            with mm.use_components("vae", device="cpu", strategy_override="model_offload"):
                raise RuntimeError("boom")

        # Hooks should still be restored in the finally block
        assert mock_apply.call_count == 1

    def test_module_back_on_cpu_after_override(self, mm):
        """After model_offload override exits, module must be on CPU."""
        vae = nn.Linear(4, 4)
        mm.register_component("vae", vae)
        mm._applied_strategy = "group_offload"

        with patch("diffusers.hooks.group_offloading.apply_group_offloading"):
            with mm.use_components("vae", device="cpu", strategy_override="model_offload"):
                pass

        assert next(vae.parameters()).device.type == "cpu"
