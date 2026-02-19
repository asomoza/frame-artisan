"""Tests for MultiModal Guidance: formula, hooks, settings, graph wiring, widgets."""

from __future__ import annotations

import tempfile

import attr
import pytest
import torch
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from frameartisan.modules.generation.constants import ADVANCED_GUIDANCE_DEFAULTS
from frameartisan.modules.generation.generation_settings import GenerationSettings
from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph
from frameartisan.modules.generation.graph.nodes.ltx2_guidance import (
    _zero_output_hook,
    calculate_guidance,
    modality_isolation_hooks,
    parse_stg_blocks,
    stg_perturbation_hooks,
)
from frameartisan.modules.generation.panels.generation_panel import GenerationPanel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@attr.s
class _FakeDirs:
    outputs_videos: str = attr.ib(default="/tmp/outputs_videos")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def gen_settings():
    return GenerationSettings()


@pytest.fixture()
def directories():
    return _FakeDirs()


@pytest.fixture()
def graph(gen_settings, directories):
    return create_default_ltx2_graph(gen_settings, directories)


@pytest.fixture()
def panel(qapp):
    settings = GenerationSettings()

    class FakePrefs:
        pass

    class FakeDirs:
        pass

    return GenerationPanel(settings, FakePrefs(), FakeDirs())


@pytest.fixture()
def qsettings():
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as f:
        path = f.name
    qs = QSettings(path, QSettings.Format.IniFormat)
    yield qs
    qs.sync()


# ---------------------------------------------------------------------------
# calculate_guidance formula tests
# ---------------------------------------------------------------------------


class TestCalculateGuidance:
    def test_cfg_only(self):
        """With only CFG active, result matches standard CFG formula."""
        cond = torch.ones(1, 4, 8)
        uncond = torch.zeros(1, 4, 8)
        cfg_scale = 3.0

        video, audio = calculate_guidance(
            cond,
            cond,
            uncond,
            uncond,
            0.0,
            0.0,
            0.0,
            0.0,
            cfg_scale=cfg_scale,
            stg_scale=0.0,
            modality_scale=1.0,
            rescale_scale=0.0,
        )

        # pred = cond + (cfg_scale - 1) * (cond - uncond) = 1 + 2*(1-0) = 3
        expected = cond + (cfg_scale - 1.0) * (cond - uncond)
        assert torch.allclose(video, expected)

    def test_stg_only(self):
        """STG term: stg_scale * (cond - perturbed)."""
        cond = torch.ones(1, 4, 8) * 2
        perturbed = torch.ones(1, 4, 8)
        stg_scale = 1.5

        video, audio = calculate_guidance(
            cond,
            cond,
            0.0,
            0.0,
            perturbed,
            perturbed,
            0.0,
            0.0,
            cfg_scale=1.0,
            stg_scale=stg_scale,
            modality_scale=1.0,
            rescale_scale=0.0,
        )

        # pred = cond + 1.5 * (2 - 1) = 2 + 1.5 = 3.5
        expected = cond + stg_scale * (cond - perturbed)
        assert torch.allclose(video, expected)

    def test_modality_only(self):
        """Modality term: (modality_scale - 1) * (cond - isolated)."""
        cond = torch.ones(1, 4, 8) * 3
        isolated = torch.ones(1, 4, 8)
        modality_scale = 2.0

        video, audio = calculate_guidance(
            cond,
            cond,
            0.0,
            0.0,
            0.0,
            0.0,
            isolated,
            isolated,
            cfg_scale=1.0,
            stg_scale=0.0,
            modality_scale=modality_scale,
            rescale_scale=0.0,
        )

        expected = cond + (modality_scale - 1.0) * (cond - isolated)
        assert torch.allclose(video, expected)

    def test_all_disabled_returns_cond(self):
        """All scales at identity values → returns cond unchanged."""
        cond = torch.randn(1, 4, 8)
        video, audio = calculate_guidance(
            cond,
            cond,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            cfg_scale=1.0,
            stg_scale=0.0,
            modality_scale=1.0,
            rescale_scale=0.0,
        )
        assert torch.allclose(video, cond)

    def test_combined_formula(self):
        """All guidance terms active simultaneously."""
        cond = torch.ones(1, 4, 8) * 5
        uncond = torch.ones(1, 4, 8) * 1
        perturbed = torch.ones(1, 4, 8) * 3
        isolated = torch.ones(1, 4, 8) * 2

        video, _ = calculate_guidance(
            cond,
            cond,
            uncond,
            uncond,
            perturbed,
            perturbed,
            isolated,
            isolated,
            cfg_scale=2.0,
            stg_scale=0.5,
            modality_scale=1.5,
            rescale_scale=0.0,
        )

        # pred = 5 + (2-1)*(5-1) + 0.5*(5-3) + (1.5-1)*(5-2) = 5 + 4 + 1 + 1.5 = 11.5
        expected = torch.ones(1, 4, 8) * 11.5
        assert torch.allclose(video, expected)

    def test_variance_rescaling(self):
        """Rescale > 0 adjusts variance toward conditional prediction."""
        torch.manual_seed(42)
        cond = torch.randn(1, 4, 8)
        uncond = torch.randn(1, 4, 8)

        video_no_rescale, _ = calculate_guidance(
            cond,
            cond,
            uncond,
            uncond,
            0.0,
            0.0,
            0.0,
            0.0,
            cfg_scale=7.0,
            stg_scale=0.0,
            modality_scale=1.0,
            rescale_scale=0.0,
        )
        video_rescale, _ = calculate_guidance(
            cond,
            cond,
            uncond,
            uncond,
            0.0,
            0.0,
            0.0,
            0.0,
            cfg_scale=7.0,
            stg_scale=0.0,
            modality_scale=1.0,
            rescale_scale=0.7,
        )

        # Rescaled result should differ from non-rescaled
        assert not torch.allclose(video_no_rescale, video_rescale)

    def test_audio_independent(self):
        """Video and audio guidance computed independently."""
        cond_v = torch.ones(1, 4, 8) * 2
        cond_a = torch.ones(1, 4, 8) * 10
        uncond_v = torch.zeros(1, 4, 8)
        uncond_a = torch.ones(1, 4, 8) * 5

        video, audio = calculate_guidance(
            cond_v,
            cond_a,
            uncond_v,
            uncond_a,
            0.0,
            0.0,
            0.0,
            0.0,
            cfg_scale=3.0,
            stg_scale=0.0,
            modality_scale=1.0,
            rescale_scale=0.0,
        )

        expected_v = cond_v + 2.0 * (cond_v - uncond_v)  # 2 + 2*2 = 6
        expected_a = cond_a + 2.0 * (cond_a - uncond_a)  # 10 + 2*5 = 20
        assert torch.allclose(video, expected_v)
        assert torch.allclose(audio, expected_a)


# ---------------------------------------------------------------------------
# parse_stg_blocks
# ---------------------------------------------------------------------------


class TestParseStgBlocks:
    def test_single_block(self):
        assert parse_stg_blocks("29") == [29]

    def test_multiple_blocks(self):
        assert parse_stg_blocks("28, 29") == [28, 29]

    def test_empty_string(self):
        assert parse_stg_blocks("") == []

    def test_whitespace(self):
        assert parse_stg_blocks("  10 , 20 , 30  ") == [10, 20, 30]


# ---------------------------------------------------------------------------
# _zero_output_hook
# ---------------------------------------------------------------------------


class TestZeroOutputHook:
    def test_returns_zeros(self):
        m = torch.nn.Linear(8, 8)
        x = torch.randn(2, 8)
        # Simulate calling the hook with (module, args, output)
        out = _zero_output_hook(m, (x,), m(x))
        assert torch.all(out == 0)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Hook context managers (using mock transformer blocks)
# ---------------------------------------------------------------------------


class _FakeAttn(torch.nn.Module):
    def forward(self, x):
        return x


class _FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _FakeAttn()
        self.audio_attn1 = _FakeAttn()
        self.audio_to_video_attn = _FakeAttn()
        self.video_to_audio_attn = _FakeAttn()


class _FakeTransformer(torch.nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_FakeBlock() for _ in range(num_blocks)])


class TestStgPerturbationHooks:
    def test_zeros_targeted_block_output(self):
        """Hook zeros output of targeted block but not others."""
        transformer = _FakeTransformer(3)
        x = torch.randn(2, 4, 8)

        with stg_perturbation_hooks(transformer, [1]):
            # Targeted block returns zeros
            out = transformer.transformer_blocks[1].attn1(x)
            assert torch.all(out == 0)
            # Untargeted block still returns identity
            out0 = transformer.transformer_blocks[0].attn1(x)
            assert torch.allclose(out0, x)

    def test_module_not_replaced(self):
        """Modules are NOT replaced — hooks are used instead."""
        transformer = _FakeTransformer(3)
        original_attn = transformer.transformer_blocks[1].attn1

        with stg_perturbation_hooks(transformer, [1]):
            assert transformer.transformer_blocks[1].attn1 is original_attn

    def test_hooks_removed_after_exit(self):
        """After context exit, output is no longer zeroed."""
        transformer = _FakeTransformer(3)
        x = torch.randn(2, 4, 8)

        with stg_perturbation_hooks(transformer, [1]):
            pass

        out = transformer.transformer_blocks[1].attn1(x)
        assert torch.allclose(out, x)

    def test_hooks_removed_on_exception(self):
        transformer = _FakeTransformer(3)
        x = torch.randn(2, 4, 8)

        with pytest.raises(RuntimeError):
            with stg_perturbation_hooks(transformer, [2]):
                raise RuntimeError("test")

        out = transformer.transformer_blocks[2].attn1(x)
        assert torch.allclose(out, x)

    def test_audio_attn_also_zeroed(self):
        transformer = _FakeTransformer(3)
        x = torch.randn(2, 4, 8)

        with stg_perturbation_hooks(transformer, [0]):
            out = transformer.transformer_blocks[0].audio_attn1(x)
            assert torch.all(out == 0)


class TestModalityIsolationHooks:
    def test_zeros_cross_attention_all_blocks(self):
        transformer = _FakeTransformer(3)
        x = torch.randn(2, 4, 8)

        with modality_isolation_hooks(transformer):
            for block in transformer.transformer_blocks:
                out_a2v = block.audio_to_video_attn(x)
                out_v2a = block.video_to_audio_attn(x)
                assert torch.all(out_a2v == 0)
                assert torch.all(out_v2a == 0)

    def test_hooks_removed_after_exit(self):
        transformer = _FakeTransformer(3)
        x = torch.randn(2, 4, 8)

        with modality_isolation_hooks(transformer):
            pass

        for block in transformer.transformer_blocks:
            assert torch.allclose(block.audio_to_video_attn(x), x)
            assert torch.allclose(block.video_to_audio_attn(x), x)

    def test_self_attention_untouched(self):
        transformer = _FakeTransformer(3)
        x = torch.randn(2, 4, 8)

        with modality_isolation_hooks(transformer):
            for block in transformer.transformer_blocks:
                assert torch.allclose(block.attn1(x), x)
                assert torch.allclose(block.audio_attn1(x), x)


# ---------------------------------------------------------------------------
# Generation Settings
# ---------------------------------------------------------------------------


class TestAdvancedGuidanceSettings:
    def test_defaults(self):
        s = GenerationSettings()
        assert s.advanced_guidance is False
        assert s.stg_scale == 0.0
        assert s.stg_blocks == "29"
        assert s.rescale_scale == 0.0
        assert s.modality_scale == 1.0
        assert s.guidance_skip_step == 0

    def test_round_trip(self, qsettings):
        s = GenerationSettings(
            advanced_guidance=True,
            stg_scale=1.5,
            stg_blocks="28,29",
            rescale_scale=0.7,
            modality_scale=3.0,
            guidance_skip_step=2,
        )
        s.save(qsettings)
        qsettings.sync()

        loaded = GenerationSettings.load(qsettings)
        assert loaded.advanced_guidance is True
        assert loaded.stg_scale == 1.5
        assert loaded.stg_blocks == "28,29"
        assert loaded.rescale_scale == 0.7
        assert loaded.modality_scale == 3.0
        assert loaded.guidance_skip_step == 2

    def test_defaults_when_absent(self, qsettings):
        loaded = GenerationSettings.load(qsettings)
        assert loaded.advanced_guidance is False
        assert loaded.stg_scale == 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestAdvancedGuidanceDefaults:
    def test_defaults_dict_has_expected_keys(self):
        assert "stg_scale" in ADVANCED_GUIDANCE_DEFAULTS
        assert "stg_blocks" in ADVANCED_GUIDANCE_DEFAULTS
        assert "rescale_scale" in ADVANCED_GUIDANCE_DEFAULTS
        assert "modality_scale" in ADVANCED_GUIDANCE_DEFAULTS
        assert "guidance_skip_step" in ADVANCED_GUIDANCE_DEFAULTS


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


class TestGraphAdvancedGuidanceNodes:
    def test_advanced_guidance_node_exists(self, graph):
        assert graph.get_node_by_name("advanced_guidance") is not None

    def test_stg_scale_node_exists(self, graph):
        assert graph.get_node_by_name("stg_scale") is not None

    def test_stg_blocks_node_exists(self, graph):
        assert graph.get_node_by_name("stg_blocks") is not None

    def test_rescale_scale_node_exists(self, graph):
        assert graph.get_node_by_name("rescale_scale") is not None

    def test_modality_scale_node_exists(self, graph):
        assert graph.get_node_by_name("modality_scale") is not None

    def test_guidance_skip_step_node_exists(self, graph):
        assert graph.get_node_by_name("guidance_skip_step") is not None

    def test_denoise_connected_to_advanced_guidance(self, graph):
        denoise = graph.get_node_by_name("denoise")
        assert "advanced_guidance" in denoise.connections

    def test_denoise_connected_to_stg_scale(self, graph):
        denoise = graph.get_node_by_name("denoise")
        assert "stg_scale" in denoise.connections

    def test_second_pass_denoise_connected(self, graph):
        sp_denoise = graph.get_node_by_name("second_pass_denoise")
        assert "advanced_guidance" in sp_denoise.connections
        assert "stg_scale" in sp_denoise.connections
        assert "modality_scale" in sp_denoise.connections


# ---------------------------------------------------------------------------
# Widget tests
# ---------------------------------------------------------------------------


class TestGenerationPanelAdvancedGuidance:
    def test_has_advanced_guidance_checkbox(self, panel):
        assert hasattr(panel, "advanced_guidance_checkbox")

    def test_advanced_guidance_frame_hidden_by_default(self, panel):
        assert panel.advanced_guidance_frame.isHidden()

    def test_advanced_guidance_frame_visible_when_checked(self, panel):
        panel.advanced_guidance_checkbox.setChecked(True)
        assert not panel.advanced_guidance_frame.isHidden()

    def test_stg_scale_slider_range(self, panel):
        assert panel.stg_scale_slider.minimum() == 0.0
        assert panel.stg_scale_slider.maximum() == 5.0

    def test_rescale_scale_slider_range(self, panel):
        assert panel.rescale_scale_slider.minimum() == 0.0
        assert panel.rescale_scale_slider.maximum() == 1.0

    def test_modality_scale_slider_range(self, panel):
        assert panel.modality_scale_slider.minimum() == 1.0
        assert panel.modality_scale_slider.maximum() == 10.0

    def test_guidance_skip_step_slider_range(self, panel):
        assert panel.guidance_skip_step_slider.minimum() == 0
        assert panel.guidance_skip_step_slider.maximum() == 10

    def test_has_stg_blocks_edit(self, panel):
        assert hasattr(panel, "stg_blocks_edit")
        assert panel.stg_blocks_edit.text() == "29"

    def test_update_panel_sets_advanced_guidance(self, panel):
        from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject

        panel.update_panel(
            640,
            352,
            24,
            4.0,
            5,
            24,
            ModelDataObject(),
            advanced_guidance=True,
            stg_scale=1.5,
            stg_blocks="28,29",
            rescale_scale=0.5,
            modality_scale=2.0,
            guidance_skip_step=3,
        )
        assert panel.advanced_guidance_checkbox.isChecked()
        assert panel.stg_scale_slider.value() == pytest.approx(1.5, abs=0.05)
        assert panel.stg_blocks_edit.text() == "28,29"
        assert panel.rescale_scale_slider.value() == pytest.approx(0.5, abs=0.05)
        assert panel.modality_scale_slider.value() == pytest.approx(2.0, abs=0.05)
        assert panel.guidance_skip_step_slider.value() == 3


# ---------------------------------------------------------------------------
# Denoise node optional inputs
# ---------------------------------------------------------------------------


class TestDenoiseNodeOptionalInputs:
    def test_advanced_guidance_in_optional_inputs(self):
        from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

        assert "advanced_guidance" in LTX2DenoiseNode.OPTIONAL_INPUTS
        assert "stg_scale" in LTX2DenoiseNode.OPTIONAL_INPUTS
        assert "stg_blocks" in LTX2DenoiseNode.OPTIONAL_INPUTS
        assert "rescale_scale" in LTX2DenoiseNode.OPTIONAL_INPUTS
        assert "modality_scale" in LTX2DenoiseNode.OPTIONAL_INPUTS
        assert "guidance_skip_step" in LTX2DenoiseNode.OPTIONAL_INPUTS
