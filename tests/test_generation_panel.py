"""Tests for GenerationPanel: sliders, update_panel API, event publishing."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from frameartisan.modules.generation.generation_settings import GenerationSettings
from frameartisan.modules.generation.panels.generation_panel import GenerationPanel


# Ensure QApplication exists for widget tests.
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def panel(qapp):
    settings = GenerationSettings()

    # Fake preferences and directories (panel only reads from them at construction)
    class FakePrefs:
        pass

    class FakeDirs:
        pass

    return GenerationPanel(settings, FakePrefs(), FakeDirs())


class TestGenerationPanelWidgets:
    def test_has_duration_slider(self, panel):
        assert hasattr(panel, "duration_slider")

    def test_has_frame_rate_slider(self, panel):
        assert hasattr(panel, "frame_rate_slider")

    def test_no_scheduler_combobox(self, panel):
        assert not hasattr(panel, "scheduler_combobox")

    def test_no_shift_slider(self, panel):
        assert not hasattr(panel, "shift_slider")

    def test_no_guidance_start_end_slider(self, panel):
        assert not hasattr(panel, "guidance_start_end_slider")

    def test_duration_slider_range(self, panel):
        assert panel.duration_slider.minimum() == 1
        assert panel.duration_slider.maximum() == 20

    def test_frame_rate_slider_range(self, panel):
        assert panel.frame_rate_slider.minimum() == 8
        assert panel.frame_rate_slider.maximum() == 50

    def test_has_use_stream_checkbox(self, panel):
        assert hasattr(panel, "use_stream_checkbox")

    def test_has_low_cpu_mem_checkbox(self, panel):
        assert hasattr(panel, "low_cpu_mem_checkbox")

    def test_group_offload_checkboxes_hidden_by_default(self, panel):
        """Checkboxes hidden when offload strategy is not group_offload."""
        assert panel.use_stream_checkbox.isHidden()
        assert panel.low_cpu_mem_checkbox.isHidden()

    def test_group_offload_checkboxes_visible_when_group_offload(self, panel):
        # Set strategy to group_offload
        for i in range(panel.offload_strategy_combobox.count()):
            if panel.offload_strategy_combobox.itemData(i) == "group_offload":
                panel.offload_strategy_combobox.setCurrentIndex(i)
                break
        assert not panel.use_stream_checkbox.isHidden()
        assert not panel.low_cpu_mem_checkbox.isHidden()

    def test_low_cpu_mem_disabled_when_stream_unchecked(self, panel):
        # Set strategy to group_offload
        for i in range(panel.offload_strategy_combobox.count()):
            if panel.offload_strategy_combobox.itemData(i) == "group_offload":
                panel.offload_strategy_combobox.setCurrentIndex(i)
                break
        panel.use_stream_checkbox.setChecked(False)
        assert not panel.low_cpu_mem_checkbox.isEnabled()

    def test_low_cpu_mem_enabled_when_stream_checked(self, panel):
        # Set strategy to group_offload
        for i in range(panel.offload_strategy_combobox.count()):
            if panel.offload_strategy_combobox.itemData(i) == "group_offload":
                panel.offload_strategy_combobox.setCurrentIndex(i)
                break
        panel.use_stream_checkbox.setChecked(True)
        assert panel.low_cpu_mem_checkbox.isEnabled()


class TestGenerationPanelUpdatePanel:
    def test_update_panel_sets_duration(self, panel):
        settings = GenerationSettings()
        panel.update_panel(
            width=640,
            height=352,
            num_inference_steps=20,
            guidance_scale=4.0,
            video_duration=10,
            frame_rate=24,
            model=settings.model,
        )
        assert panel.duration_slider.value() == 10

    def test_update_panel_sets_frame_rate(self, panel):
        settings = GenerationSettings()
        panel.update_panel(
            width=640,
            height=352,
            num_inference_steps=20,
            guidance_scale=4.0,
            video_duration=5,
            frame_rate=30,
            model=settings.model,
        )
        assert panel.frame_rate_slider.value() == 30


class TestGenerationPanelModelTypeConstraints:
    def test_distilled_disables_steps_slider(self, panel):
        panel._apply_model_type_constraints(2)
        assert not panel.steps_slider.isEnabled()

    def test_distilled_disables_guidance_slider(self, panel):
        panel._apply_model_type_constraints(2)
        assert not panel.guidance_slider.isEnabled()

    def test_distilled_sets_fixed_steps(self, panel):
        panel._apply_model_type_constraints(2)
        assert panel.steps_slider.value() == 8

    def test_distilled_sets_fixed_guidance(self, panel):
        panel._apply_model_type_constraints(2)
        assert panel.guidance_slider.value() == pytest.approx(1.0)

    def test_standard_re_enables_steps_slider(self, panel):
        panel._apply_model_type_constraints(2)
        panel._apply_model_type_constraints(1)
        assert panel.steps_slider.isEnabled()

    def test_standard_re_enables_guidance_slider(self, panel):
        panel._apply_model_type_constraints(2)
        panel._apply_model_type_constraints(1)
        assert panel.guidance_slider.isEnabled()

    def test_standard_restores_saved_steps(self, panel):
        panel.steps_slider.setValue(30)
        panel._apply_model_type_constraints(2)
        assert panel.steps_slider.value() == 8
        panel._apply_model_type_constraints(1)
        assert panel.steps_slider.value() == 30

    def test_standard_restores_saved_guidance(self, panel):
        panel.guidance_slider.setValue(5.0)
        panel._apply_model_type_constraints(2)
        assert panel.guidance_slider.value() == pytest.approx(1.0)
        panel._apply_model_type_constraints(1)
        assert panel.guidance_slider.value() == pytest.approx(5.0)


class TestSecondPassPanelWidgets:
    def test_has_second_pass_checkbox(self, panel):
        assert hasattr(panel, "second_pass_checkbox")

    def test_second_pass_checkbox_unchecked_by_default(self, panel):
        assert not panel.second_pass_checkbox.isChecked()

    def test_second_pass_frame_hidden_by_default(self, panel):
        """2nd pass frame should be hidden when checkbox is unchecked."""
        assert panel.second_pass_frame.isHidden()

    def test_second_pass_frame_visible_when_checked(self, panel):
        panel.second_pass_checkbox.setChecked(True)
        assert not panel.second_pass_frame.isHidden()

    def test_upsampled_resolution_label_text(self, panel):
        """Label should show 2x the current dimensions."""
        panel.second_pass_checkbox.setChecked(True)
        # Default dims are 640x352
        text = panel.upsampled_resolution_label.text()
        assert "1280" in text
        assert "704" in text

    def test_second_pass_distilled_constraints(self, panel):
        """When 2nd pass model type is distilled, sliders should be locked."""
        panel._apply_second_pass_model_type_constraints(2)
        assert not panel.second_pass_steps_slider.isEnabled()
        assert not panel.second_pass_guidance_slider.isEnabled()
        assert panel.second_pass_steps_slider.value() == 3
        assert panel.second_pass_guidance_slider.value() == pytest.approx(1.0)

    def test_second_pass_standard_unlocks_sliders(self, panel):
        panel._apply_second_pass_model_type_constraints(2)
        panel._apply_second_pass_model_type_constraints(1)
        assert panel.second_pass_steps_slider.isEnabled()
        assert panel.second_pass_guidance_slider.isEnabled()


class TestGenerationPanelEventBusPublish:
    def test_duration_slider_publishes_generation_change(self, panel):
        from frameartisan.app.event_bus import EventBus

        received = []
        EventBus().subscribe("generation_change", received.append)

        try:
            # Move to a different value first so the signal fires on the next set
            panel.duration_slider.setValue(1)
            panel.duration_slider.setValue(10)
        finally:
            EventBus().unsubscribe("generation_change", received.append)

        duration_events = [e for e in received if e.get("attr") == "video_duration"]
        assert duration_events, "No video_duration event published"
        assert duration_events[-1]["value"] == 10

    def test_frame_rate_slider_publishes_generation_change(self, panel):
        from frameartisan.app.event_bus import EventBus

        received = []
        EventBus().subscribe("generation_change", received.append)

        try:
            panel.frame_rate_slider.setValue(25)
        finally:
            EventBus().unsubscribe("generation_change", received.append)

        frame_rate_events = [e for e in received if e.get("attr") == "frame_rate"]
        assert frame_rate_events, "No frame_rate event published"
        assert frame_rate_events[-1]["value"] == 25
