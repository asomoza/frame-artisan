"""Tests for the Clear Graph action.

Verifies that clearing the graph resets all UI panels (LoRAs, source images,
audio, video conditioning) and clears VRAM.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QApplication

from frameartisan.app.event_bus import EventBus
from frameartisan.app.model_manager import ModelManager
from frameartisan.modules.generation.generation_settings import GenerationSettings


# Ensure QApplication exists for widget tests.
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakePrefs:
    pass


class FakeDirs:
    models_diffusers = "/tmp"
    temp_path = "/tmp"


@pytest.fixture()
def gen_settings():
    return GenerationSettings()


@pytest.fixture()
def event_bus():
    bus = EventBus()
    yield bus


def _unsubscribe_panel(panel):
    """Remove all event bus subscriptions for a panel to avoid stale callbacks."""
    bus = panel.event_bus
    for event_type in list(bus.subscribers):
        bus.subscribers[event_type] = [
            cb for cb in bus.subscribers[event_type]
            if not hasattr(cb, "__self__") or cb.__self__ is not panel
        ]


@pytest.fixture()
def lora_panel(qapp, gen_settings):
    from frameartisan.modules.generation.panels.lora_panel import LoraPanel

    panel = LoraPanel(gen_settings, FakePrefs(), FakeDirs())
    yield panel
    _unsubscribe_panel(panel)


@pytest.fixture()
def source_images_panel(qapp, gen_settings):
    from frameartisan.modules.generation.panels.source_images_panel import SourceImagesPanel

    panel = SourceImagesPanel(gen_settings, FakePrefs(), FakeDirs())
    yield panel
    _unsubscribe_panel(panel)


@pytest.fixture()
def audio_panel(qapp, gen_settings):
    from frameartisan.modules.generation.panels.audio_conditioning_panel import AudioConditioningPanel

    panel = AudioConditioningPanel(gen_settings, FakePrefs(), FakeDirs())
    yield panel
    _unsubscribe_panel(panel)


@pytest.fixture()
def video_panel(qapp, gen_settings):
    from frameartisan.modules.generation.panels.video_conditioning_panel import VideoConditioningPanel

    panel = VideoConditioningPanel(gen_settings, FakePrefs(), FakeDirs())
    yield panel
    _unsubscribe_panel(panel)


class TestClearGraphLoraPanelReset:
    def test_lora_items_cleared(self, lora_panel):
        event_bus = lora_panel.event_bus

        # Add two LoRAs
        event_bus.publish("lora", {
            "action": "add", "id": 1, "hash": "abc", "name": "LoRA 1",
            "filepath": "/path/1.safetensors", "weight": 1.0, "enabled": True,
        })
        event_bus.publish("lora", {
            "action": "add", "id": 2, "hash": "def", "name": "LoRA 2",
            "filepath": "/path/2.safetensors", "weight": 0.5, "enabled": True,
        })
        assert len(lora_panel._items) == 2

        # Fire graph_cleared
        event_bus.publish("graph_cleared", {})

        assert len(lora_panel._items) == 0


class TestClearGraphSourceImagesPanelReset:
    def test_condition_entries_cleared(self, source_images_panel):
        event_bus = source_images_panel.event_bus

        # Add two conditions
        event_bus.publish("visual_condition", {
            "action": "add", "condition_id": "img1",
            "source_thumb_path": "", "pixel_frame_index": 1, "strength": 1.0,
        })
        event_bus.publish("visual_condition", {
            "action": "add", "condition_id": "img2",
            "source_thumb_path": "", "pixel_frame_index": 5, "strength": 0.8,
        })
        assert len(source_images_panel._entries) == 2
        assert not source_images_panel.enabled_checkbox.isHidden()

        # Fire graph_cleared
        event_bus.publish("graph_cleared", {})

        assert len(source_images_panel._entries) == 0
        assert not source_images_panel.enabled_checkbox.isEnabled()


class TestClearGraphAudioPanelReset:
    def test_audio_state_cleared(self, audio_panel):
        event_bus = audio_panel.event_bus

        # Simulate audio loaded (set internal state directly since _set_audio probes the file)
        audio_panel._audio_path = "/tmp/test.wav"
        audio_panel._audio_duration = 5.0
        audio_panel.file_label.setText("test.wav")
        audio_panel.enabled_checkbox.setEnabled(True)
        audio_panel.enabled_checkbox.setChecked(True)
        audio_panel.remove_button.setEnabled(True)

        # Fire graph_cleared
        event_bus.publish("graph_cleared", {})

        assert audio_panel._audio_path is None
        assert audio_panel._audio_duration == 0.0
        assert audio_panel.file_label.text() == "No audio loaded"
        assert not audio_panel.enabled_checkbox.isEnabled()
        assert not audio_panel.enabled_checkbox.isChecked()
        assert not audio_panel.remove_button.isEnabled()

    def test_noop_when_no_audio(self, audio_panel):
        """Should not raise when no audio is loaded."""
        event_bus = audio_panel.event_bus
        event_bus.publish("graph_cleared", {})
        assert audio_panel._audio_path is None


class TestClearGraphVideoPanelReset:
    def test_video_state_cleared(self, video_panel):
        event_bus = video_panel.event_bus

        # Simulate video loaded (set internal state directly since _set_video probes the file)
        video_panel._video_path = "/tmp/test.mp4"
        video_panel._video_frame_count = 100
        video_panel._video_fps = 24.0
        video_panel._has_audio = True
        video_panel.file_label.setText("test.mp4")
        video_panel.enabled_checkbox.setEnabled(True)
        video_panel.enabled_checkbox.setChecked(True)
        video_panel.remove_button.setEnabled(True)
        video_panel.frame_slider.setEnabled(True)
        video_panel.mode_combo.setEnabled(True)

        # Fire graph_cleared
        event_bus.publish("graph_cleared", {})

        assert video_panel._video_path is None
        assert video_panel._video_frame_count == 0
        assert video_panel.file_label.text() == "No video loaded"
        assert not video_panel.enabled_checkbox.isEnabled()
        assert not video_panel.enabled_checkbox.isChecked()
        assert not video_panel.remove_button.isEnabled()
        assert not video_panel.frame_slider.isEnabled()
        assert not video_panel.mode_combo.isEnabled()

    def test_noop_when_no_video(self, video_panel):
        """Should not raise when no video is loaded."""
        event_bus = video_panel.event_bus
        event_bus.publish("graph_cleared", {})
        assert video_panel._video_path is None


class TestClearGraphAlsoClearsVRAM:
    """Verify that clear_graph also frees model memory."""

    def test_model_references_dropped(self):
        mm = ModelManager()

        model = nn.Linear(4, 4)
        mm.register_component("transformer", model)
        weak_model = weakref.ref(model)
        del model

        # Simulate what clear_graph does: calls mm.clear()
        mm.clear()
        gc.collect()

        assert weak_model() is None
