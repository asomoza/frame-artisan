from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)
from superqt import QLabeledDoubleSlider

from frameartisan.app.base_dialog import BaseSimpleDialog
from frameartisan.app.event_bus import EventBus
from frameartisan.buttons.linked_button import LinkedButton
from frameartisan.modules.generation.constants import get_default_granular_weights


NUM_LAYERS = 48

# Slider ranges
NORMAL_MIN, NORMAL_MAX = 0.0, 2.0
SLIDER_MIN, SLIDER_MAX = -30.0, 30.0


class LoraAdvancedDialog(BaseSimpleDialog):
    GRANULAR_SLIDER_MIN_HEIGHT = 200

    def __init__(self, dialog_key: str, config: dict):
        super().__init__("LoRA Advanced Dialog", minWidth=1400, minHeight=450)

        self.event_bus = EventBus()
        self.dialog_key = dialog_key
        self.lora_id = config.get("id")
        self.lora_hash = config.get("hash", "")
        self.lora_name = config.get("name", "")

        self.weight = config.get("weight", 1.0)
        self.video_strength = config.get("video_strength", 1.0)
        self.audio_strength = config.get("audio_strength", 1.0)
        self.granular_enabled = config.get("granular_transformer_weights_enabled", False)
        self.granular_weights = config.get("granular_transformer_weights", {})
        self.is_slider = config.get("is_slider", False)

        if not self.granular_weights:
            self.granular_weights = get_default_granular_weights(1)

        self.low_range = SLIDER_MIN if self.is_slider else NORMAL_MIN
        self.high_range = SLIDER_MAX if self.is_slider else NORMAL_MAX

        self.layer_sliders: dict[str, QLabeledDoubleSlider] = {}
        self.layer_linked_buttons: dict[str, LinkedButton] = {}
        self.layer_previous_values: dict[str, float] = {}
        self._updating_linked_sliders = False

        self._building = True
        self._init_ui()
        self._building = False

    def _init_ui(self):
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        # --- Top sliders in a grid ---
        sliders_layout = QGridLayout()

        self.transformer_label = QLabel("Transformer:")
        sliders_layout.addWidget(self.transformer_label, 0, 0)
        self.weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(self.low_range, self.high_range)
        self.weight_slider.setValue(self.weight)
        self.weight_slider.valueChanged.connect(self._on_weight_changed)
        sliders_layout.addWidget(self.weight_slider, 0, 1)

        video_label = QLabel("Video Strength:")
        sliders_layout.addWidget(video_label, 1, 0)
        self.video_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.video_slider.setRange(0.0, 2.0)
        self.video_slider.setValue(self.video_strength)
        self.video_slider.valueChanged.connect(self._on_video_strength_changed)
        sliders_layout.addWidget(self.video_slider, 1, 1)

        audio_label = QLabel("Audio Strength:")
        sliders_layout.addWidget(audio_label, 2, 0)
        self.audio_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.audio_slider.setRange(0.0, 2.0)
        self.audio_slider.setValue(self.audio_strength)
        self.audio_slider.valueChanged.connect(self._on_audio_strength_changed)
        sliders_layout.addWidget(self.audio_slider, 2, 1)

        self.main_layout.addLayout(sliders_layout)

        # Disable transformer slider when granular is active
        if self.granular_enabled:
            self.transformer_label.setDisabled(True)
            self.weight_slider.setDisabled(True)

        # --- Granular controls row ---
        granular_layout = QHBoxLayout()

        granular_checkbox = QCheckBox("Enable granular scales")
        granular_checkbox.setChecked(self.granular_enabled)
        granular_checkbox.toggled.connect(self._on_granular_toggled)
        granular_layout.addWidget(granular_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        self.slider_checkbox = QCheckBox("Slider")
        self.slider_checkbox.setChecked(self.is_slider)
        self.slider_checkbox.toggled.connect(self._on_slider_toggled)
        granular_layout.addWidget(self.slider_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_layout.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.preset_combo.addItems(["None", "All to 0.0", "All to 1.0"])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self.preset_combo)
        granular_layout.addLayout(preset_layout)

        self.main_layout.addLayout(granular_layout)

        # --- Granular frame ---
        self.granular_frame = QFrame()
        self.granular_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.granular_frame.setEnabled(self.granular_enabled)
        self.granular_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        sections_layout = QHBoxLayout()
        self._build_granular_layer_sliders(sections_layout)
        self.granular_frame.setLayout(sections_layout)
        self.granular_frame.setMaximumHeight(self.granular_frame.sizeHint().height())
        self.main_layout.addWidget(self.granular_frame)

        self.setLayout(self.dialog_layout)

    def _build_granular_layer_sliders(self, sections_layout: QHBoxLayout) -> None:
        self.layer_sliders.clear()
        self.layer_linked_buttons.clear()
        self.layer_previous_values.clear()

        sorted_keys = sorted(self.granular_weights.keys(), key=lambda k: int(k.split(".")[-1]))
        self._build_slider_section(sections_layout, sorted_keys, "L")

    def _build_slider_section(self, layout: QHBoxLayout, keys: list[str], label_prefix: str) -> None:
        for idx, layer_key in enumerate(keys):
            weight = self.granular_weights.get(layer_key, 1.0)
            layer_layout = QVBoxLayout()

            linked_button = LinkedButton()
            layer_layout.addWidget(linked_button, alignment=Qt.AlignmentFlag.AlignCenter)

            layer_slider = QLabeledDoubleSlider(Qt.Orientation.Vertical)
            layer_slider.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
            layer_slider.setMinimumHeight(self.GRANULAR_SLIDER_MIN_HEIGHT)
            layer_slider.setRange(self.low_range, self.high_range)
            layer_slider.setValue(weight)
            layer_slider.valueChanged.connect(lambda v, k=layer_key: self._on_slider_value_changed(k, v))
            layer_layout.addWidget(layer_slider)

            layer_label = QLabel(f"{label_prefix}{idx + 1}")
            layer_layout.addWidget(layer_label, alignment=Qt.AlignmentFlag.AlignCenter)

            layout.addLayout(layer_layout)
            self.layer_sliders[layer_key] = layer_slider
            self.layer_linked_buttons[layer_key] = linked_button
            self.layer_previous_values[layer_key] = weight

    def _on_slider_value_changed(self, layer_key: str, value: float) -> None:
        if self._updating_linked_sliders:
            return

        previous_value = self.layer_previous_values.get(layer_key, value)
        delta = value - previous_value
        self.layer_previous_values[layer_key] = value

        is_source_linked = self.layer_linked_buttons[layer_key].linked

        if is_source_linked and delta != 0:
            self._updating_linked_sliders = True
            try:
                for other_key, linked_button in self.layer_linked_buttons.items():
                    if other_key != layer_key and linked_button.linked:
                        other_slider = self.layer_sliders[other_key]
                        other_previous = self.layer_previous_values[other_key]
                        new_value = max(self.low_range, min(self.high_range, other_previous + delta))
                        self.layer_previous_values[other_key] = new_value
                        other_slider.setValue(new_value)
                        self.granular_weights[other_key] = new_value
            finally:
                self._updating_linked_sliders = False

        self.granular_weights[layer_key] = value
        self._publish_update()

    # --- Top slider handlers ---

    def _on_weight_changed(self, value: float):
        self.weight = value
        self._publish_update()

    def _on_video_strength_changed(self, value: float):
        self.video_strength = value
        self._publish_update()

    def _on_audio_strength_changed(self, value: float):
        self.audio_strength = value
        self._publish_update()

    # --- Granular controls ---

    def _on_granular_toggled(self, checked: bool):
        self.granular_enabled = checked
        self.granular_frame.setEnabled(checked)
        self.transformer_label.setDisabled(checked)
        self.weight_slider.setDisabled(checked)
        self._publish_update()

    def _on_slider_toggled(self, checked: bool):
        self.is_slider = checked

        if checked:
            self.low_range = SLIDER_MIN
            self.high_range = SLIDER_MAX
        else:
            self.low_range = NORMAL_MIN
            self.high_range = NORMAL_MAX

        self.weight_slider.setRange(self.low_range, self.high_range)
        for slider in self.layer_sliders.values():
            slider.setRange(self.low_range, self.high_range)

        self._publish_update()

    def _on_preset_selected(self, index: int):
        if index == 0:
            return

        value = 0.0 if index == 1 else 1.0

        self._updating_linked_sliders = True
        try:
            for layer_key in self.granular_weights:
                self.granular_weights[layer_key] = value
                self.layer_sliders[layer_key].setValue(value)
                self.layer_previous_values[layer_key] = value
        finally:
            self._updating_linked_sliders = False

        self.preset_combo.setCurrentIndex(0)
        self._publish_update()

    # --- Event publishing ---

    def _publish_update(self):
        if self._building:
            return
        self.event_bus.publish(
            "lora",
            {
                "action": "update_weights",
                "id": self.lora_id,
                "hash": self.lora_hash,
                "config": self._get_config(),
            },
        )

    def _get_config(self) -> dict:
        return {
            "weight": self.weight,
            "video_strength": self.video_strength,
            "audio_strength": self.audio_strength,
            "granular_transformer_weights_enabled": self.granular_enabled,
            "granular_transformer_weights": dict(self.granular_weights),
            "is_slider": self.is_slider,
        }

    def closeEvent(self, event):
        event.ignore()
        self.event_bus.publish(
            "manage_dialog",
            {
                "dialog_type": "lora_advanced",
                "action": "close",
                "dialog_key": self.dialog_key,
                "id": self.lora_id,
                "hash": self.lora_hash,
            },
        )
