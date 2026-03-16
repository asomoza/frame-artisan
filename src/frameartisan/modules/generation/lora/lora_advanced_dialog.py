from __future__ import annotations

import os
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)
from superqt import QLabeledDoubleSlider, QLabeledSlider

from frameartisan.app.base_dialog import BaseSimpleDialog
from frameartisan.app.event_bus import EventBus
from frameartisan.buttons.brush_erase_button import BrushEraseButton
from frameartisan.buttons.color_button import ColorButton
from frameartisan.buttons.linked_button import LinkedButton
from frameartisan.modules.generation.constants import get_default_granular_weights
from frameartisan.modules.generation.image.image_widget import ImageWidget
from frameartisan.modules.generation.image.layer_manager_widget import LayerManagerWidget


NUM_LAYERS = 48

# Slider ranges
NORMAL_MIN, NORMAL_MAX = 0.0, 2.0
SLIDER_MIN, SLIDER_MAX = -30.0, 30.0


class LoraAdvancedDialog(BaseSimpleDialog):
    GRANULAR_SLIDER_MIN_HEIGHT = 200

    def __init__(
        self,
        dialog_key: str,
        config: dict,
        temp_path: str = "tmp/",
        mask_width: int = 640,
        mask_height: int = 352,
    ):
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

        # Mask settings
        self.spatial_mask_enabled = config.get("spatial_mask_enabled", False)
        self.spatial_mask_path = config.get("spatial_mask_path", "")
        self.temporal_mask_enabled = config.get("temporal_mask_enabled", False)
        self.temporal_start_frame = config.get("temporal_start_frame", 0)
        self.temporal_end_frame = config.get("temporal_end_frame", -1)
        self.temporal_fade_in_frames = config.get("temporal_fade_in_frames", 0)
        self.temporal_fade_out_frames = config.get("temporal_fade_out_frames", 0)

        self.temp_path = temp_path
        self.mask_width = mask_width
        self.mask_height = mask_height

        if not self.granular_weights:
            self.granular_weights = get_default_granular_weights(1)

        self.low_range = SLIDER_MIN if self.is_slider else NORMAL_MIN
        self.high_range = SLIDER_MAX if self.is_slider else NORMAL_MAX

        self.layer_sliders: dict[str, QLabeledDoubleSlider] = {}
        self.layer_linked_buttons: dict[str, LinkedButton] = {}
        self.layer_previous_values: dict[str, float] = {}
        self._updating_linked_sliders = False
        self._saving_mask = False

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

        # --- Masking section (below granular layers) ---
        self._build_mask_section()

        self.setLayout(self.dialog_layout)

    # --- Mask section ---

    def _build_mask_section(self):
        mask_frame = QFrame()
        mask_frame.setFrameShape(QFrame.Shape.StyledPanel)
        mask_layout = QVBoxLayout()
        mask_layout.setSpacing(4)

        # --- Spatial mask ---
        spatial_header = QHBoxLayout()

        self.spatial_checkbox = QCheckBox("Spatial mask")
        self.spatial_checkbox.setChecked(self.spatial_mask_enabled)
        self.spatial_checkbox.toggled.connect(self._on_spatial_toggled)
        spatial_header.addWidget(self.spatial_checkbox)

        info_label = QLabel("Paint where the LoRA should apply (uses alpha channel)")
        info_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        spatial_header.addWidget(info_label)
        spatial_header.addStretch()

        mask_layout.addLayout(spatial_header)

        # Mask editor frame (visible when spatial mask enabled)
        self.mask_editor_frame = QFrame()
        mask_editor_layout = QVBoxLayout()
        mask_editor_layout.setContentsMargins(0, 0, 0, 0)
        mask_editor_layout.setSpacing(4)

        # Brush controls
        brush_layout = QHBoxLayout()
        brush_layout.setSpacing(8)

        brush_layout.addWidget(QLabel("Brush size:"))
        self.brush_size_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(3, 300)
        self.brush_size_slider.setValue(150)
        self.brush_size_slider.setMaximumWidth(150)
        brush_layout.addWidget(self.brush_size_slider)

        brush_layout.addWidget(QLabel("Hardness:"))
        self.brush_hardness_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_hardness_slider.setRange(0.0, 0.99)
        self.brush_hardness_slider.setValue(0.0)
        self.brush_hardness_slider.setMaximumWidth(120)
        brush_layout.addWidget(self.brush_hardness_slider)

        brush_layout.addWidget(QLabel("Steps:"))
        self.brush_steps_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_steps_slider.setRange(0.01, 10.00)
        self.brush_steps_slider.setValue(1.25)
        self.brush_steps_slider.setMaximumWidth(120)
        brush_layout.addWidget(self.brush_steps_slider)

        self.brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(self.brush_erase_button)

        self.color_button = ColorButton("Color:")
        brush_layout.addWidget(self.color_button, 0)

        brush_layout.addStretch()
        mask_editor_layout.addLayout(brush_layout)

        # Editor + layer manager side by side
        editor_row = QHBoxLayout()

        # Layer manager (left side, for switching between reference and mask layers)
        self.mask_layer_manager = LayerManagerWidget(start_expanded=True, expandable=False, use_for_mask=True)
        self.mask_layer_manager.setMaximumWidth(200)
        editor_row.addWidget(self.mask_layer_manager)

        # Image editor canvas
        self.mask_image_widget = ImageWidget(
            "Spatial Mask",
            "lora_mask",
            self.mask_width,
            self.mask_height,
            temp_path=self.temp_path,
        )
        self.mask_image_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        editor_row.addWidget(self.mask_image_widget, 4)

        mask_editor_layout.addLayout(editor_row)

        editor = self.mask_image_widget.image_editor
        editor.set_enable_copy(False)
        editor.set_enable_save(False)

        # Connect brush controls to editor
        self.brush_size_slider.valueChanged.connect(editor.set_brush_size)
        self.brush_size_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_hardness_slider.valueChanged.connect(editor.set_brush_hardness)
        self.brush_hardness_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_steps_slider.valueChanged.connect(editor.set_brush_steps)
        self.color_button.color_changed.connect(editor.set_brush_color)
        self.brush_erase_button.brush_selected.connect(self.mask_image_widget.set_erase_mode)

        # Connect layer manager to editor
        self.mask_layer_manager.layer_selected.connect(self._on_mask_layer_selected)
        self.mask_layer_manager.image_changed.connect(lambda: None)

        # Set initial brush color to black — the mask uses the alpha channel
        # (any painted pixel = alpha 255 = mask 1.0 = LoRA applies),
        # so color doesn't affect the mask value but black is the visual convention
        self.color_button.set_color((0, 0, 0))

        # Create two layers: reference image (background) + mask (foreground)
        self.reference_layer = editor.add_layer()
        self.reference_layer.layer_name = "Reference Image"
        self.mask_layer_manager.add_layer(self.reference_layer)

        self.mask_layer = editor.add_layer()
        self.mask_layer.layer_name = "Mask"
        self.mask_layer.set_opacity(1.0)
        self.mask_layer_manager.add_layer(self.mask_layer)

        # Select the mask layer for painting by default
        editor.selected_layer = self.mask_layer

        # Load existing mask if available
        if self.spatial_mask_path and os.path.exists(self.spatial_mask_path):
            prev_layer = editor.selected_layer
            editor.selected_layer = self.mask_layer
            editor.change_layer_image(self.spatial_mask_path)
            editor.selected_layer = prev_layer

        # Action buttons
        action_layout = QHBoxLayout()

        save_mask_button = QPushButton("Save mask")
        save_mask_button.setObjectName("green_button")
        save_mask_button.clicked.connect(self._on_save_mask)
        action_layout.addWidget(save_mask_button)

        clear_mask_button = QPushButton("Clear mask")
        clear_mask_button.clicked.connect(self._on_clear_mask)
        action_layout.addWidget(clear_mask_button)

        delete_mask_button = QPushButton("Delete mask")
        delete_mask_button.setObjectName("red_button")
        delete_mask_button.clicked.connect(self._on_delete_mask)
        action_layout.addWidget(delete_mask_button)

        action_layout.addStretch()
        mask_editor_layout.addLayout(action_layout)

        self.mask_editor_frame.setLayout(mask_editor_layout)
        self.mask_editor_frame.setVisible(self.spatial_mask_enabled)
        mask_layout.addWidget(self.mask_editor_frame)

        # --- Temporal mask ---
        temporal_layout = QHBoxLayout()

        self.temporal_checkbox = QCheckBox("Temporal mask")
        self.temporal_checkbox.setChecked(self.temporal_mask_enabled)
        self.temporal_checkbox.toggled.connect(self._on_temporal_toggled)
        temporal_layout.addWidget(self.temporal_checkbox)

        temporal_layout.addWidget(QLabel("Start:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 9999)
        self.start_frame_spin.setValue(self.temporal_start_frame)
        self.start_frame_spin.setEnabled(self.temporal_mask_enabled)
        self.start_frame_spin.valueChanged.connect(self._on_temporal_param_changed)
        temporal_layout.addWidget(self.start_frame_spin)

        temporal_layout.addWidget(QLabel("End:"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(-1, 9999)
        self.end_frame_spin.setSpecialValueText("last")
        self.end_frame_spin.setValue(self.temporal_end_frame)
        self.end_frame_spin.setEnabled(self.temporal_mask_enabled)
        self.end_frame_spin.valueChanged.connect(self._on_temporal_param_changed)
        temporal_layout.addWidget(self.end_frame_spin)

        temporal_layout.addWidget(QLabel("Fade in:"))
        self.fade_in_spin = QSpinBox()
        self.fade_in_spin.setRange(0, 999)
        self.fade_in_spin.setValue(self.temporal_fade_in_frames)
        self.fade_in_spin.setEnabled(self.temporal_mask_enabled)
        self.fade_in_spin.valueChanged.connect(self._on_temporal_param_changed)
        temporal_layout.addWidget(self.fade_in_spin)

        temporal_layout.addWidget(QLabel("Fade out:"))
        self.fade_out_spin = QSpinBox()
        self.fade_out_spin.setRange(0, 999)
        self.fade_out_spin.setValue(self.temporal_fade_out_frames)
        self.fade_out_spin.setEnabled(self.temporal_mask_enabled)
        self.fade_out_spin.valueChanged.connect(self._on_temporal_param_changed)
        temporal_layout.addWidget(self.fade_out_spin)

        mask_layout.addLayout(temporal_layout)

        mask_frame.setLayout(mask_layout)
        self.main_layout.addWidget(mask_frame)

    def _on_mask_layer_selected(self, layer):
        self.mask_image_widget.image_editor.selected_layer = layer

    def _on_spatial_toggled(self, checked: bool):
        self.spatial_mask_enabled = checked
        self.mask_editor_frame.setVisible(checked)
        if checked:
            self.setMinimumHeight(800)
        else:
            self.setMinimumHeight(450)
        self._publish_update()

    def _on_save_mask(self):
        """Save only the mask layer's pixmap (not the reference image)."""
        if self._saving_mask:
            return
        self._saving_mask = True

        # Get the mask layer's pixmap directly (excludes reference image)
        pixmap = self.mask_layer.pixmap_item.pixmap()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        lora_safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.lora_name)
        mask_path = os.path.join(self.temp_path, f"lora_mask_{lora_safe}_{timestamp}.png")
        pixmap.save(mask_path)

        # Remove old mask file if it was in temp
        if self.spatial_mask_path and self.temp_path in self.spatial_mask_path:
            if os.path.isfile(self.spatial_mask_path) and self.spatial_mask_path != mask_path:
                os.remove(self.spatial_mask_path)

        self.spatial_mask_path = mask_path
        self._saving_mask = False
        self._publish_update()

    def _on_clear_mask(self):
        """Clear the mask layer canvas (keep reference image)."""
        pixmap = QPixmap(self.mask_width, self.mask_height)
        pixmap.fill(Qt.GlobalColor.transparent)

        editor = self.mask_image_widget.image_editor
        prev_layer = editor.selected_layer
        editor.selected_layer = self.mask_layer
        editor.change_layer_image(pixmap)
        editor.selected_layer = prev_layer

    def _on_delete_mask(self):
        """Delete the saved mask file and clear the canvas."""
        if self.spatial_mask_path and self.temp_path in self.spatial_mask_path:
            if os.path.isfile(self.spatial_mask_path):
                os.remove(self.spatial_mask_path)
        self.spatial_mask_path = ""
        self._on_clear_mask()
        self._publish_update()

    def _on_temporal_toggled(self, checked: bool):
        self.temporal_mask_enabled = checked
        self.start_frame_spin.setEnabled(checked)
        self.end_frame_spin.setEnabled(checked)
        self.fade_in_spin.setEnabled(checked)
        self.fade_out_spin.setEnabled(checked)
        self._publish_update()

    def _on_temporal_param_changed(self):
        self.temporal_start_frame = self.start_frame_spin.value()
        self.temporal_end_frame = self.end_frame_spin.value()
        self.temporal_fade_in_frames = self.fade_in_spin.value()
        self.temporal_fade_out_frames = self.fade_out_spin.value()
        self._publish_update()

    # --- Granular layer sliders ---

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
            "spatial_mask_enabled": self.spatial_mask_enabled,
            "spatial_mask_path": self.spatial_mask_path,
            "temporal_mask_enabled": self.temporal_mask_enabled,
            "temporal_start_frame": self.temporal_start_frame,
            "temporal_end_frame": self.temporal_end_frame,
            "temporal_fade_in_frames": self.temporal_fade_in_frames,
            "temporal_fade_out_frames": self.temporal_fade_out_frames,
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
