from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from frameartisan.app.event_bus import EventBus
from frameartisan.buttons.remove_button import RemoveButton


class LoraAddedItem(QFrame):
    enabled_changed = pyqtSignal(int, bool)
    remove_clicked = pyqtSignal(int)

    def __init__(
        self,
        lora_id: int,
        lora_hash: str,
        name: str,
        weight: float = 1.0,
        enabled: bool = True,
        video_strength: float = 1.0,
        audio_strength: float = 1.0,
        granular_transformer_weights_enabled: bool = False,
        granular_transformer_weights: dict | None = None,
        is_slider: bool = False,
        apply_to_second_stage: bool = False,
    ):
        super().__init__()

        self.event_bus = EventBus()

        self.lora_id = lora_id
        self.lora_hash = lora_hash
        self.lora_name = name
        self.weight = weight
        self.video_strength = video_strength
        self.audio_strength = audio_strength
        self.granular_transformer_weights_enabled = granular_transformer_weights_enabled
        self.granular_transformer_weights = granular_transformer_weights or {}
        self.is_slider = is_slider
        self.apply_to_second_stage = apply_to_second_stage

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self.enable_checkbox = QCheckBox()
        self.enable_checkbox.setChecked(enabled)
        self.enable_checkbox.toggled.connect(self._on_enabled_toggled)
        top_layout.addWidget(self.enable_checkbox)

        self.name_label = QLabel(name)
        self.name_label.setToolTip(name)
        top_layout.addWidget(self.name_label, 1)

        remove_button = RemoveButton()
        remove_button.setFixedSize(20, 20)
        remove_button.clicked.connect(self._on_remove_clicked)
        top_layout.addWidget(remove_button)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.second_stage_checkbox = QCheckBox("2nd stage")
        self.second_stage_checkbox.setChecked(apply_to_second_stage)
        self.second_stage_checkbox.toggled.connect(self._on_second_stage_toggled)
        bottom_layout.addWidget(self.second_stage_checkbox)

        advanced_button = QPushButton("Advanced")
        advanced_button.setFixedHeight(24)
        advanced_button.clicked.connect(self._on_advanced_clicked)
        bottom_layout.addStretch()
        bottom_layout.addWidget(advanced_button)
        bottom_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def _on_advanced_clicked(self):
        self.event_bus.publish(
            "manage_dialog",
            {
                "dialog_type": "lora_advanced",
                "action": "open",
                "id": self.lora_id,
                "hash": self.lora_hash,
                "config": self.get_config(),
            },
        )

    def _on_enabled_toggled(self, checked: bool):
        self.enabled_changed.emit(self.lora_id, checked)

    def _on_second_stage_toggled(self, checked: bool):
        self.apply_to_second_stage = checked
        self.enabled_changed.emit(self.lora_id, self.enable_checkbox.isChecked())

    def _on_remove_clicked(self):
        self.remove_clicked.emit(self.lora_id)

    def update_from_dialog(self, config: dict):
        self.weight = config.get("weight", self.weight)
        self.video_strength = config.get("video_strength", self.video_strength)
        self.audio_strength = config.get("audio_strength", self.audio_strength)
        self.granular_transformer_weights_enabled = config.get(
            "granular_transformer_weights_enabled", self.granular_transformer_weights_enabled
        )
        self.granular_transformer_weights = config.get(
            "granular_transformer_weights", self.granular_transformer_weights
        )
        self.is_slider = config.get("is_slider", self.is_slider)
        self.apply_to_second_stage = config.get("apply_to_second_stage", self.apply_to_second_stage)
        self.second_stage_checkbox.setChecked(self.apply_to_second_stage)

    def get_config(self) -> dict:
        return {
            "id": self.lora_id,
            "hash": self.lora_hash,
            "name": self.lora_name,
            "weight": self.weight,
            "enabled": self.enable_checkbox.isChecked(),
            "video_strength": self.video_strength,
            "audio_strength": self.audio_strength,
            "granular_transformer_weights_enabled": self.granular_transformer_weights_enabled,
            "granular_transformer_weights": self.granular_transformer_weights,
            "is_slider": self.is_slider,
            "apply_to_second_stage": self.apply_to_second_stage,
        }
