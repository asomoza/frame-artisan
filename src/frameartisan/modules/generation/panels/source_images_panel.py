import uuid

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider, QLabeledSlider

from frameartisan.modules.generation.generation_settings import compute_num_frames
from frameartisan.modules.generation.panels.base_panel import BasePanel


_THUMB_MAX_WIDTH = 120
_THUMB_MAX_HEIGHT = 90


class _ConditionEntry(QWidget):
    """A single visual condition entry with thumbnail, frame slider, strength slider, edit/remove."""

    def __init__(self, condition_id: str, panel: "SourceImagesPanel", max_frame: int):
        super().__init__()
        self.condition_id = condition_id
        self.panel = panel
        self.thumb_path: str | None = None
        self._publishing = False

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.thumb_label)

        # Frame slider
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QLabeledSlider()
        self.frame_slider.setRange(1, max(max_frame, 1))
        self.frame_slider.setValue(1)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setOrientation(Qt.Orientation.Horizontal)
        self.frame_slider.setToolTip("Pixel frame index (1-based)")
        self.frame_slider.valueChanged.connect(self._on_settings_changed)
        frame_layout.addWidget(self.frame_slider)

        self.last_frame_btn = QPushButton("Last")
        self.last_frame_btn.setFixedWidth(40)
        self.last_frame_btn.setCheckable(True)
        self.last_frame_btn.setToolTip("Pin to last frame (-1)")
        self.last_frame_btn.toggled.connect(self._on_last_toggled)
        frame_layout.addWidget(self.last_frame_btn)
        layout.addLayout(frame_layout)

        # Strength slider
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))
        self.strength_slider = QLabeledDoubleSlider()
        self.strength_slider.setRange(0.0, 1.0)
        self.strength_slider.setSingleStep(0.05)
        self.strength_slider.setValue(1.0)
        self.strength_slider.setOrientation(Qt.Orientation.Horizontal)
        self.strength_slider.valueChanged.connect(self._on_settings_changed)
        strength_layout.addWidget(self.strength_slider)
        layout.addLayout(strength_layout)

        # Attention slider (controls how much the model "sees" this condition)
        attention_layout = QHBoxLayout()
        attention_layout.addWidget(QLabel("Attention:"))
        self.attention_slider = QLabeledDoubleSlider()
        self.attention_slider.setRange(0.0, 1.0)
        self.attention_slider.setSingleStep(0.05)
        self.attention_slider.setValue(1.0)
        self.attention_slider.setOrientation(Qt.Orientation.Horizontal)
        self.attention_slider.setToolTip(
            "Controls how strongly the model attends to this condition.\n"
            "Only affects non-first-frame conditions (keyframes).\n"
            "1.0 = full attention, 0.0 = ignored"
        )
        self.attention_slider.valueChanged.connect(self._on_settings_changed)
        attention_layout.addWidget(self.attention_slider)
        layout.addLayout(attention_layout)

        # Method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("Auto", "auto")
        self.method_combo.addItem("Replace", "replace")
        self.method_combo.addItem("Keyframe", "keyframe")
        self.method_combo.setToolTip(
            "Auto: frame 1 uses Replace, other frames use Keyframe.\n"
            "Replace: force exact image at this frame (hard constraint).\n"
            "Keyframe: append as attention reference (soft guidance)."
        )
        self.method_combo.currentIndexChanged.connect(self._on_settings_changed)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._on_edit)
        btn_layout.addWidget(edit_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("red_button")
        remove_btn.clicked.connect(self._on_remove)
        btn_layout.addWidget(remove_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def set_thumb(self, path: str) -> None:
        self.thumb_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                _THUMB_MAX_WIDTH,
                _THUMB_MAX_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.thumb_label.setPixmap(pixmap)

    def set_max_frame(self, max_frame: int) -> None:
        self.frame_slider.setRange(1, max(max_frame, 1))
        if self.last_frame_btn.isChecked():
            blocker = QSignalBlocker(self.frame_slider)
            try:
                self.frame_slider.setValue(max_frame)
            finally:
                del blocker

    @property
    def pixel_frame_index(self) -> int:
        if self.last_frame_btn.isChecked():
            return -1
        return self.frame_slider.value()

    def get_condition_data(self) -> dict:
        return {
            "pixel_frame_index": self.pixel_frame_index,
            "strength": self.strength_slider.value(),
            "attention_scale": self.attention_slider.value(),
            "method": self.method_combo.currentData() or "auto",
        }

    def _on_last_toggled(self, checked: bool) -> None:
        self.frame_slider.setEnabled(not checked)
        if checked:
            blocker = QSignalBlocker(self.frame_slider)
            try:
                self.frame_slider.setValue(self.frame_slider.maximum())
            finally:
                del blocker
        self._on_settings_changed()

    def _on_settings_changed(self) -> None:
        if self._publishing:
            return
        self._publishing = True
        try:
            self.panel.event_bus.publish(
                "visual_condition",
                {
                    "action": "update_settings",
                    "condition_id": self.condition_id,
                    "pixel_frame_index": self.pixel_frame_index,
                    "strength": self.strength_slider.value(),
                    "attention_scale": self.attention_slider.value(),
                    "method": self.method_combo.currentData() or "auto",
                },
            )
        finally:
            self._publishing = False

    def _on_edit(self) -> None:
        self.panel.event_bus.publish(
            "manage_dialog",
            {
                "dialog_type": "source_image",
                "action": "open",
                "condition_id": self.condition_id,
            },
        )

    def _on_remove(self) -> None:
        self.panel.event_bus.publish(
            "visual_condition",
            {"action": "remove", "condition_id": self.condition_id},
        )


class SourceImagesPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self._entries: dict[str, _ConditionEntry] = {}
        self._total_frames = compute_num_frames(self.gen_settings.video_duration, self.gen_settings.frame_rate)

        self.init_ui()

        self.event_bus.subscribe("visual_condition", self.on_visual_condition_event)
        self.event_bus.subscribe("generation_change", self._on_generation_change)
        self.event_bus.subscribe("graph_cleared", self._on_graph_cleared)

    def init_ui(self):
        self.main_layout = QVBoxLayout()

        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.setEnabled(False)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        self.main_layout.addWidget(self.enabled_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        # Total frames info
        self.frames_label = QLabel(self._frames_text())
        self.frames_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.frames_label)

        add_image_button = QPushButton("Add Image")
        add_image_button.clicked.connect(self.open_image_dialog)
        add_image_button.setObjectName("green_button")
        self.main_layout.addWidget(add_image_button)

        # Stretch at the end — entries are inserted before this
        self.main_layout.addStretch()
        self.setLayout(self.main_layout)

    def _frames_text(self) -> str:
        return f"Total frames: {self._total_frames}"

    def _update_total_frames(self, data: dict | None = None) -> None:
        # Use event data directly so we don't depend on gen_settings being
        # updated first (event bus subscription order is not guaranteed).
        duration = self.gen_settings.video_duration
        fps = self.gen_settings.frame_rate
        if data is not None:
            attr = data.get("attr")
            value = data.get("value")
            if attr == "video_duration" and value is not None:
                duration = value
            elif attr == "frame_rate" and value is not None:
                fps = value
        new_total = compute_num_frames(duration, fps)
        if new_total == self._total_frames:
            return
        self._total_frames = new_total
        self.frames_label.setText(self._frames_text())
        for entry in self._entries.values():
            entry.set_max_frame(self._total_frames)

    def on_enabled_changed(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "visual_conditions_enabled", "value": checked})

    def open_image_dialog(self):
        condition_id = str(uuid.uuid4())[:8]
        self.event_bus.publish(
            "manage_dialog",
            {
                "dialog_type": "source_image",
                "action": "open",
                "condition_id": condition_id,
            },
        )

    def _add_condition_entry(
        self,
        condition_id: str,
        thumb_path: str,
        pixel_frame_index: int = 1,
        strength: float = 1.0,
        attention_scale: float = 1.0,
        method: str = "auto",
    ) -> None:
        entry = _ConditionEntry(condition_id, self, self._total_frames)
        entry.set_thumb(thumb_path)

        if pixel_frame_index == -1:
            entry.last_frame_btn.setChecked(True)
        else:
            blocker = QSignalBlocker(entry.frame_slider)
            try:
                entry.frame_slider.setValue(pixel_frame_index)
            finally:
                del blocker

        blocker = QSignalBlocker(entry.strength_slider)
        try:
            entry.strength_slider.setValue(strength)
        finally:
            del blocker

        blocker = QSignalBlocker(entry.attention_slider)
        try:
            entry.attention_slider.setValue(attention_scale)
        finally:
            del blocker

        blocker = QSignalBlocker(entry.method_combo)
        try:
            idx = entry.method_combo.findData(method)
            if idx >= 0:
                entry.method_combo.setCurrentIndex(idx)
        finally:
            del blocker

        self._entries[condition_id] = entry
        # Insert before the trailing stretch
        self.main_layout.insertWidget(self.main_layout.count() - 1, entry)

        self.enabled_checkbox.setEnabled(True)
        if not self.enabled_checkbox.isChecked():
            self.enabled_checkbox.setChecked(True)
        else:
            self.on_enabled_changed(True)

    def _remove_condition_entry(self, condition_id: str) -> None:
        entry = self._entries.pop(condition_id, None)
        if entry is not None:
            self.main_layout.removeWidget(entry)
            entry.setParent(None)
            entry.deleteLater()

        if not self._entries:
            self.enabled_checkbox.setEnabled(False)
            blocker = QSignalBlocker(self.enabled_checkbox)
            try:
                self.enabled_checkbox.setChecked(False)
            finally:
                del blocker

    def _on_graph_cleared(self, _data: dict) -> None:
        for condition_id in list(self._entries):
            self._remove_condition_entry(condition_id)

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_visual_condition_event(self, data: dict):
        action = data.get("action")
        condition_id = data.get("condition_id")

        if action == "add":
            thumb_path = data.get("source_thumb_path", "")
            pixel_frame_index = data.get("pixel_frame_index", 1)
            strength = data.get("strength", 1.0)
            method = data.get("method", "auto")
            self._add_condition_entry(condition_id, thumb_path, pixel_frame_index, strength, method=method)

        elif action == "update":
            entry = self._entries.get(condition_id)
            if entry is not None:
                thumb_path = data.get("source_thumb_path")
                if thumb_path:
                    entry.set_thumb(thumb_path)

        elif action == "remove":
            self._remove_condition_entry(condition_id)

    def _on_generation_change(self, data: dict) -> None:
        attr = data.get("attr")
        if attr in ("video_duration", "frame_rate"):
            self._update_total_frames(data)
