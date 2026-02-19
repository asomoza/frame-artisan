from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frameartisan.app.event_bus import EventBus
from frameartisan.layouts.simple_flow_layout import SimpleFlowLayout
from frameartisan.modules.generation.constants import LORA_MODEL_TYPES
from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.modules.generation.widgets.model_item_widget import ModelItemWidget


try:
    from PyQt6.QtMultimedia import QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget

    _QT_MULTIMEDIA_AVAILABLE = True
except Exception:  # pragma: no cover
    QMediaPlayer = None  # type: ignore[assignment]
    QVideoWidget = None  # type: ignore[assignment]
    _QT_MULTIMEDIA_AVAILABLE = False


if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject


class LoraInfoWidget(QWidget):
    model_edit = pyqtSignal(ModelItemDataObject, QPixmap)
    model_deleted = pyqtSignal(ModelItemWidget)

    def __init__(self, model_item: ModelItemWidget, directories: DirectoriesObject):
        super().__init__()

        self.model_item = model_item
        self.pixmap = model_item.pixmap
        self.directories = directories
        self.event_bus = EventBus()

        self._preview_player: QMediaPlayer | None = None
        self._preview_video: QVideoWidget | None = None
        self._static_pixmap: QPixmap | None = None
        self._clip_path: str | None = None

        self.init_ui()
        self.load_info()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(3, 0, 3, 0)

        # Stacked widget: page 0 = static image, page 1 = video preview
        self._stack = QStackedWidget()
        self._stack.setFixedWidth(345)
        self._stack.setMaximumHeight(480)

        self.model_image_label = QLabel()
        self.model_image_label.setFixedWidth(345)
        self.model_image_label.setMaximumHeight(480)
        self.model_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self.model_image_label)  # index 0

        if _QT_MULTIMEDIA_AVAILABLE:
            self._preview_video = QVideoWidget()
            self._preview_video.setFixedWidth(345)
            self._preview_video.setMaximumHeight(480)
            self._stack.addWidget(self._preview_video)  # index 1

        self._stack.setCurrentIndex(0)
        main_layout.addWidget(self._stack)

        self.model_name_label = QLabel(self.model_item.model_data.name)
        self.model_name_label.setObjectName("model_name")
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.model_name_label)

        model_type_version_layout = QHBoxLayout()

        self.model_type_label = QLabel()
        model_type_version_layout.addWidget(self.model_type_label)

        model_type_version_layout.addStretch()

        self.version_label = QLabel()
        model_type_version_layout.addWidget(self.version_label)

        main_layout.addLayout(model_type_version_layout)
        main_layout.addSpacing(3)

        self.tags_title_label = QLabel("Tags:")
        self.tags_title_label.setObjectName("tags_title")
        self.tags_label = QLabel()
        self.tags_label.setObjectName("tags")
        main_layout.addWidget(self.tags_title_label)
        main_layout.addWidget(self.tags_label)

        # Trigger words section
        self.trigger_label = QLabel("Trigger words:")
        self.trigger_label.setObjectName("trigger_words")
        main_layout.addWidget(self.trigger_label)

        trigger_words_container = QWidget()
        self.triggers_layout = SimpleFlowLayout()
        self.triggers_layout.setSpacing(4)
        trigger_words_container.setLayout(self.triggers_layout)
        main_layout.addWidget(trigger_words_container)

        main_layout.addStretch()

        buttons_layout = QGridLayout()

        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("red_button")
        self.delete_button.clicked.connect(self.on_delete_clicked)
        buttons_layout.addWidget(self.delete_button, 0, 0)

        self.example_button = QPushButton("Example")
        self.example_button.clicked.connect(self.on_example_clicked)
        self.example_button.setVisible(False)
        buttons_layout.addWidget(self.example_button, 0, 1)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.on_edit_clicked)
        buttons_layout.addWidget(self.edit_button, 1, 0)

        self.add_lora_button = QPushButton("Add LoRA")
        self.add_lora_button.setObjectName("green_button")
        self.add_lora_button.clicked.connect(self.on_add_lora_clicked)
        buttons_layout.addWidget(self.add_lora_button, 1, 1)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def load_info(self):
        self.model_image_label.setPixmap(self.pixmap)
        self._static_pixmap = self.pixmap

        model_type = self.model_item.model_data.model_type
        model_type_string = LORA_MODEL_TYPES.get(model_type, "Unknown") if model_type is not None else "Unknown"
        self.model_type_label.setText(f"Type: {model_type_string}")

        version = self.model_item.model_data.version
        self.version_label.setText(f"v{version}" if version else "")

        self.tags_label.setText(self.model_item.model_data.tags)
        if self.model_item.model_data.tags is None:
            self.tags_title_label.setVisible(False)

        # Triggers
        if self.model_item.model_data.triggers:
            triggers_list = [tag.strip() for tag in self.model_item.model_data.triggers.split(",")]
            for trigger in triggers_list:
                if not trigger:
                    continue
                button = QPushButton(trigger)
                button.setObjectName("trigger_item")
                button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(self._on_trigger_clicked)
                self.triggers_layout.addWidget(button)
        else:
            self.trigger_label.setVisible(False)

        # Example button
        if self.model_item.model_data.example is not None:
            self.example_button.setVisible(True)

        # Preview clip path
        if self.model_item.model_data.hash:
            clip_path = os.path.join(
                self.directories.data_path, "loras", f"{self.model_item.model_data.hash}_preview.mp4"
            )
            if os.path.exists(clip_path):
                self._clip_path = clip_path

    # ------------------------------------------------------------------
    # Hover video preview
    # ------------------------------------------------------------------

    def enterEvent(self, event):
        if self._clip_path is not None and self._preview_video is not None:
            if self._preview_player is None:
                self._preview_player = QMediaPlayer(self)
                self._preview_player.setVideoOutput(self._preview_video)
                self._preview_player.mediaStatusChanged.connect(self._on_media_status_changed)
            self._preview_player.setSource(QUrl.fromLocalFile(self._clip_path))
            self._stack.setCurrentIndex(1)
            self._preview_player.play()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._preview_player is not None:
            self._preview_player.stop()
            self._stack.setCurrentIndex(0)
        super().leaveEvent(event)

    def _on_media_status_changed(self, status):
        """Loop the preview clip."""
        if self._preview_player is not None and status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._preview_player.setPosition(0)
            self._preview_player.play()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_trigger_clicked(self):
        button = self.sender()
        self.event_bus.publish("lora", {"action": "trigger_clicked", "trigger": button.text()})

    def on_example_clicked(self):
        self.event_bus.publish(
            "generate", {"action": "generate_from_json", "json_graph": self.model_item.model_data.example}
        )

    def on_delete_clicked(self):
        confirm = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Are you sure you want to delete '{self.model_item.model_data.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.model_deleted.emit(self.model_item)

    def on_edit_clicked(self):
        self.model_edit.emit(self.model_item.model_data, self.model_item.pixmap)

    def on_add_lora_clicked(self):
        self.event_bus.publish(
            "lora",
            {
                "action": "add",
                "id": self.model_item.model_data.id,
                "name": self.model_item.model_data.name,
                "filepath": self.model_item.model_data.filepath,
                "hash": self.model_item.model_data.hash,
            },
        )
