from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import QBuffer, QIODevice, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QComboBox, QGridLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from frameartisan.modules.generation.constants import MODEL_TYPES
from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.utils.database import Database


if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject
    from frameartisan.modules.generation.widgets.video_simple_widget import VideoSimpleWidget


class ModelEditWidget(QWidget):
    model_info_saved = pyqtSignal(ModelItemDataObject, object)

    def __init__(
        self,
        directories: DirectoriesObject,
        model_data: ModelItemDataObject,
        pixmap: QPixmap,
        video_viewer: VideoSimpleWidget,
    ):
        super().__init__()

        self.directories = directories
        self.model_data = model_data
        self.pixmap = pixmap
        self.video_viewer = video_viewer

        self.image_width = 345
        self.image_height = 345

        self.image_updated = False

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(3)
        main_layout.setContentsMargins(3, 0, 3, 0)

        self.model_image_label = QLabel()
        self.model_image_label.setFixedWidth(self.image_width)
        self.model_image_label.setFixedHeight(self.image_height)
        self.model_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.model_image_label.setPixmap(self.pixmap)
        main_layout.addWidget(self.model_image_label)

        self.set_image_button = QPushButton("Set current image")
        self.set_image_button.clicked.connect(self.set_model_image)
        main_layout.addWidget(self.set_image_button)

        model_layout = QGridLayout()

        model_name_label = QLabel("Name: ")
        model_layout.addWidget(model_name_label, 0, 0)
        self.name_edit = QLineEdit(self.model_data.name)
        model_layout.addWidget(self.name_edit, 0, 1)
        model_version_label = QLabel("Version:")
        model_layout.addWidget(model_version_label, 1, 0)
        self.version_edit = QLineEdit(self.model_data.version)
        model_layout.addWidget(self.version_edit, 1, 1)

        type_label = QLabel("Type:")
        model_layout.addWidget(type_label, 2, 0)
        self.model_type_combobox = QComboBox()

        for model_type, type_name in MODEL_TYPES.items():
            self.model_type_combobox.addItem(type_name, model_type)

        if self.model_data.model_type is not None:
            self.model_type_combobox.setCurrentText(MODEL_TYPES[self.model_data.model_type])

        model_layout.addWidget(self.model_type_combobox, 2, 1)

        tags_label = QLabel("Tags:")
        model_layout.addWidget(tags_label, 3, 0)
        self.tags_edit = QLineEdit(self.model_data.tags)
        model_layout.addWidget(self.tags_edit, 3, 1)

        model_layout.setColumnStretch(0, 1)
        model_layout.setColumnStretch(1, 4)
        main_layout.addLayout(model_layout)

        main_layout.addStretch()

        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("green_button")
        self.save_button.clicked.connect(self.save_model_info)
        main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)

    def set_model_image(self):
        if self.video_viewer.pixmap_item is not None:
            self.pixmap = self.video_viewer.pixmap_item.pixmap()

            scaled_pixmap = self.pixmap.scaled(
                self.image_width,
                self.image_height,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )

            x = (scaled_pixmap.width() - self.image_width) // 2
            y = (scaled_pixmap.height() - self.image_height) // 2
            cropped_pixmap = scaled_pixmap.copy(x, y, self.image_width, self.image_height)

            self.model_image_label.setPixmap(cropped_pixmap)
            self.image_updated = True

    def save_model_info(self):
        database = Database(os.path.join(self.directories.data_path, "app.db"))

        self.model_data.name = self.name_edit.text()
        self.model_data.version = self.version_edit.text()
        self.model_data.model_type = self.model_type_combobox.currentData()
        self.model_data.tags = self.tags_edit.text()

        json_graph = getattr(self.video_viewer, "json_graph", None)
        if json_graph is not None:
            self.model_data.example = json_graph

        database.update(
            "model",
            {
                "name": self.model_data.name,
                "version": self.model_data.version,
                "model_type": self.model_data.model_type,
                "tags": self.model_data.tags,
                "example": self.model_data.example,
            },
            {"id": self.model_data.id},
        )

        if self.image_updated:
            image_path = os.path.join(self.directories.data_path, "models", f"{self.model_data.hash}.webp")
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            self.model_image_label.pixmap().save(buffer, "WEBP")
            image_bytes = buffer.data().data()
            buffer.close()

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            database.update(
                "model",
                {"thumbnail": image_path},
                {"id": self.model_data.id},
            )

            self.pixmap = self.model_image_label.pixmap()

        database.disconnect()

        self.model_info_saved.emit(self.model_data, self.model_image_label.pixmap())
