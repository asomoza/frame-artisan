from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import QBuffer, QIODevice, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QComboBox, QGridLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from frameartisan.modules.generation.constants import LORA_MODEL_TYPES
from frameartisan.modules.generation.widgets.custom_text_edit_widget import CustomTextEditWidget

from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.modules.generation.lora.lora_preview_utils import extract_preview_clip
from frameartisan.utils.database import Database


if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject
    from frameartisan.modules.generation.widgets.video_simple_widget import VideoSimpleWidget


logger = logging.getLogger(__name__)

TRIGGERS_CHAR_LIMIT = 1024


class LoraEditWidget(QWidget):
    model_info_saved = pyqtSignal(ModelItemDataObject, object)

    def __init__(
        self,
        directories: DirectoriesObject,
        model_data: ModelItemDataObject,
        pixmap: QPixmap,
        video_viewer: VideoSimpleWidget,
        get_json_graph: Callable[[], str | None] | None = None,
    ):
        super().__init__()

        self.directories = directories
        self.model_data = model_data
        self.pixmap = pixmap
        self.video_viewer = video_viewer
        self.get_json_graph = get_json_graph

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

        self.set_image_button = QPushButton("Set preview from video")
        self.set_image_button.clicked.connect(self.set_model_preview)
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

        tags_label = QLabel("Tags:")
        model_layout.addWidget(tags_label, 2, 0)
        self.tags_edit = QLineEdit(self.model_data.tags)
        model_layout.addWidget(self.tags_edit, 2, 1)

        type_label = QLabel("Type:")
        model_layout.addWidget(type_label, 3, 0)
        self.model_type_combobox = QComboBox()
        for model_type, type_name in LORA_MODEL_TYPES.items():
            self.model_type_combobox.addItem(type_name, model_type)
        if self.model_data.model_type is not None:
            current_name = LORA_MODEL_TYPES.get(self.model_data.model_type)
            if current_name:
                self.model_type_combobox.setCurrentText(current_name)
        model_layout.addWidget(self.model_type_combobox, 3, 1)

        model_layout.setColumnStretch(0, 1)
        model_layout.setColumnStretch(1, 4)
        main_layout.addLayout(model_layout)

        # Triggers editing
        triggers_label = QLabel("Trigger words:")
        main_layout.addWidget(triggers_label)

        self.triggers_edit = CustomTextEditWidget(self)
        if self.model_data.triggers:
            self.triggers_edit.setPlainText(self.model_data.triggers)
        self.triggers_edit.textChanged.connect(self._on_triggers_changed)
        main_layout.addWidget(self.triggers_edit)

        self.triggers_count_label = QLabel(f"0/{TRIGGERS_CHAR_LIMIT}")
        main_layout.addWidget(self.triggers_count_label, alignment=Qt.AlignmentFlag.AlignRight)
        # Initialize counter
        self._on_triggers_changed()

        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("green_button")
        self.save_button.clicked.connect(self.save_lora_info)
        main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)

    def _on_triggers_changed(self):
        count = len(self.triggers_edit.toPlainText())
        self.triggers_count_label.setText(f"{count}/{TRIGGERS_CHAR_LIMIT}")
        if count > TRIGGERS_CHAR_LIMIT:
            self.triggers_count_label.setStyleSheet("color: red;")
        else:
            self.triggers_count_label.setStyleSheet("")

    def set_model_preview(self):
        """Capture static thumbnail + APNG preview + example graph from the video viewer."""
        # Static thumbnail from video sink
        if self.video_viewer._video_widget is not None:
            video_frame = self.video_viewer._video_widget.videoSink().videoFrame()
            qimage = video_frame.toImage()
            if not qimage.isNull():
                self.pixmap = QPixmap.fromImage(qimage)

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

        # Preview clip
        source_path = self.video_viewer.source_path
        if source_path:
            position_ms = self.video_viewer.position_ms
            clip_path = os.path.join(self.directories.data_path, "loras", f"{self.model_data.hash}_preview.mp4")
            extract_preview_clip(source_path, position_ms, clip_path)

        # Example graph
        if self.get_json_graph is not None:
            json_graph = self.get_json_graph()
            if json_graph is not None:
                json_graph = self._patch_source_image_path(json_graph, source_path)
                self.model_data.example = json_graph

    def _patch_source_image_path(self, json_graph: str, video_path: str | None) -> str:
        """Replace temp source image path in the graph JSON with the permanent output path.

        The graph serialises the ImageLoadNode path which may point to a temp file
        that gets cleaned up. Derive the permanent path from the video filename.
        If the permanent file doesn't exist yet but a temp copy does, copy it over
        (with dedup) so the example survives app restarts.
        """
        if not video_path:
            return json_graph

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        permanent_dir = str(self.directories.outputs_source_images)
        permanent_path = os.path.join(permanent_dir, f"{video_stem}.png")

        if not os.path.isfile(permanent_path):
            # Check if the source image exists in temp and copy it over
            temp_img = os.path.join(str(self.directories.temp_path), f"{video_stem}.png")
            if os.path.isfile(temp_img):
                # Dedup: check if an identical image already exists in outputs
                img_hash = hashlib.md5(open(temp_img, "rb").read()).hexdigest()
                existing_match = None
                if os.path.isdir(permanent_dir):
                    for f in os.listdir(permanent_dir):
                        if f.endswith(".png"):
                            fp = os.path.join(permanent_dir, f)
                            if hashlib.md5(open(fp, "rb").read()).hexdigest() == img_hash:
                                existing_match = fp
                                break
                if existing_match:
                    permanent_path = existing_match
                else:
                    os.makedirs(permanent_dir, exist_ok=True)
                    shutil.copy2(temp_img, permanent_path)
                    logger.debug("Copied source image to %s for LoRA example", permanent_path)
            else:
                return json_graph

        try:
            data = json.loads(json_graph)
            for node in data.get("nodes", []):
                if node.get("class") == "ImageLoadNode":
                    state = node.get("state", {})
                    if state.get("path") and state["path"] != permanent_path:
                        state["path"] = permanent_path
                    break
            return json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            return json_graph

    def save_lora_info(self):
        database = Database(os.path.join(self.directories.data_path, "app.db"))

        self.model_data.name = self.name_edit.text()
        self.model_data.version = self.version_edit.text()
        self.model_data.tags = self.tags_edit.text()
        self.model_data.triggers = self.triggers_edit.toPlainText() or None
        self.model_data.model_type = self.model_type_combobox.currentData()

        database.update(
            "lora",
            {
                "name": self.model_data.name,
                "version": self.model_data.version,
                "tags": self.model_data.tags,
                "triggers": self.model_data.triggers,
                "model_type": self.model_data.model_type,
                "example": self.model_data.example,
            },
            {"id": self.model_data.id},
        )

        if self.image_updated:
            image_path = os.path.join(self.directories.data_path, "loras", f"{self.model_data.hash}.webp")
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            self.model_image_label.pixmap().save(buffer, "WEBP")
            image_bytes = buffer.data().data()
            buffer.close()

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            database.update(
                "lora",
                {"thumbnail": image_path},
                {"id": self.model_data.id},
            )

            self.pixmap = self.model_image_label.pixmap()

        database.disconnect()

        self.model_info_saved.emit(self.model_data, self.model_image_label.pixmap())
