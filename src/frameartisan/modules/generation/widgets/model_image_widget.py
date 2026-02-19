from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QPoint, QPropertyAnimation, QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from frameartisan.modules.generation.constants import MODEL_TYPES


if TYPE_CHECKING:
    from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject


class ModelImageWidget(QWidget):
    def __init__(self, pixmap: QPixmap, model_data: ModelItemDataObject, *, type_map: dict[int, str] | None = None):
        super().__init__()

        self.model_data = model_data
        self.pixmap = pixmap
        self._type_map = type_map or MODEL_TYPES

        self.init_ui()

        self.type_animation = QPropertyAnimation(self.type_label, b"pos")
        self.version_animation = QPropertyAnimation(self.version_label, b"pos")
        self.name_animation = QPropertyAnimation(self.name_label, b"pos")

        self.setMouseTracking(True)

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.pixmap)

        self.top_widget = QWidget(self)
        top_layout = QHBoxLayout(self.top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        model_type_string = "LTX-2"
        if self.model_data.model_type is not None and self.model_data.model_type in self._type_map:
            model_type_string = self._type_map[self.model_data.model_type]

        self.type_label = QLabel(model_type_string)
        self.type_label.setObjectName("item_type")
        top_layout.addWidget(self.type_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.version_label = QLabel(self.model_data.version)
        self.version_label.setObjectName("item_version")
        top_layout.addWidget(self.version_label, alignment=Qt.AlignmentFlag.AlignRight)
        self.top_widget.raise_()

        self.name_label = QLabel(self.model_data.name, self)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setObjectName("item_name")
        self.name_label.raise_()

        if self.model_data.version is None or len(self.model_data.version) == 0:
            self.version_label.setVisible(False)

        self.top_widget.move(0, 0)
        self.name_label.move(0, self.height() - self.name_label.height())

    def enterEvent(self, _event):
        self.type_animation.stop()
        self.version_animation.stop()
        self.name_animation.stop()

        self.type_animation.setDuration(250)
        self.version_animation.setDuration(250)
        self.name_animation.setDuration(250)

        self.type_animation.setEndValue(
            QPoint(
                -self.type_label.width(),
                0,
            )
        )
        self.version_animation.setEndValue(
            QPoint(
                self.top_widget.width(),
                0,
            )
        )
        self.name_animation.setEndValue(QPoint(0, self.height()))

        self.type_animation.start()
        self.version_animation.start()
        self.name_animation.start()

    def leaveEvent(self, _event):
        self.type_animation.stop()
        self.version_animation.stop()
        self.name_animation.stop()

        self.type_animation.setDuration(250)
        self.version_animation.setDuration(250)
        self.name_animation.setDuration(250)

        self.type_animation.setEndValue(QPoint(0, 0))
        self.version_animation.setEndValue(QPoint(self.top_widget.width() - self.version_label.width(), 0))
        self.name_animation.setEndValue(QPoint(0, self.height() - self.name_label.height()))

        self.type_animation.start()
        self.version_animation.start()
        self.name_animation.start()

    def resizeEvent(self, event):
        self.top_widget.setFixedWidth(self.width())
        self.name_label.setFixedSize(QSize(self.height(), 20))
        self.name_label.move(0, self.height() - self.name_label.height())
        self.image_label.resize(event.size())
        pixmap = self.image_label.pixmap()
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

        if not self.underMouse():
            self.top_widget.move(0, 0)
            self.name_label.move(0, self.height() - self.name_label.height())

    def set_model_version(self, version: str):
        if len(version) > 0:
            self.version_label.setText(version)
            self.version_label.setVisible(True)

    def set_model_type(self, model_type: int):
        self.type_label.setText(self._type_map.get(model_type, "LTX-2"))

    def set_model_image(self, pixmap: QPixmap):
        self.pixmap = pixmap
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)
