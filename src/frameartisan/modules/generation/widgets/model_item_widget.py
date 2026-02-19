from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.modules.generation.widgets.model_image_widget import ModelImageWidget


class ModelItemWidget(QWidget):
    clicked = pyqtSignal()

    def __init__(
        self,
        model_data: ModelItemDataObject,
        pixmap: QPixmap,
        item_width: int,
        item_height: int,
        type_map: dict[int, str] | None = None,
    ):
        super().__init__()

        self.setFixedSize(item_width, item_height)

        self.model_data = model_data
        self.pixmap = pixmap
        self.type_map = type_map

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.image_widget = ModelImageWidget(self.pixmap, self.model_data, type_map=self.type_map)
        self.image_widget.setFixedSize(QSize(150, 150))
        layout.addWidget(self.image_widget)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def update_model_image(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.image_widget.set_model_image(self.pixmap)
