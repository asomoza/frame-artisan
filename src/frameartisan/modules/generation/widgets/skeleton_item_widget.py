from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget

from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject


class SkeletonItemWidget(QWidget):
    """Lightweight placeholder that mimics ModelItemWidget layout using only paint.
    No child widgets, no layouts, no animations — just drawn rectangles.
    Holds model_data for later hydration into a real ModelItemWidget.
    """

    def __init__(self, model_data: ModelItemDataObject, item_width: int, item_height: int):
        super().__init__()
        self.setFixedSize(item_width, item_height)
        self.model_data = model_data
        self._name = model_data.name or ""
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background — dark card
        p.fillRect(0, 0, w, h, QColor(40, 40, 44))

        bone = QColor(58, 58, 64)
        bone_light = QColor(50, 50, 56)
        radius = 4.0

        # Image area placeholder
        p.setBrush(bone_light)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(4, 4, w - 8, h - 28), radius, radius)

        # Type badge (top-left)
        p.setBrush(bone)
        p.drawRoundedRect(QRectF(6, 6, 40, 14), 2.0, 2.0)

        # Version badge (top-right)
        p.setBrush(bone)
        p.drawRoundedRect(QRectF(w - 30, 6, 24, 14), 2.0, 2.0)

        # Name bar (bottom)
        name_rect = QRectF(4, h - 22, w - 8, 18)
        p.setBrush(QColor(0, 0, 0, 127))
        p.drawRoundedRect(name_rect, radius, radius)

        # Name text
        p.setPen(QColor(180, 180, 180))
        font = p.font()
        font.setPointSize(7)
        p.setFont(font)
        p.drawText(name_rect, Qt.AlignmentFlag.AlignCenter, self._name)

        p.end()
