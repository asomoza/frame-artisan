from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QDialog, QSizeGrip, QVBoxLayout

from frameartisan.app.directories import DirectoriesObject
from frameartisan.app.event_bus import EventBus
from frameartisan.app.preferences import PreferencesObject


class CustomSizeGrip(QSizeGrip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(5, 5)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0, 0))
        painter.setPen(QPen(QColor(0, 0, 0, 0)))
        painter.drawRect(event.rect())


class BaseDialog(QDialog):
    border_color = QColor("#ff6b6b6b")
    MAX_MIN_HEIGHT = 700

    def __init__(self, dialog_type: str, directories: DirectoriesObject, preferences: PreferencesObject, parent=None):
        super().__init__(parent)

        self.directories = directories
        self.preferences = preferences

        self.event_bus = EventBus()

        self.dialog_type = dialog_type

        self.dialog_layout = QVBoxLayout()
        self.dialog_layout.setContentsMargins(0, 0, 0, 0)
        self.dialog_layout.setSpacing(0)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(1, 3, 0, 0)
        self.main_layout.setSpacing(0)

        self.dialog_layout.addLayout(self.main_layout)

        size_grip = CustomSizeGrip(self)
        self.dialog_layout.addWidget(
            size_grip,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        )

        self.setLayout(self.dialog_layout)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())

    def closeEvent(self, event):
        event.ignore()
        self.event_bus.publish("manage_dialog", {"dialog_type": self.dialog_type, "action": "close"})


class BaseSimpleDialog(QDialog):
    """Lightweight dialog base without directories/preferences dependencies."""

    border_color = QColor("#ff6b6b6b")

    def __init__(self, title: str, minWidth: int = 500, minHeight: int = 500):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(minWidth, minHeight)

        self.dialog_layout = QVBoxLayout()
        self.dialog_layout.setContentsMargins(0, 0, 0, 0)
        self.dialog_layout.setSpacing(0)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(1, 3, 0, 0)
        self.main_layout.setSpacing(0)

        self.dialog_layout.addLayout(self.main_layout)

        size_grip = CustomSizeGrip(self)
        self.dialog_layout.addWidget(
            size_grip,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        )

        self.setLayout(self.dialog_layout)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())
