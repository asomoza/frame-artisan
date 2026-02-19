from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel


class DropLightBox(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; font-size: 24px;")
        self.hide()

    def resizeEvent(self, _event):
        self.resize(self.parentWidget().size())  # type: ignore
