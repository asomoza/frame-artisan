from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


class PanelContainer(QWidget):
    def __init__(self):
        super().__init__()

        self.setObjectName("panel_container")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.init_ui()

    def init_ui(self):
        self.panel_layout = QVBoxLayout()
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(0)
        self.setLayout(self.panel_layout)
