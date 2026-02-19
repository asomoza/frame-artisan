from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from frameartisan.buttons.linked_button import LinkedButton
from frameartisan.buttons.transparent_button import TransparentButton


if TYPE_CHECKING:
    from frameartisan.modules.generation.image.image_editor_layer import ImageEditorLayer


class LayerWidget(QWidget):
    LINK_IMG = files("frameartisan.theme.icons").joinpath("link.png")
    UNLINK_IMG = files("frameartisan.theme.icons").joinpath("unlink.png")
    VISIBLE_IMG = files("frameartisan.theme.icons").joinpath("visible.png")
    NOT_VISIBLE_IMG = files("frameartisan.theme.icons").joinpath("not_visible.png")

    def __init__(self, layer: ImageEditorLayer):
        super().__init__()

        self.layer = layer
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(3, 0, 0, 0)
        main_layout.setSpacing(0)

        self.visible_button = TransparentButton(self.VISIBLE_IMG, 25, 25)
        self.visible_button.clicked.connect(self.on_visible_clicked)
        main_layout.addWidget(
            self.visible_button, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )

        self.layer_name_label = QLabel(self.layer.layer_name)
        main_layout.addWidget(
            self.layer_name_label, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft
        )

        self.link_button = LinkedButton(True)
        self.link_button.clicked.connect(self.on_link_clicked)
        main_layout.addWidget(self.link_button, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        main_layout.setStretch(1, 1)

        self.setLayout(main_layout)

    def on_visible_clicked(self):
        self.layer.switch_visible()
        self.visible_button.icon = self.VISIBLE_IMG if self.layer.visible else self.NOT_VISIBLE_IMG

    def on_link_clicked(self):
        self.layer.set_linked(self.link_button.linked)
