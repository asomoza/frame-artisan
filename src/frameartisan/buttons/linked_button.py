from importlib.resources import files

from frameartisan.buttons.transparent_button import TransparentButton


class LinkedButton(TransparentButton):
    LINK_IMG = files("frameartisan.theme.icons").joinpath("link.png")
    UNLINK_IMG = files("frameartisan.theme.icons").joinpath("unlink.png")

    def __init__(self, linked: bool = False):
        super().__init__(self.LINK_IMG, 25, 25)
        self.clicked.connect(self.on_lock_clicked)
        self.linked = linked
        self.icon = self.LINK_IMG if self.linked else self.UNLINK_IMG

    def on_lock_clicked(self):
        self.linked = not self.linked
        self.icon = self.LINK_IMG if self.linked else self.UNLINK_IMG
