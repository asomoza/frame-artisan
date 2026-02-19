import os
from datetime import datetime
from typing import Union

import attr
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap

from frameartisan.modules.generation.image.layer_pixmap_item import LayerPixmapItem


@attr.s(auto_attribs=True, slots=True)
class ImageEditorLayer:
    layer_id: int = attr.ib(default=None)
    layer_name: str = attr.ib(default="Background")
    width: int = attr.ib(default=1024)
    height: int = attr.ib(default=1024)
    pixmap_item: LayerPixmapItem = attr.ib(default=None)
    original_path: str = attr.ib(default=None)
    base_path: str = attr.ib(default=None)
    filename: str = attr.ib(default=None)
    image_path: str = attr.ib(default=None)
    visible: bool = attr.ib(default=True)
    locked: bool = attr.ib(default=True)
    order: int = attr.ib(default=0)
    opacity: float = attr.ib(default=1.0)
    brightness: float = attr.ib(default=0.0)
    contrast: float = attr.ib(default=1.0)
    saturation: float = attr.ib(default=1.0)
    scale: float = attr.ib(default=1.0)
    x: float = attr.ib(default=0.0)
    y: float = attr.ib(default=0.0)
    rotation: float = attr.ib(default=0.0)

    def set_pixmap_item(self, image: Union[str, QPixmap], temp_path: str):
        if self.image_path is not None and os.path.isfile(self.image_path) and temp_path in self.image_path:
            os.remove(self.image_path)
            self.image_path = None

        if image is None:
            pixmap = QPixmap(self.width, self.height)
            pixmap.fill(Qt.GlobalColor.transparent)
        else:
            if isinstance(image, str):
                pixmap = QPixmap(image)
                self.original_path = image
            else:
                pixmap = image

            self.width = pixmap.width()
            self.height = pixmap.height()

        # if the base path is set, we use that path to store the image in disk
        if self.base_path is not None and os.path.isdir(self.base_path):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename = f"{timestamp}_{self.layer_id}.png"
            self.image_path = os.path.join(self.base_path, self.filename)
            pixmap.save(self.image_path)

        alpha_pixmap = QPixmap(pixmap.size())
        alpha_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(alpha_pixmap)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        if self.pixmap_item is None:
            self.pixmap_item = LayerPixmapItem(alpha_pixmap)
        else:
            self.pixmap_item.setPixmap(alpha_pixmap)

    def switch_visible(self):
        self.visible = not self.visible
        self.pixmap_item.setVisible(self.visible)

    def set_linked(self, value: bool):
        self.locked = value

    def set_opacity(self, value: bool):
        self.opacity = value
        self.pixmap_item.setOpacity(self.opacity)

    def set_brightness(self, brightness: float):
        self.brightness = brightness
        self.pixmap_item.set_brightness(brightness)

    def set_contrast(self, contrast: float):
        self.contrast = contrast
        self.pixmap_item.set_contrast(contrast)

    def set_saturation(self, saturation: float):
        self.saturation = saturation
        self.pixmap_item.set_saturation(saturation)

    def invert_image(self):
        self.pixmap_item.invert_image()

    def mirror_image(self, horizontal: bool, vertical: bool):
        self.pixmap_item.mirror_image(horizontal, vertical)

    def update_transform_properties(self):
        if self.pixmap_item is not None:
            self.scale = self.pixmap_item.scale()
            self.x = self.pixmap_item.x()
            self.y = self.pixmap_item.y()
            self.rotation = self.pixmap_item.rotation()

    def apply_transform_properties(self):
        if self.pixmap_item is not None:
            self.pixmap_item.setScale(self.scale)
            self.pixmap_item.setPos(self.x, self.y)
            self.pixmap_item.setRotation(self.rotation)
            self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
