import os
from datetime import datetime

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap


class PixmapSaveThread(QThread):
    error = pyqtSignal(str)
    save_finished = pyqtSignal(str, str)

    def __init__(
        self,
        pixmap: QPixmap,
        prefix: str = "img",
        temp_path: str = "tmp/",
        thumb_width: int = 80,
        thumb_height: int = 80,
    ):
        super().__init__()

        self.prefix = prefix
        self.temp_path = temp_path
        self.pixmap = pixmap
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height

    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        base_name = f"{self.prefix}_{timestamp}"
        image_filename = f"{base_name}.png"
        thumb_filename = f"{base_name}_thumb.png"

        thumb_path = os.path.join(self.temp_path, thumb_filename)
        image_path = os.path.join(self.temp_path, image_filename)

        thumbnail_pixmap = self.pixmap.scaled(self.thumb_width, self.thumb_height)
        thumbnail_pixmap.save(thumb_path)
        self.pixmap.save(image_path)

        self.save_finished.emit(image_path, thumb_path)
