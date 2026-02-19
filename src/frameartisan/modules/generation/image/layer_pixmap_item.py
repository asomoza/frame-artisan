import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem


class LayerPixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0):
        super().__init__(pixmap)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self._update_lookup_tables()

    def _update_lookup_tables(self):
        # Lookup table for brightness and contrast adjustment
        self.bc_lookup_table = np.arange(256, dtype=np.float32)

        # Apply brightness
        self.bc_lookup_table = np.clip(self.bc_lookup_table + int(self.brightness * 255), 0, 255)

        # Apply contrast
        factor = (259 * (self.contrast + 255)) / (255 * (259 - self.contrast))
        self.bc_lookup_table = np.clip(factor * (self.bc_lookup_table - 128) + 128, 0, 255).astype(np.uint8)

    def paint(self, painter, option, widget=None):
        # Draw the original pixmap
        super().paint(painter, option, widget)

        if self.brightness != 0.0 or self.contrast != 1.0 or self.saturation != 1.0:
            painter.save()

            image = self.pixmap().toImage()

            bits = image.bits()
            bits.setsize(image.sizeInBytes())
            arr = np.frombuffer(bits, np.uint8).reshape((image.height(), image.width(), 4))

            # Apply brightness and contrast adjustment
            arr[:, :, 0] = self.bc_lookup_table[arr[:, :, 0]]  # Blue
            arr[:, :, 1] = self.bc_lookup_table[arr[:, :, 1]]  # Green
            arr[:, :, 2] = self.bc_lookup_table[arr[:, :, 2]]  # Red

            # Apply saturation adjustment
            if self.saturation != 1.0:
                # Convert to float for calculations
                arr_f = arr[:, :, :3].astype(np.float32) / 255.0

                # Calculate grayscale using luminosity method
                gray = 0.299 * arr_f[:, :, 2] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 0]
                gray = np.expand_dims(gray, axis=2)

                # Adjust saturation
                arr_f = arr_f * self.saturation + gray * (1 - self.saturation)

                # Convert back to uint8
                arr[:, :, :3] = np.clip(arr_f * 255, 0, 255).astype(np.uint8)

            adjusted_image = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_ARGB32)
            adjusted_pixmap = QPixmap.fromImage(adjusted_image)
            painter.drawPixmap(self.boundingRect().topLeft(), adjusted_pixmap)

            painter.restore()

    def set_brightness(self, brightness):
        self.brightness = brightness
        self._update_lookup_tables()
        self.update()

    def set_contrast(self, contrast):
        self.contrast = contrast
        self._update_lookup_tables()
        self.update()

    def set_saturation(self, saturation):
        self.saturation = max(0, saturation)
        self.update()

    def invert_image(self):
        image = self.pixmap().toImage()

        bits = image.bits()
        bits.setsize(image.sizeInBytes())
        arr = np.frombuffer(bits, np.uint8).reshape((image.height(), image.width(), 4))

        # Invert the colors
        arr[:, :, :3] = 255 - arr[:, :, :3]
        inverted_image = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_ARGB32)

        self.setPixmap(QPixmap.fromImage(inverted_image))
        self.update()

    def mirror_image(self, horizontal: bool = True, vertical: bool = False):
        image = self.pixmap().toImage()
        mirrored_image = image.mirrored(horizontal=horizontal, vertical=vertical)
        self.setPixmap(QPixmap.fromImage(mirrored_image))
        self.update()
