import copy
import logging
import os

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from superqt import QLabeledDoubleSlider, QLabeledSlider

from frameartisan.app.base_dialog import BaseDialog
from frameartisan.buttons.brush_erase_button import BrushEraseButton
from frameartisan.buttons.color_button import ColorButton
from frameartisan.buttons.eyedropper_button import EyeDropperButton
from frameartisan.modules.generation.image.image_widget import ImageWidget
from frameartisan.modules.generation.threads.pixmap_save_thread import PixmapSaveThread


logger = logging.getLogger(__name__)


class SourceImageDialog(BaseDialog):
    def __init__(
        self,
        dialog_type: str,
        directories,
        preferences,
        source_image_path=None,
        source_image_layers=None,
        image_width=640,
        image_height=352,
    ):
        self.image_width = image_width
        self.image_height = image_height

        self.source_image_path = source_image_path
        self.source_image_layers = source_image_layers

        super().__init__(dialog_type, directories, preferences)

        self.setWindowTitle("Source Image")
        self.setMinimumSize(900, 900)

        self.dialog_busy = False
        self.active_editor = None

        self.init_ui()

        self._connect_editor(self.image_widget.image_editor)

        if self.source_image_layers:
            self.image_widget.restore_layers(self.source_image_layers)
            self.add_button.setText("Update source image")
        else:
            self.image_widget.add_layer(save_temp=True)
            if self.source_image_path is not None:
                self.image_widget.image_editor.change_layer_image(self.source_image_path)
                self.add_button.setText("Update source image")
            else:
                self.add_button.setText("Add source image")

        self.image_widget.set_enabled(True)

    def init_ui(self):
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 0, 10, 0)
        content_layout.setSpacing(10)

        brush_layout = QHBoxLayout()
        brush_layout.setContentsMargins(10, 0, 10, 0)
        brush_layout.setSpacing(10)

        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label)
        self.brush_size_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(3, 300)
        self.brush_size_slider.setValue(20)
        brush_layout.addWidget(self.brush_size_slider)

        brush_hardness_label = QLabel("Brush hardness:")
        brush_layout.addWidget(brush_hardness_label)
        self.brush_hardness_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_hardness_slider.setRange(0.0, 0.99)
        self.brush_hardness_slider.setValue(0.5)
        brush_layout.addWidget(self.brush_hardness_slider)

        brush_steps_label = QLabel("Brush steps:")
        brush_layout.addWidget(brush_steps_label)
        self.brush_steps_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_steps_slider.setRange(0.01, 10.00)
        self.brush_steps_slider.setValue(0.25)
        brush_layout.addWidget(self.brush_steps_slider)

        self.brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(self.brush_erase_button)

        self.color_button = ColorButton("Color:")
        brush_layout.addWidget(self.color_button, 0)

        eyedropper_button = EyeDropperButton(25, 25)
        eyedropper_button.clicked.connect(self.on_eyedropper_clicked)
        brush_layout.addWidget(eyedropper_button, 0)

        content_layout.addLayout(brush_layout)

        self.image_widget = ImageWidget(
            "Source image",
            "src_image",
            self.image_width,
            self.image_height,
            show_layer_manager=True,
            save_directory=str(self.directories.outputs_images) if hasattr(self.directories, "outputs_images") else "",
            temp_path=str(self.directories.temp_path),
        )
        content_layout.addWidget(self.image_widget)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(5, 0, 5, 0)
        self.add_button = QPushButton("Add source image")
        self.add_button.setObjectName("green_button")
        self.add_button.clicked.connect(self.on_source_image_added)
        bottom_layout.addWidget(self.add_button)
        content_layout.addLayout(bottom_layout)

        self.main_layout.addLayout(content_layout)

    def on_eyedropper_clicked(self):
        QApplication.instance().setOverrideCursor(Qt.CursorShape.CrossCursor)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if (
            QApplication.instance().overrideCursor() == Qt.CursorShape.CrossCursor
            and event.type() == QEvent.Type.MouseButtonPress
        ):
            QApplication.instance().restoreOverrideCursor()
            QApplication.instance().removeEventFilter(self)

            global_pos = QCursor.pos()
            screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
            if screen is None:
                return super().eventFilter(obj, event)

            screen_pos = global_pos - screen.geometry().topLeft()

            pixmap = screen.grabWindow(0, screen_pos.x(), screen_pos.y(), 1, 1)
            if pixmap.isNull():
                return True

            color = QColor(pixmap.toImage().pixel(0, 0))
            rgb_color = (color.red(), color.green(), color.blue())
            self.color_button.set_color(rgb_color)
            return True

        return super().eventFilter(obj, event)

    def _connect_editor(self, editor):
        self._disconnect_active_editor()

        self.color_button.color_changed.connect(editor.set_brush_color)
        self.brush_size_slider.valueChanged.connect(editor.set_brush_size)
        self.brush_size_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_hardness_slider.valueChanged.connect(editor.set_brush_hardness)
        self.brush_hardness_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_steps_slider.valueChanged.connect(editor.set_brush_steps)
        self.brush_erase_button.brush_selected.connect(self.image_widget.set_erase_mode)

        self.brush_size_slider.setValue(editor.brush_size)
        self.color_button.set_color(editor.brush_color.getRgb()[:3])
        self.brush_hardness_slider.setValue(editor.hardness)
        self.brush_steps_slider.setValue(editor.steps)

        self.active_editor = editor

    def _disconnect_active_editor(self):
        if self.active_editor is not None:
            try:
                self.color_button.color_changed.disconnect(self.active_editor.set_brush_color)
                self.brush_size_slider.valueChanged.disconnect(self.active_editor.set_brush_size)
                self.brush_size_slider.sliderReleased.disconnect(self.active_editor.hide_brush_preview)
                self.brush_hardness_slider.valueChanged.disconnect(self.active_editor.set_brush_hardness)
                self.brush_hardness_slider.sliderReleased.disconnect(self.active_editor.hide_brush_preview)
                self.brush_steps_slider.valueChanged.disconnect(self.active_editor.set_brush_steps)
                self.brush_erase_button.brush_selected.disconnect(self.image_widget.set_erase_mode)
            except TypeError:
                logger.warning("Tried to disconnect signals that were not connected.")

            self.active_editor = None

    def on_source_image_added(self):
        if self.dialog_busy:
            return

        self.dialog_busy = True

        pixmap = self.image_widget.image_editor.get_scene_as_pixmap()

        if self.source_image_path is not None and self.directories.temp_path in self.source_image_path:
            if os.path.isfile(self.source_image_path):
                os.remove(self.source_image_path)

        self.pixmap_save_thread = PixmapSaveThread(
            pixmap, prefix="source_image", temp_path=str(self.directories.temp_path), thumb_width=150, thumb_height=150
        )

        self.pixmap_save_thread.save_finished.connect(self.on_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.connect(self.on_error)
        self.pixmap_save_thread.start()

    def on_pixmap_saved(self, image_path: str, thumbnail_path: str):
        previous_path = self.source_image_path

        if previous_path == image_path:
            return

        self.source_image_path = image_path

        self.event_bus.publish(
            "source_image",
            {
                "action": "add" if previous_path is None else "update",
                "source_image_path": image_path,
                "source_thumb_path": thumbnail_path,
            },
        )

        # Save layer state
        self.source_image_layers = self.image_widget.image_editor.get_all_layers()
        copied_layers = [copy.copy(layer) for layer in self.source_image_layers]
        for layer in copied_layers:
            layer.pixmap_item = None

        self.event_bus.publish(
            "source_image",
            {
                "action": "update_layers",
                "layers": copied_layers,
            },
        )

        self.add_button.setText("Update source image")

    def on_save_pixmap_thread_finished(self):
        self.pixmap_save_thread.save_finished.disconnect(self.on_pixmap_saved)
        self.pixmap_save_thread.finished.disconnect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.disconnect(self.on_error)
        self.pixmap_save_thread = None

        self.dialog_busy = False

    def closeEvent(self, event):
        self._save_layer_state()
        super().closeEvent(event)

    def _save_layer_state(self):
        """Persist current layer state so the dialog can be restored on reopen."""
        layers = self.image_widget.image_editor.get_all_layers()
        temp_path = str(self.directories.temp_path)

        for layer in layers:
            # Sync transform from scene items
            layer.update_transform_properties()

            # Save current pixmap to disk (captures paint strokes)
            if layer.pixmap_item is not None:
                if layer.image_path is None:
                    from datetime import datetime

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    layer.image_path = os.path.join(temp_path, f"source_img_layer_{timestamp}_{layer.layer_id}.png")
                layer.pixmap_item.pixmap().save(layer.image_path)

        copied_layers = [copy.copy(layer) for layer in layers]
        for layer in copied_layers:
            layer.pixmap_item = None

        self.event_bus.publish(
            "source_image",
            {
                "action": "update_layers",
                "layers": copied_layers,
            },
        )

    def on_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})
