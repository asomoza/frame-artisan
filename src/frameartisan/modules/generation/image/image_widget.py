from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImageReader, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from frameartisan.layouts.aspect_ratio_layout import AspectRatioLayout
from frameartisan.modules.generation.image.image_control import ImageControl
from frameartisan.modules.generation.image.image_editor import ImageEditor
from frameartisan.modules.generation.image.layer_manager_widget import LayerManagerWidget


if TYPE_CHECKING:
    from frameartisan.modules.generation.image.image_editor_layer import ImageEditorLayer


class ImageWidget(QWidget):
    image_loaded = pyqtSignal(str)
    image_changed = pyqtSignal()
    widget_updated = pyqtSignal()

    def __init__(
        self,
        text: str,
        prefix: str,
        editor_width: int,
        editor_height: int,
        show_layer_manager: bool = False,
        layer_manager_to_right: bool = False,
        save_directory: str = "",
        temp_path: str = "tmp/",
    ):
        super().__init__()

        self.text = text
        self.save_directory = save_directory
        self.temp_path = temp_path

        self.setAcceptDrops(True)

        self.prefix = prefix
        self.editor_width = editor_width
        self.editor_height = editor_height
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.show_layer_manager = show_layer_manager
        self.layer_manager_to_right = layer_manager_to_right

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        top_layout = QHBoxLayout()
        source_text_label = QLabel(self.text)
        top_layout.addWidget(source_text_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.fit_image_button = QPushButton("Fit")
        top_layout.addWidget(self.fit_image_button)
        load_image_button = QPushButton("Load")
        load_image_button.setToolTip("Load an image from your computer.")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        main_layout.addLayout(top_layout)

        middle_layout = QHBoxLayout()
        image_widget = QWidget()
        self.image_editor = ImageEditor(
            self.editor_width,
            self.editor_height,
            self.aspect_ratio,
            temp_path=self.temp_path,
            save_directory=self.save_directory,
        )
        self.image_editor.image_changed.connect(self.on_image_changed)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        self.image_editor.image_moved.connect(self.update_image_position)
        self.image_editor.image_rotated.connect(self.update_image_angle)
        self.image_editor.image_copy.connect(self.on_image_copy)
        self.image_editor.image_pasted.connect(self.on_image_pasted)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        image_widget.setLayout(editor_layout)
        middle_layout.addWidget(image_widget, 4)

        # set layer manager to the corresponding side
        if self.show_layer_manager:
            layers_layout = QVBoxLayout()
            self.layer_manager_widget = LayerManagerWidget(self.layer_manager_to_right)
            self.layer_manager_widget.layer_selected.connect(self.on_layer_selected)
            self.layer_manager_widget.layers_reordered.connect(self.image_editor.edit_all_layers_order)
            self.layer_manager_widget.add_layer_clicked.connect(self.add_layer)
            self.layer_manager_widget.delete_layer_clicked.connect(self.on_delete_layer_clicked)
            self.layer_manager_widget.image_changed.connect(self.on_image_changed)
            layers_layout.addWidget(self.layer_manager_widget)

            if self.layer_manager_to_right:
                middle_layout.addLayout(layers_layout, 0)
            else:
                middle_layout.insertLayout(0, layers_layout, 0)

        main_layout.addLayout(middle_layout)

        image_bottom_layout = QHBoxLayout()
        self.reset_image_button = QPushButton("Reset Layer")
        self.reset_image_button.setToolTip("Reset zoom and position of the image from the last update.")
        image_bottom_layout.addWidget(self.reset_image_button)
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.setToolTip("Reset zoom and position of the viewport.")
        image_bottom_layout.addWidget(self.reset_view_button)
        main_layout.addLayout(image_bottom_layout)

        image_controls_layout = QHBoxLayout()
        self.image_scale_control = ImageControl("Scale: ", 1.0, 3)
        self.image_scale_control.value_changed.connect(self.on_image_scale)
        image_controls_layout.addWidget(self.image_scale_control)
        self.image_x_pos_control = ImageControl("X Pos: ", 0, 0)
        self.image_x_pos_control.value_changed.connect(self.on_set_image_x)
        image_controls_layout.addWidget(self.image_x_pos_control)
        self.image_y_pos_control = ImageControl("Y Pos: ", 0, 0)
        self.image_y_pos_control.value_changed.connect(self.on_set_image_y)
        image_controls_layout.addWidget(self.image_y_pos_control)
        self.image_rotation_control = ImageControl("Rotation: ", 0, 0)
        self.image_rotation_control.value_changed.connect(self.on_rotate_image)
        image_controls_layout.addWidget(self.image_rotation_control)
        main_layout.addLayout(image_controls_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)

        self.setLayout(main_layout)

        self.fit_image_button.clicked.connect(self.image_editor.fit_image)
        self.reset_image_button.clicked.connect(self.image_editor.clear_and_restore)
        self.reset_view_button.clicked.connect(self.image_editor.reset_view)

    def update_image_scale(self, scale: float):
        self.image_scale_control.set_value(scale)
        self.widget_updated.emit()

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)
        self.widget_updated.emit()

    def update_image_angle(self, angle):
        self.image_rotation_control.set_value(angle)
        self.widget_updated.emit()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()

            reader = QImageReader(path)

            if reader.canRead():
                self.image_editor.change_layer_image(path)
                self.image_loaded.emit(path)

    def on_load_image(self):
        selected_path, _ = QFileDialog.getOpenFileName(self, "Load Image", self.save_directory, "Images (*.png *.jpg)")
        if selected_path:
            self.image_editor.change_layer_image(selected_path)

    def on_layer_selected(self, layer: ImageEditorLayer):
        self.image_editor.selected_layer = layer
        self.update_image_position(layer.pixmap_item.x(), layer.pixmap_item.y())
        self.update_image_angle(layer.pixmap_item.rotation())
        self.update_image_scale(layer.pixmap_item.scale())

    def add_layer(self, image_path: str = None, save_temp: bool = False):
        layer = self.image_editor.add_layer(image_path, save_temp=save_temp)
        self.layer_manager_widget.add_layer(layer)

    def restore_layers(self, layers: list[ImageEditorLayer]):
        self.image_editor.restore_layers(layers)
        self.layer_manager_widget.restore_layers(layers)

    def on_delete_layer_clicked(self):
        self.image_editor.delete_layer()
        self.layer_manager_widget.delete_layer()

    def reload_image_layer(self, image_path: str, original_path: str, order: int):
        layer_id = self.image_editor.reload_image_layer(image_path, original_path, order)
        return layer_id

    def on_image_scale(self, value: float):
        self.image_editor.set_image_scale(value)
        self.widget_updated.emit()

    def on_set_image_x(self, value: int):
        self.image_editor.set_image_x(value)
        self.widget_updated.emit()

    def on_set_image_y(self, value: int):
        self.image_editor.set_image_y(value)
        self.widget_updated.emit()

    def on_rotate_image(self, value: float):
        self.image_editor.rotate_image(value)
        self.widget_updated.emit()

    def set_layer_parameters(self, image_layer_id, scale, x, y, angle):
        self.image_editor.selected_layer_id = image_layer_id

        self.image_scale_control.set_value(scale)
        self.image_editor.set_image_scale(scale)
        self.image_x_pos_control.set_value(x)
        self.image_editor.set_image_x(x)
        self.image_y_pos_control.set_value(y)
        self.image_editor.set_image_y(y)
        self.image_rotation_control.set_value(angle)
        self.image_editor.rotate_image(angle)

    def reset_controls(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()

    def clear_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_editor.clear_all()

    def on_image_changed(self):
        self.image_changed.emit()

    def on_image_copy(self):
        pixmap = self.image_editor.get_scene_as_pixmap()
        clipboard = QGuiApplication.clipboard()
        clipboard.setPixmap(pixmap)

    def on_image_pasted(self, pixmap: QPixmap):
        self.image_editor.change_layer_image(pixmap)
        self.image_loaded.emit("")

    def set_enabled(self, enabled: bool):
        self.image_editor.setEnabled(enabled)

        self.fit_image_button.setEnabled(enabled)
        self.reset_image_button.setEnabled(enabled)
        self.reset_view_button.setEnabled(enabled)

        self.image_scale_control.setEnabled(enabled)
        self.image_x_pos_control.setEnabled(enabled)
        self.image_y_pos_control.setEnabled(enabled)
        self.image_rotation_control.setEnabled(enabled)

    def set_erase_mode(self, value: bool):
        self.image_editor.erasing = value
