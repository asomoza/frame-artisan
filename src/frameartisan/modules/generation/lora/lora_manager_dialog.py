import logging
import os
import shutil
from importlib.resources import files
from typing import Optional, cast

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout

from frameartisan.app.base_dialog import BaseDialog
from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.modules.generation.dialogs.model_items_view import ModelItemsView
from frameartisan.modules.generation.lora.lora_edit_widget import LoraEditWidget
from frameartisan.modules.generation.lora.lora_info_widget import LoraInfoWidget
from frameartisan.modules.generation.widgets.model_item_widget import ModelItemWidget


logger = logging.getLogger(__name__)


class LoraManagerDialog(BaseDialog):
    MODEL_IMG = str(files("frameartisan.theme.images").joinpath("model.webp"))

    def __init__(self, *args):
        if len(args) <= 3:
            logger.warning("LoraManagerDialog requires the video viewer argument.")
            self.video_viewer = None
            self.get_json_graph = None
            super().__init__(*args)
        elif len(args) == 4:
            self.video_viewer = args[3]
            self.get_json_graph = None
            super().__init__(*args[:3])
        else:
            self.video_viewer = args[3]
            self.get_json_graph = args[4]
            super().__init__(*args[:3], *args[5:])

        self.setWindowTitle("LoRA Manager")
        self.setMinimumSize(1160, self.MAX_MIN_HEIGHT)

        self.loading_models = False
        self.selected_model = None

        self.settings = QSettings("ZCode", "FrameArtisan")
        self.settings.beginGroup("lora_manager_dialog")
        self.load_settings()

        lora_directory = {
            "path": self.directories.models_loras,
            "format": "safetensors",
        }
        self.model_directories = (lora_directory,)
        self.database_table = "lora"

        image_dir = os.path.join(self.directories.data_path, "loras")
        if not os.path.exists(image_dir):
            try:
                os.makedirs(image_dir)
                logger.info("Directory '%s' created successfully.", image_dir)
            except OSError as e:
                logger.error("Error creating directory '%s': %s", image_dir, e)

        self.image_dir = image_dir
        self.default_pixmap = QPixmap(self.MODEL_IMG)

        self.init_ui()

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def init_ui(self):
        content_layout = QHBoxLayout()

        self.model_items_view = ModelItemsView(
            self.directories,
            self.preferences,
            self.model_directories,
            self.default_pixmap,
            self.image_dir,
            self.database_table,
        )
        self.model_items_view.error.connect(self.show_error)
        self.model_items_view.model_item_clicked.connect(self.on_model_item_clicked)
        self.model_items_view.item_imported.connect(self.on_lora_imported)
        self.model_items_view.finished_loading.connect(self.on_finished_loading)
        content_layout.addWidget(self.model_items_view)

        model_frame = QFrame()
        self.model_frame_layout = QVBoxLayout()
        model_frame.setLayout(self.model_frame_layout)
        model_frame.setFixedWidth(350)
        model_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(model_frame)

        self.main_layout.addLayout(content_layout)

    def show_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})

    def on_finished_loading(self):
        self.loading_models = False

    def on_lora_imported(self, path: str):
        if not path.endswith(".safetensors"):
            self.show_error("Only .safetensors LoRA files are supported.")
            return

        filename = os.path.basename(path)
        dest = self._get_unique_path(self.directories.models_loras, filename)

        if self.preferences.delete_lora_on_import:
            shutil.move(path, dest)
        else:
            shutil.copy2(path, dest)

        display_name = os.path.splitext(filename)[0]
        self.model_items_view.add_single_item_from_path(dest, display_name)

    def _get_unique_path(self, target_dir: str, name: str) -> str:
        new_path = os.path.join(target_dir, name)
        if not os.path.exists(new_path):
            return new_path

        base, ext = os.path.splitext(name)
        counter = 1
        while True:
            unique_name = f"{base}_{counter}{ext}"
            new_path = os.path.join(target_dir, unique_name)
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def clear_selected_model(self):
        self.selected_model = None
        for i in reversed(range(self.model_frame_layout.count())):
            widget_to_remove = self.model_frame_layout.itemAt(i).widget()
            self.model_frame_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

    def on_model_item_clicked(self, model_item_widget: ModelItemWidget):
        self.clear_selected_model()
        self.selected_model = model_item_widget

        lora_info_widget = LoraInfoWidget(model_item_widget, self.directories)
        lora_info_widget.model_edit.connect(self.on_edit_clicked)
        lora_info_widget.model_deleted.connect(self.on_lora_deleted)
        self.model_frame_layout.addWidget(lora_info_widget)

    def on_edit_clicked(self, model_data: ModelItemDataObject, pixmap: QPixmap):
        self.clear_selected_model()

        lora_edit_widget = LoraEditWidget(self.directories, model_data, pixmap, self.video_viewer, self.get_json_graph)
        lora_edit_widget.model_info_saved.connect(self.on_info_saved)
        self.model_frame_layout.addWidget(lora_edit_widget)

    def on_info_saved(self, model_data: ModelItemDataObject, pixmap: Optional[QPixmap]):
        model_items = self.model_items_view.flow_layout.items()
        edited_item = None

        for i, model in enumerate(model_items):
            model_item = cast(ModelItemWidget, model.widget())
            if model_item.model_data.id == model_data.id:
                edited_item = self.model_items_view.flow_layout.itemAt(i)
                break

        if edited_item is not None:
            edited_item = cast(ModelItemWidget, edited_item.widget())
            edited_item.model_data = model_data
            edited_item.image_widget.name_label.setText(model_data.name)
            edited_item.image_widget.set_model_version(model_data.version)
            edited_item.image_widget.set_model_type(model_data.model_type)

            if pixmap is not None:
                edited_item.update_model_image(pixmap)

        self.on_model_item_clicked(edited_item)

    def on_lora_deleted(self, model_item_widget: ModelItemWidget):
        self.model_items_view.on_delete_item(model_item_widget)
