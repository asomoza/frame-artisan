import logging
import os
import shutil
from io import BytesIO

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from frameartisan.app.directories import DirectoriesObject
from frameartisan.app.preferences import PreferencesObject
from frameartisan.layouts.flow_layout import FlowLayout
from frameartisan.modules.generation.constants import LORA_MODEL_TYPES, MODEL_TYPES
from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.modules.generation.threads.model_items_image_loader_thread import ModelItemsImageLoaderThread
from frameartisan.modules.generation.threads.model_items_loader_thread import ModelItemsLoaderThread
from frameartisan.modules.generation.threads.model_items_scanner_thread import ModelItemsScannerThread
from frameartisan.modules.generation.widgets.drop_lightbox_widget import DropLightBox
from frameartisan.modules.generation.widgets.item_selector_widget import ItemSelectorWidget
from frameartisan.modules.generation.widgets.model_item_widget import ModelItemWidget
from frameartisan.utils.database import Database
from frameartisan.utils.model_utils import calculate_file_hash


logger = logging.getLogger(__name__)


class ModelItemsView(QWidget):
    model_item_clicked = pyqtSignal(ModelItemWidget)
    finished_loading = pyqtSignal()
    item_imported = pyqtSignal(str)
    error = pyqtSignal(str)
    scanning = False
    loading = False

    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        model_directories: tuple,
        default_pixmap: QPixmap,
        image_dir: str,
        database_table: str,
    ):
        super().__init__()

        self.default_pixmap = default_pixmap
        self.thumb_width = 150
        self.thumb_height = 150

        self.directories = directories
        self.model_directories = model_directories
        self.preferences = preferences
        self.image_dir = image_dir
        self.database_table = database_table
        self._type_map = LORA_MODEL_TYPES if database_table == "lora" else MODEL_TYPES

        self.model_items_loader_thread = None
        self.model_items_image_loader_thread = None
        self.model_items_scanner_thread = None
        self.all_batches_loaded = False

        self.tags = None
        self.hide_nsfw = True

        self.setAcceptDrops(True)

        self.init_ui()

        self.load_items()

    def __del__(self):
        self._shutdown_all_threads()

    def closeEvent(self, event):
        self._shutdown_all_threads()
        super().closeEvent(event)

    def _shutdown_all_threads(self):
        self._shutdown_thread("model_items_loader_thread", request_stop=True)
        self._shutdown_thread("model_items_image_loader_thread", request_stop=True)
        self._shutdown_thread("model_items_scanner_thread", request_stop=True)

    def _shutdown_thread(self, attr_name: str, request_stop: bool = False):
        thread = getattr(self, attr_name, None)
        if thread is None:
            return

        try:
            is_running = thread.isRunning()
        except RuntimeError:
            setattr(self, attr_name, None)
            return

        if is_running:
            if request_stop:
                stop_callable = getattr(thread, "stop", None)
                if callable(stop_callable):
                    stop_callable()
                else:
                    request_interrupt = getattr(thread, "requestInterruption", None)
                    if callable(request_interrupt):
                        request_interrupt()
            try:
                thread.quit()
            except RuntimeError:
                pass
            thread.wait()

        setattr(self, attr_name, None)

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        filters_layout = QHBoxLayout()

        self.model_type_combobox = QComboBox()
        self.model_type_combobox.addItem("All", 0)
        type_map = LORA_MODEL_TYPES if self.database_table == "lora" else MODEL_TYPES
        for model_type, type_name in type_map.items():
            self.model_type_combobox.addItem(type_name, model_type)
        self.model_type_combobox.currentIndexChanged.connect(self.on_filter_changed)
        filters_layout.addWidget(self.model_type_combobox)

        tag_filter_label = QLabel("Tags:")
        filters_layout.addWidget(tag_filter_label)
        self.tags_selector = ItemSelectorWidget()
        self.tags_selector.item_changed.connect(self.on_filter_changed)
        filters_layout.addWidget(self.tags_selector)

        name_filter_label = QLabel("Name:")
        filters_layout.addWidget(name_filter_label)
        self.name_line_edit = QLineEdit()
        self.name_line_edit.textChanged.connect(self.on_filter_changed)
        filters_layout.addWidget(self.name_line_edit)

        self.order_combo_box = QComboBox()
        self.order_combo_box.addItem("Ascending", "asc")
        self.order_combo_box.addItem("Descending", "desc")
        self.order_combo_box.currentIndexChanged.connect(self.change_order_direction)
        filters_layout.addWidget(self.order_combo_box)

        self.scan_button = QPushButton("Scan")
        self.scan_button.setObjectName("scanButton")
        self.scan_button.setProperty("scanning", False)
        self.scan_button.clicked.connect(self.toggle_scan)
        filters_layout.addWidget(self.scan_button)

        import_button = QPushButton("Import")
        import_button.clicked.connect(self.on_import_model)
        filters_layout.addWidget(import_button)

        filters_layout.setStretch(0, 1)
        filters_layout.setStretch(1, 0)
        filters_layout.setStretch(2, 4)
        filters_layout.setStretch(3, 1)
        filters_layout.setStretch(4, 3)
        filters_layout.setStretch(5, 1)
        filters_layout.setStretch(6, 1)
        filters_layout.setStretch(7, 1)

        main_layout.addLayout(filters_layout)

        self.loading_widget = QWidget()
        self.loading_widget.setVisible(False)
        self.loading_layout = QHBoxLayout(self.loading_widget)
        self.loading_progress_bar = QProgressBar()
        self.loading_progress_bar.setMinimum(0)
        self.loading_layout.addWidget(self.loading_progress_bar)
        self.loading_info_label = QLabel("0/0")
        self.loading_layout.addWidget(self.loading_info_label)
        main_layout.addWidget(self.loading_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.flow_widget = QWidget()
        self.flow_widget.setObjectName("flow_widget")
        self.flow_layout = FlowLayout(self.flow_widget)
        scroll_area.setWidget(self.flow_widget)
        main_layout.addWidget(scroll_area)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

    def toggle_scan(self):
        if self.loading:
            self.error.emit("Can't use scan while items are loading")
            return

        if self.scanning:
            self.stop_scan()
        else:
            confirm = QMessageBox.question(
                self,
                "Start scan",
                "Scanning may take a while and use resources. Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.start_scan()

    def start_scan(self):
        self.scanning = True
        self.on_scanning_started()

        self.model_items_scanner_thread = ModelItemsScannerThread(
            self.model_directories, self.image_dir, self.directories.data_path, self.database_table
        )
        self.model_items_scanner_thread.item_scanned.connect(self.add_model_item)
        self.model_items_scanner_thread.item_deleted.connect(self.remove_item_from_layout)
        self.model_items_scanner_thread.scan_progress.connect(self.update_scan_progress)
        self.model_items_scanner_thread.finished_scanning.connect(self.on_scanning_finished)
        self.model_items_scanner_thread.start()

    def stop_scan(self):
        if self.model_items_scanner_thread:
            self.model_items_scanner_thread.stop()

    def on_scanning_started(self):
        self.scan_button.setText("Stop Scan")
        self.scan_button.setProperty("scanning", True)
        self.loading_progress_bar.setValue(0)
        self.loading_info_label.setText("0/0")
        self.loading_widget.setVisible(True)
        QApplication.instance().style().polish(self.scan_button)

    def on_scanning_finished(self):
        self.refresh_tags()
        self.scanning = False
        self._shutdown_thread("model_items_scanner_thread")

        self.scan_button.setText("Scan")
        self.loading_widget.setVisible(False)
        self.scan_button.setProperty("scanning", False)
        QApplication.instance().style().polish(self.scan_button)

    def update_scan_progress(self, current: int, total: int):
        self.loading_progress_bar.setMaximum(total)
        self.loading_progress_bar.setValue(current)
        self.loading_info_label.setText(f"{current}/{total}")

    def load_items(self):
        if not self.loading:
            self.loading = True
            self.flow_layout.clear()
            self.tags = []
            self.all_batches_loaded = False

            if self.model_items_loader_thread is None:
                self.model_items_loader_thread = ModelItemsLoaderThread(
                    self.directories.data_path, self.database_table, 50, self.preferences.hide_nsfw
                )
                self.model_items_loader_thread.batch_loaded.connect(self.add_batch)
                self.model_items_loader_thread.finished_loading.connect(self.on_loading_finished)

            self.model_items_loader_thread.start()

    def add_single_item_from_path(self, filepath: str, filename: str, *, component_mapping: dict | None = None):
        database = Database(os.path.join(self.directories.data_path, "app.db"))
        root_filename, _ = os.path.splitext(filename)
        image_buffer = None

        transformer_path = os.path.join(
            filepath, "transformer", "diffusion_pytorch_model-00001-of-00002.safetensors"
        )
        if os.path.isfile(transformer_path):
            hash = calculate_file_hash(transformer_path)
        else:
            transformer_dir = os.path.join(filepath, "transformer")
            if os.path.isdir(transformer_dir):
                st_files = sorted(f for f in os.listdir(transformer_dir) if f.endswith(".safetensors"))
                if st_files:
                    hash = calculate_file_hash(os.path.join(transformer_dir, st_files[0]))
                else:
                    hash = calculate_file_hash(filepath)
            else:
                hash = calculate_file_hash(filepath)

        columns = ModelItemDataObject.get_column_names()
        existing_item = database.select_one(self.database_table, columns=columns, where={"hash": hash})

        if existing_item:
            model_item = ModelItemDataObject(**existing_item)

            if model_item.deleted == 1:
                model_item.deleted = 0
                database.update(self.database_table, {"deleted": 0}, {"hash": hash})

                if model_item.thumbnail is not None and len(model_item.thumbnail) > 0:
                    image_path = os.path.join(self.image_dir, f"{hash}.webp")

                    if os.path.exists(image_path):
                        with open(image_path, "rb") as image_file:
                            img_bytes = image_file.read()
                        image_buffer = BytesIO(img_bytes)
            else:
                logger.info(f"Model {filepath} already exists, skipping import and deleting file.")
                self.error.emit("Model already exists, skipping import.")
                return
        else:
            model_item = ModelItemDataObject(
                root_filename=root_filename,
                filepath=filepath,
                name=(root_filename[:20] + "...") if len(root_filename) > 20 else root_filename,
                version="1.0",
                model_type=1,
                hash=hash,
                deleted=0,
            )

            try:
                database.insert(self.database_table, model_item.to_dict())
                model_item.id = database.last_insert_rowid()
            except Exception as e:
                logger.error(f"Error inserting model item: {e}")

        # Register component mappings if provided
        if component_mapping and model_item.id:
            try:
                from frameartisan.app.app import get_app_database_path
                from frameartisan.app.component_registry import ComponentRegistry

                db_path = get_app_database_path()
                if db_path:
                    registry = ComponentRegistry(
                        db_path,
                        os.path.join(self.directories.models_diffusers, "_components"),
                    )
                    registry.register_model_components(model_item.id, component_mapping)
                    registry.cleanup_after_registration(model_item.id, filepath)
            except Exception as e:
                logger.error(f"Error registering component mappings: {e}")

        database.disconnect()
        self.add_model_item(model_item, image_buffer)

    def add_batch(self, items: list[ModelItemDataObject]):
        model_items = []

        for item in items:
            model_item = self.add_model_item(item)
            model_items.append(model_item)

        self.load_model_images(model_items)

    def add_model_item(self, model_data: ModelItemDataObject, image_buffer: BytesIO = None, replace: bool = False):
        if image_buffer is not None:
            qimage = QImage.fromData(image_buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
        else:
            pixmap = self.default_pixmap

        model_item = ModelItemWidget(model_data, pixmap, self.thumb_width, self.thumb_height, type_map=self._type_map)
        model_item.clicked.connect(lambda: self.model_item_clicked.emit(model_item))

        if replace:
            self.remove_item_from_layout(model_data.id)

        self.flow_layout.addWidget(model_item)

        if model_data.tags is not None:
            tags = model_data.tags.split(", ")
            self.tags.extend(tags)

        return model_item

    def on_import_model(self):
        dialog = QFileDialog()
        options = QFileDialog.Option.ReadOnly | QFileDialog.Option.HideNameFilterDetails
        dialog.setOptions(options)

        filepath, _ = dialog.getOpenFileName(None, "Select a a model", "", "*.safetensors", options=options)
        self.item_imported.emit(filepath)

    def load_model_images(self, model_items: list):
        if self.model_items_image_loader_thread is None:
            self.model_items_image_loader_thread = ModelItemsImageLoaderThread(self.flow_layout, model_items)
            self.model_items_image_loader_thread.image_loaded.connect(self.update_model_image)
            self.model_items_image_loader_thread.finished_loading.connect(self.on_model_images_finished)
        else:
            self.model_items_image_loader_thread.model_items = model_items

        self.model_items_image_loader_thread.start()

    def on_model_images_finished(self):
        self._shutdown_thread("model_items_image_loader_thread")

        if not self.all_batches_loaded:
            if self.model_items_loader_thread is not None:
                self.model_items_loader_thread.resume()
            return

        self.refresh_tags()
        self.loading = False
        self.finished_loading.emit()

    def update_model_image(self, item_index: int, buffer: BytesIO):
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)

        item = self.flow_layout.itemAt(item_index)
        model_item = item.widget()
        model_item.update_model_image(pixmap)

    def refresh_tags(self):
        tags = list(set(self.tags))
        tags.sort()
        self.tags_selector.add_items(tags)

    def on_loading_finished(self):
        self.all_batches_loaded = True
        self._shutdown_thread("model_items_loader_thread")

        if self.model_items_image_loader_thread is None:
            self.refresh_tags()
            self.loading = False
            self.finished_loading.emit()

    def on_reload_items(self):
        self.tags_selector.clear_selected_items()
        self.name_line_edit.setText("")
        self.order_combo_box.setCurrentIndex(0)
        self.load_items()

    def change_order_direction(self):
        self.flow_layout.sort_direction = self.order_combo_box.currentData()
        self.flow_layout.order_by()

    def contextMenuEvent(self, event):
        pos = self.flow_widget.mapFrom(self, event.pos())
        item = self.flow_layout.itemAtPosition(pos)

        if item is not None:
            context_menu = QMenu(self)
            # delete_action: QAction | None = context_menu.addAction("Delete")
            # delete_action.triggered.connect(lambda: self.on_delete_item(item.widget()))
            context_menu.exec(event.globalPos())

    def on_delete_item(self, widget: ModelItemWidget):
        model_data = widget.model_data

        if model_data is not None:
            try:
                path = model_data.filepath
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                logger.error(f"Error deleting model item: {e}")
                self.error.emit("Model files not found.")

            database = Database(os.path.join(self.directories.data_path, "app.db"))
            database.update(self.database_table, {"deleted": 1}, {"id": model_data.id})
            self.flow_layout.remove_item(widget)
            database.disconnect()

    def on_filter_changed(self):
        tags = None
        if len(self.tags_selector.line_edit.text()) > 0:
            tags = self.tags_selector.line_edit.text().split(", ")
        name = self.name_line_edit.text()
        model_type = self.model_type_combobox.currentData()
        self.flow_layout.set_filters(tags, name, model_type)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()
            self.item_imported.emit(path)

    def remove_item_from_layout(self, item_id: int):
        for i in range(self.flow_layout.count()):
            item = self.flow_layout.itemAt(i)
            if item and isinstance(item.widget(), ModelItemWidget) and item.widget().model_data.id == item_id:
                self.flow_layout.remove_item(item.widget())
                break
