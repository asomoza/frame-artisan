import logging
import os
import shutil
from importlib.resources import files
from typing import Optional, cast

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout

from frameartisan.app.base_dialog import BaseDialog
from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.modules.generation.dialogs.model_items_view import ModelItemsView
from frameartisan.modules.generation.model.model_edit_widget import ModelEditWidget
from frameartisan.modules.generation.model.model_info_widget import ModelInfoWidget
from frameartisan.modules.generation.widgets.model_item_widget import ModelItemWidget


logger = logging.getLogger(__name__)


class ModelManagerDialog(BaseDialog):
    MODEL_IMG = str(files("frameartisan.theme.images").joinpath("model.webp"))

    def __init__(self, *args, target: str = "model"):
        self.model_target = target
        if len(args) <= 3:
            logger.warning("ModelManagerDialog requires the video viewer argument to be able to set videos previews.")
            self.video_viewer = None
            super().__init__(*args)
        else:
            self.video_viewer = args[3]
            super().__init__(*args[:3], *args[4:])

        self.setWindowTitle("Model Manager")
        self.setMinimumSize(1160, 950)

        self.loading_models = False
        self.selected_model = None

        self.settings = QSettings("ZCode", "FrameArtisan")
        self.settings.beginGroup("model_manager_dialog")
        self.load_settings()

        diffusers_directory = {
            "path": self.directories.models_diffusers,
            "format": "diffusers",
        }
        self.model_directories = (diffusers_directory,)
        self.database_table = "model"

        image_dir = os.path.join(self.directories.data_path, "models")

        if not os.path.exists(image_dir):
            try:
                os.makedirs(image_dir)
                logger.info(f"Directory '{image_dir}' created successfully.")
            except OSError as e:
                logger.error(f"Error creating directory '{image_dir}': {e}")

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
        self.model_items_view.item_imported.connect(self.on_model_imported)
        self.model_items_view.finished_loading.connect(self.on_finished_loading_models)
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

    def on_finished_loading_models(self):
        self.loading_models = False

    def on_model_imported(self, path: str):
        if path.endswith(".safetensors"):
            if self._is_transformer_safetensors(path):
                self._import_transformer_only(path)
            else:
                self._import_safetensors_model(path)
        elif os.path.isdir(path) and os.path.exists(os.path.join(path, "model_index.json")):
            self._import_diffusers_model(path)
        elif os.path.isdir(path) and self._is_component_directory(path):
            self._import_component_directory(path)

    def _is_transformer_safetensors(self, path: str) -> bool:
        """Check if a .safetensors file is a standalone transformer."""
        from frameartisan.modules.generation.component_compatibility import detect_transformer_architecture

        architecture = detect_transformer_architecture(path)
        if architecture is not None:
            logger.debug("Detected transformer architecture '%s' in %s", architecture, path)
            return True
        return False

    def _import_transformer_only(self, path: str):
        """Import a standalone transformer .safetensors as a component variant on compatible models."""
        from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel

        from frameartisan.app.app import get_app_database_path
        from frameartisan.app.component_registry import ComponentRegistry
        from frameartisan.modules.generation.component_compatibility import detect_transformer_architecture
        from frameartisan.utils.database import Database
        from frameartisan.utils.model_utils import calculate_component_hash

        db_path = get_app_database_path()
        if db_path is None:
            self.show_error("Database not available for transformer import.")
            return

        components_base_dir = os.path.join(self.directories.models_diffusers, "_components")
        registry = ComponentRegistry(db_path, components_base_dir)

        architecture = detect_transformer_architecture(path)
        if architecture is None:
            self.show_error("Could not detect transformer architecture from safetensors file.")
            return

        compatible_models = registry.get_compatible_model_ids("transformer", architecture)
        if not compatible_models:
            self.show_error(
                "No compatible models found for this transformer. Import a full diffusers model first."
            )
            return

        # Show model picker dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Import transformer")
        layout = QFormLayout(dialog)
        layout.addRow(QLabel(f"Transformer architecture: {architecture}"))

        model_combo = QComboBox()
        for model_id, model_name in compatible_models:
            model_combo.addItem(model_name, model_id)
        layout.addRow("Associate with model:", model_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        target_model_id = model_combo.currentData()
        selected_model_name = model_combo.currentText()

        # Store in _components/transformer/{hash}/
        # First, prepare a temp dir to hash from
        model_name = os.path.splitext(os.path.basename(path))[0]
        tmp_dir = os.path.join(components_base_dir, "transformer", f"_tmp_{model_name}")
        os.makedirs(tmp_dir, exist_ok=True)

        dest_file = os.path.join(tmp_dir, "diffusion_pytorch_model.safetensors")

        needs_conversion = self.needs_key_conversion(path)
        if needs_conversion:
            self.convert_and_save_transformer(path, dest_file)
            if self.preferences.delete_model_on_import:
                os.remove(path)
        else:
            if self.preferences.delete_model_on_import:
                shutil.move(path, dest_file)
            else:
                shutil.copy2(path, dest_file)

        # Copy config.json from an existing compatible transformer
        db = Database(db_path)
        rows = db.fetch_all(
            "SELECT storage_path FROM component WHERE component_type = 'transformer' AND architecture = ?",
            (architecture,),
        )
        for (existing_path,) in rows:
            existing_config = os.path.join(existing_path, "config.json")
            if os.path.isfile(existing_config):
                shutil.copy2(existing_config, os.path.join(tmp_dir, "config.json"))
                break
        db.disconnect()

        content_hash = calculate_component_hash(tmp_dir)

        # Rename temp dir to canonical hash dir
        canonical_dir = os.path.join(components_base_dir, "transformer", content_hash)
        if os.path.exists(canonical_dir):
            shutil.rmtree(tmp_dir)
        else:
            os.rename(tmp_dir, canonical_dir)

        comp_info = registry.register_component(
            component_type="transformer",
            source_path=canonical_dir,
            content_hash=content_hash,
            architecture=architecture,
        )

        # Register as variant on the selected model and all models sharing the same default transformer
        affected_ids = registry.add_component_variant_to_sharing_models(
            target_model_id, "transformer", comp_info.id
        )

        dtype_label = registry._format_dtype_label(comp_info.dtype, comp_info.config_json)
        n_models = len(affected_ids)
        if n_models > 1:
            msg = f"Imported transformer ({dtype_label}) as variant for {n_models} models"
        else:
            msg = f"Imported transformer ({dtype_label}) as variant for {selected_model_name}"
        self.event_bus.publish("show_snackbar", {"action": "show", "message": msg})

    @staticmethod
    def needs_key_conversion(path: str) -> bool:
        """Check if a safetensors file uses original (non-diffusers) key format."""
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                return key.startswith("model.diffusion_model.")
        return False

    @staticmethod
    def convert_and_save_transformer(src_path: str, dest_path: str) -> None:
        """Convert original-format keys to diffusers format and save."""
        from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers
        from safetensors.torch import load_file, save_file

        logger.info("Converting transformer keys from original to diffusers format...")
        checkpoint = load_file(src_path)
        converted = convert_z_image_transformer_checkpoint_to_diffusers(checkpoint)
        save_file(converted, dest_path)
        logger.info("Conversion complete: %s", dest_path)

    def _import_safetensors_model(self, path: str):
        self.show_error(
            "Unsupported single-file format. Import a diffusers model or a transformer-only .safetensors file."
        )

    def _import_diffusers_model(self, path: str):
        from frameartisan.app.app import get_app_database_path
        from frameartisan.app.component_registry import COMPONENT_TYPES, ComponentRegistry
        from frameartisan.utils.model_utils import calculate_component_hash

        model_dir = os.path.basename(path)
        target_dir = self.directories.models_diffusers
        new_path = self.get_unique_path(target_dir, model_dir)

        if self.preferences.delete_model_on_import:
            shutil.move(path, new_path)
        else:
            shutil.copytree(path, new_path)

        # Register components in the registry
        db_path = get_app_database_path()
        component_mapping = None

        if db_path is not None:
            try:
                components_base_dir = os.path.join(self.directories.models_diffusers, "_components")
                registry = ComponentRegistry(db_path, components_base_dir)
                component_mapping = {}

                for comp_type in COMPONENT_TYPES:
                    comp_dir = os.path.join(new_path, comp_type)
                    if not os.path.isdir(comp_dir):
                        continue
                    content_hash = calculate_component_hash(comp_dir)
                    comp_info = registry.register_component(
                        component_type=comp_type,
                        source_path=comp_dir,
                        content_hash=content_hash,
                    )
                    component_mapping[comp_type] = comp_info.id
            except Exception as e:
                logger.error("Failed to register components during import: %s", e)
                component_mapping = None

        self.model_items_view.add_single_item_from_path(
            new_path, os.path.basename(new_path), component_mapping=component_mapping
        )

    @staticmethod
    def _is_component_directory(path: str) -> bool:
        """Check if a directory is a standalone component (has config.json with architecture info)."""
        for config_name in ("config.json", "tokenizer_config.json"):
            config_path = os.path.join(path, config_name)
            if os.path.isfile(config_path):
                return True
        return False

    def _import_component_directory(self, path: str):
        """Import a standalone component directory (e.g. a quantized transformer/ or text_encoder/)."""
        import json

        from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel

        from frameartisan.app.app import get_app_database_path
        from frameartisan.app.component_registry import ComponentRegistry
        from frameartisan.utils.model_utils import calculate_component_hash

        db_path = get_app_database_path()
        if db_path is None:
            self.show_error("Database not available for component import.")
            return

        config_json = None
        architecture = None
        for config_name in ("config.json", "tokenizer_config.json"):
            config_path = os.path.join(path, config_name)
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_json = f.read()
                    cfg = json.loads(config_json)
                    architecture = (
                        cfg.get("_class_name")
                        or (cfg.get("architectures") or [None])[0]
                        or cfg.get("tokenizer_class")
                        or cfg.get("model_type")
                    )
                    break
                except Exception:
                    pass

        if architecture is None:
            self.show_error("Could not detect architecture from component directory.")
            return

        from frameartisan.modules.generation.component_compatibility import ARCHITECTURE_COMPATIBILITY

        component_type = None
        for _trans_arch, compat in ARCHITECTURE_COMPATIBILITY.items():
            for comp_type, arch_list in compat.items():
                if architecture in arch_list:
                    component_type = comp_type
                    break
            if component_type:
                break

        if component_type is None:
            arch_lower = architecture.lower()
            if "transformer" in arch_lower:
                component_type = "transformer"
            elif "autoencoder" in arch_lower or "vae" in arch_lower:
                component_type = "vae"
            elif "tokenizer" in arch_lower:
                component_type = "tokenizer"
            else:
                self.show_error(f"Cannot determine component type for architecture '{architecture}'.")
                return

        components_base_dir = os.path.join(self.directories.models_diffusers, "_components")
        registry = ComponentRegistry(db_path, components_base_dir)

        content_hash = calculate_component_hash(path)
        existing = registry.get_component_by_hash(content_hash)

        if existing is None:
            compatible_models = registry.get_compatible_model_ids(component_type, architecture)
            if not compatible_models:
                self.show_error(
                    f"No compatible models found for this {component_type}. "
                    "Import a full model first."
                )
                return

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Import {component_type}")
            layout = QFormLayout(dialog)

            dtype_preview = registry._format_dtype_label(None, config_json)
            layout.addRow(QLabel(f"Component: {component_type} ({dtype_preview or architecture})"))

            model_combo = QComboBox()
            for model_id, model_name in compatible_models:
                model_combo.addItem(model_name, model_id)
            layout.addRow("Associate with model:", model_combo)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addRow(buttons)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            target_model_id = model_combo.currentData()
            selected_model_name = model_combo.currentText()

            canonical_dir = os.path.join(components_base_dir, component_type, content_hash)
            if not os.path.exists(canonical_dir):
                os.makedirs(os.path.dirname(canonical_dir), exist_ok=True)
                if self.preferences.delete_model_on_import:
                    shutil.move(path, canonical_dir)
                else:
                    shutil.copytree(path, canonical_dir)

            comp_info = registry.register_component(
                component_type=component_type,
                source_path=canonical_dir,
                content_hash=content_hash,
                architecture=architecture,
                config_json=config_json,
            )
        else:
            comp_info = existing
            compatible_models = registry.get_compatible_model_ids(component_type, architecture)
            if not compatible_models:
                self.show_error(f"No compatible models found for this {component_type}.")
                return

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Associate {component_type}")
            layout = QFormLayout(dialog)

            dtype_label = registry._format_dtype_label(comp_info.dtype, comp_info.config_json)
            layout.addRow(QLabel(f"Component: {component_type} ({dtype_label})"))
            layout.addRow(QLabel("Already imported. Associate with another model:"))

            model_combo = QComboBox()
            for model_id, model_name in compatible_models:
                model_combo.addItem(model_name, model_id)
            layout.addRow("Model:", model_combo)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addRow(buttons)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            target_model_id = model_combo.currentData()
            selected_model_name = model_combo.currentText()

        affected_ids = registry.add_component_variant_to_sharing_models(
            target_model_id, component_type, comp_info.id
        )

        dtype_label = registry._format_dtype_label(comp_info.dtype, comp_info.config_json)
        n_models = len(affected_ids)
        if n_models > 1:
            msg = f"Imported {component_type} ({dtype_label}) for {n_models} models"
        else:
            msg = f"Imported {component_type} ({dtype_label}) for {selected_model_name}"
        self.event_bus.publish("show_snackbar", {"action": "show", "message": msg})

    def get_unique_path(self, target_dir: str, name: str) -> str:
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

        self.selected_model = ModelDataObject(
            name=model_item_widget.model_data.name,
            version=model_item_widget.model_data.version,
            filepath=model_item_widget.model_data.filepath,
            model_type=model_item_widget.model_data.model_type,
            id=model_item_widget.model_data.id,
        )

        model_info_widget = ModelInfoWidget(model_item_widget, self.directories, target=self.model_target)
        model_info_widget.model_edit.connect(self.on_model_edit_clicked)
        model_info_widget.model_deleted.connect(self.on_model_deleted)
        self.model_frame_layout.addWidget(model_info_widget)

    def on_model_edit_clicked(self, model_data: ModelItemDataObject, pixmap: QPixmap):
        self.clear_selected_model()

        model_edit_widget = ModelEditWidget(self.directories, model_data, pixmap, self.video_viewer)
        model_edit_widget.model_info_saved.connect(self.on_info_saved)
        self.model_frame_layout.addWidget(model_edit_widget)

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

    def on_model_deleted(self, model_item_widget: ModelItemWidget):
        self.model_items_view.on_delete_item(model_item_widget)
