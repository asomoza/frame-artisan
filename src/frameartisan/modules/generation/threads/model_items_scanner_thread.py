from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.utils.database import Database
from frameartisan.utils.model_utils import calculate_file_hash


if TYPE_CHECKING:
    from frameartisan.app.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


class ModelItemsScannerThread(QThread):
    status_changed = pyqtSignal(str)
    item_scanned = pyqtSignal(ModelItemDataObject, object, bool)
    item_deleted = pyqtSignal(int)
    scan_progress = pyqtSignal(int, int)
    finished_scanning = pyqtSignal()
    stop_requested = False

    def __init__(self, model_directories: tuple, image_dir: str, data_path: str, database_table: str):
        super().__init__()

        self.model_directories = model_directories
        self.image_dir = image_dir
        self.data_path = data_path
        self.database_path = os.path.join(data_path, "app.db")
        self.database_table = database_table

    def stop(self):
        self.stop_requested = True

    def run(self):
        self.database = Database(self.database_path)
        self.stop_requested = False
        self.status_changed.emit("Starting scan...")

        total_items = 0
        items_processed = 0

        files_to_check: list[str] = []

        columns = ["id", "filepath", "deleted"]
        all_items = self.database.select(self.database_table, columns)
        model_items = {item[1]: {"id": item[0], "deleted": item[2]} for item in all_items}

        for directory in self.model_directories:
            if not os.path.exists(directory["path"]):
                logger.error(f"Directory not found: {directory['path']}")
                continue

            file_format = directory.get("format")

            for filepath in os.listdir(directory["path"]):
                # Skip the _components directory used for deduplicated storage
                if filepath == "_components":
                    continue

                full_path = os.path.join(directory["path"], filepath)

                # Single-file format (e.g. safetensors LoRAs)
                if file_format and os.path.isfile(full_path) and filepath.endswith(f".{file_format}"):
                    total_items += 1
                    files_to_check.append(full_path)
                elif os.path.isdir(full_path):
                    total_items += 1
                    files_to_check.append(full_path)

        self.scan_progress.emit(items_processed, total_items)

        # Check for deleted models
        for key in model_items.keys():
            if key not in files_to_check and model_items[key]["deleted"] == 0:
                self.database.update(self.database_table, {"deleted": 1}, {"id": model_items[key]["id"]})
                self.item_deleted.emit(model_items[key]["id"])

        for directory in self.model_directories:
            file_format = directory.get("format")

            for filepath in os.listdir(directory["path"]):
                if self.stop_requested:
                    break

                # Skip the _components directory
                if filepath == "_components":
                    continue

                self.status_changed.emit(f"Scanning {filepath}...")
                image_buffer = None
                replace = False

                full_path = os.path.join(directory["path"], filepath)

                # Single-file format (e.g. safetensors LoRAs)
                if file_format and os.path.isfile(full_path) and filepath.endswith(f".{file_format}"):
                    full_filepath = full_path
                    root_filename = os.path.splitext(filepath)[0]
                    filepath = full_path
                elif os.path.isdir(full_path):
                    full_filepath = self._find_hash_file(full_path)
                    root_filename = filepath
                    filepath = full_path
                else:
                    continue

                if full_filepath is None:
                    logger.warning("No hashable file found in %s, skipping", full_path)
                    items_processed += 1
                    self.scan_progress.emit(items_processed, total_items)
                    continue

                hash = calculate_file_hash(full_filepath)
                columns = ModelItemDataObject.get_column_names()
                existing_item = self.database.select_one(self.database_table, columns=columns, where={"hash": hash})

                # Fallback: match by filepath when hash changed (e.g. legacy import hash).
                if not existing_item:
                    existing_item = self.database.select_one(
                        self.database_table, columns=columns, where={"filepath": filepath}
                    )
                    if existing_item:
                        self.database.update(self.database_table, {"hash": hash}, {"id": existing_item["id"]})

                if existing_item:
                    model_item = ModelItemDataObject(**existing_item)
                    model_item.hash = hash

                    if model_item.deleted == 1:
                        model_item.deleted = 0
                        self.database.update(
                            self.database_table,
                            {"deleted": 0, "filepath": filepath, "hash": hash},
                            {"id": model_item.id},
                        )
                    else:
                        replace = True
                        self.database.update(
                            self.database_table, {"filepath": filepath, "hash": hash}, {"id": model_item.id}
                        )

                    if model_item.thumbnail is not None and len(model_item.thumbnail) > 0:
                        image_path = os.path.join(self.image_dir, f"{hash}.webp")

                        if os.path.exists(image_path):
                            with open(image_path, "rb") as image_file:
                                img_bytes = image_file.read()
                            image_buffer = BytesIO(img_bytes)
                else:
                    model_type = self._detect_model_type(filepath)
                    model_item = ModelItemDataObject(
                        root_filename=root_filename,
                        filepath=filepath,
                        name=(root_filename[:20] + "...") if len(root_filename) > 20 else root_filename,
                        version="1.0",
                        model_type=model_type,
                        hash=hash,
                        deleted=0,
                    )

                    self.database.insert(self.database_table, model_item.to_dict())
                    model_item.id = self.database.last_insert_rowid()

                # Populate component registry
                if model_item.id is not None:
                    self._register_components(model_item.id, filepath)

                self.item_scanned.emit(model_item, image_buffer, replace)

                items_processed += 1
                self.scan_progress.emit(items_processed, total_items)

        self.database.disconnect()
        self.finished_scanning.emit()

    @staticmethod
    def _find_hash_file(model_directory: str) -> str | None:
        """Find the first .safetensors file in the transformer (or unet) subdirectory.

        Tries ``transformer/`` first, then ``unet/`` for legacy layouts.
        Returns the full path, or *None* if nothing is found.
        """
        for subdir in ("transformer", "unet"):
            comp_dir = os.path.join(model_directory, subdir)
            if not os.path.isdir(comp_dir):
                continue
            for fname in sorted(os.listdir(comp_dir)):
                if fname.endswith(".safetensors"):
                    return os.path.join(comp_dir, fname)
        return None

    @staticmethod
    def _detect_model_type(model_path: str) -> int:
        """Detect model type from transformer config.json.

        Returns:
            Model type int (1=LTX2).
        """
        import json

        config_path = os.path.join(model_path, "transformer", "config.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            return 1  # default to LTX-2

        class_name = config.get("_class_name", "")
        if "LTX2" in class_name:
            return 1

        return 1  # unknown transformer, default to LTX-2

    @staticmethod
    def _fix_ltx2_vae_config(registry: ComponentRegistry, component_id: int) -> None:
        """Fix LTX 2.0 VAE configs that have decoder_spatial_padding_mode set to 'reflect' instead of 'zeros'.

        NOTE: Only targets the LTX 2.0 VAE (timestep_conditioning=False). The LTX 2.3 VAE
        has a different config and should NOT be patched by this fix.
        """
        import json

        db = registry._db()
        row = db.fetch_one(
            "SELECT config_json, storage_path, architecture FROM component WHERE id = ?",
            (component_id,),
        )
        if row is None or row[2] != "AutoencoderKLLTX2Video":
            return

        config_json, storage_path, _ = row
        if not config_json:
            return

        try:
            config = json.loads(config_json)
        except Exception:
            return

        # Only fix the LTX 2.0 VAE — the 2.3 VAE has a different config
        if config.get("timestep_conditioning") is not False:
            return

        if config.get("decoder_spatial_padding_mode") != "reflect":
            return

        config["decoder_spatial_padding_mode"] = "zeros"
        updated_json = json.dumps(config, indent=2)

        # Update the database
        db.execute("UPDATE component SET config_json = ? WHERE id = ?", (updated_json, component_id))

        # Update the on-disk config.json
        config_path = os.path.join(storage_path, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "w") as f:
                    f.write(updated_json + "\n")
                logger.info("Fixed LTX2 VAE config: decoder_spatial_padding_mode → zeros (%s)", config_path)
            except Exception as e:
                logger.warning("Could not update on-disk VAE config: %s", e)

    def _register_components(self, model_id: int, model_path: str) -> None:
        """Register component entries for a diffusers model if not already present."""
        try:
            from frameartisan.app.component_registry import COMPONENT_TYPES, ComponentRegistry
            from frameartisan.utils.model_utils import calculate_component_hash

            # Determine diffusers base dir from model path
            diffusers_dir = os.path.dirname(model_path)
            components_base_dir = os.path.join(diffusers_dir, "_components")
            registry = ComponentRegistry(self.database_path, components_base_dir)

            if registry.model_has_components(model_id):
                return

            component_mapping: dict[str, int] = {}
            for comp_type in COMPONENT_TYPES:
                comp_dir = os.path.join(model_path, comp_type)
                if not os.path.isdir(comp_dir):
                    # Component missing locally — check _components/ for an
                    # already-canonical copy (e.g. deduplicated before this
                    # component type was tracked).
                    canonical_type_dir = os.path.join(components_base_dir, comp_type)
                    if os.path.isdir(canonical_type_dir):
                        # Pick the first (usually only) hash directory inside.
                        for hash_dir in os.listdir(canonical_type_dir):
                            candidate = os.path.join(canonical_type_dir, hash_dir)
                            if os.path.isdir(candidate):
                                comp_dir = candidate
                                break
                        else:
                            continue
                    else:
                        continue
                content_hash = calculate_component_hash(comp_dir)
                comp_info = registry.register_component(
                    component_type=comp_type,
                    source_path=comp_dir,
                    content_hash=content_hash,
                )
                component_mapping[comp_type] = comp_info.id

            if component_mapping:
                registry.register_model_components(model_id, component_mapping)
                registry.cleanup_after_registration(model_id, model_path)

                # Fix LTX2 VAE config: decoder_spatial_padding_mode should be "zeros"
                if "vae" in component_mapping:
                    self._fix_ltx2_vae_config(registry, component_mapping["vae"])
        except Exception as e:
            logger.debug("Failed to register components for model %d: %s", model_id, e)
