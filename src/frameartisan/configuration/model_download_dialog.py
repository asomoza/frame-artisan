import logging
import os

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from frameartisan.app.directories import DirectoriesObject
from frameartisan.app.preferences import PreferencesObject
from frameartisan.configuration.model_download_thread import VARIANT_CONFIG
from frameartisan.utils.database import Database


logger = logging.getLogger(__name__)


# Ordered list of variant keys for UI display.
_VARIANT_ORDER = [
    "normal",
    "distilled",
    "sdnq_4bit",
    "sdnq_8bit",
    "distilled_sdnq_4bit",
    "distilled_sdnq_8bit",
    "latent_upsampler",
]

# Base variants get full model entries; quantized variants become component variants.
_BASE_VARIANTS = {"normal", "distilled", "latent_upsampler"}

# Maps quantized variant → which base model its transformer belongs to.
_TRANSFORMER_PARENT = {
    "sdnq_4bit": "normal",
    "sdnq_8bit": "normal",
    "distilled_sdnq_4bit": "distilled",
    "distilled_sdnq_8bit": "distilled",
}


class ModelDownloadDialog(QDialog):
    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        database: Database,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.directories = directories
        self.preferences = preferences
        self.database = database
        self._download_thread = None
        self._completed_variants: dict[str, str] = {}  # variant_key -> local_path

        self.setWindowTitle("Download Models")
        self.setMinimumWidth(550)

        self._checkboxes: dict[str, QCheckBox] = {}
        self._already_downloaded: set[str] = self._detect_downloaded_variants()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("Download Models")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        if self._already_downloaded:
            subtitle_text = "Select additional models to download from HuggingFace."
        else:
            subtitle_text = "No models found. Select models to download from HuggingFace."
        subtitle = QLabel(subtitle_text)
        subtitle.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        for variant_key in _VARIANT_ORDER:
            cfg = VARIANT_CONFIG[variant_key]
            cb = QCheckBox(cfg["display_name"])
            if variant_key in self._already_downloaded:
                cb.setChecked(True)
                cb.setEnabled(False)
                cb.setText(cfg["display_name"] + "  (downloaded)")
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[variant_key] = cb
            layout.addWidget(cb)

        layout.addSpacing(4)

        info_label = QLabel(
            "Shared components are downloaded once and reused.\n"
            "Quantized variants appear as options in the model info panel."
        )
        info_label.setStyleSheet("color: #9aa0a6; font-style: italic;")
        layout.addWidget(info_label)

        layout.addSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        layout.addStretch()

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self._skip_button = QPushButton("Skip")
        self._skip_button.clicked.connect(self.close)
        buttons_layout.addWidget(self._skip_button)

        self._download_button = QPushButton("Download")
        self._download_button.setEnabled(False)
        self._download_button.clicked.connect(self._on_download_clicked)
        buttons_layout.addWidget(self._download_button)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def _detect_downloaded_variants(self) -> set[str]:
        """Check which variants already exist on disk with a valid transformer."""
        downloaded = set()
        models_dir = self.directories.models_diffusers
        if not models_dir or not os.path.isdir(models_dir):
            return downloaded

        for variant_key, cfg in VARIANT_CONFIG.items():
            variant_path = os.path.join(models_dir, cfg["dir_name"])
            if variant_key in self._STANDALONE_VARIANTS:
                # Standalone models don't have transformer/ — check for config.json
                if os.path.isfile(os.path.join(variant_path, "config.json")):
                    downloaded.add(variant_key)
            elif self._find_hash_file(variant_path) is not None:
                downloaded.add(variant_key)

        return downloaded

    # ------------------------------------------------------------------
    # Checkbox logic
    # ------------------------------------------------------------------

    # Variants that are standalone and don't require the "normal" base model.
    _STANDALONE_VARIANTS = {"latent_upsampler"}

    def _on_checkbox_changed(self):
        any_non_normal = any(
            cb.isChecked()
            for key, cb in self._checkboxes.items()
            if key != "normal" and key not in self._STANDALONE_VARIANTS
        )

        normal_cb = self._checkboxes["normal"]
        if "normal" not in self._already_downloaded:
            if any_non_normal:
                normal_cb.blockSignals(True)
                normal_cb.setChecked(True)
                normal_cb.setEnabled(False)
                normal_cb.blockSignals(False)
            else:
                normal_cb.setEnabled(True)

        # Enable download only if there are new (not already downloaded) variants checked
        any_new_checked = any(
            cb.isChecked() for key, cb in self._checkboxes.items() if key not in self._already_downloaded
        )
        self._download_button.setEnabled(any_new_checked)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _on_download_clicked(self):
        selected = [
            key for key, cb in self._checkboxes.items() if cb.isChecked() and key not in self._already_downloaded
        ]
        if not selected:
            return

        # Disable UI during download
        for cb in self._checkboxes.values():
            cb.setEnabled(False)
        self._download_button.setEnabled(False)
        self._skip_button.setText("Cancel")

        self._progress_bar.setVisible(True)
        self._status_label.setVisible(True)

        from frameartisan.configuration.model_download_thread import ModelDownloadThread

        self._download_thread = ModelDownloadThread(
            models_dir=self.directories.models_diffusers,
            selected_variants=selected,
        )
        self._download_thread.progress.connect(self._on_progress)
        self._download_thread.variant_completed.connect(self._on_variant_completed)
        self._download_thread.download_finished.connect(self._on_download_finished)
        self._download_thread.download_error.connect(self._on_download_error)
        self._download_thread.start()

        # Re-wire skip/cancel to abort
        self._skip_button.clicked.disconnect()
        self._skip_button.clicked.connect(self._on_cancel_clicked)

    def _on_cancel_clicked(self):
        if self._download_thread is not None:
            self._download_thread.request_stop()
            self._status_label.setText("Cancelling...")

    def _on_progress(self, message: str, _current: int, _total: int):
        self._status_label.setText(message)

    def _on_variant_completed(self, variant_key: str, local_path: str):
        self._completed_variants[variant_key] = local_path
        if variant_key in _BASE_VARIANTS:
            self._register_model(variant_key, local_path)

    def _on_download_finished(self):
        self._register_component_variants()

        self._progress_bar.setVisible(False)
        self._status_label.setText("Download complete!")

        self._skip_button.setText("Close")
        self._skip_button.clicked.disconnect()
        self._skip_button.clicked.connect(self.close)
        self._download_button.setVisible(False)

    def _on_download_error(self, message: str):
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Error: {message}")
        logger.error("Download error: %s", message)

        QMessageBox.warning(self, "Download Error", message)

        # Re-enable UI
        self._skip_button.setText("Close")
        self._skip_button.clicked.disconnect()
        self._skip_button.clicked.connect(self.close)

    # ------------------------------------------------------------------
    # Model registration (runs on main thread via signal)
    # ------------------------------------------------------------------

    def _register_model(self, variant_key: str, model_path: str) -> int | None:
        """Register a downloaded model variant in the database, returning the model ID."""
        from frameartisan.app.component_registry import COMPONENT_TYPES, ComponentRegistry
        from frameartisan.configuration.model_download_thread import TE_SHARED_FROM
        from frameartisan.utils.model_utils import calculate_component_hash, calculate_file_hash

        cfg = VARIANT_CONFIG[variant_key]

        # Find hash file (first .safetensors in transformer/)
        hash_file = self._find_hash_file(model_path)
        if hash_file is None:
            logger.warning("No hashable file found in %s", model_path)
            return None

        file_hash = calculate_file_hash(hash_file)

        # Check for existing model with same hash
        existing = self.database.select_one("model", ["id"], {"hash": file_hash})
        if existing:
            return existing["id"]

        self.database.insert(
            "model",
            {
                "root_filename": cfg["dir_name"],
                "filepath": model_path,
                "name": cfg["display_name"],
                "version": cfg.get("version", "1.0"),
                "model_type": cfg["model_type"],
                "hash": file_hash,
                "tags": cfg.get("tags", ""),
                "deleted": 0,
            },
        )
        model_id = self.database.last_insert_rowid()

        # Register components
        diffusers_dir = os.path.dirname(model_path)
        components_base_dir = os.path.join(diffusers_dir, "_components")
        registry = ComponentRegistry(self.database.db_path, components_base_dir)

        # If this variant shares a TE with a sibling, resolve the sibling's
        # model ID so we can copy its text_encoder component directly.
        sibling_components: dict[str, int] = {}
        te_sibling = TE_SHARED_FROM.get(variant_key)
        if te_sibling:
            sibling_path = self._completed_variants.get(te_sibling)
            if sibling_path:
                sibling_hash_file = self._find_hash_file(sibling_path)
                if sibling_hash_file:
                    sibling_hash = calculate_file_hash(sibling_hash_file)
                    sibling_row = self.database.select_one("model", ["id"], {"hash": sibling_hash})
                    if sibling_row:
                        sibling_comps = registry.get_model_components(sibling_row["id"])
                        sibling_components = {ct: info.id for ct, info in sibling_comps.items()}

        component_mapping: dict[str, int] = {}
        for comp_type in COMPONENT_TYPES:
            comp_dir = os.path.join(model_path, comp_type)
            if not os.path.isdir(comp_dir):
                # If the sibling already has this component, use it directly
                if comp_type in sibling_components:
                    component_mapping[comp_type] = sibling_components[comp_type]
                    continue

                # Try _components/ directory
                canonical_type_dir = os.path.join(components_base_dir, comp_type)
                if os.path.isdir(canonical_type_dir):
                    for hash_dir in os.listdir(canonical_type_dir):
                        candidate = os.path.join(canonical_type_dir, hash_dir)
                        if os.path.isdir(candidate):
                            comp_dir = candidate
                            break
                    else:
                        comp_dir = None
                else:
                    comp_dir = None

                # Fall back to an already-registered component of this type in the DB
                if comp_dir is None:
                    existing_comp = self.database.fetch_one(
                        "SELECT id FROM component WHERE component_type = ? LIMIT 1",
                        (comp_type,),
                    )
                    if existing_comp:
                        component_mapping[comp_type] = existing_comp[0]
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

        return model_id

    # ------------------------------------------------------------------
    # Component variant registration
    # ------------------------------------------------------------------

    def _register_component_variants(self):
        """Register quantized downloads as component variants on their base models."""
        from frameartisan.app.component_registry import ComponentRegistry
        from frameartisan.utils.model_utils import calculate_component_hash, calculate_file_hash

        diffusers_dir = self.directories.models_diffusers
        components_base_dir = os.path.join(diffusers_dir, "_components")
        registry = ComponentRegistry(self.database.db_path, components_base_dir)

        # Include both newly downloaded and previously existing variants
        all_available: dict[str, str] = {}
        for variant_key in self._already_downloaded:
            cfg = VARIANT_CONFIG[variant_key]
            all_available[variant_key] = os.path.join(diffusers_dir, cfg["dir_name"])
        all_available.update(self._completed_variants)

        # Resolve base model IDs
        base_model_ids: dict[str, int] = {}
        for base_key in ("normal", "distilled"):
            local_path = all_available.get(base_key)
            if local_path is None:
                continue
            hash_file = self._find_hash_file(local_path)
            if hash_file is None:
                continue
            file_hash = calculate_file_hash(hash_file)
            row = self.database.select_one("model", ["id"], {"hash": file_hash})
            if row:
                base_model_ids[base_key] = row["id"]

        # Register quantized variants
        for variant_key, local_path in all_available.items():
            if variant_key in _BASE_VARIANTS:
                continue

            parent_key = _TRANSFORMER_PARENT.get(variant_key)
            if parent_key is None or parent_key not in base_model_ids:
                continue

            parent_model_id = base_model_ids[parent_key]

            # Register transformer component as variant
            transformer_dir = os.path.join(local_path, "transformer")
            if os.path.isdir(transformer_dir):
                content_hash = calculate_component_hash(transformer_dir)
                comp_info = registry.register_component(
                    component_type="transformer",
                    source_path=transformer_dir,
                    content_hash=content_hash,
                )
                registry.add_component_variant_to_sharing_models(
                    parent_model_id, "transformer", comp_info.id
                )
                logger.info(
                    "Registered transformer variant from %s on model %d",
                    variant_key, parent_model_id,
                )

            # Register text_encoder component as variant on ALL base models
            # (TE is shared across normal and distilled)
            te_dir = os.path.join(local_path, "text_encoder")
            if os.path.isdir(te_dir):
                content_hash = calculate_component_hash(te_dir)
                comp_info = registry.register_component(
                    component_type="text_encoder",
                    source_path=te_dir,
                    content_hash=content_hash,
                )
                for base_key, base_id in base_model_ids.items():
                    registry.add_component_variant_to_sharing_models(
                        base_id, "text_encoder", comp_info.id
                    )
                logger.info(
                    "Registered text_encoder variant from %s on %d base models",
                    variant_key, len(base_model_ids),
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_hash_file(model_directory: str) -> str | None:
        for subdir in ("transformer", "unet"):
            comp_dir = os.path.join(model_directory, subdir)
            if not os.path.isdir(comp_dir):
                continue
            for fname in sorted(os.listdir(comp_dir)):
                if fname.endswith(".safetensors"):
                    return os.path.join(comp_dir, fname)
        return None
