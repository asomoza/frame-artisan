import gc
import json
import logging
import os
import re
import uuid

import torch
from PyQt6.QtCore import QObject, QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QProgressBar, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget

from frameartisan.app.model_manager import get_model_manager
from frameartisan.configuration.model_download_dialog import ModelDownloadDialog
from frameartisan.modules.base_module import BaseModule
from frameartisan.modules.generation.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    LTX2_LATENT_UPSAMPLER_DIR,
    LTX2_TINY_VAE_DIR,
    LTX2_TINY_VAE_FILENAME,
    MODEL_TYPE_DEFAULTS,
    SECOND_PASS_MODEL_TYPE_DEFAULTS,
)
from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
from frameartisan.modules.generation.generation_settings import GenerationSettings, compute_num_frames
from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph
from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES
from frameartisan.modules.generation.lora.lora_advanced_dialog import LoraAdvancedDialog
from frameartisan.modules.generation.lora.lora_manager_dialog import LoraManagerDialog
from frameartisan.modules.generation.menus.generation_right_menu import GenerationRightMenu
from frameartisan.modules.generation.model.model_manager_dialog import ModelManagerDialog
from frameartisan.modules.generation.source_image.source_image_dialog import SourceImageDialog
from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread
from frameartisan.modules.generation.widgets.prompt_bar_widget import PromptBarWidget
from frameartisan.modules.generation.widgets.video_simple_widget import VideoSimpleWidget
from frameartisan.utils.database import Database
from frameartisan.utils.json_utils import extract_dict_from_json_graph


logger = logging.getLogger(__name__)


class StatusBarLogHandler(logging.Handler):
    """Forwards INFO log records from node loggers to a Qt signal for the status bar."""

    class _Emitter(QObject):
        message = pyqtSignal(str)

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.emitter = self._Emitter()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        # Strip long filesystem paths for cleaner display:
        # "Loading transformer from /home/.../models/ltx2" → "Loading transformer"
        msg = re.sub(r"\s+from\s+\S*[/\\]\S+", "", msg)
        self.emitter.message.emit(msg)


class GenerationModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.settings = QSettings("ZCode", "FrameArtisan")

        self.gen_settings = GenerationSettings.load(self.settings)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        self.node_graph: FrameArtisanNodeGraph | None = None
        self.thread: NodeGraphThread | None = None
        self._last_run_json_graph: str | None = None

        # Stage 1 preview state
        self._stage1_preview_active: bool = False

        self._source_image_path: str | None = None
        self._source_image_layers: list = []

        # Visual conditions (multi-image conditioning)
        self._visual_conditions: dict[str, dict] = {}
        # Each: {"id": str, "image_path": str, "layers": list, "pixel_frame_index": int, "strength": float}

        self._pending_condition_id: str | None = None  # condition_id for dialog in progress

        # Video conditioning
        self._video_condition: dict | None = None
        # {"video_path": str, "node_name": str, "pixel_frame_index": int, "strength": float}

        # Audio conditioning
        self._audio_path: str | None = None
        self._audio_from_video: bool = False

        self._status_log_handler = StatusBarLogHandler()
        self._status_log_handler.emitter.message.connect(self._on_status_log_message)
        logging.getLogger("frameartisan.modules.generation.graph.nodes").addHandler(self._status_log_handler)

        self.init_ui()
        self.build_graph()
        self.subscribe_events()
        self._publish_compile_cache_size()

    def init_ui(self):
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left column: video viewer + prompt bar.
        left_container = QWidget(self)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.video_viewer = VideoSimpleWidget(self.directories, self.preferences)
        self.video_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.video_viewer)

        spacer = QSpacerItem(5, 5, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        left_layout.addSpacerItem(spacer)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(self.gen_settings.num_inference_steps)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.prompt_bar = PromptBarWidget()
        self.prompt_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.prompt_bar)

        left_layout.setStretch(0, 8)
        left_layout.setStretch(2, 0)
        left_layout.setStretch(3, 3)

        root_layout.addWidget(left_container)

        # Right menu: sibling of the whole left column so it spans the full height.
        self.right_menu = GenerationRightMenu(self.gen_settings, self.preferences, self.directories)
        root_layout.addWidget(self.right_menu)
        root_layout.setStretch(0, 1)
        root_layout.setStretch(1, 0)

        self.setLayout(root_layout)

        self.prompt_bar.negative_prompt.setPlainText(DEFAULT_NEGATIVE_PROMPT)
        self.prompt_bar.previous_negative_prompt = DEFAULT_NEGATIVE_PROMPT

        self.prompt_bar.generate_clicked.connect(self.on_generate)
        self.prompt_bar.abort_clicked.connect(self._on_abort_clicked)
        self.prompt_bar.continue_stage2_clicked.connect(self._continue_stage2)
        self.prompt_bar.retry_stage1_clicked.connect(self._retry_stage1)

        self.prompt_bar.generate_button.auto_save = self.preferences.auto_save_videos
        self.prompt_bar.generate_button.auto_save_changed.connect(self._on_auto_save_changed)

        self.video_viewer.set_drop_callback(self._on_video_dropped)

    def build_graph(self) -> None:
        # Set up compile cache env vars early so compilation on first
        # generation can find the persistent cache from previous sessions.
        self._setup_compile_cache()

        # Sync offload strategy to ModelManager
        mm = get_model_manager()
        mm.offload_strategy = self.gen_settings.offload_strategy
        mm.group_offload_use_stream = self.gen_settings.group_offload_use_stream
        mm.group_offload_low_cpu_mem = self.gen_settings.group_offload_low_cpu_mem

        self.node_graph = create_default_ltx2_graph(self.gen_settings, self.directories)

        self.thread = NodeGraphThread(
            directories=self.directories,
            node_graph=self.node_graph,
            dtype=self.dtype,
            device=self.device,
            graph_factory=FrameArtisanNodeGraph,
            node_classes=NODE_CLASSES,
        )
        self.thread.save_video_metadata = self.preferences.save_video_metadata
        self._sync_preview_decode_settings()

        self.thread.generation_finished.connect(self._on_generation_finished)
        self.thread.generation_error.connect(self._on_generation_error)
        self.thread.generation_aborted.connect(self._on_generation_aborted)
        self.thread.progress_update.connect(lambda step, _latents: self.step_progress_update(step + 1))
        self.thread.preview_video_ready.connect(self._on_preview_video_ready)
        self.thread.stage_started.connect(self._on_stage_started)
        self.thread.status_changed.connect(self._on_status_changed)

        # Apply 2nd pass toggle if settings say it's enabled (checkbox restored from session)
        if self.gen_settings.second_pass_enabled:
            self._toggle_second_pass(True)

    def subscribe_events(self) -> None:
        self.event_bus.subscribe("generation_change", self.on_generation_change)
        self.event_bus.subscribe("module", self.on_module_event)
        self.event_bus.subscribe("manage_dialog", self.on_manage_dialog_event)
        self.event_bus.subscribe("model", self.on_model_event)
        self.event_bus.subscribe("lora", self.on_lora_event)
        self.event_bus.subscribe("generate", self.on_generate_event)
        self.event_bus.subscribe("source_image", self.on_source_image_event)
        self.event_bus.subscribe("visual_condition", self.on_visual_condition_event)
        self.event_bus.subscribe("audio_condition", self.on_audio_condition_event)
        self.event_bus.subscribe("video_condition", self.on_video_condition_event)

    def _toggle_second_pass(self, enabled: bool) -> None:
        """Enable or disable 2nd pass (upsample) nodes and rewire decode connections."""
        if self.node_graph is None:
            return

        # Toggle node enabled states
        for name in (
            "upsample",
            "second_pass_model",
            "second_pass_lora",
            "second_pass_latents",
            "second_pass_denoise",
        ):
            node = self.node_graph.get_node_by_name(name)
            if node is not None:
                node.enabled = enabled
                node.set_updated()

        decode_node = self.node_graph.get_node_by_name("decode")
        if decode_node is None:
            return

        if enabled:
            # Rewire decode to read from 2nd pass outputs
            sp_denoise = self.node_graph.get_node_by_name("second_pass_denoise")
            sp_latents = self.node_graph.get_node_by_name("second_pass_latents")
            denoise_node = self.node_graph.get_node_by_name("denoise")
            latents_node = self.node_graph.get_node_by_name("latents")

            if sp_denoise and sp_latents and denoise_node and latents_node:
                # Disconnect old sources
                decode_node.disconnect("video_latents", denoise_node, "video_latents")
                decode_node.disconnect("audio_latents", denoise_node, "audio_latents")
                decode_node.disconnect("latent_num_frames", latents_node, "latent_num_frames")
                decode_node.disconnect("latent_height", latents_node, "latent_height")
                decode_node.disconnect("latent_width", latents_node, "latent_width")
                decode_node.disconnect("audio_num_frames", latents_node, "audio_num_frames")

                # Connect new sources
                decode_node.connect("video_latents", sp_denoise, "video_latents")
                decode_node.connect("audio_latents", sp_denoise, "audio_latents")
                decode_node.connect("latent_num_frames", sp_latents, "latent_num_frames")
                decode_node.connect("latent_height", sp_latents, "latent_height")
                decode_node.connect("latent_width", sp_latents, "latent_width")
                decode_node.connect("audio_num_frames", sp_latents, "audio_num_frames")
        else:
            # Rewire decode back to first pass outputs
            denoise_node = self.node_graph.get_node_by_name("denoise")
            latents_node = self.node_graph.get_node_by_name("latents")
            sp_denoise = self.node_graph.get_node_by_name("second_pass_denoise")
            sp_latents = self.node_graph.get_node_by_name("second_pass_latents")

            if denoise_node and latents_node and sp_denoise and sp_latents:
                # Disconnect 2nd pass sources (if connected)
                try:
                    decode_node.disconnect("video_latents", sp_denoise, "video_latents")
                    decode_node.disconnect("audio_latents", sp_denoise, "audio_latents")
                    decode_node.disconnect("latent_num_frames", sp_latents, "latent_num_frames")
                    decode_node.disconnect("latent_height", sp_latents, "latent_height")
                    decode_node.disconnect("latent_width", sp_latents, "latent_width")
                    decode_node.disconnect("audio_num_frames", sp_latents, "audio_num_frames")
                except Exception:
                    pass

                # Reconnect first pass sources
                decode_node.connect("video_latents", denoise_node, "video_latents")
                decode_node.connect("audio_latents", denoise_node, "audio_latents")
                decode_node.connect("latent_num_frames", latents_node, "latent_num_frames")
                decode_node.connect("latent_height", latents_node, "latent_height")
                decode_node.connect("latent_width", latents_node, "latent_width")
                decode_node.connect("audio_num_frames", latents_node, "audio_num_frames")

        # Progress bar always shows first-pass steps initially;
        # stage_started signal resets it for the 2nd pass at runtime.
        self.progress_bar.setMaximum(self.gen_settings.num_inference_steps)

    def get_dialog_specs(self):
        return {
            "model_manager": {
                "key": lambda _data: "model_manager",
                "factory": lambda data: ModelManagerDialog(
                    "model_manager",
                    self.directories,
                    self.preferences,
                    self.video_viewer,
                    target=data.get("target", "model"),
                ),
            },
            "model_download": {
                "key": lambda _data: "model_download",
                "factory": lambda _data: self._create_download_dialog(),
            },
            "lora_manager": {
                "key": lambda _data: "lora_manager",
                "factory": lambda _data: LoraManagerDialog(
                    "lora_manager",
                    self.directories,
                    self.preferences,
                    self.video_viewer,
                    lambda: self._last_run_json_graph,
                ),
            },
            "lora_advanced": {
                "key": lambda data: f"lora_advanced_{data.get('id', '')}",
                "close_key": lambda data: data.get("dialog_key", f"lora_advanced_{data.get('id', '')}"),
                "factory": lambda data: LoraAdvancedDialog(
                    f"lora_advanced_{data.get('id', '')}",
                    data.get("config", {}),
                    temp_path=str(self.directories.temp_path),
                    mask_width=self.gen_settings.video_width,
                    mask_height=self.gen_settings.video_height,
                ),
            },
            "source_image": {
                "key": lambda data: "source_image",
                "factory": lambda data: self._create_source_image_dialog(data),
            },
        }

    def _create_source_image_dialog(self, data: dict) -> SourceImageDialog:
        condition_id = data.get("condition_id")
        self._pending_condition_id = condition_id
        if condition_id and condition_id in self._visual_conditions:
            cond = self._visual_conditions[condition_id]
            image_path = cond.get("image_path")
            layers = cond.get("layers", [])
        else:
            image_path = self._source_image_path
            layers = self._source_image_layers
        return SourceImageDialog(
            "source_image",
            self.directories,
            self.preferences,
            image_path,
            layers,
            self.gen_settings.video_width,
            self.gen_settings.video_height,
        )

    def _create_download_dialog(self) -> ModelDownloadDialog:
        from frameartisan.app.app import get_app_database_path

        database = Database(get_app_database_path())
        return ModelDownloadDialog(self.directories, self.preferences, database)

    def open_dialog(self, dialog_key, dialog_factory):
        if dialog_key not in self.dialogs:
            self.dialogs[dialog_key] = dialog_factory()
            self.dialogs[dialog_key].setParent(self, Qt.WindowType.Window)
            self.dialogs[dialog_key].show()
        else:
            self.dialogs[dialog_key].raise_()
            self.dialogs[dialog_key].activateWindow()

    def close_dialog(self, dialog_key):
        if dialog_key in self.dialogs:
            dialog = self.dialogs.pop(dialog_key)
            dialog.hide()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_generate(
        self,
        seed: int,
        positive_prompt: str,
        negative_prompt: str,
        positive_prompt_changed: bool,
        negative_prompt_changed: bool,
        seed_changed: bool,
    ) -> None:
        if self.thread is None or self.node_graph is None:
            return

        if self.thread.isRunning():
            logger.warning("Generation already running; ignoring generate request")
            return

        if positive_prompt_changed:
            prompt_node = self.node_graph.get_node_by_name("prompt")
            if prompt_node is not None:
                prompt_node.update_value(positive_prompt)

        if negative_prompt_changed:
            neg_node = self.node_graph.get_node_by_name("negative_prompt")
            if neg_node is not None:
                neg_node.update_value(negative_prompt)

        if seed_changed:
            seed_node = self.node_graph.get_node_by_name("seed")
            if seed_node is not None:
                seed_node.update_value(seed)

        self._start_generation()

    def _setup_compile_cache(self) -> None:
        """Configure torch inductor persistent cache directory."""
        cache_path = getattr(self.directories, "cache_path", None)
        if not cache_path:
            logger.warning("No cache_path configured — compile cache disabled")
            return
        compile_cache_dir = os.path.join(cache_path, "torch_compile")
        os.makedirs(compile_cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache_dir
        logger.info("Compile cache dir: %s", compile_cache_dir)

    def _start_generation(self) -> None:
        """Serialize the current graph and kick off generation."""
        if self.thread is None or self.node_graph is None:
            return
        if self.thread.isRunning():
            logger.warning("Generation already running; ignoring generate request")
            return

        self._setup_compile_cache()

        # Check upsampler model exists when 2nd pass is enabled
        if self.gen_settings.second_pass_enabled:
            upsampler_path = os.path.join(str(self.directories.models_diffusers), LTX2_LATENT_UPSAMPLER_DIR)
            if not os.path.isdir(upsampler_path):
                self.event_bus.publish(
                    "show_snackbar",
                    {"action": "show", "message": "Latent upsampler model not found. Download it first."},
                )
                return

            # Set upsampler path on the node
            upsample_node = self.node_graph.get_node_by_name("upsample")
            if upsample_node is not None:
                upsample_node.upsampler_model_path = upsampler_path
                upsample_node.set_updated()

        # Stage 1 preview: temporarily disable stage 2 nodes so only
        # stage 1 + decode runs.  The staged graph keeps stage 2 enabled
        # so "continue to stage 2" can re-enable them via update_from_json.
        use_preview = (
            self.gen_settings.second_pass_enabled
            and self.gen_settings.preview_stage1
            and not self._stage1_preview_active
            and not getattr(self, "_skip_preview", False)
        )
        if use_preview:
            self._stage1_preview_active = True
            self._toggle_second_pass(False)

        self.progress_bar.setValue(0)
        self.thread.save_video_metadata = self.preferences.save_video_metadata
        self.thread.save_source_images = self.preferences.save_source_images
        self.thread.save_source_audio = self.preferences.save_source_audio
        self.thread.save_source_video = self.preferences.save_source_video
        self.thread.auto_save_videos = self.prompt_bar.generate_button.auto_save
        # Debug: log audio_encode state at generation time
        audio_encode = self.node_graph.get_node_by_name("audio_encode")
        if audio_encode is not None:
            logger.debug(
                "Generation start: audio_encode enabled=%s, updated=%s, path=%s, gen_settings.audio_conditioning_enabled=%s",
                audio_encode.enabled, audio_encode.updated, audio_encode.audio_path,
                self.gen_settings.audio_conditioning_enabled,
            )

        json_graph = self.node_graph.to_json()
        self._last_run_json_graph = json_graph
        self.thread.start_generation(json_graph)
        self.prompt_bar.set_generating(True)

    _EXPORT_ATTRS = {"video_codec", "video_crf", "video_preset", "audio_codec", "audio_bitrate_kbps"}

    # Settings attrs whose graph node has a shorter name (strip "video_" prefix).
    _SETTINGS_TO_NODE: dict[str, str] = {
        "video_width": "width",
        "video_height": "height",
    }

    _NUM_FRAMES_TRIGGERS = {"video_duration", "frame_rate"}

    @staticmethod
    def _read_component_overrides(model_id: int | None) -> dict[str, int] | None:
        """Read current component overrides from the DB for a model."""
        if model_id is None:
            return None
        try:
            from frameartisan.app.app import get_app_database_path
            from frameartisan.utils.database import Database

            db_path = get_app_database_path()
            if db_path:
                db = Database(db_path)
                rows = db.fetch_all(
                    "SELECT component_type, component_id FROM model_component_override WHERE model_id = ?",
                    (model_id,),
                )
                return {ct: cid for ct, cid in rows} if rows else None
        except Exception:
            pass
        return None

    def on_generation_change(self, data: dict) -> None:
        attr = data.get("attr")
        value = data.get("value")
        if attr is None:
            return

        if attr == "active_loras":
            self.gen_settings.active_loras = value
            self.gen_settings.save(self.settings)
            if self.node_graph is not None:
                lora_node = self.node_graph.get_node_by_name("lora")
                if lora_node is not None:
                    lora_node.update_loras(value)
                second_pass_lora_node = self.node_graph.get_node_by_name("second_pass_lora")
                if second_pass_lora_node is not None:
                    filtered = [c for c in value if c.get("apply_to_second_stage", False)]
                    second_pass_lora_node.update_loras(filtered)
            return

        if hasattr(self.gen_settings, attr):
            setattr(self.gen_settings, attr, value)
            self.gen_settings.save(self.settings)

        if attr == "num_inference_steps":
            self.progress_bar.setMaximum(int(value))

        if attr == "component_variant_changed":
            if self.node_graph is not None:
                target = data.get("target", "model")
                node_name = "second_pass_model" if target == "second_pass_model" else "model"
                node = self.node_graph.get_node_by_name(node_name)
                if node is not None:
                    node.component_overrides = self._read_component_overrides(
                        node.model_id
                    )
                    node.set_updated()
            return

        if attr == "use_torch_compile":
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.set_updated()

        if attr == "torch_compile_max_autotune":
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.set_updated()

        if attr == "offload_strategy":
            get_model_manager().offload_strategy = value
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.offload_strategy = value
                    model_node.set_updated()

        if attr == "group_offload_use_stream":
            get_model_manager().group_offload_use_stream = value
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.group_offload_use_stream = value
                    model_node.set_updated()

        if attr == "group_offload_low_cpu_mem":
            get_model_manager().group_offload_low_cpu_mem = value
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.group_offload_low_cpu_mem = value
                    model_node.set_updated()

        if attr == "streaming_decode":
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.streaming_decode = value
                    model_node.set_updated()

        if attr == "ff_chunking":
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.ff_chunking = value
                    model_node.set_updated()

        if attr == "ff_num_chunks":
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.ff_num_chunks = value
                    model_node.set_updated()

        if attr in ("preview_decode", "preview_time_upscale", "preview_space_upscale"):
            self._sync_preview_decode_settings()

        if attr == "second_pass_enabled":
            self._toggle_second_pass(value)

        if attr == "second_pass_steps":
            if self.node_graph is not None:
                node = self.node_graph.get_node_by_name("second_pass_steps")
                if node is not None:
                    node.update_value(int(value))

        if attr == "second_pass_guidance":
            if self.node_graph is not None:
                node = self.node_graph.get_node_by_name("second_pass_guidance")
                if node is not None:
                    node.update_value(float(value))

        if attr == "source_image_enabled":
            if self.node_graph is not None and self._source_image_path:
                source_node = self.node_graph.get_node_by_name("source_image")
                encode_node = self.node_graph.get_node_by_name("image_encode")
                if source_node is not None:
                    source_node.enabled = value
                    source_node.set_updated()
                if encode_node is not None:
                    encode_node.enabled = value
                    encode_node.set_updated()

        if attr == "visual_conditions_enabled":
            if self.node_graph is not None:
                condition_encode = self.node_graph.get_node_by_name("condition_encode")
                if condition_encode is not None:
                    has_video = self._video_condition is not None and self.gen_settings.video_conditioning_enabled
                    condition_encode.enabled = value or has_video
                    condition_encode.set_updated()
                # Enable/disable all dynamic ImageLoadNodes for conditions
                for cond_data in self._visual_conditions.values():
                    node_name = cond_data.get("node_name")
                    if node_name:
                        img_node = self.node_graph.get_node_by_name(node_name)
                        if img_node is not None:
                            img_node.enabled = value
                            img_node.set_updated()

        if attr == "audio_conditioning_enabled":
            if self.node_graph is not None:
                audio_encode = self.node_graph.get_node_by_name("audio_encode")
                if audio_encode is not None:
                    audio_encode.enabled = value
                    audio_encode.set_updated()

        if attr == "video_conditioning_enabled":
            if self.node_graph is not None and self._video_condition is not None:
                node_name_vc = self._video_condition.get("node_name")
                if node_name_vc:
                    vid_node = self.node_graph.get_node_by_name(node_name_vc)
                    if vid_node is not None:
                        vid_node.enabled = value
                        vid_node.set_updated()
                # Enable/disable condition_encode (check if image conditions also exist)
                condition_encode = self.node_graph.get_node_by_name("condition_encode")
                if condition_encode is not None:
                    has_images = bool(self._visual_conditions) and self.gen_settings.visual_conditions_enabled
                    condition_encode.enabled = value or has_images
                    condition_encode.set_updated()

        node_name = self._SETTINGS_TO_NODE.get(attr, attr)

        if self.node_graph is not None:
            if node_name in self._EXPORT_ATTRS:
                video_send_node = self.node_graph.get_node_by_name("video_send")
                if video_send_node is not None:
                    setattr(video_send_node, node_name, value)
            else:
                node = self.node_graph.get_node_by_name(node_name)
                if node is not None and hasattr(node, "update_value"):
                    node.update_value(value)

            # Recompute num_frames when duration or fps changes.
            if attr in self._NUM_FRAMES_TRIGGERS:
                num_frames = compute_num_frames(
                    self.gen_settings.video_duration,
                    self.gen_settings.frame_rate,
                )
                nf_node = self.node_graph.get_node_by_name("num_frames")
                if nf_node is not None and hasattr(nf_node, "update_value"):
                    nf_node.update_value(num_frames)

    def on_model_event(self, data: dict) -> None:
        action = data.get("action")
        target = data.get("target", "model")  # "model" or "second_pass_model"

        if action == "update" and target == "second_pass_model":
            model_data_object = data.get("model_data_object")
            if model_data_object is None:
                return
            self.gen_settings.second_pass_model = ModelDataObject(
                name=model_data_object.name,
                version=model_data_object.version,
                filepath=model_data_object.filepath,
                model_type=model_data_object.model_type,
                id=model_data_object.id,
            )
            self.gen_settings.save(self.settings)
            if self.node_graph is not None:
                sp_model_node = self.node_graph.get_node_by_name("second_pass_model")
                if sp_model_node is not None:
                    sp_model_node.update_value(model_data_object.filepath, model_data_object.id)
                    sp_model_node.component_overrides = self._read_component_overrides(
                        model_data_object.id
                    )

                sp_model_type_node = self.node_graph.get_node_by_name("second_pass_model_type")
                if sp_model_type_node is not None:
                    sp_model_type_node.update_value(model_data_object.model_type)

            # Apply model-type defaults to 2nd pass steps/guidance
            mt = model_data_object.model_type
            defaults = SECOND_PASS_MODEL_TYPE_DEFAULTS.get(mt, {})
            if defaults:
                steps = int(defaults.get("num_inference_steps", self.gen_settings.second_pass_steps))
                guidance = float(defaults.get("guidance_scale", self.gen_settings.second_pass_guidance))

                self.gen_settings.second_pass_steps = steps
                self.gen_settings.second_pass_guidance = guidance
                self.gen_settings.save(self.settings)

                if self.node_graph is not None:
                    sp_steps = self.node_graph.get_node_by_name("second_pass_steps")
                    if sp_steps is not None:
                        sp_steps.update_value(steps)
                    sp_guidance = self.node_graph.get_node_by_name("second_pass_guidance")
                    if sp_guidance is not None:
                        sp_guidance.update_value(guidance)

            self.event_bus.publish("second_pass_model_type_changed", {"model_type": mt})
            return

        if action == "update":
            model_data_object = data.get("model_data_object")
            if model_data_object is None:
                return
            self.gen_settings.model = ModelDataObject(
                name=model_data_object.name,
                version=model_data_object.version,
                filepath=model_data_object.filepath,
                model_type=model_data_object.model_type,
                id=model_data_object.id,
            )
            self.gen_settings.save(self.settings)
            if self.node_graph is not None:
                model_node = self.node_graph.get_node_by_name("model")
                if model_node is not None:
                    model_node.update_value(model_data_object.filepath, model_data_object.id)
                    model_node.component_overrides = self._read_component_overrides(
                        model_data_object.id
                    )

                model_type_node = self.node_graph.get_node_by_name("model_type")
                if model_type_node is not None:
                    model_type_node.update_value(model_data_object.model_type)

            # Apply model-type defaults to steps/guidance graph nodes and settings.
            mt = model_data_object.model_type
            defaults = MODEL_TYPE_DEFAULTS.get(mt, {})
            if defaults:
                steps = int(defaults.get("num_inference_steps", self.gen_settings.num_inference_steps))
                guidance = float(defaults.get("guidance_scale", self.gen_settings.guidance_scale))

                self.gen_settings.num_inference_steps = steps
                self.gen_settings.guidance_scale = guidance
                self.gen_settings.save(self.settings)

                self.progress_bar.setMaximum(steps)

                if self.node_graph is not None:
                    steps_node = self.node_graph.get_node_by_name("num_inference_steps")
                    if steps_node is not None:
                        steps_node.update_value(steps)
                    guidance_node = self.node_graph.get_node_by_name("guidance_scale")
                    if guidance_node is not None:
                        guidance_node.update_value(guidance)

            self.event_bus.publish("model_type_changed", {"model_type": mt})

    def on_module_event(self, data: dict) -> None:
        action = data.get("action")
        if action == "clear_vram":
            if self.thread is not None:
                self.thread.abort_graph()
                self.thread.clear_node_values()
            get_model_manager().clear()
            gc.collect()
            gc.collect()
        elif action == "clear_graph":
            # Clear VRAM first (same as clear_vram)
            if self.thread is not None:
                self.thread.abort_graph()
                self.thread.clear_node_values()
            get_model_manager().clear()

            # Reset internal conditioning state
            self._source_image_path = None
            self._source_image_layers = []
            self._visual_conditions.clear()
            self._pending_condition_id = None
            self._video_condition = None
            self._audio_path = None
            self._audio_from_video = False
            self.gen_settings.active_loras = []

            # Tell UI panels to reset themselves
            self.event_bus.publish("graph_cleared", {})

            # Reset prompts to defaults
            self.prompt_bar.positive_prompt.setPlainText("")
            self.prompt_bar.previous_positive_prompt = None
            self.prompt_bar.negative_prompt.setPlainText(DEFAULT_NEGATIVE_PROMPT)
            self.prompt_bar.previous_negative_prompt = DEFAULT_NEGATIVE_PROMPT

            self.build_graph()
        elif action == "clear_compile_cache":
            self._clear_compile_cache()
        elif action == "request_compile_cache_size":
            self._publish_compile_cache_size()

    def _get_compile_cache_dir(self) -> str | None:
        cache_path = getattr(self.directories, "cache_path", None)
        if not cache_path:
            return None
        return os.path.join(cache_path, "torch_compile")

    def _compute_dir_size(self, path: str) -> int:
        """Return total size in bytes of all files under *path*."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    def _publish_compile_cache_size(self) -> None:
        cache_dir = self._get_compile_cache_dir()
        if cache_dir and os.path.isdir(cache_dir):
            size_bytes = self._compute_dir_size(cache_dir)
        else:
            size_bytes = 0
        self.event_bus.publish("compile_cache_size", {"size_bytes": size_bytes})

    def _clear_compile_cache(self) -> None:
        import shutil

        cache_dir = self._get_compile_cache_dir()
        if cache_dir and os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("Compile cache cleared: %s", cache_dir)
        self._publish_compile_cache_size()

    def on_lora_event(self, data: dict) -> None:
        action = data.get("action")
        if action == "trigger_clicked":
            trigger = data.get("trigger", "")
            if trigger:
                self.prompt_bar.positive_prompt.insertTriggerAtCursor(trigger)

    def on_source_image_event(self, data: dict) -> None:
        action = data.get("action")

        # Bridge to visual_condition system when a condition dialog is pending
        condition_id = self._pending_condition_id
        if condition_id and action in ("add", "update"):
            vc_action = "add" if condition_id not in self._visual_conditions else "update"
            self.event_bus.publish(
                "visual_condition",
                {
                    "action": vc_action,
                    "condition_id": condition_id,
                    "source_image_path": data.get("source_image_path"),
                    "source_thumb_path": data.get("source_thumb_path"),
                    "layers": data.get("layers", []),
                },
            )
            return

        if action in ("add", "update"):
            self._source_image_path = data.get("source_image_path")
            self._source_image_layers = data.get("layers", self._source_image_layers)
            if self.node_graph is not None:
                source_node = self.node_graph.get_node_by_name("source_image")
                if source_node is not None:
                    source_node.update_path(self._source_image_path)
        elif action == "update_layers":
            self._source_image_layers = data.get("layers", [])
        elif action == "remove":
            self._source_image_path = None
            self._source_image_layers = []
            self.gen_settings.source_image_enabled = False
            if self.node_graph is not None:
                source_node = self.node_graph.get_node_by_name("source_image")
                encode_node = self.node_graph.get_node_by_name("image_encode")
                if source_node is not None:
                    source_node.enabled = False
                    source_node.set_updated()
                if encode_node is not None:
                    encode_node.enabled = False
                    encode_node.set_updated()

    def on_visual_condition_event(self, data: dict) -> None:
        action = data.get("action")
        condition_id = data.get("condition_id")

        if action == "add":
            image_path = data.get("source_image_path")
            layers = data.get("layers", [])
            pixel_frame_index = data.get("pixel_frame_index", 1)
            strength = data.get("strength", 1.0)

            if self.node_graph is None:
                return

            from frameartisan.modules.generation.graph.nodes.image_load_node import ImageLoadNode

            node_name = f"source_image_{condition_id}"
            img_node = ImageLoadNode()
            img_node.enabled = self.gen_settings.visual_conditions_enabled
            self.node_graph.add_node(img_node, name=node_name)
            if image_path:
                img_node.update_path(image_path)

            # Connect to condition_encode.images (multi-connection)
            condition_encode = self.node_graph.get_node_by_name("condition_encode")
            if condition_encode is not None:
                condition_encode.connect("images", img_node, "image")
                condition_encode.enabled = True

            self._visual_conditions[condition_id] = {
                "id": condition_id,
                "image_path": image_path,
                "layers": layers,
                "node_name": node_name,
                "pixel_frame_index": pixel_frame_index,
                "strength": strength,
                "attention_scale": data.get("attention_scale", 1.0),
                "method": data.get("method", "auto"),
            }

            self._sync_conditions_to_node()

        elif action == "update":
            cond = self._visual_conditions.get(condition_id)
            if cond is None:
                return
            image_path = data.get("source_image_path")
            layers = data.get("layers", cond.get("layers", []))
            if image_path:
                cond["image_path"] = image_path
                cond["layers"] = layers
                if self.node_graph is not None:
                    img_node = self.node_graph.get_node_by_name(cond["node_name"])
                    if img_node is not None:
                        img_node.update_path(image_path)

        elif action == "update_settings":
            cond = self._visual_conditions.get(condition_id)
            if cond is None:
                return
            if "pixel_frame_index" in data:
                cond["pixel_frame_index"] = data["pixel_frame_index"]
            if "strength" in data:
                cond["strength"] = data["strength"]
            if "attention_scale" in data:
                cond["attention_scale"] = data["attention_scale"]
            if "method" in data:
                cond["method"] = data["method"]
            self._sync_conditions_to_node()

        elif action == "remove":
            cond = self._visual_conditions.pop(condition_id, None)
            if cond is None:
                return
            if self.node_graph is not None:
                node_name = cond.get("node_name")
                img_node = self.node_graph.get_node_by_name(node_name)
                if img_node is not None:
                    condition_encode = self.node_graph.get_node_by_name("condition_encode")
                    if condition_encode is not None:
                        condition_encode.disconnect_from_node(img_node)
                    self.node_graph.delete_node(img_node)

                self._sync_conditions_to_node()

                if not self._visual_conditions:
                    self.gen_settings.visual_conditions_enabled = False
                    has_video = self._video_condition is not None and self.gen_settings.video_conditioning_enabled
                    if condition_encode is not None and not has_video:
                        condition_encode.enabled = False
                        condition_encode.set_updated()

    def _sync_conditions_to_node(self) -> None:
        """Sync visual and video conditions metadata to the condition_encode node."""
        if self.node_graph is None:
            return
        condition_encode = self.node_graph.get_node_by_name("condition_encode")
        if condition_encode is None:
            return
        conditions = [
            {
                "type": "image",
                "pixel_frame_index": cond["pixel_frame_index"],
                "strength": cond["strength"],
                "attention_scale": cond.get("attention_scale", 1.0),
                "method": cond.get("method", "auto"),
            }
            for cond in self._visual_conditions.values()
        ]
        if self._video_condition is not None:
            video_cond = {
                "type": "video",
                "mode": self._video_condition.get("mode", "replace"),
                "pixel_frame_index": self._video_condition["pixel_frame_index"],
                "strength": self._video_condition["strength"],
                "attention_scale": self._video_condition.get("attention_scale", 1.0),
            }
            if self._video_condition.get("source_frame_start") is not None:
                video_cond["source_frame_start"] = self._video_condition["source_frame_start"]
            if self._video_condition.get("source_frame_end") is not None:
                video_cond["source_frame_end"] = self._video_condition["source_frame_end"]
            conditions.append(video_cond)
        condition_encode.update_conditions(conditions)

    def on_audio_condition_event(self, data: dict) -> None:
        action = data.get("action")

        if action == "add":
            audio_path = data.get("audio_path")
            self._audio_path = audio_path
            self._audio_from_video = data.get("from_video", False)
            trim_start = data.get("trim_start_s")
            trim_end = data.get("trim_end_s")
            logger.debug(
                "Audio condition add: path=%s, from_video=%s, trim=(%s, %s)",
                audio_path, self._audio_from_video, trim_start, trim_end,
            )
            if self.node_graph is not None:
                audio_encode = self.node_graph.get_node_by_name("audio_encode")
                if audio_encode is not None:
                    audio_encode.update_path(audio_path)
                    audio_encode.update_trim(trim_start, trim_end)
                    audio_encode.enabled = True
                    audio_encode.set_updated()
                    logger.debug(
                        "Audio encode node: enabled=%s, path=%s, trim=(%s, %s), updated=%s",
                        audio_encode.enabled, audio_encode.audio_path,
                        audio_encode.trim_start_s, audio_encode.trim_end_s,
                        audio_encode.updated,
                    )

        elif action == "update_trim":
            if self.node_graph is not None:
                audio_encode = self.node_graph.get_node_by_name("audio_encode")
                if audio_encode is not None:
                    trim_start = data.get("trim_start_s")
                    trim_end = data.get("trim_end_s")
                    audio_encode.update_trim(trim_start, trim_end)

        elif action == "remove":
            self._audio_path = None
            self._audio_from_video = False
            self.gen_settings.audio_conditioning_enabled = False
            self.gen_settings.save(self.settings)
            if self.node_graph is not None:
                audio_encode = self.node_graph.get_node_by_name("audio_encode")
                if audio_encode is not None:
                    audio_encode.enabled = False
                    audio_encode.update_path(None)
                    audio_encode.set_updated()

    def on_video_condition_event(self, data: dict) -> None:
        action = data.get("action")

        if action == "add":
            video_path = data.get("video_path")
            pixel_frame_index = data.get("pixel_frame_index", 1)
            strength = data.get("strength", 1.0)

            if self.node_graph is None:
                return

            # Remove existing video condition node if any
            if self._video_condition is not None:
                old_name = self._video_condition.get("node_name")
                if old_name:
                    old_node = self.node_graph.get_node_by_name(old_name)
                    if old_node is not None:
                        condition_encode = self.node_graph.get_node_by_name("condition_encode")
                        if condition_encode is not None:
                            condition_encode.disconnect_from_node(old_node)
                        self.node_graph.delete_node(old_node)

            from frameartisan.modules.generation.graph.nodes.video_load_node import VideoLoadNode

            node_name = "condition_video"
            vid_node = VideoLoadNode()
            vid_node.enabled = self.gen_settings.video_conditioning_enabled
            self.node_graph.add_node(vid_node, name=node_name)
            vid_node.update_path(video_path)

            # Connect to condition_encode.videos (multi-connection)
            condition_encode = self.node_graph.get_node_by_name("condition_encode")
            if condition_encode is not None:
                condition_encode.connect("videos", vid_node, "frames")
                condition_encode.enabled = True

            self._video_condition = {
                "video_path": video_path,
                "node_name": node_name,
                "pixel_frame_index": pixel_frame_index,
                "strength": strength,
                "attention_scale": data.get("attention_scale", 1.0),
                "mode": data.get("mode", "replace"),
                "source_frame_start": data.get("source_frame_start"),
                "source_frame_end": data.get("source_frame_end"),
            }

            self._sync_conditions_to_node()

        elif action == "update_settings":
            if self._video_condition is None:
                return
            if "pixel_frame_index" in data:
                self._video_condition["pixel_frame_index"] = data["pixel_frame_index"]
            if "strength" in data:
                self._video_condition["strength"] = data["strength"]
            if "attention_scale" in data:
                self._video_condition["attention_scale"] = data["attention_scale"]
            if "mode" in data:
                self._video_condition["mode"] = data["mode"]
            if "source_frame_start" in data:
                self._video_condition["source_frame_start"] = data["source_frame_start"]
            if "source_frame_end" in data:
                self._video_condition["source_frame_end"] = data["source_frame_end"]
            self._sync_conditions_to_node()

        elif action == "remove":
            if self._video_condition is None:
                return
            if self.node_graph is not None:
                node_name = self._video_condition.get("node_name")
                vid_node = self.node_graph.get_node_by_name(node_name)
                if vid_node is not None:
                    condition_encode = self.node_graph.get_node_by_name("condition_encode")
                    if condition_encode is not None:
                        condition_encode.disconnect_from_node(vid_node)
                    video_send = self.node_graph.get_node_by_name("video_send")
                    if video_send is not None:
                        video_send.disconnect_from_node(vid_node)
                    self.node_graph.delete_node(vid_node)

            self._video_condition = None
            self.gen_settings.video_conditioning_enabled = False

            self._sync_conditions_to_node()

            # Disable condition_encode if no image conditions either
            if self.node_graph is not None and not self._visual_conditions:
                condition_encode = self.node_graph.get_node_by_name("condition_encode")
                if condition_encode is not None:
                    condition_encode.enabled = False
                    condition_encode.set_updated()

    def on_generate_event(self, data: dict) -> None:
        action = data.get("action")
        if action == "generate_from_json":
            json_graph_str = data.get("json_graph")
            if json_graph_str:
                try:
                    graph_data = json.loads(json_graph_str) if isinstance(json_graph_str, str) else json_graph_str
                    if isinstance(graph_data, dict) and "nodes" in graph_data:
                        self._apply_loaded_graph_subset(graph_data)
                        # LoRA example generations always go to temp
                        saved_auto_save = self.prompt_bar.generate_button.auto_save
                        self.prompt_bar.generate_button.auto_save = False
                        self._start_generation()
                        self.prompt_bar.generate_button.auto_save = saved_auto_save
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse example graph: %s", e)

    def on_manage_dialog_event(self, data):
        dialog_type = data.get("dialog_type")
        action = data.get("action")

        spec = self.get_dialog_specs().get(dialog_type)
        if not spec:
            return

        if action == "open":
            dialog_key = spec["key"](data)
            if dialog_key is None:
                return
            self.open_dialog(dialog_key, lambda: spec["factory"](data))
        elif action == "close":
            close_key_fn = spec.get("close_key", spec["key"])
            dialog_key = close_key_fn(data)
            if dialog_key is None:
                return
            self.close_dialog(dialog_key)
            if dialog_type == "source_image":
                self._pending_condition_id = None

    # ------------------------------------------------------------------
    # Thread signal handlers
    # ------------------------------------------------------------------

    def _reset_progress_bar(self, value: int = 0) -> None:
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMaximum(self.gen_settings.num_inference_steps)
        self.progress_bar.setValue(value)

    def _on_generation_finished(self, result, duration) -> None:
        self._reset_progress_bar(self.gen_settings.num_inference_steps)
        if isinstance(result, str):
            # Video path from LTX2VideoSendNode
            self.video_viewer.load_video(result, autoplay=True)

        if self._stage1_preview_active:
            # Stage 1 preview complete — show "Continue to Stage 2" button
            self.prompt_bar.set_generating(False)
            self.prompt_bar.set_stage1_preview_mode(True)
            self.event_bus.publish(
                "status_message",
                {"action": "change", "message": f"Stage 1 preview ready ({duration:.1f}s). Continue or retry."},
            )
        else:
            self.prompt_bar.set_generating(False)
            if duration is not None:
                self.event_bus.publish(
                    "status_message", {"action": "change", "message": f"Generation completed in {duration:.2f}s"}
                )
        self._publish_compile_cache_size()

    def _continue_stage2(self) -> None:
        """Continue from stage 1 preview to stage 2.

        Re-enables stage 2 nodes on the staged graph and runs again.  The
        persistent run graph preserves stage 1 latents so only stage 2 +
        decode actually execute.
        """
        self._stage1_preview_active = False
        self._skip_preview = True
        self.prompt_bar.set_stage1_preview_mode(False)
        # Re-enable stage 2 on the staged graph — the serialized diff
        # triggers update_from_json to mark only the changed nodes.
        self._toggle_second_pass(True)
        self._start_generation()
        self._skip_preview = False

    def _retry_stage1(self) -> None:
        """Discard stage 1 preview and re-run with the current settings."""
        self._stage1_preview_active = False
        self.prompt_bar.set_stage1_preview_mode(False)
        # Keep stage 2 disabled for another preview run
        self._start_generation()

    def _on_generation_error(self, message: str, _recoverable: bool) -> None:
        logger.error("Generation error: %s", message)
        self._reset_progress_bar()
        self._stage1_preview_active = False
        self.prompt_bar.set_stage1_preview_mode(False)
        self.prompt_bar.set_generating(False)
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})
        self.event_bus.publish("status_message", {"action": "change", "message": "Generation failed"})

    def _on_generation_aborted(self) -> None:
        self._reset_progress_bar()
        self._stage1_preview_active = False
        self.prompt_bar.set_stage1_preview_mode(False)
        self.prompt_bar.set_generating(False)
        self.event_bus.publish("status_message", {"action": "change", "message": "Generation aborted"})

    def _on_abort_clicked(self) -> None:
        if self.thread is not None:
            self.thread.abort_graph()

    def _on_preview_video_ready(self, video_path: str) -> None:
        """Load the tiny-VAE decoded preview video into the player."""
        self.video_viewer.load_preview_video(video_path)

    def _sync_preview_decode_settings(self) -> None:
        """Push preview_decode flag and tiny VAE path to the generation thread."""
        if self.thread is None:
            return
        self.thread.preview_decode = self.gen_settings.preview_decode
        self.thread.preview_time_upscale = self.gen_settings.preview_time_upscale
        self.thread.preview_space_upscale = self.gen_settings.preview_space_upscale
        self.thread.tiny_vae_path = os.path.join(
            self.directories.models_diffusers, LTX2_TINY_VAE_DIR, LTX2_TINY_VAE_FILENAME
        )

    def _on_auto_save_changed(self, checked: bool) -> None:
        self.preferences.auto_save_videos = checked
        self.settings.setValue("auto_save_videos", checked)

    def step_progress_update(self, step: int):
        if step == 1:
            self._reset_progress_bar()
        self.progress_bar.setValue(step)

    def _on_stage_started(self, total_steps: int) -> None:
        """Reset progress bar for a new generation stage."""
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(total_steps)

    def _on_status_changed(self, message: str) -> None:
        if not message:
            # Empty message = return to normal progress mode
            self._reset_progress_bar(self.progress_bar.value())
            return
        self.progress_bar.setFormat(message)
        self.progress_bar.setRange(0, 0)  # indeterminate / busy mode
        self.event_bus.publish("status_message", {"action": "change", "message": message})

    def _on_status_log_message(self, message: str) -> None:
        self.event_bus.publish("status_message", {"action": "change", "message": message})

    # ------------------------------------------------------------------
    # Drag-and-drop: restore generation params from MP4 metadata
    # ------------------------------------------------------------------

    def _on_video_dropped(self, mp4_path: str) -> None:
        try:
            import av

            with av.open(mp4_path) as container:
                graph_json_str = container.metadata.get("comment")
        except Exception as e:
            logger.warning("Failed to read MP4 metadata: %s", e)
            self.event_bus.publish("show_snackbar", {"action": "show", "message": "Failed to read video metadata"})
            return

        if not graph_json_str:
            self.event_bus.publish("show_snackbar", {"action": "show", "message": "No generation data found in video"})
            return

        try:
            graph_data = json.loads(graph_json_str)
            if not isinstance(graph_data, dict) or "nodes" not in graph_data:
                raise ValueError("not a graph")
        except (json.JSONDecodeError, ValueError):
            self.event_bus.publish("show_snackbar", {"action": "show", "message": "No generation data found in video"})
            return

        self._apply_loaded_graph_subset(graph_data)
        self.video_viewer.load_video(mp4_path, autoplay=False)

    @staticmethod
    def _resolve_path(path: str | None) -> str | None:
        """Return *path* if the file exists on disk, else ``None``."""
        if path and os.path.isfile(path):
            return path
        return None

    def _apply_loaded_graph_subset(self, graph_data: dict) -> None:
        """Restore generation parameters from a saved graph JSON dict."""
        # --- Extract simple params via the standard helper ---
        wanted = [
            "prompt",
            "negative_prompt",
            "seed",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "num_frames",
            "frame_rate",
        ]
        subset = extract_dict_from_json_graph(graph_data, wanted, include_missing=True)

        # --- Prompts ---
        prompt = subset.get("prompt", "")
        neg_prompt = subset.get("negative_prompt", "")
        if prompt is not None:
            self.prompt_bar.positive_prompt.setPlainText(str(prompt))
            self.prompt_bar.previous_positive_prompt = str(prompt)
            if self.node_graph:
                node = self.node_graph.get_node_by_name("prompt")
                if node and hasattr(node, "update_value"):
                    node.update_value(str(prompt))
        if neg_prompt is not None:
            self.prompt_bar.negative_prompt.setPlainText(str(neg_prompt))
            self.prompt_bar.previous_negative_prompt = str(neg_prompt)
            if self.node_graph:
                node = self.node_graph.get_node_by_name("negative_prompt")
                if node and hasattr(node, "update_value"):
                    node.update_value(str(neg_prompt))

        # --- Seed ---
        seed = subset.get("seed")
        if seed is not None:
            self.prompt_bar.seed_text.setText(str(int(seed)))
            self.prompt_bar.seed_text.setDisabled(False)
            self.prompt_bar.random_checkbox.setChecked(False)
            self.prompt_bar.use_random_seed = False
            self.prompt_bar.previous_seed = int(seed)
            if self.node_graph:
                node = self.node_graph.get_node_by_name("seed")
                if node and hasattr(node, "update_value"):
                    node.update_value(int(seed))

        # --- Generation params ---
        width = subset.get("width")
        height = subset.get("height")
        steps = subset.get("num_inference_steps")
        guidance = subset.get("guidance_scale")
        frame_rate = subset.get("frame_rate")
        num_frames = subset.get("num_frames")

        if width is not None:
            self.gen_settings.video_width = int(width)
            if self.node_graph:
                node = self.node_graph.get_node_by_name("width")
                if node and hasattr(node, "update_value"):
                    node.update_value(int(width))

        if height is not None:
            self.gen_settings.video_height = int(height)
            if self.node_graph:
                node = self.node_graph.get_node_by_name("height")
                if node and hasattr(node, "update_value"):
                    node.update_value(int(height))

        if steps is not None:
            self.gen_settings.num_inference_steps = int(steps)
            self.progress_bar.setMaximum(int(steps))
            if self.node_graph:
                node = self.node_graph.get_node_by_name("num_inference_steps")
                if node and hasattr(node, "update_value"):
                    node.update_value(int(steps))

        if guidance is not None:
            self.gen_settings.guidance_scale = float(guidance)
            if self.node_graph:
                node = self.node_graph.get_node_by_name("guidance_scale")
                if node and hasattr(node, "update_value"):
                    node.update_value(float(guidance))

        if frame_rate is not None:
            self.gen_settings.frame_rate = int(frame_rate)
            if self.node_graph:
                node = self.node_graph.get_node_by_name("frame_rate")
                if node and hasattr(node, "update_value"):
                    node.update_value(int(frame_rate))

        # Reverse-compute duration from num_frames and frame_rate.
        # Use round() (not ceil) to recover the original duration — ceil is used
        # in compute_num_frames to snap UP to 8n+1, so reversing with ceil
        # would overshoot (e.g. 305 frames at 30fps → ceil(10.13) = 11, not 10).
        if num_frames is not None and frame_rate is not None:
            duration = max(1, round((int(num_frames) - 1) / int(frame_rate)))
            self.gen_settings.video_duration = duration
            if self.node_graph:
                nf_node = self.node_graph.get_node_by_name("num_frames")
                if nf_node and hasattr(nf_node, "update_value"):
                    nf_node.update_value(int(num_frames))

        # --- Model restoration (parse directly from graph nodes) ---
        nodes = graph_data.get("nodes", [])
        model_node_data = None
        lora_node_data = None
        for node in nodes:
            if not isinstance(node, dict):
                continue
            cls_name = node.get("class", "")
            if cls_name == "LTX2ModelNode" and model_node_data is None:
                model_node_data = node
            elif cls_name == "LTX2LoraNode" and lora_node_data is None:
                lora_node_data = node

        if model_node_data is not None:
            state = model_node_data.get("state", {})
            model_id = state.get("model_id")
            if model_id:
                try:
                    from frameartisan.app.app import get_app_database_path

                    db = Database(get_app_database_path())
                    row = db.select_one(
                        "model",
                        ["id", "name", "version", "filepath", "model_type"],
                        {"id": model_id, "deleted": 0},
                    )
                    if row:
                        # Apply saved component overrides to DB so the UI reflects them
                        component_overrides = state.get("component_overrides")
                        if component_overrides:
                            from frameartisan.app.component_registry import ComponentRegistry

                            db_path = get_app_database_path()
                            components_base_dir = os.path.join(
                                os.path.dirname(row["filepath"]), "_components"
                            )
                            registry = ComponentRegistry(db_path, components_base_dir)
                            for comp_type, comp_id in component_overrides.items():
                                registry.set_component_override(model_id, comp_type, int(comp_id))

                        mdo = ModelDataObject(
                            name=row["name"],
                            version=row["version"],
                            filepath=row["filepath"],
                            model_type=row["model_type"],
                            id=row["id"],
                        )
                        self.on_model_event({"action": "update", "model_data_object": mdo})
                    else:
                        self.event_bus.publish(
                            "show_snackbar",
                            {"action": "show", "message": "Model not found in database, using current model"},
                        )
                except Exception as e:
                    logger.warning("Failed to restore model from metadata: %s", e)

        # --- LoRA restoration ---
        if lora_node_data is not None:
            state = lora_node_data.get("state", {})
            lora_configs = state.get("lora_configs", [])

            # Clear existing LoRAs first
            lora_panel = self.right_menu.panel_instances.get("LoRA")
            if lora_panel is not None:
                for lora_id in list(lora_panel._items.keys()):
                    lora_panel._remove_item(lora_id)
                lora_panel._sync_to_graph()

            for cfg in lora_configs:
                lora_id = cfg.get("id")
                if not lora_id:
                    continue
                try:
                    from frameartisan.app.app import get_app_database_path

                    db = Database(get_app_database_path())
                    row = db.select_one(
                        "lora",
                        ["id", "name", "filepath", "hash"],
                        {"id": lora_id, "deleted": 0},
                    )
                    if row:
                        self.event_bus.publish(
                            "lora",
                            {
                                "action": "add",
                                "id": row["id"],
                                "name": row["name"],
                                "filepath": row["filepath"],
                                "hash": row["hash"],
                                "weight": cfg.get("weight", 1.0),
                                "enabled": cfg.get("enabled", True),
                                "video_strength": cfg.get("video_strength", 1.0),
                                "audio_strength": cfg.get("audio_strength", 1.0),
                                "granular_transformer_weights_enabled": cfg.get(
                                    "granular_transformer_weights_enabled", False
                                ),
                                "granular_transformer_weights": cfg.get("granular_transformer_weights"),
                                "is_slider": cfg.get("is_slider", False),
                                "apply_to_second_stage": cfg.get("apply_to_second_stage", False),
                            },
                        )
                    else:
                        lora_name = cfg.get("name", f"ID {lora_id}")
                        self.event_bus.publish(
                            "show_snackbar",
                            {"action": "show", "message": f"LoRA '{lora_name}' not found, skipping"},
                        )
                except Exception as e:
                    logger.warning("Failed to restore LoRA from metadata: %s", e)

        # --- Visual conditions restoration ---
        # Collect source_image_* nodes and the condition_encode state
        source_image_nodes: dict[str, dict] = {}  # node name → node data
        condition_encode_data = None
        legacy_source_image_data = None
        for node in nodes:
            if not isinstance(node, dict):
                continue
            cls_name = node.get("class", "")
            node_name = node.get("name", "")
            if cls_name == "ImageLoadNode":
                if node_name.startswith("source_image_"):
                    source_image_nodes[node_name] = node
                elif node_name == "source_image" and legacy_source_image_data is None:
                    legacy_source_image_data = node
            elif cls_name == "LTX2ConditionEncodeNode":
                condition_encode_data = node

        # Clear any existing visual conditions first
        for cid in list(self._visual_conditions.keys()):
            self.event_bus.publish("visual_condition", {"action": "remove", "condition_id": cid})

        if condition_encode_data is not None and source_image_nodes:
            # New multi-condition path
            conditions_list = condition_encode_data.get("state", {}).get("conditions", [])
            # Match conditions to image nodes by order
            sorted_img_names = sorted(source_image_nodes.keys())
            restored_count = 0
            enabled_count = 0
            for i, node_name in enumerate(sorted_img_names):
                img_data = source_image_nodes[node_name]
                if not img_data.get("enabled", False):
                    continue
                enabled_count += 1
                state = img_data.get("state", {})
                original_path = state.get("path")
                restored_path = self._resolve_path(original_path)
                if not restored_path:
                    continue

                restored_count += 1
                # Extract condition_id from node name (source_image_<id>)
                condition_id = node_name.removeprefix("source_image_")
                # Get frame/strength from conditions list if available
                cond_meta = conditions_list[i] if i < len(conditions_list) else {}
                pixel_frame_index = cond_meta.get("pixel_frame_index", 1)
                strength = cond_meta.get("strength", 1.0)
                method = cond_meta.get("method", "auto")

                self.event_bus.publish(
                    "visual_condition",
                    {
                        "action": "add",
                        "condition_id": condition_id,
                        "source_image_path": restored_path,
                        "source_thumb_path": restored_path,
                        "pixel_frame_index": pixel_frame_index,
                        "strength": strength,
                        "method": method,
                    },
                )

            if enabled_count > 0 and restored_count == 0:
                self.event_bus.publish(
                    "show_snackbar",
                    {"action": "show", "message": "Source image(s) not found, restoring as text-to-video"},
                )

        elif legacy_source_image_data is not None and legacy_source_image_data.get("enabled", False):
            # Legacy single-image path — restore as a visual condition
            state = legacy_source_image_data.get("state", {})
            original_path = state.get("path")
            restored_path = self._resolve_path(original_path)

            if restored_path:
                condition_id = str(uuid.uuid4())[:8]
                self.event_bus.publish(
                    "visual_condition",
                    {
                        "action": "add",
                        "condition_id": condition_id,
                        "source_image_path": restored_path,
                        "source_thumb_path": restored_path,
                    },
                )
            else:
                self.event_bus.publish(
                    "show_snackbar",
                    {"action": "show", "message": "Source image not found, restoring as text-to-video"},
                )

        # --- Audio condition restoration ---
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("class") == "LTX2AudioEncodeNode" and node.get("name") == "audio_encode":
                if not node.get("enabled", False):
                    break
                state = node.get("state", {})
                audio_path = state.get("audio_path")
                restored_audio = self._resolve_path(audio_path)
                if restored_audio:
                    self.event_bus.publish(
                        "audio_condition",
                        {"action": "add", "audio_path": restored_audio},
                    )
                    trim_start = state.get("trim_start_s")
                    trim_end = state.get("trim_end_s")
                    if trim_start is not None or trim_end is not None:
                        self.event_bus.publish(
                            "audio_condition",
                            {"action": "update_trim", "trim_start_s": trim_start, "trim_end_s": trim_end},
                        )
                break

        # --- Video condition restoration ---
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("class") == "VideoLoadNode" and (node.get("name") or "").startswith("condition_video"):
                if not node.get("enabled", False):
                    break
                state = node.get("state", {})
                video_path = state.get("path")
                restored_video = self._resolve_path(video_path)
                if restored_video:
                    # Get condition settings from the condition_encode node
                    ce_state = (condition_encode_data or {}).get("state", {})
                    conditions_list = ce_state.get("conditions", [])
                    # Video condition is usually the last condition or has type "video"
                    video_cond_meta = {}
                    for cond in conditions_list:
                        if cond.get("type") == "video":
                            video_cond_meta = cond
                            break

                    self.event_bus.publish(
                        "video_condition",
                        {
                            "action": "add",
                            "video_path": restored_video,
                            "pixel_frame_index": video_cond_meta.get("pixel_frame_index", 1),
                            "strength": video_cond_meta.get("strength", 1.0),
                            "mode": video_cond_meta.get("mode", "replace"),
                            "source_frame_start": video_cond_meta.get("source_frame_start"),
                            "source_frame_end": video_cond_meta.get("source_frame_end"),
                        },
                    )
                break

        # --- Refresh UI panel ---
        gen_panel = self.right_menu.panel_instances.get("Generation")
        if gen_panel is not None:
            gen_panel.update_panel(
                self.gen_settings.video_width,
                self.gen_settings.video_height,
                self.gen_settings.num_inference_steps,
                self.gen_settings.guidance_scale,
                self.gen_settings.video_duration,
                self.gen_settings.frame_rate,
                self.gen_settings.model,
                self.gen_settings.use_torch_compile,
                self.gen_settings.attention_backend,
                self.gen_settings.offload_strategy,
                self.gen_settings.group_offload_use_stream,
                self.gen_settings.group_offload_low_cpu_mem,
                self.gen_settings.second_pass_enabled,
                self.gen_settings.second_pass_model,
                self.gen_settings.second_pass_steps,
                self.gen_settings.second_pass_guidance,
                self.gen_settings.streaming_decode,
                self.gen_settings.ff_chunking,
                self.gen_settings.ff_num_chunks,
            )

        self.gen_settings.save(self.settings)
