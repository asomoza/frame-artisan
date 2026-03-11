from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from superqt import QLabeledDoubleSlider, QLabeledSlider

from frameartisan.modules.generation.constants import (
    ADVANCED_GUIDANCE_DEFAULTS,
    LTX2_LATENT_UPSAMPLER_DIR,
    LTX2_TINY_VAE_DIR,
    LTX2_TINY_VAE_FILENAME,
    MODEL_TYPE_DEFAULTS,
    OFFLOAD_STRATEGIES,
)
from frameartisan.modules.generation.panels.base_panel import BasePanel
from frameartisan.modules.generation.widgets.video_dimensions_widget import VideoDimensionsWidget
from frameartisan.utils.json_utils import cast_model


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


_HAS_TRITON = _triton_available()


if TYPE_CHECKING:
    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject


class GenerationPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.update_panel(
            self.gen_settings.video_width,
            self.gen_settings.video_height,
            self.gen_settings.num_inference_steps,
            self.gen_settings.guidance_scale,
            self.gen_settings.video_duration,
            self.gen_settings.frame_rate,
            self.gen_settings.model,
            bool(getattr(self.gen_settings, "use_torch_compile", False)),
            str(getattr(self.gen_settings, "attention_backend", "native")),
            str(getattr(self.gen_settings, "offload_strategy", "auto")),
            bool(getattr(self.gen_settings, "group_offload_use_stream", False)),
            bool(getattr(self.gen_settings, "group_offload_low_cpu_mem", False)),
            bool(getattr(self.gen_settings, "second_pass_enabled", False)),
            getattr(self.gen_settings, "second_pass_model", None),
            int(getattr(self.gen_settings, "second_pass_steps", 10)),
            float(getattr(self.gen_settings, "second_pass_guidance", 4.0)),
            bool(getattr(self.gen_settings, "streaming_decode", False)),
            bool(getattr(self.gen_settings, "ff_chunking", False)),
            int(getattr(self.gen_settings, "ff_num_chunks", 2)),
            bool(getattr(self.gen_settings, "advanced_guidance", False)),
            float(getattr(self.gen_settings, "stg_scale", 0.0)),
            str(getattr(self.gen_settings, "stg_blocks", "29")),
            float(getattr(self.gen_settings, "rescale_scale", 0.0)),
            float(getattr(self.gen_settings, "modality_scale", 1.0)),
            int(getattr(self.gen_settings, "guidance_skip_step", 0)),
        )

        self.event_bus.subscribe("model", self.on_model_event)
        self.event_bus.subscribe("json_graph", self.on_json_graph_event)
        self.event_bus.subscribe("model_type_changed", self._on_model_type_changed)
        self.event_bus.subscribe("second_pass_model_type_changed", self._on_second_pass_model_type_changed)
        self.event_bus.subscribe("generation_change", self._on_generation_change_for_resolution)
        self.event_bus.subscribe("compile_cache_size", self._on_compile_cache_size)
        self.event_bus.subscribe("model_downloaded", self._on_model_downloaded)

        # Saved slider values for restoring after distilled mode.
        self._saved_steps: int | None = None
        self._saved_guidance: float | None = None

        # Apply constraints for the initially selected model.
        self._apply_model_type_constraints(self.gen_settings.model.model_type)

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.video_dimensions = VideoDimensionsWidget()
        main_layout.addWidget(self.video_dimensions)

        step_guidance_layout = QGridLayout()
        steps_label = QLabel("Steps:")
        step_guidance_layout.addWidget(steps_label, 0, 0)

        self.steps_slider = QLabeledSlider()
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setSingleStep(1)
        self.steps_slider.setOrientation(Qt.Orientation.Horizontal)
        self.steps_slider.valueChanged.connect(self.on_steps_value_changed)
        step_guidance_layout.addWidget(self.steps_slider, 0, 1)

        guidance_label = QLabel("Guidance")
        guidance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_guidance_layout.addWidget(guidance_label, 1, 0)

        self.guidance_slider = QLabeledDoubleSlider()
        self.guidance_slider.setObjectName("guidance_slider")
        self.guidance_slider.setRange(1.0, 20.0)
        self.guidance_slider.setSingleStep(0.1)
        self.guidance_slider.setOrientation(Qt.Orientation.Horizontal)
        self.guidance_slider.valueChanged.connect(self.on_guidance_value_changed)
        step_guidance_layout.addWidget(self.guidance_slider, 1, 1)

        main_layout.addLayout(step_guidance_layout)

        # --- Advanced Guidance section ---
        self.advanced_guidance_checkbox = QCheckBox("Advanced Guidance")
        self.advanced_guidance_checkbox.toggled.connect(self.on_advanced_guidance_toggled)
        main_layout.addWidget(self.advanced_guidance_checkbox)

        self.advanced_guidance_frame = QFrame()
        self.advanced_guidance_frame.setObjectName("advanced_guidance_frame")
        ag_layout = QGridLayout()
        ag_layout.setContentsMargins(6, 6, 6, 6)

        ag_layout.addWidget(QLabel("STG Scale:"), 0, 0)
        self.stg_scale_slider = QLabeledDoubleSlider()
        self.stg_scale_slider.setRange(0.0, 5.0)
        self.stg_scale_slider.setSingleStep(0.1)
        self.stg_scale_slider.setOrientation(Qt.Orientation.Horizontal)
        self.stg_scale_slider.valueChanged.connect(self.on_stg_scale_changed)
        ag_layout.addWidget(self.stg_scale_slider, 0, 1)

        ag_layout.addWidget(QLabel("Rescale:"), 1, 0)
        self.rescale_scale_slider = QLabeledDoubleSlider()
        self.rescale_scale_slider.setRange(0.0, 1.0)
        self.rescale_scale_slider.setSingleStep(0.05)
        self.rescale_scale_slider.setOrientation(Qt.Orientation.Horizontal)
        self.rescale_scale_slider.valueChanged.connect(self.on_rescale_scale_changed)
        ag_layout.addWidget(self.rescale_scale_slider, 1, 1)

        ag_layout.addWidget(QLabel("Modality:"), 2, 0)
        self.modality_scale_slider = QLabeledDoubleSlider()
        self.modality_scale_slider.setRange(1.0, 10.0)
        self.modality_scale_slider.setSingleStep(0.1)
        self.modality_scale_slider.setOrientation(Qt.Orientation.Horizontal)
        self.modality_scale_slider.valueChanged.connect(self.on_modality_scale_changed)
        ag_layout.addWidget(self.modality_scale_slider, 2, 1)

        ag_layout.addWidget(QLabel("Skip Step:"), 3, 0)
        self.guidance_skip_step_slider = QLabeledSlider()
        self.guidance_skip_step_slider.setRange(0, 10)
        self.guidance_skip_step_slider.setSingleStep(1)
        self.guidance_skip_step_slider.setOrientation(Qt.Orientation.Horizontal)
        self.guidance_skip_step_slider.valueChanged.connect(self.on_guidance_skip_step_changed)
        ag_layout.addWidget(self.guidance_skip_step_slider, 3, 1)

        ag_layout.addWidget(QLabel("STG Blocks:"), 4, 0)
        self.stg_blocks_edit = QLineEdit("29")
        self.stg_blocks_edit.editingFinished.connect(self.on_stg_blocks_changed)
        ag_layout.addWidget(self.stg_blocks_edit, 4, 1)

        self.advanced_guidance_frame.setLayout(ag_layout)
        main_layout.addWidget(self.advanced_guidance_frame)
        self._update_advanced_guidance_visibility()

        # --- duration slider (1–20 seconds) ---
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration (s):")
        duration_layout.addWidget(duration_label)

        self.duration_slider = QLabeledSlider()
        self.duration_slider.setRange(1, 20)
        self.duration_slider.setSingleStep(1)
        self.duration_slider.setOrientation(Qt.Orientation.Horizontal)
        self.duration_slider.valueChanged.connect(self.on_duration_value_changed)
        duration_layout.addWidget(self.duration_slider)
        main_layout.addLayout(duration_layout)

        # --- frame_rate slider ---
        frame_rate_layout = QHBoxLayout()
        frame_rate_label = QLabel("FPS:")
        frame_rate_layout.addWidget(frame_rate_label)

        self.frame_rate_slider = QLabeledSlider()
        self.frame_rate_slider.setObjectName("frame_rate_slider")
        self.frame_rate_slider.setRange(8, 50)
        self.frame_rate_slider.setSingleStep(1)
        self.frame_rate_slider.setOrientation(Qt.Orientation.Horizontal)
        self.frame_rate_slider.valueChanged.connect(self.on_frame_rate_value_changed)
        frame_rate_layout.addWidget(self.frame_rate_slider)
        main_layout.addLayout(frame_rate_layout)

        main_layout.addSpacing(10)

        select_base_model_button = QPushButton("Load model")
        select_base_model_button.clicked.connect(self.open_model_manager_dialog)
        main_layout.addWidget(select_base_model_button)

        base_model_layout = QHBoxLayout()
        base_model_label = QLabel("Model: ")
        base_model_layout.addWidget(base_model_label, 0)
        self.selected_base_model_label = QLabel("no model selected")
        base_model_layout.addWidget(self.selected_base_model_label, 1, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(base_model_layout)

        self.use_torch_compile_checkbox = QCheckBox("Use torch.compile")
        self.use_torch_compile_checkbox.setChecked(bool(getattr(self.gen_settings, "use_torch_compile", False)))
        self.use_torch_compile_checkbox.toggled.connect(self.on_use_torch_compile_toggled)
        self.use_torch_compile_checkbox.setVisible(_HAS_TRITON)
        main_layout.addWidget(self.use_torch_compile_checkbox)

        self.max_autotune_checkbox = QCheckBox("Max Autotune")
        self.max_autotune_checkbox.setChecked(bool(getattr(self.gen_settings, "torch_compile_max_autotune", False)))
        self.max_autotune_checkbox.toggled.connect(self.on_max_autotune_toggled)
        main_layout.addWidget(self.max_autotune_checkbox)
        self._update_max_autotune_visibility()

        # Attention backend dropdown
        attention_layout = QHBoxLayout()
        attention_label = QLabel("Attention:")
        attention_layout.addWidget(attention_label)

        self.attention_backend_combobox = QComboBox()
        self._populate_attention_backends()
        self.attention_backend_combobox.currentIndexChanged.connect(self.on_attention_backend_changed)
        attention_layout.addWidget(self.attention_backend_combobox, 1)
        main_layout.addLayout(attention_layout)

        # Offload strategy dropdown
        offload_layout = QHBoxLayout()
        offload_label = QLabel("Offload:")
        offload_layout.addWidget(offload_label)

        self.offload_strategy_combobox = QComboBox()
        for strategy_id, display_name in OFFLOAD_STRATEGIES.items():
            self.offload_strategy_combobox.addItem(display_name, strategy_id)
        self.offload_strategy_combobox.currentIndexChanged.connect(self.on_offload_strategy_changed)
        offload_layout.addWidget(self.offload_strategy_combobox, 1)
        main_layout.addLayout(offload_layout)

        # Group offload options (visible only when group_offload is explicitly selected)
        self.use_stream_checkbox = QCheckBox("Use CUDA Streams")
        self.use_stream_checkbox.setChecked(bool(getattr(self.gen_settings, "group_offload_use_stream", False)))
        self.use_stream_checkbox.toggled.connect(self.on_use_stream_toggled)
        main_layout.addWidget(self.use_stream_checkbox)

        self.low_cpu_mem_checkbox = QCheckBox("Low CPU Memory")
        self.low_cpu_mem_checkbox.setChecked(bool(getattr(self.gen_settings, "group_offload_low_cpu_mem", False)))
        self.low_cpu_mem_checkbox.toggled.connect(self.on_low_cpu_mem_toggled)
        main_layout.addWidget(self.low_cpu_mem_checkbox)

        self._update_group_offload_options_visibility()

        # Streaming decode (temporal tiling with lower peak VRAM)
        self.streaming_decode_checkbox = QCheckBox("Streaming Decode")
        self.streaming_decode_checkbox.setChecked(bool(getattr(self.gen_settings, "streaming_decode", False)))
        self.streaming_decode_checkbox.toggled.connect(self.on_streaming_decode_toggled)
        main_layout.addWidget(self.streaming_decode_checkbox)

        # Feed-forward chunking (lower peak activation VRAM during denoise)
        ff_chunking_layout = QHBoxLayout()
        self.ff_chunking_checkbox = QCheckBox("FF Chunking")
        self.ff_chunking_checkbox.setChecked(bool(getattr(self.gen_settings, "ff_chunking", False)))
        self.ff_chunking_checkbox.toggled.connect(self.on_ff_chunking_toggled)
        ff_chunking_layout.addWidget(self.ff_chunking_checkbox)

        ff_chunks_label = QLabel("Chunks:")
        ff_chunking_layout.addWidget(ff_chunks_label)
        self.ff_num_chunks_spinbox = QSpinBox()
        self.ff_num_chunks_spinbox.setRange(2, 16)
        self.ff_num_chunks_spinbox.setValue(int(getattr(self.gen_settings, "ff_num_chunks", 2)))
        self.ff_num_chunks_spinbox.setToolTip("Number of chunks (higher = more VRAM savings, slower)")
        self.ff_num_chunks_spinbox.valueChanged.connect(self.on_ff_num_chunks_changed)
        ff_chunking_layout.addWidget(self.ff_num_chunks_spinbox)
        ff_chunking_layout.addStretch()

        self._ff_chunks_label = ff_chunks_label
        self._update_ff_chunks_visibility(self.ff_chunking_checkbox.isChecked())

        main_layout.addLayout(ff_chunking_layout)

        # Live preview (requires tiny VAE)
        self.preview_decode_checkbox = QCheckBox("Live Preview")
        self.preview_decode_checkbox.setToolTip(
            "Show a live video preview during denoising using a tiny VAE decoder.\n"
            "Requires the Tiny VAE model to be downloaded."
        )
        self.preview_decode_checkbox.setChecked(bool(getattr(self.gen_settings, "preview_decode", False)))
        self.preview_decode_checkbox.toggled.connect(self.on_preview_decode_toggled)
        main_layout.addWidget(self.preview_decode_checkbox)
        self._update_preview_decode_availability()

        # --- 2nd pass (upsample) section ---
        main_layout.addSpacing(10)

        self.second_pass_checkbox = QCheckBox("2nd Pass (Upsample)")
        self.second_pass_checkbox.toggled.connect(self.on_second_pass_toggled)
        main_layout.addWidget(self.second_pass_checkbox)

        self.second_pass_frame = QFrame()
        self.second_pass_frame.setObjectName("second_pass_frame")
        sp_layout = QVBoxLayout()
        sp_layout.setContentsMargins(6, 6, 6, 6)

        self.upsampled_resolution_label = QLabel("Upsampled: 0 x 0")
        sp_layout.addWidget(self.upsampled_resolution_label)

        self.select_second_pass_model_button = QPushButton("Load 2nd Pass Model")
        self.select_second_pass_model_button.clicked.connect(self.open_second_pass_model_manager_dialog)
        sp_layout.addWidget(self.select_second_pass_model_button)

        sp_model_layout = QHBoxLayout()
        sp_model_label = QLabel("Model: ")
        sp_model_layout.addWidget(sp_model_label, 0)
        self.selected_second_pass_model_label = QLabel("no model selected")
        sp_model_layout.addWidget(self.selected_second_pass_model_label, 1, alignment=Qt.AlignmentFlag.AlignRight)
        sp_layout.addLayout(sp_model_layout)

        sp_grid = QGridLayout()
        sp_steps_label = QLabel("Steps:")
        sp_grid.addWidget(sp_steps_label, 0, 0)
        self.second_pass_steps_slider = QLabeledSlider()
        self.second_pass_steps_slider.setRange(1, 100)
        self.second_pass_steps_slider.setSingleStep(1)
        self.second_pass_steps_slider.setOrientation(Qt.Orientation.Horizontal)
        self.second_pass_steps_slider.valueChanged.connect(self.on_second_pass_steps_changed)
        sp_grid.addWidget(self.second_pass_steps_slider, 0, 1)

        sp_guidance_label = QLabel("Guidance:")
        sp_grid.addWidget(sp_guidance_label, 1, 0)
        self.second_pass_guidance_slider = QLabeledDoubleSlider()
        self.second_pass_guidance_slider.setRange(1.0, 20.0)
        self.second_pass_guidance_slider.setSingleStep(0.1)
        self.second_pass_guidance_slider.setOrientation(Qt.Orientation.Horizontal)
        self.second_pass_guidance_slider.valueChanged.connect(self.on_second_pass_guidance_changed)
        sp_grid.addWidget(self.second_pass_guidance_slider, 1, 1)

        sp_layout.addLayout(sp_grid)
        self.second_pass_frame.setLayout(sp_layout)
        main_layout.addWidget(self.second_pass_frame)

        self._update_second_pass_visibility()

        main_layout.addStretch()

        download_models_button = QPushButton("Download Models")
        download_models_button.clicked.connect(self.on_download_models_clicked)
        main_layout.addWidget(download_models_button)

        clear_graph_button = QPushButton("Clear Graph")
        clear_graph_button.setObjectName("red_button")
        clear_graph_button.clicked.connect(self.on_clear_graph_clicked)
        main_layout.addWidget(clear_graph_button)

        clear_vram_button = QPushButton("Clear VRAM")
        clear_vram_button.setObjectName("red_button")
        clear_vram_button.clicked.connect(self.on_clear_vram_clicked)
        main_layout.addWidget(clear_vram_button)

        # Compile cache size + clear button
        self.compile_cache_label = QLabel("Compile cache: –")
        self.compile_cache_label.setStyleSheet("color: #9aa0a6;")
        self.compile_cache_label.setVisible(_HAS_TRITON)
        main_layout.addWidget(self.compile_cache_label)

        self.clear_compile_cache_button = QPushButton("Clear Compile Cache")
        self.clear_compile_cache_button.setObjectName("red_button")
        self.clear_compile_cache_button.clicked.connect(self.on_clear_compile_cache_clicked)
        self.clear_compile_cache_button.setVisible(_HAS_TRITON)
        main_layout.addWidget(self.clear_compile_cache_button)

        self.setLayout(main_layout)

    def on_steps_value_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "num_inference_steps", "value": value})

    def on_guidance_value_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "guidance_scale", "value": value})

    def on_duration_value_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "video_duration", "value": value})

    def on_frame_rate_value_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "frame_rate", "value": value})

    def on_advanced_guidance_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "advanced_guidance", "value": bool(checked)})
        if checked:
            # Pre-populate sliders with recommended defaults on first enable
            for attr, default_val in ADVANCED_GUIDANCE_DEFAULTS.items():
                widget = self._guidance_widget_for_attr(attr)
                if widget is not None:
                    current = widget.value() if hasattr(widget, "value") else None
                    if current is not None and current == 0 and default_val != 0:
                        widget.setValue(default_val)
                        self.event_bus.publish("generation_change", {"attr": attr, "value": default_val})
        self._update_advanced_guidance_visibility()

    def _guidance_widget_for_attr(self, attr: str):
        return {
            "stg_scale": self.stg_scale_slider,
            "rescale_scale": self.rescale_scale_slider,
            "modality_scale": self.modality_scale_slider,
            "guidance_skip_step": self.guidance_skip_step_slider,
        }.get(attr)

    def on_stg_scale_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "stg_scale", "value": value})

    def on_rescale_scale_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "rescale_scale", "value": value})

    def on_modality_scale_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "modality_scale", "value": value})

    def on_guidance_skip_step_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "guidance_skip_step", "value": value})

    def on_stg_blocks_changed(self):
        self.event_bus.publish("generation_change", {"attr": "stg_blocks", "value": self.stg_blocks_edit.text()})

    def _update_advanced_guidance_visibility(self):
        visible = self.advanced_guidance_checkbox.isChecked()
        self.advanced_guidance_frame.setVisible(visible)

    def update_panel(
        self,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        video_duration: int,
        frame_rate: int,
        model: ModelDataObject,
        use_torch_compile: bool = False,
        attention_backend: str = "native",
        offload_strategy: str = "auto",
        group_offload_use_stream: bool = False,
        group_offload_low_cpu_mem: bool = False,
        second_pass_enabled: bool = False,
        second_pass_model: ModelDataObject | None = None,
        second_pass_steps: int = 10,
        second_pass_guidance: float = 4.0,
        streaming_decode: bool = False,
        ff_chunking: bool = False,
        ff_num_chunks: int = 2,
        advanced_guidance: bool = False,
        stg_scale: float = 0.0,
        stg_blocks: str = "29",
        rescale_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_skip_step: int = 0,
    ):
        blockers = [
            QSignalBlocker(self.video_dimensions.width_slider),
            QSignalBlocker(self.video_dimensions.height_slider),
            QSignalBlocker(self.steps_slider),
            QSignalBlocker(self.guidance_slider),
            QSignalBlocker(self.duration_slider),
            QSignalBlocker(self.frame_rate_slider),
            QSignalBlocker(self.use_torch_compile_checkbox),
            QSignalBlocker(self.max_autotune_checkbox),
            QSignalBlocker(self.attention_backend_combobox),
            QSignalBlocker(self.offload_strategy_combobox),
            QSignalBlocker(self.use_stream_checkbox),
            QSignalBlocker(self.low_cpu_mem_checkbox),
            QSignalBlocker(self.second_pass_checkbox),
            QSignalBlocker(self.second_pass_steps_slider),
            QSignalBlocker(self.second_pass_guidance_slider),
            QSignalBlocker(self.streaming_decode_checkbox),
            QSignalBlocker(self.ff_chunking_checkbox),
            QSignalBlocker(self.ff_num_chunks_spinbox),
            QSignalBlocker(self.advanced_guidance_checkbox),
            QSignalBlocker(self.stg_scale_slider),
            QSignalBlocker(self.rescale_scale_slider),
            QSignalBlocker(self.modality_scale_slider),
            QSignalBlocker(self.guidance_skip_step_slider),
        ]
        try:
            self.video_dimensions.width_slider.setValue(int(width))
            self.video_dimensions.height_slider.setValue(int(height))
            self.video_dimensions.video_width_value_label.setText(str(int(width)))
            self.video_dimensions.video_height_value_label.setText(str(int(height)))
            self.video_dimensions._prev_width = int(width)
            self.video_dimensions._prev_height = int(height)
            self.steps_slider.setValue(int(num_inference_steps))
            self.guidance_slider.setValue(float(guidance_scale))
            self.duration_slider.setValue(int(video_duration))
            self.frame_rate_slider.setValue(int(frame_rate))
            self.selected_base_model_label.setText(model.name)
            self.use_torch_compile_checkbox.setChecked(bool(use_torch_compile))
            self.max_autotune_checkbox.setChecked(
                bool(getattr(self.gen_settings, "torch_compile_max_autotune", False))
            )
            self.use_stream_checkbox.setChecked(bool(group_offload_use_stream))
            self.low_cpu_mem_checkbox.setChecked(bool(group_offload_low_cpu_mem))
            self.second_pass_checkbox.setChecked(bool(second_pass_enabled))
            self.second_pass_steps_slider.setValue(int(second_pass_steps))
            self.second_pass_guidance_slider.setValue(float(second_pass_guidance))
            self.streaming_decode_checkbox.setChecked(bool(streaming_decode))
            self.ff_chunking_checkbox.setChecked(bool(ff_chunking))
            self.ff_num_chunks_spinbox.setValue(int(ff_num_chunks))
            self._update_ff_chunks_visibility(bool(ff_chunking))
            self.advanced_guidance_checkbox.setChecked(bool(advanced_guidance))
            self.stg_scale_slider.setValue(float(stg_scale))
            self.rescale_scale_slider.setValue(float(rescale_scale))
            self.modality_scale_slider.setValue(float(modality_scale))
            self.guidance_skip_step_slider.setValue(int(guidance_skip_step))
            self.stg_blocks_edit.setText(str(stg_blocks))
            if second_pass_model is not None:
                self.selected_second_pass_model_label.setText(second_pass_model.name)
        finally:
            for b in blockers:
                del b

        self._set_attention_backend_ui(attention_backend)
        self._set_offload_strategy_ui(offload_strategy)
        self._update_group_offload_options_visibility()
        self._update_ff_stream_exclusivity()
        self._update_max_autotune_visibility()
        self._update_second_pass_visibility()
        self._update_upsampled_resolution_label()
        self._update_advanced_guidance_visibility()

    def on_second_pass_toggled(self, checked: bool):
        if checked and not self._has_upsampler_model():
            blocker = QSignalBlocker(self.second_pass_checkbox)
            try:
                self.second_pass_checkbox.setChecked(False)
            finally:
                del blocker
            self.event_bus.publish(
                "show_snackbar",
                {"action": "show", "message": "Latent upsampler model not found. Download it first."},
            )
            return
        self.event_bus.publish("generation_change", {"attr": "second_pass_enabled", "value": bool(checked)})
        self._update_second_pass_visibility()

    def on_second_pass_steps_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "second_pass_steps", "value": value})

    def on_second_pass_guidance_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "second_pass_guidance", "value": value})

    def open_second_pass_model_manager_dialog(self):
        self.event_bus.publish(
            "manage_dialog",
            {"dialog_type": "model_manager", "action": "open", "target": "second_pass_model"},
        )

    def _has_upsampler_model(self) -> bool:
        return os.path.isdir(os.path.join(str(self.directories.models_diffusers), LTX2_LATENT_UPSAMPLER_DIR))

    def _update_second_pass_visibility(self):
        visible = self.second_pass_checkbox.isChecked()
        self.second_pass_frame.setVisible(visible)
        self._update_torch_compile_availability()

    def _update_torch_compile_availability(self):
        """Disable torch.compile when 2nd pass is enabled (shape changes cause recompilation)."""
        second_pass_on = self.second_pass_checkbox.isChecked()
        self.use_torch_compile_checkbox.setEnabled(not second_pass_on)
        if second_pass_on and self.use_torch_compile_checkbox.isChecked():
            self.use_torch_compile_checkbox.setChecked(False)
        self._update_max_autotune_visibility()

    def _update_upsampled_resolution_label(self):
        w = self.gen_settings.video_width * 2
        h = self.gen_settings.video_height * 2
        self.upsampled_resolution_label.setText(f"Upsampled: {w} x {h}")

    def _on_second_pass_model_type_changed(self, data: dict) -> None:
        model_type = data.get("model_type", 1)
        self._apply_second_pass_model_type_constraints(model_type)

    def _on_generation_change_for_resolution(self, data: dict) -> None:
        attr = data.get("attr")
        if attr in ("video_width", "video_height"):
            self._update_upsampled_resolution_label()

    def _apply_second_pass_model_type_constraints(self, model_type: int) -> None:
        defaults = MODEL_TYPE_DEFAULTS.get(model_type, {})
        if model_type == 2:
            # Distilled: lock steps and guidance for 2nd pass
            blocker_s = QSignalBlocker(self.second_pass_steps_slider)
            blocker_g = QSignalBlocker(self.second_pass_guidance_slider)
            try:
                self.second_pass_steps_slider.setValue(3)
                self.second_pass_guidance_slider.setValue(1.0)
            finally:
                del blocker_s
                del blocker_g
            self.second_pass_steps_slider.setEnabled(False)
            self.second_pass_guidance_slider.setEnabled(False)
        else:
            self.second_pass_steps_slider.setEnabled(True)
            self.second_pass_guidance_slider.setEnabled(True)
            if defaults:
                blocker_s = QSignalBlocker(self.second_pass_steps_slider)
                blocker_g = QSignalBlocker(self.second_pass_guidance_slider)
                try:
                    self.second_pass_steps_slider.setValue(
                        int(defaults.get("num_inference_steps", self.gen_settings.second_pass_steps))
                    )
                    self.second_pass_guidance_slider.setValue(
                        float(defaults.get("guidance_scale", self.gen_settings.second_pass_guidance))
                    )
                finally:
                    del blocker_s
                    del blocker_g

    def on_download_models_clicked(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "model_download", "action": "open"})

    def open_model_manager_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "model_manager", "action": "open"})

    def on_use_torch_compile_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "use_torch_compile", "value": bool(checked)})
        self._update_max_autotune_visibility()

    def on_max_autotune_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "torch_compile_max_autotune", "value": bool(checked)})

    def _update_max_autotune_visibility(self):
        self.max_autotune_checkbox.setVisible(self.use_torch_compile_checkbox.isChecked())

    def _populate_attention_backends(self):
        from frameartisan.app.model_manager import get_model_manager

        mm = get_model_manager()
        available_backends = mm.get_available_attention_backends()

        self.attention_backend_combobox.clear()
        for backend_id, display_name in available_backends:
            self.attention_backend_combobox.addItem(display_name, backend_id)

    def _set_attention_backend_ui(self, backend: str):
        for i in range(self.attention_backend_combobox.count()):
            if self.attention_backend_combobox.itemData(i) == backend:
                blocker = QSignalBlocker(self.attention_backend_combobox)
                try:
                    self.attention_backend_combobox.setCurrentIndex(i)
                finally:
                    del blocker
                return
        blocker = QSignalBlocker(self.attention_backend_combobox)
        try:
            self.attention_backend_combobox.setCurrentIndex(0)
        finally:
            del blocker

    def on_attention_backend_changed(self, index: int):
        backend = self.attention_backend_combobox.itemData(index)
        if backend:
            self.event_bus.publish("generation_change", {"attr": "attention_backend", "value": backend})

    def on_offload_strategy_changed(self, index: int):
        strategy = self.offload_strategy_combobox.itemData(index)
        if strategy:
            self.event_bus.publish("generation_change", {"attr": "offload_strategy", "value": strategy})
            self._update_group_offload_options_visibility()
            self._update_ff_stream_exclusivity()

    def on_use_stream_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "group_offload_use_stream", "value": bool(checked)})
        self.low_cpu_mem_checkbox.setEnabled(checked)
        self._update_ff_stream_exclusivity()

    def on_low_cpu_mem_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "group_offload_low_cpu_mem", "value": bool(checked)})

    def on_streaming_decode_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "streaming_decode", "value": bool(checked)})

    def on_ff_chunking_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "ff_chunking", "value": bool(checked)})
        self._update_ff_chunks_visibility(checked)
        self._update_ff_stream_exclusivity()

    def on_ff_num_chunks_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "ff_num_chunks", "value": int(value)})

    def on_preview_decode_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "preview_decode", "value": bool(checked)})

    def _update_preview_decode_availability(self):
        """Enable the live preview checkbox only if the tiny VAE is downloaded."""
        models_dir = getattr(self.directories, "models_diffusers", None)
        if not models_dir:
            self.preview_decode_checkbox.setEnabled(False)
            self.preview_decode_checkbox.setChecked(False)
            return
        tiny_vae_path = os.path.join(models_dir, LTX2_TINY_VAE_DIR, LTX2_TINY_VAE_FILENAME)
        available = os.path.isfile(tiny_vae_path)
        self.preview_decode_checkbox.setEnabled(available)
        if not available:
            self.preview_decode_checkbox.setChecked(False)

    def _on_model_downloaded(self, data: dict) -> None:
        if "tiny_vae" in data.get("variants", []):
            self._update_preview_decode_availability()

    def _update_ff_chunks_visibility(self, visible: bool):
        self._ff_chunks_label.setVisible(visible)
        self.ff_num_chunks_spinbox.setVisible(visible)

    def _update_group_offload_options_visibility(self):
        strategy = self.offload_strategy_combobox.currentData()
        visible = strategy in ("group_offload", "sequential_group_offload")
        self.use_stream_checkbox.setVisible(visible)
        self.low_cpu_mem_checkbox.setVisible(visible)
        if visible:
            self.low_cpu_mem_checkbox.setEnabled(self.use_stream_checkbox.isChecked())

    def _update_ff_stream_exclusivity(self):
        """FF chunking and sequential_group_offload + CUDA streams are incompatible.

        FF chunking calls the FF module multiple times per block forward, which
        disrupts the stream-based execution-order tracing used for prefetching.
        When one is active, disable and uncheck the other.
        """
        strategy = self.offload_strategy_combobox.currentData()
        is_seq = strategy == "sequential_group_offload"
        stream_on = self.use_stream_checkbox.isChecked() and self.use_stream_checkbox.isVisible()
        ff_on = self.ff_chunking_checkbox.isChecked()

        if is_seq and stream_on and ff_on:
            # Both active — CUDA streams take priority, disable FF chunking
            self.ff_chunking_checkbox.setChecked(False)
            self.ff_chunking_checkbox.setEnabled(False)
        elif is_seq and stream_on:
            self.ff_chunking_checkbox.setEnabled(False)
        elif is_seq and ff_on:
            self.use_stream_checkbox.setEnabled(False)
        else:
            self.ff_chunking_checkbox.setEnabled(True)
            if self.use_stream_checkbox.isVisible():
                self.use_stream_checkbox.setEnabled(True)

    def _set_offload_strategy_ui(self, strategy: str):
        for i in range(self.offload_strategy_combobox.count()):
            if self.offload_strategy_combobox.itemData(i) == strategy:
                blocker = QSignalBlocker(self.offload_strategy_combobox)
                try:
                    self.offload_strategy_combobox.setCurrentIndex(i)
                finally:
                    del blocker
                return
        blocker = QSignalBlocker(self.offload_strategy_combobox)
        try:
            self.offload_strategy_combobox.setCurrentIndex(0)
        finally:
            del blocker

    def _confirm_destructive_action(self, title: str, text: str) -> bool:
        res = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return res == QMessageBox.StandardButton.Yes

    def on_clear_graph_clicked(self):
        if not self._confirm_destructive_action(
            "Clear Graph",
            "This will reset the node graph to defaults (removing LoRAs/source image wiring). Continue?",
        ):
            return
        self.event_bus.publish("module", {"action": "clear_graph"})

    def on_clear_vram_clicked(self):
        if not self._confirm_destructive_action(
            "Clear VRAM",
            "This will abort any running generation and unload models from GPU memory. Continue?",
        ):
            return
        self.event_bus.publish("module", {"action": "clear_vram"})

    def on_clear_compile_cache_clicked(self):
        if not self._confirm_destructive_action(
            "Clear Compile Cache",
            "This will delete the torch.compile disk cache. The next compilation will be slower. Continue?",
        ):
            return
        self.event_bus.publish("module", {"action": "clear_compile_cache"})

    def _on_compile_cache_size(self, data: dict) -> None:
        size_bytes = data.get("size_bytes", 0)
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        self.compile_cache_label.setText(f"Compile cache: {size_str}")

    # ------------------------------------------------------------------
    # Model-type constraints
    # ------------------------------------------------------------------

    def _on_model_type_changed(self, data: dict) -> None:
        model_type = data.get("model_type", 1)
        self._apply_model_type_constraints(model_type)

    def _apply_model_type_constraints(self, model_type: int) -> None:
        defaults = MODEL_TYPE_DEFAULTS.get(model_type, {})
        if model_type == 2:
            # Save current values before locking.
            self._saved_steps = self.steps_slider.value()
            self._saved_guidance = self.guidance_slider.value()

            steps = int(defaults.get("num_inference_steps", 8))
            guidance = float(defaults.get("guidance_scale", 1.0))

            blocker_s = QSignalBlocker(self.steps_slider)
            blocker_g = QSignalBlocker(self.guidance_slider)
            try:
                self.steps_slider.setValue(steps)
                self.guidance_slider.setValue(guidance)
            finally:
                del blocker_s
                del blocker_g

            self.steps_slider.setEnabled(False)
            self.guidance_slider.setEnabled(False)
            # Distilled models don't support advanced guidance
            self.advanced_guidance_checkbox.setEnabled(False)
            if self.advanced_guidance_checkbox.isChecked():
                self.advanced_guidance_checkbox.setChecked(False)
        else:
            self.steps_slider.setEnabled(True)
            self.guidance_slider.setEnabled(True)
            self.advanced_guidance_checkbox.setEnabled(True)

            # Restore previously saved values if coming from distilled.
            if self._saved_steps is not None:
                self.steps_slider.setValue(self._saved_steps)
                self._saved_steps = None
            elif defaults:
                blocker = QSignalBlocker(self.steps_slider)
                try:
                    self.steps_slider.setValue(
                        int(defaults.get("num_inference_steps", self.gen_settings.num_inference_steps))
                    )
                finally:
                    del blocker

            if self._saved_guidance is not None:
                self.guidance_slider.setValue(self._saved_guidance)
                self._saved_guidance = None
            elif defaults:
                blocker = QSignalBlocker(self.guidance_slider)
                try:
                    self.guidance_slider.setValue(
                        float(defaults.get("guidance_scale", self.gen_settings.guidance_scale))
                    )
                finally:
                    del blocker

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_model_event(self, data):
        action = data.get("action")
        target = data.get("target", "model")
        if action == "update" and target == "second_pass_model":
            model_data_object = data.get("model_data_object")
            if model_data_object is not None:
                self.selected_second_pass_model_label.setText(model_data_object.name)
            else:
                self.selected_second_pass_model_label.setText("no model selected")
            return
        if action == "update":
            model_data_object = data.get("model_data_object")
            if model_data_object is not None:
                self.selected_base_model_label.setText(model_data_object.name)
            else:
                self.selected_base_model_label.setText("no model selected")

    def on_json_graph_event(self, data):
        action = data.get("action")
        if action == "loaded":
            data = data.get("data", {})

            width = data.get("video_width", self.gen_settings.video_width)
            height = data.get("video_height", self.gen_settings.video_height)
            num_inference_steps = data.get("num_inference_steps", self.gen_settings.num_inference_steps)
            guidance_scale = data.get("guidance_scale", self.gen_settings.guidance_scale)
            video_duration = data.get("video_duration", self.gen_settings.video_duration)
            frame_rate = data.get("frame_rate", self.gen_settings.frame_rate)
            model = cast_model(data.get("model", self.gen_settings.model))
            use_torch_compile = self.gen_settings.use_torch_compile
            attention_backend = self.gen_settings.attention_backend
            offload_strategy = self.gen_settings.offload_strategy
            group_offload_use_stream = self.gen_settings.group_offload_use_stream
            group_offload_low_cpu_mem = self.gen_settings.group_offload_low_cpu_mem

            self.update_panel(
                width,
                height,
                num_inference_steps,
                guidance_scale,
                video_duration,
                frame_rate,
                model,
                use_torch_compile,
                attention_backend,
                offload_strategy,
                group_offload_use_stream,
                group_offload_low_cpu_mem,
            )
