import math
from dataclasses import dataclass, field

from PyQt6.QtCore import QSettings

from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
from frameartisan.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from frameartisan.utils.json_utils import (
    cast_model,
    cast_number_range,
    cast_scheduler,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_json,
    coerce_str,
)


_MAX_DURATION_SECONDS = 20


def compute_num_frames(duration_seconds: int | float, frame_rate: int | float) -> int:
    """Compute the number of frames snapped to the next LTX2 8n+1 value.

    The VAE temporal compression ratio is 8, so valid frame counts are
    9, 17, 25, ... (i.e. ``8 * n + 1`` for n >= 1).

    Rounds *up* so the video is at least as long as the requested duration.
    """
    raw = int(round(duration_seconds * frame_rate))
    # Ceil to the next 8n+1 value so the video meets the requested duration.
    n = math.ceil((raw - 1) / 8)
    n = max(n, 1)  # minimum n=1 → 9 frames
    return 8 * n + 1


@dataclass(slots=True)
class GenerationSettings:
    right_menu_expanded: bool = True

    video_width: int = 640
    video_height: int = 352
    num_inference_steps: int = 24
    guidance_scale: float = 4.0
    guidance_start_end: list[float] = field(default_factory=lambda: [0.0, 1.0])
    scheduler: SchedulerDataObject = field(default_factory=SchedulerDataObject)
    strength: float = 0.5
    video_duration: int = 5
    frame_rate: int = 24
    model: ModelDataObject = field(default_factory=ModelDataObject)
    use_torch_compile: bool = False
    torch_compile_max_autotune: bool = False
    attention_backend: str = "native"
    offload_strategy: str = "auto"
    group_offload_use_stream: bool = False
    group_offload_low_cpu_mem: bool = False
    streaming_decode: bool = False
    ff_chunking: bool = False
    ff_num_chunks: int = 2
    preview_decode: bool = False
    preview_time_upscale: bool = False
    preview_space_upscale: bool = True

    video_codec: str = "libx264"
    video_crf: int = 23
    video_preset: str = "medium"
    audio_codec: str = "aac"
    audio_bitrate_kbps: int = 192

    source_image_enabled: bool = False  # legacy, migrated to visual_conditions_enabled
    visual_conditions_enabled: bool = False
    audio_conditioning_enabled: bool = False
    video_conditioning_enabled: bool = False
    video_conditioning_mode: str = "replace"

    second_pass_enabled: bool = False
    second_pass_model: ModelDataObject = field(default_factory=ModelDataObject)
    second_pass_steps: int = 10
    second_pass_guidance: float = 4.0

    advanced_guidance: bool = False
    stg_scale: float = 0.0
    stg_blocks: str = "29"
    rescale_scale: float = 0.0
    modality_scale: float = 1.0
    guidance_skip_step: int = 0

    active_loras: list = field(default_factory=list)

    GROUP: str = "generation"

    @classmethod
    def load(cls, qsettings: QSettings) -> "GenerationSettings":
        settings = cls()
        qsettings.beginGroup(settings.GROUP)

        try:
            settings.right_menu_expanded = coerce_bool(
                qsettings.value("right_menu_expanded", settings.right_menu_expanded),
                settings.right_menu_expanded,
            )

            settings.video_width = coerce_int(
                qsettings.value("video_width", settings.video_width), settings.video_width
            )
            settings.video_height = coerce_int(
                qsettings.value("video_height", settings.video_height), settings.video_height
            )
            settings.num_inference_steps = coerce_int(
                qsettings.value("num_inference_steps", settings.num_inference_steps),
                settings.num_inference_steps,
            )

            settings.guidance_scale = coerce_float(
                qsettings.value("guidance_scale", settings.guidance_scale), settings.guidance_scale
            )

            raw_range = coerce_json(qsettings.value("guidance_start_end", settings.guidance_start_end))
            try:
                settings.guidance_start_end = cast_number_range(raw_range)
            except Exception:
                settings.guidance_start_end = [0.0, 1.0]

            raw_sched = coerce_json(qsettings.value("scheduler", SchedulerDataObject().to_dict()))
            settings.scheduler = cast_scheduler(raw_sched)

            settings.strength = coerce_float(qsettings.value("strength", settings.strength), settings.strength)

            settings.video_duration = coerce_int(
                qsettings.value("video_duration", settings.video_duration), settings.video_duration
            )
            settings.frame_rate = coerce_int(qsettings.value("frame_rate", settings.frame_rate), settings.frame_rate)

            raw_model = coerce_json(qsettings.value("model", ModelDataObject().to_dict()))
            settings.model = cast_model(raw_model)

            settings.use_torch_compile = coerce_bool(
                qsettings.value("use_torch_compile", settings.use_torch_compile),
                settings.use_torch_compile,
            )

            settings.torch_compile_max_autotune = coerce_bool(
                qsettings.value("torch_compile_max_autotune", settings.torch_compile_max_autotune),
                settings.torch_compile_max_autotune,
            )

            settings.attention_backend = coerce_str(
                qsettings.value("attention_backend", settings.attention_backend),
                settings.attention_backend,
            )

            settings.offload_strategy = coerce_str(
                qsettings.value("offload_strategy", settings.offload_strategy),
                settings.offload_strategy,
            )

            settings.group_offload_use_stream = coerce_bool(
                qsettings.value("group_offload_use_stream", settings.group_offload_use_stream),
                settings.group_offload_use_stream,
            )

            settings.group_offload_low_cpu_mem = coerce_bool(
                qsettings.value("group_offload_low_cpu_mem", settings.group_offload_low_cpu_mem),
                settings.group_offload_low_cpu_mem,
            )

            settings.streaming_decode = coerce_bool(
                qsettings.value("streaming_decode", settings.streaming_decode),
                settings.streaming_decode,
            )

            settings.ff_chunking = coerce_bool(
                qsettings.value("ff_chunking", settings.ff_chunking),
                settings.ff_chunking,
            )

            settings.ff_num_chunks = int(
                qsettings.value("ff_num_chunks", settings.ff_num_chunks)
            )

            settings.preview_decode = coerce_bool(
                qsettings.value("preview_decode", settings.preview_decode),
                settings.preview_decode,
            )

            settings.preview_time_upscale = coerce_bool(
                qsettings.value("preview_time_upscale", settings.preview_time_upscale),
                settings.preview_time_upscale,
            )

            settings.preview_space_upscale = coerce_bool(
                qsettings.value("preview_space_upscale", settings.preview_space_upscale),
                settings.preview_space_upscale,
            )

            settings.source_image_enabled = coerce_bool(
                qsettings.value("source_image_enabled", settings.source_image_enabled),
                settings.source_image_enabled,
            )

            # Migrate source_image_enabled → visual_conditions_enabled
            visual_cond_raw = qsettings.value("visual_conditions_enabled", None)
            if visual_cond_raw is not None:
                settings.visual_conditions_enabled = coerce_bool(visual_cond_raw, settings.visual_conditions_enabled)
            else:
                # First run after migration: inherit from old key
                settings.visual_conditions_enabled = settings.source_image_enabled

            settings.audio_conditioning_enabled = coerce_bool(
                qsettings.value("audio_conditioning_enabled", settings.audio_conditioning_enabled),
                settings.audio_conditioning_enabled,
            )

            settings.video_conditioning_enabled = coerce_bool(
                qsettings.value("video_conditioning_enabled", settings.video_conditioning_enabled),
                settings.video_conditioning_enabled,
            )

            settings.video_conditioning_mode = coerce_str(
                qsettings.value("video_conditioning_mode", settings.video_conditioning_mode),
                settings.video_conditioning_mode,
            )

            settings.advanced_guidance = coerce_bool(
                qsettings.value("advanced_guidance", settings.advanced_guidance),
                settings.advanced_guidance,
            )
            settings.stg_scale = coerce_float(qsettings.value("stg_scale", settings.stg_scale), settings.stg_scale)
            settings.stg_blocks = coerce_str(qsettings.value("stg_blocks", settings.stg_blocks), settings.stg_blocks)
            settings.rescale_scale = coerce_float(
                qsettings.value("rescale_scale", settings.rescale_scale), settings.rescale_scale
            )
            settings.modality_scale = coerce_float(
                qsettings.value("modality_scale", settings.modality_scale), settings.modality_scale
            )
            settings.guidance_skip_step = coerce_int(
                qsettings.value("guidance_skip_step", settings.guidance_skip_step),
                settings.guidance_skip_step,
            )

            settings.second_pass_enabled = coerce_bool(
                qsettings.value("second_pass_enabled", settings.second_pass_enabled),
                settings.second_pass_enabled,
            )

            raw_sp_model = coerce_json(qsettings.value("second_pass_model", ModelDataObject().to_dict()))
            settings.second_pass_model = cast_model(raw_sp_model)

            settings.second_pass_steps = coerce_int(
                qsettings.value("second_pass_steps", settings.second_pass_steps),
                settings.second_pass_steps,
            )

            settings.second_pass_guidance = coerce_float(
                qsettings.value("second_pass_guidance", settings.second_pass_guidance),
                settings.second_pass_guidance,
            )

            settings.video_codec = coerce_str(
                qsettings.value("video_codec", settings.video_codec),
                settings.video_codec,
            )
            settings.video_crf = coerce_int(qsettings.value("video_crf", settings.video_crf), settings.video_crf)
            settings.video_preset = coerce_str(
                qsettings.value("video_preset", settings.video_preset),
                settings.video_preset,
            )
            settings.audio_codec = coerce_str(
                qsettings.value("audio_codec", settings.audio_codec),
                settings.audio_codec,
            )
            settings.audio_bitrate_kbps = coerce_int(
                qsettings.value("audio_bitrate_kbps", settings.audio_bitrate_kbps),
                settings.audio_bitrate_kbps,
            )

            return settings
        finally:
            qsettings.endGroup()

    def save(self, qsettings: QSettings) -> None:
        qsettings.beginGroup(self.GROUP)
        try:
            qsettings.setValue("right_menu_expanded", bool(self.right_menu_expanded))
            qsettings.setValue("video_width", int(self.video_width))
            qsettings.setValue("video_height", int(self.video_height))
            qsettings.setValue("num_inference_steps", int(self.num_inference_steps))
            qsettings.setValue("guidance_scale", float(self.guidance_scale))
            qsettings.setValue("guidance_start_end", list(self.guidance_start_end))
            qsettings.setValue("scheduler", self.scheduler.to_dict())
            qsettings.setValue("strength", float(self.strength))
            qsettings.setValue("video_duration", int(self.video_duration))
            qsettings.setValue("frame_rate", int(self.frame_rate))
            qsettings.setValue("model", self.model.to_dict())
            qsettings.setValue("use_torch_compile", bool(self.use_torch_compile))
            qsettings.setValue("torch_compile_max_autotune", bool(self.torch_compile_max_autotune))
            qsettings.setValue("attention_backend", str(self.attention_backend))
            qsettings.setValue("offload_strategy", str(self.offload_strategy))
            qsettings.setValue("group_offload_use_stream", bool(self.group_offload_use_stream))
            qsettings.setValue("group_offload_low_cpu_mem", bool(self.group_offload_low_cpu_mem))
            qsettings.setValue("streaming_decode", bool(self.streaming_decode))
            qsettings.setValue("ff_chunking", bool(self.ff_chunking))
            qsettings.setValue("ff_num_chunks", int(self.ff_num_chunks))
            qsettings.setValue("preview_decode", bool(self.preview_decode))
            qsettings.setValue("preview_time_upscale", bool(self.preview_time_upscale))
            qsettings.setValue("preview_space_upscale", bool(self.preview_space_upscale))
            qsettings.setValue("source_image_enabled", bool(self.source_image_enabled))
            qsettings.setValue("visual_conditions_enabled", bool(self.visual_conditions_enabled))
            qsettings.setValue("audio_conditioning_enabled", bool(self.audio_conditioning_enabled))
            qsettings.setValue("video_conditioning_enabled", bool(self.video_conditioning_enabled))
            qsettings.setValue("video_conditioning_mode", str(self.video_conditioning_mode))
            qsettings.setValue("advanced_guidance", bool(self.advanced_guidance))
            qsettings.setValue("stg_scale", float(self.stg_scale))
            qsettings.setValue("stg_blocks", str(self.stg_blocks))
            qsettings.setValue("rescale_scale", float(self.rescale_scale))
            qsettings.setValue("modality_scale", float(self.modality_scale))
            qsettings.setValue("guidance_skip_step", int(self.guidance_skip_step))
            qsettings.setValue("second_pass_enabled", bool(self.second_pass_enabled))
            qsettings.setValue("second_pass_model", self.second_pass_model.to_dict())
            qsettings.setValue("second_pass_steps", int(self.second_pass_steps))
            qsettings.setValue("second_pass_guidance", float(self.second_pass_guidance))
            qsettings.setValue("video_codec", str(self.video_codec))
            qsettings.setValue("video_crf", int(self.video_crf))
            qsettings.setValue("video_preset", str(self.video_preset))
            qsettings.setValue("audio_codec", str(self.audio_codec))
            qsettings.setValue("audio_bitrate_kbps", int(self.audio_bitrate_kbps))
        finally:
            qsettings.endGroup()
