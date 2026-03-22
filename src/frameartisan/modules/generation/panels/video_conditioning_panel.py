from __future__ import annotations

import logging
import os

from PyQt6.QtCore import QMimeData, QSignalBlocker, Qt
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from frameartisan.modules.generation.widgets.drop_lightbox_widget import DropLightBox
from superqt import QLabeledDoubleSlider, QLabeledRangeSlider, QLabeledSlider

from frameartisan.modules.generation.generation_settings import compute_num_frames
from frameartisan.modules.generation.panels.base_panel import BasePanel

logger = logging.getLogger(__name__)


class VideoConditioningPanel(BasePanel):
    _VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, *args):
        super().__init__(*args)
        self.setAcceptDrops(True)

        self._video_path: str | None = None
        self._video_frame_count: int = 0
        self._video_fps: float = 0.0
        self._has_audio: bool = False
        self._using_video_audio: bool = False
        self._publishing = False

        self._total_frames = compute_num_frames(self.gen_settings.video_duration, self.gen_settings.frame_rate)

        self.init_ui()

        self.event_bus.subscribe("video_condition", self.on_video_condition_event)
        self.event_bus.subscribe("generation_change", self._on_generation_change)
        self.event_bus.subscribe("graph_cleared", self._on_graph_cleared)

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.setEnabled(False)
        self.enabled_checkbox.toggled.connect(self._on_enabled_changed)
        main_layout.addWidget(self.enabled_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        load_button = QPushButton("Load Video")
        load_button.clicked.connect(self._on_load_video)
        load_button.setObjectName("green_button")
        main_layout.addWidget(load_button)

        self.file_label = QLabel("No video loaded")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        main_layout.addWidget(self.file_label)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        main_layout.addWidget(self.info_label)

        self.usage_label = QLabel("")
        self.usage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.usage_label.setWordWrap(True)
        main_layout.addWidget(self.usage_label)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Replace", "replace")
        self.mode_combo.addItem("Concat (IC-LoRA)", "concat")
        self.mode_combo.addItem("Keyframe", "keyframe")
        self.mode_combo.setToolTip(
            "Replace: overwrite latent tokens in-place (hard constraint).\n"
            "Concat: append reference tokens (IC-LoRA style, for control conditioning).\n"
            "Keyframe: append as attention reference (soft guidance)."
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.mode_combo.setEnabled(False)
        mode_layout.addWidget(self.mode_combo)
        main_layout.addLayout(mode_layout)

        # Keyframe options (visible only in keyframe mode)
        self._keyframe_options_frame = QVBoxLayout()

        # Speed multiplier — skips frames to compress time (motion speeds up)
        kf_speed_layout = QHBoxLayout()
        self._kf_speed_label = QLabel("Speed:")
        kf_speed_layout.addWidget(self._kf_speed_label)
        self.keyframe_speed_combo = QComboBox()
        for val, label in [(1, "1x"), (2, "2x"), (3, "3x"), (4, "4x")]:
            self.keyframe_speed_combo.addItem(label, val)
        cur_speed = int(getattr(self.gen_settings, "keyframe_speed", 1))
        for i in range(self.keyframe_speed_combo.count()):
            if self.keyframe_speed_combo.itemData(i) == cur_speed:
                self.keyframe_speed_combo.setCurrentIndex(i)
                break
        self.keyframe_speed_combo.setToolTip(
            "Speed up motion by skipping frames.\n"
            "2x = every 2nd frame (motion plays twice as fast)."
        )
        self.keyframe_speed_combo.currentIndexChanged.connect(self._on_keyframe_speed_changed)
        kf_speed_layout.addWidget(self.keyframe_speed_combo)
        self._keyframe_options_frame.addLayout(kf_speed_layout)

        self._kf_downscale_label = QLabel("Keyframe Downsampling")
        self._kf_downscale_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._keyframe_options_frame.addWidget(self._kf_downscale_label)
        kf_spatial_layout = QHBoxLayout()
        kf_spatial_layout.addWidget(QLabel("Spatial:"))
        self.keyframe_spatial_combo = QComboBox()
        for val, label in [(1, "1x (Full)"), (2, "2x"), (4, "4x")]:
            self.keyframe_spatial_combo.addItem(label, val)
        cur_spatial = int(getattr(self.gen_settings, "keyframe_spatial_downscale", 1))
        for i in range(self.keyframe_spatial_combo.count()):
            if self.keyframe_spatial_combo.itemData(i) == cur_spatial:
                self.keyframe_spatial_combo.setCurrentIndex(i)
                break
        self.keyframe_spatial_combo.currentIndexChanged.connect(self._on_keyframe_ds_changed)
        kf_spatial_layout.addWidget(self.keyframe_spatial_combo)

        kf_temporal_layout = QHBoxLayout()
        kf_temporal_layout.addWidget(QLabel("Temporal:"))
        self.keyframe_temporal_combo = QComboBox()
        for val, label in [(1, "All frames"), (2, "Every 2nd"), (4, "Every 4th"), (6, "Every 6th")]:
            self.keyframe_temporal_combo.addItem(label, val)
        cur_temporal = int(getattr(self.gen_settings, "keyframe_temporal_stride", 1))
        for i in range(self.keyframe_temporal_combo.count()):
            if self.keyframe_temporal_combo.itemData(i) == cur_temporal:
                self.keyframe_temporal_combo.setCurrentIndex(i)
                break
        self.keyframe_temporal_combo.currentIndexChanged.connect(self._on_keyframe_ds_changed)
        kf_temporal_layout.addWidget(self.keyframe_temporal_combo)

        self._keyframe_options_frame.addLayout(kf_spatial_layout)
        self._keyframe_options_frame.addLayout(kf_temporal_layout)
        main_layout.addLayout(self._keyframe_options_frame)

        self._kf_spatial_label = kf_spatial_layout.itemAt(0).widget()
        self._kf_temporal_label = kf_temporal_layout.itemAt(0).widget()
        self._update_keyframe_options_visibility()

        # Start frame slider
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Start Frame:"))
        self.frame_slider = QLabeledSlider()
        self.frame_slider.setRange(1, max(self._total_frames, 1))
        self.frame_slider.setValue(1)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setOrientation(Qt.Orientation.Horizontal)
        self.frame_slider.setToolTip("Pixel frame where the video starts (1-based)")
        self.frame_slider.valueChanged.connect(self._on_settings_changed)
        self.frame_slider.setEnabled(False)
        frame_layout.addWidget(self.frame_slider)

        self.last_frame_btn = QPushButton("Last")
        self.last_frame_btn.setFixedWidth(40)
        self.last_frame_btn.setCheckable(True)
        self.last_frame_btn.setToolTip("Pin video to end at last frame")
        self.last_frame_btn.toggled.connect(self._on_last_toggled)
        self.last_frame_btn.setEnabled(False)
        frame_layout.addWidget(self.last_frame_btn)
        main_layout.addLayout(frame_layout)

        # Strength slider
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))
        self.strength_slider = QLabeledDoubleSlider()
        self.strength_slider.setRange(0.0, 1.0)
        self.strength_slider.setSingleStep(0.05)
        self.strength_slider.setValue(1.0)
        self.strength_slider.setOrientation(Qt.Orientation.Horizontal)
        self.strength_slider.valueChanged.connect(self._on_settings_changed)
        self.strength_slider.setEnabled(False)
        strength_layout.addWidget(self.strength_slider)
        main_layout.addLayout(strength_layout)

        # Attention slider
        attention_layout = QHBoxLayout()
        attention_layout.addWidget(QLabel("Attention:"))
        self.attention_slider = QLabeledDoubleSlider()
        self.attention_slider.setRange(0.0, 1.0)
        self.attention_slider.setSingleStep(0.05)
        self.attention_slider.setValue(1.0)
        self.attention_slider.setOrientation(Qt.Orientation.Horizontal)
        self.attention_slider.setToolTip(
            "Controls how strongly the model attends to this condition.\n"
            "Only affects non-first-frame conditions (keyframes).\n"
            "1.0 = full attention, 0.0 = ignored"
        )
        self.attention_slider.valueChanged.connect(self._on_settings_changed)
        self.attention_slider.setEnabled(False)
        attention_layout.addWidget(self.attention_slider)
        main_layout.addLayout(attention_layout)

        # Source frame range slider
        source_range_layout = QHBoxLayout()
        source_range_layout.addWidget(QLabel("Source Frames:"))
        self.source_range_slider = QLabeledRangeSlider()
        self.source_range_slider.setRange(1, 1)
        self.source_range_slider.setValue((1, 1))
        self.source_range_slider.setOrientation(Qt.Orientation.Horizontal)
        self.source_range_slider.setToolTip("Range of source video frames to use (1-based)")
        self.source_range_slider.valueChanged.connect(self._on_settings_changed)
        self.source_range_slider.setEnabled(False)
        source_range_layout.addWidget(self.source_range_slider)
        main_layout.addLayout(source_range_layout)

        self.use_audio_checkbox = QCheckBox("Use video audio")
        self.use_audio_checkbox.setEnabled(False)
        self.use_audio_checkbox.setToolTip("Video has no audio track")
        self.use_audio_checkbox.toggled.connect(self._on_use_audio_toggled)
        main_layout.addWidget(self.use_audio_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        self.remove_button = QPushButton("Remove Video")
        self.remove_button.setObjectName("red_button")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self._on_remove_video)
        main_layout.addWidget(self.remove_button)

        main_layout.addStretch()
        self.setLayout(main_layout)

        self._drop_lightbox = DropLightBox(self)
        self._drop_lightbox.setText("Drop video here")

    @property
    def pixel_frame_index(self) -> int:
        if self.last_frame_btn.isChecked():
            return -1
        return self.frame_slider.value()

    @property
    def source_frame_range(self) -> tuple[int, int]:
        """1-based inclusive range of source video frames to use."""
        val = self.source_range_slider.value()
        return (int(val[0]), int(val[1]))

    @property
    def video_mode(self) -> str:
        return self.mode_combo.currentData() or "replace"

    def _update_keyframe_options_visibility(self) -> None:
        visible = self.mode_combo.currentData() == "keyframe"
        self._kf_speed_label.setVisible(visible)
        self.keyframe_speed_combo.setVisible(visible)
        self._kf_downscale_label.setVisible(visible)
        self.keyframe_spatial_combo.setVisible(visible)
        self.keyframe_temporal_combo.setVisible(visible)
        self._kf_spatial_label.setVisible(visible)
        self._kf_temporal_label.setVisible(visible)

    def _on_mode_changed(self) -> None:
        self._update_keyframe_options_visibility()
        self._on_settings_changed()

    def _on_keyframe_speed_changed(self) -> None:
        speed = self.keyframe_speed_combo.currentData()
        if speed is not None:
            self.event_bus.publish("generation_change", {"attr": "keyframe_speed", "value": int(speed)})

    def _on_keyframe_ds_changed(self) -> None:
        spatial = self.keyframe_spatial_combo.currentData()
        temporal = self.keyframe_temporal_combo.currentData()
        if spatial is not None:
            self.event_bus.publish("generation_change", {"attr": "keyframe_spatial_downscale", "value": int(spatial)})
        if temporal is not None:
            self.event_bus.publish("generation_change", {"attr": "keyframe_temporal_stride", "value": int(temporal)})

    def _on_enabled_changed(self, checked: bool) -> None:
        self.event_bus.publish("generation_change", {"attr": "video_conditioning_enabled", "value": checked})

    @staticmethod
    def _get_video_url(mime: QMimeData) -> str | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if os.path.splitext(path)[1].lower() in VideoConditioningPanel._VIDEO_EXTENSIONS:
                return path
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if self._get_video_url(event.mimeData()):
            event.acceptProposedAction()
            self._drop_lightbox.show()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if self._get_video_url(event.mimeData()):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # type: ignore[override]
        self._drop_lightbox.hide()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._drop_lightbox.hide()
        path = self._get_video_url(event.mimeData())
        if not path:
            return
        self._set_video(path)
        sf_start, sf_end = self.source_frame_range
        self.event_bus.publish(
            "video_condition",
            {
                "action": "add",
                "video_path": path,
                "pixel_frame_index": self.pixel_frame_index,
                "strength": self.strength_slider.value(),
                "attention_scale": self.attention_slider.value(),
                "mode": self.video_mode,
                "source_frame_start": sf_start,
                "source_frame_end": sf_end,
            },
        )
        event.acceptProposedAction()

    def _on_load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if path:
            self._set_video(path)
            sf_start, sf_end = self.source_frame_range
            self.event_bus.publish(
                "video_condition",
                {
                    "action": "add",
                    "video_path": path,
                    "pixel_frame_index": self.pixel_frame_index,
                    "strength": self.strength_slider.value(),
                    "mode": self.video_mode,
                    "source_frame_start": sf_start,
                    "source_frame_end": sf_end,
                },
            )

    def _set_video(self, path: str) -> None:
        self._video_path = path
        self.file_label.setText(os.path.basename(path))

        # Probe video for frame count, fps, and audio presence
        self._video_frame_count, self._video_fps, self._has_audio = self._probe_video(path)

        if self._video_fps > 0:
            duration = self._video_frame_count / self._video_fps
            self.info_label.setText(
                f"Video: {self._video_frame_count} frames @ {self._video_fps:.1f}fps ({duration:.1f}s)"
            )
        else:
            self.info_label.setText(f"Video: {self._video_frame_count} frames")

        # Configure source frame range slider for the loaded video
        if self._video_frame_count > 0:
            blocker = QSignalBlocker(self.source_range_slider)
            try:
                self.source_range_slider.setRange(1, self._video_frame_count)
                self.source_range_slider.setValue((1, self._video_frame_count))
            finally:
                del blocker

        self._update_usage_label()

        self.enabled_checkbox.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.last_frame_btn.setEnabled(True)
        self.strength_slider.setEnabled(True)
        self.attention_slider.setEnabled(True)
        self.source_range_slider.setEnabled(self._video_frame_count > 1)
        self.mode_combo.setEnabled(True)
        self.remove_button.setEnabled(True)

        # Update audio checkbox state
        if self._has_audio:
            self.use_audio_checkbox.setEnabled(True)
            self.use_audio_checkbox.setToolTip("Extract and use the audio track from this video")
        else:
            self.use_audio_checkbox.setEnabled(False)
            self.use_audio_checkbox.setToolTip("Video has no audio track")
            if self.use_audio_checkbox.isChecked():
                blocker = QSignalBlocker(self.use_audio_checkbox)
                try:
                    self.use_audio_checkbox.setChecked(False)
                finally:
                    del blocker
                # Clean up audio if we were using video audio
                if self._using_video_audio:
                    self._using_video_audio = False
                    self.event_bus.publish("audio_condition", {"action": "remove", "from_video": True})

        # Re-extract audio if checkbox is already checked (video replaced)
        if self.use_audio_checkbox.isChecked() and self._has_audio:
            self._publish_video_audio()

        if not self.enabled_checkbox.isChecked():
            self.enabled_checkbox.setChecked(True)
        else:
            self._on_enabled_changed(True)

    @staticmethod
    def _probe_video(path: str) -> tuple[int, float, bool]:
        try:
            import av

            with av.open(path) as container:
                stream = container.streams.video[0]
                frame_count = stream.frames
                if frame_count == 0:
                    # Some containers don't report frame count; decode to count
                    frame_count = sum(1 for _ in container.decode(video=0))
                fps = float(stream.average_rate) if stream.average_rate else 0.0
                has_audio = len(container.streams.audio) > 0
                return frame_count, fps, has_audio
        except Exception:
            return 0, 0.0, False

    @staticmethod
    def _extract_audio(video_path: str, output_path: str) -> bool:
        try:
            import av

            with av.open(video_path) as inp:
                if len(inp.streams.audio) == 0:
                    return False
                audio_stream = inp.streams.audio[0]

                with av.open(output_path, "w") as out:
                    out_stream = out.add_stream("pcm_s16le", rate=audio_stream.rate)
                    out_stream.layout = "mono" if audio_stream.channels == 1 else "stereo"
                    for frame in inp.decode(audio=0):
                        for packet in out_stream.encode(frame):
                            out.mux(packet)
                    for packet in out_stream.encode(None):
                        out.mux(packet)
            return True
        except Exception:
            logger.exception("Failed to extract audio from video")
            return False

    def _video_audio_trim(self) -> tuple[float | None, float | None]:
        """Compute audio trim times (seconds) matching the selected source frame range."""
        if self._video_fps <= 0:
            return None, None
        sf_start, sf_end = self.source_frame_range
        trim_start = (sf_start - 1) / self._video_fps
        trim_end = sf_end / self._video_fps
        return trim_start, trim_end

    def _publish_video_audio(self) -> None:
        if self._video_path is None:
            return
        output_path = os.path.join(self.directories.temp_path, "video_audio.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self._extract_audio(self._video_path, output_path):
            file_size = os.path.getsize(output_path)
            trim_start, trim_end = self._video_audio_trim()
            logger.debug(
                "Video audio extracted: %s → %s (%d bytes), trim=%.2f-%.2fs",
                self._video_path, output_path, file_size,
                trim_start or 0, trim_end or 0,
            )
            self._using_video_audio = True
            self.event_bus.publish(
                "audio_condition",
                {
                    "action": "add",
                    "audio_path": output_path,
                    "from_video": True,
                    "trim_start_s": trim_start,
                    "trim_end_s": trim_end,
                },
            )
        else:
            logger.warning("Video audio extraction failed for %s", self._video_path)
            self._using_video_audio = False
            blocker = QSignalBlocker(self.use_audio_checkbox)
            try:
                self.use_audio_checkbox.setChecked(False)
            finally:
                del blocker

    def _on_use_audio_toggled(self, checked: bool) -> None:
        if checked:
            if self._video_path and self._has_audio:
                self._publish_video_audio()
        else:
            if self._using_video_audio:
                self._using_video_audio = False
                self.event_bus.publish("audio_condition", {"action": "remove", "from_video": True})

    def _update_usage_label(self) -> None:
        if self._video_frame_count == 0:
            self.usage_label.setText("")
            return

        sf_start, sf_end = self.source_frame_range
        source_count = sf_end - sf_start + 1

        start = self.frame_slider.value()
        available = self._total_frames - (start - 1)
        available = max(available, 0)
        used = min(source_count, available)

        # Align to VAE temporal grid (8n+1) — pad up
        import math

        if used > 0:
            aligned = math.ceil((used - 1) / 8) * 8 + 1
            if aligned <= 0:
                aligned = 1
        else:
            aligned = 0

        if aligned < source_count:
            self.usage_label.setText(f"Using: {aligned} of {source_count} frames (trimmed to fit)")
        else:
            self.usage_label.setText(f"Using: all {aligned} frames")

    def _on_last_toggled(self, checked: bool) -> None:
        self.frame_slider.setEnabled(not checked)
        if checked:
            blocker = QSignalBlocker(self.frame_slider)
            try:
                self.frame_slider.setValue(self.frame_slider.maximum())
            finally:
                del blocker
        self._on_settings_changed()

    def _on_settings_changed(self) -> None:
        if self._publishing:
            return
        self._publishing = True
        try:
            self._update_usage_label()
            if self._video_path is not None:
                sf_start, sf_end = self.source_frame_range
                self.event_bus.publish(
                    "video_condition",
                    {
                        "action": "update_settings",
                        "pixel_frame_index": self.pixel_frame_index,
                        "strength": self.strength_slider.value(),
                        "attention_scale": self.attention_slider.value(),
                        "mode": self.video_mode,
                        "source_frame_start": sf_start,
                        "source_frame_end": sf_end,
                    },
                )
                # Sync audio trim to match source frame range
                if self._using_video_audio:
                    trim_start, trim_end = self._video_audio_trim()
                    self.event_bus.publish(
                        "audio_condition",
                        {
                            "action": "update_trim",
                            "trim_start_s": trim_start,
                            "trim_end_s": trim_end,
                        },
                    )
        finally:
            self._publishing = False

    def _on_remove_video(self) -> None:
        # Clean up video audio before clearing state
        if self._using_video_audio:
            self._using_video_audio = False
            self.event_bus.publish("audio_condition", {"action": "remove", "from_video": True})

        self._video_path = None
        self._video_frame_count = 0
        self._video_fps = 0.0
        self._has_audio = False
        self.file_label.setText("No video loaded")
        self.info_label.setText("")
        self.usage_label.setText("")

        self.frame_slider.setEnabled(False)
        self.last_frame_btn.setEnabled(False)
        self.strength_slider.setEnabled(False)
        self.attention_slider.setEnabled(False)
        self.source_range_slider.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.enabled_checkbox.setEnabled(False)
        self.use_audio_checkbox.setEnabled(False)
        self.use_audio_checkbox.setToolTip("Video has no audio track")

        blocker = QSignalBlocker(self.enabled_checkbox)
        try:
            self.enabled_checkbox.setChecked(False)
        finally:
            del blocker

        blocker2 = QSignalBlocker(self.use_audio_checkbox)
        try:
            self.use_audio_checkbox.setChecked(False)
        finally:
            del blocker2

        self.event_bus.publish("video_condition", {"action": "remove"})

    def _update_total_frames(self, data: dict | None = None) -> None:
        # Use event data directly (event bus order not guaranteed)
        duration = self.gen_settings.video_duration
        fps = self.gen_settings.frame_rate
        if data is not None:
            attr = data.get("attr")
            value = data.get("value")
            if attr == "video_duration" and value is not None:
                duration = value
            elif attr == "frame_rate" and value is not None:
                fps = value
        new_total = compute_num_frames(duration, fps)
        if new_total == self._total_frames:
            return
        self._total_frames = new_total
        self.frame_slider.setRange(1, max(self._total_frames, 1))
        if self.last_frame_btn.isChecked():
            blocker = QSignalBlocker(self.frame_slider)
            try:
                self.frame_slider.setValue(self._total_frames)
            finally:
                del blocker
        self._update_usage_label()

    def _on_generation_change(self, data: dict) -> None:
        attr = data.get("attr")
        if attr in ("video_duration", "frame_rate"):
            self._update_total_frames(data)

    def _on_graph_cleared(self, _data: dict) -> None:
        if self._video_path is not None:
            self._video_path = None
            self._video_frame_count = 0
            self._video_fps = 0.0
            self._has_audio = False
            self._using_video_audio = False
            self.file_label.setText("No video loaded")
            self.info_label.setText("")
            self.usage_label.setText("")
            self.frame_slider.setEnabled(False)
            self.last_frame_btn.setEnabled(False)
            self.strength_slider.setEnabled(False)
            self.source_range_slider.setEnabled(False)
            self.mode_combo.setEnabled(False)
            self.remove_button.setEnabled(False)
            self.enabled_checkbox.setEnabled(False)
            self.use_audio_checkbox.setEnabled(False)
            self.use_audio_checkbox.setToolTip("Video has no audio track")
            blocker = QSignalBlocker(self.enabled_checkbox)
            try:
                self.enabled_checkbox.setChecked(False)
            finally:
                del blocker
            blocker2 = QSignalBlocker(self.use_audio_checkbox)
            try:
                self.use_audio_checkbox.setChecked(False)
            finally:
                del blocker2

    def on_video_condition_event(self, data: dict) -> None:
        action = data.get("action")
        if action == "add":
            path = data.get("video_path")
            if path and path != self._video_path:
                self._set_video(path)
        elif action == "remove":
            if self._video_path is not None:
                self._on_remove_video()
