"""Video editor dialog — trim, scale, and prepare source videos."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from superqt import QLabeledRangeSlider

from frameartisan.app.base_dialog import BaseDialog
from frameartisan.modules.generation.video_editor.video_editor_controls import VideoEditorControls
from frameartisan.modules.generation.video_editor.video_process_thread import VideoProcessThread
from frameartisan.modules.generation.widgets.video_simple_widget import VideoSimpleWidget

if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject
    from frameartisan.app.preferences import PreferencesObject

logger = logging.getLogger(__name__)


class VideoEditorDialog(BaseDialog):
    def __init__(
        self,
        dialog_type: str,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        video_path: str,
        gen_width: int,
        gen_height: int,
        editor_state: dict | None = None,
    ):
        super().__init__(dialog_type, directories, preferences)
        self.setWindowTitle("Video Editor")
        self.setMinimumSize(640, 500)

        self._settings = QSettings("ZCode", "FrameArtisan_VideoEditor")
        self._video_path = video_path
        self._gen_width = gen_width
        self._gen_height = gen_height
        self._video_info: dict | None = None
        self._dialog_busy = False
        self._process_thread: VideoProcessThread | None = None
        self._trim_enforcing = False  # prevents recursive position enforcement

        self._init_ui()

        if video_path:
            self.load_video(video_path, editor_state=editor_state)

        # Restore saved geometry
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def _init_ui(self) -> None:
        # Toolbar area — reserved for future preprocess tools
        self._toolbar_layout = QHBoxLayout()
        self.main_layout.addLayout(self._toolbar_layout)

        # Video info label (source resolution, FPS, codec)
        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #8e8e8e; font-size: 11px;")
        self.main_layout.addWidget(self._info_label)

        # Video preview
        self._video_widget = VideoSimpleWidget(self.directories, self.preferences)
        self._video_widget.setMinimumHeight(250)
        self.main_layout.addWidget(self._video_widget, 1)

        # Frame counter
        self._frame_label = QLabel("Frame: — / —")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet("color: #eff0f1; font-size: 11px;")
        self.main_layout.addWidget(self._frame_label)

        # Trim section
        trim_layout = QVBoxLayout()
        trim_layout.setContentsMargins(8, 4, 8, 4)

        trim_header = QHBoxLayout()
        trim_header.addWidget(QLabel("Trim Range (frames):"))
        trim_header.addStretch()
        self._duration_label = QLabel()
        trim_header.addWidget(self._duration_label)
        trim_layout.addLayout(trim_header)

        self._trim_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self._trim_slider.setRange(1, 1)
        self._trim_slider.setValue((1, 1))
        self._trim_slider.valueChanged.connect(self._on_trim_changed)
        trim_layout.addWidget(self._trim_slider)

        # Trim navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()

        goto_start_btn = QPushButton("Go to Start")
        goto_start_btn.setFixedWidth(80)
        goto_start_btn.setToolTip("Seek to trim start (Home)")
        goto_start_btn.clicked.connect(self._seek_to_trim_start)
        nav_layout.addWidget(goto_start_btn)

        step_back_btn = QPushButton("|<")
        step_back_btn.setFixedWidth(32)
        step_back_btn.setToolTip("Step start back 1 frame")
        step_back_btn.clicked.connect(lambda: self._step_trim(-1, "start"))
        nav_layout.addWidget(step_back_btn)

        step_fwd_btn = QPushButton(">|")
        step_fwd_btn.setFixedWidth(32)
        step_fwd_btn.setToolTip("Step end forward 1 frame")
        step_fwd_btn.clicked.connect(lambda: self._step_trim(1, "end"))
        nav_layout.addWidget(step_fwd_btn)

        goto_end_btn = QPushButton("Go to End")
        goto_end_btn.setFixedWidth(80)
        goto_end_btn.setToolTip("Seek to trim end (End)")
        goto_end_btn.clicked.connect(self._seek_to_trim_end)
        nav_layout.addWidget(goto_end_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(50)
        reset_btn.setToolTip("Reset trim to full range and resolution to default")
        reset_btn.clicked.connect(self._on_reset)
        nav_layout.addWidget(reset_btn)

        nav_layout.addStretch()
        trim_layout.addLayout(nav_layout)

        self.main_layout.addLayout(trim_layout)

        # Controls (resolution + apply)
        self._controls = VideoEditorControls()
        self._controls.set_generation_resolution(self._gen_width, self._gen_height)
        self._controls.apply_clicked.connect(self._on_apply)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(8, 4, 8, 8)
        controls_layout.addWidget(self._controls)
        self.main_layout.addLayout(controls_layout)

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self._step_frame_back)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._step_frame_forward)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self).activated.connect(self._seek_to_trim_start)
        QShortcut(QKeySequence(Qt.Key.Key_End), self).activated.connect(self._seek_to_trim_end)

    def load_video(self, path: str, editor_state: dict | None = None) -> None:
        """Probe video and configure the dialog.

        Args:
            editor_state: Optional dict with ``trim_start``, ``trim_end``,
                ``resolution`` to restore from a previous Apply.
        """
        self._video_path = path
        self._video_info = self._probe_video(path)
        if self._video_info is None:
            return

        fc = self._video_info["frame_count"]
        fps = self._video_info["fps"]
        w = self._video_info["width"]
        h = self._video_info["height"]

        # Video info display
        self._info_label.setText(f"{w}\u00d7{h}  |  {fps:.2f} fps  |  {fc} frames")

        # Configure trim slider — restore previous trim if available
        self._trim_slider.setRange(1, max(fc, 1))
        if editor_state and "trim_start" in editor_state and "trim_end" in editor_state:
            start = max(1, min(editor_state["trim_start"], fc))
            end = max(start, min(editor_state["trim_end"], fc))
            self._trim_slider.setValue((start, end))
        else:
            self._trim_slider.setValue((1, fc))

        # Configure resolution presets
        self._controls.set_source_resolution(
            self._video_info["width"], self._video_info["height"],
        )

        # Restore resolution selection if available
        if editor_state and "resolution" in editor_state:
            res = editor_state["resolution"]
            for i in range(self._controls.resolution_combo.count()):
                if self._controls.resolution_combo.itemData(i) == tuple(res):
                    self._controls.resolution_combo.setCurrentIndex(i)
                    break

        # Load into video player
        self._video_widget.load_video(path, autoplay=False)

        # Connect position monitoring for trim enforcement
        player = self._video_widget._player
        if player is not None:
            try:
                player.positionChanged.disconnect(self._on_player_position)
            except (TypeError, RuntimeError):
                pass
            player.positionChanged.connect(self._on_player_position)

        self._update_duration_label()
        logger.info(
            "Video editor loaded: %s (%d frames, %.1f fps, %dx%d)",
            path, fc, fps, self._video_info["width"], self._video_info["height"],
        )

    def _frame_to_ms(self, frame: int) -> int:
        """Convert 1-based frame number to milliseconds."""
        if self._video_info is None:
            return 0
        return int((frame - 1) / self._video_info["fps"] * 1000)

    def _on_player_position(self, position_ms: int) -> None:
        """Enforce trim boundaries and update frame counter."""
        if self._video_info is None:
            return

        # Update frame counter
        fps = self._video_info["fps"]
        current_frame = int(position_ms / 1000.0 * fps) + 1
        total_frames = self._video_info["frame_count"]
        start, end = self._trim_slider.value()
        self._frame_label.setText(
            f"Frame: {current_frame} / {total_frames}  (trim: {start}\u2013{end})"
        )

        # Enforce trim boundaries
        if self._trim_enforcing:
            return
        player = self._video_widget._player
        if player is None:
            return

        start_ms = self._frame_to_ms(start)
        end_ms = self._frame_to_ms(end + 1)  # end is inclusive, so +1

        if position_ms < start_ms:
            self._trim_enforcing = True
            player.setPosition(start_ms)
            self._trim_enforcing = False
        elif position_ms >= end_ms:
            self._trim_enforcing = True
            if self._video_widget._looping:
                player.setPosition(start_ms)
            else:
                player.pause()
                player.setPosition(start_ms)
            self._trim_enforcing = False

    def _on_trim_changed(self) -> None:
        self._update_duration_label()
        # Seek to trim start when range changes
        player = self._video_widget._player
        if player is not None and self._video_info is not None:
            start, _ = self._trim_slider.value()
            player.setPosition(self._frame_to_ms(start))

    def _update_duration_label(self) -> None:
        if self._video_info is None:
            self._duration_label.setText("")
            return
        start, end = self._trim_slider.value()
        frame_count = end - start + 1
        duration = frame_count / self._video_info["fps"]
        self._duration_label.setText(f"{duration:.1f}s ({frame_count} frames)")

    def _step_trim(self, delta: int, handle: str) -> None:
        start, end = self._trim_slider.value()
        if handle == "start":
            start = max(self._trim_slider.minimum(), min(start + delta, end))
        else:
            end = min(self._trim_slider.maximum(), max(end + delta, start))
        self._trim_slider.setValue((start, end))

    # ------------------------------------------------------------------
    # Navigation & keyboard shortcuts
    # ------------------------------------------------------------------

    def _toggle_play_pause(self) -> None:
        player = self._video_widget._player
        if player is None:
            return
        from PyQt6.QtMultimedia import QMediaPlayer

        if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            player.pause()
        else:
            # If at or past trim end, restart from trim start
            start, end = self._trim_slider.value()
            if self._video_info and player.position() >= self._frame_to_ms(end + 1):
                player.setPosition(self._frame_to_ms(start))
            player.play()

    def _step_frame_back(self) -> None:
        """Step one frame backward (Left arrow)."""
        player = self._video_widget._player
        if player is None or self._video_info is None:
            return
        fps = self._video_info["fps"]
        frame_ms = int(1000 / fps)
        new_pos = max(self._frame_to_ms(self._trim_slider.value()[0]), player.position() - frame_ms)
        player.pause()
        player.setPosition(new_pos)

    def _step_frame_forward(self) -> None:
        """Step one frame forward (Right arrow)."""
        player = self._video_widget._player
        if player is None or self._video_info is None:
            return
        fps = self._video_info["fps"]
        frame_ms = int(1000 / fps)
        end_ms = self._frame_to_ms(self._trim_slider.value()[1] + 1)
        new_pos = min(end_ms - 1, player.position() + frame_ms)
        player.pause()
        player.setPosition(new_pos)

    def _seek_to_trim_start(self) -> None:
        """Seek to the trim start point (Home)."""
        player = self._video_widget._player
        if player is not None and self._video_info is not None:
            start, _ = self._trim_slider.value()
            player.setPosition(self._frame_to_ms(start))

    def _seek_to_trim_end(self) -> None:
        """Seek to the trim end point (End)."""
        player = self._video_widget._player
        if player is not None and self._video_info is not None:
            _, end = self._trim_slider.value()
            player.setPosition(self._frame_to_ms(end))

    def _on_reset(self) -> None:
        """Reset trim to full range and resolution to first preset."""
        if self._video_info is None:
            return
        fc = self._video_info["frame_count"]
        self._trim_slider.setValue((1, fc))
        self._controls.resolution_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _on_apply(self) -> None:
        if self._dialog_busy or self._video_info is None:
            return

        self._dialog_busy = True
        self._controls.set_busy(True)

        start, end = self._trim_slider.value()
        # Convert 1-based inclusive to 0-based [start, end)
        trim_start = start - 1
        trim_end = end  # 1-based inclusive end → 0-based exclusive

        w, h = self._controls.get_output_resolution()
        fps = self._video_info["fps"]

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(
            str(self.directories.temp_path),
            f"video_edit_{timestamp}.mp4",
        )

        self._process_thread = VideoProcessThread(
            source_path=self._video_path,
            output_path=output_path,
            trim_start_frame=trim_start,
            trim_end_frame=trim_end,
            output_width=w,
            output_height=h,
            fps=fps,
            include_audio=self._video_info.get("has_audio", False),
        )
        self._process_thread.progress.connect(self._on_process_progress)
        self._process_thread.finished_path.connect(self._on_process_finished)
        self._process_thread.error.connect(self._on_process_error)
        self._process_thread.finished.connect(self._on_thread_done)
        self._process_thread.start()

    def _on_process_progress(self, percent: int) -> None:
        self._controls.set_progress(percent)

    def _on_process_finished(self, output_path: str) -> None:
        start, end = self._trim_slider.value()
        w, h = self._controls.get_output_resolution()
        self.event_bus.publish(
            "video_condition",
            {
                "action": "update_video",
                "video_path": output_path,
                "editor_state": {
                    "trim_start": start,
                    "trim_end": end,
                    "resolution": (w, h),
                },
            },
        )

    def _on_process_error(self, message: str) -> None:
        self.event_bus.publish(
            "show_snackbar",
            {"action": "show", "message": f"Video processing failed: {message}"},
        )

    def _on_thread_done(self) -> None:
        """Clean up thread references after thread finishes (success or error)."""
        if self._process_thread is not None:
            self._process_thread.progress.disconnect(self._on_process_progress)
            self._process_thread.finished_path.disconnect(self._on_process_finished)
            self._process_thread.error.disconnect(self._on_process_error)
            self._process_thread.finished.disconnect(self._on_thread_done)
            self._process_thread = None

        self._dialog_busy = False
        self._controls.set_busy(False)

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    @staticmethod
    def _probe_video(path: str) -> dict | None:
        """Probe video file for metadata. Returns None on failure."""
        try:
            import av

            with av.open(path) as container:
                stream = container.streams.video[0]
                frame_count = stream.frames
                if frame_count == 0:
                    # Some containers don't report frame count; decode to count
                    frame_count = sum(1 for _ in container.decode(video=0))
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                width = stream.width
                height = stream.height
                has_audio = len(container.streams.audio) > 0
            return {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "has_audio": has_audio,
            }
        except Exception as e:
            logger.warning("Failed to probe video %s: %s", path, e)
            return None
