from __future__ import annotations

import hashlib
import logging
import shutil
from importlib.resources import files
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QMimeData, Qt, QTime, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frameartisan.modules.generation.widgets.drop_lightbox_widget import DropLightBox


try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget

    _QT_MULTIMEDIA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on platform/runtime Qt plugins
    QAudioOutput = None  # type: ignore[assignment]
    QMediaPlayer = None  # type: ignore[assignment]
    QVideoWidget = None  # type: ignore[assignment]
    _QT_MULTIMEDIA_AVAILABLE = False


if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject
    from frameartisan.app.preferences import PreferencesObject


logger = logging.getLogger(__name__)

_APP_BG = "#29292b"
_CONTROLS_BG = "#252629"
_BORDER = "#1a1b1d"


def _ms_to_str(ms: int) -> str:
    t = QTime(0, 0).addMSecs(max(0, ms))
    return t.toString("h:mm:ss") if t.hour() > 0 else t.toString("m:ss")


class VideoSimpleWidget(QWidget):
    """Video viewer with play/pause, seek and volume controls."""

    def __init__(
        self,
        directories: Optional[DirectoriesObject] = None,
        preferences: Optional[PreferencesObject] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.directories = directories
        self.preferences = preferences

        self.setObjectName("video_simple_widget")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"VideoSimpleWidget {{ background-color: {_APP_BG}; }}")
        self.setAcceptDrops(True)

        self._drop_callback: Callable[[str], None] | None = None
        self._drop_lightbox = DropLightBox(self)
        self._drop_lightbox.setText("Drop video here")

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

        self._player: QMediaPlayer | None = None
        self._audio_output: QAudioOutput | None = None
        self._video_widget: QVideoWidget | None = None
        self._stack: QStackedWidget | None = None
        self._seeking = False
        self._looping = False

        # Control widgets (created in _init_multimedia or remain None)
        self._play_btn: QPushButton | None = None
        self._loop_btn: QPushButton | None = None
        self._save_btn: QPushButton | None = None
        self._time_label: QLabel | None = None
        self._seeker: QSlider | None = None
        self._volume_slider: QSlider | None = None
        self._controls_bar: QWidget | None = None

        if _QT_MULTIMEDIA_AVAILABLE:
            self._init_multimedia()
        else:
            self._init_placeholder()

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def set_drop_callback(self, callback: Callable[[str], None]) -> None:
        self._drop_callback = callback

    @staticmethod
    def _get_mp4_url(mime: QMimeData) -> str | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if path.lower().endswith(".mp4"):
                return path
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if self._get_mp4_url(event.mimeData()):
            event.acceptProposedAction()
            self._drop_lightbox.show()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if self._get_mp4_url(event.mimeData()):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # type: ignore[override]
        self._drop_lightbox.hide()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._drop_lightbox.hide()
        path = self._get_mp4_url(event.mimeData())
        if not path:
            return
        if self._drop_callback is not None:
            self._drop_callback(path)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_multimedia(self) -> None:
        assert QMediaPlayer is not None
        assert QAudioOutput is not None
        assert QVideoWidget is not None

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(1.0)
        self._player.setAudioOutput(self._audio_output)

        # --- video area via a stack so the app-background shows when idle ---
        self._stack = QStackedWidget(self)
        self._stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Page 0 – no-video placeholder
        idle_page = QLabel("No video loaded")
        idle_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        idle_page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        idle_page.setStyleSheet(f"color: #6e6e6e; background-color: {_APP_BG};")
        self._stack.addWidget(idle_page)  # index 0

        # Page 1 – actual video
        self._video_widget = QVideoWidget(self)
        self._video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if hasattr(self._video_widget, "setAspectRatioMode"):
            try:
                self._video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
            except Exception:
                pass
        self._stack.addWidget(self._video_widget)  # index 1
        self._stack.setCurrentIndex(0)

        self._layout.addWidget(self._stack)

        # --- controls bar ---
        self._controls_bar = self._build_controls()
        self._layout.addWidget(self._controls_bar)

        # --- wire player signals ---
        self._player.setVideoSink(self._video_widget.videoSink())
        self._player.errorOccurred.connect(self._on_player_error)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._set_controls_enabled(False)

    def _build_controls(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("video_controls")
        bar.setFixedHeight(36)
        bar.setStyleSheet(
            f"QWidget#video_controls {{ background-color: {_CONTROLS_BG}; border-top: 1px solid {_BORDER}; }}"
        )

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Play / pause
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(26, 26)
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._on_play_pause_clicked)
        layout.addWidget(self._play_btn)

        # Loop toggle
        self._loop_btn = QPushButton()
        self._loop_btn.setFixedSize(26, 26)
        self._loop_btn.setToolTip("Loop")
        self._loop_btn.setCheckable(True)
        loop_icon = files("frameartisan.theme.icons").joinpath("loop.png")
        self._loop_btn.setStyleSheet(
            f"QPushButton {{ border: none; background: transparent;"
            f"  image: url({loop_icon}); opacity: 0.5; }}"
            f"QPushButton:checked {{ image: url({loop_icon}); opacity: 1.0;"
            f"  border: 1px solid #58a6ff; border-radius: 4px; background: rgba(88,166,255,0.15); }}"
        )
        self._loop_btn.clicked.connect(self._on_loop_clicked)
        layout.addWidget(self._loop_btn)

        # Save
        self._save_btn = QPushButton("Save")
        self._save_btn.setFixedSize(42, 26)
        self._save_btn.setToolTip("Save video to outputs (Ctrl+S)")
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._save_btn.setEnabled(False)
        layout.addWidget(self._save_btn)

        self._save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self._save_shortcut.activated.connect(self._on_save_clicked)

        # Time label — fixed width to prevent layout jitter
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFixedWidth(100)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._time_label.setStyleSheet("color: #eff0f1; background: transparent;")
        layout.addWidget(self._time_label)

        # Seeker
        self._seeker = QSlider(Qt.Orientation.Horizontal)
        self._seeker.setRange(0, 0)
        self._seeker.setToolTip("Seek")
        self._seeker.sliderPressed.connect(self._on_seeker_pressed)
        self._seeker.sliderReleased.connect(self._on_seeker_released)
        layout.addWidget(self._seeker, 1)

        # Volume label + slider
        vol_label = QLabel("Vol")
        vol_label.setStyleSheet("color: #eff0f1; background: transparent;")
        layout.addWidget(vol_label)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(80)
        self._volume_slider.setToolTip("Volume")
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        layout.addWidget(self._volume_slider)

        return bar

    def _init_placeholder(self) -> None:
        label = QLabel("Video playback unavailable (QtMultimedia not found).")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._layout.addWidget(label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def source_path(self) -> str | None:
        """Return the local file path of the currently loaded video, or None."""
        if self._player is None:
            return None
        url = self._player.source()
        if url.isEmpty():
            return None
        return url.toLocalFile() or None

    @property
    def position_ms(self) -> int:
        """Return the current playback position in milliseconds."""
        if self._player is None:
            return 0
        return self._player.position()

    def setAlignment(self, alignment: Qt.AlignmentFlag | Qt.Alignment) -> None:  # noqa: N802
        # Aspect ratio is handled by QVideoWidget internally; nothing to do here.
        pass

    def load_video(self, path: str | Path, *, autoplay: bool = False) -> bool:
        if self._player is None:
            logger.warning("Cannot load video: QtMultimedia is not available")
            return False

        video_path = Path(path).expanduser()
        if not video_path.exists():
            logger.warning("Video file does not exist: %s", video_path)
            return False

        self._player.setSource(QUrl.fromLocalFile(str(video_path.resolve())))
        self._stack.setCurrentIndex(1)  # show video widget
        self._set_controls_enabled(True)

        # Enable save button only when the video is in the temp directory
        if self._save_btn is not None:
            in_temp = (
                self.directories is not None
                and self.directories.temp_path
                and str(video_path.resolve()).startswith(str(Path(self.directories.temp_path).resolve()))
            )
            self._save_btn.setEnabled(bool(in_temp))

        if autoplay:
            self.play()
        return True

    def play(self) -> None:
        if self._player is not None:
            self._player.play()

    def pause(self) -> None:
        if self._player is not None:
            self._player.pause()

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        if self._play_btn is not None:
            self._play_btn.setEnabled(enabled)
        if self._seeker is not None:
            self._seeker.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_loop_clicked(self, checked: bool) -> None:
        self._looping = checked
        if self._player is not None:
            if checked:
                self._player.setLoops(QMediaPlayer.Loops.Infinite)
            else:
                self._player.setLoops(1)

    def _on_play_pause_clicked(self) -> None:
        if self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_position_changed(self, position_ms: int) -> None:
        if not self._seeking and self._seeker is not None:
            self._seeker.setValue(position_ms)
        if self._time_label is not None and self._player is not None:
            duration_ms = self._player.duration()
            self._time_label.setText(f"{_ms_to_str(position_ms)} / {_ms_to_str(duration_ms)}")

    def _on_duration_changed(self, duration_ms: int) -> None:
        if self._seeker is not None:
            self._seeker.setRange(0, duration_ms)
        if self._time_label is not None:
            self._time_label.setText(f"0:00 / {_ms_to_str(duration_ms)}")

    def _on_playback_state_changed(self, state) -> None:
        if self._play_btn is None:
            return
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("⏸")
        else:
            self._play_btn.setText("▶")

    def _on_seeker_pressed(self) -> None:
        self._seeking = True

    def _on_seeker_released(self) -> None:
        if self._player is not None and self._seeker is not None:
            self._player.setPosition(self._seeker.value())
        self._seeking = False

    def _on_volume_changed(self, value: int) -> None:
        if self._audio_output is not None:
            self._audio_output.setVolume(value / 100.0)

    def _on_save_clicked(self) -> None:
        if self._save_btn is not None and not self._save_btn.isEnabled():
            return

        src = self.source_path
        if not src:
            return

        src_path = Path(src)
        if not src_path.exists() or self.directories is None:
            return

        default_dir = str(Path(self.directories.outputs_videos))
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video",
            str(Path(default_dir) / src_path.name),
            "Video Files (*.mp4);;All Files (*)",
        )
        if not dest_path:
            return

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(src_path), str(dest))
            logger.info("Video saved to %s", dest)
        except Exception as exc:
            logger.warning("Failed to save video: %s", exc)
            return

        # Copy source images to the configured outputs directory (with dedup)
        if self.preferences is not None and self.preferences.save_source_images:
            stem = src_path.stem
            temp_dir = Path(self.directories.temp_path)
            img_dest_dir = Path(self.directories.outputs_source_images)
            # Try single legacy name and indexed names
            candidates = [temp_dir / f"{stem}.png"]
            for i in range(20):
                candidates.append(temp_dir / f"{stem}_{i}.png")
            for src_img in candidates:
                if not src_img.exists():
                    continue
                img_hash = hashlib.md5(src_img.read_bytes()).hexdigest()
                duplicate = False
                if img_dest_dir.exists():
                    for existing in img_dest_dir.glob("*.png"):
                        if hashlib.md5(existing.read_bytes()).hexdigest() == img_hash:
                            logger.debug("Source image already exists at %s, skipping", existing)
                            duplicate = True
                            break
                if not duplicate:
                    img_dest_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(str(src_img), str(img_dest_dir / src_img.name))
                        logger.debug("Source image saved to %s", img_dest_dir / src_img.name)
                    except Exception as exc:
                        logger.warning("Failed to save source image: %s", exc)

        # Copy source audio to the configured outputs directory (with dedup)
        if self.preferences is not None and self.preferences.save_source_audio:
            stem = src_path.stem
            temp_dir = Path(self.directories.temp_path)
            audio_dest_dir = Path(self.directories.outputs_source_audio)
            for src_audio in temp_dir.glob(f"{stem}.*"):
                if src_audio.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                    continue
                audio_hash = hashlib.md5(src_audio.read_bytes()).hexdigest()
                duplicate = False
                if audio_dest_dir.exists():
                    for existing in audio_dest_dir.iterdir():
                        if existing.is_file() and hashlib.md5(existing.read_bytes()).hexdigest() == audio_hash:
                            logger.debug("Source audio already exists at %s, skipping", existing)
                            duplicate = True
                            break
                if not duplicate:
                    audio_dest_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(str(src_audio), str(audio_dest_dir / src_audio.name))
                        logger.debug("Source audio saved to %s", audio_dest_dir / src_audio.name)
                    except Exception as exc:
                        logger.warning("Failed to save source audio: %s", exc)

        # Copy source video to the configured outputs directory (with dedup)
        if self.preferences is not None and self.preferences.save_source_video:
            stem = src_path.stem
            temp_dir = Path(self.directories.temp_path)
            video_dest_dir = Path(self.directories.outputs_source_videos)
            src_video = temp_dir / f"{stem}_source_video.mp4"
            if src_video.exists():
                video_hash = hashlib.md5(src_video.read_bytes()).hexdigest()
                duplicate = False
                if video_dest_dir.exists():
                    for existing in video_dest_dir.iterdir():
                        if existing.is_file() and hashlib.md5(existing.read_bytes()).hexdigest() == video_hash:
                            logger.debug("Source video already exists at %s, skipping", existing)
                            duplicate = True
                            break
                if not duplicate:
                    video_dest_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(str(src_video), str(video_dest_dir / src_video.name))
                        logger.debug("Source video saved to %s", video_dest_dir / src_video.name)
                    except Exception as exc:
                        logger.warning("Failed to save source video: %s", exc)

        if self._save_btn is not None:
            self._save_btn.setEnabled(False)

    def _on_player_error(self, error, error_string: str) -> None:
        logger.warning("QMediaPlayer error %s: %s", error, error_string)
