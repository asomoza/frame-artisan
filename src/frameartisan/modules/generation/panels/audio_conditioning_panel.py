import logging
import os

from PyQt6.QtCore import QMimeData, QSignalBlocker, Qt
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from frameartisan.modules.generation.widgets.drop_lightbox_widget import DropLightBox
from superqt import QLabeledRangeSlider

from frameartisan.modules.generation.panels.base_panel import BasePanel

logger = logging.getLogger(__name__)


class AudioConditioningPanel(BasePanel):
    _AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self, *args):
        super().__init__(*args)
        self.setAcceptDrops(True)

        self._audio_path: str | None = None
        self._audio_duration: float = 0.0
        self._from_video: bool = False
        self._publishing = False

        self.init_ui()

        self.event_bus.subscribe("audio_condition", self.on_audio_condition_event)

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.setEnabled(False)
        self.enabled_checkbox.toggled.connect(self._on_enabled_changed)
        main_layout.addWidget(self.enabled_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        load_button = QPushButton("Load Audio")
        load_button.clicked.connect(self._on_load_audio)
        load_button.setObjectName("green_button")
        main_layout.addWidget(load_button)

        self.file_label = QLabel("No audio loaded")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        main_layout.addWidget(self.file_label)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        main_layout.addWidget(self.info_label)

        # Source audio trim range (in tenths of a second)
        source_range_layout = QHBoxLayout()
        source_range_layout.addWidget(QLabel("Trim (0.1s):"))
        self.source_range_slider = QLabeledRangeSlider()
        self.source_range_slider.setRange(0, 1)
        self.source_range_slider.setValue((0, 1))
        self.source_range_slider.setOrientation(Qt.Orientation.Horizontal)
        self.source_range_slider.setToolTip("Range of source audio to use (in tenths of a second)")
        self.source_range_slider.valueChanged.connect(self._on_trim_changed)
        self.source_range_slider.setEnabled(False)
        source_range_layout.addWidget(self.source_range_slider)
        main_layout.addLayout(source_range_layout)

        self.remove_button = QPushButton("Remove Audio")
        self.remove_button.setObjectName("red_button")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self._on_remove_audio)
        main_layout.addWidget(self.remove_button)

        main_layout.addStretch()
        self.setLayout(main_layout)

        self._drop_lightbox = DropLightBox(self)
        self._drop_lightbox.setText("Drop audio here")

    @property
    def trim_range(self) -> tuple[float, float]:
        """Start and end of the source audio trim range in seconds."""
        val = self.source_range_slider.value()
        return (int(val[0]) / 10.0, int(val[1]) / 10.0)

    def _on_enabled_changed(self, checked: bool) -> None:
        self.event_bus.publish("generation_change", {"attr": "audio_conditioning_enabled", "value": checked})

    @staticmethod
    def _probe_audio_duration(path: str) -> float:
        try:
            import av

            with av.open(path) as container:
                stream = container.streams.audio[0]
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
                # Fallback: decode to measure duration
                duration = 0.0
                for frame in container.decode(audio=0):
                    duration = max(duration, float(frame.pts * stream.time_base) + float(frame.samples) / frame.rate)
                return duration
        except Exception:
            return 0.0

    def _set_audio(self, path: str, from_video: bool = False) -> None:
        self._audio_path = path
        self._from_video = from_video

        if from_video:
            self.file_label.setText("Audio from video")
        else:
            self.file_label.setText(os.path.basename(path))

        self._audio_duration = self._probe_audio_duration(path)
        if self._audio_duration > 0:
            self.info_label.setText(f"Duration: {self._audio_duration:.1f}s")
            max_tenths = round(self._audio_duration * 10)
            blocker = QSignalBlocker(self.source_range_slider)
            try:
                self.source_range_slider.setRange(0, max_tenths)
                self.source_range_slider.setValue((0, max_tenths))
            finally:
                del blocker
            self.source_range_slider.setEnabled(True)
        else:
            self.info_label.setText("")
            self.source_range_slider.setEnabled(False)

        self.enabled_checkbox.setEnabled(True)
        self.remove_button.setEnabled(not from_video)

    def _on_trim_changed(self) -> None:
        if self._publishing:
            return
        self._publishing = True
        try:
            if self._audio_path is not None:
                val = self.source_range_slider.value()
                is_full_range = int(val[0]) == self.source_range_slider.minimum() and int(
                    val[1]
                ) == self.source_range_slider.maximum()
                if is_full_range:
                    # Full range selected — clear trim so node uses full audio
                    self.event_bus.publish(
                        "audio_condition",
                        {"action": "update_trim", "trim_start_s": None, "trim_end_s": None},
                    )
                else:
                    trim_start, trim_end = self.trim_range
                    self.event_bus.publish(
                        "audio_condition",
                        {"action": "update_trim", "trim_start_s": trim_start, "trim_end_s": trim_end},
                    )
        finally:
            self._publishing = False

    @staticmethod
    def _get_audio_url(mime: QMimeData) -> str | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if os.path.splitext(path)[1].lower() in AudioConditioningPanel._AUDIO_EXTENSIONS:
                return path
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if self._get_audio_url(event.mimeData()):
            event.acceptProposedAction()
            self._drop_lightbox.show()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if self._get_audio_url(event.mimeData()):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # type: ignore[override]
        self._drop_lightbox.hide()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._drop_lightbox.hide()
        path = self._get_audio_url(event.mimeData())
        if not path:
            return
        self._set_audio(path)

        if not self.enabled_checkbox.isChecked():
            self.enabled_checkbox.setChecked(True)
        else:
            self._on_enabled_changed(True)

        self.event_bus.publish(
            "audio_condition",
            {"action": "add", "audio_path": path},
        )
        event.acceptProposedAction()

    def _on_load_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)",
        )
        if path:
            self._set_audio(path)

            if not self.enabled_checkbox.isChecked():
                self.enabled_checkbox.setChecked(True)
            else:
                self._on_enabled_changed(True)

            self.event_bus.publish(
                "audio_condition",
                {"action": "add", "audio_path": path},
            )

    def _on_remove_audio(self) -> None:
        self._audio_path = None
        self._audio_duration = 0.0
        self.file_label.setText("No audio loaded")
        self.info_label.setText("")
        self.enabled_checkbox.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.source_range_slider.setEnabled(False)

        blocker = QSignalBlocker(self.enabled_checkbox)
        try:
            self.enabled_checkbox.setChecked(False)
        finally:
            del blocker

        self.event_bus.publish("audio_condition", {"action": "remove"})

    def on_audio_condition_event(self, data: dict) -> None:
        action = data.get("action")
        if action == "add":
            path = data.get("audio_path")
            if path and path != self._audio_path:
                from_video = data.get("from_video", False)
                self._set_audio(path, from_video=from_video)
                if not self.enabled_checkbox.isChecked():
                    self.enabled_checkbox.setChecked(True)
        elif action == "remove":
            if self._audio_path is not None:
                self._from_video = False
                self._on_remove_audio()
