from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout
from superqt import QLabeledSlider

from frameartisan.modules.generation.panels.base_panel import BasePanel


_VIDEO_CODECS = [
    ("libx264", "H.264"),
    ("libx265", "H.265"),
    ("libaom-av1", "AV1"),
]

_PRESETS = [
    ("ultrafast", "Ultra Fast"),
    ("superfast", "Super Fast"),
    ("veryfast", "Very Fast"),
    ("faster", "Faster"),
    ("medium", "Medium"),
    ("slow", "Slow"),
    ("slower", "Slower"),
    ("veryslow", "Very Slow"),
]

_AUDIO_CODECS = [
    ("aac", "AAC"),
    ("libmp3lame", "MP3"),
    ("libopus", "Opus"),
]

_AUDIO_BITRATES = [64, 128, 192, 256, 320]

_CRF_MAX = {"libx264": 51, "libx265": 51, "libaom-av1": 63}


class ExportPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)
        self.init_ui()
        self._load_from_settings()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Video codec ---
        codec_layout = QHBoxLayout()
        codec_layout.addWidget(QLabel("Video Codec:"))
        self.video_codec_combo = QComboBox()
        for codec_id, label in _VIDEO_CODECS:
            self.video_codec_combo.addItem(label, codec_id)
        self.video_codec_combo.currentIndexChanged.connect(self._on_video_codec_changed)
        codec_layout.addWidget(self.video_codec_combo, 1)
        layout.addLayout(codec_layout)

        # --- CRF slider ---
        crf_layout = QHBoxLayout()
        crf_layout.addWidget(QLabel("CRF:"))
        self.crf_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.crf_slider.setRange(0, 51)
        self.crf_slider.setSingleStep(1)
        self.crf_slider.valueChanged.connect(self._on_crf_changed)
        crf_layout.addWidget(self.crf_slider, 1)
        layout.addLayout(crf_layout)

        # --- Preset ---
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for preset_id, label in _PRESETS:
            self.preset_combo.addItem(label, preset_id)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo, 1)
        layout.addLayout(preset_layout)

        # --- Audio codec ---
        audio_codec_layout = QHBoxLayout()
        audio_codec_layout.addWidget(QLabel("Audio Codec:"))
        self.audio_codec_combo = QComboBox()
        for codec_id, label in _AUDIO_CODECS:
            self.audio_codec_combo.addItem(label, codec_id)
        self.audio_codec_combo.currentIndexChanged.connect(self._on_audio_codec_changed)
        audio_codec_layout.addWidget(self.audio_codec_combo, 1)
        layout.addLayout(audio_codec_layout)

        # --- Audio bitrate ---
        bitrate_layout = QHBoxLayout()
        bitrate_layout.addWidget(QLabel("Audio Bitrate:"))
        self.audio_bitrate_combo = QComboBox()
        for kbps in _AUDIO_BITRATES:
            self.audio_bitrate_combo.addItem(f"{kbps} kbps", kbps)
        self.audio_bitrate_combo.currentIndexChanged.connect(self._on_audio_bitrate_changed)
        bitrate_layout.addWidget(self.audio_bitrate_combo, 1)
        layout.addLayout(bitrate_layout)

        layout.addStretch()
        self.setLayout(layout)

    def _load_from_settings(self):
        # Video codec (also updates CRF max)
        self._set_combo(self.video_codec_combo, self.gen_settings.video_codec)
        self._update_crf_max(self.gen_settings.video_codec)

        blocker = QSignalBlocker(self.crf_slider)
        try:
            self.crf_slider.setValue(int(self.gen_settings.video_crf))
        finally:
            del blocker

        self._set_combo(self.preset_combo, self.gen_settings.video_preset)
        self._set_combo(self.audio_codec_combo, self.gen_settings.audio_codec)
        self._set_combo_by_value(self.audio_bitrate_combo, self.gen_settings.audio_bitrate_kbps)

    def _set_combo(self, combo: QComboBox, data_value):
        blocker = QSignalBlocker(combo)
        try:
            for i in range(combo.count()):
                if combo.itemData(i) == data_value:
                    combo.setCurrentIndex(i)
                    return
            combo.setCurrentIndex(0)
        finally:
            del blocker

    def _set_combo_by_value(self, combo: QComboBox, value):
        self._set_combo(combo, value)

    def _update_crf_max(self, codec: str):
        new_max = _CRF_MAX.get(codec, 51)
        blocker = QSignalBlocker(self.crf_slider)
        try:
            current = self.crf_slider.value()
            self.crf_slider.setMaximum(new_max)
            if current > new_max:
                self.crf_slider.setValue(new_max)
        finally:
            del blocker

    def _on_video_codec_changed(self, index: int):
        codec = self.video_codec_combo.itemData(index)
        if not codec:
            return
        self._update_crf_max(codec)
        # Clamp and re-publish CRF if it changed
        clamped_crf = min(self.crf_slider.value(), self.crf_slider.maximum())
        self.event_bus.publish("generation_change", {"attr": "video_codec", "value": codec})
        self.event_bus.publish("generation_change", {"attr": "video_crf", "value": clamped_crf})

    def _on_crf_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "video_crf", "value": value})

    def _on_preset_changed(self, index: int):
        preset = self.preset_combo.itemData(index)
        if preset:
            self.event_bus.publish("generation_change", {"attr": "video_preset", "value": preset})

    def _on_audio_codec_changed(self, index: int):
        codec = self.audio_codec_combo.itemData(index)
        if codec:
            self.event_bus.publish("generation_change", {"attr": "audio_codec", "value": codec})

    def _on_audio_bitrate_changed(self, index: int):
        kbps = self.audio_bitrate_combo.itemData(index)
        if kbps is not None:
            self.event_bus.publish("generation_change", {"attr": "audio_bitrate_kbps", "value": kbps})
