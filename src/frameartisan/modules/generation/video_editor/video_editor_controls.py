"""Resolution picker and action controls for the video editor dialog."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class VideoEditorControls(QWidget):
    """Bottom controls: resolution preset picker + apply button + progress bar."""

    apply_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._source_w: int = 0
        self._source_h: int = 0
        self._gen_w: int = 0
        self._gen_h: int = 0

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Resolution row
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.setToolTip("Output resolution for the processed video.")
        res_layout.addWidget(self.resolution_combo)
        layout.addLayout(res_layout)

        self.resolution_info = QLabel()
        layout.addWidget(self.resolution_info)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("green_button")
        self.apply_button.clicked.connect(self.apply_clicked)
        layout.addWidget(self.apply_button)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)

    def set_source_resolution(self, w: int, h: int) -> None:
        self._source_w = w
        self._source_h = h
        self._rebuild_presets()

    def set_generation_resolution(self, w: int, h: int) -> None:
        self._gen_w = w
        self._gen_h = h
        self._rebuild_presets()

    def get_output_resolution(self) -> tuple[int, int]:
        """Return the selected output resolution (width, height)."""
        data = self.resolution_combo.currentData()
        if data is None:
            return self._source_w, self._source_h
        return data

    def set_busy(self, busy: bool) -> None:
        self.apply_button.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if not busy:
            self.progress_bar.setValue(0)

    def set_progress(self, percent: int) -> None:
        self.progress_bar.setValue(percent)

    def _rebuild_presets(self) -> None:
        self.resolution_combo.clear()
        sw, sh = self._source_w, self._source_h
        gw, gh = self._gen_w, self._gen_h

        if gw > 0 and gh > 0:
            self.resolution_combo.addItem(
                f"Match generation ({gw}\u00d7{gh})",
                (gw, gh),
            )

        if sw > 0 and sh > 0:
            half_w, half_h = _even(sw // 2), _even(sh // 2)
            quarter_w, quarter_h = _even(sw // 4), _even(sh // 4)
            self.resolution_combo.addItem(
                f"1/2 ({half_w}\u00d7{half_h})",
                (half_w, half_h),
            )
            self.resolution_combo.addItem(
                f"1/4 ({quarter_w}\u00d7{quarter_h})",
                (quarter_w, quarter_h),
            )
            self.resolution_combo.addItem(
                f"Original ({sw}\u00d7{sh})",
                (sw, sh),
            )

        self._on_resolution_changed()

    def _on_resolution_changed(self) -> None:
        data = self.resolution_combo.currentData()
        if data is not None:
            w, h = data
            self.resolution_info.setText(f"Output: {w}\u00d7{h}")
        else:
            self.resolution_info.setText("")


def _even(v: int) -> int:
    """Round down to the nearest even number (codec requirement)."""
    return max(v & ~1, 2)
