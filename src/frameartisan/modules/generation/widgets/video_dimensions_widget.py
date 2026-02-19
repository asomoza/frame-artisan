from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QSlider, QVBoxLayout, QWidget

from frameartisan.app.event_bus import EventBus
from frameartisan.buttons.linked_button import LinkedButton


# LTX2 requires both width and height to be divisible by 32.
_MIN_DIM = 256
_MAX_DIM = 2048
_STEP = 32
ALLOWED_VALUES = list(range(_MIN_DIM, _MAX_DIM + 1, _STEP))


def _snap(value: int) -> int:
    clamped = max(_MIN_DIM, min(_MAX_DIM, value))
    return min(ALLOWED_VALUES, key=lambda x: abs(x - clamped))


class VideoDimensionsWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.event_bus = EventBus()
        self._updating_linked = False
        self._prev_width = _MIN_DIM
        self._prev_height = _MIN_DIM

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        sliders_layout = QGridLayout()

        width_label = QLabel("Width")
        sliders_layout.addWidget(width_label, 0, 0)

        self.width_slider = QSlider()
        self.width_slider.setRange(_MIN_DIM, _MAX_DIM)
        self.width_slider.setSingleStep(_STEP)
        self.width_slider.setPageStep(_STEP)
        self.width_slider.setOrientation(Qt.Orientation.Horizontal)
        sliders_layout.addWidget(self.width_slider, 0, 1)

        self.video_width_value_label = QLabel()
        sliders_layout.addWidget(self.video_width_value_label, 0, 2)

        self.linked_button = LinkedButton()
        sliders_layout.addWidget(self.linked_button, 0, 3, 2, 1, Qt.AlignmentFlag.AlignCenter)

        height_label = QLabel("Height")
        sliders_layout.addWidget(height_label, 1, 0)

        self.height_slider = QSlider()
        self.height_slider.setRange(_MIN_DIM, _MAX_DIM)
        self.height_slider.setSingleStep(_STEP)
        self.height_slider.setPageStep(_STEP)
        self.height_slider.setOrientation(Qt.Orientation.Horizontal)
        sliders_layout.addWidget(self.height_slider, 1, 1)

        self.video_height_value_label = QLabel()
        sliders_layout.addWidget(self.video_height_value_label, 1, 2)

        self.width_slider.valueChanged.connect(self.on_slider_value_changed)
        self.height_slider.valueChanged.connect(self.on_slider_value_changed)

        main_layout.addLayout(sliders_layout)

        self.setLayout(main_layout)

    def on_slider_value_changed(self):
        if self._updating_linked:
            return

        slider = self.sender()
        nearest_value = _snap(slider.value())

        if slider == self.width_slider:
            delta = nearest_value - self._prev_width
            self._prev_width = nearest_value
            self.video_width_value_label.setText(str(nearest_value))
            self.event_bus.publish("generation_change", {"attr": "video_width", "value": nearest_value})

            if self.linked_button.linked and delta != 0:
                new_height = _snap(self._prev_height + delta)
                self._prev_height = new_height
                self._update_other_slider(
                    self.height_slider, self.video_height_value_label, "video_height", new_height
                )
        else:
            delta = nearest_value - self._prev_height
            self._prev_height = nearest_value
            self.video_height_value_label.setText(str(nearest_value))
            self.event_bus.publish("generation_change", {"attr": "video_height", "value": nearest_value})

            if self.linked_button.linked and delta != 0:
                new_width = _snap(self._prev_width + delta)
                self._prev_width = new_width
                self._update_other_slider(self.width_slider, self.video_width_value_label, "video_width", new_width)

    def _update_other_slider(self, slider: QSlider, label: QLabel, attr: str, value: int) -> None:
        self._updating_linked = True
        try:
            slider.setValue(value)
            label.setText(str(value))
            self.event_bus.publish("generation_change", {"attr": attr, "value": value})
        finally:
            self._updating_linked = False
