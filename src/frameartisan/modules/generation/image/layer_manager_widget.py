from importlib.resources import files

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from frameartisan.buttons.expand_contract_button import ExpandContractButton
from frameartisan.buttons.transparent_button import TransparentButton
from frameartisan.modules.generation.image.image_editor_layer import ImageEditorLayer
from frameartisan.modules.generation.image.layer_list_widget import LayerListWidget
from frameartisan.modules.generation.image.layer_widget import LayerWidget


class LayerManagerWidget(QWidget):
    ADD_LAYER_IMG = files("frameartisan.theme.icons").joinpath("add_layer.png")
    DELETE_LAYER_IMG = files("frameartisan.theme.icons").joinpath("delete_layer.png")

    layer_selected = pyqtSignal(ImageEditorLayer)
    layer_visibility_changed = pyqtSignal(int, bool)
    layer_lock_changed = pyqtSignal(int, bool)
    layers_reordered = pyqtSignal(list)
    add_layer_clicked = pyqtSignal()
    delete_layer_clicked = pyqtSignal()
    image_changed = pyqtSignal()

    EXPANDED_WIDTH = 250
    NORMAL_WIDTH = 25

    def __init__(
        self,
        right_side_version: bool = False,
        start_expanded: bool = False,
        expandable: bool = True,
        use_for_mask: bool = False,
    ):
        super().__init__()

        self.right_side_version = right_side_version
        self.start_expanded = start_expanded
        self.expandable = expandable
        self.use_for_mask = use_for_mask

        if self.start_expanded:
            self.setMinimumSize(self.EXPANDED_WIDTH, 50)
            self.setMaximumWidth(self.EXPANDED_WIDTH)
            self.expanded = True
        else:
            self.setMinimumSize(self.NORMAL_WIDTH, 50)
            self.setMaximumWidth(self.NORMAL_WIDTH)
            self.expanded = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        if self.expandable:
            self.animating = False
            self.animation_min = QPropertyAnimation(self, b"minimumWidth")
            self.animation_max = QPropertyAnimation(self, b"maximumWidth")
            self.animation_min.finished.connect(self.animation_finished)
            self.animation_min.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.animation_max.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.animation_min.setDuration(300)
            self.animation_max.setDuration(300)

        self.init_ui()

        self.selected_layer = None

        if self.start_expanded:
            self.list_widget.setVisible(True)
            self.layers_controls_widget.setVisible(True)
        else:
            self.list_widget.setVisible(False)
            self.layers_controls_widget.setVisible(False)

    def init_ui(self):
        button_alignment = Qt.AlignmentFlag.AlignLeft if self.right_side_version else Qt.AlignmentFlag.AlignRight

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        if self.expandable:
            self.expand_btn = ExpandContractButton(25, 25, False, self.right_side_version)
            self.expand_btn.clicked.connect(self.on_expand_clicked)
            main_layout.addWidget(self.expand_btn, alignment=Qt.AlignmentFlag.AlignTop | button_alignment)

        self.list_widget = LayerListWidget()
        self.list_widget.currentItemChanged.connect(self.handle_item_selected)
        self.list_widget.layers_reordered.connect(self.on_layers_reordered)
        self.list_widget.setSpacing(0)
        main_layout.addWidget(self.list_widget)

        self.layers_controls_widget = QWidget()
        layers_controls_layout = QVBoxLayout()

        if not self.use_for_mask:
            layer_add_remove_layout = QHBoxLayout()
            self.add_layer_button = TransparentButton(self.ADD_LAYER_IMG, 28, 28)
            self.add_layer_button.setObjectName("bottom_layer_control")
            self.add_layer_button.clicked.connect(self.on_add_layer_clicked)
            layer_add_remove_layout.addWidget(self.add_layer_button, alignment=Qt.AlignmentFlag.AlignLeft)
            self.delete_layer_button = TransparentButton(self.DELETE_LAYER_IMG, 28, 28)
            self.delete_layer_button.clicked.connect(self.on_delete_layer_clicked)
            self.delete_layer_button.setObjectName("bottom_layer_control")
            layer_add_remove_layout.addWidget(self.delete_layer_button, alignment=Qt.AlignmentFlag.AlignRight)
            layers_controls_layout.addLayout(layer_add_remove_layout)

        image_buttons_layout = QHBoxLayout()
        invert_button = QPushButton("Invert")
        invert_button.clicked.connect(self.on_invert_image)
        image_buttons_layout.addWidget(invert_button)
        horizontal_mirror_button = QPushButton("H. Mirror")
        horizontal_mirror_button.clicked.connect(lambda: self.on_mirror_image(True, False))
        image_buttons_layout.addWidget(horizontal_mirror_button)
        vertical_mirror_button = QPushButton("V. Mirror")
        vertical_mirror_button.clicked.connect(lambda: self.on_mirror_image(False, True))
        image_buttons_layout.addWidget(vertical_mirror_button)
        layers_controls_layout.addLayout(image_buttons_layout)

        layer_advanced_layout = QGridLayout()

        opacity_label = QLabel("Opacity")
        layer_advanced_layout.addWidget(opacity_label, 0, 0)
        self.opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0.0, 1.0)
        self.opacity_slider.setValue(1.0)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        layer_advanced_layout.addWidget(self.opacity_slider, 0, 1)

        brightness_label = QLabel("Brightness")
        layer_advanced_layout.addWidget(brightness_label, 1, 0)
        self.brightness_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-1.0, 1.0)
        self.brightness_slider.setValue(0.0)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        layer_advanced_layout.addWidget(self.brightness_slider, 1, 1)

        contrast_label = QLabel("Contrast")
        layer_advanced_layout.addWidget(contrast_label, 2, 0)
        self.contrast_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-150.0, 150.0)
        self.contrast_slider.setValue(1.0)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        layer_advanced_layout.addWidget(self.contrast_slider, 2, 1)

        saturation_label = QLabel("Saturation")
        layer_advanced_layout.addWidget(saturation_label, 3, 0)
        self.saturation_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(0.0, 3.0)
        self.saturation_slider.setValue(1.0)
        self.saturation_slider.valueChanged.connect(self.on_saturation_changed)
        layer_advanced_layout.addWidget(self.saturation_slider, 3, 1)

        layers_controls_layout.addLayout(layer_advanced_layout)

        self.layers_controls_widget.setLayout(layers_controls_layout)
        main_layout.addWidget(self.layers_controls_widget)

        self.setLayout(main_layout)

    def on_add_layer_clicked(self):
        self.add_layer_clicked.emit()

    def add_layer(self, layer: ImageEditorLayer):
        item = QListWidgetItem()
        widget = LayerWidget(layer)
        item.setSizeHint(widget.sizeHint())

        self.list_widget.insertItem(0, item)
        self.list_widget.setItemWidget(item, widget)
        self.list_widget.setCurrentItem(item)

    def restore_layers(self, layers: list[ImageEditorLayer]):
        self.list_widget.clear()

        for layer in sorted(layers, key=lambda layer: layer.order, reverse=True):
            item = QListWidgetItem()
            widget = LayerWidget(layer)
            item.setSizeHint(widget.sizeHint())

            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def on_delete_layer_clicked(self):
        self.delete_layer_clicked.emit()

    def delete_layer(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            layer_widget = self.list_widget.itemWidget(item)
            if layer_widget.layer == self.selected_layer:
                self.list_widget.takeItem(i)
                break

    def get_layer_name(self, layer_id):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget.layer.layer_id == layer_id:
                return widget.layer.layer_name
        return None

    def handle_item_selected(self, current_item, previous_item):
        if current_item is not None:
            self.selected_layer = self.list_widget.itemWidget(current_item).layer
            self.opacity_slider.setValue(self.selected_layer.opacity)
            self.brightness_slider.setValue(self.selected_layer.brightness)
            self.contrast_slider.setValue(self.selected_layer.contrast)
            self.saturation_slider.setValue(self.selected_layer.saturation)
            self.layer_selected.emit(self.selected_layer)

    def on_layers_reordered(self, layers: list):
        self.layers_reordered.emit(layers)

    def on_expand_clicked(self):
        if not self.expandable:
            return

        if self.expanded:
            self.contract()
        else:
            self.expand()

    def expand(self):
        if not self.expandable or self.animating:
            return

        self.animation_min.setStartValue(self.NORMAL_WIDTH)
        self.animation_min.setEndValue(self.EXPANDED_WIDTH)
        self.animation_max.setStartValue(self.NORMAL_WIDTH)
        self.animation_max.setEndValue(self.EXPANDED_WIDTH)
        self.animation_min.start()
        self.animation_max.start()
        self.animating = True

    def contract(self):
        if not self.expandable or self.animating:
            return

        self.list_widget.setVisible(False)
        self.layers_controls_widget.setVisible(False)
        self.animation_min.setStartValue(self.EXPANDED_WIDTH)
        self.animation_min.setEndValue(self.NORMAL_WIDTH)
        self.animation_max.setStartValue(self.EXPANDED_WIDTH)
        self.animation_max.setEndValue(self.NORMAL_WIDTH)
        self.animation_min.start()
        self.animation_max.start()
        self.animating = True

    def animation_finished(self):
        if self.expanded:
            self.expanded = False
        else:
            self.list_widget.setVisible(True)
            self.layers_controls_widget.setVisible(True)
            self.expanded = True

        self.animating = False

    def on_opacity_changed(self, value):
        self.selected_layer.set_opacity(value)
        self.image_changed.emit()

    def on_brightness_changed(self, value):
        self.selected_layer.set_brightness(value)
        self.image_changed.emit()

    def on_contrast_changed(self, value):
        self.selected_layer.set_contrast(value)
        self.image_changed.emit()

    def on_saturation_changed(self, value):
        self.selected_layer.set_saturation(value)
        self.image_changed.emit()

    def on_invert_image(self):
        self.selected_layer.invert_image()
        self.image_changed.emit()

    def on_mirror_image(self, horizontal: bool, vertical: bool):
        self.selected_layer.mirror_image(horizontal, vertical)
        self.image_changed.emit()
