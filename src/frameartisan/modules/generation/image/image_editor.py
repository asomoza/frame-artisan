from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Union

from PyQt6.QtCore import QPoint, QPointF, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QGuiApplication,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import QApplication, QFileDialog, QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QMenu

from frameartisan.modules.generation.image.image_editor_layer import ImageEditorLayer
from frameartisan.modules.generation.image.layer_manager import LayerManager
from frameartisan.modules.generation.image.layer_pixmap_item import LayerPixmapItem


class ImageEditor(QGraphicsView):
    image_changed = pyqtSignal()
    image_moved = pyqtSignal(float, float)
    image_scaled = pyqtSignal(float)
    image_rotated = pyqtSignal(float)
    image_pasted = pyqtSignal(QPixmap)
    image_copy = pyqtSignal()

    def __init__(
        self,
        target_width: int,
        target_height: int,
        aspect_ratio: float,
        temp_path: str = "tmp/",
        save_directory: str = "",
    ):
        super(ImageEditor, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.save_directory = save_directory
        self.temp_path = temp_path
        self.target_width = target_width
        self.target_height = target_height
        self.aspect_ratio = aspect_ratio

        self.setSceneRect(0, 0, self.target_width, self.target_height)

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scene = QGraphicsScene(0, 0, 0, 0)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        self.layer_manager = LayerManager(self.target_width, self.target_height, self.temp_path)
        self.selected_layer = None

        self.drawing = False
        self.last_point = QPoint()
        self.brush_color = QColor(0, 0, 0, 255)
        self.brush_size = 20
        self.hardness = 0.5
        self.steps = 0.1
        self.brush_preview = None
        self.last_cursor_size = None
        self.accumulated_distance = 0.0

        self.erasing = False

        self.moving = False
        self.last_mouse_position = None
        self.setMouseTracking(True)
        self.current_scale_factor = 1.0
        self.total_translation = QPointF(0, 0)

        self.enable_copy = True
        self.enable_save = True

        # Cached zoom factor to prevent cursor size fluctuations from spurious resize events
        self._cached_zoom_factor = None
        self._cached_viewport_size = None
        self._oscillation_sizes = set()
        self._stable_viewport_size = None
        self._editor_id = id(self)

        # Scene-based brush outline (circle that follows the mouse)
        self._brush_outline = None
        self._last_outline_color_is_light = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_cursor)

        self.brush_preview_timer = QTimer()
        self.brush_preview_timer.timeout.connect(self.hide_brush_preview)
        self.brush_preview_timer.setSingleShot(True)

    def sizeHint(self):
        return QSize(self.target_width, self.target_height)

    def resizeEvent(self, event):
        if not self.isVisible():
            super().resizeEvent(event)
            return

        current_viewport = (self.viewport().width(), self.viewport().height())

        if self._cached_viewport_size is not None and current_viewport != self._cached_viewport_size:
            self._oscillation_sizes.add(current_viewport)
            self._oscillation_sizes.add(self._cached_viewport_size)

            if len(self._oscillation_sizes) == 2:
                larger_size = max(self._oscillation_sizes, key=lambda s: s[0] * s[1])

                if current_viewport != larger_size:
                    super().resizeEvent(event)
                    return

                self._stable_viewport_size = larger_size

        rect = self.sceneRect()
        self.resetTransform()
        self.scale(self.viewport().width() / rect.width(), self.viewport().height() / rect.height())

        self._cached_zoom_factor = self.transform().m11()
        self._cached_viewport_size = current_viewport

        super().resizeEvent(event)

    def enterEvent(self, event):
        self.setFocus()
        self.drawing = False
        self.moving = False
        self.timer.start(100)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.timer.stop()
        self._hide_brush_outline()
        super().leaveEvent(event)

    def showEvent(self, event):
        self._cached_zoom_factor = None
        self._cached_viewport_size = None
        self._oscillation_sizes.clear()
        self._stable_viewport_size = None
        self._hide_brush_outline()
        super().showEvent(event)

    def add_layer(self, image_path: str = None, save_temp: bool = False) -> ImageEditorLayer:
        temp_path = self.temp_path if save_temp else None

        layer = self.layer_manager.add_new_layer(image_path, base_path=temp_path)
        self.scene.addItem(layer.pixmap_item)

        layer.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        layer.pixmap_item.setZValue(layer.order)
        self.image_changed.emit()

        self.selected_layer = layer

        return layer

    def restore_layers(self, layers: list[ImageEditorLayer]):
        self.layer_manager.delete_all()
        self.scene.clear()

        for layer in layers:
            self.layer_manager.add_layer_object(layer)

            if layer.pixmap_item is None:
                pixmap = QPixmap(layer.image_path)
                layer.pixmap_item = LayerPixmapItem(
                    pixmap, brightness=layer.brightness, contrast=layer.contrast, saturation=layer.saturation
                )
                layer.pixmap_item.setOpacity(layer.opacity)
                layer.pixmap_item.setVisible(layer.visible)
                layer.apply_transform_properties()

            self.scene.addItem(layer.pixmap_item)

            layer.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            layer.pixmap_item.setZValue(layer.order)

        if layers:
            self.selected_layer = layers[-1]
        else:
            self.selected_layer = None

        self.image_changed.emit()

    def change_layer_image(self, image: Union[str, QPixmap]):
        self.selected_layer.set_pixmap_item(image, self.temp_path)
        self.image_changed.emit()

    def reload_image_layer(self, image_path: str, original_path: str, order: int):
        pixmap = QPixmap(image_path)
        layer = self.layer_manager.reload_layer(pixmap, image_path, original_path, order)
        self.scene.addItem(layer.pixmap_item)

        layer.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        layer.pixmap_item.setZValue(layer.order)

        return layer.layer_id

    def get_all_layers(self) -> list[ImageEditorLayer]:
        return self.layer_manager.layers

    def set_layer_locked(self, layer_id: int, locked: bool):
        layer = self.layer_manager.get_layer_by_id(layer_id)

        if layer is not None:
            layer.locked = locked

    def set_layer_visible(self, layer_id: int, visible: bool):
        layer = self.layer_manager.get_layer_by_id(layer_id)

        if layer is not None and layer.pixmap_item is not None:
            layer.visible = visible
            layer.pixmap_item.setVisible(visible)

    def set_layer_order(self, layer_id: int, order: int):
        layer = self.layer_manager.get_layer_by_id(layer_id)

        if layer is not None and layer.order != order:
            self.layer_manager.move_layer(layer_id, order)

            for layer in self.layer_manager.get_layers():
                layer.pixmap_item.setZValue(layer.order)

    def edit_all_layers_order(self, layers: list):
        for layer_id, order in layers:
            for layer in self.layer_manager.layers:
                if layer.layer_id == layer_id:
                    layer.order = order
                    layer.pixmap_item.setZValue(layer.order)

    def fit_image(self):
        pixmap_size = self.selected_layer.pixmap_item.pixmap().size()

        self.selected_layer.pixmap_item.setRotation(0)

        width_scale = (self.mapToScene(self.viewport().rect()).boundingRect().width()) / pixmap_size.width()
        height_scale = (self.mapToScene(self.viewport().rect()).boundingRect().height()) / pixmap_size.height()
        scale_factor = min(width_scale, height_scale)

        self.set_image_scale(scale_factor)

        scaled_width = pixmap_size.width() * scale_factor
        scaled_height = pixmap_size.height() * scale_factor
        width_diff = pixmap_size.width() - scaled_width
        height_diff = pixmap_size.height() - scaled_height

        new_x = width_diff / 2
        new_y = height_diff / 2

        if self.selected_layer.locked:
            self.set_image_x(-new_x)
            self.set_image_y(-new_y)
        else:
            self.selected_layer.pixmap_item.setPos(-new_x, -new_y)

        self.image_scaled.emit(scale_factor)
        self.image_moved.emit(-new_x, -new_y)
        self.image_rotated.emit(0)

    def set_image_scale(self, scale_factor):
        if self.selected_layer.pixmap_item is not None:
            scale_ratio = scale_factor / self.selected_layer.pixmap_item.scale()

            if self.selected_layer.locked:
                reference_point = self.selected_layer.pixmap_item.pos()

                self.selected_layer.pixmap_item.setTransformOriginPoint(
                    self.selected_layer.pixmap_item.boundingRect().center()
                )
                self.selected_layer.pixmap_item.setScale(scale_factor)
                self.selected_layer.scale = scale_factor

                new_reference_point = self.selected_layer.pixmap_item.pos()
                position_delta = new_reference_point - reference_point

                for layer in self.layer_manager.get_layers():
                    if layer.layer_id != self.selected_layer.layer_id and layer.locked:
                        relative_pos = layer.pixmap_item.pos() - reference_point

                        layer.pixmap_item.setTransformOriginPoint(layer.pixmap_item.boundingRect().center())
                        layer.pixmap_item.setScale(layer.pixmap_item.scale() * scale_ratio)
                        layer.scale = layer.pixmap_item.scale()

                        new_pos = reference_point + relative_pos * scale_ratio + position_delta
                        layer.pixmap_item.setPos(new_pos)
                        layer.x = new_pos.x()
                        layer.y = new_pos.y()
            else:
                self.selected_layer.pixmap_item.setTransformOriginPoint(
                    self.selected_layer.pixmap_item.boundingRect().center()
                )
                self.selected_layer.pixmap_item.setScale(scale_factor)
                self.selected_layer.scale = scale_factor

    def set_image_x(self, x_position):
        x_delta = x_position - self.selected_layer.pixmap_item.x()

        if self.selected_layer.pixmap_item is not None:
            self.selected_layer.pixmap_item.setX(x_position)
            self.selected_layer.x = x_position

            if self.selected_layer.locked:
                for layer in self.layer_manager.get_layers():
                    if layer.layer_id != self.selected_layer.layer_id and layer.locked:
                        new_x = layer.pixmap_item.x() + x_delta
                        layer.pixmap_item.setX(new_x)
                        layer.x = new_x

    def set_image_y(self, y_position):
        y_delta = y_position - self.selected_layer.pixmap_item.y()

        self.selected_layer.pixmap_item.setY(y_position)
        self.selected_layer.y = y_position

        if self.selected_layer.locked:
            for layer in self.layer_manager.get_layers():
                if layer.layer_id != self.selected_layer.layer_id and layer.locked:
                    new_y = layer.pixmap_item.y() + y_delta
                    layer.pixmap_item.setY(new_y)
                    layer.y = new_y

    def rotate_image(self, angle):
        rotation_delta = angle - self.selected_layer.pixmap_item.rotation()

        self.selected_layer.pixmap_item.setTransformOriginPoint(
            self.selected_layer.pixmap_item.boundingRect().center()
        )
        self.selected_layer.pixmap_item.setRotation(angle)
        self.selected_layer.rotation = angle

        if self.selected_layer.locked:
            for layer in self.layer_manager.get_layers():
                if layer.layer_id != self.selected_layer.layer_id and layer.locked:
                    layer.pixmap_item.setTransformOriginPoint(layer.pixmap_item.boundingRect().center())
                    new_rotation = layer.pixmap_item.rotation() + rotation_delta
                    layer.pixmap_item.setRotation(new_rotation)
                    layer.rotation = new_rotation

    def clear_and_restore(self):
        if self.selected_layer.original_path is None:
            alpha_pixmap = QPixmap(self.target_width, self.target_height)
            alpha_pixmap.fill(Qt.GlobalColor.transparent)
        else:
            pixmap = QPixmap(self.selected_layer.original_path)

            alpha_pixmap = QPixmap(pixmap.size())
            alpha_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(alpha_pixmap)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()

        self.selected_layer.pixmap_item.setPixmap(alpha_pixmap)
        self.selected_layer.pixmap_item.setScale(1)
        self.selected_layer.pixmap_item.setX(0)
        self.selected_layer.pixmap_item.setY(0)
        self.selected_layer.pixmap_item.setRotation(0)
        self.image_scaled.emit(1)
        self.image_moved.emit(0, 0)
        self.image_rotated.emit(0)

        self.image_changed.emit()

    def delete_layer(self):
        if (
            self.selected_layer.image_path is not None
            and os.path.isfile(self.selected_layer.image_path)
            and self.temp_path in self.selected_layer.image_path
        ):
            os.remove(self.selected_layer.image_path)

        if (
            self.selected_layer.original_path is not None
            and os.path.isfile(self.selected_layer.original_path)
            and self.temp_path in self.selected_layer.original_path
        ):
            os.remove(self.selected_layer.original_path)

        self.scene.removeItem(self.selected_layer.pixmap_item)
        self.layer_manager.delete_layer(self.selected_layer.layer_id)

    def clear_all(self):
        self.scene.clear()
        self.layer_manager.delete_all()

    def draw(self, point):
        pixmap = self.selected_layer.pixmap_item.pixmap()
        painter = QPainter(pixmap)

        if self.erasing:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        gradient = QRadialGradient(point, self.brush_size / 2)
        gradient.setColorAt(0, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 255))
        gradient.setColorAt(self.hardness, self.brush_color)
        gradient.setColorAt(1, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 0))

        brush = QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(point, self.brush_size / 2, self.brush_size / 2)
        painter.end()

        self.selected_layer.pixmap_item.setPixmap(pixmap)
        self.update()

    def draw_line(self, start_point, end_point):
        if start_point == end_point:
            return

        dx = end_point.x() - start_point.x()
        dy = end_point.y() - start_point.y()
        distance = (dx * dx + dy * dy) ** 0.5

        self.accumulated_distance += distance

        base_step_size = max(1.0, self.brush_size * 0.1)
        step_size = base_step_size * self.steps

        if self.accumulated_distance >= step_size:
            self.draw(end_point)
            self.accumulated_distance = 0.0

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.timer.stop()
                self.last_mouse_position = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self.drawing = True
                self.last_point = self.selected_layer.pixmap_item.mapFromScene(self.mapToScene(event.pos()))
                self.accumulated_distance = 0.0
                self.draw(self.last_point)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._brush_outline is not None and self.selected_layer and self.selected_layer.pixmap_item:
            scene_pos = self.mapToScene(event.pos())
            scale_factor = self.selected_layer.pixmap_item.scale()
            brush_size_in_scene = self.brush_size * scale_factor
            radius = brush_size_in_scene / 2
            self._brush_outline.setRect(
                scene_pos.x() - radius,
                scene_pos.y() - radius,
                brush_size_in_scene,
                brush_size_in_scene,
            )

        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.moving:
                self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                delta = self.mapToScene(self.last_mouse_position) - self.mapToScene(event.pos())
                self.setSceneRect(self.sceneRect().translated(delta.x(), delta.y()))
                self.translate(delta.x() / self.current_scale_factor, delta.y() / self.current_scale_factor)
                self.total_translation += delta
                self.last_mouse_position = event.pos()
            elif self.drawing:
                current_point = self.selected_layer.pixmap_item.mapFromScene(self.mapToScene(event.pos()))
                self.draw_line(self.last_point, current_point)
                self.last_point = current_point
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.drawing:
                self.image_changed.emit()
                self.drawing = False
                self.timer.start(100)
            else:
                self.timer.start(100)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)
        self.current_scale_factor *= zoomFactor

        self._cached_zoom_factor = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self.moving:
                self.timer.stop()
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.moving = True

            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.moving = False
            self.timer.start(100)
        else:
            super().keyReleaseEvent(event)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        if self.enable_copy:
            copy_action = QAction("Copy Image", self)
            copy_action.triggered.connect(self.copy_image)
            context_menu.addAction(copy_action)

        paste_action = QAction("Paste Image", self)
        paste_action.triggered.connect(self.paste_image)

        clipboard = QGuiApplication.clipboard()
        mime_data = clipboard.mimeData()
        paste_action.setDisabled(True)

        if mime_data.hasImage():
            paste_action.setEnabled(True)

        context_menu.addAction(paste_action)

        if self.enable_save:
            save_action = QAction("Save Image", self)
            save_action.triggered.connect(self.save_image)
            context_menu.addAction(save_action)

        context_menu.exec(event.globalPos())

    def save_image(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            os.path.join(self.save_directory, f"{timestamp}.png"),
            "Images (*.png *.jpg)",
        )

        pixmap = self.get_scene_as_pixmap()
        pixmap.save(save_path)

    def paste_image(self):
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()

        if not pixmap.isNull():
            self.image_pasted.emit(pixmap)

    def copy_image(self):
        self.image_copy.emit()

    def _update_brush_outline(self, scene_pos: QPointF, brush_size_in_scene: float, use_light_color: bool):
        """Update or create the scene-based brush outline."""
        radius = brush_size_in_scene / 2

        if self._brush_outline is None:
            self._brush_outline = QGraphicsEllipseItem()
            self._brush_outline.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self._brush_outline.setZValue(10000)
            self.scene.addItem(self._brush_outline)

        if self._last_outline_color_is_light != use_light_color:
            if use_light_color:
                outline_color = QColor(255, 255, 255, 200)
            else:
                outline_color = QColor(0, 0, 0, 200)
            self._brush_outline.setPen(QPen(outline_color, 1.5))
            self._last_outline_color_is_light = use_light_color

        self._brush_outline.setRect(
            scene_pos.x() - radius,
            scene_pos.y() - radius,
            brush_size_in_scene,
            brush_size_in_scene,
        )
        self._brush_outline.setVisible(True)

    def _hide_brush_outline(self):
        """Hide the scene-based brush outline."""
        if self._brush_outline is not None:
            self._brush_outline.setVisible(False)

    def update_cursor(self):
        if self.selected_layer is None or self.selected_layer.pixmap_item is None:
            return

        cursor_pos = QCursor.pos()
        view_pos = self.viewport().mapFromGlobal(cursor_pos)
        scene_pos = self.mapToScene(view_pos)

        scale_factor = self.selected_layer.pixmap_item.scale()
        brush_size_in_scene = self.brush_size * scale_factor

        bg_color = self.get_color_under_cursor()
        if bg_color is not None:
            brightness = (bg_color.red() * 299 + bg_color.green() * 587 + bg_color.blue() * 114) / 1000
            use_light_color = brightness < 128
        else:
            use_light_color = True

        self._update_brush_outline(scene_pos, brush_size_in_scene, use_light_color)

        self.setCursor(Qt.CursorShape.CrossCursor)

    def get_color_under_cursor(self):
        cursor_pos = QCursor.pos()
        view_pos = self.viewport().mapFromGlobal(cursor_pos)

        if not self.viewport().rect().contains(view_pos):
            return None

        pixmap = self.viewport().grab(QRect(view_pos.x(), view_pos.y(), 1, 1))

        if pixmap.isNull():
            self.logger.error("Failed to grab pixmap from viewport.")
            return None

        image = pixmap.toImage()

        if image.isNull():
            self.logger.error("Failed to convert pixmap to image.")
            return None

        return QColor(image.pixelColor(0, 0))

    def get_color_under_point(self, point):
        global_point = self.mapToGlobal(point)

        screen = QGuiApplication.primaryScreen()
        pixmap = screen.grabWindow(0, global_point.x(), global_point.y(), 1, 1)
        image = pixmap.toImage()
        color = QColor(image.pixelColor(0, 0))

        return color

    def set_brush_color(self, color: tuple):
        self.brush_color = QColor(int(color[0]), int(color[1]), int(color[2]), 255)

    def set_brush_size(self, value):
        self.brush_size = value
        self.show_brush_preview()

    def set_brush_hardness(self, value):
        self.hardness = value
        self.show_brush_preview()

    def set_brush_steps(self, value):
        self.steps = max(0.1, min(10.0, value))

    def show_brush_preview(self):
        if self.brush_preview is not None:
            self.scene.removeItem(self.brush_preview)
            self.brush_preview = None

        center = self.sceneRect().center()

        brightness = (
            self.brush_color.red() * 299 + self.brush_color.green() * 587 + self.brush_color.blue() * 114
        ) / 1000

        if brightness < 128:
            brush_preview_color = Qt.GlobalColor.white
        else:
            brush_preview_color = Qt.GlobalColor.black

        self.brush_preview = QGraphicsEllipseItem(
            center.x() - self.brush_size / 2, center.y() - self.brush_size / 2, self.brush_size, self.brush_size
        )
        self.brush_preview.setPen(QPen(brush_preview_color))

        gradient = QRadialGradient(center, self.brush_size / 2)
        gradient.setColorAt(0, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 255))
        gradient.setColorAt(self.hardness, self.brush_color)
        gradient.setColorAt(1, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 0))

        self.brush_preview.setBrush(QBrush(gradient))

        z_value = len(self.layer_manager.layers) + 1
        self.brush_preview.setZValue(z_value)
        self.scene.addItem(self.brush_preview)

        self.brush_preview_timer.start(500)

    def hide_brush_preview(self):
        if self.brush_preview is not None:
            self.scene.removeItem(self.brush_preview)
            self.brush_preview = None

    def reset_view(self):
        self.scale(1 / self.current_scale_factor, 1 / self.current_scale_factor)
        self.setSceneRect(self.sceneRect().translated(-self.total_translation.x(), -self.total_translation.y()))
        self.current_scale_factor = 1.0
        self.total_translation = QPointF(0, 0)

    def get_scene_as_pixmap(self):
        pixmap = QPixmap(self.sceneRect().size().toSize())
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.render(painter)
        painter.end()

        return pixmap

    def set_enable_copy(self, enable: bool):
        self.enable_copy = enable

    def set_enable_save(self, enable: bool):
        self.enable_save = enable

    def dragMoveEvent(self, event):
        pass

    def dragEnterEvent(self, event):
        pass

    def dropEvent(self, event):
        pass
