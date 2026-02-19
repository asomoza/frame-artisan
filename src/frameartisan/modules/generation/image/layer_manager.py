from PyQt6.QtGui import QPixmap

from frameartisan.modules.generation.image.image_editor_layer import ImageEditorLayer
from frameartisan.modules.generation.image.layer_pixmap_item import LayerPixmapItem


class LayerManager:
    def __init__(self, target_width: int, target_height: int, temp_path: str):
        self.layers = []
        self.next_layer_id = 0
        self.target_width = target_width
        self.target_height = target_height
        self.temp_path = temp_path

    def shift_order(self, order: int):
        for layer in self.layers:
            if layer.order >= order:
                layer.order += 1

    def add_new_layer(
        self, image_path: str = None, base_path: str = None, locked: bool = True, order: int = None
    ) -> ImageEditorLayer:
        if order is not None:
            self.shift_order(order)
        else:
            order = max(layer.order for layer in self.layers) + 1 if self.layers else 0

        if self.next_layer_id == 0:
            layer_name = "Background"
        else:
            layer_name = f"Layer #{self.next_layer_id}"

        layer = ImageEditorLayer(
            layer_id=self.next_layer_id,
            layer_name=layer_name,
            width=self.target_width,
            height=self.target_height,
            locked=locked,
            order=order,
            base_path=base_path,
        )
        layer.set_pixmap_item(image_path, self.temp_path)
        self.layers.append(layer)
        self.next_layer_id += 1

        return layer

    def add_layer_object(self, layer: ImageEditorLayer, order: int = None) -> ImageEditorLayer:
        if order is not None:
            self.shift_order(order)
            layer.order = order
        else:
            layer.order = max(layer.order for layer in self.layers) + 1 if self.layers else 0

        # Ensure the layer has a unique ID in this manager
        layer.layer_id = self.next_layer_id

        self.layers.append(layer)
        self.next_layer_id += 1

        return layer

    def reload_layer(self, pixmap: QPixmap, image_path: str, original_path: str, order: int):
        pixmap_item = LayerPixmapItem(pixmap)
        layer = ImageEditorLayer(
            pixmap_item=pixmap_item,
            image_path=image_path,
            original_path=original_path,
            locked=True,
            order=order,
        )

        layer.layer_id = self.next_layer_id
        self.layers.append(layer)
        self.next_layer_id += 1

        return layer

    def edit_layer(
        self,
        layer_id: int,
        pixmap: QPixmap = None,
        image_path: str = None,
        locked: bool = None,
        order: int = None,
    ) -> ImageEditorLayer:
        layer = self.get_layer_by_id(layer_id)
        if layer is not None:
            if pixmap is not None:
                layer.pixmap_item.setPixmap(pixmap)

            layer.image_path = image_path if image_path is not None else layer.image_path
            layer.locked = locked if locked is not None else layer.locked

            if order is not None and order != layer.order:
                self.move_layer(layer_id, order)

            return layer
        return None

    def delete_layer(self, layer_id: int):
        deleted_order = None
        for i, layer in enumerate(self.layers):
            if layer.layer_id == layer_id:
                deleted_order = layer.order
                del self.layers[i]
                break

        if deleted_order is not None:
            for layer in self.layers:
                if layer.order > deleted_order:
                    layer.order -= 1

    def move_layer(self, layer_id: int, new_order: int):
        layer = self.get_layer_by_id(layer_id)
        if layer is not None:
            self.layers.remove(layer)

            for iter_layer in self.layers:
                if iter_layer.order >= new_order:
                    iter_layer.order += 1

            layer.order = new_order
            self.layers.append(layer)

            self.reorder_layers()

    def reorder_layers(self):
        self.layers.sort(key=lambda layer: layer.order)

    def get_layers(self):
        return self.layers

    def get_layer_by_id(self, layer_id: int) -> ImageEditorLayer:
        if len(self.layers) > 0:
            for layer in self.layers:
                if layer.layer_id == layer_id:
                    return layer
        return None

    def get_layer_at_order(self, order: int) -> ImageEditorLayer:
        if len(self.layers) > 0:
            for layer in self.layers:
                if layer.order == order:
                    return layer
        return None

    def get_first_layer(self) -> ImageEditorLayer:
        if self.layers:
            return self.layers[0]
        return None

    def get_last_layer(self) -> ImageEditorLayer:
        if self.layers:
            return self.layers[-1]
        return None

    def delete_all(self):
        self.layers = []
        self.next_layer_id = 0
