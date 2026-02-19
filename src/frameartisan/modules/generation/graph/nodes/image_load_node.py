from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


class ImageLoadNode(Node):
    PRIORITY = 2
    OUTPUTS = ["image"]

    SERIALIZE_EXCLUDE = {"image"}

    def __init__(self, path: str | None = None, image: Optional[np.ndarray] = None, grayscale: bool = False):
        super().__init__()
        self.path = path
        self.image = image
        self.grayscale = grayscale

    def update_value(self, path: str):
        self.path = path
        self.set_updated()

    def update_image(self, image: np.ndarray):
        self.image = image
        self.set_updated()

    def update_path(self, path: str):
        self.path = path

        try:
            self.image = self._load_numpy(self.path, grayscale=self.grayscale)
        except Exception as e:
            raise NodeError(e, self.__class__.__name__)

        self.set_updated()

    @staticmethod
    def _load_numpy(path: str, *, grayscale: bool = False) -> np.ndarray:
        if not path:
            raise FileNotFoundError("No image path provided")

        if grayscale:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(path)

            # If image has alpha channel (4 channels), extract it
            # Otherwise convert to grayscale
            if raw.ndim == 3 and raw.shape[2] == 4:
                gray = raw[:, :, 3]  # Alpha channel
            elif raw.ndim == 3:
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            else:
                gray = raw

            mask_image = gray.astype(np.float32) / 255.0
            mask_image = np.expand_dims(mask_image, axis=-1)

            return np.ascontiguousarray(mask_image)

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def __call__(self):
        if self.image is None:
            try:
                image = self._load_numpy(self.path, grayscale=self.grayscale)
            except Exception as e:
                raise NodeError(e, self.__class__.__name__)
        else:
            image = self.image

        self.values["image"] = image

        return self.values
