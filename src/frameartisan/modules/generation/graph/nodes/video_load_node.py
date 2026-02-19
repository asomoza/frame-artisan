from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class VideoLoadNode(Node):
    """Load a video file and output all frames as a numpy array.

    Outputs ``frames`` as a numpy array of shape ``[F, H, W, 3]`` (uint8 RGB).
    """

    PRIORITY = 2
    OUTPUTS = ["frames"]

    SERIALIZE_EXCLUDE = {"frames"}

    def __init__(self, path: str | None = None, frames: Optional[np.ndarray] = None):
        super().__init__()
        self.path = path
        self.frames = frames

    def update_path(self, path: str):
        self.path = path

        try:
            self.frames = self._load_frames(self.path)
        except Exception as e:
            raise NodeError(e, self.__class__.__name__)

        self.set_updated()

    @staticmethod
    def _load_frames(path: str) -> np.ndarray:
        if not path:
            raise FileNotFoundError("No video path provided")

        import av

        frames = []
        with av.open(path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            raise NodeError("Video contains no frames", "VideoLoadNode")

        return np.stack(frames)

    @property
    def frame_count(self) -> int:
        if self.frames is not None:
            return self.frames.shape[0]
        return 0

    def __call__(self):
        if self.frames is None:
            try:
                self.frames = self._load_frames(self.path)
            except Exception as e:
                raise NodeError(e, self.__class__.__name__)

        self.values["frames"] = self.frames

        return self.values
