"""Tests for VideoLoadNode."""

from __future__ import annotations

import numpy as np
import pytest

from frameartisan.modules.generation.graph.nodes.video_load_node import VideoLoadNode


class TestVideoLoadNodeMetadata:
    def test_outputs(self):
        assert VideoLoadNode.OUTPUTS == ["frames"]

    def test_serialize_exclude(self):
        assert "frames" in VideoLoadNode.SERIALIZE_EXCLUDE

    def test_priority(self):
        assert VideoLoadNode.PRIORITY == 2


class TestVideoLoadNodeInit:
    def test_default_init(self):
        node = VideoLoadNode()
        assert node.path is None
        assert node.frames is None

    def test_init_with_path(self):
        node = VideoLoadNode(path="/some/video.mp4")
        assert node.path == "/some/video.mp4"
        assert node.frames is None

    def test_init_with_frames(self):
        frames = np.zeros((10, 64, 64, 3), dtype=np.uint8)
        node = VideoLoadNode(frames=frames)
        assert node.frames is frames

    def test_frame_count_none(self):
        node = VideoLoadNode()
        assert node.frame_count == 0

    def test_frame_count_with_frames(self):
        frames = np.zeros((25, 64, 64, 3), dtype=np.uint8)
        node = VideoLoadNode(frames=frames)
        assert node.frame_count == 25


class TestVideoLoadNodeCall:
    def test_call_with_preloaded_frames(self):
        frames = np.zeros((10, 64, 64, 3), dtype=np.uint8)
        node = VideoLoadNode(frames=frames)
        result = node()
        assert "frames" in result
        assert result["frames"] is frames

    def test_call_without_frames_no_path_raises(self):
        node = VideoLoadNode()
        with pytest.raises(Exception):
            node()


class TestVideoLoadNodeUpdatePath:
    def test_update_path_sets_updated(self, tmp_path):
        """update_path with a valid video should set_updated."""
        # Create a tiny test video using PyAV
        import av

        video_path = str(tmp_path / "test.mp4")
        with av.open(video_path, "w") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.height = 64
            stream.width = 64
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "30", "preset": "ultrafast"}
            for i in range(9):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 28, dtype=np.uint8), format="rgb24")
                for pkt in stream.encode(frame):
                    container.mux(pkt)
            for pkt in stream.encode(None):
                container.mux(pkt)

        node = VideoLoadNode()
        node.update_path(video_path)

        assert node.updated is True
        assert node.frames is not None
        assert node.frames.ndim == 4
        assert node.frames.shape[0] == 9
        assert node.frames.shape[1] == 64
        assert node.frames.shape[2] == 64
        assert node.frames.shape[3] == 3
        assert node.frame_count == 9

    def test_update_path_invalid_raises(self):
        node = VideoLoadNode()
        with pytest.raises(Exception):
            node.update_path("/nonexistent/video.mp4")


class TestVideoLoadNodeSerialization:
    def test_frames_excluded_from_state(self):
        frames = np.zeros((10, 64, 64, 3), dtype=np.uint8)
        node = VideoLoadNode(path="/test.mp4", frames=frames)
        state = node.get_state()
        assert "frames" not in state
        assert "path" in state

    def test_round_trip(self):
        node = VideoLoadNode(path="/test/video.mp4")
        state = node.get_state()

        node2 = VideoLoadNode()
        node2.apply_state(state)
        assert node2.path == "/test/video.mp4"
        assert node2.frames is None  # frames not serialized


class TestVideoLoadFrames:
    def test_load_frames_static(self, tmp_path):
        """_load_frames returns correct numpy array."""
        import av

        video_path = str(tmp_path / "test.mp4")
        with av.open(video_path, "w") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.height = 32
            stream.width = 48
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "30", "preset": "ultrafast"}
            for _ in range(5):
                frame = av.VideoFrame.from_ndarray(np.zeros((32, 48, 3), dtype=np.uint8), format="rgb24")
                for pkt in stream.encode(frame):
                    container.mux(pkt)
            for pkt in stream.encode(None):
                container.mux(pkt)

        frames = VideoLoadNode._load_frames(video_path)
        assert isinstance(frames, np.ndarray)
        assert frames.dtype == np.uint8
        assert frames.shape == (5, 32, 48, 3)

    def test_load_frames_no_path_raises(self):
        with pytest.raises(FileNotFoundError):
            VideoLoadNode._load_frames("")

    def test_load_frames_nonexistent_raises(self):
        with pytest.raises(Exception):
            VideoLoadNode._load_frames("/nonexistent/file.mp4")
