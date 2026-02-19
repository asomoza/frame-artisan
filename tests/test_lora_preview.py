"""Tests for LoRA preview, trigger words, and example graph features."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

from frameartisan.app.event_bus import EventBus
from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.modules.generation.lora.lora_preview_utils import extract_preview_clip


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_data(**overrides):
    defaults = dict(
        root_filename="test.safetensors",
        filepath="/tmp/test.safetensors",
        name="TestLoRA",
        version="1.0",
        model_type=1,
        hash="abc123",
        tags="anime, style",
        id=42,
    )
    defaults.update(overrides)
    return ModelItemDataObject(**defaults)


def _make_model_item(model_data=None):
    """Create a minimal ModelItemWidget mock."""
    if model_data is None:
        model_data = _make_model_data()
    item = MagicMock()
    item.model_data = model_data
    from PyQt6.QtGui import QPixmap

    item.pixmap = QPixmap(10, 10)
    return item


def _create_test_video(path, num_frames=40, width=64, height=64, fps=15):
    """Create a tiny test video with PyAV."""
    import av
    import numpy as np

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        arr = np.full((height, width, 3), fill_value=(i * 6) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ---------------------------------------------------------------------------
# TestPreviewClipExtraction
# ---------------------------------------------------------------------------


class TestPreviewClipExtraction:
    def test_creates_mp4_from_video(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        output_path = str(tmp_path / "preview.mp4")
        _create_test_video(video_path, num_frames=40)

        result = extract_preview_clip(video_path, 0, output_path)

        assert result is True
        assert os.path.exists(output_path)

        # Verify it's a valid video with PyAV
        import av

        with av.open(output_path) as c:
            stream = c.streams.video[0]
            assert stream.codec_context.width <= 346  # max 345, even-rounded
            assert stream.codec_context.height <= 346

    def test_handles_missing_video(self, tmp_path):
        output_path = str(tmp_path / "preview.mp4")
        result = extract_preview_clip("/nonexistent/video.mp4", 0, output_path)
        assert result is False
        assert not os.path.exists(output_path)

    def test_short_video_succeeds(self, tmp_path):
        video_path = str(tmp_path / "short.mp4")
        output_path = str(tmp_path / "preview.mp4")
        _create_test_video(video_path, num_frames=5)

        result = extract_preview_clip(video_path, 0, output_path)

        assert result is True
        assert os.path.exists(output_path)

    def test_output_fits_max_dimension(self, tmp_path):
        video_path = str(tmp_path / "wide.mp4")
        output_path = str(tmp_path / "preview.mp4")
        _create_test_video(video_path, width=1280, height=720)

        result = extract_preview_clip(video_path, 0, output_path)

        assert result is True
        import av

        with av.open(output_path) as c:
            stream = c.streams.video[0]
            # Long side should be ~345
            assert max(stream.codec_context.width, stream.codec_context.height) <= 346

    def test_clip_is_smaller_than_source(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        output_path = str(tmp_path / "preview.mp4")
        _create_test_video(video_path, num_frames=60, width=512, height=512)

        extract_preview_clip(video_path, 0, output_path)

        src_size = os.path.getsize(video_path)
        clip_size = os.path.getsize(output_path)
        assert clip_size < src_size


# ---------------------------------------------------------------------------
# TestTriggerDisplay
# ---------------------------------------------------------------------------


class TestTriggerDisplay:
    def test_triggers_shown_as_buttons(self, qapp):
        from frameartisan.modules.generation.lora.lora_info_widget import LoraInfoWidget

        model_data = _make_model_data(triggers="style1, style2, style3")
        model_item = _make_model_item(model_data)

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()

        widget = LoraInfoWidget(model_item, dirs)

        assert widget.triggers_layout.count() == 3
        labels = []
        for i in range(widget.triggers_layout.count()):
            item = widget.triggers_layout.itemAt(i)
            labels.append(item.widget().text())
        assert labels == ["style1", "style2", "style3"]

    def test_trigger_click_publishes_event(self, qapp):
        from frameartisan.modules.generation.lora.lora_info_widget import LoraInfoWidget

        model_data = _make_model_data(triggers="my_trigger")
        model_item = _make_model_item(model_data)

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()

        widget = LoraInfoWidget(model_item, dirs)

        received = []
        EventBus().subscribe("lora", lambda data: received.append(data))

        button = widget.triggers_layout.itemAt(0).widget()
        button.click()

        assert len(received) >= 1
        trigger_events = [e for e in received if e.get("action") == "trigger_clicked"]
        assert len(trigger_events) == 1
        assert trigger_events[0]["trigger"] == "my_trigger"

    def test_empty_triggers_hides_label(self, qapp):
        from frameartisan.modules.generation.lora.lora_info_widget import LoraInfoWidget

        model_data = _make_model_data(triggers=None)
        model_item = _make_model_item(model_data)

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()

        widget = LoraInfoWidget(model_item, dirs)

        assert widget.trigger_label.isHidden()
        assert widget.triggers_layout.count() == 0


# ---------------------------------------------------------------------------
# TestExampleGraph
# ---------------------------------------------------------------------------


class TestExampleGraph:
    def test_example_button_hidden_when_no_example(self, qapp):
        from frameartisan.modules.generation.lora.lora_info_widget import LoraInfoWidget

        model_data = _make_model_data(example=None)
        model_item = _make_model_item(model_data)

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()

        widget = LoraInfoWidget(model_item, dirs)
        assert widget.example_button.isHidden()

    def test_example_button_visible_when_example_exists(self, qapp):
        from frameartisan.modules.generation.lora.lora_info_widget import LoraInfoWidget

        graph_json = json.dumps({"nodes": [], "connections": []})
        model_data = _make_model_data(example=graph_json)
        model_item = _make_model_item(model_data)

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()

        widget = LoraInfoWidget(model_item, dirs)
        assert not widget.example_button.isHidden()

    def test_example_click_publishes_event(self, qapp):
        from frameartisan.modules.generation.lora.lora_info_widget import LoraInfoWidget

        graph_json = json.dumps({"nodes": [{"class": "test"}], "connections": []})
        model_data = _make_model_data(example=graph_json)
        model_item = _make_model_item(model_data)

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()

        widget = LoraInfoWidget(model_item, dirs)

        received = []
        EventBus().subscribe("generate", lambda data: received.append(data))

        widget.example_button.click()

        gen_events = [e for e in received if e.get("action") == "generate_from_json"]
        assert len(gen_events) == 1
        assert gen_events[0]["json_graph"] == graph_json

    def test_example_captured_on_set_preview(self, qapp):
        from frameartisan.modules.generation.lora.lora_edit_widget import LoraEditWidget

        model_data = _make_model_data()

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()
        os.makedirs(os.path.join(dirs.data_path, "loras"), exist_ok=True)

        video_viewer = MagicMock()
        video_viewer._video_widget = None
        video_viewer.source_path = None

        graph_json = '{"nodes": [], "connections": []}'
        get_json_graph = MagicMock(return_value=graph_json)

        from PyQt6.QtGui import QPixmap

        widget = LoraEditWidget(dirs, model_data, QPixmap(10, 10), video_viewer, get_json_graph)
        widget.set_model_preview()

        assert model_data.example == graph_json
        get_json_graph.assert_called_once()

    def test_example_patches_temp_source_image_path(self, qapp, tmp_path):
        from frameartisan.modules.generation.lora.lora_edit_widget import LoraEditWidget

        model_data = _make_model_data()

        # Set up directories with outputs_source_images
        source_images_dir = tmp_path / "outputs" / "source_images"
        source_images_dir.mkdir(parents=True)
        permanent_img = source_images_dir / "ltx2_20260101_120000.png"
        permanent_img.write_bytes(b"fake png")

        dirs = MagicMock()
        dirs.data_path = str(tmp_path)
        dirs.outputs_source_images = str(source_images_dir)
        os.makedirs(os.path.join(str(tmp_path), "loras"), exist_ok=True)

        video_viewer = MagicMock()
        video_viewer._video_widget = None
        video_viewer.source_path = str(tmp_path / "videos" / "ltx2_20260101_120000.mp4")

        temp_path = "/tmp/source_image_abc123.png"
        graph_json = json.dumps(
            {
                "nodes": [
                    {"class": "ImageLoadNode", "state": {"path": temp_path}},
                    {"class": "OtherNode", "state": {}},
                ],
                "connections": [],
            }
        )
        get_json_graph = MagicMock(return_value=graph_json)

        from PyQt6.QtGui import QPixmap

        widget = LoraEditWidget(dirs, model_data, QPixmap(10, 10), video_viewer, get_json_graph)
        widget.set_model_preview()

        saved = json.loads(model_data.example)
        assert saved["nodes"][0]["state"]["path"] == str(permanent_img)

    def test_example_keeps_path_when_no_permanent_image(self, qapp, tmp_path):
        from frameartisan.modules.generation.lora.lora_edit_widget import LoraEditWidget

        model_data = _make_model_data()

        dirs = MagicMock()
        dirs.data_path = str(tmp_path)
        dirs.outputs_source_images = str(tmp_path / "outputs" / "source_images")
        os.makedirs(os.path.join(str(tmp_path), "loras"), exist_ok=True)

        video_viewer = MagicMock()
        video_viewer._video_widget = None
        video_viewer.source_path = str(tmp_path / "videos" / "ltx2_20260101_120000.mp4")

        temp_path = "/tmp/source_image_abc123.png"
        graph_json = json.dumps(
            {
                "nodes": [{"class": "ImageLoadNode", "state": {"path": temp_path}}],
                "connections": [],
            }
        )
        get_json_graph = MagicMock(return_value=graph_json)

        from PyQt6.QtGui import QPixmap

        widget = LoraEditWidget(dirs, model_data, QPixmap(10, 10), video_viewer, get_json_graph)
        widget.set_model_preview()

        # No permanent image exists, so path should remain unchanged
        saved = json.loads(model_data.example)
        assert saved["nodes"][0]["state"]["path"] == temp_path

    def test_triggers_saved_to_db(self, qapp):
        from frameartisan.modules.generation.lora.lora_edit_widget import LoraEditWidget

        model_data = _make_model_data()

        dirs = MagicMock()
        dirs.data_path = tempfile.mkdtemp()
        os.makedirs(os.path.join(dirs.data_path, "loras"), exist_ok=True)

        video_viewer = MagicMock()
        video_viewer._video_widget = None

        from PyQt6.QtGui import QPixmap

        widget = LoraEditWidget(dirs, model_data, QPixmap(10, 10), video_viewer)
        widget.triggers_edit.setPlainText("trigger1, trigger2")

        with patch("frameartisan.modules.generation.lora.lora_edit_widget.Database") as MockDB:
            mock_db = MagicMock()
            MockDB.return_value = mock_db

            widget.save_lora_info()

            mock_db.update.assert_called_once()
            call_args = mock_db.update.call_args
            update_dict = call_args[0][1]
            assert update_dict["triggers"] == "trigger1, trigger2"
