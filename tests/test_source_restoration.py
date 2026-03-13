"""Tests for source file restoration from saved graph metadata.

Covers both the video-drop path (_apply_loaded_graph_subset) and the
LoRA example path (lora_edit_widget → persist_source_paths_in_graph),
ensuring images, audio, and video sources are correctly saved and restored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frameartisan.utils.database import Database
from frameartisan.utils.json_utils import persist_source_paths_in_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_db(db_path: str) -> Database:
    db = Database(db_path)
    db.create_table(
        "source_file",
        [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "kind TEXT NOT NULL",
            "content_hash TEXT NOT NULL",
            "filepath TEXT NOT NULL",
            "UNIQUE(kind, content_hash)",
        ],
    )
    return db


def _make_graph_json(*nodes: dict, connections: list | None = None) -> str:
    return json.dumps({
        "format_version": 1,
        "nodes": list(nodes),
        "connections": connections or [],
    })


def _image_node(name: str, path: str, enabled: bool = True) -> dict:
    return {
        "class": "ImageLoadNode",
        "name": name,
        "enabled": enabled,
        "state": {"path": path},
    }


def _audio_node(path: str, enabled: bool = True, trim_start: float | None = None, trim_end: float | None = None) -> dict:
    return {
        "class": "LTX2AudioEncodeNode",
        "name": "audio_encode",
        "enabled": enabled,
        "state": {"audio_path": path, "trim_start_s": trim_start, "trim_end_s": trim_end},
    }


def _video_load_node(name: str, path: str, enabled: bool = True) -> dict:
    return {
        "class": "VideoLoadNode",
        "name": name,
        "enabled": enabled,
        "state": {"path": path},
    }


def _condition_encode_node(conditions: list | None = None) -> dict:
    return {
        "class": "LTX2ConditionEncodeNode",
        "name": "condition_encode",
        "enabled": True,
        "state": {"conditions": conditions or []},
    }


def _model_node() -> dict:
    return {
        "class": "LTX2ModelNode",
        "name": "model",
        "enabled": True,
        "state": {"model_id": None},
    }


@pytest.fixture()
def env(tmp_path):
    source_dir = tmp_path / "user_files"
    source_dir.mkdir()

    img = source_dir / "photo.png"
    img.write_bytes(b"fake-png-content-1234")

    audio = source_dir / "clip.wav"
    audio.write_bytes(b"fake-wav-content-5678")

    video = source_dir / "scene.mp4"
    video.write_bytes(b"fake-mp4-content-abcd")

    out_images = tmp_path / "outputs" / "source_images"
    out_audio = tmp_path / "outputs" / "source_audio"
    out_video = tmp_path / "outputs" / "source_videos"

    db_path = str(tmp_path / "app.db")
    Database(db_path).disconnect()
    db = _create_db(db_path)

    yield {
        "tmp_path": tmp_path,
        "img": img,
        "audio": audio,
        "video": video,
        "out_images": out_images,
        "out_audio": out_audio,
        "out_video": out_video,
        "db_path": db_path,
        "db": db,
    }

    db.disconnect()


def _persist(env, graph_json: str) -> str:
    with patch("frameartisan.app.app.get_app_database_path", return_value=env["db_path"]):
        return persist_source_paths_in_graph(
            graph_json,
            source_image_dir=str(env["out_images"]),
            source_audio_dir=str(env["out_audio"]),
            source_video_dir=str(env["out_video"]),
        )


# ---------------------------------------------------------------------------
# Test: LoRA example saves ALL source types via persist_source_paths_in_graph
# ---------------------------------------------------------------------------

class TestLoraExampleSourcePersistence:
    """LoRA example graph persistence uses persist_source_paths_in_graph
    which handles images, audio, and video with content-hash naming."""

    def test_lora_example_persists_source_image(self, env):
        """Source image in LoRA example graph is persisted with content-hash naming."""
        graph = _make_graph_json(
            _image_node("source_image_abc", str(env["img"])),
            _condition_encode_node([{"pixel_frame_index": 1, "strength": 0.8}]),
        )
        result = _persist(env, graph)

        data = json.loads(result)
        new_path = data["nodes"][0]["state"]["path"]
        assert str(env["out_images"]) in new_path
        assert os.path.isfile(new_path)
        # Uses content-hash naming, not video-stem naming
        assert "source_image_abc_" in os.path.basename(new_path)

    def test_lora_example_persists_audio(self, env):
        """Audio in LoRA example graph is persisted."""
        graph = _make_graph_json(
            _audio_node(str(env["audio"]), trim_start=1.0, trim_end=5.0),
        )
        result = _persist(env, graph)

        data = json.loads(result)
        new_path = data["nodes"][0]["state"]["audio_path"]
        assert str(env["out_audio"]) in new_path
        assert os.path.isfile(new_path)

    def test_lora_example_persists_video(self, env):
        """Video source in LoRA example graph is persisted."""
        graph = _make_graph_json(
            _video_load_node("condition_video", str(env["video"])),
        )
        result = _persist(env, graph)

        data = json.loads(result)
        new_path = data["nodes"][0]["state"]["path"]
        assert str(env["out_video"]) in new_path
        assert os.path.isfile(new_path)

    def test_lora_example_persists_all_sources(self, env):
        """A complex graph with image + audio + video all get persisted."""
        graph = _make_graph_json(
            _image_node("source_image_abc", str(env["img"])),
            _audio_node(str(env["audio"])),
            _video_load_node("condition_video", str(env["video"])),
            _condition_encode_node([{"pixel_frame_index": 1, "strength": 1.0}]),
        )
        result = _persist(env, graph)
        data = json.loads(result)

        img_path = data["nodes"][0]["state"]["path"]
        audio_path = data["nodes"][1]["state"]["audio_path"]
        video_path = data["nodes"][2]["state"]["path"]

        assert os.path.isfile(img_path)
        assert os.path.isfile(audio_path)
        assert os.path.isfile(video_path)
        # All in different output dirs
        assert str(env["out_images"]) in img_path
        assert str(env["out_audio"]) in audio_path
        assert str(env["out_video"]) in video_path

    def test_persisted_path_survives_resolve(self, env):
        """After persist, the path should be resolvable (file exists on disk)."""
        graph = _make_graph_json(
            _image_node("source_image_abc", str(env["img"])),
        )
        result = _persist(env, graph)
        data = json.loads(result)
        path = data["nodes"][0]["state"]["path"]

        # Simulate _resolve_path
        assert os.path.isfile(path), f"Persisted path {path} should exist on disk"

    def test_video_drop_and_lora_produce_same_naming(self, env):
        """Both flows use content-hash naming, so a file persisted by video save
        is found when the same graph is loaded from a LoRA example."""
        graph = _make_graph_json(
            _image_node("source_image_abc", str(env["img"])),
        )

        # First persist (simulates video save)
        result1 = _persist(env, graph)
        path1 = json.loads(result1)["nodes"][0]["state"]["path"]

        # Second persist (simulates LoRA example save) — same source file
        result2 = _persist(env, graph)
        path2 = json.loads(result2)["nodes"][0]["state"]["path"]

        # Both should point to the same deduplicated file
        assert path1 == path2


# ---------------------------------------------------------------------------
# Test: _apply_loaded_graph_subset restores audio and video conditions
# ---------------------------------------------------------------------------

class TestApplyLoadedGraphSubset:
    """Test that _apply_loaded_graph_subset restores audio and video conditions
    from saved graph metadata."""

    def _make_mock_module(self, tmp_path):
        """Create a minimal mock of GenerationModule with required attributes."""
        from frameartisan.app.event_bus import EventBus

        module = MagicMock()
        module.event_bus = EventBus()
        module._visual_conditions = {}
        module._audio_path = None
        module._video_condition = None

        # Track events published
        events = []
        original_publish = module.event_bus.publish

        def tracking_publish(topic, data):
            events.append((topic, data))
            original_publish(topic, data)

        module.event_bus.publish = tracking_publish
        module._events = events

        # Use real _resolve_path
        from frameartisan.modules.generation.generation_module import GenerationModule
        module._resolve_path = GenerationModule._resolve_path

        return module

    def test_audio_condition_restored(self, env):
        """Audio condition is restored from saved graph metadata."""
        module = self._make_mock_module(env["tmp_path"])

        # Persist the audio file first
        graph = _make_graph_json(
            _model_node(),
            _audio_node(str(env["audio"])),
        )
        persisted = _persist(env, graph)
        graph_data = json.loads(persisted)

        # Now call _apply_loaded_graph_subset
        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        audio_events = [(t, d) for t, d in module._events if t == "audio_condition"]
        assert len(audio_events) >= 1
        add_event = audio_events[0]
        assert add_event[1]["action"] == "add"
        assert os.path.isfile(add_event[1]["audio_path"])

    def test_audio_condition_with_trim_restored(self, env):
        """Audio condition with trim settings is restored."""
        module = self._make_mock_module(env["tmp_path"])

        graph = _make_graph_json(
            _model_node(),
            _audio_node(str(env["audio"]), trim_start=2.5, trim_end=8.0),
        )
        persisted = _persist(env, graph)
        graph_data = json.loads(persisted)

        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        audio_events = [(t, d) for t, d in module._events if t == "audio_condition"]
        # Should have add + update_trim
        assert len(audio_events) == 2
        assert audio_events[0][1]["action"] == "add"
        assert audio_events[1][1]["action"] == "update_trim"
        assert audio_events[1][1]["trim_start_s"] == 2.5
        assert audio_events[1][1]["trim_end_s"] == 8.0

    def test_disabled_audio_not_restored(self, env):
        """Disabled audio node should not be restored."""
        module = self._make_mock_module(env["tmp_path"])

        graph = _make_graph_json(
            _model_node(),
            _audio_node(str(env["audio"]), enabled=False),
        )
        persisted = _persist(env, graph)
        graph_data = json.loads(persisted)

        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        audio_events = [(t, d) for t, d in module._events if t == "audio_condition"]
        assert len(audio_events) == 0

    def test_video_condition_restored(self, env):
        """Video condition is restored from saved graph metadata."""
        module = self._make_mock_module(env["tmp_path"])

        graph = _make_graph_json(
            _model_node(),
            _video_load_node("condition_video", str(env["video"])),
            _condition_encode_node([
                {"type": "video", "pixel_frame_index": 5, "strength": 0.7, "mode": "concat",
                 "source_frame_start": 10, "source_frame_end": 50},
            ]),
        )
        persisted = _persist(env, graph)
        graph_data = json.loads(persisted)

        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        video_events = [(t, d) for t, d in module._events if t == "video_condition"]
        assert len(video_events) == 1
        evt = video_events[0][1]
        assert evt["action"] == "add"
        assert os.path.isfile(evt["video_path"])
        assert evt["pixel_frame_index"] == 5
        assert evt["strength"] == 0.7
        assert evt["mode"] == "concat"
        assert evt["source_frame_start"] == 10
        assert evt["source_frame_end"] == 50

    def test_disabled_video_not_restored(self, env):
        """Disabled video node should not be restored."""
        module = self._make_mock_module(env["tmp_path"])

        graph = _make_graph_json(
            _model_node(),
            _video_load_node("condition_video", str(env["video"]), enabled=False),
            _condition_encode_node([{"type": "video"}]),
        )
        persisted = _persist(env, graph)
        graph_data = json.loads(persisted)

        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        video_events = [(t, d) for t, d in module._events if t == "video_condition"]
        assert len(video_events) == 0

    def test_missing_audio_file_not_restored(self, env):
        """If the audio file doesn't exist, it should not be restored."""
        module = self._make_mock_module(env["tmp_path"])

        graph_data = json.loads(_make_graph_json(
            _model_node(),
            _audio_node("/nonexistent/audio.wav"),
        ))

        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        audio_events = [(t, d) for t, d in module._events if t == "audio_condition"]
        assert len(audio_events) == 0

    def test_missing_video_file_not_restored(self, env):
        """If the video file doesn't exist, it should not be restored."""
        module = self._make_mock_module(env["tmp_path"])

        graph_data = json.loads(_make_graph_json(
            _model_node(),
            _video_load_node("condition_video", "/nonexistent/video.mp4"),
            _condition_encode_node([{"type": "video"}]),
        ))

        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        video_events = [(t, d) for t, d in module._events if t == "video_condition"]
        assert len(video_events) == 0

    def test_all_sources_restored_together(self, env):
        """Image + audio + video all restored from a single graph."""
        module = self._make_mock_module(env["tmp_path"])

        graph = _make_graph_json(
            _model_node(),
            _image_node("source_image_abc", str(env["img"])),
            _audio_node(str(env["audio"])),
            _video_load_node("condition_video", str(env["video"])),
            _condition_encode_node([
                {"pixel_frame_index": 1, "strength": 1.0},
                {"type": "video", "pixel_frame_index": 1, "strength": 1.0},
            ]),
        )
        persisted = _persist(env, graph)
        graph_data = json.loads(persisted)

        from frameartisan.modules.generation.generation_module import GenerationModule
        GenerationModule._apply_loaded_graph_subset(module, graph_data)

        visual_events = [(t, d) for t, d in module._events if t == "visual_condition"]
        audio_events = [(t, d) for t, d in module._events if t == "audio_condition"]
        video_events = [(t, d) for t, d in module._events if t == "video_condition"]

        assert len(visual_events) >= 1  # at least the add
        assert len(audio_events) >= 1
        assert len(video_events) == 1
