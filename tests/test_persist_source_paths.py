"""Tests for persist_source_paths_in_graph (source file deduplication & path rewriting)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from frameartisan.utils.database import Database
from frameartisan.utils.json_utils import persist_source_paths_in_graph


def _create_db(db_path: str) -> Database:
    """Create a fresh database with the source_file table."""
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


def _make_graph_json(*nodes: dict) -> str:
    return json.dumps({"format_version": 1, "nodes": list(nodes), "connections": []})


def _image_node(name: str, path: str, enabled: bool = True) -> dict:
    return {
        "class": "ImageLoadNode",
        "name": name,
        "enabled": enabled,
        "state": {"path": path},
    }


def _audio_node(path: str, enabled: bool = True) -> dict:
    return {
        "class": "LTX2AudioEncodeNode",
        "name": "audio_encode",
        "enabled": enabled,
        "state": {"audio_path": path, "trim_start_s": 0, "trim_end_s": 0},
    }


def _video_node(name: str, path: str, enabled: bool = True) -> dict:
    return {
        "class": "VideoLoadNode",
        "name": name,
        "enabled": enabled,
        "state": {"path": path},
    }


@pytest.fixture()
def env(tmp_path):
    """Set up temp dirs, db, and a sample source image."""
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
    # Clear any stale thread-local DB connection from prior tests
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

    # Clean up thread-local DB connection so it doesn't leak between tests
    db.disconnect()


def _call_persist(env, graph_json: str) -> str:
    """Call persist_source_paths_in_graph with the test db patched in."""
    with patch("frameartisan.app.app.get_app_database_path", return_value=env["db_path"]):
        return persist_source_paths_in_graph(
            graph_json,
            source_image_dir=str(env["out_images"]),
            source_audio_dir=str(env["out_audio"]),
            source_video_dir=str(env["out_video"]),
        )


class TestPersistSourcePaths:
    """Core persist + dedup behaviour."""

    def test_image_copied_and_path_rewritten(self, env):
        graph = _make_graph_json(_image_node("source_image_abc", str(env["img"])))
        result = _call_persist(env, graph)

        data = json.loads(result)
        new_path = data["nodes"][0]["state"]["path"]
        assert new_path != str(env["img"])
        assert str(env["out_images"]) in new_path
        assert os.path.isfile(new_path)
        assert Path(new_path).read_bytes() == env["img"].read_bytes()

    def test_audio_copied_and_path_rewritten(self, env):
        graph = _make_graph_json(_audio_node(str(env["audio"])))
        result = _call_persist(env, graph)

        data = json.loads(result)
        new_path = data["nodes"][0]["state"]["audio_path"]
        assert new_path != str(env["audio"])
        assert str(env["out_audio"]) in new_path
        assert os.path.isfile(new_path)

    def test_video_copied_and_path_rewritten(self, env):
        graph = _make_graph_json(_video_node("condition_video", str(env["video"])))
        result = _call_persist(env, graph)

        data = json.loads(result)
        new_path = data["nodes"][0]["state"]["path"]
        assert new_path != str(env["video"])
        assert str(env["out_video"]) in new_path
        assert os.path.isfile(new_path)

    def test_dedup_reuses_existing_file(self, env):
        """Second call with same content should reuse the file, not create a new one."""
        graph = _make_graph_json(_image_node("source_image_abc", str(env["img"])))

        result1 = _call_persist(env, graph)
        path1 = json.loads(result1)["nodes"][0]["state"]["path"]

        # Call again — same source file, should dedup
        result2 = _call_persist(env, graph)
        path2 = json.loads(result2)["nodes"][0]["state"]["path"]

        assert path1 == path2
        # Only one file should exist in the output dir
        files = list(env["out_images"].glob("*"))
        assert len(files) == 1

    def test_dedup_different_content_creates_new_file(self, env):
        """Different content for same kind should create a second file."""
        graph1 = _make_graph_json(_image_node("source_image_abc", str(env["img"])))
        result1 = _call_persist(env, graph1)
        path1 = json.loads(result1)["nodes"][0]["state"]["path"]

        # Write different content to a new source file
        img2 = env["tmp_path"] / "user_files" / "photo2.png"
        img2.write_bytes(b"different-content-9999")
        graph2 = _make_graph_json(_image_node("source_image_abc", str(img2)))
        result2 = _call_persist(env, graph2)
        path2 = json.loads(result2)["nodes"][0]["state"]["path"]

        assert path1 != path2
        files = list(env["out_images"].glob("*"))
        assert len(files) == 2

    def test_stale_db_entry_cleaned_up(self, env):
        """If the DB points to a deleted file, the entry should be removed and a fresh copy made."""
        graph = _make_graph_json(_image_node("source_image_abc", str(env["img"])))

        result1 = _call_persist(env, graph)
        path1 = json.loads(result1)["nodes"][0]["state"]["path"]

        # Delete the persisted file to simulate stale entry
        os.remove(path1)

        result2 = _call_persist(env, graph)
        path2 = json.loads(result2)["nodes"][0]["state"]["path"]

        assert os.path.isfile(path2)
        # A new file was created (old one was deleted)
        assert path1 == path2 or os.path.isfile(path2)

    def test_already_in_target_dir_no_op(self, env):
        """If the source is already in the target directory, no copy is made."""
        env["out_images"].mkdir(parents=True, exist_ok=True)
        already_there = env["out_images"] / "existing.png"
        already_there.write_bytes(b"already-there")

        graph = _make_graph_json(_image_node("source_image_abc", str(already_there)))
        result = _call_persist(env, graph)

        # Path should be unchanged
        data = json.loads(result)
        assert data["nodes"][0]["state"]["path"] == str(already_there)
        # Still only one file
        assert len(list(env["out_images"].glob("*"))) == 1


class TestPersistEdgeCases:
    """Edge cases and skip conditions."""

    def test_missing_source_file_unchanged(self, env):
        """If the source file doesn't exist, the path should stay unchanged."""
        graph = _make_graph_json(_image_node("source_image_abc", "/nonexistent/photo.png"))
        result = _call_persist(env, graph)

        data = json.loads(result)
        assert data["nodes"][0]["state"]["path"] == "/nonexistent/photo.png"

    def test_no_dir_configured_skips(self, env):
        """If source_image_dir is None, images are skipped."""
        graph = _make_graph_json(_image_node("source_image_abc", str(env["img"])))
        with patch("frameartisan.app.app.get_app_database_path", return_value=env["db_path"]):
            result = persist_source_paths_in_graph(
                graph,
                source_image_dir=None,
                source_audio_dir=str(env["out_audio"]),
                source_video_dir=str(env["out_video"]),
            )

        data = json.loads(result)
        assert data["nodes"][0]["state"]["path"] == str(env["img"])

    def test_non_matching_node_name_skipped(self, env):
        """ImageLoadNode not named source_image_* should be left alone."""
        graph = _make_graph_json(_image_node("some_other_node", str(env["img"])))
        result = _call_persist(env, graph)

        data = json.loads(result)
        assert data["nodes"][0]["state"]["path"] == str(env["img"])

    def test_invalid_json_returned_unchanged(self, env):
        result = _call_persist(env, "not valid json {{{")
        assert result == "not valid json {{{"

    def test_multiple_images_persisted(self, env):
        """Multiple condition images should each get their own persisted copy."""
        img2 = env["tmp_path"] / "user_files" / "photo2.png"
        img2.write_bytes(b"second-image-content")

        graph = _make_graph_json(
            _image_node("source_image_aaa", str(env["img"])),
            _image_node("source_image_bbb", str(img2)),
        )
        result = _call_persist(env, graph)

        data = json.loads(result)
        path_a = data["nodes"][0]["state"]["path"]
        path_b = data["nodes"][1]["state"]["path"]
        assert path_a != path_b
        assert os.path.isfile(path_a)
        assert os.path.isfile(path_b)
        assert len(list(env["out_images"].glob("*"))) == 2

    def test_no_db_still_copies(self, env):
        """Without a DB, files should still be copied (just no dedup)."""
        graph = _make_graph_json(_image_node("source_image_abc", str(env["img"])))
        with patch("frameartisan.app.app.get_app_database_path", return_value=None):
            result = persist_source_paths_in_graph(
                graph,
                source_image_dir=str(env["out_images"]),
            )

        data = json.loads(result)
        new_path = data["nodes"][0]["state"]["path"]
        assert os.path.isfile(new_path)
        assert str(env["out_images"]) in new_path
