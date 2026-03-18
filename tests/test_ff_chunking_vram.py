"""Heavy GPU test: verify FF chunking works with each offload strategy.

Runs single-stage inference with the 4-bit distilled model and FF chunking
enabled for each offload strategy.  Each test is independent and should be
run separately to avoid excessive GPU memory pressure::

    GPU_HEAVY=1 uv run --extra test pytest tests/test_ff_chunking_vram.py -v -s -k model_offload
    GPU_HEAVY=1 uv run --extra test pytest tests/test_ff_chunking_vram.py -v -s -k group_offload_no_stream
    GPU_HEAVY=1 uv run --extra test pytest tests/test_ff_chunking_vram.py -v -s -k group_offload_stream
    GPU_HEAVY=1 uv run --extra test pytest tests/test_ff_chunking_vram.py -v -s -k seq_group_offload_no_stream
    GPU_HEAVY=1 uv run --extra test pytest tests/test_ff_chunking_vram.py -v -s -k seq_group_offload_stream

Activated by setting the environment variable ``GPU_HEAVY=1``.
"""

from __future__ import annotations

import gc
import logging
import os
from collections import deque

import attr
import pytest
import torch

from frameartisan.app.app import set_app_database_path
from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
from frameartisan.modules.generation.generation_settings import GenerationSettings
from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph
from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.utils.database import Database

# Suppress noisy "unexecuted_layers" warnings from diffusers group offloading
logging.getLogger("diffusers.hooks.group_offloading").setLevel(logging.ERROR)

pytestmark = pytest.mark.gpu_heavy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@attr.s
class _FakeDirs:
    outputs_videos: str = attr.ib(default="")


def _discover_model(db_path: str, version: str) -> dict | None:
    db = Database(db_path)
    rows = db.fetch_all(
        "SELECT id, name, filepath, model_type, version FROM model "
        "WHERE deleted = 0 AND model_type = 2 AND version = ?",
        (version,),
    )
    db.disconnect()
    if not rows:
        return None
    row_id, name, filepath, model_type, ver = rows[0]
    return {"id": row_id, "name": name, "filepath": filepath, "model_type": model_type, "version": ver}


def _topo_sort(graph):
    sorted_nodes: deque = deque()
    visited: set = set()
    visiting: set = set()

    def dfs(node):
        visiting.add(node)
        for dep in sorted(node.dependencies, key=lambda x: x.PRIORITY, reverse=True):
            if dep in visiting:
                raise ValueError("Graph contains a cycle")
            if dep not in visited:
                dfs(dep)
        visiting.remove(node)
        visited.add(node)
        sorted_nodes.append(node)

    for node in sorted(graph.nodes, key=lambda x: x.PRIORITY, reverse=True):
        if node not in visited:
            dfs(node)

    return sorted_nodes


def _full_cleanup():
    get_model_manager().clear()
    set_global_model_manager(ModelManager())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_single_stage(
    tmp_dir: str,
    model_info: dict,
    offload_strategy: str,
    group_offload_use_stream: bool = False,
    group_offload_low_cpu_mem: bool = False,
) -> list[str]:
    """Run single-stage inference with FF chunking and return video paths."""
    device = torch.device("cuda")
    mm = get_model_manager()

    model_do = ModelDataObject(
        name=model_info["name"],
        filepath=model_info["filepath"],
        model_type=model_info["model_type"],
        id=model_info["id"],
    )

    settings = GenerationSettings(
        model=model_do,
        video_width=512,
        video_height=320,
        video_duration=2,
        frame_rate=24,
        num_inference_steps=8,
        guidance_scale=1.0,
        offload_strategy=offload_strategy,
        group_offload_use_stream=group_offload_use_stream,
        group_offload_low_cpu_mem=group_offload_low_cpu_mem,
        ff_chunking=True,
    )

    dirs = _FakeDirs(outputs_videos=tmp_dir)
    graph = create_default_ltx2_graph(settings, dirs)
    graph.device = device
    graph.dtype = torch.bfloat16

    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = "a red ball bouncing on a table"

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    sorted_nodes = _topo_sort(graph)

    with mm.device_scope(device=device, dtype=torch.bfloat16):
        with torch.inference_mode():
            for node in sorted_nodes:
                if not (node.updated and node.enabled):
                    continue

                node.device = device
                node.dtype = torch.bfloat16
                node_name = node.name or node.__class__.__name__

                try:
                    node()
                except (RuntimeError, NodeError) as e:
                    if "out of memory" in str(e).lower():
                        del graph
                        _full_cleanup()
                        pytest.skip(f"OOM during {node_name}: {e}")
                    raise

                node.updated = False

            mm.apply_offload_strategy(device)

    return video_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_4bit(app_db_path):
    model = _discover_model(app_db_path, "4bit")
    if model is None:
        pytest.skip("LTX2 Distilled SDNQ 4-bit not found in DB")
    return model


@pytest.fixture(autouse=True)
def setup_cleanup(app_db_path):
    set_app_database_path(app_db_path)
    set_global_model_manager(ModelManager())
    yield
    _full_cleanup()


# ---------------------------------------------------------------------------
# Tests — run individually to avoid GPU memory pressure
# ---------------------------------------------------------------------------


def test_ff_chunking_model_offload(tmp_path, model_4bit):
    videos = _run_single_stage(
        str(tmp_path),
        model_4bit,
        offload_strategy="model_offload",
    )
    assert len(videos) == 1
    assert os.path.isfile(videos[0])


def test_ff_chunking_group_offload_no_stream(tmp_path, model_4bit):
    videos = _run_single_stage(
        str(tmp_path),
        model_4bit,
        offload_strategy="group_offload",
    )
    assert len(videos) == 1
    assert os.path.isfile(videos[0])


def test_ff_chunking_group_offload_stream(tmp_path, model_4bit):
    videos = _run_single_stage(
        str(tmp_path),
        model_4bit,
        offload_strategy="group_offload",
        group_offload_use_stream=True,
    )
    assert len(videos) == 1
    assert os.path.isfile(videos[0])


def test_ff_chunking_seq_group_offload_no_stream(tmp_path, model_4bit):
    videos = _run_single_stage(
        str(tmp_path),
        model_4bit,
        offload_strategy="sequential_group_offload",
    )
    assert len(videos) == 1
    assert os.path.isfile(videos[0])


def test_ff_chunking_seq_group_offload_stream(tmp_path, model_4bit):
    videos = _run_single_stage(
        str(tmp_path),
        model_4bit,
        offload_strategy="sequential_group_offload",
        group_offload_use_stream=True,
    )
    assert len(videos) == 1
    assert os.path.isfile(videos[0])
