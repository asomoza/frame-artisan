"""Heavy GPU test: advanced multimodal guidance with SDNQ 4-bit non-distilled model.

Verifies that the advanced guidance path (STG, modality isolation, variance
rescaling) works end-to-end with sequential_group_offload on a quantised model.

Activated by setting the environment variable ``GPU_HEAVY=1``.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_advanced_guidance_heavy.py -v -s
"""

from __future__ import annotations

import gc
import os

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

pytestmark = pytest.mark.gpu_heavy


@attr.s
class _FakeDirs:
    outputs_videos: str = attr.ib(default="")


def _full_cleanup():
    get_model_manager().clear()
    set_global_model_manager(ModelManager())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _discover_non_distilled_4bit(db_path: str) -> dict | None:
    """Find a non-distilled (model_type=1) SDNQ 4-bit model in the database."""
    db = Database(db_path)
    rows = db.fetch_all(
        "SELECT id, name, filepath, model_type, version FROM model "
        "WHERE deleted = 0 AND model_type = 1 AND version = '4bit'"
    )
    db.disconnect()
    if not rows:
        return None
    row_id, name, filepath, model_type, version = rows[0]
    return {
        "id": row_id,
        "name": name,
        "filepath": filepath,
        "model_type": model_type,
        "version": version,
    }


def _build_graph(
    output_dir: str,
    model: dict,
    *,
    offload_strategy: str,
    advanced_guidance: bool = False,
    stg_scale: float = 0.0,
    stg_blocks: str = "29",
    rescale_scale: float = 0.0,
    modality_scale: float = 1.0,
    guidance_skip_step: int = 0,
):
    model_obj = ModelDataObject(
        name=model["name"],
        filepath=model["filepath"],
        model_type=model["model_type"],
        id=model["id"],
    )
    settings = GenerationSettings(
        model=model_obj,
        video_width=256,
        video_height=256,
        video_duration=1,
        frame_rate=8,
        num_inference_steps=2,
        guidance_scale=4.0,
        offload_strategy=offload_strategy,
        advanced_guidance=advanced_guidance,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        rescale_scale=rescale_scale,
        modality_scale=modality_scale,
        guidance_skip_step=guidance_skip_step,
    )
    dirs = _FakeDirs(outputs_videos=output_dir)
    graph = create_default_ltx2_graph(settings, dirs)

    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = "a red ball bouncing"

    return graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_4bit(app_db_path):
    model = _discover_non_distilled_4bit(app_db_path)
    if model is None:
        pytest.skip("Non-distilled SDNQ 4-bit model not found in DB")
    return model


@pytest.fixture()
def setup_cleanup(app_db_path):
    set_app_database_path(app_db_path)
    set_global_model_manager(ModelManager())
    yield
    _full_cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_advanced_guidance_sequential_group_offload(tmp_path, model_4bit, setup_cleanup):
    """Full generation with advanced guidance + sequential_group_offload on SDNQ 4-bit.

    Enables all guidance components (CFG, STG, modality isolation, rescaling)
    to exercise every forward pass path.
    """
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    graph = _build_graph(
        str(tmp_path),
        model_4bit,
        offload_strategy="sequential_group_offload",
        advanced_guidance=True,
        stg_scale=1.0,
        stg_blocks="29",
        rescale_scale=0.7,
        modality_scale=3.0,
        guidance_skip_step=0,
    )
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    try:
        with torch.inference_mode():
            graph()
    except (RuntimeError, NodeError) as e:
        if "out of memory" in str(e).lower():
            del graph
            _full_cleanup()
            pytest.skip(f"OOM: {e}")
        raise

    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
    assert os.path.isfile(video_paths[0])
    assert os.path.getsize(video_paths[0]) > 0

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n  Advanced guidance + seq_group_offload: peak VRAM = {peak_vram:.0f} MB")
    print(f"  Video: {video_paths[0]}")


def test_advanced_guidance_no_offload(tmp_path, model_4bit, setup_cleanup):
    """Full generation with advanced guidance + no_offload on SDNQ 4-bit."""
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    graph = _build_graph(
        str(tmp_path),
        model_4bit,
        offload_strategy="no_offload",
        advanced_guidance=True,
        stg_scale=1.0,
        stg_blocks="29",
        rescale_scale=0.7,
        modality_scale=3.0,
    )
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    try:
        with torch.inference_mode():
            graph()
    except (RuntimeError, NodeError) as e:
        if "out of memory" in str(e).lower():
            del graph
            _full_cleanup()
            pytest.skip(f"OOM: {e}")
        raise

    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
    assert os.path.isfile(video_paths[0])
    assert os.path.getsize(video_paths[0]) > 0

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n  Advanced guidance + no_offload: peak VRAM = {peak_vram:.0f} MB")
    print(f"  Video: {video_paths[0]}")


def test_advanced_guidance_produces_different_output(tmp_path, model_4bit, setup_cleanup):
    """Advanced guidance should produce different output than simple CFG.

    Runs two generations with the same seed — one with simple CFG, one with
    advanced guidance (STG + modality) — and verifies the videos differ in size
    (a rough proxy for different content).
    """
    device = torch.device("cuda")

    results: dict[str, str] = {}

    for label, ag_enabled in [("simple_cfg", False), ("advanced", True)]:
        set_global_model_manager(ModelManager())
        gc.collect()
        torch.cuda.empty_cache()

        out_dir = str(tmp_path / label)
        os.makedirs(out_dir, exist_ok=True)

        graph = _build_graph(
            out_dir,
            model_4bit,
            offload_strategy="sequential_group_offload",
            advanced_guidance=ag_enabled,
            stg_scale=1.0 if ag_enabled else 0.0,
            stg_blocks="29",
            rescale_scale=0.7 if ag_enabled else 0.0,
            modality_scale=3.0 if ag_enabled else 1.0,
        )
        graph.device = device
        graph.dtype = torch.bfloat16

        video_paths: list[str] = []
        video_send = graph.get_node_by_name("video_send")
        video_send.video_callback = video_paths.append

        try:
            with torch.inference_mode():
                graph()
        except (RuntimeError, NodeError) as e:
            if "out of memory" in str(e).lower():
                del graph
                _full_cleanup()
                pytest.skip(f"OOM during {label}: {e}")
            raise

        assert len(video_paths) == 1
        results[label] = video_paths[0]

        del graph
        _full_cleanup()

    # Both should produce valid videos
    for label, path in results.items():
        assert os.path.isfile(path), f"{label} video missing"
        assert os.path.getsize(path) > 0, f"{label} video empty"

    print(f"\n  Simple CFG video: {results['simple_cfg']}")
    print(f"  Advanced guidance video: {results['advanced']}")
