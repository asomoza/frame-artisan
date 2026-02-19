"""Heavy GPU tests: run real distilled models from the user's database.

Activated by setting the environment variable ``GPU_HEAVY=1``.
Skipped otherwise.

    GPU_HEAVY=1 uv run --extra test pytest tests/ -v -m gpu_heavy -s

Tests the following models (if present in the database):
- LTX2 Distilled (bf16)          — model_type=2, version="1.0"
- LTX2 Distilled SDNQ 4-bit     — model_type=2, version="4bit"
- LTX2 Distilled SDNQ 8-bit     — model_type=2, version="8bit"

Each model is tested with three offload strategies: no_offload, model_offload, group_offload.
"""

from __future__ import annotations

import gc
import json
import logging
import os

import attr
import psutil
import pytest
import torch

from frameartisan.app.app import set_app_database_path
from frameartisan.app.model_manager import get_model_manager
from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
from frameartisan.modules.generation.generation_settings import GenerationSettings
from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph
from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.utils.database import Database


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.gpu_heavy

# Target distilled models to test
_TARGET_MODELS = [
    {"model_type": 2, "version": "1.0", "label": "distilled_bf16"},
    {"model_type": 2, "version": "4bit", "label": "distilled_sdnq_4bit"},
    {"model_type": 2, "version": "8bit", "label": "distilled_sdnq_8bit"},
]

_OFFLOAD_STRATEGIES = ["no_offload", "model_offload", "group_offload"]


def _vram_info() -> str:
    """Return a human-readable string with current VRAM and RAM usage."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    ram = psutil.virtual_memory()
    return (
        f"VRAM: {free / 1024**3:.1f}G free / {total / 1024**3:.1f}G total "
        f"(allocated={allocated / 1024**3:.1f}G, reserved={reserved / 1024**3:.1f}G) | "
        f"RAM: {ram.available / 1024**3:.1f}G free / {ram.total / 1024**3:.1f}G total"
    )


def _full_cleanup():
    """Release all GPU memory: clear model manager, force GC, empty CUDA cache."""
    mm = get_model_manager()
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@attr.s
class _FakeDirs:
    outputs_videos: str = attr.ib(default="")


def _discover_models(db_path: str) -> list[dict]:
    """Query the database for the target distilled models."""
    db = Database(db_path)
    rows = db.fetch_all(
        "SELECT id, name, filepath, model_type, version FROM model WHERE deleted = 0 AND model_type = 2"
    )
    db.disconnect()

    found = []
    for target in _TARGET_MODELS:
        for row in rows:
            row_id, name, filepath, model_type, version = row
            if model_type == target["model_type"] and version == target["version"]:
                found.append(
                    {
                        "id": row_id,
                        "name": name,
                        "filepath": filepath,
                        "model_type": model_type,
                        "version": version,
                        "label": target["label"],
                    }
                )
                break
    return found


def _build_graph(output_dir: str, model: dict):
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
        guidance_scale=1.0,
        use_torch_compile=False,
    )
    dirs = _FakeDirs(outputs_videos=output_dir)
    graph = create_default_ltx2_graph(settings, dirs)

    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = "a red ball"

    return graph


def _run_generation(graph) -> list[str]:
    device = torch.device("cuda")
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    graph()
    return video_paths


def _make_test_params(db_path: str) -> list:
    """Build parametrize args: [(model_dict, strategy, test_id), ...]."""
    models = _discover_models(db_path)
    params = []
    for model in models:
        for strategy in _OFFLOAD_STRATEGIES:
            test_id = f"{model['label']}-{strategy}"
            params.append(pytest.param(model, strategy, id=test_id))
    return params


def _get_db_path() -> str | None:
    """Read QSettings to find app.db, return None if not available."""
    try:
        from PyQt6.QtCore import QSettings

        settings = QSettings("ZCode", "FrameArtisan")
        data_path = settings.value("data_path", "")
        if not data_path:
            return None
        db_path = os.path.join(data_path, "app.db")
        return db_path if os.path.isfile(db_path) else None
    except Exception:
        return None


# Discover at module level so pytest can parametrize
_DB_PATH = _get_db_path()
_TEST_PARAMS = _make_test_params(_DB_PATH) if _DB_PATH else []


@pytest.mark.parametrize(
    "model,strategy",
    _TEST_PARAMS
    if _TEST_PARAMS
    else [pytest.param(None, None, marks=pytest.mark.skip("no distilled models found in DB"))],
)
def test_distilled_generation(tmp_path, app_db_path, model, strategy):
    """Run a distilled model with a given offload strategy and verify output."""
    set_app_database_path(app_db_path)
    mm = get_model_manager()
    graph = None

    try:
        _full_cleanup()
        print(f"\n  START: {model['label']} / {strategy} — {_vram_info()}")

        mm.offload_strategy = strategy

        output_dir = str(tmp_path)
        graph = _build_graph(output_dir, model)

        try:
            video_paths = _run_generation(graph)
        except (RuntimeError, NodeError) as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM: {model['label']} / {strategy} — {_vram_info()}")
                # Drop graph refs before cleanup so tensors can be freed
                del graph
                graph = None
                _full_cleanup()
                pytest.skip(f"OOM: {model['label']} / {strategy}")
            raise

        assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
        assert os.path.isfile(video_paths[0]), f"Video file not found: {video_paths[0]}"
        assert os.path.getsize(video_paths[0]) > 0, "Video file is empty"
        print(f"  PASS: {model['label']} / {strategy} — {video_paths[0]}")
    finally:
        # Drop graph refs first so model tensors can be freed
        del graph
        _full_cleanup()
        print(f"  CLEANUP: {_vram_info()}")


# ---------------------------------------------------------------------------
# Component-loading validation (replicates the UI startup flow)
# ---------------------------------------------------------------------------


def _read_quantization_from_config(config_path: str) -> str | None:
    """Read weights_dtype from a component's config.json quantization_config."""
    try:
        with open(os.path.join(config_path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config")
        if qc is None:
            return None
        return qc.get("weights_dtype")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _discover_model_by_version(db_path: str, version: str) -> dict | None:
    """Find a single distilled model (model_type=2) by version string."""
    db = Database(db_path)
    rows = db.fetch_all(
        "SELECT id, name, filepath, model_type, version FROM model "
        "WHERE deleted = 0 AND model_type = 2 AND version = ?",
        (version,),
    )
    db.disconnect()
    if not rows:
        return None
    row_id, name, filepath, model_type, row_version = rows[0]
    return {
        "id": row_id,
        "name": name,
        "filepath": filepath,
        "model_type": model_type,
        "version": row_version,
    }


def test_distilled_8bit_loads_correct_components(tmp_path, app_db_path):
    """Replicate the UI startup flow for the distilled 8-bit model and validate components.

    Steps mirror GenerationModule:
    1. Load GenerationSettings with the 8-bit model (as if saved in QSettings).
    2. Sync offload strategy to ModelManager.
    3. Build the graph via create_default_ltx2_graph (same as build_graph).
    4. Serialise → deserialise the graph (same as NodeGraphThread).
    5. Run the graph so the model node loads components.
    6. Assert transformer and text_encoder are both sdnq int8.
    """
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled SDNQ 8-bit not found in DB")

    set_app_database_path(app_db_path)
    mm = get_model_manager()
    graph = None

    try:
        _full_cleanup()

        # -- Step 1: build GenerationSettings exactly as the UI does --
        model_obj = ModelDataObject(
            name=model["name"],
            filepath=model["filepath"],
            model_type=model["model_type"],
            id=model["id"],
        )
        gen_settings = GenerationSettings(
            model=model_obj,
            video_width=256,
            video_height=256,
            video_duration=1,
            frame_rate=8,
            num_inference_steps=2,
            guidance_scale=1.0,
            offload_strategy="model_offload",
        )

        # -- Step 2: sync offload strategy (GenerationModule.build_graph) --
        mm.offload_strategy = gen_settings.offload_strategy

        # -- Step 3: build graph --
        dirs = _FakeDirs(outputs_videos=str(tmp_path))
        graph = create_default_ltx2_graph(gen_settings, dirs)

        # Verify model_id was passed through to the node
        model_node = graph.get_node_by_name("model")
        assert model_node.model_id == model["id"], (
            f"model_id mismatch: node has {model_node.model_id}, expected {model['id']}"
        )

        # -- Step 4: serialise → deserialise (same as NodeGraphThread) --
        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES

        json_graph = graph.to_json()
        run_graph = FrameArtisanNodeGraph()
        run_graph.from_json(json_graph, node_classes=NODE_CLASSES)
        run_graph.device = torch.device("cuda")
        run_graph.dtype = torch.bfloat16

        # -- Verify model_id survives serialisation round-trip --
        rt_model_node = run_graph.get_node_by_name("model")
        assert rt_model_node.model_id == model["id"], (
            f"model_id lost after serialisation: node has {rt_model_node.model_id}, expected {model['id']}"
        )

        # -- Step 5: validate component paths BEFORE running --
        resolved_paths = rt_model_node._resolve_component_paths(rt_model_node.model_path)
        te_path = resolved_paths.get("text_encoder", "")
        tr_path = resolved_paths.get("transformer", "")

        te_quant = _read_quantization_from_config(te_path)
        tr_quant = _read_quantization_from_config(tr_path)

        print(f"\n  Resolved TE path: {te_path}")
        print(f"  Resolved TR path: {tr_path}")
        print(f"  TE config weights_dtype: {te_quant}")
        print(f"  TR config weights_dtype: {tr_quant}")

        assert te_quant == "int8", f"text_encoder should be int8 but config says {te_quant!r} (path: {te_path})"
        assert tr_quant == "int8", f"transformer should be int8 but config says {tr_quant!r} (path: {tr_path})"

        # -- Step 6: run the full graph and validate loaded models --
        prompt_node = run_graph.get_node_by_name("prompt")
        prompt_node.text = "a red ball"

        video_paths = []
        video_send = run_graph.get_node_by_name("video_send")
        video_send.video_callback = video_paths.append
        video_send.output_dir = str(tmp_path)

        try:
            run_graph()
        except (RuntimeError, NodeError) as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"OOM: {e}")
            raise

        # Validate the loaded models have sdnq int8 quantization
        rt_model_node = run_graph.get_node_by_name("model")
        te = rt_model_node.values["text_encoder"]
        tr = rt_model_node.values["transformer"]

        te_desc = rt_model_node._describe_model(te)
        tr_desc = rt_model_node._describe_model(tr)

        print(f"  Loaded TE description: {te_desc}")
        print(f"  Loaded TR description: {tr_desc}")

        assert "int8" in te_desc, f"text_encoder should be sdnq int8 but got {te_desc!r}"
        assert "int8" in tr_desc, f"transformer should be sdnq int8 but got {tr_desc!r}"

        assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
        print(f"  PASS: 8-bit component validation — {video_paths[0]}")

    finally:
        del graph
        _full_cleanup()
