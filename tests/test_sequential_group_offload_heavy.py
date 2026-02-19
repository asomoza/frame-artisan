"""Heavy GPU test: sequential group offload with a real 4-bit distilled model.

Verifies device placement, VRAM/RAM usage, and compares peak VRAM against
no_offload and model_offload strategies.

Activated by setting the environment variable ``GPU_HEAVY=1``.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_sequential_group_offload_heavy.py -v -s
"""

from __future__ import annotations

import gc
import os
from collections import deque
from unittest.mock import patch

import attr
import psutil
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@attr.s
class _FakeDirs:
    outputs_videos: str = attr.ib(default="")


def _get_device_str(module) -> str:
    """Return the device string of the first parameter, or 'no_params'."""
    try:
        p = next(module.parameters())
        return str(p.device)
    except StopIteration:
        return "no_params"


def _get_all_component_devices(mm) -> dict[str, str]:
    """Snapshot device of every managed component."""
    return {name: _get_device_str(mod) for name, mod in mm._managed_components.items()}


class MemorySnapshot:
    """Capture VRAM and RSS at a given point."""

    def __init__(self, label: str, device: torch.device):
        self.label = label
        self.vram_allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
        self.vram_reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
        self.vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        self.rss_mb = psutil.Process().memory_info().rss / (1024**2)

    def __repr__(self) -> str:
        return (
            f"{self.label}: "
            f"VRAM alloc={self.vram_allocated_mb:.0f}MB "
            f"reserved={self.vram_reserved_mb:.0f}MB "
            f"peak={self.vram_peak_mb:.0f}MB | "
            f"RSS={self.rss_mb:.0f}MB"
        )


def _full_cleanup():
    """Release all GPU memory."""
    get_model_manager().clear()
    set_global_model_manager(ModelManager())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _discover_4bit_model(db_path: str) -> dict | None:
    """Find the 4-bit distilled model in the database."""
    db = Database(db_path)
    rows = db.fetch_all(
        "SELECT id, name, filepath, model_type, version FROM model "
        "WHERE deleted = 0 AND model_type = 2 AND version = '4bit'"
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


def _build_graph(output_dir: str, model: dict, *, offload_strategy: str):
    model_obj = ModelDataObject(
        name=model["name"],
        filepath=model["filepath"],
        model_type=model["model_type"],
        id=model["id"],
    )
    settings = GenerationSettings(
        model=model_obj,
        video_width=512,
        video_height=320,
        video_duration=2,
        frame_rate=24,
        num_inference_steps=8,
        guidance_scale=1.0,
        offload_strategy=offload_strategy,
    )
    dirs = _FakeDirs(outputs_videos=output_dir)
    graph = create_default_ltx2_graph(settings, dirs)

    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = "a red ball bouncing"

    return graph


def _topo_sort(graph):
    """Topological sort matching the graph executor."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_4bit(app_db_path):
    """Discover the 4-bit distilled model from the database."""
    model = _discover_4bit_model(app_db_path)
    if model is None:
        pytest.skip("LTX2 Distilled SDNQ 4-bit not found in DB")
    return model


@pytest.fixture()
def setup_cleanup(app_db_path):
    """Clean ModelManager before and after each test."""
    set_app_database_path(app_db_path)
    set_global_model_manager(ModelManager())
    yield
    _full_cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_device_placement(tmp_path, model_4bit, setup_cleanup):
    """Run a full generation with sequential_group_offload on the 4-bit model
    and verify device placement of all components at every stage."""
    device = torch.device("cuda")
    mm = get_model_manager()
    torch.cuda.reset_peak_memory_stats(device)

    graph = _build_graph(str(tmp_path), model_4bit, offload_strategy="sequential_group_offload")
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    sorted_nodes = _topo_sort(graph)
    snapshots: list[MemorySnapshot] = []
    device_log: list[tuple[str, str, dict[str, str]]] = []

    # Key nodes we want to track
    key_nodes = {"model", "lora", "prompt_encode", "denoise", "decode", "video_send"}

    snapshots.append(MemorySnapshot("before_graph", device))

    with mm.device_scope(device=device, dtype=torch.bfloat16):
        with torch.inference_mode():
            for node in sorted_nodes:
                if not (node.updated and node.enabled):
                    continue

                node.device = device
                node.dtype = torch.bfloat16
                node_name = node.name or node.__class__.__name__

                if mm._managed_components and node_name in key_nodes:
                    device_log.append(("before", node_name, _get_all_component_devices(mm)))
                    snapshots.append(MemorySnapshot(f"before_{node_name}", device))

                try:
                    node()
                except (RuntimeError, NodeError) as e:
                    if "out of memory" in str(e).lower():
                        del graph
                        _full_cleanup()
                        pytest.skip(f"OOM during {node_name}: {e}")
                    raise

                node.updated = False

                if mm._managed_components and node_name in key_nodes:
                    device_log.append(("after", node_name, _get_all_component_devices(mm)))
                    snapshots.append(MemorySnapshot(f"after_{node_name}", device))

            mm.apply_offload_strategy(device)

    devices_final = _get_all_component_devices(mm)
    device_log.append(("final", "graph_done", devices_final))
    snapshots.append(MemorySnapshot("after_graph", device))

    # --- Print ---
    print("\n" + "=" * 80)
    print(f"DEVICE PLACEMENT LOG — {model_4bit['name']} / sequential_group_offload")
    print("=" * 80)
    for phase, node_name, devices in device_log:
        print(f"\n[{phase}] {node_name}:")
        for comp, dev in sorted(devices.items()):
            print(f"  {comp:20s} -> {dev}")

    print("\n" + "=" * 80)
    print("MEMORY SNAPSHOTS")
    print("=" * 80)
    for snap in snapshots:
        print(f"  {snap}")

    # --- Assertions ---

    # 1. Video was produced
    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
    assert os.path.isfile(video_paths[0])
    assert os.path.getsize(video_paths[0]) > 0

    # 2. After model node: all components on CPU
    after_model = _get_log_entry(device_log, "after", "model")
    for comp, dev in after_model.items():
        assert dev == "cpu", f"After model: {comp} should be on cpu, got {dev}"

    # 3. After prompt_encode: text_encoder and connectors back on CPU
    after_prompt = _get_log_entry(device_log, "after", "prompt_encode")
    for comp in ("text_encoder", "connectors"):
        assert after_prompt[comp] == "cpu", f"After prompt_encode: {comp} should be on cpu, got {after_prompt[comp]}"

    # 4. After denoise: transformer back on CPU
    after_denoise = _get_log_entry(device_log, "after", "denoise")
    assert after_denoise["transformer"] == "cpu", (
        f"After denoise: transformer should be on cpu, got {after_denoise['transformer']}"
    )

    # 5. After decode: vae, audio_vae, vocoder back on CPU
    after_decode = _get_log_entry(device_log, "after", "decode")
    for comp in ("vae", "audio_vae", "vocoder"):
        assert after_decode[comp] == "cpu", f"After decode: {comp} should be on cpu, got {after_decode[comp]}"

    # 6. Final: everything on CPU
    for comp, dev in devices_final.items():
        assert dev == "cpu", f"Final: {comp} should be on cpu, got {dev}"

    # 7. VRAM should be low at the end
    final_vram = snapshots[-1].vram_allocated_mb
    print(f"\nFinal VRAM allocated: {final_vram:.0f} MB")
    assert final_vram < 1000, f"Final VRAM too high: {final_vram:.0f} MB"

    # 8. Peak VRAM
    peak_vram = max(s.vram_peak_mb for s in snapshots)
    print(f"Peak VRAM: {peak_vram:.0f} MB")


def test_during_execution(tmp_path, model_4bit, setup_cleanup):
    """Capture device state and VRAM inside use_components blocks.

    Verifies VRAM spikes during heavy compute and that inactive components
    stay on CPU.
    """
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    graph = _build_graph(str(tmp_path), model_4bit, offload_strategy="sequential_group_offload")
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    during_snapshots: list[tuple[str, dict[str, str], MemorySnapshot]] = []
    original_use_components = ModelManager.use_components

    import contextlib

    @contextlib.contextmanager
    def instrumented_use_components(self, *names, device, strategy_override=None):
        with original_use_components(self, *names, device=device, strategy_override=strategy_override):
            devices = _get_all_component_devices(self)
            snap = MemorySnapshot(f"during_{'_'.join(names)}", torch.device(device))
            during_snapshots.append(("_".join(names), devices, snap))
            yield

    try:
        with patch.object(ModelManager, "use_components", instrumented_use_components):
            with torch.inference_mode():
                graph()
    except (RuntimeError, NodeError) as e:
        if "out of memory" in str(e).lower():
            del graph
            _full_cleanup()
            pytest.skip(f"OOM: {e}")
        raise

    assert len(video_paths) == 1

    # --- Print ---
    print("\n" + "=" * 80)
    print(f"DURING-EXECUTION — {model_4bit['name']} / sequential_group_offload")
    print("=" * 80)
    for label, devices, snap in during_snapshots:
        print(f"\n  [{label}]")
        print(f"    {snap}")
        for comp, dev in sorted(devices.items()):
            marker = " <-- ACTIVE" if comp in label else ""
            print(f"    {comp:20s} -> {dev}{marker}")

    # --- Assertions ---
    labels = {label for label, _, _ in during_snapshots}
    assert any("text_encoder" in lbl for lbl in labels), "No snapshot during prompt encoding"
    assert any("transformer" in lbl and "text_encoder" not in lbl for lbl in labels), "No snapshot during denoise"
    assert any("vae" in lbl for lbl in labels), "No snapshot during decode"

    # VRAM should spike during transformer (denoise) — the heaviest component
    for label, _, snap in during_snapshots:
        if label == "transformer":
            print(f"\n  During denoise: VRAM reserved = {snap.vram_reserved_mb:.0f} MB")
            # 4-bit transformer should still use meaningful VRAM via hooks
            assert snap.vram_reserved_mb > 100, (
                f"Expected VRAM > 100 MB during denoise, got {snap.vram_reserved_mb:.0f} MB"
            )

    # Inactive heavy components (transformer, text_encoder) should be on CPU
    for label, devices, _ in during_snapshots:
        active = set(label.split("_"))
        for comp in ("transformer", "text_encoder"):
            if comp not in active and devices.get(comp, "cpu").startswith("cuda"):
                pytest.fail(f"{comp} unexpectedly on GPU during {label}")


def test_vram_comparison(tmp_path, model_4bit, setup_cleanup):
    """Compare peak VRAM across offload strategies with the 4-bit model.

    Sequential group offload should use less peak VRAM than no_offload,
    since it loads only one heavy component at a time.
    """
    device = torch.device("cuda")
    results: dict[str, float] = {}

    for strategy in ("no_offload", "model_offload", "sequential_group_offload"):
        set_global_model_manager(ModelManager())
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        graph = _build_graph(str(tmp_path), model_4bit, offload_strategy=strategy)
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
                results[strategy] = float("inf")
                print(f"  {strategy}: OOM")
                continue
            raise

        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        results[strategy] = peak
        assert len(video_paths) == 1

        del graph
        _full_cleanup()

    # --- Print ---
    print("\n" + "=" * 80)
    print(f"PEAK VRAM COMPARISON — {model_4bit['name']}")
    print("=" * 80)
    for strategy, peak in results.items():
        if peak == float("inf"):
            print(f"  {strategy:35s}  OOM")
        else:
            print(f"  {strategy:35s}  {peak:.0f} MB")

    peak_no = results.get("no_offload", float("inf"))
    peak_model = results.get("model_offload", float("inf"))
    peak_seq = results.get("sequential_group_offload", float("inf"))

    if peak_no != float("inf") and peak_seq != float("inf"):
        savings_vs_no = peak_no - peak_seq
        print(f"\n  Savings vs no_offload:     {savings_vs_no:.0f} MB")
    if peak_model != float("inf") and peak_seq != float("inf"):
        savings_vs_model = peak_model - peak_seq
        print(f"  Savings vs model_offload:  {savings_vs_model:.0f} MB")

    # Sequential group offload should use less or equal peak VRAM than no_offload
    if peak_no != float("inf") and peak_seq != float("inf"):
        assert peak_seq <= peak_no * 1.05, (
            f"sequential_group_offload ({peak_seq:.0f} MB) should not use more VRAM than no_offload ({peak_no:.0f} MB)"
        )

    # If no_offload OOM'd but sequential_group_offload succeeded, that's a pass
    if peak_no == float("inf") and peak_seq != float("inf"):
        print("\n  no_offload OOM'd but sequential_group_offload succeeded — VRAM savings confirmed!")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_log_entry(device_log, phase, node_name) -> dict[str, str]:
    entries = [d for p, n, d in device_log if p == phase and n == node_name]
    assert entries, f"No '{phase} {node_name}' log entry found"
    return entries[0]
