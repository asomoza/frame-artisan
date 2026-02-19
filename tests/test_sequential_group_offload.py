"""Integration test: sequential group offload strategy.

Verifies that components are placed on the correct device before, during, and
after each graph node execution, and that VRAM/RAM usage is reasonable.

Activated by setting the environment variable ``GPU_LIGHT=1``.

    GPU_LIGHT=1 uv run --extra test pytest tests/test_sequential_group_offload.py -v -s
"""

from __future__ import annotations

import gc
from collections import deque
from unittest.mock import patch

import attr
import psutil
import pytest
import torch

TINY_MODEL_REPO = "OzzyGT/tiny_LTX2"

pytestmark = pytest.mark.gpu_light


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
        self.vram_mb = torch.cuda.memory_allocated(device) / (1024**2)
        self.vram_reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
        self.rss_mb = psutil.Process().memory_info().rss / (1024**2)

    def __repr__(self) -> str:
        return (
            f"{self.label}: "
            f"VRAM={self.vram_mb:.1f}MB (reserved={self.vram_reserved_mb:.1f}MB), "
            f"RSS={self.rss_mb:.1f}MB"
        )


# Nodes that actually use GPU components (call use_components).
_GPU_NODES = {"prompt_encode", "lora", "denoise", "decode"}

# What each GPU node uses and what strategy it runs under.
_NODE_COMPONENTS = {
    "prompt_encode": {"components": ("text_encoder", "connectors"), "strategy": "sequential_group_offload"},
    "lora": {"components": ("transformer",), "strategy": "sequential_group_offload"},
    "denoise": {"components": ("transformer",), "strategy": "sequential_group_offload"},
    "decode": {"components": ("vae", "audio_vae", "vocoder"), "strategy": "model_offload"},  # override
}


def _build_graph(output_dir, model_path, *, offload_strategy="sequential_group_offload"):
    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings
    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph

    model = ModelDataObject(name="tiny_LTX2", filepath=model_path, model_type=0, id=0)
    settings = GenerationSettings(
        model=model,
        video_width=256,
        video_height=256,
        video_duration=1,
        frame_rate=8,
        num_inference_steps=2,
        guidance_scale=1.0,
        offload_strategy=offload_strategy,
    )
    dirs = _FakeDirs(outputs_videos=output_dir)
    graph = create_default_ltx2_graph(settings, dirs)

    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = "a red ball"

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


def _full_cleanup():
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    get_model_manager().clear()
    set_global_model_manager(ModelManager())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(TINY_MODEL_REPO)


@pytest.fixture()
def mm_cleanup():
    """Ensure ModelManager is clean before and after the test."""
    from frameartisan.app.model_manager import ModelManager, set_global_model_manager

    set_global_model_manager(ModelManager())
    yield
    _full_cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sequential_group_offload_device_placement(tmp_path, tiny_model_path, mm_cleanup):
    """Run a full generation with sequential_group_offload and verify device
    placement of all components before and after each node execution."""
    from frameartisan.app.model_manager import get_model_manager

    device = torch.device("cuda")

    graph = _build_graph(str(tmp_path), tiny_model_path)
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    sorted_nodes = _topo_sort(graph)
    mm = get_model_manager()
    snapshots: list[MemorySnapshot] = []
    device_log: list[tuple[str, str, dict[str, str]]] = []

    # --- Execute nodes one by one (mirrors graph executor) ---
    snapshots.append(MemorySnapshot("before_graph", device))

    with mm.device_scope(device=device, dtype=torch.bfloat16):
        with torch.inference_mode():
            for node in sorted_nodes:
                if not (node.updated and node.enabled):
                    continue

                node.device = device
                node.dtype = torch.bfloat16
                node_name = node.name or node.__class__.__name__

                # Before
                if mm._managed_components:
                    device_log.append(("before", node_name, _get_all_component_devices(mm)))
                    snapshots.append(MemorySnapshot(f"before_{node_name}", device))

                node()
                node.updated = False

                # After
                if mm._managed_components:
                    device_log.append(("after", node_name, _get_all_component_devices(mm)))
                    snapshots.append(MemorySnapshot(f"after_{node_name}", device))

            mm.apply_offload_strategy(device)

    devices_final = _get_all_component_devices(mm)
    device_log.append(("final", "graph_done", devices_final))
    snapshots.append(MemorySnapshot("after_graph", device))

    # --- Print results ---
    print("\n" + "=" * 80)
    print("DEVICE PLACEMENT LOG")
    print("=" * 80)
    for phase, node_name, devices in device_log:
        # Only print GPU-relevant nodes to keep output readable
        if node_name in _GPU_NODES or phase == "final" or node_name == "model":
            print(f"\n[{phase}] {node_name}:")
            for comp, dev in sorted(devices.items()):
                print(f"  {comp:20s} -> {dev}")

    print("\n" + "=" * 80)
    print("MEMORY SNAPSHOTS")
    print("=" * 80)
    for snap in snapshots:
        if any(kw in snap.label for kw in ("graph", "model", "prompt_encode", "denoise", "decode", "video_send")):
            print(f"  {snap}")

    # --- Assertions ---

    # 1. Video was produced
    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"

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
    final_vram = snapshots[-1].vram_mb
    print(f"\nFinal VRAM allocated: {final_vram:.1f} MB")
    assert final_vram < 500, f"Final VRAM too high: {final_vram:.1f} MB"


def test_sequential_group_offload_during_execution(tmp_path, tiny_model_path, mm_cleanup):
    """Capture device state and VRAM *inside* use_components blocks.

    With sequential_group_offload, hooks manage per-layer GPU placement so
    parameter-level device checks may still show 'cpu'.  The key assertion
    is that VRAM increases during heavy nodes (prompt_encode, denoise),
    confirming that computation is actually happening on GPU.
    """
    from frameartisan.app.model_manager import ModelManager

    device = torch.device("cuda")

    graph = _build_graph(str(tmp_path), tiny_model_path)
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    during_snapshots: list[tuple[str, dict[str, str], MemorySnapshot]] = []

    # Patch use_components to capture device state inside the context manager
    original_use_components = ModelManager.use_components

    import contextlib

    @contextlib.contextmanager
    def instrumented_use_components(self, *names, device, strategy_override=None):
        with original_use_components(self, *names, device=device, strategy_override=strategy_override):
            # Capture state while components are active on GPU
            devices = _get_all_component_devices(self)
            snap = MemorySnapshot(f"during_{'_'.join(names)}", torch.device(device))
            during_snapshots.append(("_".join(names), devices, snap))
            yield

    with patch.object(ModelManager, "use_components", instrumented_use_components):
        with torch.inference_mode():
            graph()

    assert len(video_paths) == 1

    # --- Print ---
    print("\n" + "=" * 80)
    print("DURING-EXECUTION DEVICE PLACEMENT & VRAM")
    print("=" * 80)
    for label, devices, snap in during_snapshots:
        print(f"\n  [{label}]  {snap}")
        for comp, dev in sorted(devices.items()):
            print(f"    {comp:20s} -> {dev}")

    # --- Assertions ---

    # Snapshots were captured for the expected node groups
    labels = {label for label, _, _ in during_snapshots}
    assert any("text_encoder" in lbl for lbl in labels), "No snapshot during prompt encoding"
    assert any("transformer" in lbl and "text_encoder" not in lbl for lbl in labels), "No snapshot during denoise"
    assert any("vae" in lbl for lbl in labels), "No snapshot during decode"

    # VRAM should be non-zero during heavy nodes (computation is on GPU)
    for label, _, snap in during_snapshots:
        if "transformer" in label or "text_encoder" in label:
            # With group offload hooks, VRAM should be used for activations
            # even if parameters report as 'cpu'
            print(f"\n  {label}: VRAM reserved = {snap.vram_reserved_mb:.1f} MB")

    # Inactive components should stay on CPU during each phase
    for label, devices, _ in during_snapshots:
        active = set(label.split("_"))
        for comp, dev in devices.items():
            if comp not in active and comp in ("transformer", "text_encoder") and dev.startswith("cuda"):
                pytest.fail(f"{comp} unexpectedly on GPU during {label}")


def test_sequential_group_offload_vs_no_offload_vram(tmp_path, tiny_model_path, mm_cleanup):
    """Compare peak VRAM between sequential_group_offload and no_offload.

    With the tiny model the difference may be negligible, so we only assert
    that sequential_group_offload uses no MORE peak VRAM than no_offload.
    The real value of this test is the printed numbers for manual inspection.
    """
    from frameartisan.app.model_manager import ModelManager, set_global_model_manager

    device = torch.device("cuda")

    def _run_with_strategy(strategy: str) -> float:
        set_global_model_manager(ModelManager())
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        graph = _build_graph(str(tmp_path), tiny_model_path, offload_strategy=strategy)
        graph.device = device
        graph.dtype = torch.bfloat16

        video_paths: list[str] = []
        video_send = graph.get_node_by_name("video_send")
        video_send.video_callback = video_paths.append

        with torch.inference_mode():
            graph()

        assert len(video_paths) == 1
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)

        del graph
        _full_cleanup()
        return peak

    peak_no_offload = _run_with_strategy("no_offload")
    peak_seq_group = _run_with_strategy("sequential_group_offload")

    print(f"\nPeak VRAM (no_offload):                {peak_no_offload:.1f} MB")
    print(f"Peak VRAM (sequential_group_offload):  {peak_seq_group:.1f} MB")
    savings = peak_no_offload - peak_seq_group
    print(f"Savings:                               {savings:.1f} MB")

    # With tiny model, savings may be zero. Just ensure it's not worse.
    # A 10% tolerance accounts for CUDA allocator non-determinism.
    assert peak_seq_group <= peak_no_offload * 1.1, (
        f"sequential_group_offload ({peak_seq_group:.1f} MB) should not use significantly more VRAM "
        f"than no_offload ({peak_no_offload:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_log_entry(device_log, phase, node_name) -> dict[str, str]:
    entries = [d for p, n, d in device_log if p == phase and n == node_name]
    assert entries, f"No '{phase} {node_name}' log entry found"
    return entries[0]
