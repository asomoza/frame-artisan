"""Heavy GPU test: two-stage inference with sequential group offload + streaming decode.

Uses an 8-bit distilled model for the first pass and a 4-bit distilled model
for the second pass, with sequential_group_offload + CUDA streams and streaming
decode enabled.  Tracks VRAM at every graph node to identify spikes.

Activated by setting the environment variable ``GPU_HEAVY=1``.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_two_stage_streaming_heavy.py -v -s
"""

from __future__ import annotations

import gc
import os
from collections import deque

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
from frameartisan.modules.generation.constants import LTX2_LATENT_UPSAMPLER_DIR
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
        torch.cuda.synchronize(device)
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


def _discover_model(db_path: str, version: str) -> dict | None:
    """Find a distilled model (model_type=2) with the given version in the database."""
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
    return {
        "id": row_id,
        "name": name,
        "filepath": filepath,
        "model_type": model_type,
        "version": ver,
    }


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


def _find_upsampler_path() -> str | None:
    """Find the latent upsampler model directory."""
    from PyQt6.QtCore import QSettings

    settings = QSettings("ZCode", "FrameArtisan")
    models_path = settings.value("models_diffusers", "")
    if not models_path:
        return None
    path = os.path.join(models_path, LTX2_LATENT_UPSAMPLER_DIR)
    return path if os.path.isdir(path) else None


def _enable_second_pass(graph):
    """Enable 2nd pass nodes and rewire decode connections (mirrors GenerationModule._toggle_second_pass)."""
    for name in (
        "upsample",
        "second_pass_model",
        "second_pass_lora",
        "second_pass_latents",
        "second_pass_denoise",
    ):
        node = graph.get_node_by_name(name)
        if node is not None:
            node.enabled = True
            node.set_updated()

    decode_node = graph.get_node_by_name("decode")
    sp_denoise = graph.get_node_by_name("second_pass_denoise")
    sp_latents = graph.get_node_by_name("second_pass_latents")
    denoise_node = graph.get_node_by_name("denoise")
    latents_node = graph.get_node_by_name("latents")

    if not all([decode_node, sp_denoise, sp_latents, denoise_node, latents_node]):
        return

    # Disconnect first pass sources
    decode_node.disconnect("video_latents", denoise_node, "video_latents")
    decode_node.disconnect("audio_latents", denoise_node, "audio_latents")
    decode_node.disconnect("latent_num_frames", latents_node, "latent_num_frames")
    decode_node.disconnect("latent_height", latents_node, "latent_height")
    decode_node.disconnect("latent_width", latents_node, "latent_width")
    decode_node.disconnect("audio_num_frames", latents_node, "audio_num_frames")

    # Connect second pass sources
    decode_node.connect("video_latents", sp_denoise, "video_latents")
    decode_node.connect("audio_latents", sp_denoise, "audio_latents")
    decode_node.connect("latent_num_frames", sp_latents, "latent_num_frames")
    decode_node.connect("latent_height", sp_latents, "latent_height")
    decode_node.connect("latent_width", sp_latents, "latent_width")
    decode_node.connect("audio_num_frames", sp_latents, "audio_num_frames")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def upsampler_path():
    """Find the latent upsampler model directory."""
    path = _find_upsampler_path()
    if path is None:
        pytest.skip("LTX2 latent upsampler not found")
    return path


@pytest.fixture(scope="module")
def model_8bit(app_db_path):
    """Discover the 8-bit distilled model from the database."""
    model = _discover_model(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled SDNQ 8-bit not found in DB")
    return model


@pytest.fixture(scope="module")
def model_4bit(app_db_path):
    """Discover the 4-bit distilled model from the database."""
    model = _discover_model(app_db_path, "4bit")
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


def test_two_stage_streaming_vram(tmp_path, model_8bit, model_4bit, upsampler_path, setup_cleanup):
    """Run full two-stage generation with sequential_group_offload + CUDA stream
    + streaming decode.  Track VRAM at every node to identify spikes.

    First pass:  8-bit distilled
    Second pass: 4-bit distilled
    """
    device = torch.device("cuda")
    mm = get_model_manager()
    torch.cuda.reset_peak_memory_stats(device)

    # Build graph with 8-bit as primary, 4-bit as second pass
    first_pass_model = ModelDataObject(
        name=model_8bit["name"],
        filepath=model_8bit["filepath"],
        model_type=model_8bit["model_type"],
        id=model_8bit["id"],
    )
    second_pass_model = ModelDataObject(
        name=model_4bit["name"],
        filepath=model_4bit["filepath"],
        model_type=model_4bit["model_type"],
        id=model_4bit["id"],
    )

    settings = GenerationSettings(
        model=first_pass_model,
        second_pass_model=second_pass_model,
        second_pass_enabled=True,
        second_pass_steps=3,
        second_pass_guidance=1.0,
        video_width=512,
        video_height=320,
        video_duration=5,
        frame_rate=24,
        num_inference_steps=8,
        guidance_scale=1.0,
        offload_strategy="sequential_group_offload",
        group_offload_use_stream=True,
        group_offload_low_cpu_mem=False,
        streaming_decode=True,
    )

    dirs = _FakeDirs(outputs_videos=str(tmp_path))
    graph = create_default_ltx2_graph(settings, dirs)

    # Set upsampler path and enable 2nd pass
    graph.get_node_by_name("upsample").upsampler_model_path = upsampler_path
    _enable_second_pass(graph)

    graph.device = device
    graph.dtype = torch.bfloat16

    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = "a red ball bouncing on a table"

    video_paths: list[str] = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    sorted_nodes = _topo_sort(graph)
    snapshots: list[MemorySnapshot] = []
    device_log: list[tuple[str, str, dict[str, str]]] = []

    snapshots.append(MemorySnapshot("before_graph", device))

    with mm.device_scope(device=device, dtype=torch.bfloat16):
        with torch.inference_mode():
            for node in sorted_nodes:
                if not (node.updated and node.enabled):
                    continue

                node.device = device
                node.dtype = torch.bfloat16
                node_name = node.name or node.__class__.__name__

                # Reset peak before each node to get per-node peak
                torch.cuda.reset_peak_memory_stats(device)
                snap_before = MemorySnapshot(f"before_{node_name}", device)
                snapshots.append(snap_before)

                if mm._managed_components:
                    device_log.append(("before", node_name, _get_all_component_devices(mm)))

                try:
                    node()
                except (RuntimeError, NodeError) as e:
                    if "out of memory" in str(e).lower():
                        del graph
                        _full_cleanup()
                        pytest.skip(f"OOM during {node_name}: {e}")
                    raise

                node.updated = False

                snap_after = MemorySnapshot(f"after_{node_name}", device)
                snapshots.append(snap_after)

                if mm._managed_components:
                    device_log.append(("after", node_name, _get_all_component_devices(mm)))

            mm.apply_offload_strategy(device)

    snapshots.append(MemorySnapshot("final", device))

    # --- Print ---
    print("\n" + "=" * 80)
    print("TWO-STAGE STREAMING DECODE — VRAM PROFILE")
    print(f"  1st pass: {model_8bit['name']} (8-bit)")
    print(f"  2nd pass: {model_4bit['name']} (4-bit)")
    print("  Strategy: sequential_group_offload + CUDA stream + streaming decode")
    print("=" * 80)

    print("\n--- Per-node VRAM (peak during node execution) ---")
    # Pair up before/after snapshots
    i = 1  # skip "before_graph"
    while i < len(snapshots) - 1:
        before = snapshots[i]
        after = snapshots[i + 1]
        node_name = before.label.replace("before_", "")
        peak_during = after.vram_peak_mb
        alloc_before = before.vram_allocated_mb
        alloc_after = after.vram_allocated_mb
        print(
            f"  {node_name:30s}  "
            f"peak={peak_during:6.0f}MB  "
            f"alloc_before={alloc_before:6.0f}MB  "
            f"alloc_after={alloc_after:6.0f}MB"
        )
        i += 2

    print("\n--- Device placement log ---")
    for phase, node_name, devices in device_log:
        print(f"\n  [{phase}] {node_name}:")
        for comp, dev in sorted(devices.items()):
            print(f"    {comp:25s} -> {dev}")

    print("\n--- All snapshots ---")
    for snap in snapshots:
        print(f"  {snap}")

    # --- Assertions ---

    # 1. Video was produced
    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
    assert os.path.isfile(video_paths[0])
    assert os.path.getsize(video_paths[0]) > 0
    print(f"\nVideo saved: {video_paths[0]} ({os.path.getsize(video_paths[0]) / 1024:.0f} KB)")

    # 2. After decode: vae, audio_vae, vocoder back on CPU
    after_decode = _get_log_entry(device_log, "after", "decode")
    for comp in ("vae", "audio_vae", "vocoder"):
        assert after_decode[comp] == "cpu", f"After decode: {comp} should be on cpu, got {after_decode[comp]}"

    # 3. Final: everything on CPU
    final_devices = _get_all_component_devices(mm)
    for comp, dev in final_devices.items():
        assert dev == "cpu", f"Final: {comp} should be on cpu, got {dev}"

    # 4. Decode node peak VRAM should be reasonable with streaming decode.
    #    Without streaming, the full decoded video tensor would spike VRAM.
    #    With streaming, peak during decode should be well below the denoise peak.
    decode_peak = _get_node_peak(snapshots, "decode")
    denoise_peak = _get_node_peak(snapshots, "denoise")
    sp_denoise_peak = _get_node_peak(snapshots, "second_pass_denoise")
    max_denoise_peak = max(denoise_peak, sp_denoise_peak)

    print(f"\n  Decode peak VRAM:          {decode_peak:.0f} MB")
    print(f"  Max denoise peak VRAM:     {max_denoise_peak:.0f} MB")

    if decode_peak > 0 and max_denoise_peak > 0:
        print(f"  Decode / Denoise ratio:    {decode_peak / max_denoise_peak:.2f}")
        # Streaming decode peak should not exceed denoise peak by more than 50%
        assert decode_peak < max_denoise_peak * 1.5, (
            f"Decode peak VRAM ({decode_peak:.0f}MB) is too high relative to "
            f"denoise peak ({max_denoise_peak:.0f}MB) — streaming decode may not be working"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_log_entry(device_log, phase, node_name) -> dict[str, str]:
    entries = [d for p, n, d in device_log if p == phase and n == node_name]
    assert entries, f"No '{phase} {node_name}' log entry found"
    return entries[0]


def _get_node_peak(snapshots: list[MemorySnapshot], node_name: str) -> float:
    """Get the peak VRAM recorded after a node executed."""
    for snap in snapshots:
        if snap.label == f"after_{node_name}":
            return snap.vram_peak_mb
    return 0.0
