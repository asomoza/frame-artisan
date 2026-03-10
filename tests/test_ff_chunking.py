"""GPU test: feed-forward chunking numerical comparison for LTX2 transformer.

Verifies that enabling chunked feed-forward produces identical output to the
standard (non-chunked) forward pass.

The tests run on **CPU** to assert exact zero diff — this proves the chunking
logic is mathematically correct.  GPU matmul kernels are non-deterministic
across different input shapes (different CUDA kernel tiling / accumulation
order), so GPU diffs are expected and do not indicate a bug.

Activated by setting the environment variable ``GPU_HEAVY=1``.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_ff_chunking.py -v -s
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

from frameartisan.app.app import set_app_database_path
from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
from frameartisan.modules.generation.graph.nodes.ff_chunking import (
    _chunked_ff_forward,
)
from frameartisan.utils.database import Database

pytestmark = pytest.mark.gpu_heavy


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


def _resolve_component_path(db_path: str, model_info: dict, component: str) -> str:
    """Resolve the storage path for a component via the registry, fallback to subdir."""
    try:
        from frameartisan.app.component_registry import ComponentRegistry

        components_base_dir = os.path.join(os.path.dirname(model_info["filepath"]), "_components")
        registry = ComponentRegistry(db_path, components_base_dir)
        comps = registry.get_model_components(model_info["id"])
        if comps and component in comps:
            return comps[component].storage_path
    except Exception:
        pass
    return os.path.join(model_info["filepath"], component)


@pytest.fixture(scope="module")
def model_info(app_db_path):
    # Use the full bf16 model to avoid SDNQ quantized matmul non-determinism
    model = _discover_model(app_db_path, "1.0")
    if model is None:
        pytest.skip("No LTX2 Distilled bf16 model found in DB")
    return model


@pytest.fixture(scope="module")
def transformer_on_cpu(app_db_path, model_info):
    """Load the full bf16 transformer once, keep on CPU.

    Tests run entirely on CPU for deterministic matmul results.
    """
    from diffusers import LTX2VideoTransformer3DModel

    tr_path = _resolve_component_path(app_db_path, model_info, "transformer")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(tr_path, device_map="cpu", torch_dtype=torch.bfloat16)
    transformer.eval()

    yield transformer

    del transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def setup_cleanup(app_db_path):
    set_app_database_path(app_db_path)
    set_global_model_manager(ModelManager())
    yield
    get_model_manager().clear()
    set_global_model_manager(ModelManager())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 1: Isolated FF layer — prove in-place chunking is mathematically correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_chunks", [2, 4, 16])
def test_isolated_ff_chunking_zero_diff(transformer_on_cpu, num_chunks):
    """Chunking a single FF layer on a fixed input must produce bit-identical output.

    Runs on CPU where matmul is deterministic regardless of input shape, so any
    diff would indicate a real bug in the chunking logic.
    """
    block = transformer_on_cpu.transformer_blocks[0]
    ff = block.ff

    gen = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(1, 64, 4096, generator=gen, dtype=torch.bfloat16, device="cpu")

    with torch.inference_mode():
        out_std = ff(x.clone()).clone()

        # Simulate the in-place chunked forward
        ff._ff_num_chunks = num_chunks
        x_chunked = x.clone()
        out_chunked = _chunked_ff_forward(ff, x_chunked)

    diff = (out_std.float() - out_chunked.float()).abs().max().item()
    print(f"\n  Isolated FF num_chunks={num_chunks}: max_diff={diff}")

    assert diff == 0.0, f"Isolated FF differs with num_chunks={num_chunks}: max_diff={diff}"


# ---------------------------------------------------------------------------
# Test 2: Isolated audio FF layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_chunks", [2, 4, 8])
def test_isolated_audio_ff_chunking_zero_diff(transformer_on_cpu, num_chunks):
    """Same as above but for the audio FF path (audio_inner_dim=2048)."""
    block = transformer_on_cpu.transformer_blocks[0]
    audio_ff = block.audio_ff

    gen = torch.Generator(device="cpu").manual_seed(123)
    x = torch.randn(1, 16, 2048, generator=gen, dtype=torch.bfloat16, device="cpu")

    with torch.inference_mode():
        out_std = audio_ff(x.clone()).clone()

        audio_ff._ff_num_chunks = num_chunks
        x_chunked = x.clone()
        out_chunked = _chunked_ff_forward(audio_ff, x_chunked)

    diff = (out_std.float() - out_chunked.float()).abs().max().item()
    print(f"\n  Isolated audio FF num_chunks={num_chunks}: max_diff={diff}")

    assert diff == 0.0, f"Isolated audio FF differs with num_chunks={num_chunks}: max_diff={diff}"
