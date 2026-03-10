"""GPU test: verify FF chunking actually reduces peak VRAM usage.

Loads a single transformer FF layer on GPU and measures peak memory with
and without chunking on a realistic sequence length.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_ff_chunking_vram_savings.py -v -s

Activated by setting the environment variable ``GPU_HEAVY=1``.
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
    model = _discover_model(app_db_path, "1.0")
    if model is None:
        pytest.skip("No LTX2 Distilled bf16 model found in DB")
    return model


@pytest.fixture(scope="module")
def ff_on_gpu(app_db_path, model_info):
    """Load one transformer block's FF layer onto GPU."""
    from diffusers import LTX2VideoTransformer3DModel

    tr_path = _resolve_component_path(app_db_path, model_info, "transformer")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(tr_path, device_map="cpu", torch_dtype=torch.bfloat16)
    transformer.eval()

    ff = transformer.transformer_blocks[0].ff.cuda()

    yield ff

    del ff, transformer
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def setup_cleanup(app_db_path):
    set_app_database_path(app_db_path)
    set_global_model_manager(ModelManager())
    yield
    get_model_manager().clear()
    set_global_model_manager(ModelManager())
    gc.collect()
    torch.cuda.empty_cache()


def _measure_peak_mb(ff, x, num_chunks: int) -> float:
    """Run FF forward and return peak GPU memory allocated in MB."""
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    with torch.inference_mode():
        if num_chunks > 1:
            ff._ff_num_chunks = num_chunks
            out = _chunked_ff_forward(ff, x)
        else:
            out = ff(x)

    peak = torch.cuda.max_memory_allocated() - baseline
    del out
    return peak / (1024 * 1024)


def test_ff_chunking_saves_vram(ff_on_gpu):
    """Chunked FF should use measurably less peak VRAM than standard FF."""
    seq_len = 4096
    hidden_dim = 4096

    # Warm up
    with torch.inference_mode():
        _ = ff_on_gpu(torch.randn(1, seq_len, hidden_dim, dtype=torch.bfloat16, device="cuda"))
    torch.cuda.empty_cache()

    x_standard = torch.randn(1, seq_len, hidden_dim, dtype=torch.bfloat16, device="cuda")
    peak_standard = _measure_peak_mb(ff_on_gpu, x_standard, num_chunks=1)
    del x_standard
    torch.cuda.empty_cache()

    x_chunked = torch.randn(1, seq_len, hidden_dim, dtype=torch.bfloat16, device="cuda")
    peak_chunked = _measure_peak_mb(ff_on_gpu, x_chunked, num_chunks=2)
    del x_chunked
    torch.cuda.empty_cache()

    savings_mb = peak_standard - peak_chunked
    savings_pct = (savings_mb / peak_standard) * 100 if peak_standard > 0 else 0

    print(f"\n  Sequence length: {seq_len}")
    print(f"  Standard FF peak: {peak_standard:.1f} MB")
    print(f"  Chunked FF peak:  {peak_chunked:.1f} MB")
    print(f"  Savings:          {savings_mb:.1f} MB ({savings_pct:.1f}%)")

    assert peak_chunked < peak_standard, (
        f"Chunked FF ({peak_chunked:.1f} MB) should use less peak VRAM "
        f"than standard FF ({peak_standard:.1f} MB)"
    )
