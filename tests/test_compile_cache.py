"""Test that torch.compile persistent disk cache works correctly.

Simulates the real app's regional compilation pattern:
- A model with multiple repeated blocks (like LTX2's 48 transformer blocks)
- compile_repeated_blocks-style per-block compilation
- Dynamo reset between runs (simulates closing and reopening the app)
- TORCHINDUCTOR_CACHE_DIR for persistent disk cache

Verifies:
1. First compilation populates the cache directory
2. Second compilation with the same shapes hits the cache (faster)
"""

import os
import time

import pytest
import torch
import torch.nn as nn


gpu_light = pytest.mark.gpu_light

# Number of repeated blocks — enough to make the cache speedup measurable
# without being so large that the test takes too long.
NUM_BLOCKS = 6


class _TinyTransformerBlock(nn.Module):
    """Minimal transformer-like block matching the pattern compiled by
    ``compile_repeated_blocks`` in diffusers (attention + FF + norms)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class _TinyTransformer(nn.Module):
    """Model with repeated blocks, mimicking LTX2VideoTransformer3DModel."""

    _repeated_blocks = ["_TinyTransformerBlock"]

    def __init__(self, num_blocks: int = NUM_BLOCKS, dim: int = 128):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([_TinyTransformerBlock(dim) for _ in range(num_blocks)])
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)

    def compile_repeated_blocks(self, **kwargs):
        """Regional compilation — same logic as diffusers ModelMixin."""
        for submod in self.modules():
            if submod.__class__.__name__ in self._repeated_blocks:
                torch.compile(submod, **kwargs)


def _dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(path) for f in fns)


def _build_compile_and_run(cache_dir: str, mode: str = "default"):
    """Build a fresh model, apply regional compile, and run one forward pass.

    Resets Dynamo first to simulate a fresh app launch (no in-memory cache).
    """
    torch._dynamo.reset()

    model = _TinyTransformer().cuda().to(torch.bfloat16)

    # Mirror the real app's settings
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 1000

    # Regional compile — each block is compiled individually (in-place)
    for submod in model.modules():
        if submod.__class__.__name__ in model._repeated_blocks:
            submod.compile(mode=mode)

    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)

    t0 = time.monotonic()
    _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    return elapsed


@gpu_light
def test_compile_cache_populates_directory(tmp_path):
    """Regional compilation should populate the cache directory with files."""
    cache_dir = str(tmp_path / "torch_compile")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    _build_compile_and_run(cache_dir)

    total_size = _dir_size(cache_dir)
    assert total_size > 0, f"Cache directory {cache_dir} is empty after compilation"


@gpu_light
def test_compile_cache_second_run_faster(tmp_path):
    """Second run after Dynamo reset should be faster thanks to disk cache."""
    cache_dir = str(tmp_path / "torch_compile")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    # First run: cold compile (no cache)
    time_first = _build_compile_and_run(cache_dir)
    cache_size_after_first = _dir_size(cache_dir)

    # Second run: Dynamo reset + fresh model (simulates app restart)
    time_second = _build_compile_and_run(cache_dir)
    cache_size_after_second = _dir_size(cache_dir)

    # Cache size should stay roughly the same (hit, not new entries)
    assert cache_size_after_second == cache_size_after_first, (
        f"Cache grew on second run ({cache_size_after_first} → {cache_size_after_second}), suggesting a cache miss"
    )

    assert time_second < time_first, (
        f"Second compilation ({time_second:.3f}s) was not faster than first ({time_first:.3f}s)"
    )
