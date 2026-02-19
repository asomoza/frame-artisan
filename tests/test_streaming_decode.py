"""Compare VAE streaming temporal decode vs standard single-shot decode.

The streaming path keeps latents on CPU and tiles them to GPU one at a time,
reducing peak VRAM.  This test verifies numerical equivalence.

Activated by setting the environment variable ``GPU_HEAVY=1``.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_streaming_decode.py -v -s
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

pytestmark = pytest.mark.gpu_heavy


def _full_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_vae_path(app_db_path: str) -> str | None:
    """Resolve the VAE component path from the first LTX2 model in the DB."""
    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.component_registry import ComponentRegistry
    from frameartisan.utils.database import Database

    set_app_database_path(app_db_path)
    db = Database(app_db_path)
    rows = db.fetch_all(
        "SELECT id, filepath FROM model WHERE deleted = 0 AND model_type = 2 LIMIT 1",
    )
    db.disconnect()
    if not rows:
        return None

    model_id, model_path = rows[0]
    components_base_dir = os.path.join(os.path.dirname(model_path), "_components")
    registry = ComponentRegistry(app_db_path, components_base_dir)
    comps = registry.get_model_components(model_id)
    if comps and "vae" in comps:
        return comps["vae"].storage_path
    resolved = registry.resolve_component_paths(model_path) or {}
    return resolved.get("vae", os.path.join(model_path, "vae"))


def test_streaming_decode_matches_standard(app_db_path):
    """Standard VAE decode vs streaming temporal decode must produce identical output."""
    from diffusers import AutoencoderKLLTX2Video

    from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
        vae_temporal_decode_streaming,
    )

    vae_path = _get_vae_path(app_db_path)
    if vae_path is None or not os.path.isdir(vae_path):
        pytest.skip("VAE path not found")

    device = torch.device("cuda")
    vae = AutoencoderKLLTX2Video.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
    vae.to(device)
    vae.enable_tiling()
    # Enable framewise (temporal) decoding — this is what the decode node uses
    # and what triggers _temporal_tiled_decode inside the VAE.
    vae.use_framewise_decoding = True

    # Determine latent dimensions.
    # Use enough temporal frames to trigger tiling (> tile_sample_min_num_frames // temporal_compression_ratio).
    temporal_ratio = vae.temporal_compression_ratio
    tile_latent_min = vae.tile_sample_min_num_frames // temporal_ratio

    # We need latent_frames > tile_latent_min to exercise the tiling path.
    # Use 2x the minimum to get at least 2 tiles.
    latent_frames = tile_latent_min * 2 + 1
    latent_h = 8  # small spatial to keep VRAM manageable
    latent_w = 8
    latent_channels = vae.config.latent_channels

    print(f"\n  VAE tiling params: tile_latent_min={tile_latent_min}, temporal_ratio={temporal_ratio}")
    print(f"  Test latents: [1, {latent_channels}, {latent_frames}, {latent_h}, {latent_w}]")

    # Create deterministic random latents
    gen = torch.Generator("cpu").manual_seed(42)
    latents = torch.randn(1, latent_channels, latent_frames, latent_h, latent_w, generator=gen, dtype=torch.bfloat16)

    # Timestep embedding (matching decode node)
    temb = None
    if getattr(vae.config, "timestep_conditioning", False):
        temb = torch.tensor([0.0], device=device, dtype=torch.bfloat16)

    # --- Standard decode (full latents on GPU) ---
    print("  Running standard decode...")
    with torch.inference_mode():
        standard_output = vae.decode(latents.to(device), temb=temb, return_dict=False)[0].cpu()

    _full_cleanup()

    # --- Streaming decode (latents on CPU, tiles streamed) ---
    print("  Running streaming decode...")
    with torch.inference_mode():
        streaming_output = vae_temporal_decode_streaming(
            vae,
            latents.cpu(),
            device=device,
            temb=temb,
        )

    _full_cleanup()

    # --- Compare ---
    assert standard_output.shape == streaming_output.shape, (
        f"Shape mismatch: standard={standard_output.shape} vs streaming={streaming_output.shape}"
    )

    diff = (standard_output.float() - streaming_output.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")

    # Temporal tiling involves blending, so the standard single-shot decode also
    # uses temporal tiling internally (since tiling is enabled and frames > threshold).
    # Both paths should produce identical results.
    assert max_diff == 0, f"Outputs differ: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}"

    print("  PASS: streaming decode matches standard decode")

    del vae
    _full_cleanup()
