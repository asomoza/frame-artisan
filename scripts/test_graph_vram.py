#!/usr/bin/env python3
"""Standalone graph-based LTX-2.0 inference script for VRAM debugging.

Builds and runs the same node graph the app uses, but without the GUI.
Reads model paths from the app's database so component resolution works.

Usage:
    uv run python scripts/test_graph_vram.py [options]

Options:
    --model-name NAME       Model name substring to match (default: first LTX-2 found)
    --width W               Video width (default: 768)
    --height H              Video height (default: 512)
    --duration S            Video duration in seconds (default: 5)
    --frame-rate FPS        Frame rate (default: 24)
    --steps N               Inference steps (default: 24)
    --guidance G            Guidance scale (default: 4.0)
    --seed S                Seed (default: random)
    --offload STRATEGY      Offload strategy (default: group_offload)
    --use-stream            Enable group_offload use_stream (default: off)
    --low-cpu-mem           Enable low_cpu_mem_usage (default: off)
    --streaming-decode      Enable streaming VAE decode (default: off)
    --ff-chunking           Enable feed-forward chunking (default: off)
    --two-stage             Enable two-stage with spatial upscaler
    --sp-model-name NAME    Second-pass model name (for --two-stage)
    --upsampler-path PATH   Latent upsampler path (for --two-stage)
    --prompt TEXT            Prompt (default: test prompt)
    --output-dir DIR        Output directory (default: ./outputs)
    --lora PATH             LoRA safetensors file to apply (can repeat)
    --lora-weight W         LoRA weight (default: 1.0, applies to last --lora)
"""

import argparse
import ctypes
import gc
import logging
import os
import sys
import time

import torch

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def vram_info() -> str:
    if not torch.cuda.is_available():
        return "no CUDA"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB"


def get_db_path() -> str:
    """Read the app's QSettings to find the database."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    settings = QSettings("ZCode", "FrameArtisan")
    data_path = settings.value("data_path", "")
    if not data_path:
        raise RuntimeError(
            "No data_path in QSettings. Run the app once first to set up the database."
        )
    db_path = os.path.join(data_path, "app.db")
    if not os.path.isfile(db_path):
        raise RuntimeError(f"Database not found at {db_path}")
    return db_path


def find_model(db_path: str, name_filter: str | None = None, model_type: int | None = None):
    """Find a model in the database."""
    from frameartisan.utils.database import Database
    db = Database(db_path)
    rows = db.fetch_all(
        "SELECT id, name, filepath, model_type, version FROM model WHERE deleted = 0"
    )
    db.disconnect()

    for row in rows:
        row_id, name, filepath, mt, version = row
        if model_type is not None and mt != model_type:
            continue
        if name_filter and name_filter.lower() not in name.lower():
            continue
        return {"id": row_id, "name": name, "filepath": filepath, "model_type": mt, "version": version}

    available = [(r[1], r[3], r[4]) for r in rows]
    raise RuntimeError(
        f"No model found matching filter={name_filter!r} type={model_type}.\n"
        f"Available: {available}"
    )


def list_models(db_path: str):
    """Print all available models."""
    from frameartisan.utils.database import Database
    db = Database(db_path)
    rows = db.fetch_all(
        "SELECT id, name, filepath, model_type, version FROM model WHERE deleted = 0"
    )
    db.disconnect()
    print("\nAvailable models:")
    print(f"  {'ID':>4}  {'Type':>4}  {'Version':<10}  {'Name'}")
    print(f"  {'──':>4}  {'────':>4}  {'───────':<10}  {'────'}")
    for row in rows:
        row_id, name, filepath, mt, version = row
        print(f"  {row_id:>4}  {mt:>4}  {version:<10}  {name}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph-based LTX-2 inference for VRAM debugging")
    parser.add_argument("--model-name", type=str, default=None, help="Model name substring")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--frame-rate", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", type=str, default="group_offload",
                        choices=["auto", "no_offload", "model_offload",
                                 "sequential_group_offload", "group_offload"])
    parser.add_argument("--use-stream", action="store_true")
    parser.add_argument("--low-cpu-mem", action="store_true")
    parser.add_argument("--streaming-decode", action="store_true")
    parser.add_argument("--ff-chunking", action="store_true")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--sp-model-name", type=str, default=None,
                        help="Second-pass model name (for --two-stage)")
    parser.add_argument("--upsampler-path", type=str, default=None,
                        help="Latent upsampler model path (for --two-stage)")
    parser.add_argument("--prompt", type=str,
                        default="A cinematic shot of a golden retriever running through a field of sunflowers.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--lora", type=str, action="append", default=[], help="LoRA file path")
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    db_path = get_db_path()
    logger.info("Database: %s", db_path)

    if args.list_models:
        list_models(db_path)
        return

    # Register DB path with the app module so component resolution works
    from frameartisan.app.app import set_app_database_path
    set_app_database_path(db_path)

    from frameartisan.app.model_manager import get_model_manager
    from frameartisan.modules.generation.constants import DEFAULT_NEGATIVE_PROMPT
    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings, compute_num_frames
    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph

    # ── Find models ──────────────────────────────────────────────────────
    model = find_model(db_path, args.model_name)
    logger.info("Model: %s (id=%d, type=%d, path=%s)", model["name"], model["id"],
                model["model_type"], model["filepath"])

    model_obj = ModelDataObject(
        name=model["name"],
        filepath=model["filepath"],
        model_type=model["model_type"],
        id=model["id"],
    )

    # Second-pass model (for two-stage)
    sp_model_obj = ModelDataObject()
    if args.two_stage:
        if not args.sp_model_name:
            raise RuntimeError("--sp-model-name required with --two-stage")
        if not args.upsampler_path:
            raise RuntimeError("--upsampler-path required with --two-stage")
        sp_model = find_model(db_path, args.sp_model_name)
        logger.info("Second-pass model: %s (id=%d)", sp_model["name"], sp_model["id"])
        sp_model_obj = ModelDataObject(
            name=sp_model["name"],
            filepath=sp_model["filepath"],
            model_type=sp_model["model_type"],
            id=sp_model["id"],
        )

    # ── Build settings ───────────────────────────────────────────────────
    seed = args.seed or torch.randint(0, 2**32, (1,)).item()
    logger.info("Seed: %d", seed)

    settings = GenerationSettings(
        model=model_obj,
        second_pass_model=sp_model_obj,
        video_width=args.width,
        video_height=args.height,
        video_duration=args.duration,
        frame_rate=args.frame_rate,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        offload_strategy=args.offload,
        group_offload_use_stream=args.use_stream,
        group_offload_low_cpu_mem=args.low_cpu_mem,
        streaming_decode=args.streaming_decode,
        ff_chunking=args.ff_chunking,
        use_torch_compile=False,
        second_pass_enabled=args.two_stage,
    )

    num_frames = compute_num_frames(args.duration, args.frame_rate)
    logger.info("Resolution: %dx%d, duration: %ds, frames: %d, fps: %d",
                args.width, args.height, args.duration, num_frames, args.frame_rate)

    # ── Build LoRA configs ───────────────────────────────────────────────
    lora_configs = []
    for i, lora_path in enumerate(args.lora):
        import hashlib
        lora_hash = hashlib.sha256(lora_path.encode()).hexdigest()[:12]
        lora_configs.append({
            "name": os.path.basename(lora_path),
            "hash": lora_hash,
            "filepath": lora_path,
            "enabled": True,
            "weight": args.lora_weight,
            "video_strength": 1.0,
            "audio_strength": 1.0,
        })
    settings.active_loras = lora_configs

    # ── Directories stub ─────────────────────────────────────────────────
    import attr

    @attr.s
    class _Dirs:
        data_path = attr.ib(default="")
        models_diffusers = attr.ib(default="")
        models_loras = attr.ib(default="")
        models_controlnets = attr.ib(default="")
        outputs_videos = attr.ib(default="")

    os.makedirs(args.output_dir, exist_ok=True)
    dirs = _Dirs(outputs_videos=args.output_dir)

    # ── Build graph ──────────────────────────────────────────────────────
    logger.info("Building graph...")
    graph = create_default_ltx2_graph(settings, dirs)

    # Set prompt
    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = args.prompt

    # Set seed
    seed_node = graph.get_node_by_name("seed")
    seed_node.number = seed

    # Enable two-stage nodes if requested
    if args.two_stage:
        for name in ["second_pass_model", "second_pass_lora", "upsample",
                      "second_pass_latents", "second_pass_denoise"]:
            node = graph.get_node_by_name(name)
            if node:
                node.enabled = True
                node.updated = True

        upsample_node = graph.get_node_by_name("upsample")
        upsample_node.upsampler_model_path = args.upsampler_path

        # Stage 1 runs at half resolution; the upsampler doubles to target.
        width_node = graph.get_node_by_name("width")
        height_node = graph.get_node_by_name("height")
        width_node.number = args.width // 2
        height_node.number = args.height // 2
        logger.info("Two-stage: stage 1 at %dx%d, stage 2 at %dx%d",
                    args.width // 2, args.height // 2, args.width, args.height)

        # Rewire decode to read from second_pass_denoise instead of denoise.
        # Must disconnect old connections first (connect() appends, not replaces).
        decode_node = graph.get_node_by_name("decode")
        denoise_node_ref = graph.get_node_by_name("denoise")
        latents_node_ref = graph.get_node_by_name("latents")
        sp_denoise = graph.get_node_by_name("second_pass_denoise")
        sp_latents = graph.get_node_by_name("second_pass_latents")
        upsample = graph.get_node_by_name("upsample")

        decode_node.disconnect("video_latents", denoise_node_ref, "video_latents")
        decode_node.disconnect("audio_latents", denoise_node_ref, "audio_latents")
        decode_node.disconnect("latent_num_frames", latents_node_ref, "latent_num_frames")
        decode_node.disconnect("latent_height", latents_node_ref, "latent_height")
        decode_node.disconnect("latent_width", latents_node_ref, "latent_width")
        decode_node.disconnect("audio_num_frames", latents_node_ref, "audio_num_frames")

        decode_node.connect("video_latents", sp_denoise, "video_latents")
        decode_node.connect("audio_latents", sp_denoise, "audio_latents")
        decode_node.connect("latent_num_frames", sp_latents, "latent_num_frames")
        decode_node.connect("latent_height", upsample, "latent_height")
        decode_node.connect("latent_width", upsample, "latent_width")
        decode_node.connect("audio_num_frames", sp_latents, "audio_num_frames")

    # ── Callbacks ────────────────────────────────────────────────────────
    video_paths = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    denoise_node = graph.get_node_by_name("denoise")

    def _on_step(step, t, _preview):
        logger.info("  Step %d/%d  t=%.4f  %s", step + 1, args.steps, t.item(), vram_info())

    denoise_node.callback = _on_step

    if args.two_stage:
        sp_denoise_node = graph.get_node_by_name("second_pass_denoise")

        def _on_sp_step(step, t, _preview):
            logger.info("  2nd pass step %d/3  t=%.4f  %s", step + 1, t.item(), vram_info())

        sp_denoise_node.callback = _on_sp_step

    # ── Run ──────────────────────────────────────────────────────────────
    device = torch.device("cuda")
    graph.device = device
    graph.dtype = torch.bfloat16

    logger.info("Starting generation...  %s", vram_info())
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    try:
        graph()
    except Exception as e:
        logger.error("Generation failed: %s", e)
        logger.error("VRAM at failure: %s", vram_info())
        logger.error("Peak VRAM: %.2f GB", torch.cuda.max_memory_allocated() / 1024**3)
        raise

    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    logger.info("Generation complete in %.1fs", elapsed)
    logger.info("Peak VRAM: %.2f GB", peak_vram)
    logger.info("Final VRAM: %s", vram_info())

    if video_paths:
        logger.info("Output: %s", video_paths[0])
    else:
        logger.warning("No video output produced")

    # ── Cleanup ──────────────────────────────────────────────────────────
    del graph
    mm = get_model_manager()
    mm.clear()
    flush()


if __name__ == "__main__":
    main()
