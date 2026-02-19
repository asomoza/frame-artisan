"""Step-by-step T2V 2-stage comparison: pipeline vs graph nodes.

Each test isolates a single stage of the T2V pipeline and compares the
numerical output against the equivalent graph node using **shared model
instances** to eliminate quantization non-determinism.

Activated by setting the environment variable ``GPU_HEAVY=1``.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_t2v_stages.py -v -s -k <test_name>
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

pytestmark = pytest.mark.gpu_heavy

INTERMEDIATES_DIR = "/tmp/frame_artisan_t2v_stages"

# Shared test parameters
PROMPT = "a red ball"
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
WIDTH = 256
HEIGHT = 256
NUM_FRAMES = 9  # 8*1+1
FRAME_RATE = 8
SEED = 42


def _ensure_intermediates_dir():
    os.makedirs(INTERMEDIATES_DIR, exist_ok=True)


def _save_tensor(name: str, tensor: torch.Tensor):
    _ensure_intermediates_dir()
    torch.save(tensor, os.path.join(INTERMEDIATES_DIR, f"{name}.pt"))


def _load_tensor(name: str) -> torch.Tensor:
    path = os.path.join(INTERMEDIATES_DIR, f"{name}.pt")
    if not os.path.isfile(path):
        pytest.skip(f"Intermediate tensor not found: {path} — run previous stage first")
    return torch.load(path, weights_only=True)


def _full_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_db_path() -> str | None:
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


def _discover_model_by_version(db_path: str, version: str) -> dict | None:
    from frameartisan.utils.database import Database

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


def _resolve_paths(model_id: int, model_path: str) -> dict[str, str]:
    from frameartisan.app.app import get_app_database_path
    from frameartisan.app.component_registry import ComponentRegistry

    db_path = get_app_database_path()
    components_base_dir = os.path.join(os.path.dirname(model_path), "_components")
    registry = ComponentRegistry(db_path, components_base_dir)
    comps = registry.get_model_components(model_id)
    if comps:
        return {ct: info.storage_path for ct, info in comps.items()}
    return registry.resolve_component_paths(model_path) or {}


def _find_upsampler_path() -> str | None:
    from PyQt6.QtCore import QSettings

    settings = QSettings("ZCode", "FrameArtisan")
    models_path = settings.value("models_diffusers", "")
    if not models_path:
        return None
    path = os.path.join(models_path, "LTX2_latent_upsampler")
    return path if os.path.isdir(path) else None


def _load_components(model_info: dict):
    """Load individual pipeline components from resolved paths."""
    try:
        from sdnq import SDNQConfig  # noqa: F401
    except ImportError:
        pass

    from diffusers import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
        FlowMatchEulerDiscreteScheduler,
        LTX2VideoTransformer3DModel,
    )
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer

    model_path = model_info["filepath"]
    paths = _resolve_paths(model_info["id"], model_path)

    def _p(comp_type):
        return paths.get(comp_type, os.path.join(model_path, comp_type))

    heavy_kwargs = {"device_map": "cpu", "torch_dtype": torch.bfloat16}

    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(_p("text_encoder"), **heavy_kwargs)
    transformer = LTX2VideoTransformer3DModel.from_pretrained(_p("transformer"), **heavy_kwargs)
    tokenizer = GemmaTokenizer.from_pretrained(_p("tokenizer"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(_p("scheduler"))
    vae = AutoencoderKLLTX2Video.from_pretrained(_p("vae"), torch_dtype=torch.bfloat16)
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(_p("audio_vae"), torch_dtype=torch.bfloat16)
    connectors = LTX2TextConnectors.from_pretrained(_p("connectors"), torch_dtype=torch.bfloat16)
    vocoder = LTX2Vocoder.from_pretrained(_p("vocoder"), torch_dtype=torch.bfloat16)

    return {
        "text_encoder": text_encoder,
        "transformer": transformer,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
        "audio_vae": audio_vae,
        "connectors": connectors,
        "vocoder": vocoder,
    }


class _ValueNode:
    """Minimal stand-in that exposes .values and .enabled for node connections."""

    def __init__(self, **kwargs):
        self.values = kwargs
        self.enabled = True
        self.dependents = []


def _connect_node(node, input_name, source_node, output_name):
    """Wire a node input to a _ValueNode output without full graph machinery."""
    node.connections[input_name] = [(source_node, output_name)]
    if source_node not in node.dependencies:
        node.dependencies.append(source_node)


# ---------------------------------------------------------------------------
# Test 1: T2V Latent Preparation (pure noise, no image conditioning)
# ---------------------------------------------------------------------------


def test_t2v_stage_prepare_latents(app_db_path):
    """Compare T2V latent preparation: pipeline prepare_latents vs LatentsNode (no image)."""
    import warnings

    from diffusers import LTX2Pipeline

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "4bit")
    if model is None:
        pytest.skip("LTX2 Distilled 4-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    # --- Pipeline side: prepare_latents + prepare_audio_latents ---
    pipe = LTX2Pipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    generator = torch.Generator("cuda").manual_seed(SEED)
    num_channels_latents = pipe.transformer.config.in_channels

    with torch.inference_mode():
        pipe_latents = pipe.prepare_latents(
            1,
            num_channels_latents,
            HEIGHT,
            WIDTH,
            NUM_FRAMES,
            0.0,
            torch.float32,
            device,
            generator,
        )

    # Audio latents
    audio_vae = components["audio_vae"]
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        audio_vae.config.sample_rate
        / audio_vae.config.mel_hop_length
        / float(getattr(audio_vae, "temporal_compression_ratio", 4))
    )
    pipe_audio_num_frames = round(duration_s * audio_latents_per_second)

    with torch.inference_mode():
        pipe_audio_latents = pipe.prepare_audio_latents(
            1,
            audio_vae.config.latent_channels,
            pipe_audio_num_frames,
            audio_vae.config.mel_bins,
            0.0,
            torch.float32,
            device,
            generator,
        )

    pipe_latents_cpu = pipe_latents.cpu()
    pipe_audio_cpu = pipe_audio_latents.cpu()

    print(f"  Pipeline: video={pipe_latents_cpu.shape}, audio={pipe_audio_cpu.shape}")

    del pipe
    _full_cleanup()

    # --- Node side: LatentsNode (no image inputs) ---
    from frameartisan.modules.generation.graph.nodes.ltx2_latents_node import LTX2LatentsNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.apply_offload_strategy(device)

    lat_node = LTX2LatentsNode()
    lat_node.device = device
    lat_source = _ValueNode(
        transformer=components["transformer"],
        vae=components["vae"],
        audio_vae=components["audio_vae"],
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        frame_rate=float(FRAME_RATE),
        seed=SEED,
    )
    for inp in ["transformer", "vae", "audio_vae", "num_frames", "height", "width", "frame_rate", "seed"]:
        _connect_node(lat_node, inp, lat_source, inp)

    with torch.inference_mode():
        lat_node()

    node_video = lat_node.values["video_latents"].cpu()
    node_audio = lat_node.values["audio_latents"].cpu()
    node_mask = lat_node.values["conditioning_mask"]

    print(f"  Node: video={node_video.shape}, audio={node_audio.shape}, cond_mask={node_mask}")

    # T2V should produce no conditioning_mask
    assert node_mask is None, f"T2V should have no conditioning_mask, got {type(node_mask)}"

    # --- Compare ---
    video_diff = (pipe_latents_cpu.float() - node_video.float()).abs()
    audio_diff = (pipe_audio_cpu.float() - node_audio.float()).abs()

    print(f"  Video latents: max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}")
    print(f"  Audio latents: max_diff={audio_diff.max():.8f}, mean_diff={audio_diff.mean():.8f}")

    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"

    # Save for next test
    _save_tensor("prepared_video_latents", node_video)
    _save_tensor("prepared_audio_latents", node_audio)

    print("  PASS: T2V latent preparation matches")

    del components, lat_node
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 2: T2V First-stage Denoise (no conditioning mask)
# ---------------------------------------------------------------------------


def test_t2v_stage_denoise(app_db_path):
    """Compare T2V first-stage denoising: manual pipeline loop vs DenoiseNode (no cond mask)."""
    import copy
    import warnings

    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import calculate_shift

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "4bit")
    if model is None:
        pytest.skip("LTX2 Distilled 4-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    # --- Prepare shared inputs using pipeline ---
    pipe = LTX2Pipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    # Encode prompt
    with torch.inference_mode():
        (
            prompt_embeds,
            prompt_attention_mask,
            _neg_embeds,
            _neg_mask,
        ) = pipe.encode_prompt(PROMPT, NEGATIVE_PROMPT, do_classifier_free_guidance=False)

    # Connectors
    pipe.connectors.to(device)
    with torch.inference_mode():
        additive_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        video_embeds, audio_embeds, connector_mask = pipe.connectors(
            prompt_embeds,
            additive_mask,
            additive_mask=True,
        )
    pipe.connectors.to("cpu")

    # Prepare latents
    generator = torch.Generator("cuda").manual_seed(SEED)
    num_channels_latents = pipe.transformer.config.in_channels

    with torch.inference_mode():
        video_latents = pipe.prepare_latents(
            1,
            num_channels_latents,
            HEIGHT,
            WIDTH,
            NUM_FRAMES,
            0.0,
            torch.float32,
            device,
            generator,
        )

    # Audio latents
    audio_vae = components["audio_vae"]
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)

    with torch.inference_mode():
        audio_latents = pipe.prepare_audio_latents(
            1,
            audio_vae.config.latent_channels,
            audio_num_frames,
            audio_vae.config.mel_bins,
            0.0,
            torch.float32,
            device,
            generator,
        )

    # Latent dimensions
    vae_temporal_ratio = pipe.vae_temporal_compression_ratio
    vae_spatial_ratio = pipe.vae_spatial_compression_ratio
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = HEIGHT // vae_spatial_ratio
    latent_width = WIDTH // vae_spatial_ratio
    video_sequence_length = latent_num_frames * latent_height * latent_width

    # RoPE coords
    transformer = components["transformer"]
    video_coords = transformer.rope.prepare_video_coords(
        video_latents.shape[0],
        latent_num_frames,
        latent_height,
        latent_width,
        video_latents.device,
        fps=float(FRAME_RATE),
    )
    audio_coords = transformer.audio_rope.prepare_audio_coords(
        audio_latents.shape[0],
        audio_num_frames,
        audio_latents.device,
    )

    # Clone shared inputs
    shared_video_latents = video_latents.clone()
    shared_audio_latents = audio_latents.clone()

    print(f"  Inputs: video={video_latents.shape}, audio={audio_latents.shape}")
    print(f"  Latent dims: F={latent_num_frames}, H={latent_height}, W={latent_width}")

    del pipe
    _full_cleanup()

    # --- Pipeline side: manual denoising loop (T2V: no conditioning mask) ---
    from diffusers import FlowMatchEulerDiscreteScheduler

    sigmas = DISTILLED_SIGMA_VALUES
    num_steps = 8

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(components["scheduler"].config)
    audio_scheduler = copy.deepcopy(scheduler)

    mu = calculate_shift(
        video_sequence_length,
        scheduler.config.get("base_image_seq_len", 1024),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.95),
        scheduler.config.get("max_shift", 2.05),
    )
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    audio_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = scheduler.timesteps

    pipe_video = shared_video_latents.clone()
    pipe_audio = shared_audio_latents.clone()

    print(f"  Running pipeline-style T2V loop: {len(timesteps)} steps...")

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("transformer", transformer)
    mm.apply_offload_strategy(device)

    with torch.inference_mode(), mm.use_components("transformer", device=device):
        for i, t in enumerate(timesteps):
            latent_model_input = pipe_video.to(video_embeds.dtype)
            audio_model_input = pipe_audio.to(audio_embeds.dtype)

            timestep = t.expand(latent_model_input.shape[0])

            with transformer.cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = transformer(
                    hidden_states=latent_model_input,
                    audio_hidden_states=audio_model_input,
                    encoder_hidden_states=video_embeds,
                    audio_encoder_hidden_states=audio_embeds,
                    timestep=timestep,
                    encoder_attention_mask=connector_mask,
                    audio_encoder_attention_mask=connector_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=float(FRAME_RATE),
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    return_dict=False,
                )

            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            # T2V: simple scheduler step on full latents (no frame separation)
            pipe_video = scheduler.step(noise_pred_video, t, pipe_video, return_dict=False)[0]
            pipe_audio = audio_scheduler.step(noise_pred_audio, t, pipe_audio, return_dict=False)[0]

    pipe_video_cpu = pipe_video.cpu()
    pipe_audio_cpu = pipe_audio.cpu()

    print(f"  Pipeline done: video={pipe_video_cpu.shape}, audio={pipe_audio_cpu.shape}")

    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()

    # --- Node side: DenoiseNode (no conditioning_mask) ---
    from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("transformer", transformer)
    mm.apply_offload_strategy(device)

    node = LTX2DenoiseNode()
    node.device = device

    source = _ValueNode(
        transformer=transformer,
        scheduler_config=components["scheduler"].config,
        prompt_embeds=video_embeds,
        audio_prompt_embeds=audio_embeds,
        attention_mask=connector_mask,
        video_latents=shared_video_latents.clone(),
        audio_latents=shared_audio_latents.clone(),
        video_coords=video_coords,
        audio_coords=audio_coords,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_num_frames=audio_num_frames,
        num_inference_steps=num_steps,
        guidance_scale=1.0,
        frame_rate=float(FRAME_RATE),
        model_type=2,  # distilled
        stage=1,
        transformer_component_name="transformer",
    )

    for inp in LTX2DenoiseNode.REQUIRED_INPUTS + LTX2DenoiseNode.OPTIONAL_INPUTS:
        if hasattr(source, "values") and inp in source.values:
            _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_video = node.values["video_latents"].cpu()
    node_audio = node.values["audio_latents"].cpu()

    print(f"  Node done: video={node_video.shape}, audio={node_audio.shape}")

    # --- Compare ---
    assert pipe_video_cpu.shape == node_video.shape, (
        f"Video shape mismatch: pipe={pipe_video_cpu.shape} vs node={node_video.shape}"
    )
    assert pipe_audio_cpu.shape == node_audio.shape, (
        f"Audio shape mismatch: pipe={pipe_audio_cpu.shape} vs node={node_audio.shape}"
    )

    video_diff = (pipe_video_cpu.float() - node_video.float()).abs()
    audio_diff = (pipe_audio_cpu.float() - node_audio.float()).abs()

    print(f"  Video latents: max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}")
    print(f"  Audio latents: max_diff={audio_diff.max():.8f}, mean_diff={audio_diff.mean():.8f}")

    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"

    _save_tensor("denoised_video_latents", node_video)
    _save_tensor("denoised_audio_latents", node_audio)

    print("  PASS: T2V first-stage denoise matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 3: T2V Second Pass Latent Preparation (adds noise to video + audio)
# ---------------------------------------------------------------------------


def test_t2v_stage_second_pass_latents(app_db_path):
    """Compare T2V stage-2 latent preparation: pipeline vs SecondPassLatentsNode (T2V mode)."""
    import warnings

    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "4bit")
    if model is None:
        pytest.skip("LTX2 Distilled 4-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    # Stage 2 operates at 2x spatial resolution
    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2

    vae = components["vae"]
    vae_temporal_ratio = getattr(vae, "temporal_compression_ratio", 8)
    vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 32)
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = stage2_height // vae_spatial_ratio
    latent_width = stage2_width // vae_spatial_ratio
    video_seq_len = latent_num_frames * latent_height * latent_width

    # Create fake upsampled video latents (packed 3D)
    gen_fake = torch.Generator("cpu").manual_seed(99)
    fake_video_latents = torch.randn(1, video_seq_len, 128, generator=gen_fake, dtype=torch.float32)

    # Create fake audio latents from stage 1
    audio_vae = components["audio_vae"]
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        audio_vae.config.sample_rate
        / audio_vae.config.mel_hop_length
        / float(getattr(audio_vae, "temporal_compression_ratio", 4))
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    latent_mel_bins = audio_vae.config.mel_bins // getattr(audio_vae, "mel_compression_ratio", 4)
    audio_feat = audio_vae.config.latent_channels * latent_mel_bins
    fake_audio_latents = torch.randn(1, audio_num_frames, audio_feat, generator=gen_fake, dtype=torch.float32)

    noise_scale = STAGE_2_DISTILLED_SIGMA_VALUES[0]

    print(f"  Fake inputs: video={fake_video_latents.shape}, audio={fake_audio_latents.shape}")
    print(f"  noise_scale={noise_scale}")

    # --- Pipeline side ---
    pipe = LTX2Pipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    num_channels_latents = pipe.transformer.config.in_channels

    generator = torch.Generator("cuda").manual_seed(SEED)

    with torch.inference_mode():
        pipe_video = pipe.prepare_latents(
            1,
            num_channels_latents,
            stage2_height,
            stage2_width,
            NUM_FRAMES,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents=fake_video_latents.clone().to(device),
        )

    # Audio
    num_mel_bins = audio_vae.config.mel_bins
    with torch.inference_mode():
        pipe_audio = pipe.prepare_audio_latents(
            1,
            audio_vae.config.latent_channels,
            audio_num_frames,
            num_mel_bins,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents=fake_audio_latents.clone().to(device),
        )

    pipe_video_cpu = pipe_video.cpu()
    pipe_audio_cpu = pipe_audio.cpu()

    print(f"  Pipeline: video={pipe_video_cpu.shape}, audio={pipe_audio_cpu.shape}")

    del pipe
    _full_cleanup()

    # --- Node side: SecondPassLatentsNode (T2V mode — no conditioning_mask) ---
    from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import LTX2SecondPassLatentsNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.apply_offload_strategy(device)

    transformer = components["transformer"]
    audio_coords = transformer.audio_rope.prepare_audio_coords(1, audio_num_frames, device)

    node = LTX2SecondPassLatentsNode()
    node.device = device

    # T2V mode: no conditioning_mask input
    source = _ValueNode(
        transformer=transformer,
        audio_vae=components["audio_vae"],
        video_latents=fake_video_latents.clone().to(device),
        audio_latents=fake_audio_latents.clone().to(device),
        audio_coords=audio_coords,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_num_frames=audio_num_frames,
        frame_rate=float(FRAME_RATE),
        seed=SEED,
        model_type=2,
    )
    for inp in LTX2SecondPassLatentsNode.REQUIRED_INPUTS:
        if inp in source.values:
            _connect_node(node, inp, source, inp)
    # Do NOT connect conditioning_mask — T2V mode

    with torch.inference_mode():
        node()

    node_video = node.values["video_latents"].cpu()
    node_audio = node.values["audio_latents"].cpu()
    node_mask = node.values["conditioning_mask"]

    print(f"  Node: video={node_video.shape}, audio={node_audio.shape}, mask={node_mask}")

    # T2V should produce no conditioning_mask
    assert node_mask is None, f"T2V SecondPassLatents should have no conditioning_mask, got {type(node_mask)}"

    # --- Compare ---
    video_diff = (pipe_video_cpu.float() - node_video.float()).abs()
    audio_diff = (pipe_audio_cpu.float() - node_audio.float()).abs()

    print(f"  Video latents: max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}")
    print(f"  Audio latents: max_diff={audio_diff.max():.8f}, mean_diff={audio_diff.mean():.8f}")

    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"

    print("  PASS: T2V second pass latent preparation matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 4: T2V Second-stage Denoise (no conditioning mask)
# ---------------------------------------------------------------------------


def test_t2v_stage_second_denoise(app_db_path):
    """Compare T2V second-stage denoising: manual pipeline loop vs DenoiseNode (stage=2, no cond mask)."""
    import copy
    import warnings

    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import calculate_shift

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "4bit")
    if model is None:
        pytest.skip("LTX2 Distilled 4-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    # Stage 2 operates at 2x spatial resolution
    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2

    # --- Prepare shared inputs using pipeline ---
    pipe = LTX2Pipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    # Encode prompt
    with torch.inference_mode():
        (
            prompt_embeds,
            prompt_attention_mask,
            _neg_embeds,
            _neg_mask,
        ) = pipe.encode_prompt(PROMPT, NEGATIVE_PROMPT, do_classifier_free_guidance=False)

    # Connectors
    pipe.connectors.to(device)
    with torch.inference_mode():
        additive_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        video_embeds, audio_embeds, connector_mask = pipe.connectors(
            prompt_embeds,
            additive_mask,
            additive_mask=True,
        )
    pipe.connectors.to("cpu")

    # Create fake stage-2 latents
    vae_temporal_ratio = pipe.vae_temporal_compression_ratio
    vae_spatial_ratio = pipe.vae_spatial_compression_ratio
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = stage2_height // vae_spatial_ratio
    latent_width = stage2_width // vae_spatial_ratio
    video_seq_len = latent_num_frames * latent_height * latent_width

    noise_scale = STAGE_2_DISTILLED_SIGMA_VALUES[0]
    num_channels_latents = pipe.transformer.config.in_channels

    gen_fake = torch.Generator("cpu").manual_seed(99)
    fake_video_latents = torch.randn(1, video_seq_len, 128, generator=gen_fake, dtype=torch.float32).to(device)

    audio_vae = components["audio_vae"]
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    latent_mel_bins = audio_vae.config.mel_bins // pipe.audio_vae_mel_compression_ratio
    audio_feat = audio_vae.config.latent_channels * latent_mel_bins
    fake_audio_latents = torch.randn(1, audio_num_frames, audio_feat, generator=gen_fake, dtype=torch.float32).to(
        device
    )

    generator = torch.Generator("cuda").manual_seed(SEED)

    with torch.inference_mode():
        video_latents = pipe.prepare_latents(
            1,
            num_channels_latents,
            stage2_height,
            stage2_width,
            NUM_FRAMES,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents=fake_video_latents.clone(),
        )
        audio_latents = pipe.prepare_audio_latents(
            1,
            audio_vae.config.latent_channels,
            audio_num_frames,
            audio_vae.config.mel_bins,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents=fake_audio_latents.clone(),
        )

    video_sequence_length = latent_num_frames * latent_height * latent_width

    # RoPE coords
    transformer = components["transformer"]
    video_coords = transformer.rope.prepare_video_coords(
        video_latents.shape[0],
        latent_num_frames,
        latent_height,
        latent_width,
        video_latents.device,
        fps=float(FRAME_RATE),
    )
    audio_coords = transformer.audio_rope.prepare_audio_coords(
        audio_latents.shape[0],
        audio_num_frames,
        audio_latents.device,
    )

    shared_video_latents = video_latents.clone()
    shared_audio_latents = audio_latents.clone()

    print(f"  Inputs: video={video_latents.shape}, audio={audio_latents.shape}")
    print(f"  Latent dims: F={latent_num_frames}, H={latent_height}, W={latent_width}")

    del pipe
    _full_cleanup()

    # --- Pipeline side: manual stage 2 T2V denoising loop ---
    sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
    num_steps = 3

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        components["scheduler"].config, use_dynamic_shifting=False, shift_terminal=None
    )
    audio_scheduler = copy.deepcopy(scheduler)

    mu = calculate_shift(
        video_sequence_length,
        scheduler.config.get("base_image_seq_len", 1024),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.95),
        scheduler.config.get("max_shift", 2.05),
    )
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    audio_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = scheduler.timesteps

    pipe_video = shared_video_latents.clone()
    pipe_audio = shared_audio_latents.clone()

    print(f"  Running pipeline-style T2V stage 2 loop: {len(timesteps)} steps...")

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("transformer", transformer)
    mm.apply_offload_strategy(device)

    with torch.inference_mode(), mm.use_components("transformer", device=device):
        for i, t in enumerate(timesteps):
            latent_model_input = pipe_video.to(video_embeds.dtype)
            audio_model_input = pipe_audio.to(audio_embeds.dtype)

            timestep = t.expand(latent_model_input.shape[0])

            with transformer.cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = transformer(
                    hidden_states=latent_model_input,
                    audio_hidden_states=audio_model_input,
                    encoder_hidden_states=video_embeds,
                    audio_encoder_hidden_states=audio_embeds,
                    timestep=timestep,
                    encoder_attention_mask=connector_mask,
                    audio_encoder_attention_mask=connector_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=float(FRAME_RATE),
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    return_dict=False,
                )

            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            # T2V: simple scheduler step on full latents
            pipe_video = scheduler.step(noise_pred_video, t, pipe_video, return_dict=False)[0]
            pipe_audio = audio_scheduler.step(noise_pred_audio, t, pipe_audio, return_dict=False)[0]

    pipe_video_cpu = pipe_video.cpu()
    pipe_audio_cpu = pipe_audio.cpu()

    print(f"  Pipeline done: video={pipe_video_cpu.shape}, audio={pipe_audio_cpu.shape}")

    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()

    # --- Node side: DenoiseNode stage=2, no conditioning_mask ---
    from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("transformer", transformer)
    mm.apply_offload_strategy(device)

    node = LTX2DenoiseNode()
    node.device = device

    source = _ValueNode(
        transformer=transformer,
        scheduler_config=components["scheduler"].config,
        prompt_embeds=video_embeds,
        audio_prompt_embeds=audio_embeds,
        attention_mask=connector_mask,
        video_latents=shared_video_latents.clone(),
        audio_latents=shared_audio_latents.clone(),
        video_coords=video_coords,
        audio_coords=audio_coords,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_num_frames=audio_num_frames,
        num_inference_steps=num_steps,
        guidance_scale=1.0,
        frame_rate=float(FRAME_RATE),
        model_type=2,
        stage=2,
        transformer_component_name="transformer",
    )

    for inp in LTX2DenoiseNode.REQUIRED_INPUTS + LTX2DenoiseNode.OPTIONAL_INPUTS:
        if hasattr(source, "values") and inp in source.values:
            _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_video = node.values["video_latents"].cpu()
    node_audio = node.values["audio_latents"].cpu()

    print(f"  Node done: video={node_video.shape}, audio={node_audio.shape}")

    # --- Compare ---
    assert pipe_video_cpu.shape == node_video.shape, (
        f"Video shape mismatch: pipe={pipe_video_cpu.shape} vs node={node_video.shape}"
    )
    assert pipe_audio_cpu.shape == node_audio.shape, (
        f"Audio shape mismatch: pipe={pipe_audio_cpu.shape} vs node={node_audio.shape}"
    )

    video_diff = (pipe_video_cpu.float() - node_video.float()).abs()
    audio_diff = (pipe_audio_cpu.float() - node_audio.float()).abs()

    print(f"  Video latents: max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}")
    print(f"  Audio latents: max_diff={audio_diff.max():.8f}, mean_diff={audio_diff.mean():.8f}")

    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"

    print("  PASS: T2V second-stage denoise matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()
