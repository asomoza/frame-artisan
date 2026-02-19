"""Step-by-step I2V 2-stage comparison: pipeline vs graph nodes.

Each test isolates a single stage of the pipeline and compares the numerical
output against the equivalent graph node.  Intermediate tensors are saved to
``/tmp/frame_artisan_i2v_stages/`` so later tests can load them without
re-running previous stages.

Activated by setting the environment variable ``GPU_HEAVY=1``.

    GPU_HEAVY=1 uv run --extra test pytest tests/test_i2v_stages.py -v -s -k <test_name>
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

pytestmark = pytest.mark.gpu_heavy

INTERMEDIATES_DIR = "/tmp/frame_artisan_i2v_stages"

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
# Test 1: Prompt Encoding
# ---------------------------------------------------------------------------


def test_i2v_stage_prompt_encode(app_db_path):
    """Compare prompt encoding: pipeline encode_prompt + connectors vs PromptEncodeNode."""
    import warnings

    from frameartisan.app.app import set_app_database_path

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    # Use the SAME model instances for both pipeline and node to eliminate
    # quantization rounding differences from separate loads.

    # --- Pipeline side: encode_prompt + connectors ---
    from diffusers import LTX2ImageToVideoPipeline

    pipe = LTX2ImageToVideoPipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    with torch.inference_mode():
        (
            pipe_prompt_embeds,
            pipe_prompt_attention_mask,
            pipe_negative_prompt_embeds,
            pipe_negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            PROMPT,
            NEGATIVE_PROMPT,
            do_classifier_free_guidance=False,
        )

    print(f"  Pipeline encode_prompt: embeds={pipe_prompt_embeds.shape}, mask={pipe_prompt_attention_mask.shape}")

    # Run connectors (this is what the pipeline does internally in __call__)
    pipe.connectors.to(device)
    with torch.inference_mode():
        additive_mask = (1 - pipe_prompt_attention_mask.to(pipe_prompt_embeds.dtype)) * -1000000.0
        pipe_video_embeds, pipe_audio_embeds, pipe_connector_mask = pipe.connectors(
            pipe_prompt_embeds,
            additive_mask,
            additive_mask=True,
        )

    pipe_video_embeds_cpu = pipe_video_embeds.cpu()
    pipe_audio_embeds_cpu = pipe_audio_embeds.cpu()
    pipe_connector_mask_cpu = pipe_connector_mask.cpu()

    # Save pre-connector pipeline embeds for reference
    _save_tensor("pipe_raw_prompt_embeds", pipe_prompt_embeds.cpu())
    _save_tensor("pipe_raw_attention_mask", pipe_prompt_attention_mask.cpu())

    print(f"  Pipeline post-connectors: video={pipe_video_embeds.shape}, audio={pipe_audio_embeds.shape}")

    # Move all components back to CPU before reusing in node
    pipe.connectors.to("cpu")
    del pipe
    _full_cleanup()

    # --- Node side: PromptEncodeNode (reuse same model instances) ---
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_prompt_encode_node import LTX2PromptEncodeNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"

    mm.register_component("text_encoder", components["text_encoder"])
    mm.register_component("connectors", components["connectors"])
    mm.apply_offload_strategy(device)

    node = LTX2PromptEncodeNode()
    node.device = device

    source = _ValueNode(
        tokenizer=components["tokenizer"],
        text_encoder=components["text_encoder"],
        connectors=components["connectors"],
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=1.0,
    )

    for inp in ["tokenizer", "text_encoder", "connectors", "prompt"]:
        _connect_node(node, inp, source, inp)
    _connect_node(node, "negative_prompt", source, "negative_prompt")
    _connect_node(node, "guidance_scale", source, "guidance_scale")

    with torch.inference_mode():
        node()

    node_video_embeds = node.values["prompt_embeds"]
    node_audio_embeds = node.values["audio_prompt_embeds"]
    node_connector_mask = node.values["attention_mask"]

    print(f"  Node post-connectors: video={node_video_embeds.shape}, audio={node_audio_embeds.shape}")

    # --- Compare ---
    assert pipe_video_embeds_cpu.shape == node_video_embeds.shape, (
        f"Video embeds shape mismatch: pipe={pipe_video_embeds_cpu.shape} vs node={node_video_embeds.shape}"
    )
    assert pipe_audio_embeds_cpu.shape == node_audio_embeds.shape, (
        f"Audio embeds shape mismatch: pipe={pipe_audio_embeds_cpu.shape} vs node={node_audio_embeds.shape}"
    )

    video_diff = (pipe_video_embeds_cpu.float() - node_video_embeds.float()).abs()
    audio_diff = (pipe_audio_embeds_cpu.float() - node_audio_embeds.float()).abs()
    mask_diff = (pipe_connector_mask_cpu.float() - node_connector_mask.float()).abs()

    print(f"  Video embeds: max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}")
    print(f"  Audio embeds: max_diff={audio_diff.max():.8f}, mean_diff={audio_diff.mean():.8f}")
    print(f"  Mask: max_diff={mask_diff.max():.8f}")

    assert video_diff.max() == 0, f"Video embeds differ: max_diff={video_diff.max():.8f}"
    assert audio_diff.max() == 0, f"Audio embeds differ: max_diff={audio_diff.max():.8f}"
    assert mask_diff.max() == 0, f"Mask differs: max_diff={mask_diff.max():.8f}"

    # Save intermediates for next stage
    _save_tensor("prompt_embeds", node_video_embeds)
    _save_tensor("audio_prompt_embeds", node_audio_embeds)
    _save_tensor("attention_mask", node_connector_mask)

    print("  PASS: Prompt encoding matches")

    # Cleanup
    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 2: Image Encoding (VAE encode + normalize)
# ---------------------------------------------------------------------------


def _make_fake_image(height: int, width: int, seed: int = 0):
    """Create a deterministic fake RGB image as numpy HWC uint8."""
    import numpy as np

    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (height, width, 3), dtype=np.uint8)


def test_i2v_stage_image_encode(app_db_path):
    """Compare image encoding: pipeline prepare_latents (image path) vs ImageEncodeNode."""
    import warnings

    from diffusers import LTX2ImageToVideoPipeline
    from PIL import Image

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    fake_image_np = _make_fake_image(HEIGHT, WIDTH)
    fake_image_pil = Image.fromarray(fake_image_np)

    # --- Pipeline side: preprocess image + VAE encode ---
    pipe = LTX2ImageToVideoPipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    # The pipeline preprocesses with VaeImageProcessor then encodes inside
    # prepare_latents. We replicate the steps manually to extract the
    # normalized, repeated, packed image latents and the conditioning mask.
    pipe_image = pipe.video_processor.preprocess(fake_image_pil, height=HEIGHT, width=WIDTH)
    pipe_image = pipe_image.to(device=device, dtype=torch.bfloat16)

    vae = components["vae"]
    vae.to(device)

    with torch.inference_mode():
        # Pipeline path: encode → .mode() → cast to float32 → normalize → repeat → blend → pack
        # For this test we compare just the image latents part (before noise blend).
        init_latents = vae.encode(pipe_image[0].unsqueeze(0).unsqueeze(2)).latent_dist.mode()
        init_latents = init_latents.to(torch.float32)
        pipe_norm = pipe._normalize_latents(init_latents, vae.latents_mean, vae.latents_std)

        latent_num_frames = (NUM_FRAMES - 1) // pipe.vae_temporal_compression_ratio + 1
        pipe_repeated = pipe_norm.repeat(1, 1, latent_num_frames, 1, 1)
        pipe_packed = pipe._pack_latents(
            pipe_repeated, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
        )

    pipe_packed_cpu = pipe_packed.cpu()
    print(f"  Pipeline image latents (packed): {pipe_packed_cpu.shape}, dtype={pipe_packed_cpu.dtype}")

    # Also get the conditioning mask
    latent_height = HEIGHT // pipe.vae_spatial_compression_ratio
    latent_width = WIDTH // pipe.vae_spatial_compression_ratio
    mask_shape = (1, 1, latent_num_frames, latent_height, latent_width)
    pipe_cond_mask = torch.zeros(mask_shape, device=device, dtype=torch.float32)
    pipe_cond_mask[:, :, 0] = 1.0
    pipe_cond_mask_packed = pipe._pack_latents(
        pipe_cond_mask, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
    ).squeeze(-1)
    pipe_cond_mask_cpu = pipe_cond_mask_packed.cpu()

    vae.to("cpu")
    del pipe
    _full_cleanup()

    # --- Node side: ImageEncodeNode ---
    from frameartisan.modules.generation.graph.nodes.ltx2_image_encode_node import LTX2ImageEncodeNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("vae", vae)
    mm.apply_offload_strategy(device)

    node = LTX2ImageEncodeNode()
    node.device = device
    source = _ValueNode(
        vae=vae,
        image=fake_image_np,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
    )
    for inp in ["vae", "image", "num_frames", "height", "width"]:
        _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_packed = node.values["image_latents"].cpu()
    node_mask = node.values["conditioning_mask"].cpu()

    print(f"  Node image latents (packed): {node_packed.shape}, dtype={node_packed.dtype}")

    # --- Compare ---
    assert pipe_packed_cpu.shape == node_packed.shape, (
        f"Shape mismatch: pipe={pipe_packed_cpu.shape} vs node={node_packed.shape}"
    )

    latent_diff = (pipe_packed_cpu.float() - node_packed.float()).abs()
    mask_diff = (pipe_cond_mask_cpu.float() - node_mask.float()).abs()

    print(f"  Image latents: max_diff={latent_diff.max():.8f}, mean_diff={latent_diff.mean():.8f}")
    print(f"  Cond mask: max_diff={mask_diff.max():.8f}")

    assert latent_diff.max() == 0, f"Image latents differ: max_diff={latent_diff.max():.8f}"
    assert mask_diff.max() == 0, f"Cond mask differs: max_diff={mask_diff.max():.8f}"

    # Save for next test
    _save_tensor("image_latents", node_packed)
    _save_tensor("conditioning_mask", node_mask)

    print("  PASS: Image encoding matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 3: Prepare Latents (noise generation + image/noise blend)
# ---------------------------------------------------------------------------


def test_i2v_stage_prepare_latents(app_db_path):
    """Compare latent preparation: pipeline prepare_latents vs LatentsNode."""
    import warnings

    from diffusers import LTX2ImageToVideoPipeline
    from PIL import Image

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    fake_image_np = _make_fake_image(HEIGHT, WIDTH)
    fake_image_pil = Image.fromarray(fake_image_np)

    # --- Pipeline side: full prepare_latents + prepare_audio_latents ---
    pipe = LTX2ImageToVideoPipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    pipe_image = pipe.video_processor.preprocess(fake_image_pil, height=HEIGHT, width=WIDTH)
    pipe_image = pipe_image.to(device=device, dtype=torch.bfloat16)

    generator = torch.Generator("cuda").manual_seed(SEED)
    num_channels_latents = pipe.transformer.config.in_channels

    with torch.inference_mode():
        pipe_latents, pipe_cond_mask = pipe.prepare_latents(
            pipe_image,
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
    pipe_cond_mask_cpu = pipe_cond_mask.cpu()
    pipe_audio_cpu = pipe_audio_latents.cpu()

    print(
        f"  Pipeline: video={pipe_latents_cpu.shape}, cond_mask={pipe_cond_mask_cpu.shape}, audio={pipe_audio_cpu.shape}"
    )

    del pipe
    _full_cleanup()

    # --- Node side: ImageEncodeNode + LatentsNode ---
    from frameartisan.modules.generation.graph.nodes.ltx2_image_encode_node import LTX2ImageEncodeNode
    from frameartisan.modules.generation.graph.nodes.ltx2_latents_node import LTX2LatentsNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("vae", components["vae"])
    mm.apply_offload_strategy(device)

    # Image encode
    ie_node = LTX2ImageEncodeNode()
    ie_node.device = device
    ie_source = _ValueNode(
        vae=components["vae"],
        image=fake_image_np,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
    )
    for inp in ["vae", "image", "num_frames", "height", "width"]:
        _connect_node(ie_node, inp, ie_source, inp)

    with torch.inference_mode():
        ie_node()

    # Latents
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
    _connect_node(lat_node, "image_latents", ie_node, "image_latents")
    _connect_node(lat_node, "conditioning_mask", ie_node, "conditioning_mask")

    with torch.inference_mode():
        lat_node()

    node_video = lat_node.values["video_latents"].cpu()
    node_audio = lat_node.values["audio_latents"].cpu()
    node_mask = lat_node.values["conditioning_mask"].cpu()

    print(f"  Node: video={node_video.shape}, audio={node_audio.shape}, cond_mask={node_mask.shape}")

    # --- Compare ---
    video_diff = (pipe_latents_cpu.float() - node_video.float()).abs()
    audio_diff = (pipe_audio_cpu.float() - node_audio.float()).abs()
    mask_diff = (pipe_cond_mask_cpu.float() - node_mask.float()).abs()

    print(f"  Video latents: max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}")
    print(f"  Audio latents: max_diff={audio_diff.max():.8f}, mean_diff={audio_diff.mean():.8f}")
    print(f"  Cond mask: max_diff={mask_diff.max():.8f}")

    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"
    assert mask_diff.max() == 0, f"Cond mask differs: max_diff={mask_diff.max():.8f}"

    # Save for next test
    _save_tensor("prepared_video_latents", node_video)
    _save_tensor("prepared_audio_latents", node_audio)
    _save_tensor("prepared_cond_mask", node_mask)

    print("  PASS: Latent preparation matches")

    del components, ie_node, lat_node
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 4: First-stage Denoise (transformer forward + scheduler step)
# ---------------------------------------------------------------------------


def test_i2v_stage_denoise(app_db_path):
    """Compare first-stage denoising: manual pipeline loop vs DenoiseNode."""
    import copy
    import warnings

    from PIL import Image

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
        calculate_shift,
        pack_latents,
        unpack_latents,
    )

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    fake_image_np = _make_fake_image(HEIGHT, WIDTH)
    fake_image_pil = Image.fromarray(fake_image_np)

    # --- Prepare inputs using pipeline (verified identical in tests 1-3) ---
    from diffusers import LTX2ImageToVideoPipeline

    pipe = LTX2ImageToVideoPipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    # Encode prompt (no CFG for distilled)
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

    # Prepare latents (image encode + noise blend)
    pipe_image = pipe.video_processor.preprocess(fake_image_pil, height=HEIGHT, width=WIDTH)
    pipe_image = pipe_image.to(device=device, dtype=torch.bfloat16)

    generator = torch.Generator("cuda").manual_seed(SEED)
    num_channels_latents = pipe.transformer.config.in_channels

    with torch.inference_mode():
        video_latents, conditioning_mask = pipe.prepare_latents(
            pipe_image,
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
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    num_mel_bins = pipe.audio_vae.config.mel_bins
    num_channels_audio = pipe.audio_vae.config.latent_channels

    with torch.inference_mode():
        audio_latents = pipe.prepare_audio_latents(
            1,
            num_channels_audio,
            audio_num_frames,
            num_mel_bins,
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

    # Clone shared inputs so pipeline and node start from exactly the same state
    shared_video_latents = video_latents.clone()
    shared_audio_latents = audio_latents.clone()
    shared_conditioning_mask = conditioning_mask.clone()

    print(f"  Inputs: video={video_latents.shape}, audio={audio_latents.shape}, cond_mask={conditioning_mask.shape}")
    print(f"  Latent dims: F={latent_num_frames}, H={latent_height}, W={latent_width}, seq={video_sequence_length}")

    del pipe
    _full_cleanup()

    # --- Pipeline side: manual denoising loop ---
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

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

    patch_size = getattr(transformer.config, "patch_size", 1)
    patch_size_t = getattr(transformer.config, "patch_size_t", 1)

    pipe_video = shared_video_latents.clone()
    pipe_audio = shared_audio_latents.clone()

    print(f"  Running pipeline-style loop: {len(timesteps)} steps...")

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
            video_timestep = timestep.unsqueeze(-1) * (1 - shared_conditioning_mask)

            with transformer.cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = transformer(
                    hidden_states=latent_model_input,
                    audio_hidden_states=audio_model_input,
                    encoder_hidden_states=video_embeds,
                    audio_encoder_hidden_states=audio_embeds,
                    timestep=video_timestep,
                    audio_timestep=timestep,
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

            # Unpack, separate frame 0, step on frames 1+, repack
            noise_pred_video = unpack_latents(
                noise_pred_video, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
            )
            pipe_video = unpack_latents(
                pipe_video, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
            )

            noise_pred_frames = noise_pred_video[:, :, 1:]
            noise_latents = pipe_video[:, :, 1:]
            pred_latents = scheduler.step(noise_pred_frames, t, noise_latents, return_dict=False)[0]

            pipe_video = torch.cat([pipe_video[:, :, :1], pred_latents], dim=2)
            pipe_video = pack_latents(pipe_video, patch_size, patch_size_t)

            pipe_audio = audio_scheduler.step(noise_pred_audio, t, pipe_audio, return_dict=False)[0]

    pipe_video_cpu = pipe_video.cpu()
    pipe_audio_cpu = pipe_audio.cpu()

    print(f"  Pipeline done: video={pipe_video_cpu.shape}, audio={pipe_audio_cpu.shape}")

    # Free transformer from GPU
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()

    # --- Node side: DenoiseNode ---
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
        conditioning_mask=shared_conditioning_mask.clone(),
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

    # Save for next test
    _save_tensor("denoised_video_latents", node_video)
    _save_tensor("denoised_audio_latents", node_audio)

    print("  PASS: First-stage denoise matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 5: Latent Upsampling
# ---------------------------------------------------------------------------


def test_i2v_stage_upsample(app_db_path):
    """Compare latent upsampling: pipeline LTX2LatentUpsamplePipeline vs LatentUpsampleNode."""
    import warnings

    from diffusers import AutoencoderKLLTX2Video, LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    upsampler_path = _find_upsampler_path()
    if upsampler_path is None:
        pytest.skip("Latent upsampler not found")

    _full_cleanup()
    device = torch.device("cuda")

    model_path = model["filepath"]
    paths = _resolve_paths(model["id"], model_path)

    def _p(comp_type):
        return paths.get(comp_type, os.path.join(model_path, comp_type))

    # Load shared components
    vae = AutoencoderKLLTX2Video.from_pretrained(_p("vae"), torch_dtype=torch.bfloat16)
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(upsampler_path, torch_dtype=torch.bfloat16)

    # Create fake denoised latents (packed, normalized, float32).
    # In the real graph, the denoise node outputs normalized packed latents.
    # The upsample node denormalizes before upsampling (matching the pipeline,
    # where output_type="latent" returns denormalized 5D and the upsampler
    # receives them already denormalized).
    # We tell the pipeline latents_normalized=True so both sides denormalize.
    latent_num_frames = (NUM_FRAMES - 1) // vae.temporal_compression_ratio + 1
    latent_height = HEIGHT // vae.spatial_compression_ratio
    latent_width = WIDTH // vae.spatial_compression_ratio
    seq_len = latent_num_frames * latent_height * latent_width

    gen = torch.Generator("cpu").manual_seed(SEED)
    fake_latents = torch.randn(1, seq_len, 128, generator=gen, dtype=torch.float32)

    print(f"\n  Fake input latents: {fake_latents.shape}")
    print(f"  Latent dims: F={latent_num_frames}, H={latent_height}, W={latent_width}")

    # --- Pipeline side ---
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler)
    upsample_pipe.enable_model_cpu_offload(device=device)

    with torch.inference_mode():
        pipe_result = upsample_pipe(
            latents=fake_latents.clone(),
            latents_normalized=True,
            output_type="latent",
            return_dict=False,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
        )[0]

    pipe_result_cpu = pipe_result.cpu()
    print(f"  Pipeline result: {pipe_result_cpu.shape}, dtype={pipe_result_cpu.dtype}")

    del upsample_pipe
    _full_cleanup()

    # --- Node side ---
    from frameartisan.modules.generation.graph.nodes.ltx2_latent_upsample_node import LTX2LatentUpsampleNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.apply_offload_strategy(device)

    node = LTX2LatentUpsampleNode(upsampler_model_path=upsampler_path)
    node.device = device

    source = _ValueNode(
        video_latents=fake_latents.clone(),
        vae=vae,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    for inp in LTX2LatentUpsampleNode.REQUIRED_INPUTS:
        _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_result = node.values["video_latents"].cpu()
    print(f"  Node result: {node_result.shape}, dtype={node_result.dtype}")

    # Pipeline returns [B,C,F,H*2,W*2] unpacked; node returns [B,seq,feat] packed.
    # Pack pipeline result for comparison if it's 5D.
    if pipe_result_cpu.ndim == 5:
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents

        pipe_result_cpu = pack_latents(pipe_result_cpu, patch_size=1, patch_size_t=1)
        print(f"  Pipeline packed: {pipe_result_cpu.shape}")

    # --- Compare ---
    assert pipe_result_cpu.shape == node_result.shape, (
        f"Shape mismatch: pipe={pipe_result_cpu.shape} vs node={node_result.shape}"
    )

    diff = (pipe_result_cpu.float() - node_result.float()).abs()
    print(f"  Upsample diff: max={diff.max():.8f}, mean={diff.mean():.8f}")

    assert diff.max() == 0, f"Upsample differs: max_diff={diff.max():.8f}"

    _save_tensor("upsampled_video_latents", node_result)

    print("  PASS: Latent upsampling matches")

    del node, source, vae, latent_upsampler
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 6: Second Pass Latent Preparation (noise mixing for stage 2)
# ---------------------------------------------------------------------------


def test_i2v_stage_second_pass_latents(app_db_path):
    """Compare stage-2 latent preparation: pipeline prepare_latents/prepare_audio_latents vs SecondPassLatentsNode."""
    import warnings

    from diffusers import LTX2ImageToVideoPipeline
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
    from PIL import Image

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    fake_image_np = _make_fake_image(HEIGHT, WIDTH)
    fake_image_pil = Image.fromarray(fake_image_np)

    # Stage 2 operates at 2x spatial resolution
    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2

    vae = components["vae"]
    vae_temporal_ratio = getattr(vae, "temporal_compression_ratio", 8)
    vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 32)
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = stage2_height // vae_spatial_ratio  # 2x
    latent_width = stage2_width // vae_spatial_ratio  # 2x
    video_seq_len = latent_num_frames * latent_height * latent_width

    # Create fake upsampled video latents in denormalized 5D space (matching
    # the real flow: upsampler outputs denormalized [B, C, F, H, W]).
    # The pipeline's prepare_latents normalizes 5D latents, and our node also
    # normalizes the packed 3D equivalent.
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents

    gen_fake = torch.Generator("cpu").manual_seed(99)
    fake_video_latents_5d = torch.randn(
        1,
        128,
        latent_num_frames,
        latent_height,
        latent_width,
        generator=gen_fake,
        dtype=torch.float32,
    )
    # Packed version for the node (which receives packed 3D from upsample node)
    fake_video_latents = pack_latents(fake_video_latents_5d, patch_size=1, patch_size_t=1)

    # Create fake audio latents from stage 1 (packed 3D)
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
    pipe = LTX2ImageToVideoPipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    num_channels_latents = pipe.transformer.config.in_channels

    generator = torch.Generator("cuda").manual_seed(SEED)

    with torch.inference_mode():
        pipe_video, pipe_cond_mask = pipe.prepare_latents(
            pipe.video_processor.preprocess(fake_image_pil, height=stage2_height, width=stage2_width).to(
                device=device, dtype=torch.bfloat16
            ),
            1,
            num_channels_latents,
            stage2_height,
            stage2_width,
            NUM_FRAMES,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents=fake_video_latents_5d.clone().to(device),
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
    pipe_cond_mask_cpu = pipe_cond_mask.cpu()

    print(f"  Pipeline: video={pipe_video_cpu.shape}, audio={pipe_audio_cpu.shape}, mask={pipe_cond_mask_cpu.shape}")

    del pipe
    _full_cleanup()

    # --- Node side ---
    from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import LTX2SecondPassLatentsNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.apply_offload_strategy(device)

    transformer = components["transformer"]
    audio_coords = transformer.audio_rope.prepare_audio_coords(1, audio_num_frames, device)

    node = LTX2SecondPassLatentsNode()
    node.device = device

    # Pass a dummy conditioning_mask to signal I2V mode (the actual mask is created
    # fresh inside the node at stage 2 latent dimensions).
    dummy_cond_mask = torch.ones(1, video_seq_len, device=device)

    source = _ValueNode(
        transformer=transformer,
        vae=components["vae"],
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
        conditioning_mask=dummy_cond_mask,
    )
    for inp in LTX2SecondPassLatentsNode.REQUIRED_INPUTS + LTX2SecondPassLatentsNode.OPTIONAL_INPUTS:
        if inp in source.values:
            _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_video = node.values["video_latents"].cpu()
    node_audio = node.values["audio_latents"].cpu()

    print(f"  Node: video={node_video.shape}, audio={node_audio.shape}")

    # --- Compare ---
    video_diff = (pipe_video_cpu.float() - node_video.float()).abs()
    audio_diff = (pipe_audio_cpu.float() - node_audio.float()).abs()

    print(f"  Video latents: max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}")
    print(f"  Audio latents: max_diff={audio_diff.max():.8f}, mean_diff={audio_diff.mean():.8f}")

    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"

    print("  PASS: Second pass latent preparation matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 7: Second-stage Denoise
# ---------------------------------------------------------------------------


def test_i2v_stage_second_denoise(app_db_path):
    """Compare second-stage denoising: manual pipeline loop vs DenoiseNode (stage=2)."""
    import copy
    import warnings

    from PIL import Image

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
        calculate_shift,
        pack_latents,
        unpack_latents,
    )

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    fake_image_np = _make_fake_image(HEIGHT, WIDTH)
    fake_image_pil = Image.fromarray(fake_image_np)

    # Stage 2 operates at 2x spatial resolution
    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2

    # --- Prepare shared inputs ---
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ImageToVideoPipeline
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

    pipe = LTX2ImageToVideoPipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    # Encode prompt (no CFG for distilled)
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

    # Prepare latents for stage 2 (using pipeline)
    pipe_image = pipe.video_processor.preprocess(fake_image_pil, height=stage2_height, width=stage2_width)
    pipe_image = pipe_image.to(device=device, dtype=torch.bfloat16)

    noise_scale = STAGE_2_DISTILLED_SIGMA_VALUES[0]
    num_channels_latents = pipe.transformer.config.in_channels
    generator = torch.Generator("cuda").manual_seed(SEED)

    # Create fake upsampled latents
    vae_temporal_ratio = pipe.vae_temporal_compression_ratio
    vae_spatial_ratio = pipe.vae_spatial_compression_ratio
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = stage2_height // vae_spatial_ratio
    latent_width = stage2_width // vae_spatial_ratio
    video_seq_len = latent_num_frames * latent_height * latent_width

    gen_fake = torch.Generator("cpu").manual_seed(99)
    fake_video_latents = torch.randn(1, video_seq_len, 128, generator=gen_fake, dtype=torch.float32).to(device)

    audio_vae = components["audio_vae"]
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    num_mel_bins = audio_vae.config.mel_bins
    latent_mel_bins = num_mel_bins // pipe.audio_vae_mel_compression_ratio
    audio_feat = audio_vae.config.latent_channels * latent_mel_bins
    fake_audio_latents = torch.randn(1, audio_num_frames, audio_feat, generator=gen_fake, dtype=torch.float32).to(
        device
    )

    with torch.inference_mode():
        video_latents, conditioning_mask = pipe.prepare_latents(
            pipe_image,
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
            num_mel_bins,
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

    # Clone shared inputs
    shared_video_latents = video_latents.clone()
    shared_audio_latents = audio_latents.clone()
    shared_conditioning_mask = conditioning_mask.clone()

    print(f"  Inputs: video={video_latents.shape}, audio={audio_latents.shape}, cond_mask={conditioning_mask.shape}")
    print(f"  Latent dims: F={latent_num_frames}, H={latent_height}, W={latent_width}")

    del pipe
    _full_cleanup()

    # --- Pipeline side: manual stage 2 denoising loop ---
    sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
    num_steps = 3

    # Stage 2 uses modified scheduler config
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

    patch_size = getattr(transformer.config, "patch_size", 1)
    patch_size_t = getattr(transformer.config, "patch_size_t", 1)

    pipe_video = shared_video_latents.clone()
    pipe_audio = shared_audio_latents.clone()

    print(f"  Running pipeline-style stage 2 loop: {len(timesteps)} steps...")

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
            video_timestep = timestep.unsqueeze(-1) * (1 - shared_conditioning_mask)

            with transformer.cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = transformer(
                    hidden_states=latent_model_input,
                    audio_hidden_states=audio_model_input,
                    encoder_hidden_states=video_embeds,
                    audio_encoder_hidden_states=audio_embeds,
                    timestep=video_timestep,
                    audio_timestep=timestep,
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

            # Unpack, separate frame 0, step on frames 1+, repack
            noise_pred_video = unpack_latents(
                noise_pred_video, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
            )
            pipe_video = unpack_latents(
                pipe_video, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
            )

            noise_pred_frames = noise_pred_video[:, :, 1:]
            noise_latents = pipe_video[:, :, 1:]
            pred_latents = scheduler.step(noise_pred_frames, t, noise_latents, return_dict=False)[0]

            pipe_video = torch.cat([pipe_video[:, :, :1], pred_latents], dim=2)
            pipe_video = pack_latents(pipe_video, patch_size, patch_size_t)

            pipe_audio = audio_scheduler.step(noise_pred_audio, t, pipe_audio, return_dict=False)[0]

    pipe_video_cpu = pipe_video.cpu()
    pipe_audio_cpu = pipe_audio.cpu()

    print(f"  Pipeline done: video={pipe_video_cpu.shape}, audio={pipe_audio_cpu.shape}")

    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()

    # --- Node side: DenoiseNode with stage=2 ---
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
        conditioning_mask=shared_conditioning_mask.clone(),
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

    print("  PASS: Second-stage denoise matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 8: Decode (VAE decode + postprocess)
# ---------------------------------------------------------------------------


def test_i2v_stage_decode(app_db_path):
    """Compare decode: pipeline unpack+denorm+VAE decode vs DecodeNode at stage 2 resolution."""
    import warnings

    import numpy as np

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
        denormalize_latents,
        unpack_latents,
    )

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    # Stage 2 operates at 2x spatial resolution
    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2

    vae = components["vae"]
    audio_vae = components["audio_vae"]
    vocoder = components["vocoder"]

    vae_temporal_ratio = getattr(vae, "temporal_compression_ratio", 8)
    vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 32)
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = stage2_height // vae_spatial_ratio
    latent_width = stage2_width // vae_spatial_ratio
    video_seq_len = latent_num_frames * latent_height * latent_width

    # Audio dims
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        audio_vae.config.sample_rate
        / audio_vae.config.mel_hop_length
        / float(getattr(audio_vae, "temporal_compression_ratio", 4))
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    num_mel_bins = audio_vae.config.mel_bins
    latent_mel_bins = num_mel_bins // getattr(audio_vae, "mel_compression_ratio", 4)
    audio_feat = audio_vae.config.latent_channels * latent_mel_bins

    # Create deterministic fake denoised latents (packed 3D, float32)
    gen_fake = torch.Generator("cpu").manual_seed(123)
    fake_video_latents = torch.randn(1, video_seq_len, 128, generator=gen_fake, dtype=torch.float32)
    fake_audio_latents = torch.randn(1, audio_num_frames, audio_feat, generator=gen_fake, dtype=torch.float32)

    print(f"  Fake denoised: video={fake_video_latents.shape}, audio={fake_audio_latents.shape}")
    print(f"  Latent dims: F={latent_num_frames}, H={latent_height}, W={latent_width}")

    # --- Pipeline side: manual decode ---
    video_latents_5d = unpack_latents(
        fake_video_latents.clone().to(device),
        latent_num_frames,
        latent_height,
        latent_width,
        patch_size=1,
        patch_size_t=1,
    )
    video_latents_5d = denormalize_latents(
        video_latents_5d,
        vae.latents_mean,
        vae.latents_std,
        vae.config.scaling_factor,
    )

    timestep = None
    if getattr(vae.config, "timestep_conditioning", False):
        timestep = torch.tensor([0.0], device=device, dtype=video_latents_5d.dtype)

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("vae", vae)
    mm.register_component("audio_vae", audio_vae)
    mm.register_component("vocoder", vocoder)
    mm.apply_offload_strategy(device)

    with torch.inference_mode(), mm.use_components("vae", "audio_vae", "vocoder", device=device):
        video_latents_5d = video_latents_5d.to(device=device, dtype=vae.dtype)
        pipe_video = vae.decode(video_latents_5d, timestep, return_dict=False)[0]

    # Post-process to match graph's decode node
    pipe_video = (pipe_video.float() * 0.5 + 0.5).clamp(0, 1)
    pipe_video = pipe_video[0].permute(1, 0, 2, 3)  # [C, F, H, W] → [F, C, H, W]
    pipe_video = pipe_video.permute(0, 2, 3, 1)  # [F, C, H, W] → [F, H, W, C]
    pipe_video_np = (pipe_video.cpu().numpy() * 255).round().astype(np.uint8)

    print(f"  Pipeline decoded video: {pipe_video_np.shape}, dtype={pipe_video_np.dtype}")

    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()

    # --- Node side: DecodeNode ---
    from frameartisan.modules.generation.graph.nodes.ltx2_decode_node import LTX2DecodeNode

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"
    mm.register_component("vae", vae)
    mm.register_component("audio_vae", audio_vae)
    mm.register_component("vocoder", vocoder)
    mm.apply_offload_strategy(device)

    node = LTX2DecodeNode()
    node.device = device

    source = _ValueNode(
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        video_latents=fake_video_latents.clone().to(device),
        audio_latents=fake_audio_latents.clone().to(device),
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_num_frames=audio_num_frames,
        num_frames=NUM_FRAMES,
        frame_rate=float(FRAME_RATE),
    )
    for inp in LTX2DecodeNode.REQUIRED_INPUTS:
        _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_video_np = node.values["video"]

    print(f"  Node decoded video: {node_video_np.shape}, dtype={node_video_np.dtype}")

    # --- Compare ---
    assert pipe_video_np.shape == node_video_np.shape, (
        f"Shape mismatch: pipe={pipe_video_np.shape} vs node={node_video_np.shape}"
    )

    diff = np.abs(pipe_video_np.astype(np.int16) - node_video_np.astype(np.int16))
    max_diff = diff.max()
    mean_diff = diff.astype(np.float32).mean()

    print(f"  Video decode: max_diff={max_diff}, mean_diff={mean_diff:.4f}")

    assert max_diff == 0, f"Decoded video differs: max_diff={max_diff}, mean_diff={mean_diff:.4f}"

    print("  PASS: Decode matches")

    del components, node, source
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()


# ---------------------------------------------------------------------------
# Test 9: Full end-to-end I2V 2-stage (pipeline vs graph)
# ---------------------------------------------------------------------------


def test_i2v_full_e2e_two_stage(app_db_path):
    """Run the full I2V 2-stage pipeline and graph end-to-end and compare decoded video.

    Both sides share the same loaded components so precision differences from
    independent model loads don't obscure real divergences.
    """
    import warnings

    import attr
    import numpy as np
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        LTX2ImageToVideoPipeline,
        LTX2LatentUpsamplePipeline,
    )
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
    from PIL import Image

    from frameartisan.app.app import set_app_database_path
    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings
    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    set_app_database_path(app_db_path)
    model = _discover_model_by_version(app_db_path, "8bit")
    if model is None:
        pytest.skip("LTX2 Distilled 8-bit not found in DB")

    upsampler_path = _find_upsampler_path()
    if upsampler_path is None:
        pytest.skip("Latent upsampler not found")

    _full_cleanup()
    device = torch.device("cuda")

    print(f"\n  Loading components for {model['name']}...")
    components = _load_components(model)

    fake_image_np = _make_fake_image(HEIGHT, WIDTH)
    fake_image_pil = Image.fromarray(fake_image_np)

    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2

    # --- Pipeline side: full 2-stage I2V ---
    pipe = LTX2ImageToVideoPipeline(**components)
    pipe.enable_model_cpu_offload(device=device)

    # Stage 1
    generator = torch.Generator("cuda").manual_seed(SEED)
    with torch.inference_mode():
        video_latent, audio_latent = pipe(
            image=fake_image_pil,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=FRAME_RATE,
            num_inference_steps=8,
            guidance_scale=1.0,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )
    scheduler_config = pipe.scheduler.config

    # Upsample
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(upsampler_path, torch_dtype=torch.bfloat16)
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
    upsample_pipe.enable_model_cpu_offload(device=device)

    with torch.inference_mode():
        upscaled = upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
        )[0]

    del upsample_pipe, latent_upsampler
    _full_cleanup()

    # Stage 2 with image conditioning
    stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        scheduler_config, use_dynamic_shifting=False, shift_terminal=None
    )
    pipe.scheduler = stage2_scheduler

    generator_s2 = torch.Generator("cuda").manual_seed(SEED)
    with torch.inference_mode():
        pipe_result = pipe(
            image=fake_image_pil,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            latents=upscaled,
            audio_latents=audio_latent,
            height=stage2_height,
            width=stage2_width,
            num_frames=NUM_FRAMES,
            frame_rate=FRAME_RATE,
            num_inference_steps=3,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            guidance_scale=1.0,
            generator=generator_s2,
            output_type="np",
            return_dict=False,
        )

    pipe_video_np = pipe_result[0]  # [B, F, H, W, C]
    pipe_video_np = (pipe_video_np[0] * 255).round().astype(np.uint8)  # [F, H, W, C]

    # Remove pipeline hooks so components can be reused by the graph
    pipe.remove_all_hooks()
    del pipe
    _full_cleanup()

    # --- Graph side: full 2-stage I2V using the SAME components ---
    @attr.s
    class _FakeDirs:
        outputs_videos: str = attr.ib(default="")

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "model_offload"

    model_obj = ModelDataObject(
        name=model["name"],
        filepath=model["filepath"],
        model_type=model["model_type"],
        id=model["id"],
    )
    gen_settings = GenerationSettings(
        model=model_obj,
        second_pass_model=model_obj,
        video_width=WIDTH,
        video_height=HEIGHT,
        video_duration=NUM_FRAMES / FRAME_RATE,
        frame_rate=FRAME_RATE,
        num_inference_steps=8,
        guidance_scale=1.0,
        second_pass_enabled=True,
        second_pass_steps=3,
        second_pass_guidance=1.0,
        use_torch_compile=False,
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dirs = _FakeDirs(outputs_videos=tmpdir)
        graph = create_default_ltx2_graph(gen_settings, dirs)

        graph.get_node_by_name("prompt").text = PROMPT
        graph.get_node_by_name("seed").number = SEED

        # Enable source image
        source_node = graph.get_node_by_name("source_image")
        source_node.enabled = True
        source_node.image = fake_image_np
        source_node.set_updated()

        encode_node = graph.get_node_by_name("image_encode")
        encode_node.enabled = True
        encode_node.set_updated()

        # Enable 2nd pass nodes
        for name in (
            "upsample",
            "second_pass_model",
            "second_pass_lora",
            "second_pass_latents",
            "second_pass_denoise",
        ):
            node = graph.get_node_by_name(name)
            node.enabled = True
            node.set_updated()

        # Set upsampler path
        graph.get_node_by_name("upsample").upsampler_model_path = upsampler_path

        # Pre-populate model nodes with the shared components so the graph
        # uses exactly the same weights as the pipeline (no second load).
        base_component_values = {
            "tokenizer": components["tokenizer"],
            "text_encoder": components["text_encoder"],
            "transformer": components["transformer"],
            "vae": components["vae"],
            "audio_vae": components["audio_vae"],
            "connectors": components["connectors"],
            "vocoder": components["vocoder"],
            "scheduler_config": scheduler_config,
            "transformer_component_name": "transformer",
        }
        # Primary model node
        model_node = graph.get_node_by_name("model")
        model_node.values = dict(base_component_values)
        model_node.updated = False

        # Second pass model node — uses "sp_" prefix for transformer registration
        sp_model_node = graph.get_node_by_name("second_pass_model")
        sp_model_node.values = dict(base_component_values)
        sp_model_node.values["transformer_component_name"] = "sp_transformer"
        sp_model_node.updated = False

        # Pre-populate lora nodes (pass-through when no LoRAs configured)
        lora_node = graph.get_node_by_name("lora")
        lora_node.values = {
            "transformer": components["transformer"],
            "transformer_component_name": "transformer",
        }
        lora_node.updated = False

        sp_lora_node = graph.get_node_by_name("second_pass_lora")
        sp_lora_node.values = {
            "transformer": components["transformer"],
            "transformer_component_name": "sp_transformer",
        }
        sp_lora_node.updated = False

        # Register components with ModelManager for offload
        mm.register_component("text_encoder", components["text_encoder"])
        mm.register_component("transformer", components["transformer"])
        mm.register_component("sp_transformer", components["transformer"])
        mm.register_component("vae", components["vae"])
        mm.register_component("audio_vae", components["audio_vae"])
        mm.register_component("connectors", components["connectors"])
        mm.register_component("vocoder", components["vocoder"])
        mm.apply_offload_strategy(device)

        # Enable VAE tiling (normally done by model node)
        components["vae"].enable_tiling()

        # Rewire decode to read from 2nd pass
        decode_node = graph.get_node_by_name("decode")
        denoise_node = graph.get_node_by_name("denoise")
        latents_node = graph.get_node_by_name("latents")
        sp_denoise = graph.get_node_by_name("second_pass_denoise")
        sp_latents = graph.get_node_by_name("second_pass_latents")

        decode_node.disconnect("video_latents", denoise_node, "video_latents")
        decode_node.disconnect("audio_latents", denoise_node, "audio_latents")
        decode_node.disconnect("latent_num_frames", latents_node, "latent_num_frames")
        decode_node.disconnect("latent_height", latents_node, "latent_height")
        decode_node.disconnect("latent_width", latents_node, "latent_width")
        decode_node.disconnect("audio_num_frames", latents_node, "audio_num_frames")

        decode_node.connect("video_latents", sp_denoise, "video_latents")
        decode_node.connect("audio_latents", sp_denoise, "audio_latents")
        decode_node.connect("latent_num_frames", sp_latents, "latent_num_frames")
        decode_node.connect("latent_height", sp_latents, "latent_height")
        decode_node.connect("latent_width", sp_latents, "latent_width")
        decode_node.connect("audio_num_frames", sp_latents, "audio_num_frames")

        graph.device = device
        graph.dtype = torch.bfloat16

        video_paths = []
        graph.get_node_by_name("video_send").video_callback = video_paths.append

        graph()

        graph_video_np = graph.get_node_by_name("decode").values["video"]

    # --- Compare ---
    assert pipe_video_np.shape == graph_video_np.shape, (
        f"Shape mismatch: pipe={pipe_video_np.shape} vs graph={graph_video_np.shape}"
    )

    diff = np.abs(pipe_video_np.astype(np.int16) - graph_video_np.astype(np.int16))
    max_diff = diff.max()
    mean_diff = diff.astype(np.float32).mean()

    print(f"  E2E video: max_diff={max_diff}, mean_diff={mean_diff:.4f}")
    print(f"  Per-frame max diff: {[int(diff[f].max()) for f in range(diff.shape[0])]}")

    if max_diff > 1:
        print(f"  WARNING: Large divergence detected (max_diff={max_diff})")

    assert max_diff <= 1, f"E2E I2V 2-stage diverged: max_diff={max_diff}, mean_diff={mean_diff:.4f}"

    print("  PASS: Full E2E I2V 2-stage comparison")

    del graph, components
    mm.clear()
    set_global_model_manager(ModelManager())
    _full_cleanup()
