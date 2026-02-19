"""Step-by-step T2V 2nd pass comparison: pipeline vs graph nodes.

Each test isolates a single stage of the 2nd pass and compares the
numerical output against the equivalent graph node using **shared model
instances** to eliminate any non-determinism from separate model loads.

Activated by setting ``GPU_LIGHT=1``.

    GPU_LIGHT=1 uv run --extra test pytest tests/test_second_pass_stages.py -v -s
"""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.gpu_light

# Shared test parameters (match _SECOND_PASS_PARAMS in test_ltx2_integration)
PROMPT = "a red ball"
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
WIDTH = 256
HEIGHT = 256
NUM_FRAMES = 9
FRAME_RATE = 8
SEED = 42
NUM_STEPS_S1 = 2
NUM_STEPS_S2 = 2
GUIDANCE_SCALE = 1.0  # no_cfg — the failing case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ValueNode:
    """Minimal stand-in for a graph node with .values and .enabled."""

    def __init__(self, **kwargs):
        self.values = kwargs
        self.enabled = True
        self.dependents = []


def _connect_node(node, input_name, source_node, output_name):
    node.connections[input_name] = [(source_node, output_name)]
    if source_node not in node.dependencies:
        node.dependencies.append(source_node)


def _assert_zero_diff(label: str, a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, f"{label} shape mismatch: {a.shape} vs {b.shape}"
    diff = (a.float() - b.float()).abs()
    print(f"  {label}: max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}")
    assert diff.max() == 0, f"{label} differ: max_diff={diff.max():.8f}"


# ---------------------------------------------------------------------------
# Stage 1: Run full stage 1 on both pipeline and graph, return intermediates
# (We need stage 1 outputs as inputs for stages 2+)
# ---------------------------------------------------------------------------


def _run_shared_stage1(pipe, components, device):
    """Run stage 1 via pipeline and return intermediates needed for stage 2."""

    generator = torch.Generator(device).manual_seed(SEED)

    with torch.inference_mode():
        video_latent, audio_latent = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            frame_rate=FRAME_RATE,
            num_inference_steps=NUM_STEPS_S1,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )

    return video_latent, audio_latent


# ---------------------------------------------------------------------------
# Test 1: Upsample — pipeline vs node
# ---------------------------------------------------------------------------


def test_second_pass_upsample(tiny_model_path, tiny_upsampler_path):
    """Compare upsample: LTX2LatentUpsamplePipeline vs LTX2LatentUpsampleNode."""
    from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_latent_upsample_node import LTX2LatentUpsampleNode
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents

    device = torch.device("cpu")

    # Load shared components
    pipe = LTX2Pipeline.from_pretrained(tiny_model_path, torch_dtype=torch.bfloat16).to(device)
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(tiny_upsampler_path, torch_dtype=torch.bfloat16)

    # Run stage 1 to get video_latent
    video_latent, _audio_latent = _run_shared_stage1(pipe, None, device)
    print(f"\n  Stage 1 video_latent: shape={video_latent.shape}, dtype={video_latent.dtype}")

    vae = pipe.vae
    vae_temporal_ratio = pipe.vae_temporal_compression_ratio
    vae_spatial_ratio = pipe.vae_spatial_compression_ratio
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = HEIGHT // vae_spatial_ratio
    latent_width = WIDTH // vae_spatial_ratio

    # --- Pipeline side: upsample ---
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler).to(device)
    pipe_upsampled = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]
    print(f"  Pipeline upsampled: shape={pipe_upsampled.shape}, dtype={pipe_upsampled.dtype}")

    # --- Node side: upsample ---
    # The upsample node expects packed 3D input. The stage 1 pipeline returns
    # latents in whatever format the pipeline uses. We need to match what the
    # graph would receive from stage 1 denoise.
    #
    # In the graph: denoise outputs packed 3D latents → upsample node.
    # In the pipeline: stage 1 returns 5D latents → upsample_pipe.
    #
    # The upsample pipeline handles 5D internally (unpack if needed).
    # The upsample node expects packed 3D.
    #
    # To make this a fair comparison, we simulate what the graph does:
    # 1. Pack the stage 1 output to 3D (same as denoise node output)
    # 2. Feed to upsample node
    #
    # But we also need to check: does the upsample pipeline produce the same
    # result as the upsample node?

    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "no_offload"

    # The node gets packed 3D latents from denoise. The pipeline gives 5D.
    # We need to convert the pipeline's 5D to packed 3D to feed the node.
    if video_latent.ndim == 5:
        node_input = pack_latents(video_latent, patch_size=1, patch_size_t=1)
    else:
        node_input = video_latent

    node = LTX2LatentUpsampleNode(upsampler_model_path=tiny_upsampler_path)
    node.device = device

    source = _ValueNode(
        video_latents=node_input,
        vae=vae,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    for inp in LTX2LatentUpsampleNode.REQUIRED_INPUTS:
        _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_upsampled = node.values["video_latents"]  # packed 3D
    print(f"  Node upsampled: shape={node_upsampled.shape}, dtype={node_upsampled.dtype}")

    # Pipeline output is 5D [B, C, F, H, W]. Pack it for comparison.
    if pipe_upsampled.ndim == 5:
        pipe_upsampled_packed = pack_latents(pipe_upsampled, patch_size=1, patch_size_t=1)
    else:
        pipe_upsampled_packed = pipe_upsampled

    _assert_zero_diff("Upsampled video latents", pipe_upsampled_packed, node_upsampled)

    # Also verify the node reports correct dimensions
    assert node.values["latent_height"] == latent_height * 2
    assert node.values["latent_width"] == latent_width * 2

    print("  PASS: Upsample matches")

    del pipe, upsample_pipe, node
    mm.clear()
    set_global_model_manager(ModelManager())


# ---------------------------------------------------------------------------
# Test 2: Second pass latent preparation — pipeline vs node
# ---------------------------------------------------------------------------


def test_second_pass_latent_prep(tiny_model_path, tiny_upsampler_path):
    """Compare stage-2 latent preparation: pipeline prepare_latents/prepare_audio_latents vs SecondPassLatentsNode."""
    from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import LTX2SecondPassLatentsNode
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents

    device = torch.device("cpu")

    # Load shared components
    pipe = LTX2Pipeline.from_pretrained(tiny_model_path, torch_dtype=torch.bfloat16).to(device)
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(tiny_upsampler_path, torch_dtype=torch.bfloat16)

    vae = pipe.vae
    audio_vae = pipe.audio_vae
    transformer = pipe.transformer
    vae_temporal_ratio = pipe.vae_temporal_compression_ratio
    vae_spatial_ratio = pipe.vae_spatial_compression_ratio

    # Run stage 1
    video_latent, audio_latent = _run_shared_stage1(pipe, None, device)
    print(f"\n  Stage 1: video={video_latent.shape}, audio={audio_latent.shape}")

    # Upsample (shared upsampler)
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler).to(device)
    upscaled_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]
    del upsample_pipe
    print(f"  Upscaled: shape={upscaled_latent.shape}")

    # Stage 2 dimensions
    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height_s2 = stage2_height // vae_spatial_ratio
    latent_width_s2 = stage2_width // vae_spatial_ratio

    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)

    # --- Pipeline side: prepare_latents + prepare_audio_latents ---
    # The pipeline's stage 2 call passes noise_scale=0.0 (default)
    generator_s2 = torch.Generator(device).manual_seed(SEED)
    num_channels_latents = pipe.transformer.config.in_channels

    with torch.inference_mode():
        pipe_video = pipe.prepare_latents(
            1,
            num_channels_latents,
            stage2_height,
            stage2_width,
            NUM_FRAMES,
            0.0,  # noise_scale=0.0 (pipeline default)
            torch.float32,
            device,
            generator_s2,
            upscaled_latent,
        )
        pipe_audio = pipe.prepare_audio_latents(
            1,
            audio_vae.config.latent_channels,
            audio_num_frames,
            audio_vae.config.mel_bins,
            0.0,  # noise_scale=0.0
            torch.float32,
            device,
            generator_s2,
            audio_latent,
        )

    print(f"  Pipeline prepared: video={pipe_video.shape}, audio={pipe_audio.shape}")

    # --- Node side: SecondPassLatentsNode ---
    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "no_offload"

    # The node gets packed 3D video from upsample node (which does its own
    # denormalize + upsample + pack). We need to simulate that.
    # The upsample node: unpack → denormalize → upsample → pack
    # The upsample pipeline: if 5D and latents_normalized=False → just upsamples
    # The key: the upsample pipeline returns 5D DENORMALIZED latents.
    # The upsample node returns packed 3D DENORMALIZED latents.
    # Pack the pipeline's 5D upsampled latents to get the same 3D format.
    if upscaled_latent.ndim == 5:
        node_video_input = pack_latents(upscaled_latent, patch_size=1, patch_size_t=1)
    else:
        node_video_input = upscaled_latent

    # The audio_latent from stage 1 — in the pipeline, this is 5D or 3D.
    # In the graph, denoise outputs packed 3D. We need the same format.
    if audio_latent.ndim == 4:
        # [B, C, L, M] → pack
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_audio_latents

        node_audio_input = pack_audio_latents(audio_latent)
    elif audio_latent.ndim == 3:
        node_audio_input = audio_latent
    else:
        node_audio_input = audio_latent

    audio_coords = transformer.audio_rope.prepare_audio_coords(1, audio_num_frames, device)

    node = LTX2SecondPassLatentsNode()
    node.device = device

    source = _ValueNode(
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        video_latents=node_video_input,
        audio_latents=node_audio_input,
        audio_coords=audio_coords,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height_s2,
        latent_width=latent_width_s2,
        audio_num_frames=audio_num_frames,
        frame_rate=float(FRAME_RATE),
        seed=SEED,
        model_type=0,  # non-distilled tiny model
    )
    for inp in LTX2SecondPassLatentsNode.REQUIRED_INPUTS:
        if inp in source.values:
            _connect_node(node, inp, source, inp)

    with torch.inference_mode():
        node()

    node_video = node.values["video_latents"]
    node_audio = node.values["audio_latents"]

    print(f"  Node prepared: video={node_video.shape}, audio={node_audio.shape}")

    # --- Compare ---
    _assert_zero_diff("Stage 2 video latents", pipe_video, node_video)
    _assert_zero_diff("Stage 2 audio latents", pipe_audio, node_audio)

    print("  PASS: Stage 2 latent preparation matches")

    del pipe, node
    mm.clear()
    set_global_model_manager(ModelManager())


# ---------------------------------------------------------------------------
# Test 3: Second pass denoise — pipeline vs node (shared inputs)
# ---------------------------------------------------------------------------


def test_second_pass_denoise(tiny_model_path, tiny_upsampler_path):
    """Compare stage-2 denoise: manual pipeline loop vs DenoiseNode (shared models + inputs)."""
    import numpy as np

    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import calculate_shift

    device = torch.device("cpu")

    # Load shared components
    pipe = LTX2Pipeline.from_pretrained(tiny_model_path, torch_dtype=torch.bfloat16).to(device)
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(tiny_upsampler_path, torch_dtype=torch.bfloat16)

    vae = pipe.vae
    audio_vae = pipe.audio_vae
    transformer = pipe.transformer
    scheduler_config = pipe.scheduler.config

    vae_temporal_ratio = pipe.vae_temporal_compression_ratio
    vae_spatial_ratio = pipe.vae_spatial_compression_ratio

    # Run stage 1
    video_latent, audio_latent = _run_shared_stage1(pipe, None, device)

    # Upsample
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler).to(device)
    upscaled_latent = upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]
    del upsample_pipe

    # Stage 2 prepare latents (pipeline side)
    stage2_height = HEIGHT * 2
    stage2_width = WIDTH * 2
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal_ratio + 1
    latent_height = stage2_height // vae_spatial_ratio
    latent_width = stage2_width // vae_spatial_ratio

    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)

    generator_s2 = torch.Generator(device).manual_seed(SEED)
    num_channels_latents = transformer.config.in_channels

    with torch.inference_mode():
        prepared_video = pipe.prepare_latents(
            1,
            num_channels_latents,
            stage2_height,
            stage2_width,
            NUM_FRAMES,
            0.0,
            torch.float32,
            device,
            generator_s2,
            upscaled_latent,
        )
        prepared_audio = pipe.prepare_audio_latents(
            1,
            audio_vae.config.latent_channels,
            audio_num_frames,
            audio_vae.config.mel_bins,
            0.0,
            torch.float32,
            device,
            generator_s2,
            audio_latent,
        )

    print(f"\n  Prepared: video={prepared_video.shape}, audio={prepared_audio.shape}")

    # Encode prompt (shared)
    with torch.inference_mode():
        prompt_embeds, prompt_attention_mask, _neg_embeds, _neg_mask = pipe.encode_prompt(
            PROMPT, NEGATIVE_PROMPT, do_classifier_free_guidance=False
        )

    pipe.connectors.to(device)
    with torch.inference_mode():
        additive_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        video_embeds, audio_embeds, connector_mask = pipe.connectors(prompt_embeds, additive_mask, additive_mask=True)
    pipe.connectors.to("cpu")

    # RoPE coords
    video_coords = transformer.rope.prepare_video_coords(
        prepared_video.shape[0],
        latent_num_frames,
        latent_height,
        latent_width,
        prepared_video.device,
        fps=float(FRAME_RATE),
    )
    audio_coords = transformer.audio_rope.prepare_audio_coords(
        prepared_audio.shape[0],
        audio_num_frames,
        prepared_audio.device,
    )

    # Clone for both sides
    shared_video = prepared_video.clone()
    shared_audio = prepared_audio.clone()

    # --- Pipeline side: manual stage 2 denoise loop ---
    video_sequence_length = latent_num_frames * latent_height * latent_width
    sigmas = list(np.linspace(1.0, 1.0 / NUM_STEPS_S2, NUM_STEPS_S2))

    stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        scheduler_config, use_dynamic_shifting=False, shift_terminal=None
    )
    audio_scheduler = copy.deepcopy(stage2_scheduler)

    mu = calculate_shift(
        video_sequence_length,
        stage2_scheduler.config.get("base_image_seq_len", 1024),
        stage2_scheduler.config.get("max_image_seq_len", 4096),
        stage2_scheduler.config.get("base_shift", 0.95),
        stage2_scheduler.config.get("max_shift", 2.05),
    )
    stage2_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    audio_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = stage2_scheduler.timesteps

    pipe_video = shared_video.clone()
    pipe_audio = shared_audio.clone()

    print(f"  Running pipeline-style stage 2 loop: {len(timesteps)} steps...")

    with torch.inference_mode():
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

            pipe_video = stage2_scheduler.step(noise_pred_video, t, pipe_video, return_dict=False)[0]
            pipe_audio = audio_scheduler.step(noise_pred_audio, t, pipe_audio, return_dict=False)[0]

    pipe_video_cpu = pipe_video.cpu()
    pipe_audio_cpu = pipe_audio.cpu()
    print(f"  Pipeline done: video={pipe_video_cpu.shape}, audio={pipe_audio_cpu.shape}")

    # --- Node side: DenoiseNode ---
    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "no_offload"

    node = LTX2DenoiseNode()
    node.device = device

    source = _ValueNode(
        transformer=transformer,
        scheduler_config=scheduler_config,
        prompt_embeds=video_embeds,
        audio_prompt_embeds=audio_embeds,
        attention_mask=connector_mask,
        video_latents=shared_video.clone(),
        audio_latents=shared_audio.clone(),
        video_coords=video_coords,
        audio_coords=audio_coords,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_num_frames=audio_num_frames,
        num_inference_steps=NUM_STEPS_S2,
        guidance_scale=GUIDANCE_SCALE,
        frame_rate=float(FRAME_RATE),
        model_type=0,
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
    _assert_zero_diff("Stage 2 denoise video", pipe_video_cpu, node_video)
    _assert_zero_diff("Stage 2 denoise audio", pipe_audio_cpu, node_audio)

    print("  PASS: Stage 2 denoise matches")

    del pipe, node
    mm.clear()
    set_global_model_manager(ModelManager())


# ---------------------------------------------------------------------------
# Test 4: Full two-stage flow — pipeline vs graph, SEPARATE model loads
# (mirrors the original failing test to confirm the root cause)
# ---------------------------------------------------------------------------


def test_second_pass_full_diagnose(tmp_path, tiny_model_path, tiny_upsampler_path):
    """Diagnose: run both pipeline and graph for 2-stage, compare every intermediate."""
    from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers import FlowMatchEulerDiscreteScheduler

    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents, pack_audio_latents

    device = torch.device("cpu")

    # --- Pipeline side: full 2-stage ---
    pipe = LTX2Pipeline.from_pretrained(tiny_model_path, torch_dtype=torch.bfloat16).to(device)
    generator = torch.Generator(device).manual_seed(SEED)
    pipe_s1_video, pipe_s1_audio = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        frame_rate=FRAME_RATE,
        num_inference_steps=NUM_STEPS_S1,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )
    scheduler_config = pipe.scheduler.config

    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(tiny_upsampler_path, torch_dtype=torch.bfloat16)
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler).to(device)
    pipe_upscaled = upsample_pipe(latents=pipe_s1_video, output_type="latent", return_dict=False)[0]
    del upsample_pipe, latent_upsampler

    # Capture stage 2 prepared latents
    stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        scheduler_config, use_dynamic_shifting=False, shift_terminal=None
    )
    pipe.scheduler = stage2_scheduler
    generator_s2 = torch.Generator(device).manual_seed(SEED)

    # Get the prepared latents by calling prepare_latents/prepare_audio_latents directly
    vae_temporal = pipe.vae_temporal_compression_ratio
    vae_spatial = pipe.vae_spatial_compression_ratio
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal + 1
    latent_h = (HEIGHT * 2) // vae_spatial
    latent_w = (WIDTH * 2) // vae_spatial
    duration_s = NUM_FRAMES / FRAME_RATE
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    num_channels = pipe.transformer.config.in_channels

    with torch.inference_mode():
        pipe_s2_prep_video = pipe.prepare_latents(
            1,
            num_channels,
            HEIGHT * 2,
            WIDTH * 2,
            NUM_FRAMES,
            0.0,
            torch.float32,
            device,
            generator_s2,
            pipe_upscaled,
        )
        pipe_s2_prep_audio = pipe.prepare_audio_latents(
            1,
            pipe.audio_vae.config.latent_channels,
            audio_num_frames,
            pipe.audio_vae.config.mel_bins,
            0.0,
            torch.float32,
            device,
            generator_s2,
            pipe_s1_audio,
        )

    # Also get the re-encoded prompt embeddings for stage 2
    with torch.inference_mode():
        pe, pam, _, _ = pipe.encode_prompt(PROMPT, NEGATIVE_PROMPT, do_classifier_free_guidance=False)
    pipe.connectors.to(device)
    with torch.inference_mode():
        additive_mask = (1 - pam.to(pe.dtype)) * -1000000.0
        pipe_s2_vid_embeds, pipe_s2_aud_embeds, pipe_s2_cmask = pipe.connectors(pe, additive_mask, additive_mask=True)
    pipe.connectors.to("cpu")

    del pipe
    print(f"\n  Pipeline s1: video={pipe_s1_video.shape}, audio={pipe_s1_audio.shape}")
    print(f"  Pipeline upscaled: {pipe_upscaled.shape}")
    print(f"  Pipeline s2 prep: video={pipe_s2_prep_video.shape}, audio={pipe_s2_prep_audio.shape}")

    # --- Graph side ---
    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.clear()

    graph = _build_graph_for_shared(
        str(tmp_path),
        tiny_model_path,
        tiny_upsampler_path,
        num_inference_steps=NUM_STEPS_S1,
        guidance_scale=GUIDANCE_SCALE,
        second_pass_steps=NUM_STEPS_S2,
        second_pass_guidance=GUIDANCE_SCALE,
    )
    graph.get_node_by_name("prompt").text = PROMPT
    graph.get_node_by_name("seed").number = SEED
    graph.get_node_by_name("upsample").upsampler_model_path = tiny_upsampler_path
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths = []
    graph.get_node_by_name("video_send").video_callback = video_paths.append

    graph()

    # Extract intermediates from graph nodes
    graph_s1_denoise = graph.get_node_by_name("denoise")
    graph_s1_video = graph_s1_denoise.values["video_latents"].cpu()
    graph_s1_audio = graph_s1_denoise.values["audio_latents"].cpu()

    graph_upsample = graph.get_node_by_name("upsample")
    graph_upscaled = graph_upsample.values["video_latents"].cpu()

    graph_s2_latents = graph.get_node_by_name("second_pass_latents")
    graph_s2_prep_video = graph_s2_latents.values["video_latents"].cpu()
    graph_s2_prep_audio = graph_s2_latents.values["audio_latents"].cpu()

    graph_prompt = graph.get_node_by_name("prompt_encode")
    graph_vid_embeds = graph_prompt.values["prompt_embeds"].cpu()
    graph_aud_embeds = graph_prompt.values["audio_prompt_embeds"].cpu()
    graph_cmask = graph_prompt.values["attention_mask"].cpu()

    graph_s2_denoise = graph.get_node_by_name("second_pass_denoise")
    graph_s2_video = graph_s2_denoise.values["video_latents"].cpu()

    # --- Compare each intermediate ---
    def _diff(label, a, b):
        if a.shape != b.shape:
            print(f"  {label}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            return
        d = (a.float() - b.float()).abs()
        status = "OK" if d.max() == 0 else f"DIFF max={d.max():.8f}"
        print(f"  {label}: {status}")

    print("\n  --- Intermediate comparison ---")

    # Pack pipeline outputs for comparison
    pipe_s1_v_packed = pack_latents(pipe_s1_video, 1, 1) if pipe_s1_video.ndim == 5 else pipe_s1_video
    pipe_s1_a_packed = pack_audio_latents(pipe_s1_audio) if pipe_s1_audio.ndim == 4 else pipe_s1_audio
    pipe_up_packed = pack_latents(pipe_upscaled, 1, 1) if pipe_upscaled.ndim == 5 else pipe_upscaled

    _diff("Stage 1 video", pipe_s1_v_packed.cpu(), graph_s1_video)
    _diff("Stage 1 audio", pipe_s1_a_packed.cpu(), graph_s1_audio)
    _diff("Upscaled video", pipe_up_packed.cpu(), graph_upscaled)
    _diff("Prompt vid embeds", pipe_s2_vid_embeds.cpu(), graph_vid_embeds)
    _diff("Prompt aud embeds", pipe_s2_aud_embeds.cpu(), graph_aud_embeds)
    _diff("Prompt mask", pipe_s2_cmask.cpu(), graph_cmask)
    _diff("S2 prep video", pipe_s2_prep_video.cpu(), graph_s2_prep_video)
    _diff("S2 prep audio", pipe_s2_prep_audio.cpu(), graph_s2_prep_audio)

    # Check if the graph's two model nodes share the same transformer
    graph_s1_model = graph.get_node_by_name("model")
    graph_s2_model = graph.get_node_by_name("second_pass_model")
    s1_transformer = graph_s1_model.values["transformer"]
    s2_transformer = graph_s2_model.values["transformer"]
    same_obj = s1_transformer is s2_transformer
    print(f"  Transformer same object: {same_obj} (id: {id(s1_transformer)} vs {id(s2_transformer)})")

    # Check weight equality between pipeline and graph transformers
    # (the pipeline was deleted, so reload for comparison)
    pipe2 = LTX2Pipeline.from_pretrained(tiny_model_path, torch_dtype=torch.bfloat16).to(device)
    pipe_tr = pipe2.transformer
    graph_tr = s1_transformer

    max_weight_diff = 0.0
    worst_param = "(none)"
    for (pname, pparam), (gname, gparam) in zip(pipe_tr.named_parameters(), graph_tr.named_parameters()):
        d = (pparam.float() - gparam.float()).abs().max().item()
        if d > max_weight_diff:
            max_weight_diff = d
            worst_param = pname
    print(f"  Transformer weight diff: max={max_weight_diff:.8f} ({worst_param})")
    del pipe2

    # Now run the stage 2 denoise manually using the GRAPH's transformer
    # (which has residual state from stage 1) and the pipeline's inputs
    import copy
    import numpy as np
    from frameartisan.modules.generation.graph.nodes.ltx2_utils import calculate_shift

    sigmas = list(np.linspace(1.0, 1.0 / NUM_STEPS_S2, NUM_STEPS_S2))
    s2_sched = FlowMatchEulerDiscreteScheduler.from_config(
        dict(graph_s2_model.values["scheduler_config"]),
        use_dynamic_shifting=False,
        shift_terminal=None,
    )
    audio_sched = copy.deepcopy(s2_sched)
    video_seq_len = latent_num_frames * latent_h * latent_w
    mu = calculate_shift(
        video_seq_len,
        s2_sched.config.get("base_image_seq_len", 1024),
        s2_sched.config.get("max_image_seq_len", 4096),
        s2_sched.config.get("base_shift", 0.95),
        s2_sched.config.get("max_shift", 2.05),
    )
    s2_sched.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    audio_sched.set_timesteps(sigmas=sigmas, device=device, mu=mu)

    s2_video_coords = graph_s2_latents.values["video_coords"]
    s2_audio_coords = graph_s2_latents.values["audio_coords"]

    manual_video = graph_s2_prep_video.clone().to(device)
    manual_audio = graph_s2_prep_audio.clone().to(device)

    with torch.inference_mode():
        for i, t in enumerate(s2_sched.timesteps):
            lmi = manual_video.to(graph_vid_embeds.dtype)
            ami = manual_audio.to(graph_aud_embeds.dtype)
            ts = t.expand(lmi.shape[0])

            with graph_tr.cache_context("cond_uncond"):
                npv, npa = graph_tr(
                    hidden_states=lmi,
                    audio_hidden_states=ami,
                    encoder_hidden_states=graph_vid_embeds.to(device),
                    audio_encoder_hidden_states=graph_aud_embeds.to(device),
                    timestep=ts,
                    encoder_attention_mask=graph_cmask.to(device),
                    audio_encoder_attention_mask=graph_cmask.to(device),
                    num_frames=latent_num_frames,
                    height=latent_h,
                    width=latent_w,
                    fps=float(FRAME_RATE),
                    audio_num_frames=audio_num_frames,
                    video_coords=s2_video_coords,
                    audio_coords=s2_audio_coords,
                    return_dict=False,
                )
            manual_video = s2_sched.step(npv.float(), t, manual_video, return_dict=False)[0]
            manual_audio = audio_sched.step(npa.float(), t, manual_audio, return_dict=False)[0]

    _diff("Manual s2 denoise (using graph tr after s1) vs graph s2", manual_video.cpu(), graph_s2_video)

    # Compare manual result vs what pipeline WOULD produce
    # This tells us if the residual state affects the graph's transformer
    # Now run the PIPELINE's actual stage 2 via pipe() and compare with manual loop
    # using the PIPELINE'S OWN transformer
    pipe3 = LTX2Pipeline.from_pretrained(tiny_model_path, torch_dtype=torch.bfloat16).to(device)
    # Run stage 1 first (to match state)
    gen1 = torch.Generator(device).manual_seed(SEED)
    pipe3_s1_v, pipe3_s1_a = pipe3(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        frame_rate=FRAME_RATE,
        num_inference_steps=NUM_STEPS_S1,
        guidance_scale=GUIDANCE_SCALE,
        generator=gen1,
        output_type="latent",
        return_dict=False,
    )
    # Upsample
    lups = LTX2LatentUpsamplerModel.from_pretrained(tiny_upsampler_path, torch_dtype=torch.bfloat16)
    upipe = LTX2LatentUpsamplePipeline(vae=pipe3.vae, latent_upsampler=lups).to(device)
    ups3 = upipe(latents=pipe3_s1_v, output_type="latent", return_dict=False)[0]
    del upipe, lups

    # Get stage 2 prepared latents from this pipeline
    s2_sched_pipe = FlowMatchEulerDiscreteScheduler.from_config(
        pipe3.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
    )
    pipe3.scheduler = s2_sched_pipe
    gen2 = torch.Generator(device).manual_seed(SEED)

    pipe3_prep_v = pipe3.prepare_latents(
        1,
        pipe3.transformer.config.in_channels,
        HEIGHT * 2,
        WIDTH * 2,
        NUM_FRAMES,
        0.0,
        torch.float32,
        device,
        gen2,
        ups3,
    )
    pipe3_prep_a = pipe3.prepare_audio_latents(
        1,
        pipe3.audio_vae.config.latent_channels,
        audio_num_frames,
        pipe3.audio_vae.config.mel_bins,
        0.0,
        torch.float32,
        device,
        gen2,
        pipe3_s1_a,
    )

    # Run manual loop using pipe3's transformer
    s2_sched3 = FlowMatchEulerDiscreteScheduler.from_config(
        dict(pipe3.scheduler.config), use_dynamic_shifting=False, shift_terminal=None
    )
    audio_sched3 = copy.deepcopy(s2_sched3)
    mu3 = calculate_shift(
        video_seq_len,
        s2_sched3.config.get("base_image_seq_len", 1024),
        s2_sched3.config.get("max_image_seq_len", 4096),
        s2_sched3.config.get("base_shift", 0.95),
        s2_sched3.config.get("max_shift", 2.05),
    )
    s2_sched3.set_timesteps(sigmas=sigmas, device=device, mu=mu3)
    audio_sched3.set_timesteps(sigmas=sigmas, device=device, mu=mu3)

    # Encode prompt using pipe3
    with torch.inference_mode():
        pe3, pam3, _, _ = pipe3.encode_prompt(PROMPT, NEGATIVE_PROMPT, do_classifier_free_guidance=False)
    pipe3.connectors.to(device)
    with torch.inference_mode():
        am3 = (1 - pam3.to(pe3.dtype)) * -1000000.0
        ve3, ae3, cm3 = pipe3.connectors(pe3, am3, additive_mask=True)
    pipe3.connectors.to("cpu")

    vc3 = pipe3.transformer.rope.prepare_video_coords(
        1, latent_num_frames, latent_h, latent_w, device, fps=float(FRAME_RATE)
    )
    ac3 = pipe3.transformer.audio_rope.prepare_audio_coords(1, audio_num_frames, device)

    m_v = pipe3_prep_v.clone()
    m_a = pipe3_prep_a.clone()

    with torch.inference_mode():
        for i, t in enumerate(s2_sched3.timesteps):
            lmi = m_v.to(ve3.dtype)
            ami = m_a.to(ae3.dtype)
            ts = t.expand(lmi.shape[0])
            with pipe3.transformer.cache_context("cond_uncond"):
                npv3, npa3 = pipe3.transformer(
                    hidden_states=lmi,
                    audio_hidden_states=ami,
                    encoder_hidden_states=ve3,
                    audio_encoder_hidden_states=ae3,
                    timestep=ts,
                    encoder_attention_mask=cm3,
                    audio_encoder_attention_mask=cm3,
                    num_frames=latent_num_frames,
                    height=latent_h,
                    width=latent_w,
                    fps=float(FRAME_RATE),
                    audio_num_frames=audio_num_frames,
                    video_coords=vc3,
                    audio_coords=ac3,
                    return_dict=False,
                )
            m_v = s2_sched3.step(npv3.float(), t, m_v, return_dict=False)[0]
            m_a = audio_sched3.step(npa3.float(), t, m_a, return_dict=False)[0]

    # Now run pipe3's actual full stage 2 via __call__
    gen3 = torch.Generator(device).manual_seed(SEED)
    pipe3.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe3.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
    )
    pipe3_s2_v, pipe3_s2_a = pipe3(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        latents=ups3,
        audio_latents=pipe3_s1_a,
        width=WIDTH * 2,
        height=HEIGHT * 2,
        num_frames=NUM_FRAMES,
        frame_rate=FRAME_RATE,
        num_inference_steps=NUM_STEPS_S2,
        guidance_scale=GUIDANCE_SCALE,
        generator=gen3,
        output_type="latent",
        return_dict=False,
    )
    # Pipeline output is 5D denormalized — pack for comparison
    pipe3_s2_v_packed = pack_latents(pipe3_s2_v, 1, 1) if pipe3_s2_v.ndim == 5 else pipe3_s2_v

    _diff("Manual loop (pipe3 tr) vs graph", m_v.cpu(), graph_s2_video)
    _diff("Pipeline __call__ vs manual loop (same tr)", pipe3_s2_v_packed.cpu(), m_v.cpu())

    del pipe3

    # The real comparison
    # Need to run pipe stage 2 fully for the final denoise comparison
    # but we already extracted the prep latents. The divergence location is what matters.

    del graph
    mm.clear()
    set_global_model_manager(ModelManager())


def _build_graph_for_shared(
    output_dir,
    model_path,
    upsampler_path,
    *,
    num_inference_steps,
    guidance_scale,
    second_pass_steps,
    second_pass_guidance,
):
    """Build a 2nd pass graph (same as test_ltx2_integration._build_second_pass_graph)."""
    import attr

    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings
    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph

    @attr.s
    class _FakeDirs:
        outputs_videos: str = attr.ib(default="")

    model = ModelDataObject(name="tiny_LTX2", filepath=model_path, model_type=0, id=0)
    settings = GenerationSettings(
        model=model,
        second_pass_model=model,
        video_width=WIDTH,
        video_height=HEIGHT,
        video_duration=1,
        frame_rate=FRAME_RATE,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        second_pass_enabled=True,
        second_pass_steps=second_pass_steps,
        second_pass_guidance=second_pass_guidance,
        use_torch_compile=False,
    )
    dirs = _FakeDirs(outputs_videos=output_dir)
    graph = create_default_ltx2_graph(settings, dirs)

    for name in ("upsample", "second_pass_model", "second_pass_lora", "second_pass_latents", "second_pass_denoise"):
        node = graph.get_node_by_name(name)
        node.enabled = True
        node.set_updated()

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

    return graph
