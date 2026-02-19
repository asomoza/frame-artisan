"""Integration test: build the full LTX2 graph, load the tiny model from HF,
and generate a video end-to-end.

Activated by setting the environment variable ``GPU_LIGHT=1``.
Skipped otherwise.

    GPU_LIGHT=1 uv run --extra test pytest tests/ -v -m gpu_light -s

The test downloads ``OzzyGT/tiny_LTX2`` (≈67 MB) on first run. Subsequent
runs use the HuggingFace cache.
"""

from __future__ import annotations

import os

import attr
import pytest
import torch

TINY_MODEL_REPO = "OzzyGT/tiny_LTX2"

pytestmark = pytest.mark.gpu_light


@attr.s
class _FakeDirs:
    outputs_videos: str = attr.ib(default="")


@pytest.fixture(scope="session")
def tiny_model_path():
    """Download the tiny model once per session and return its local path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(TINY_MODEL_REPO)


def _build_graph(
    output_dir: str,
    model_path: str,
    *,
    num_inference_steps: int = 2,
    guidance_scale: float = 1.0,
):
    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings
    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph

    model = ModelDataObject(
        name="tiny_LTX2",
        filepath=model_path,
        model_type=0,
        id=0,
    )
    settings = GenerationSettings(
        model=model,
        video_width=256,
        video_height=256,
        video_duration=1,
        frame_rate=8,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_torch_compile=False,
    )
    dirs = _FakeDirs(outputs_videos=output_dir)
    graph = create_default_ltx2_graph(settings, dirs)

    prompt_node = graph.get_node_by_name("prompt")
    prompt_node.text = "a red ball"

    return graph


def _run_generation(graph) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths = []
    video_send = graph.get_node_by_name("video_send")
    video_send.video_callback = video_paths.append

    graph()
    return video_paths


def test_tiny_no_cfg(tmp_path, tiny_model_path):
    """Tiny model without CFG (guidance_scale=1.0)."""
    output_dir = str(tmp_path)
    graph = _build_graph(
        output_dir,
        tiny_model_path,
        num_inference_steps=2,
        guidance_scale=1.0,
    )

    video_paths = _run_generation(graph)

    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
    assert os.path.isfile(video_paths[0]), f"Video file not found: {video_paths[0]}"
    assert os.path.getsize(video_paths[0]) > 0, "Video file is empty"


def test_tiny_with_cfg(tmp_path, tiny_model_path):
    """Tiny model with CFG (guidance_scale=4.0)."""
    output_dir = str(tmp_path)
    graph = _build_graph(
        output_dir,
        tiny_model_path,
        num_inference_steps=2,
        guidance_scale=4.0,
    )

    video_paths = _run_generation(graph)

    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
    assert os.path.isfile(video_paths[0]), f"Video file not found: {video_paths[0]}"
    assert os.path.getsize(video_paths[0]) > 0, "Video file is empty"


# ---------------------------------------------------------------------------
# Numerical comparison: graph nodes vs diffusers LTX2Pipeline
# ---------------------------------------------------------------------------


def _to_packed_3d(tensor):
    """Normalize a latent tensor to packed 3D [B, seq, C] for comparison.

    Pipeline returns 5D [B, C, F, H, W] (video) or 4D [B, C, T, mel] (audio),
    while graph nodes return packed 3D [B, seq, C].
    """
    if tensor.ndim == 5:
        # Video: [B, C, F, H, W] → [B, F*H*W, C]
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_latents

        return pack_latents(tensor, patch_size=1, patch_size_t=1)
    if tensor.ndim == 4:
        # Audio: [B, C, T, mel] → [B, T, C*mel]
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_audio_latents

        return pack_audio_latents(tensor)
    return tensor


_COMPARISON_PARAMS = dict(
    prompt="a red ball",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    height=256,
    width=256,
    num_frames=9,  # 8*1+1
    frame_rate=8,
    num_inference_steps=2,
    seed=42,
)


def _run_pipeline_latent(model_path: str, *, guidance_scale: float):
    """Run the diffusers LTX2Pipeline and return (video_latent, audio_latent)."""
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cpu")

    generator = torch.Generator("cpu").manual_seed(_COMPARISON_PARAMS["seed"])

    video_latent, audio_latent = pipe(
        prompt=_COMPARISON_PARAMS["prompt"],
        negative_prompt=_COMPARISON_PARAMS["negative_prompt"],
        height=_COMPARISON_PARAMS["height"],
        width=_COMPARISON_PARAMS["width"],
        num_frames=_COMPARISON_PARAMS["num_frames"],
        frame_rate=_COMPARISON_PARAMS["frame_rate"],
        num_inference_steps=_COMPARISON_PARAMS["num_inference_steps"],
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )

    return _to_packed_3d(video_latent).cpu(), _to_packed_3d(audio_latent).cpu()


def _run_graph_latent(
    model_path: str,
    output_dir: str,
    *,
    guidance_scale: float,
):
    """Run the node graph and return (video_latent, audio_latent) from the denoise node."""
    graph = _build_graph(
        output_dir,
        model_path,
        num_inference_steps=_COMPARISON_PARAMS["num_inference_steps"],
        guidance_scale=guidance_scale,
    )

    graph.get_node_by_name("prompt").text = _COMPARISON_PARAMS["prompt"]
    graph.get_node_by_name("seed").number = _COMPARISON_PARAMS["seed"]

    device = torch.device("cpu")
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths = []
    graph.get_node_by_name("video_send").video_callback = video_paths.append

    graph()

    denoise_node = graph.get_node_by_name("denoise")
    return denoise_node.values["video_latents"].cpu(), denoise_node.values["audio_latents"].cpu()


@pytest.mark.parametrize(
    "guidance_scale",
    [1.0, 4.0],
    ids=["no_cfg", "with_cfg"],
)
def test_numerical_match_pipeline_vs_graph(tmp_path, tiny_model_path, guidance_scale):
    """Verify that the decomposed graph produces identical latents to LTX2Pipeline."""
    from frameartisan.app.model_manager import get_model_manager

    get_model_manager().clear()

    pipe_video, pipe_audio = _run_pipeline_latent(tiny_model_path, guidance_scale=guidance_scale)
    graph_video, graph_audio = _run_graph_latent(
        tiny_model_path,
        str(tmp_path),
        guidance_scale=guidance_scale,
    )

    # --- Video latent comparison ---
    assert pipe_video.shape == graph_video.shape, (
        f"Video shape mismatch: pipeline {pipe_video.shape} vs graph {graph_video.shape}"
    )
    video_diff = (pipe_video.float() - graph_video.float()).abs()
    print(
        f"\n  Video latents: shape={pipe_video.shape}, max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}"
    )
    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"

    # --- Audio latent comparison ---
    assert pipe_audio.shape == graph_audio.shape, (
        f"Audio shape mismatch: pipeline {pipe_audio.shape} vs graph {graph_audio.shape}"
    )
    audio_diff = (pipe_audio.float() - graph_audio.float()).abs()
    print(f"  Audio latents: shape={pipe_audio.shape}, max_diff={audio_diff.max():.8f}")
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"


# ---------------------------------------------------------------------------
# Numerical comparison: group offload with leaf_level + use_stream
# ---------------------------------------------------------------------------

_GROUP_OFFLOAD_PARAMS = dict(
    prompt="a red ball",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    height=256,
    width=256,
    num_frames=9,  # 8*1+1
    frame_rate=8,
    num_inference_steps=2,
    seed=42,
)


def _run_pipeline_group_offload_latent(model_path: str, *, guidance_scale: float):
    """Run LTX2Pipeline with group_offload (leaf_level, use_stream=True) on CUDA, return latents."""
    import warnings

    from diffusers import LTX2Pipeline

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    pipe = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    onload_device = torch.device("cuda")
    pipe.enable_group_offload(
        onload_device=onload_device,
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True,
    )

    # Use CUDA generator to match the graph's latents node which creates
    # torch.Generator(device) — CPU and CUDA generators produce different
    # random sequences even with the same seed.
    generator = torch.Generator("cuda").manual_seed(_GROUP_OFFLOAD_PARAMS["seed"])

    video_latent, audio_latent = pipe(
        prompt=_GROUP_OFFLOAD_PARAMS["prompt"],
        negative_prompt=_GROUP_OFFLOAD_PARAMS["negative_prompt"],
        height=_GROUP_OFFLOAD_PARAMS["height"],
        width=_GROUP_OFFLOAD_PARAMS["width"],
        num_frames=_GROUP_OFFLOAD_PARAMS["num_frames"],
        frame_rate=_GROUP_OFFLOAD_PARAMS["frame_rate"],
        num_inference_steps=_GROUP_OFFLOAD_PARAMS["num_inference_steps"],
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )

    return _to_packed_3d(video_latent).cpu(), _to_packed_3d(audio_latent).cpu()


def _run_graph_group_offload_latent(
    model_path: str,
    output_dir: str,
    *,
    guidance_scale: float,
):
    """Run the node graph with group_offload (leaf_level, use_stream=True) on CUDA, return latents."""
    import warnings

    from frameartisan.app.model_manager import ModelManager, get_model_manager, set_global_model_manager

    warnings.filterwarnings("ignore", message=".*layers were not executed.*")

    # Fresh ModelManager so no stale hooks
    set_global_model_manager(ModelManager())
    mm = get_model_manager()
    mm.offload_strategy = "group_offload"

    # Patch apply_offload_strategy to use use_stream=True
    _original_apply = ModelManager.apply_offload_strategy

    def _patched_apply(self, device):
        from diffusers.hooks.group_offloading import apply_group_offloading

        strategy = self.resolve_offload_strategy(device)

        with self._lock:
            if not self._managed_components:
                self._applied_strategy = strategy
                return strategy
            if self._applied_strategy == strategy:
                return strategy

        self.prepare_strategy_transition(strategy, device)

        if strategy == "group_offload":
            offload_kwargs = dict(
                onload_device=torch.device(device) if isinstance(device, str) else device,
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
            )
            with self._lock:
                components = list(self._managed_components.items())
            for name, mod in components:
                apply_group_offloading(mod, **offload_kwargs)

        return strategy

    ModelManager.apply_offload_strategy = _patched_apply

    try:
        graph = _build_graph(
            output_dir,
            model_path,
            num_inference_steps=_GROUP_OFFLOAD_PARAMS["num_inference_steps"],
            guidance_scale=guidance_scale,
        )

        graph.get_node_by_name("prompt").text = _GROUP_OFFLOAD_PARAMS["prompt"]
        graph.get_node_by_name("seed").number = _GROUP_OFFLOAD_PARAMS["seed"]

        device = torch.device("cuda")
        graph.device = device
        graph.dtype = torch.bfloat16

        video_paths = []
        graph.get_node_by_name("video_send").video_callback = video_paths.append

        graph()

        denoise_node = graph.get_node_by_name("denoise")
        return denoise_node.values["video_latents"].cpu(), denoise_node.values["audio_latents"].cpu()
    finally:
        ModelManager.apply_offload_strategy = _original_apply
        set_global_model_manager(ModelManager())


@pytest.mark.parametrize(
    "guidance_scale",
    [1.0, 4.0],
    ids=["no_cfg", "with_cfg"],
)
def test_numerical_match_group_offload_streams(tmp_path, tiny_model_path, guidance_scale):
    """Verify pipeline vs graph produce identical latents with group_offload + use_stream=True."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for group_offload stream test")

    pipe_video, pipe_audio = _run_pipeline_group_offload_latent(
        tiny_model_path,
        guidance_scale=guidance_scale,
    )
    graph_video, graph_audio = _run_graph_group_offload_latent(
        tiny_model_path,
        str(tmp_path),
        guidance_scale=guidance_scale,
    )

    # --- Video latent comparison ---
    assert pipe_video.shape == graph_video.shape, (
        f"Video shape mismatch: pipeline {pipe_video.shape} vs graph {graph_video.shape}"
    )
    video_diff = (pipe_video.float() - graph_video.float()).abs()
    print(f"\n  [group_offload+stream] Video latents: shape={pipe_video.shape}, max_diff={video_diff.max():.8f}")
    assert video_diff.max() == 0, f"Video latents differ: max_diff={video_diff.max():.8f}"

    # --- Audio latent comparison ---
    assert pipe_audio.shape == graph_audio.shape, (
        f"Audio shape mismatch: pipeline {pipe_audio.shape} vs graph {graph_audio.shape}"
    )
    audio_diff = (pipe_audio.float() - graph_audio.float()).abs()
    print(f"  [group_offload+stream] Audio latents: shape={pipe_audio.shape}, max_diff={audio_diff.max():.8f}")
    assert audio_diff.max() == 0, f"Audio latents differ: max_diff={audio_diff.max():.8f}"


# ---------------------------------------------------------------------------
# Numerical comparison: I2V pipeline vs graph with image conditioning
# ---------------------------------------------------------------------------

_I2V_PARAMS = dict(
    prompt="a red ball",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    height=256,
    width=256,
    num_frames=9,  # 8*1+1
    frame_rate=8,
    num_inference_steps=2,
    seed=42,
)


def _make_solid_red_image_pil(height: int, width: int):
    """Create a solid red PIL Image."""
    from PIL import Image

    return Image.new("RGB", (width, height), (255, 0, 0))


def _run_i2v_pipeline_latent(model_path: str, *, guidance_scale: float):
    """Run the diffusers LTX2ImageToVideoPipeline and return (video_latent, audio_latent)."""
    from diffusers import LTX2ImageToVideoPipeline

    pipe = LTX2ImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cpu")

    generator = torch.Generator("cpu").manual_seed(_I2V_PARAMS["seed"])
    image = _make_solid_red_image_pil(_I2V_PARAMS["height"], _I2V_PARAMS["width"])

    video_latent, audio_latent = pipe(
        image=image,
        prompt=_I2V_PARAMS["prompt"],
        negative_prompt=_I2V_PARAMS["negative_prompt"],
        height=_I2V_PARAMS["height"],
        width=_I2V_PARAMS["width"],
        num_frames=_I2V_PARAMS["num_frames"],
        frame_rate=_I2V_PARAMS["frame_rate"],
        num_inference_steps=_I2V_PARAMS["num_inference_steps"],
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )

    return _to_packed_3d(video_latent).cpu(), _to_packed_3d(audio_latent).cpu()


def _run_graph_i2v_latent(
    model_path: str,
    output_dir: str,
    *,
    guidance_scale: float,
):
    """Run the node graph with image conditioning and return (video_latent, audio_latent)."""
    import numpy as np

    graph = _build_graph(
        output_dir,
        model_path,
        num_inference_steps=_I2V_PARAMS["num_inference_steps"],
        guidance_scale=guidance_scale,
    )

    graph.get_node_by_name("prompt").text = _I2V_PARAMS["prompt"]
    graph.get_node_by_name("seed").number = _I2V_PARAMS["seed"]

    # Create a solid red image as numpy (same as the PIL image for the pipeline)
    fake_image = np.zeros((_I2V_PARAMS["height"], _I2V_PARAMS["width"], 3), dtype=np.uint8)
    fake_image[:, :, 0] = 255

    # Enable source image nodes
    source_node = graph.get_node_by_name("source_image")
    source_node.enabled = True
    source_node.image = fake_image
    source_node.set_updated()

    encode_node = graph.get_node_by_name("image_encode")
    encode_node.enabled = True
    encode_node.set_updated()

    device = torch.device("cpu")
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths = []
    graph.get_node_by_name("video_send").video_callback = video_paths.append

    graph()

    denoise_node = graph.get_node_by_name("denoise")
    return denoise_node.values["video_latents"].cpu(), denoise_node.values["audio_latents"].cpu()


@pytest.mark.parametrize(
    "guidance_scale",
    [1.0, 4.0],
    ids=["no_cfg", "with_cfg"],
)
def test_numerical_match_i2v_pipeline_vs_graph(tmp_path, tiny_model_path, guidance_scale):
    """Verify that the decomposed graph with image conditioning produces identical latents to LTX2ImageToVideoPipeline."""
    from frameartisan.app.model_manager import get_model_manager

    get_model_manager().clear()

    pipe_video, pipe_audio = _run_i2v_pipeline_latent(tiny_model_path, guidance_scale=guidance_scale)
    graph_video, graph_audio = _run_graph_i2v_latent(
        tiny_model_path,
        str(tmp_path),
        guidance_scale=guidance_scale,
    )

    # --- Video latent comparison ---
    assert pipe_video.shape == graph_video.shape, (
        f"Video shape mismatch: pipeline {pipe_video.shape} vs graph {graph_video.shape}"
    )
    video_diff = (pipe_video.float() - graph_video.float()).abs()
    print(
        f"\n  I2V Video latents: shape={pipe_video.shape}, max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}"
    )
    assert video_diff.max() == 0, f"I2V Video latents differ: max_diff={video_diff.max():.8f}"

    # --- Audio latent comparison ---
    assert pipe_audio.shape == graph_audio.shape, (
        f"Audio shape mismatch: pipeline {pipe_audio.shape} vs graph {graph_audio.shape}"
    )
    audio_diff = (pipe_audio.float() - graph_audio.float()).abs()
    print(f"  I2V Audio latents: shape={pipe_audio.shape}, max_diff={audio_diff.max():.8f}")
    assert audio_diff.max() == 0, f"I2V Audio latents differ: max_diff={audio_diff.max():.8f}"


# ---------------------------------------------------------------------------
# Image conditioning: verify that a source image changes the output
# ---------------------------------------------------------------------------


def test_image_conditioning_changes_output(tmp_path, tiny_model_path):
    """Run generation with and without a source image and verify the outputs differ.

    Validates the full image conditioning pipeline end-to-end: ImageLoadNode →
    LTX2ImageEncodeNode (VAE encode) → LTX2LatentsNode (latent blending) →
    LTX2DenoiseNode (timestep masking + frame-0 preservation).
    """
    import gc

    import numpy as np

    from frameartisan.app.model_manager import get_model_manager

    mm = get_model_manager()
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Run 1: baseline (no image conditioning) ---
    graph_baseline = _build_graph(
        str(tmp_path / "baseline"),
        tiny_model_path,
        num_inference_steps=2,
        guidance_scale=1.0,
    )
    os.makedirs(str(tmp_path / "baseline"), exist_ok=True)
    graph_baseline.get_node_by_name("seed").number = 42
    graph_baseline.device = device
    graph_baseline.dtype = torch.bfloat16
    video_paths_baseline = []
    graph_baseline.get_node_by_name("video_send").video_callback = video_paths_baseline.append
    graph_baseline()

    decode_baseline = graph_baseline.get_node_by_name("decode")
    video_baseline = decode_baseline.values["video"].copy()

    del graph_baseline
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Run 2: with a solid-color source image ---
    graph_img = _build_graph(
        str(tmp_path / "img_cond"),
        tiny_model_path,
        num_inference_steps=2,
        guidance_scale=1.0,
    )
    os.makedirs(str(tmp_path / "img_cond"), exist_ok=True)
    graph_img.get_node_by_name("seed").number = 42

    # Create a solid red image (256x256 RGB uint8)
    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    fake_image[:, :, 0] = 255  # red channel

    # Enable source_image and image_encode nodes, inject the fake image directly
    source_node = graph_img.get_node_by_name("source_image")
    source_node.enabled = True
    source_node.image = fake_image
    source_node.set_updated()

    encode_node = graph_img.get_node_by_name("image_encode")
    encode_node.enabled = True
    encode_node.set_updated()

    graph_img.device = device
    graph_img.dtype = torch.bfloat16
    video_paths_img = []
    graph_img.get_node_by_name("video_send").video_callback = video_paths_img.append
    graph_img()

    decode_img = graph_img.get_node_by_name("decode")
    video_img = decode_img.values["video"].copy()

    # --- Assertions ---
    assert len(video_paths_baseline) == 1, f"Baseline: expected 1 video, got {len(video_paths_baseline)}"
    assert len(video_paths_img) == 1, f"Image cond: expected 1 video, got {len(video_paths_img)}"

    assert video_baseline.shape == video_img.shape, (
        f"Shape mismatch: baseline {video_baseline.shape} vs img_cond {video_img.shape}"
    )

    # The outputs MUST differ — image conditioning should change the result
    diff = np.abs(video_baseline.astype(np.int16) - video_img.astype(np.int16))
    max_diff = diff.max()
    mean_diff = diff.astype(np.float32).mean()
    print(f"\n  Image conditioning effect: max_diff={max_diff}, mean_diff={mean_diff:.4f}")
    assert max_diff > 0, "Image conditioning had no effect — outputs are identical"
    assert mean_diff > 1.0, f"Image conditioning effect too small: mean_diff={mean_diff:.4f}"

    # Cleanup
    del graph_img
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Offload strategy switching: replicates the full UI flow
# ---------------------------------------------------------------------------

_OFFLOAD_STRATEGIES = ["no_offload", "model_offload", "group_offload"]


def test_offload_strategy_switching(tmp_path, tiny_model_path):
    """Switch between all offload strategies in one session, replicating the UI flow.

    Mirrors what GenerationModule + NodeGraphThread do:
    1. Build a staged graph (like GenerationModule.build_graph).
    2. For the first strategy: serialise → from_json into a persistent run graph → run.
    3. For subsequent strategies: change mm.offload_strategy, mark model node updated
       on the staged graph (like on_generation_change), serialise → update_from_json
       on the same persistent run graph → run.

    This catches issues with stale offload hooks, leftover GPU placement, and
    strategy transitions between all combinations.
    """
    import gc

    from frameartisan.app.model_manager import get_model_manager
    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings
    from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph
    from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES

    mm = get_model_manager()
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Build the staged graph (lives for the entire session, like GenerationModule.node_graph) --
    model_obj = ModelDataObject(name="tiny_LTX2", filepath=tiny_model_path, model_type=0, id=0)
    gen_settings = GenerationSettings(
        model=model_obj,
        video_width=256,
        video_height=256,
        video_duration=1,
        frame_rate=8,
        num_inference_steps=2,
        guidance_scale=1.0,
        offload_strategy=_OFFLOAD_STRATEGIES[0],
    )
    mm.offload_strategy = gen_settings.offload_strategy

    staged_graph = create_default_ltx2_graph(gen_settings, _FakeDirs(outputs_videos=str(tmp_path)))
    staged_graph.get_node_by_name("prompt").text = "a red ball"

    # The persistent run graph lives across generations (like NodeGraphThread._persistent_run_graph)
    persistent_run_graph = None

    for i, strategy in enumerate(_OFFLOAD_STRATEGIES):
        print(f"\n  [{i + 1}/{len(_OFFLOAD_STRATEGIES)}] Testing strategy: {strategy}")

        # -- Replicate on_generation_change for offload_strategy --
        mm.offload_strategy = strategy
        model_node = staged_graph.get_node_by_name("model")
        model_node.offload_strategy = strategy
        model_node.set_updated()

        # -- Replicate on_generate: serialise the staged graph --
        json_graph = staged_graph.to_json()

        # -- Replicate NodeGraphThread.create_run_graph_from_json --
        if persistent_run_graph is None:
            persistent_run_graph = FrameArtisanNodeGraph()
            persistent_run_graph.from_json(json_graph, node_classes=NODE_CLASSES)
        else:
            persistent_run_graph.update_from_json(json_graph, node_classes=NODE_CLASSES)

        persistent_run_graph.device = device
        persistent_run_graph.dtype = torch.bfloat16

        # -- Wire callbacks (like NodeGraphThread.wire_callbacks) --
        video_paths = []
        video_send = persistent_run_graph.get_node_by_name("video_send")
        video_send.video_callback = video_paths.append
        video_send.output_dir = str(tmp_path)

        # -- Run the graph --
        persistent_run_graph()

        assert len(video_paths) == 1, f"Strategy {strategy}: expected 1 video, got {len(video_paths)}"
        assert os.path.isfile(video_paths[0]), f"Strategy {strategy}: video not found: {video_paths[0]}"
        assert os.path.getsize(video_paths[0]) > 0, f"Strategy {strategy}: video file is empty"

        print(f"  PASS: {strategy} — {video_paths[0]}")

    # Final cleanup
    del persistent_run_graph
    del staged_graph
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 2nd pass (upsample) end-to-end test with mocked upsampler
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_upsampler_path(tiny_model_path):
    """Return the path to the tiny latent upsampler inside the tiny_LTX2 repo."""
    return os.path.join(tiny_model_path, "latent_upsampler")


def _build_second_pass_graph(
    output_dir: str,
    model_path: str,
    *,
    num_inference_steps: int = 2,
    guidance_scale: float = 1.0,
    second_pass_steps: int = 2,
    second_pass_guidance: float = 1.0,
):
    """Build a graph with 2nd pass enabled, using the same model for both passes."""
    from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings
    from frameartisan.modules.generation.graph.new_graph import create_default_ltx2_graph

    model = ModelDataObject(
        name="tiny_LTX2",
        filepath=model_path,
        model_type=0,
        id=0,
    )
    settings = GenerationSettings(
        model=model,
        second_pass_model=model,
        video_width=256,
        video_height=256,
        video_duration=1,
        frame_rate=8,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        second_pass_enabled=True,
        second_pass_steps=second_pass_steps,
        second_pass_guidance=second_pass_guidance,
        use_torch_compile=False,
    )
    dirs = _FakeDirs(outputs_videos=output_dir)
    graph = create_default_ltx2_graph(settings, dirs)

    graph.get_node_by_name("prompt").text = "a red ball"

    # Enable 2nd pass nodes
    for name in ("upsample", "second_pass_model", "second_pass_lora", "second_pass_latents", "second_pass_denoise"):
        node = graph.get_node_by_name(name)
        node.enabled = True
        node.set_updated()

    # Rewire decode to read from 2nd pass outputs
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


def _run_second_pass_e2e(tmp_path, tiny_model_path, tiny_upsampler_path, offload_strategy="auto"):
    """Shared logic for 2nd pass end-to-end tests with configurable offload strategy."""
    import gc

    from frameartisan.app.model_manager import get_model_manager

    mm = get_model_manager()
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mm.offload_strategy = offload_strategy

    graph = _build_second_pass_graph(
        str(tmp_path),
        tiny_model_path,
        num_inference_steps=2,
        guidance_scale=1.0,
        second_pass_steps=2,
        second_pass_guidance=1.0,
    )

    # Set offload strategy on both model nodes
    model_node = graph.get_node_by_name("model")
    model_node.offload_strategy = offload_strategy
    sp_model_node = graph.get_node_by_name("second_pass_model")
    sp_model_node.offload_strategy = offload_strategy

    graph.get_node_by_name("seed").number = 42

    # Point upsampler to the tiny upsampler model
    upsample_node = graph.get_node_by_name("upsample")
    upsample_node.upsampler_model_path = tiny_upsampler_path

    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths = []
    graph.get_node_by_name("video_send").video_callback = video_paths.append

    graph()

    assert len(video_paths) == 1, f"Expected 1 video, got {len(video_paths)}"
    assert os.path.isfile(video_paths[0]), f"Video file not found: {video_paths[0]}"
    assert os.path.getsize(video_paths[0]) > 0, "Video file is empty"

    # The decode output should be at the upsampled resolution (2x the latent dims).
    # Base: 256x256 → latent 32x32 → upsampled latent 64x64 → decoded 512x512
    decode_node = graph.get_node_by_name("decode")
    video = decode_node.values["video"]
    # video shape: [F, H, W, C]
    _, h, w, _ = video.shape
    print(f"\n  2nd pass ({offload_strategy}) video: shape={video.shape}, resolution={w}x{h}")
    assert h == 512, f"Expected height 512 (2x 256), got {h}"
    assert w == 512, f"Expected width 512 (2x 256), got {w}"

    # Cleanup
    del graph
    mm.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.gpu_light
def test_second_pass_end_to_end(tmp_path, tiny_model_path, tiny_upsampler_path):
    """Run the full 2-stage pipeline with auto offload (no_offload on large GPU)."""
    _run_second_pass_e2e(tmp_path, tiny_model_path, tiny_upsampler_path, offload_strategy="auto")


@pytest.mark.gpu_light
def test_second_pass_model_offload(tmp_path, tiny_model_path, tiny_upsampler_path):
    """Run the full 2-stage pipeline with model_offload.

    This specifically tests that use_components correctly moves the transformer
    to GPU for the second pass denoise, even after the upsample node registers
    a new component (which resets _applied_strategy).
    """
    _run_second_pass_e2e(tmp_path, tiny_model_path, tiny_upsampler_path, offload_strategy="model_offload")


@pytest.mark.gpu_light
def test_second_pass_group_offload(tmp_path, tiny_model_path, tiny_upsampler_path):
    """Run the full 2-stage pipeline with group_offload.

    Tests that group offload hooks are properly applied to all components
    including the latent upsampler and second pass transformer.
    """
    _run_second_pass_e2e(tmp_path, tiny_model_path, tiny_upsampler_path, offload_strategy="group_offload")


# ---------------------------------------------------------------------------
# 2nd pass numerical comparison: pipeline vs graph
# ---------------------------------------------------------------------------

_SECOND_PASS_PARAMS = dict(
    prompt="a red ball",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    height=256,
    width=256,
    num_frames=9,
    frame_rate=8,
    num_inference_steps=2,
    second_pass_steps=2,
    seed=42,
)


def _run_pipeline_two_stage_latent(model_path: str, upsampler_path: str, *, guidance_scale: float):
    """Run 2-stage diffusers pipeline on CPU and return (video_latent, audio_latent)."""
    from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    p = _SECOND_PASS_PARAMS
    device = torch.device("cpu")

    # --- Stage 1 ---
    pipe = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    generator = torch.Generator(device).manual_seed(p["seed"])

    video_latent, audio_latent = pipe(
        prompt=p["prompt"],
        negative_prompt=p["negative_prompt"],
        width=p["width"],
        height=p["height"],
        num_frames=p["num_frames"],
        frame_rate=p["frame_rate"],
        num_inference_steps=p["num_inference_steps"],
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )
    scheduler_config = pipe.scheduler.config

    # --- Upsample ---
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(upsampler_path, torch_dtype=torch.bfloat16)
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler).to(device)

    upscaled_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    del upsample_pipe, latent_upsampler

    # --- Stage 2 ---
    stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        scheduler_config, use_dynamic_shifting=False, shift_terminal=None
    )
    pipe.scheduler = stage2_scheduler

    generator_s2 = torch.Generator(device).manual_seed(p["seed"])

    video_latent_s2, audio_latent_s2 = pipe(
        prompt=p["prompt"],
        negative_prompt=p["negative_prompt"],
        latents=upscaled_latent,
        audio_latents=audio_latent,
        width=p["width"] * 2,
        height=p["height"] * 2,
        num_frames=p["num_frames"],
        frame_rate=p["frame_rate"],
        num_inference_steps=p["second_pass_steps"],
        guidance_scale=guidance_scale,
        generator=generator_s2,
        output_type="latent",
        return_dict=False,
    )

    del pipe
    return _to_packed_3d(video_latent_s2).cpu(), _to_packed_3d(audio_latent_s2).cpu()


def _run_graph_two_stage_latent(
    model_path: str,
    upsampler_path: str,
    output_dir: str,
    *,
    guidance_scale: float,
):
    """Run the 2-stage node graph on CPU and return (video_latent, audio_latent)."""
    from frameartisan.app.model_manager import get_model_manager

    mm = get_model_manager()
    mm.clear()

    p = _SECOND_PASS_PARAMS
    graph = _build_second_pass_graph(
        output_dir,
        model_path,
        num_inference_steps=p["num_inference_steps"],
        guidance_scale=guidance_scale,
        second_pass_steps=p["second_pass_steps"],
        second_pass_guidance=guidance_scale,
    )

    graph.get_node_by_name("prompt").text = p["prompt"]
    graph.get_node_by_name("seed").number = p["seed"]
    graph.get_node_by_name("upsample").upsampler_model_path = upsampler_path

    # Use no_offload for CPU numerical comparison — "auto" resolves to
    # "group_offload" on CPU which is unnecessary and the pipeline doesn't
    # use offload hooks either.
    mm.offload_strategy = "no_offload"
    graph.get_node_by_name("model").offload_strategy = "no_offload"
    graph.get_node_by_name("second_pass_model").offload_strategy = "no_offload"

    device = torch.device("cpu")
    graph.device = device
    graph.dtype = torch.bfloat16

    video_paths = []
    graph.get_node_by_name("video_send").video_callback = video_paths.append

    graph()

    sp_denoise = graph.get_node_by_name("second_pass_denoise")
    video_latent = sp_denoise.values["video_latents"].cpu()
    audio_latent = sp_denoise.values["audio_latents"].cpu()

    del graph
    get_model_manager().clear()
    return video_latent, audio_latent


@pytest.mark.parametrize(
    "guidance_scale",
    [1.0, 4.0],
    ids=["no_cfg", "with_cfg"],
)
def test_numerical_match_second_pass_pipeline_vs_graph(tmp_path, tiny_model_path, tiny_upsampler_path, guidance_scale):
    """Verify that the 2-stage graph produces identical latents to 2-stage pipeline."""
    from frameartisan.app.model_manager import get_model_manager

    get_model_manager().clear()

    pipe_video, pipe_audio = _run_pipeline_two_stage_latent(
        tiny_model_path, tiny_upsampler_path, guidance_scale=guidance_scale
    )
    graph_video, graph_audio = _run_graph_two_stage_latent(
        tiny_model_path, tiny_upsampler_path, str(tmp_path), guidance_scale=guidance_scale
    )

    # --- Video latent comparison ---
    assert pipe_video.shape == graph_video.shape, (
        f"Video shape mismatch: pipeline {pipe_video.shape} vs graph {graph_video.shape}"
    )
    video_diff = (pipe_video.float() - graph_video.float()).abs()
    print(
        f"\n  2nd pass video latents: shape={pipe_video.shape}, max_diff={video_diff.max():.8f}, mean_diff={video_diff.mean():.8f}"
    )
    assert video_diff.max() == 0, f"2nd pass video latents differ: max_diff={video_diff.max():.8f}"

    # --- Audio latent comparison ---
    assert pipe_audio.shape == graph_audio.shape, (
        f"Audio shape mismatch: pipeline {pipe_audio.shape} vs graph {graph_audio.shape}"
    )
    audio_diff = (pipe_audio.float() - graph_audio.float()).abs()
    print(f"  2nd pass audio latents: shape={pipe_audio.shape}, max_diff={audio_diff.max():.8f}")
    assert audio_diff.max() == 0, f"2nd pass audio latents differ: max_diff={audio_diff.max():.8f}"
