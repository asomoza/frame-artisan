from __future__ import annotations

from typing import TYPE_CHECKING

from frameartisan.modules.generation.constants import DEFAULT_NEGATIVE_PROMPT
from frameartisan.modules.generation.generation_settings import compute_num_frames
from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
from frameartisan.modules.generation.graph.nodes.boolean_node import BooleanNode
from frameartisan.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from frameartisan.modules.generation.graph.nodes.ltx2_audio_encode_node import LTX2AudioEncodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_condition_encode_node import LTX2ConditionEncodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_decode_node import LTX2DecodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_denoise_node import LTX2DenoiseNode
from frameartisan.modules.generation.graph.nodes.ltx2_image_encode_node import LTX2ImageEncodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_latent_upsample_node import LTX2LatentUpsampleNode
from frameartisan.modules.generation.graph.nodes.ltx2_latents_node import LTX2LatentsNode
from frameartisan.modules.generation.graph.nodes.ltx2_lora_node import LTX2LoraNode
from frameartisan.modules.generation.graph.nodes.ltx2_model_node import LTX2ModelNode
from frameartisan.modules.generation.graph.nodes.ltx2_prompt_encode_node import LTX2PromptEncodeNode
from frameartisan.modules.generation.graph.nodes.ltx2_second_pass_latents_node import LTX2SecondPassLatentsNode
from frameartisan.modules.generation.graph.nodes.ltx2_video_send_node import LTX2VideoSendNode
from frameartisan.modules.generation.graph.nodes.number_node import NumberNode
from frameartisan.modules.generation.graph.nodes.text_node import TextNode


if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject
    from frameartisan.modules.generation.generation_settings import GenerationSettings


def create_default_ltx2_graph(
    gen_settings: GenerationSettings,
    directories: DirectoriesObject,
) -> FrameArtisanNodeGraph:
    graph = FrameArtisanNodeGraph()

    # --- Source nodes ---
    model_node = LTX2ModelNode(
        model_path=gen_settings.model.filepath,
        model_id=gen_settings.model.id,
        offload_strategy=gen_settings.offload_strategy,
        group_offload_use_stream=gen_settings.group_offload_use_stream,
        group_offload_low_cpu_mem=gen_settings.group_offload_low_cpu_mem,
        streaming_decode=gen_settings.streaming_decode,
        ff_chunking=gen_settings.ff_chunking,
        ff_num_chunks=gen_settings.ff_num_chunks,
    )
    graph.add_node(model_node, name="model")

    lora_node = LTX2LoraNode(lora_configs=gen_settings.active_loras)
    graph.add_node(lora_node, name="lora")

    prompt_node = TextNode(text="")
    graph.add_node(prompt_node, name="prompt")

    negative_prompt_node = TextNode(text=DEFAULT_NEGATIVE_PROMPT)
    graph.add_node(negative_prompt_node, name="negative_prompt")

    seed_node = NumberNode(number=0)
    graph.add_node(seed_node, name="seed")

    steps_node = NumberNode(number=gen_settings.num_inference_steps)
    graph.add_node(steps_node, name="num_inference_steps")

    guidance_node = NumberNode(number=gen_settings.guidance_scale)
    graph.add_node(guidance_node, name="guidance_scale")

    width_node = NumberNode(number=gen_settings.video_width)
    graph.add_node(width_node, name="width")

    height_node = NumberNode(number=gen_settings.video_height)
    graph.add_node(height_node, name="height")

    num_frames_node = NumberNode(number=compute_num_frames(gen_settings.video_duration, gen_settings.frame_rate))
    graph.add_node(num_frames_node, name="num_frames")

    frame_rate_node = NumberNode(number=gen_settings.frame_rate)
    graph.add_node(frame_rate_node, name="frame_rate")

    model_type_node = NumberNode(number=gen_settings.model.model_type)
    graph.add_node(model_type_node, name="model_type")

    use_torch_compile_node = BooleanNode(value=gen_settings.use_torch_compile)
    graph.add_node(use_torch_compile_node, name="use_torch_compile")

    torch_compile_max_autotune_node = BooleanNode(value=gen_settings.torch_compile_max_autotune)
    graph.add_node(torch_compile_max_autotune_node, name="torch_compile_max_autotune")

    # --- Advanced guidance source nodes ---
    advanced_guidance_node = BooleanNode(value=gen_settings.advanced_guidance)
    graph.add_node(advanced_guidance_node, name="advanced_guidance")

    stg_scale_node = NumberNode(number=gen_settings.stg_scale)
    graph.add_node(stg_scale_node, name="stg_scale")

    stg_blocks_node = TextNode(text=gen_settings.stg_blocks)
    graph.add_node(stg_blocks_node, name="stg_blocks")

    rescale_scale_node = NumberNode(number=gen_settings.rescale_scale)
    graph.add_node(rescale_scale_node, name="rescale_scale")

    modality_scale_node = NumberNode(number=gen_settings.modality_scale)
    graph.add_node(modality_scale_node, name="modality_scale")

    guidance_skip_step_node = NumberNode(number=gen_settings.guidance_skip_step)
    graph.add_node(guidance_skip_step_node, name="guidance_skip_step")

    # --- Processing nodes ---
    prompt_encode_node = LTX2PromptEncodeNode()
    graph.add_node(prompt_encode_node, name="prompt_encode")

    latents_node = LTX2LatentsNode()
    graph.add_node(latents_node, name="latents")

    denoise_node = LTX2DenoiseNode()
    graph.add_node(denoise_node, name="denoise")

    decode_node = LTX2DecodeNode()
    graph.add_node(decode_node, name="decode")

    video_send_node = LTX2VideoSendNode()
    video_send_node.output_dir = str(directories.outputs_videos)
    video_send_node.video_codec = gen_settings.video_codec
    video_send_node.video_crf = gen_settings.video_crf
    video_send_node.video_preset = gen_settings.video_preset
    video_send_node.audio_codec = gen_settings.audio_codec
    video_send_node.audio_bitrate_kbps = gen_settings.audio_bitrate_kbps
    graph.add_node(video_send_node, name="video_send")

    # --- Connections: model_node → lora_node ---
    lora_node.connect("transformer", model_node, "transformer")
    lora_node.connect("transformer_component_name", model_node, "transformer_component_name")

    # --- Connections: model_node → torch.compile settings ---
    model_node.connect("use_torch_compile", use_torch_compile_node, "value")
    model_node.connect("torch_compile_max_autotune", torch_compile_max_autotune_node, "value")

    # --- Connections: model_node + text → prompt_encode ---
    prompt_encode_node.connect("tokenizer", model_node, "tokenizer")
    prompt_encode_node.connect("text_encoder", model_node, "text_encoder")
    prompt_encode_node.connect("connectors", model_node, "connectors")
    prompt_encode_node.connect("prompt", prompt_node, "value")
    prompt_encode_node.connect("negative_prompt", negative_prompt_node, "value")
    prompt_encode_node.connect("guidance_scale", guidance_node, "value")

    # --- Connections: lora_node + model_node + params → latents ---
    latents_node.connect("transformer", lora_node, "transformer")
    latents_node.connect("vae", model_node, "vae")
    latents_node.connect("audio_vae", model_node, "audio_vae")
    latents_node.connect("num_frames", num_frames_node, "value")
    latents_node.connect("height", height_node, "value")
    latents_node.connect("width", width_node, "value")
    latents_node.connect("frame_rate", frame_rate_node, "value")
    latents_node.connect("seed", seed_node, "value")

    # --- Connections: → denoise ---
    denoise_node.connect("transformer", lora_node, "transformer")
    denoise_node.connect("transformer_component_name", lora_node, "transformer_component_name")
    denoise_node.connect("scheduler_config", model_node, "scheduler_config")
    denoise_node.connect("prompt_embeds", prompt_encode_node, "prompt_embeds")
    denoise_node.connect("audio_prompt_embeds", prompt_encode_node, "audio_prompt_embeds")
    denoise_node.connect("attention_mask", prompt_encode_node, "attention_mask")
    denoise_node.connect("video_latents", latents_node, "video_latents")
    denoise_node.connect("audio_latents", latents_node, "audio_latents")
    denoise_node.connect("video_coords", latents_node, "video_coords")
    denoise_node.connect("audio_coords", latents_node, "audio_coords")
    denoise_node.connect("latent_num_frames", latents_node, "latent_num_frames")
    denoise_node.connect("latent_height", latents_node, "latent_height")
    denoise_node.connect("latent_width", latents_node, "latent_width")
    denoise_node.connect("audio_num_frames", latents_node, "audio_num_frames")
    denoise_node.connect("num_inference_steps", steps_node, "value")
    denoise_node.connect("guidance_scale", guidance_node, "value")
    denoise_node.connect("frame_rate", frame_rate_node, "value")
    denoise_node.connect("model_type", model_type_node, "value")
    denoise_node.connect("use_torch_compile", use_torch_compile_node, "value")
    denoise_node.connect("advanced_guidance", advanced_guidance_node, "value")
    denoise_node.connect("stg_scale", stg_scale_node, "value")
    denoise_node.connect("stg_blocks", stg_blocks_node, "value")
    denoise_node.connect("rescale_scale", rescale_scale_node, "value")
    denoise_node.connect("modality_scale", modality_scale_node, "value")
    denoise_node.connect("guidance_skip_step", guidance_skip_step_node, "value")

    # --- Connections: → decode ---
    decode_node.connect("vae", model_node, "vae")
    decode_node.connect("audio_vae", model_node, "audio_vae")
    decode_node.connect("vocoder", model_node, "vocoder")
    decode_node.connect("video_latents", denoise_node, "video_latents")
    decode_node.connect("audio_latents", denoise_node, "audio_latents")
    decode_node.connect("latent_num_frames", latents_node, "latent_num_frames")
    decode_node.connect("latent_height", latents_node, "latent_height")
    decode_node.connect("latent_width", latents_node, "latent_width")
    decode_node.connect("audio_num_frames", latents_node, "audio_num_frames")
    decode_node.connect("num_frames", num_frames_node, "value")
    decode_node.connect("frame_rate", frame_rate_node, "value")

    # --- Connections: decode → video_send ---
    video_send_node.connect("video", decode_node, "video")
    video_send_node.connect("audio", decode_node, "audio")
    video_send_node.connect("frame_rate_out", decode_node, "frame_rate_out")

    # --- Source image nodes (disabled by default, kept for backward compat) ---
    source_image_node = ImageLoadNode()
    source_image_node.enabled = False
    graph.add_node(source_image_node, name="source_image")

    image_encode_node = LTX2ImageEncodeNode()
    image_encode_node.enabled = False
    graph.add_node(image_encode_node, name="image_encode")

    # --- Connections: source image → image encode (legacy) ---
    image_encode_node.connect("vae", model_node, "vae")
    image_encode_node.connect("image", source_image_node, "image")
    image_encode_node.connect("num_frames", num_frames_node, "value")
    image_encode_node.connect("height", height_node, "value")
    image_encode_node.connect("width", width_node, "value")

    # --- Generalized condition encode node (disabled by default) ---
    condition_encode_node = LTX2ConditionEncodeNode()
    condition_encode_node.enabled = False
    graph.add_node(condition_encode_node, name="condition_encode")

    condition_encode_node.connect("vae", model_node, "vae")
    condition_encode_node.connect("num_frames", num_frames_node, "value")
    condition_encode_node.connect("height", height_node, "value")
    condition_encode_node.connect("width", width_node, "value")
    condition_encode_node.connect("frame_rate", frame_rate_node, "value")

    # --- Connections: condition encode → latents (new path) ---
    latents_node.connect("clean_latents", condition_encode_node, "clean_latents")
    latents_node.connect("clean_conditioning_mask", condition_encode_node, "conditioning_mask")

    # --- Connections: condition encode → latents (concat path) ---
    latents_node.connect("concat_latents", condition_encode_node, "concat_latents")
    latents_node.connect("concat_positions", condition_encode_node, "concat_positions")
    latents_node.connect("concat_conditioning_mask", condition_encode_node, "concat_conditioning_mask")

    # --- Connections: lora → condition_encode (downscale_factor for concat mode) ---
    condition_encode_node.connect("reference_downscale_factor", lora_node, "reference_downscale_factor")

    # --- Connections: legacy image encode → latents (backward compat) ---
    latents_node.connect("image_latents", image_encode_node, "image_latents")
    latents_node.connect("conditioning_mask", image_encode_node, "conditioning_mask")

    # --- Connections: latents → denoise (conditioning) ---
    denoise_node.connect("conditioning_mask", latents_node, "conditioning_mask")
    denoise_node.connect("clean_latents", latents_node, "clean_latents")
    denoise_node.connect("base_num_tokens", latents_node, "base_num_tokens")

    # --- Audio encode node (disabled by default) ---
    audio_encode_node = LTX2AudioEncodeNode()
    audio_encode_node.enabled = False
    graph.add_node(audio_encode_node, name="audio_encode")

    audio_encode_node.connect("audio_vae", model_node, "audio_vae")
    audio_encode_node.connect("num_frames", num_frames_node, "value")
    audio_encode_node.connect("frame_rate", frame_rate_node, "value")

    # --- Connections: audio encode → latents ---
    latents_node.connect("clean_audio_latents", audio_encode_node, "clean_audio_latents")
    latents_node.connect("audio_conditioning_mask", audio_encode_node, "audio_conditioning_mask")

    # --- Connections: latents → denoise (audio conditioning) ---
    denoise_node.connect("clean_audio_latents", latents_node, "clean_audio_latents")
    denoise_node.connect("audio_conditioning_mask", latents_node, "audio_conditioning_mask")

    # --- Connections: latents → decode (audio conditioning) ---
    decode_node.connect("clean_audio_latents", latents_node, "clean_audio_latents")
    decode_node.connect("audio_conditioning_mask", latents_node, "audio_conditioning_mask")

    # ===================================================================
    # 2nd pass (upsample) nodes — all disabled by default
    # ===================================================================

    # --- 2nd pass source nodes ---
    second_pass_model_node = LTX2ModelNode(
        model_path=gen_settings.second_pass_model.filepath,
        model_id=gen_settings.second_pass_model.id,
        offload_strategy=gen_settings.offload_strategy,
        group_offload_use_stream=gen_settings.group_offload_use_stream,
        group_offload_low_cpu_mem=gen_settings.group_offload_low_cpu_mem,
        component_prefix="sp_",
    )
    second_pass_model_node.enabled = False
    graph.add_node(second_pass_model_node, name="second_pass_model")

    second_pass_lora_node = LTX2LoraNode()
    second_pass_lora_node.enabled = False
    graph.add_node(second_pass_lora_node, name="second_pass_lora")

    second_pass_model_type_node = NumberNode(number=gen_settings.second_pass_model.model_type)
    graph.add_node(second_pass_model_type_node, name="second_pass_model_type")

    second_pass_steps_node = NumberNode(number=gen_settings.second_pass_steps)
    graph.add_node(second_pass_steps_node, name="second_pass_steps")

    second_pass_guidance_node = NumberNode(number=gen_settings.second_pass_guidance)
    graph.add_node(second_pass_guidance_node, name="second_pass_guidance")

    stage_node = NumberNode(number=2)
    graph.add_node(stage_node, name="stage")

    # --- 2nd pass processing nodes ---
    upsample_node = LTX2LatentUpsampleNode()
    upsample_node.enabled = False
    graph.add_node(upsample_node, name="upsample")

    second_pass_latents_node = LTX2SecondPassLatentsNode()
    second_pass_latents_node.enabled = False
    graph.add_node(second_pass_latents_node, name="second_pass_latents")

    second_pass_denoise_node = LTX2DenoiseNode()
    second_pass_denoise_node.enabled = False
    graph.add_node(second_pass_denoise_node, name="second_pass_denoise")

    # --- 2nd pass connections: second_pass_model_node ---
    second_pass_model_node.connect("use_torch_compile", use_torch_compile_node, "value")

    # --- 2nd pass connections: second_pass_lora_node ---
    second_pass_lora_node.connect("transformer", second_pass_model_node, "transformer")
    second_pass_lora_node.connect("transformer_component_name", second_pass_model_node, "transformer_component_name")

    # --- 2nd pass connections: upsample_node ---
    upsample_node.connect("video_latents", denoise_node, "video_latents")
    upsample_node.connect("vae", model_node, "vae")
    upsample_node.connect("latent_num_frames", latents_node, "latent_num_frames")
    upsample_node.connect("latent_height", latents_node, "latent_height")
    upsample_node.connect("latent_width", latents_node, "latent_width")

    # --- 2nd pass connections: second_pass_latents_node ---
    second_pass_latents_node.connect("transformer", second_pass_lora_node, "transformer")
    second_pass_latents_node.connect("vae", model_node, "vae")
    second_pass_latents_node.connect("audio_vae", model_node, "audio_vae")
    second_pass_latents_node.connect("video_latents", upsample_node, "video_latents")
    second_pass_latents_node.connect("audio_latents", denoise_node, "audio_latents")
    second_pass_latents_node.connect("audio_coords", latents_node, "audio_coords")
    second_pass_latents_node.connect("latent_num_frames", upsample_node, "latent_num_frames")
    second_pass_latents_node.connect("latent_height", upsample_node, "latent_height")
    second_pass_latents_node.connect("latent_width", upsample_node, "latent_width")
    second_pass_latents_node.connect("audio_num_frames", latents_node, "audio_num_frames")
    second_pass_latents_node.connect("frame_rate", frame_rate_node, "value")
    second_pass_latents_node.connect("seed", seed_node, "value")
    second_pass_latents_node.connect("model_type", second_pass_model_type_node, "value")
    second_pass_latents_node.connect("conditioning_mask", latents_node, "conditioning_mask")
    second_pass_latents_node.connect("clean_latents", latents_node, "clean_latents")
    second_pass_latents_node.connect("clean_audio_latents", latents_node, "clean_audio_latents")
    second_pass_latents_node.connect("audio_conditioning_mask", latents_node, "audio_conditioning_mask")

    # --- 2nd pass connections: second_pass_denoise_node ---
    second_pass_denoise_node.connect("transformer", second_pass_lora_node, "transformer")
    second_pass_denoise_node.connect("transformer_component_name", second_pass_lora_node, "transformer_component_name")
    second_pass_denoise_node.connect("scheduler_config", second_pass_model_node, "scheduler_config")
    second_pass_denoise_node.connect("prompt_embeds", prompt_encode_node, "prompt_embeds")
    second_pass_denoise_node.connect("audio_prompt_embeds", prompt_encode_node, "audio_prompt_embeds")
    second_pass_denoise_node.connect("attention_mask", prompt_encode_node, "attention_mask")
    second_pass_denoise_node.connect("video_latents", second_pass_latents_node, "video_latents")
    second_pass_denoise_node.connect("audio_latents", second_pass_latents_node, "audio_latents")
    second_pass_denoise_node.connect("video_coords", second_pass_latents_node, "video_coords")
    second_pass_denoise_node.connect("audio_coords", second_pass_latents_node, "audio_coords")
    second_pass_denoise_node.connect("latent_num_frames", second_pass_latents_node, "latent_num_frames")
    second_pass_denoise_node.connect("latent_height", second_pass_latents_node, "latent_height")
    second_pass_denoise_node.connect("latent_width", second_pass_latents_node, "latent_width")
    second_pass_denoise_node.connect("audio_num_frames", second_pass_latents_node, "audio_num_frames")
    second_pass_denoise_node.connect("num_inference_steps", second_pass_steps_node, "value")
    second_pass_denoise_node.connect("guidance_scale", second_pass_guidance_node, "value")
    second_pass_denoise_node.connect("frame_rate", frame_rate_node, "value")
    second_pass_denoise_node.connect("model_type", second_pass_model_type_node, "value")
    second_pass_denoise_node.connect("stage", stage_node, "value")
    second_pass_denoise_node.connect("conditioning_mask", second_pass_latents_node, "conditioning_mask")
    second_pass_denoise_node.connect("clean_latents", second_pass_latents_node, "clean_latents")
    second_pass_denoise_node.connect("base_num_tokens", second_pass_latents_node, "base_num_tokens")
    second_pass_denoise_node.connect("clean_audio_latents", second_pass_latents_node, "clean_audio_latents")
    second_pass_denoise_node.connect("audio_conditioning_mask", second_pass_latents_node, "audio_conditioning_mask")
    second_pass_denoise_node.connect("advanced_guidance", advanced_guidance_node, "value")
    second_pass_denoise_node.connect("stg_scale", stg_scale_node, "value")
    second_pass_denoise_node.connect("stg_blocks", stg_blocks_node, "value")
    second_pass_denoise_node.connect("rescale_scale", rescale_scale_node, "value")
    second_pass_denoise_node.connect("modality_scale", modality_scale_node, "value")
    second_pass_denoise_node.connect("guidance_skip_step", guidance_skip_step_node, "value")

    return graph
