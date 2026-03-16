from __future__ import annotations

import copy
import logging
from typing import ClassVar

import numpy as np

from frameartisan.modules.generation.constants import LTX2_DISTILLED_SIGMAS, LTX2_STAGE2_DISTILLED_SIGMAS
from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class _AbortedError(Exception):
    """Raised inside the denoising loop to interrupt generation."""


class LTX2DenoiseNode(Node):
    PRIORITY = 0
    REQUIRED_INPUTS: ClassVar[list[str]] = [
        "transformer",
        "scheduler_config",
        "prompt_embeds",
        "audio_prompt_embeds",
        "attention_mask",
        "video_latents",
        "audio_latents",
        "video_coords",
        "audio_coords",
        "latent_num_frames",
        "latent_height",
        "latent_width",
        "audio_num_frames",
        "num_inference_steps",
        "guidance_scale",
        "frame_rate",
    ]
    OPTIONAL_INPUTS: ClassVar[list[str]] = [
        "model_type",
        "conditioning_mask",
        "clean_latents",
        "clean_audio_latents",
        "audio_conditioning_mask",
        "stage",
        "transformer_component_name",
        "use_torch_compile",
        "advanced_guidance",
        "stg_scale",
        "stg_blocks",
        "rescale_scale",
        "modality_scale",
        "guidance_skip_step",
        "base_num_tokens",
        "keyframe_group_sizes",
        "keyframe_attention_scales",
    ]
    OUTPUTS: ClassVar[list[str]] = ["video_latents", "audio_latents"]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    def __init__(self):
        super().__init__()
        self.callback = None
        self.on_start_callback = None
        self.status_callback = None
        # Exposed for live preview: set during __call__ so callers can unpack latents.
        self._latent_height: int = 0
        self._latent_width: int = 0
        self._latent_num_frames: int = 0

    def __call__(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler

        from frameartisan.app.model_manager import get_model_manager
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            calculate_shift,
            pack_latents,
            unpack_latents,
        )

        transformer = self.transformer
        scheduler_config = self.scheduler_config
        prompt_embeds = self.prompt_embeds
        audio_prompt_embeds = self.audio_prompt_embeds
        attention_mask = self.attention_mask
        video_latents = self.video_latents
        audio_latents = self.audio_latents
        video_coords = self.video_coords
        audio_coords = self.audio_coords
        latent_num_frames = int(self.latent_num_frames)
        latent_height = int(self.latent_height)
        latent_width = int(self.latent_width)
        audio_num_frames = int(self.audio_num_frames)
        num_inference_steps = int(self.num_inference_steps)
        guidance_scale = float(self.guidance_scale)
        frame_rate = float(self.frame_rate)
        model_type = int(self.model_type) if self.model_type is not None else None
        conditioning_mask = self.conditioning_mask  # [B, seq_len] or None
        clean_latents = self.clean_latents  # [B, seq_len, feat] or None (for x0 blending)
        clean_audio_latents = self.clean_audio_latents  # packed audio latents or None
        audio_conditioning_mask = self.audio_conditioning_mask  # [B, seq_len] or None
        stage = int(self.stage) if self.stage is not None else 1
        transformer_component_name = self.transformer_component_name or "transformer"

        # Advanced guidance params (None → disabled)
        advanced_guidance = bool(self.advanced_guidance) if self.advanced_guidance is not None else False
        stg_scale = float(self.stg_scale) if self.stg_scale is not None else 0.0
        stg_blocks_str = str(self.stg_blocks) if self.stg_blocks is not None else "29"
        rescale_scale = float(self.rescale_scale) if self.rescale_scale is not None else 0.0
        modality_scale = float(self.modality_scale) if self.modality_scale is not None else 1.0
        guidance_skip_step = int(self.guidance_skip_step) if self.guidance_skip_step is not None else 0

        # Expose spatial dims for live latent preview
        self._latent_height = latent_height
        self._latent_width = latent_width
        self._latent_num_frames = latent_num_frames

        if transformer is None:
            raise NodeError("transformer input is not connected", self.__class__.__name__)

        device = self.device

        # Distilled model: force fixed parameters and sigma schedule
        sigmas = None
        if model_type == 2:
            if stage == 2:
                num_inference_steps = 3
                guidance_scale = 1.0
                sigmas = LTX2_STAGE2_DISTILLED_SIGMAS
            else:
                num_inference_steps = 8
                guidance_scale = 1.0
                sigmas = LTX2_DISTILLED_SIGMAS
            advanced_guidance = False

        # The pipeline's prepare_latents always returns float32 latents.
        # Cast here to match and avoid bf16 precision loss in the scheduler step.
        video_latents = video_latents.to(device=device, dtype=torch.float32)
        audio_latents = audio_latents.to(device=device, dtype=torch.float32)

        # Move embeddings and clean latents to device
        prompt_embeds = prompt_embeds.to(device=device)
        audio_prompt_embeds = audio_prompt_embeds.to(device=device)
        attention_mask = attention_mask.to(device=device)
        if clean_latents is not None:
            clean_latents = clean_latents.to(device=device, dtype=torch.float32)
        if clean_audio_latents is not None:
            clean_audio_latents = clean_audio_latents.to(device=device, dtype=torch.float32)
        if audio_conditioning_mask is not None:
            audio_conditioning_mask = audio_conditioning_mask.to(device=device, dtype=torch.float32)

        # Determine CFG: prompt_encode may produce batch=2 [negative, positive].
        # Use CFG only if both conditions are met.
        embeds_have_cfg = prompt_embeds.shape[0] == 2
        do_cfg = embeds_have_cfg and guidance_scale > 1.0

        # If embeddings have CFG but we don't want it, take only the positive half
        if embeds_have_cfg and not do_cfg:
            prompt_embeds = prompt_embeds[1:2]
            audio_prompt_embeds = audio_prompt_embeds[1:2]
            attention_mask = attention_mask[1:2]

        # Create scheduler from config — stage 2 uses modified config
        if stage == 2:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                scheduler_config, use_dynamic_shifting=False, shift_terminal=None
            )
        else:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        audio_scheduler = copy.deepcopy(scheduler)

        # Prepare sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        # Calculate resolution-dependent shift
        video_sequence_length = latent_num_frames * latent_height * latent_width
        mu = calculate_shift(
            video_sequence_length,
            scheduler.config.get("base_image_seq_len", 1024),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.95),
            scheduler.config.get("max_shift", 2.05),
        )

        # Set timesteps on both schedulers
        audio_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = scheduler.timesteps

        # --- Advanced guidance: separate forward passes per component ---
        if advanced_guidance:
            self._run_advanced_guidance_loop(
                transformer=transformer,
                prompt_embeds=prompt_embeds,
                audio_prompt_embeds=audio_prompt_embeds,
                attention_mask=attention_mask,
                video_latents=video_latents,
                audio_latents=audio_latents,
                video_coords=video_coords,
                audio_coords=audio_coords,
                conditioning_mask=conditioning_mask,
                clean_latents=clean_latents,
                clean_audio_latents=clean_audio_latents,
                audio_conditioning_mask=audio_conditioning_mask,
                latent_num_frames=latent_num_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                audio_num_frames=audio_num_frames,
                frame_rate=frame_rate,
                guidance_scale=guidance_scale,
                stg_scale=stg_scale,
                stg_blocks_str=stg_blocks_str,
                rescale_scale=rescale_scale,
                modality_scale=modality_scale,
                guidance_skip_step=guidance_skip_step,
                embeds_have_cfg=embeds_have_cfg,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
                timesteps=timesteps,
                transformer_component_name=transformer_component_name,
                device=device,
            )
            return self.values

        # --- Standard batch=2 CFG path (unchanged) ---
        # Duplicate latents/coords/mask for CFG
        if do_cfg:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
            if conditioning_mask is not None:
                conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])
            if audio_conditioning_mask is not None:
                audio_conditioning_mask = torch.cat([audio_conditioning_mask, audio_conditioning_mask])

        # --- Keyframe self-attention mask ---
        keyframe_group_sizes = self.keyframe_group_sizes
        keyframe_hooks: list[tuple] | None = None

        logger.info(
            "LTX2 denoising: %d steps, guidance_scale=%.1f, seq_len=%d",
            len(timesteps),
            guidance_scale,
            video_sequence_length,
        )

        mm = get_model_manager()

        if self.on_start_callback is not None:
            self.on_start_callback(len(timesteps))

        try:
            with mm.use_components(transformer_component_name, device=device):
                # Install keyframe attention mask after transformer is on device
                if keyframe_group_sizes:
                    keyframe_hooks = self._install_keyframe_attention(
                        transformer, video_latents, keyframe_group_sizes,
                        self.keyframe_attention_scales, do_cfg, device,
                    )
                for i, t in enumerate(timesteps):
                    if i == 0 and self.use_torch_compile and self.status_callback is not None:
                        self.status_callback("Compiling transformer (this may take a while)...")
                    # Prepare model input (duplicate for CFG)
                    latent_model_input = torch.cat([video_latents] * 2) if do_cfg else video_latents
                    latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                    # Audio input: when mask is present, use blended audio_latents (from latents node).
                    # When no mask but clean audio exists, pass clean audio directly (full conditioning).
                    if audio_conditioning_mask is not None:
                        audio_input = audio_latents
                    elif clean_audio_latents is not None:
                        audio_input = clean_audio_latents
                    else:
                        audio_input = audio_latents
                    audio_model_input = torch.cat([audio_input] * 2) if do_cfg else audio_input
                    audio_model_input = audio_model_input.to(prompt_embeds.dtype)

                    timestep = t.expand(latent_model_input.shape[0])

                    # Timestep masking for conditioning:
                    # Conditioned tokens see timestep=0 (fully denoised)
                    audio_timestep_kwarg = {}
                    if conditioning_mask is not None:
                        cm = conditioning_mask.to(device=device, dtype=timestep.dtype)
                        # timestep: [B] → [B, 1]; cm: [B, seq_len]
                        video_timestep = timestep.unsqueeze(-1) * (1 - cm)
                        # Audio uses the original scalar timestep (different token count)
                        audio_timestep_kwarg["audio_timestep"] = timestep
                    else:
                        video_timestep = timestep

                    # Audio timestep masking: per-token for partial conditioning,
                    # zeros for full conditioning, scalar for no conditioning
                    if audio_conditioning_mask is not None:
                        acm = audio_conditioning_mask[: audio_model_input.shape[0]]
                        audio_timestep_kwarg["audio_timestep"] = timestep.unsqueeze(-1) * (1 - acm)
                    elif clean_audio_latents is not None:
                        audio_timestep_kwarg["audio_timestep"] = torch.zeros_like(timestep)

                    with transformer.cache_context("cond_uncond"):
                        noise_pred_video, noise_pred_audio = transformer(
                            hidden_states=latent_model_input,
                            audio_hidden_states=audio_model_input,
                            encoder_hidden_states=prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep,
                            encoder_attention_mask=attention_mask,
                            audio_encoder_attention_mask=attention_mask,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_coords,
                            audio_coords=audio_coords,
                            return_dict=False,
                            **audio_timestep_kwarg,
                        )

                    noise_pred_video = noise_pred_video.float()
                    noise_pred_audio = noise_pred_audio.float()

                    # Apply classifier-free guidance
                    if do_cfg:
                        noise_pred_video_uncond, noise_pred_video_text = noise_pred_video.chunk(2)
                        noise_pred_video = noise_pred_video_uncond + guidance_scale * (
                            noise_pred_video_text - noise_pred_video_uncond
                        )
                        noise_pred_audio_uncond, noise_pred_audio_text = noise_pred_audio.chunk(2)
                        noise_pred_audio = noise_pred_audio_uncond + guidance_scale * (
                            noise_pred_audio_text - noise_pred_audio_uncond
                        )

                    # Predicted clean sample (x0) for preview callback
                    # For flow matching: x0 = latents - sigma * velocity
                    pred_x0 = video_latents - t * noise_pred_video

                    # Scheduler step
                    if conditioning_mask is not None and clean_latents is not None:
                        # Generalized x0-space blending
                        sigma = t
                        batch_size = video_latents.shape[0]
                        # Blend x0 with clean latents using mask
                        cm_batch = conditioning_mask[:batch_size]
                        mask_3d = cm_batch.unsqueeze(-1)  # [B, seq, 1]
                        x0_blended = pred_x0 * (1 - mask_3d) + clean_latents[:batch_size] * mask_3d
                        # x0 → blended velocity
                        noise_pred_blended = (video_latents - x0_blended) / sigma
                        video_latents = scheduler.step(noise_pred_blended, t, video_latents, return_dict=False)[0]
                    elif conditioning_mask is not None:
                        # Legacy path: unpack → preserve frame 0 → step 1+ → repack
                        patch_size = getattr(transformer.config, "patch_size", 1)
                        patch_size_t = getattr(transformer.config, "patch_size_t", 1)

                        noise_pred_video = unpack_latents(
                            noise_pred_video, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
                        )
                        video_latents = unpack_latents(
                            video_latents, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
                        )

                        noise_pred_frames = noise_pred_video[:, :, 1:]
                        noise_latents = video_latents[:, :, 1:]
                        pred_latents = scheduler.step(noise_pred_frames, t, noise_latents, return_dict=False)[0]

                        video_latents = torch.cat([video_latents[:, :, :1], pred_latents], dim=2)
                        video_latents = pack_latents(video_latents, patch_size, patch_size_t)
                    else:
                        video_latents = scheduler.step(noise_pred_video, t, video_latents, return_dict=False)[0]

                    # Audio scheduler step
                    if audio_conditioning_mask is not None and clean_audio_latents is not None:
                        # Partial audio conditioning: x0-space blending (mirrors video pattern)
                        sigma = t
                        batch_size = audio_latents.shape[0]
                        x0_audio = audio_latents - sigma * noise_pred_audio
                        acm_batch = audio_conditioning_mask[:batch_size]
                        acm_3d = acm_batch.unsqueeze(-1)  # [B, seq, 1]
                        x0_audio_blended = x0_audio * (1 - acm_3d) + clean_audio_latents[:batch_size] * acm_3d
                        blended_pred_audio = (audio_latents - x0_audio_blended) / sigma
                        audio_latents = audio_scheduler.step(blended_pred_audio, t, audio_latents, return_dict=False)[
                            0
                        ]
                    elif clean_audio_latents is None:
                        # No audio conditioning: normal denoise
                        audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]
                    # else: full audio conditioning (mask=None, clean present) — skip, audio stays constant

                    # Progress callback — use pred_x0 for early steps (cleaner preview),
                    # switch to video_latents for the last 2 steps (converged, no overshoot)
                    if self.callback is not None:
                        preview = video_latents if i >= len(timesteps) - 2 else pred_x0
                        self.callback(i, t, preview)

                    # Clear compile status after first step (compilation is done)
                    if i == 0 and self.use_torch_compile and self.status_callback is not None:
                        self.status_callback("")

                    # Abort check
                    if self.abort:
                        raise _AbortedError

        except _AbortedError:
            self.abort = True
            self._cleanup_keyframe_hooks(keyframe_hooks)
            video_latents = self._strip_concat_tokens(video_latents)
            self.values = {
                "video_latents": video_latents,
                "audio_latents": audio_latents,
            }
            return self.values
        except Exception as e:
            self._cleanup_keyframe_hooks(keyframe_hooks)
            raise NodeError(f"LTX2 denoising failed: {e}", self.__class__.__name__) from e

        self._cleanup_keyframe_hooks(keyframe_hooks)
        video_latents = self._strip_concat_tokens(video_latents)
        self.values = {
            "video_latents": video_latents,
            "audio_latents": audio_latents,
        }
        return self.values

    def _strip_concat_tokens(self, video_latents):
        """Remove appended concat/keyframe tokens if base_num_tokens is set."""
        base_num_tokens = self.base_num_tokens
        if base_num_tokens is not None:
            video_latents = video_latents[:, : int(base_num_tokens), :]
        return video_latents

    @staticmethod
    def _install_keyframe_attention(
        transformer, video_latents, keyframe_group_sizes, keyframe_attention_scales, do_cfg, device
    ):
        """Build and install the keyframe self-attention mask on the transformer."""
        import torch

        from frameartisan.modules.generation.graph.nodes.ltx2_keyframe_attention import (
            install_keyframe_mask,
        )
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            build_keyframe_attention_mask,
        )

        total_tokens = video_latents.shape[1]
        num_keyframe_tokens = sum(keyframe_group_sizes)
        num_base_tokens = total_tokens - num_keyframe_tokens

        # Use the transformer's compute dtype so the mask matches query dtype in SDPA
        model_dtype = next(transformer.parameters()).dtype

        mask = build_keyframe_attention_mask(
            num_base_tokens=num_base_tokens,
            keyframe_group_sizes=keyframe_group_sizes,
            attention_scales=keyframe_attention_scales,
            dtype=model_dtype,
            device=device,
        )
        if mask is None:
            return None

        # Expand for CFG batch if needed
        batch_size = 2 if do_cfg else 1
        if batch_size > 1:
            mask = mask.expand(batch_size, -1, -1).contiguous()

        logger.info(
            "Keyframe attention mask installed: %d base tokens, %d keyframe tokens (%d groups), mask shape %s",
            num_base_tokens,
            num_keyframe_tokens,
            len(keyframe_group_sizes),
            mask.shape,
        )

        return install_keyframe_mask(transformer, mask)

    @staticmethod
    def _cleanup_keyframe_hooks(hooks):
        """Remove keyframe attention mask hooks if installed."""
        if hooks is not None:
            from frameartisan.modules.generation.graph.nodes.ltx2_keyframe_attention import (
                remove_keyframe_mask,
            )

            remove_keyframe_mask(hooks)

    def _run_advanced_guidance_loop(
        self,
        *,
        transformer,
        prompt_embeds,
        audio_prompt_embeds,
        attention_mask,
        video_latents,
        audio_latents,
        video_coords,
        audio_coords,
        conditioning_mask,
        clean_latents,
        clean_audio_latents,
        audio_conditioning_mask,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        audio_num_frames: int,
        frame_rate: float,
        guidance_scale: float,
        stg_scale: float,
        stg_blocks_str: str,
        rescale_scale: float,
        modality_scale: float,
        guidance_skip_step: int,
        embeds_have_cfg: bool,
        scheduler,
        audio_scheduler,
        timesteps,
        transformer_component_name: str,
        device,
    ) -> None:
        """Advanced guidance loop with separate forward passes for CFG, STG, modality isolation."""
        import torch

        from frameartisan.app.model_manager import get_model_manager
        from frameartisan.modules.generation.graph.nodes.ltx2_guidance import (
            calculate_guidance,
            modality_isolation_hooks,
            parse_stg_blocks,
            stg_perturbation_hooks,
        )
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            pack_latents,
            unpack_latents,
        )

        # Split embeddings: [negative, positive] → separate tensors
        if embeds_have_cfg:
            neg_prompt_embeds = prompt_embeds[0:1]
            pos_prompt_embeds = prompt_embeds[1:2]
            neg_audio_prompt_embeds = audio_prompt_embeds[0:1]
            pos_audio_prompt_embeds = audio_prompt_embeds[1:2]
            neg_attention_mask = attention_mask[0:1]
            pos_attention_mask = attention_mask[1:2]
        else:
            pos_prompt_embeds = prompt_embeds
            pos_audio_prompt_embeds = audio_prompt_embeds
            pos_attention_mask = attention_mask
            neg_prompt_embeds = None
            neg_audio_prompt_embeds = None
            neg_attention_mask = None

        do_cfg = guidance_scale != 1.0 and neg_prompt_embeds is not None
        do_stg = stg_scale != 0.0
        do_modality = modality_scale != 1.0

        stg_block_indices = parse_stg_blocks(stg_blocks_str) if do_stg else []

        video_sequence_length = latent_num_frames * latent_height * latent_width
        logger.info(
            "LTX2 advanced guidance: %d steps, cfg=%.1f, stg=%.1f, modality=%.1f, rescale=%.2f, seq_len=%d",
            len(timesteps),
            guidance_scale,
            stg_scale,
            modality_scale,
            rescale_scale,
            video_sequence_length,
        )

        mm = get_model_manager()

        if self.on_start_callback is not None:
            self.on_start_callback(len(timesteps))

        # Cache for skip-step reuse
        last_noise_pred_video = None
        last_noise_pred_audio = None

        # Keyframe attention mask for advanced guidance
        keyframe_group_sizes = self.keyframe_group_sizes
        keyframe_hooks: list[tuple] | None = None

        def _transformer_forward(v_input, a_input, p_embeds, a_p_embeds, attn_mask, video_ts, audio_ts_kwarg):
            with transformer.cache_context("cond_uncond"):
                pred_v, pred_a = transformer(
                    hidden_states=v_input,
                    audio_hidden_states=a_input,
                    encoder_hidden_states=p_embeds,
                    audio_encoder_hidden_states=a_p_embeds,
                    timestep=video_ts,
                    encoder_attention_mask=attn_mask,
                    audio_encoder_attention_mask=attn_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    return_dict=False,
                    **audio_ts_kwarg,
                )
            return pred_v.float(), pred_a.float()

        try:
            with mm.use_components(transformer_component_name, device=device):
                # Install keyframe attention mask (batch_size=1 for advanced guidance)
                if keyframe_group_sizes:
                    keyframe_hooks = self._install_keyframe_attention(
                        transformer, video_latents, keyframe_group_sizes, False, device
                    )
                for i, t in enumerate(timesteps):
                    if i == 0 and self.use_torch_compile and self.status_callback is not None:
                        self.status_callback("Compiling transformer (this may take a while)...")

                    # Skip step logic: reuse last prediction
                    if (
                        guidance_skip_step > 0
                        and i > 0
                        and i % (guidance_skip_step + 1) != 0
                        and last_noise_pred_video is not None
                    ):
                        noise_pred_video = last_noise_pred_video
                        noise_pred_audio = last_noise_pred_audio
                    else:
                        # Prepare single-batch inputs
                        v_input = video_latents.to(pos_prompt_embeds.dtype)
                        # Audio input: blended for partial conditioning, clean for full, noisy otherwise
                        if audio_conditioning_mask is not None:
                            audio_input = audio_latents
                        elif clean_audio_latents is not None:
                            audio_input = clean_audio_latents
                        else:
                            audio_input = audio_latents
                        a_input = audio_input.to(pos_prompt_embeds.dtype)
                        timestep = t.expand(v_input.shape[0])

                        audio_timestep_kwarg = {}
                        if conditioning_mask is not None:
                            cm = conditioning_mask.to(device=device, dtype=timestep.dtype)
                            video_timestep = timestep.unsqueeze(-1) * (1 - cm)
                            audio_timestep_kwarg["audio_timestep"] = timestep
                        else:
                            video_timestep = timestep

                        # Audio timestep masking: per-token for partial, zeros for full, scalar for none
                        if audio_conditioning_mask is not None:
                            acm = audio_conditioning_mask[: v_input.shape[0]]
                            audio_timestep_kwarg["audio_timestep"] = timestep.unsqueeze(-1) * (1 - acm)
                        elif clean_audio_latents is not None:
                            audio_timestep_kwarg["audio_timestep"] = torch.zeros_like(timestep)

                        # 1. Conditional forward (always)
                        cond_video, cond_audio = _transformer_forward(
                            v_input,
                            a_input,
                            pos_prompt_embeds,
                            pos_audio_prompt_embeds,
                            pos_attention_mask,
                            video_timestep,
                            audio_timestep_kwarg,
                        )

                        # 2. Unconditional forward (if CFG)
                        if do_cfg:
                            uncond_video, uncond_audio = _transformer_forward(
                                v_input,
                                a_input,
                                neg_prompt_embeds,
                                neg_audio_prompt_embeds,
                                neg_attention_mask,
                                video_timestep,
                                audio_timestep_kwarg,
                            )
                        else:
                            uncond_video, uncond_audio = 0.0, 0.0

                        # 3. Perturbed forward (if STG)
                        if do_stg:
                            with stg_perturbation_hooks(transformer, stg_block_indices):
                                perturbed_video, perturbed_audio = _transformer_forward(
                                    v_input,
                                    a_input,
                                    pos_prompt_embeds,
                                    pos_audio_prompt_embeds,
                                    pos_attention_mask,
                                    video_timestep,
                                    audio_timestep_kwarg,
                                )
                        else:
                            perturbed_video, perturbed_audio = 0.0, 0.0

                        # 4. Modality-isolated forward (if modality guidance)
                        if do_modality:
                            with modality_isolation_hooks(transformer):
                                isolated_video, isolated_audio = _transformer_forward(
                                    v_input,
                                    a_input,
                                    pos_prompt_embeds,
                                    pos_audio_prompt_embeds,
                                    pos_attention_mask,
                                    video_timestep,
                                    audio_timestep_kwarg,
                                )
                        else:
                            isolated_video, isolated_audio = 0.0, 0.0

                        # Combine guidance
                        noise_pred_video, noise_pred_audio = calculate_guidance(
                            cond_video,
                            cond_audio,
                            uncond_video,
                            uncond_audio,
                            perturbed_video,
                            perturbed_audio,
                            isolated_video,
                            isolated_audio,
                            cfg_scale=guidance_scale,
                            stg_scale=stg_scale,
                            modality_scale=modality_scale,
                            rescale_scale=rescale_scale,
                        )

                        last_noise_pred_video = noise_pred_video
                        last_noise_pred_audio = noise_pred_audio

                    # Predicted clean sample (x0) for preview callback
                    pred_x0 = video_latents - t * noise_pred_video

                    # Scheduler step
                    if conditioning_mask is not None and clean_latents is not None:
                        # Generalized x0-space blending
                        sigma = t
                        mask_3d = conditioning_mask.unsqueeze(-1)  # [B, seq, 1]
                        x0_blended = pred_x0 * (1 - mask_3d) + clean_latents * mask_3d
                        noise_pred_blended = (video_latents - x0_blended) / sigma
                        video_latents = scheduler.step(noise_pred_blended, t, video_latents, return_dict=False)[0]
                    elif conditioning_mask is not None:
                        # Legacy path
                        patch_size = getattr(transformer.config, "patch_size", 1)
                        patch_size_t = getattr(transformer.config, "patch_size_t", 1)

                        noise_pred_video = unpack_latents(
                            noise_pred_video, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
                        )
                        video_latents = unpack_latents(
                            video_latents, latent_num_frames, latent_height, latent_width, patch_size, patch_size_t
                        )

                        noise_pred_frames = noise_pred_video[:, :, 1:]
                        noise_latents = video_latents[:, :, 1:]
                        pred_latents = scheduler.step(noise_pred_frames, t, noise_latents, return_dict=False)[0]

                        video_latents = torch.cat([video_latents[:, :, :1], pred_latents], dim=2)
                        video_latents = pack_latents(video_latents, patch_size, patch_size_t)
                    else:
                        video_latents = scheduler.step(noise_pred_video, t, video_latents, return_dict=False)[0]

                    # Audio scheduler step
                    if audio_conditioning_mask is not None and clean_audio_latents is not None:
                        # Partial audio conditioning: x0-space blending
                        sigma = t
                        x0_audio = audio_latents - sigma * noise_pred_audio
                        acm_3d = audio_conditioning_mask.unsqueeze(-1)  # [B, seq, 1]
                        x0_audio_blended = x0_audio * (1 - acm_3d) + clean_audio_latents * acm_3d
                        blended_pred_audio = (audio_latents - x0_audio_blended) / sigma
                        audio_latents = audio_scheduler.step(blended_pred_audio, t, audio_latents, return_dict=False)[
                            0
                        ]
                    elif clean_audio_latents is None:
                        audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]
                    # else: full audio conditioning — skip

                    if self.callback is not None:
                        preview = video_latents if i >= len(timesteps) - 2 else pred_x0
                        self.callback(i, t, preview)

                    if i == 0 and self.use_torch_compile and self.status_callback is not None:
                        self.status_callback("")

                    if self.abort:
                        raise _AbortedError

        except _AbortedError:
            self.abort = True
            self._cleanup_keyframe_hooks(keyframe_hooks)
            video_latents = self._strip_concat_tokens(video_latents)
            self.values = {
                "video_latents": video_latents,
                "audio_latents": audio_latents,
            }
            return
        except Exception as e:
            self._cleanup_keyframe_hooks(keyframe_hooks)
            raise NodeError(f"LTX2 advanced guidance denoising failed: {e}", self.__class__.__name__) from e

        self._cleanup_keyframe_hooks(keyframe_hooks)
        video_latents = self._strip_concat_tokens(video_latents)
        self.values = {
            "video_latents": video_latents,
            "audio_latents": audio_latents,
        }
