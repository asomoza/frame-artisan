from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)

# Key renames applied when converting ComfyUI/non-diffusers LoRA keys to diffusers format.
_DIFFUSION_MODEL_RENAMES: dict[str, str] = {
    "patchify_proj": "proj_in",
    "audio_patchify_proj": "audio_proj_in",
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

_CONNECTOR_RENAMES: dict[str, str] = {
    "aggregate_embed": "text_proj_in",
}


def convert_non_diffusers_lora(
    state_dict: dict,
    non_diffusers_prefix: str = "diffusion_model",
) -> dict:
    """Convert ComfyUI / non-diffusers LoRA keys to diffusers format.

    Handles both ``diffusion_model.*`` (transformer) and
    ``text_embedding_projection.*`` (connectors) prefixes.
    """
    # Filter and strip the prefix
    prefix_dot = f"{non_diffusers_prefix}."
    stripped = {k.removeprefix(prefix_dot): v for k, v in state_dict.items() if k.startswith(prefix_dot)}

    # Choose rename table
    renames = _DIFFUSION_MODEL_RENAMES if non_diffusers_prefix == "diffusion_model" else _CONNECTOR_RENAMES

    # Apply renames
    renamed = {}
    for key, value in stripped.items():
        new_key = key
        for old_pat, new_pat in renames.items():
            new_key = new_key.replace(old_pat, new_pat)
        renamed[new_key] = value

    # Handle adaln_single → time_embed / audio_adaln_single → audio_time_embed
    final = {}
    for key, value in renamed.items():
        if key.startswith("adaln_single."):
            key = key.replace("adaln_single.", "time_embed.", 1)
        elif key.startswith("audio_adaln_single."):
            key = key.replace("audio_adaln_single.", "audio_time_embed.", 1)
        final[key] = value

    # Add destination prefix
    dest_prefix = "transformer" if non_diffusers_prefix == "diffusion_model" else "connectors"
    return {f"{dest_prefix}.{k}": v for k, v in final.items()}


def _load_lora_state_dict(
    filepath: str,
    video_strength: float = 1.0,
    audio_strength: float = 1.0,
) -> dict:
    """Load a LoRA state dict, converting non-diffusers format if needed.

    LoRA files from ComfyUI and other tools use ``diffusion_model.`` as the
    key prefix (e.g. ``diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight``).
    The diffusers ``load_lora_adapter`` expects a ``transformer.`` prefix instead.

    This function detects the format and applies the same key conversion that
    ``LTX2Pipeline.load_lora_weights`` performs internally.

    When *video_strength* or *audio_strength* differ from 1.0, the corresponding
    LoRA weights are scaled (via lora_A) or removed (when strength == 0).
    """
    from safetensors.torch import load_file

    state_dict = load_file(filepath)

    is_non_diffusers = any(k.startswith("diffusion_model.") for k in state_dict)
    if not is_non_diffusers:
        state_dict = dict(state_dict)
    else:
        converted = convert_non_diffusers_lora(state_dict)

        # Also convert connectors keys if present
        has_connector = any(k.startswith("text_embedding_projection.") for k in state_dict)
        if has_connector:
            converted.update(convert_non_diffusers_lora(state_dict, "text_embedding_projection"))

        logger.info("Converted non-diffusers LoRA format (diffusion_model → transformer)")
        state_dict = converted

    if video_strength != 1.0 or audio_strength != 1.0:
        state_dict = _apply_strength_filtering(state_dict, video_strength, audio_strength)

    return state_dict


_AUDIO_PATTERNS = ("video_to_audio_attn", "audio_to_video_attn", "audio_attn", "audio_ff.net")


def _apply_strength_filtering(state_dict: dict, video_strength: float, audio_strength: float) -> dict:
    """Scale or remove LoRA keys based on audio/video strength."""
    filtered = {}
    for key, tensor in state_dict.items():
        is_audio = any(pat in key for pat in _AUDIO_PATTERNS)
        strength = audio_strength if is_audio else video_strength

        if strength == 0.0:
            continue

        if strength != 1.0 and "lora_A" in key:
            tensor = tensor * strength

        filtered[key] = tensor
    return filtered


class LTX2LoraNode(Node):
    PRIORITY = 1
    OUTPUTS = ["transformer", "transformer_component_name", "reference_downscale_factor", "lora_masks"]
    REQUIRED_INPUTS: ClassVar[list] = ["transformer"]
    OPTIONAL_INPUTS: ClassVar[list] = ["transformer_component_name"]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = {"lora_configs"}

    def __init__(self, lora_configs: list[dict] | None = None):
        super().__init__()
        self.lora_configs: list[dict] = lora_configs or []
        self._loaded_adapters: dict[
            str, tuple[str, float, float]
        ] = {}  # adapter_name → (filepath, video_str, audio_str)

    @staticmethod
    def _read_reference_downscale_factor(filepath: str) -> int:
        """Read reference_downscale_factor from safetensors metadata, default 1."""
        try:
            from safetensors import safe_open

            with safe_open(filepath, framework="pt") as f:
                metadata = f.metadata()
            if metadata and "reference_downscale_factor" in metadata:
                return int(metadata["reference_downscale_factor"])
        except Exception:
            pass
        return 1

    @staticmethod
    def _resolve_lora_filepath(cfg: dict) -> str:
        """Look up LoRA filepath from DB by id, falling back to config filepath."""
        lora_id = cfg.get("id")
        if lora_id:
            try:
                from frameartisan.app.app import get_app_database_path
                from frameartisan.utils.database import Database

                db = Database(get_app_database_path())
                row = db.select_one("lora", ["filepath"], {"id": lora_id, "deleted": 0})
                if row:
                    return row["filepath"]
            except Exception:
                pass
        return cfg.get("filepath", "")

    def __call__(self):
        from frameartisan.app.model_manager import get_model_manager

        transformer = self.transformer
        transformer_component_name = self.transformer_component_name or "transformer"

        # Build desired adapter state from enabled configs
        desired: dict[str, dict] = {}
        for cfg in self.lora_configs:
            if not cfg.get("enabled", True):
                continue
            name = cfg.get("hash") or cfg.get("name", "")
            if not name:
                continue
            desired[name] = cfg

        # Remove adapters that are no longer desired
        for adapter_name in list(self._loaded_adapters):
            if adapter_name not in desired:
                try:
                    transformer.delete_adapters(adapter_name)
                    logger.info("Unloaded LoRA adapter: %s", adapter_name)
                except Exception as e:
                    logger.warning("Failed to delete adapter '%s': %s", adapter_name, e)
                del self._loaded_adapters[adapter_name]

        if not desired:
            # No LoRAs active — disable any remaining adapters
            if self._loaded_adapters:
                try:
                    transformer.disable_adapters()
                except Exception:
                    pass
                self._loaded_adapters.clear()
            self.values = {
                "transformer": transformer,
                "transformer_component_name": transformer_component_name,
                "reference_downscale_factor": 1,
                "lora_masks": None,
            }
            return self.values

        # Load new adapters
        mm = get_model_manager()
        with mm.use_components(transformer_component_name, device=self.device):
            for adapter_name, cfg in desired.items():
                filepath = self._resolve_lora_filepath(cfg)
                video_str = float(cfg.get("video_strength", 1.0))
                audio_str = float(cfg.get("audio_strength", 1.0))
                load_key = (filepath, video_str, audio_str)
                # Check if adapter was removed externally (e.g. by other stage's LoRA node)
                if adapter_name in self._loaded_adapters:
                    peft_config = getattr(transformer, "peft_config", None)
                    if peft_config is None or (isinstance(peft_config, dict) and adapter_name not in peft_config):
                        logger.info("Adapter '%s' was removed externally, will reload", adapter_name)
                        del self._loaded_adapters[adapter_name]

                if adapter_name not in self._loaded_adapters:
                    # Check if adapter already exists on transformer (loaded by other stage's node)
                    peft_cfg = getattr(transformer, "peft_config", None)
                    if isinstance(peft_cfg, dict) and adapter_name in peft_cfg:
                        self._loaded_adapters[adapter_name] = load_key
                        logger.info("Adopted existing adapter '%s' from shared transformer", adapter_name)
                        continue
                    if not filepath:
                        continue
                    try:
                        state_dict = _load_lora_state_dict(filepath, video_str, audio_str)
                        transformer.load_lora_adapter(state_dict, adapter_name=adapter_name)
                        del state_dict
                        self._loaded_adapters[adapter_name] = load_key
                        logger.info("Loaded LoRA adapter: %s from %s", adapter_name, filepath)
                    except Exception as e:
                        raise NodeError(
                            f"Failed to load LoRA '{cfg.get('name', adapter_name)}': {e}",
                            self.__class__.__name__,
                        ) from e
                elif self._loaded_adapters[adapter_name] != load_key:
                    # Filepath or strengths changed — reload
                    try:
                        transformer.delete_adapters(adapter_name)
                    except Exception:
                        pass
                    try:
                        state_dict = _load_lora_state_dict(filepath, video_str, audio_str)
                        transformer.load_lora_adapter(state_dict, adapter_name=adapter_name)
                        del state_dict
                        self._loaded_adapters[adapter_name] = load_key
                    except Exception as e:
                        raise NodeError(
                            f"Failed to reload LoRA '{cfg.get('name', adapter_name)}': {e}",
                            self.__class__.__name__,
                        ) from e

            # Set active adapters with weights — use granular dict when enabled
            names = list(desired.keys())
            weights = []
            for n in names:
                cfg = desired[n]
                if cfg.get("granular_transformer_weights_enabled") and cfg.get("granular_transformer_weights"):
                    weights.append(cfg["granular_transformer_weights"])
                else:
                    weights.append(float(cfg.get("weight", 1.0)))
            try:
                transformer.set_adapters(names, weights)
                logger.info(
                    "Active LoRA adapters: %s",
                    ", ".join(
                        f"{n}=granular" if isinstance(w, dict) else f"{n}={w:.2f}" for n, w in zip(names, weights)
                    ),
                )
            except Exception as e:
                raise NodeError(
                    f"Failed to set LoRA adapters: {e}",
                    self.__class__.__name__,
                ) from e

        # Re-apply group offload hooks so new LoRA submodules are covered
        mm.reapply_group_offload(transformer_component_name, self.device)

        # Extract max reference_downscale_factor from active LoRA metadata
        max_downscale = 1
        for adapter_name, cfg in desired.items():
            filepath = self._resolve_lora_filepath(cfg)
            if filepath:
                ds = self._read_reference_downscale_factor(filepath)
                max_downscale = max(max_downscale, ds)

        # Collect per-adapter mask configs
        lora_masks = self._collect_lora_masks(desired)

        self.values = {
            "transformer": transformer,
            "transformer_component_name": transformer_component_name,
            "reference_downscale_factor": max_downscale,
            "lora_masks": lora_masks if lora_masks else None,
        }
        return self.values

    def update_loras(self, configs: list[dict]) -> None:
        if configs != self.lora_configs:
            self.lora_configs = configs
            self.set_updated()

    @staticmethod
    def _collect_lora_masks(desired: dict[str, dict]) -> list[tuple]:
        """Collect mask data for adapters that have spatial or temporal masking enabled.

        Returns:
            List of (adapter_name, spatial_mask_tensor | None, temporal_config | None) tuples.
        """
        from frameartisan.modules.generation.graph.nodes.lora_mask import load_spatial_mask

        masks = []
        for adapter_name, cfg in desired.items():
            spatial_mask = None
            temporal_config = None

            if cfg.get("spatial_mask_enabled") and cfg.get("spatial_mask_path"):
                spatial_mask = load_spatial_mask(cfg["spatial_mask_path"])

            if cfg.get("temporal_mask_enabled"):
                temporal_config = {
                    "start_frame": cfg.get("temporal_start_frame", 0),
                    "end_frame": cfg.get("temporal_end_frame", -1),
                    "fade_in_frames": cfg.get("temporal_fade_in_frames", 0),
                    "fade_out_frames": cfg.get("temporal_fade_out_frames", 0),
                }

            if spatial_mask is not None or temporal_config is not None:
                masks.append((adapter_name, spatial_mask, temporal_config))

        return masks
