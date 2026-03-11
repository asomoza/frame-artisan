from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2ModelNode(Node):
    PRIORITY = 2
    OUTPUTS = [
        "tokenizer",
        "text_encoder",
        "transformer",
        "vae",
        "audio_vae",
        "connectors",
        "vocoder",
        "scheduler_config",
        "transformer_component_name",
    ]
    REQUIRED_INPUTS: ClassVar[list] = []
    OPTIONAL_INPUTS: ClassVar[list] = ["use_torch_compile", "torch_compile_max_autotune"]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = {
        "model_path",
        "model_id",
        "offload_strategy",
        "group_offload_use_stream",
        "group_offload_low_cpu_mem",
        "streaming_decode",
        "ff_chunking",
        "ff_num_chunks",
        "component_prefix",
        "component_overrides",
    }

    _RUNTIME_EXCLUDE: ClassVar[set[str]] = set()

    def __init__(
        self,
        model_path: str = "",
        model_id: int | None = None,
        offload_strategy: str = "auto",
        group_offload_use_stream: bool = False,
        group_offload_low_cpu_mem: bool = False,
        streaming_decode: bool = False,
        ff_chunking: bool = False,
        ff_num_chunks: int = 2,
        component_prefix: str = "",
        component_overrides: dict[str, int] | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.model_id = model_id
        self.offload_strategy = offload_strategy
        self.group_offload_use_stream = group_offload_use_stream
        self.group_offload_low_cpu_mem = group_offload_low_cpu_mem
        self.streaming_decode = streaming_decode
        self.ff_chunking = ff_chunking
        self.ff_num_chunks = ff_num_chunks
        self.component_prefix = component_prefix
        self.component_overrides = component_overrides
        self.status_callback = None
        self._prev_component_paths: dict[str, str] = {}
        self._prev_values: dict[str, object] = {}

    def _resolve_component_paths(self, model_path: str) -> dict[str, str]:
        """Resolve actual storage paths for components via the registry.

        Uses the node's in-memory ``component_overrides`` when available so
        that first-pass and second-pass model nodes can have independent
        variant selections even when they share the same model_id.  Falls
        back to DB overrides when ``component_overrides`` is ``None``, and
        to local subfolders when the registry is unavailable.
        """
        try:
            from frameartisan.app.app import get_app_database_path
            from frameartisan.app.component_registry import ComponentRegistry

            db_path = get_app_database_path()
            if db_path is not None:
                import os

                components_base_dir = os.path.join(os.path.dirname(model_path), "_components")
                registry = ComponentRegistry(db_path, components_base_dir)
                if self.model_id is not None:
                    if self.component_overrides:
                        comps = registry.resolve_model_components_with_overrides(
                            self.model_id, self.component_overrides
                        )
                    else:
                        comps = registry.resolve_model_components(self.model_id)
                    if comps:
                        return {ct: info.storage_path for ct, info in comps.items()}
                paths = registry.resolve_component_paths(model_path)
                if paths:
                    return paths
        except Exception as e:
            logger.debug("Could not resolve component paths from registry: %s", e)
        return {}

    def _load_component(self, cls, path: str, label: str, **kwargs):
        """Load a single component via from_pretrained, wrapping errors."""
        logger.info("Loading %s from %s", label, path)
        try:
            return cls.from_pretrained(path, **kwargs)
        except Exception as e:
            raise NodeError(f"Failed to load {label}: {e}", self.__class__.__name__) from e

    def get_state(self) -> dict:
        return super().get_state()

    def __call__(self):
        import os

        import torch
        from diffusers import (
            AutoencoderKLLTX2Audio,
            AutoencoderKLLTX2Video,
            FlowMatchEulerDiscreteScheduler,
            LTX2VideoTransformer3DModel,
        )
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
        from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

        from frameartisan.app.model_manager import get_model_manager

        try:
            from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer
        except ImportError:
            raise NodeError(
                "transformers package not found. Install transformers>=4.50 for LTX2 support.",
                self.__class__.__name__,
            )

        model_path = self.model_path
        if not model_path:
            raise NodeError("model_path is empty. Set a path to the LTX2 model directory.", self.__class__.__name__)

        mm = get_model_manager()

        # Resolve component paths from the registry; fall back to local subfolders.
        component_paths = self._resolve_component_paths(model_path)

        def _path(comp_type: str) -> str:
            return component_paths.get(comp_type, os.path.join(model_path, comp_type))

        prefix = self.component_prefix
        transformer_only = bool(prefix)

        # Build resolved path map to detect what changed since last run.
        # When prefix is set (2nd pass), only load the transformer and scheduler.
        if transformer_only:
            all_comp_types = ["transformer", "scheduler"]
        else:
            all_comp_types = [
                "text_encoder",
                "transformer",
                "tokenizer",
                "scheduler",
                "vae",
                "audio_vae",
                "connectors",
                "vocoder",
            ]
        resolved_paths = {ct: _path(ct) for ct in all_comp_types}
        changed = {ct for ct in all_comp_types if resolved_paths[ct] != self._prev_component_paths.get(ct)}
        if changed:
            logger.info("Components to reload: %s", ", ".join(sorted(changed)))
        else:
            logger.info("No component paths changed — full reload")
            changed = set(all_comp_types)

        # --- Heavy components (cached in ModelManager) ---
        tr_path = resolved_paths["transformer"]
        te_path = resolved_paths.get("text_encoder", "")

        # Detect quantization from the actual resolved paths (not the base model dir)
        quant_type = self._detect_quantization_from_path(tr_path)
        if not transformer_only:
            quant_type = quant_type or self._detect_quantization_from_path(te_path)
        logger.info("LTX2 quantization type detected: %s", quant_type or "full-precision")

        # Register SDNQ quantization type before loading any components
        sdnq_available = False
        _sdnqconfig_patched = False
        if quant_type == "sdnq":
            try:
                from sdnq import SDNQConfig  # noqa: F401 — registers sdnq in diffusers/transformers
                from sdnq.common import check_torch_compile

                sdnq_available = True

                if not check_torch_compile():
                    # The saved config may store use_quantized_matmul which
                    # will error on load if triton is not present
                    _orig_sdnqconfig_init = SDNQConfig.__init__

                    def _patched_init(self, *args, use_quantized_matmul=False, **kwargs):  # noqa: ARG001
                        _orig_sdnqconfig_init(self, *args, use_quantized_matmul=False, **kwargs)

                    SDNQConfig.__init__ = _patched_init
                    _sdnqconfig_patched = True
                    logger.info("Triton / torch.compile unavailable — SDNQ quantized matmul disabled for loading")
            except ImportError:
                logger.warning("sdnq package not available — SDNQ models will load dequantized (higher memory usage)")

        tr_hash = mm.component_hash(tr_path)
        transformer = mm.get_cached(tr_hash)

        # Always pass torch_dtype=bfloat16. For SDNQ models this casts
        # non-quantized layers (norms, biases) to bf16 while preserving
        # quantized weights
        heavy_kwargs: dict = {"device_map": "cpu", "torch_dtype": torch.bfloat16}

        if not transformer_only:
            te_hash = mm.component_hash(te_path)
            text_encoder = mm.get_cached(te_hash)
            if text_encoder is None:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                text_encoder = self._load_component(
                    Gemma3ForConditionalGeneration,
                    te_path,
                    "text_encoder",
                    **heavy_kwargs,
                )
                mm.set_cached(te_hash, text_encoder)
                logger.info("text_encoder loaded and cached (%s)", self._describe_model(text_encoder))
            else:
                logger.info("text_encoder loaded from cache (%s)", self._describe_model(text_encoder))

        if transformer is None:
            transformer = self._load_component(
                LTX2VideoTransformer3DModel,
                tr_path,
                "transformer",
                **heavy_kwargs,
            )
            mm.set_cached(tr_hash, transformer)
            logger.info("transformer loaded and cached (%s)", self._describe_model(transformer))
        else:
            logger.info("transformer loaded from cache (%s)", self._describe_model(transformer))

        # --- Light components (reuse from previous run if path unchanged) ---
        def _load_or_reuse(comp_type, cls, **kwargs):
            if comp_type not in changed and comp_type in self._prev_values:
                logger.info("Reusing %s (unchanged)", comp_type)
                return self._prev_values[comp_type]
            return self._load_component(cls, resolved_paths[comp_type], comp_type, **kwargs)

        light_kwargs: dict = {"torch_dtype": torch.bfloat16}

        # Load scheduler (needed for both primary and prefix nodes)
        scheduler = (
            self._prev_values["scheduler_obj"]
            if "scheduler" not in changed and "scheduler_obj" in self._prev_values
            else FlowMatchEulerDiscreteScheduler.from_pretrained(resolved_paths["scheduler"])
        )

        if not transformer_only:
            tokenizer = _load_or_reuse("tokenizer", GemmaTokenizer)
            vae = _load_or_reuse("vae", AutoencoderKLLTX2Video, **light_kwargs)
            audio_vae = _load_or_reuse("audio_vae", AutoencoderKLLTX2Audio, **light_kwargs)
            connectors = _load_or_reuse("connectors", LTX2TextConnectors, **light_kwargs)
            vocoder = _load_or_reuse("vocoder", LTX2Vocoder, **light_kwargs)

        # Restore original SDNQConfig.__init__ if we patched it
        if _sdnqconfig_patched:
            SDNQConfig.__init__ = _orig_sdnqconfig_init  # noqa: F821

        # SDNQ post-processing: enable quantized matmul when triton is available
        if quant_type == "sdnq" and sdnq_available:
            try:
                from sdnq.common import check_torch_compile as _ctc

                if _ctc():
                    from sdnq.loader import apply_sdnq_options_to_model

                    transformer = apply_sdnq_options_to_model(transformer, use_quantized_matmul=True)
                    if not transformer_only:
                        text_encoder = apply_sdnq_options_to_model(text_encoder, use_quantized_matmul=True)
                    logger.info("SDNQ quantized matmul enabled for transformer%s", " and text_encoder" if not transformer_only else "")
                else:
                    logger.info("Triton / torch.compile not available — SDNQ models using standard matmul")
            except Exception:
                logger.info("triton not available — SDNQ models will use standard matmul")

        # Register nn.Module components for lifecycle management.
        # When component_prefix is set (2nd pass), only register the transformer
        # under a prefixed name — shared components (vae, text_encoder, etc.)
        # are already registered by the primary model node.
        if transformer_only:
            mm.register_component(f"{prefix}transformer", transformer)
        else:
            nn_components = {
                "transformer": transformer,
                "text_encoder": text_encoder,
                "connectors": connectors,
                "vae": vae,
                "audio_vae": audio_vae,
                "vocoder": vocoder,
            }
            for name, mod in nn_components.items():
                mm.register_component(name, mod)

        # Free any previous light components that were just replaced in
        # managed_components.  Without this, old vae/vocoder/etc. instances
        # stay alive until the next GC cycle, doubling RAM for these models.
        gc.collect()

        # Propagate offload settings to ModelManager and apply
        mm.offload_strategy = self.offload_strategy
        mm.group_offload_use_stream = self.group_offload_use_stream
        mm.group_offload_low_cpu_mem = self.group_offload_low_cpu_mem
        strategy = mm.apply_offload_strategy(self.device)
        logger.info("Offload strategy: %s", strategy)

        # Regional compilation for the transformer — compiles only the repeated
        # transformer blocks (much faster cold start than full torch.compile).
        # Applied AFTER offload hooks so Dynamo can trace through wrappers.
        if self.use_torch_compile:
            compile_mode = "max-autotune-no-cudagraphs" if self.torch_compile_max_autotune else "default"
            logger.info(
                "torch.compile: max_autotune=%s → mode=%s",
                self.torch_compile_max_autotune,
                compile_mode,
            )
            if self.status_callback is not None:
                self.status_callback("Compiling transformer (this may take a while)...")
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 1000

            import torch._inductor.config as inductor_config

            # Suppress verbose autotune benchmarking output
            inductor_config.max_autotune_report_choices_stats = False
            inductor_config.autotune_num_choices_displayed = 0
            logger.info("Applying regional compile to transformer blocks (mode=%s)", compile_mode)
            transformer.compile_repeated_blocks(mode=compile_mode)

        if not transformer_only:
            # Streaming temporal decode (external tiling with lower peak VRAM)
            vae._use_streaming_decode = bool(self.streaming_decode)
            if self.streaming_decode:
                logger.info("VAE streaming decode enabled")

        # Feed-forward chunking (lower peak activation VRAM during denoise)
        from frameartisan.modules.generation.graph.nodes.ff_chunking import (
            disable_forward_chunking,
            enable_forward_chunking,
        )

        if self.ff_chunking:
            enable_forward_chunking(transformer, num_chunks=self.ff_num_chunks)
            logger.info("Feed-forward chunking enabled (num_chunks=%d)", self.ff_num_chunks)
        else:
            disable_forward_chunking(transformer)

        # Extract scheduler config as a plain dict for downstream nodes
        scheduler_config = dict(scheduler.config)

        transformer_component_name = f"{prefix}transformer" if prefix else "transformer"

        if transformer_only:
            self.values = {
                "transformer": transformer,
                "scheduler_config": scheduler_config,
                "transformer_component_name": transformer_component_name,
            }
            self._prev_component_paths = resolved_paths
            self._prev_values = {
                "transformer": transformer,
                "scheduler_obj": scheduler,
            }
        else:
            self.values = {
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae,
                "audio_vae": audio_vae,
                "connectors": connectors,
                "vocoder": vocoder,
                "scheduler_config": scheduler_config,
                "transformer_component_name": transformer_component_name,
            }
            self._prev_component_paths = resolved_paths
            self._prev_values = {
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae,
                "audio_vae": audio_vae,
                "connectors": connectors,
                "vocoder": vocoder,
                "scheduler_obj": scheduler,
            }

        return self.values

    def update_value(self, model_path: str, model_id: int | None = None) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.updated = True

    def clear_models(self):
        from frameartisan.app.model_manager import get_model_manager

        get_model_manager().clear()

    @staticmethod
    def _describe_model(model) -> str:
        """Return a human-readable description of a model's dtype/quantization."""
        config = getattr(model, "config", None)
        if config is not None:
            qc = getattr(config, "quantization_config", None)
            if qc is not None:
                qc_dict = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
                quant_type = str(qc_dict.get("quant_type", qc_dict.get("quant_method", ""))).lower()

                # SDNQ uses weights_dtype (e.g. "int4", "int8")
                weights_dtype = str(qc_dict.get("weights_dtype", "")).lower()
                bits = qc_dict.get("weight_bits") or qc_dict.get("bits")

                if "sdnq" in quant_type:
                    if weights_dtype:
                        return f"sdnq {weights_dtype}"
                    return f"sdnq {bits}-bit" if bits else "sdnq"

                if qc_dict.get("load_in_4bit") or bits == 4:
                    return "bnb 4-bit"
                if qc_dict.get("load_in_8bit") or bits == 8:
                    return "bnb 8-bit"
                return f"quantized ({quant_type})" if quant_type else "quantized"

        # Fall back to parameter dtype
        try:
            param = next(model.parameters())
            return str(param.dtype).replace("torch.", "")
        except StopIteration:
            return "unknown"

    @staticmethod
    def _detect_quantization_from_path(component_path: str) -> str | None:
        """Detect quantization from a component directory's config.json."""
        config_path = Path(component_path) / "config.json"
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        qc = config.get("quantization_config")
        if qc is None:
            return None

        quant_type = str(qc.get("quant_type", qc.get("quant_method", ""))).lower()
        if "sdnq" in quant_type:
            return "sdnq"

        if qc.get("load_in_4bit") or qc.get("load_in_8bit") or "bitsandbytes" in quant_type:
            return "bnb"

        return None
