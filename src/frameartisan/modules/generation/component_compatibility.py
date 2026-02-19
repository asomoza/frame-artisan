"""Architecture compatibility mapping for component reuse.

Defines which text_encoder, VAE, and tokenizer architectures are compatible
with each transformer architecture. Used for transformer-only imports to
auto-detect usable shared components.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


ARCHITECTURE_COMPATIBILITY: dict[str, dict[str, list[str]]] = {
    "ZImageTransformer2DModel": {
        "text_encoder": ["Qwen3Model", "Qwen3ForCausalLM"],
        "vae": ["AutoencoderKL"],
        "tokenizer": ["Qwen2Tokenizer", "Qwen2TokenizerFast"],
    },
    "Flux2Transformer2DModel": {
        "text_encoder": ["Qwen3ForCausalLM"],
        "vae": ["AutoencoderKLFlux2"],
        "tokenizer": ["Qwen2TokenizerFast"],
    },
    "LTX2VideoTransformer3DModel": {
        "text_encoder": ["Gemma3ForConditionalGeneration"],
        "vae": ["AutoencoderKLLTXVideo"],
        "tokenizer": ["GemmaTokenizerFast", "Gemma2TokenizerFast"],
    },
}

# Safetensors key patterns used to detect transformer architecture.
# Each entry maps a frozenset of "signature keys" to an architecture name.
# A file matches if ALL signature keys are present in its tensor key set.
_ARCHITECTURE_SIGNATURES: list[tuple[frozenset[str], str]] = [
    # Diffusers key format (bare keys)
    (
        frozenset({"context_refiner.weight", "all_final_layer.adaLN_modulation.1.weight"}),
        "ZImageTransformer2DModel",
    ),
    # Original key format (model.diffusion_model. prefix)
    (
        frozenset(
            {
                "model.diffusion_model.final_layer.adaLN_modulation.1.weight",
                "model.diffusion_model.context_refiner.0.attention.out.weight",
            }
        ),
        "ZImageTransformer2DModel",
    ),
    # Flux2 Klein diffusers key format
    (
        frozenset({"transformer_blocks.0.ff.net.0.proj.weight", "single_transformer_blocks.0.attn.to_q.weight"}),
        "Flux2Transformer2DModel",
    ),
    # LTX-Video 2 diffusers key format
    (
        frozenset({"patchify_proj.weight", "transformer_blocks.0.attn.to_q.weight"}),
        "LTX2VideoTransformer3DModel",
    ),
]


def detect_transformer_architecture(safetensors_path: str) -> str | None:
    """Detect transformer architecture from a .safetensors file's tensor keys.

    Args:
        safetensors_path: Path to a .safetensors file.

    Returns:
        Architecture name string if detected, None otherwise.
    """
    try:
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

        for signature_keys, architecture in _ARCHITECTURE_SIGNATURES:
            if signature_keys.issubset(keys):
                return architecture

    except Exception as e:
        logger.debug("Failed to detect architecture from %s: %s", safetensors_path, e)

    return None
