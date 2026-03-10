import logging
import os

from PyQt6.QtCore import QThread, pyqtSignal


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------


def build_variant_tags(
    transformer_tag: str,
    text_encoder_tag: str,
    quant_method: str = "",
) -> str:
    """Compose comma-separated tags from component-level metadata."""
    parts: list[str] = []
    if quant_method:
        parts.append(quant_method)
    parts.append(transformer_tag)
    parts.append(text_encoder_tag)
    return ", ".join(parts)


VARIANT_CONFIG = {
    "normal": {
        "repo_id": "OzzyGT/LTX2",
        "display_name": "LTX2 (bf16)",
        "model_type": 1,
        "allow_patterns": None,  # download everything
        "dir_name": "LTX2",
        "quant_method": "",
        "transformer_tag": "bf16 transformer",
        "text_encoder_tag": "bf16 text encoder",
        "tags": build_variant_tags("bf16 transformer", "bf16 text encoder"),
        "version": "1.0",
    },
    "distilled": {
        "repo_id": "OzzyGT/LTX2_distilled",
        "display_name": "LTX2 Distilled (bf16)",
        "model_type": 2,
        "allow_patterns": ["transformer/**", "scheduler/**", "model_index.json"],
        "dir_name": "LTX2_Distilled",
        "quant_method": "",
        "transformer_tag": "bf16 transformer",
        "text_encoder_tag": "bf16 text encoder",
        "tags": build_variant_tags("bf16 transformer", "bf16 text encoder"),
        "version": "1.0",
    },
    "sdnq_4bit": {
        "repo_id": "OzzyGT/LTX2_SDNQ_4bit_dynamic",
        "display_name": "LTX2 SDNQ 4-bit",
        "model_type": 1,
        "allow_patterns": ["transformer/**", "text_encoder/**", "model_index.json"],
        "dir_name": "LTX2_SDNQ_4bit",
        "quant_method": "sdnq",
        "transformer_tag": "4bit transformer",
        "text_encoder_tag": "4bit text encoder",
        "tags": build_variant_tags("4bit transformer", "4bit text encoder", "sdnq"),
        "version": "4bit",
    },
    "sdnq_8bit": {
        "repo_id": "OzzyGT/LTX2_SDNQ_8bit_dynamic",
        "display_name": "LTX2 SDNQ 8-bit",
        "model_type": 1,
        "allow_patterns": ["transformer/**", "text_encoder/**", "model_index.json"],
        "dir_name": "LTX2_SDNQ_8bit",
        "quant_method": "sdnq",
        "transformer_tag": "8bit transformer",
        "text_encoder_tag": "8bit text encoder",
        "tags": build_variant_tags("8bit transformer", "8bit text encoder", "sdnq"),
        "version": "8bit",
    },
    "distilled_sdnq_4bit": {
        "repo_id": "OzzyGT/LTX2_distilled_SDNQ_4bit_dynamic",
        "display_name": "LTX2 Distilled SDNQ 4-bit",
        "model_type": 2,
        "allow_patterns": ["transformer/**", "text_encoder/**", "scheduler/**", "model_index.json"],
        "dir_name": "LTX2_Distilled_SDNQ_4bit",
        "quant_method": "sdnq",
        "transformer_tag": "4bit transformer",
        "text_encoder_tag": "4bit text encoder",
        "tags": build_variant_tags("4bit transformer", "4bit text encoder", "sdnq"),
        "version": "4bit",
    },
    "distilled_sdnq_8bit": {
        "repo_id": "OzzyGT/LTX2_distilled_SDNQ_8bit_dynamic",
        "display_name": "LTX2 Distilled SDNQ 8-bit",
        "model_type": 2,
        "allow_patterns": ["transformer/**", "text_encoder/**", "scheduler/**", "model_index.json"],
        "dir_name": "LTX2_Distilled_SDNQ_8bit",
        "quant_method": "sdnq",
        "transformer_tag": "8bit transformer",
        "text_encoder_tag": "8bit text encoder",
        "tags": build_variant_tags("8bit transformer", "8bit text encoder", "sdnq"),
        "version": "8bit",
    },
    "latent_upsampler": {
        "repo_id": "OzzyGT/LTX2_latent_upsampler",
        "display_name": "LTX2 Latent Upsampler",
        "model_type": 0,
        "allow_patterns": None,
        "dir_name": "LTX2_latent_upsampler",
        "quant_method": "",
        "transformer_tag": "",
        "text_encoder_tag": "",
        "tags": "",
        "version": "1.0",
    },
}

# Text-encoder sharing: SDNQ variants at the same bit-width share the same TE,
# regardless of whether they are distilled or not.
# key = variant key, value = variant key whose TE can be reused.
TE_SHARED_FROM = {
    "distilled_sdnq_4bit": "sdnq_4bit",
    "distilled_sdnq_8bit": "sdnq_8bit",
    "sdnq_4bit": "distilled_sdnq_4bit",
    "sdnq_8bit": "distilled_sdnq_8bit",
}


class ModelDownloadThread(QThread):
    """Downloads model variants from HuggingFace using ``huggingface_hub.snapshot_download``."""

    progress = pyqtSignal(str, int, int)  # (status_message, current_bytes, total_bytes)
    variant_completed = pyqtSignal(str, str)  # (variant_key, local_path)
    download_finished = pyqtSignal()
    download_error = pyqtSignal(str)

    def __init__(self, models_dir: str, selected_variants: list[str]):
        super().__init__()
        self.models_dir = models_dir
        self.selected_variants = selected_variants
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            from huggingface_hub import snapshot_download

            # Track which text-encoder quantizations have been downloaded
            downloaded_te: set[str] = set()

            # Download base models first: normal (shared components), then distilled.
            ordered = []
            for base in ("normal", "distilled"):
                if base in self.selected_variants:
                    ordered.append(base)
            for v in self.selected_variants:
                if v not in ("normal", "distilled"):
                    ordered.append(v)

            for variant_key in ordered:
                if self._stop_requested:
                    break

                cfg = VARIANT_CONFIG.get(variant_key)
                if cfg is None:
                    continue

                local_dir = os.path.join(self.models_dir, cfg["dir_name"])
                allow_patterns = cfg["allow_patterns"]

                # Optimise: if the TE was already downloaded from a sibling variant,
                # only download transformer + scheduler + model_index.json
                if allow_patterns is not None and "text_encoder/**" in allow_patterns:
                    te_sibling = TE_SHARED_FROM.get(variant_key)
                    if te_sibling and te_sibling in downloaded_te:
                        reduced = ["transformer/**", "model_index.json"]
                        if "scheduler/**" in allow_patterns:
                            reduced.insert(1, "scheduler/**")
                        allow_patterns = reduced

                self.progress.emit(f"Downloading {cfg['display_name']}...", 0, 0)

                try:
                    snapshot_download(
                        repo_id=cfg["repo_id"],
                        local_dir=local_dir,
                        allow_patterns=allow_patterns,
                    )
                except Exception as e:
                    self.download_error.emit(f"Failed to download {cfg['display_name']}: {e}")
                    return

                # Track TE downloads
                if allow_patterns is None or "text_encoder/**" in (cfg["allow_patterns"] or []):
                    downloaded_te.add(variant_key)

                self.variant_completed.emit(variant_key, local_dir)

            self.download_finished.emit()

        except Exception as e:
            logger.exception("Download thread error")
            self.download_error.emit(str(e))
