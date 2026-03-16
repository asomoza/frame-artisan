from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
from frameartisan.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject

logger = logging.getLogger(__name__)


def extract_dict_from_json_graph(json_graph: Any, wanted: Iterable[Any], *, include_missing: bool = False) -> dict:
    data = coerce_to_dict(json_graph)
    if not data:
        return {}

    nodes = (data.get("nodes") or []) if isinstance(data, dict) else []

    by_name: dict[str, dict] = {}
    loras: list[dict[str, Any]] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        name = node.get("name") or (node.get("state") or {}).get("name")
        if name:
            by_name[name] = node

        if node.get("class") == "LoraNode":
            state = node.get("state") or {}
            loras.append(
                {
                    "id": state.get("id", node.get("id")),
                    "name": state.get("name", node.get("name")),
                    "adapter_name": state.get("adapter_name", None),
                    "lora_name": state.get("lora_name", None),
                    "path": state.get("path", None),
                    "transformer_weight": state.get("transformer_weight", 1.0),
                    "version": state.get("version", None),
                    "enabled": state.get("lora_enabled", state.get("enabled", node.get("enabled"))),
                    "is_slider": state.get("is_slider", False),
                    "database_id": state.get("database_id", 0),
                    "granular_transformer_weights_enabled": state.get("granular_transformer_weights_enabled", False),
                    "granular_transformer_weights": state.get("transformer_granular_weights", {}),
                }
            )

    def auto_value(node_state: dict) -> Any:
        if "text" in node_state:
            return node_state.get("text")
        if "number" in node_state:
            return node_state.get("number")
        if "model_name" in node_state:
            return node_state.get("model_name")
        if "scheduler_data_object" in node_state:
            return node_state.get("scheduler_data_object")
        if "value" in node_state:
            return node_state.get("value")
        if "path" in node_state:
            return node_state.get("path")
        return None

    out: dict[str, Any] = {}
    for spec in wanted or []:
        out_key: str | None
        node_name: str | None
        state_key: str | None
        default: Any

        if isinstance(spec, str):
            out_key = spec
            node_name = spec
            state_key = None
            default = None
        elif isinstance(spec, dict):
            out_key = spec.get("out") or spec.get("name")
            node_name = spec.get("name")
            state_key = spec.get("key")
            default = spec.get("default")
        else:
            continue

        if not out_key or not node_name:
            continue

        # Special case: "loras" returns a list
        if node_name == "loras":
            value = loras
            if (value is None or value == []) and include_missing:
                value = default if default is not None else []
            if value is None and not include_missing:
                continue
            out[out_key] = value
            continue

        # Special case: "model" is stored at the top-level of the node dict
        if node_name == "model":
            node = by_name.get(node_name)
            if not node:
                if include_missing:
                    out[out_key] = default
                continue

            model_name = node.get("model_name")
            path = node.get("path")
            version = node.get("version")
            model_type = node.get("model_type")
            model_id = node.get("db_model_id", 0)

            if model_name is None and path is None and version is None and model_type is None:
                if include_missing:
                    out[out_key] = default
                continue

            out[out_key] = {
                "name": model_name or "",
                "version": version or "",
                "filepath": path or "",
                "model_type": model_type or 0,
                "id": model_id or 0,
            }
            continue

        node = by_name.get(node_name)
        if not node:
            if include_missing:
                out[out_key] = default
            continue

        state = node.get("state") or {}
        value = state.get(state_key) if state_key else auto_value(state)
        if value is None:
            value = default

        if value is None and not include_missing:
            continue

        out[out_key] = value

    return out


def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def coerce_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return default


def coerce_to_dict(json_graph: Any) -> dict:
    if isinstance(json_graph, dict):
        return json_graph
    if isinstance(json_graph, str):
        try:
            parsed = json.loads(json_graph)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def cast_number_range(value) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("the number range must be a list of 2 numbers")

    a, b = value[0], value[1]
    if isinstance(a, bool) or isinstance(b, bool):
        raise ValueError("the number range values must be int|float (not bool)")

    return [float(a), float(b)]


def cast_scheduler(value) -> SchedulerDataObject:
    if isinstance(value, SchedulerDataObject):
        return value

    if value is None:
        return SchedulerDataObject()

    if isinstance(value, dict):
        try:
            return SchedulerDataObject.from_dict(value)
        except Exception:
            return SchedulerDataObject()

    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return SchedulerDataObject.from_dict(data)
        except Exception:
            pass
        return SchedulerDataObject()

    return SchedulerDataObject()


def cast_model(value: Any) -> ModelDataObject | None:
    if isinstance(value, ModelDataObject):
        return value

    if value is None:
        return ModelDataObject()

    if isinstance(value, dict):
        try:
            return ModelDataObject.from_dict(value)
        except Exception:
            return ModelDataObject()

    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return ModelDataObject.from_dict(data)
        except Exception:
            pass
        return ModelDataObject()

    return ModelDataObject()


# ---------------------------------------------------------------------------
# Source file persistence — copy source files to permanent directories and
# rewrite paths in graph JSON so that metadata embedded in videos always
# references stable, deduplicated copies.
# ---------------------------------------------------------------------------


def _find_existing_source(db: Any, kind: str, content_hash: str) -> str | None:
    """Look up an existing source file by kind and content hash."""
    row = db.fetch_one(
        "SELECT id, filepath FROM source_file WHERE kind = ? AND content_hash = ?",
        (kind, content_hash),
    )
    if row is None:
        return None
    row_id, filepath = row
    if os.path.isfile(filepath):
        return filepath
    db.execute("DELETE FROM source_file WHERE id = ?", (row_id,))
    return None


def _record_source(db: Any, kind: str, content_hash: str, filepath: str) -> None:
    """Record a source file in the database, upserting on conflict."""
    db.execute(
        "INSERT INTO source_file (kind, content_hash, filepath) VALUES (?, ?, ?) "
        "ON CONFLICT(kind, content_hash) DO UPDATE SET filepath = excluded.filepath",
        (kind, content_hash, filepath),
    )


def persist_source_paths_in_graph(
    json_graph: str,
    *,
    source_image_dir: str | None = None,
    source_audio_dir: str | None = None,
    source_video_dir: str | None = None,
    lora_mask_dir: str | None = None,
) -> str:
    """Copy source files to permanent directories and rewrite paths in graph JSON.

    Walks the graph nodes looking for:
    - ``ImageLoadNode`` named ``source_image_*`` → images
    - ``LTX2AudioEncodeNode`` named ``audio_encode`` → audio (``audio_path``)
    - ``VideoLoadNode`` named ``condition_video*`` → video

    Uses content-hash deduplication via the ``source_file`` DB table: if an
    identical file was already saved for the same kind, the existing copy is
    reused and the path is rewritten to point there.

    Returns the (possibly updated) JSON string.
    """
    from frameartisan.app.app import get_app_database_path
    from frameartisan.utils.database import Database

    try:
        data = json.loads(json_graph)
    except Exception:
        return json_graph

    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return json_graph

    db_path = get_app_database_path()
    db = Database(db_path) if db_path else None

    updated = False

    for node in nodes:
        if not isinstance(node, dict):
            continue

        cls_name = node.get("class", "")
        node_name = node.get("name", "")
        state = node.get("state")
        if not isinstance(state, dict):
            continue

        # --- Condition images (ImageLoadNode named source_image_*) ---
        if cls_name == "ImageLoadNode" and node_name.startswith("source_image_") and source_image_dir:
            src_path = state.get("path")
            result = _persist_file(db, src_path, source_image_dir, kind=node_name)
            if result is not None:
                state["path"] = result
                updated = True

        # --- Audio (LTX2AudioEncodeNode named audio_encode) ---
        elif cls_name == "LTX2AudioEncodeNode" and node_name == "audio_encode" and source_audio_dir:
            src_path = state.get("audio_path")
            result = _persist_file(db, src_path, source_audio_dir, kind="audio")
            if result is not None:
                state["audio_path"] = result
                updated = True

        # --- Source video (VideoLoadNode named condition_video*) ---
        elif cls_name == "VideoLoadNode" and node_name.startswith("condition_video") and source_video_dir:
            src_path = state.get("path")
            result = _persist_file(db, src_path, source_video_dir, kind=node_name)
            if result is not None:
                state["path"] = result
                updated = True

        # --- LoRA masks (LTX2LoraNode with spatial_mask_path in lora_configs) ---
        elif cls_name == "LTX2LoraNode" and lora_mask_dir:
            lora_configs = state.get("lora_configs")
            if isinstance(lora_configs, list):
                for cfg in lora_configs:
                    if not isinstance(cfg, dict):
                        continue
                    mask_path = cfg.get("spatial_mask_path")
                    if mask_path:
                        lora_name = cfg.get("name", cfg.get("hash", "lora"))
                        kind = f"lora_mask_{lora_name}"
                        result = _persist_file(db, mask_path, lora_mask_dir, kind=kind)
                        if result is not None:
                            cfg["spatial_mask_path"] = result
                            updated = True

    if not updated:
        return json_graph

    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json_graph


def _persist_file(
    db: Any,
    src_path: str | None,
    dest_dir: str,
    *,
    kind: str,
) -> str | None:
    """Copy a single source file to *dest_dir* with dedup. Returns the new path, or ``None`` if nothing to do."""
    if not src_path or not isinstance(src_path, str) or not src_path.strip():
        return None

    src = Path(src_path)
    if not src.is_file():
        return None

    dest_dir_path = Path(dest_dir)

    # Already in the target directory — nothing to do.
    try:
        if src.parent.resolve() == dest_dir_path.resolve():
            return None
    except Exception:
        pass

    content_hash = hashlib.md5(src.read_bytes()).hexdigest()

    if db is not None:
        existing = _find_existing_source(db, kind, content_hash)
        if existing is not None:
            return existing

    dest_dir_path.mkdir(parents=True, exist_ok=True)
    ext = src.suffix or ".bin"
    dest = dest_dir_path / f"{kind}_{content_hash[:12]}{ext}"

    # Handle unlikely collision
    if dest.exists():
        for seq in range(1, 10_000):
            candidate = dest_dir_path / f"{kind}_{content_hash[:12]}_{seq}{ext}"
            if not candidate.exists():
                dest = candidate
                break

    try:
        shutil.copy2(src, dest)
    except Exception:
        logger.warning("Failed to copy source file %s → %s", src, dest)
        return None

    dest_str = str(dest)
    if db is not None:
        _record_source(db, kind, content_hash, dest_str)

    return dest_str
