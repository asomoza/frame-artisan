from __future__ import annotations

import json
from typing import Any, Iterable

from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject
from frameartisan.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject


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
