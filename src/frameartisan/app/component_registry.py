from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from frameartisan.utils.database import Database


logger = logging.getLogger(__name__)

COMPONENT_TYPES = (
    "tokenizer",
    "text_encoder",
    "transformer",
    "vae",
    "audio_vae",
    "vocoder",
    "connectors",
    "scheduler",
)


@dataclass
class ComponentInfo:
    id: int
    component_type: str
    content_hash: str
    storage_path: str
    size_bytes: int = 0
    architecture: Optional[str] = None
    config_json: Optional[str] = None
    dtype: Optional[str] = None


class ComponentRegistry:
    def __init__(self, db_path: str, components_base_dir: str):
        self.db_path = db_path
        self.components_base_dir = components_base_dir

    def _db(self) -> Database:
        return Database(self.db_path)

    def register_component(
        self,
        component_type: str,
        source_path: str,
        content_hash: str,
        *,
        architecture: str | None = None,
        config_json: str | None = None,
    ) -> ComponentInfo:
        """Register a component by its hash. If the hash already exists, return the existing entry."""
        db = self._db()

        existing = self.get_component_by_hash(content_hash)
        if existing is not None:
            if existing.dtype is None:
                dtype = self._detect_dtype(source_path, existing.config_json)
                if dtype:
                    db.execute("UPDATE component SET dtype = ? WHERE id = ?", (dtype, existing.id))
                    existing.dtype = dtype
            return existing

        size_bytes = 0
        if os.path.isdir(source_path):
            for dirpath, _dirnames, filenames in os.walk(source_path):
                for f in filenames:
                    size_bytes += os.path.getsize(os.path.join(dirpath, f))
        elif os.path.isfile(source_path):
            size_bytes = os.path.getsize(source_path)

        if config_json is None:
            # Try config.json first, then tokenizer_config.json for tokenizers
            for config_name in ("config.json", "tokenizer_config.json"):
                config_path = os.path.join(source_path, config_name)
                if os.path.isfile(config_path):
                    try:
                        with open(config_path, "r") as f:
                            config_json = f.read()
                        break
                    except Exception:
                        pass

        if architecture is None and config_json is not None:
            try:
                cfg = json.loads(config_json)
                architecture = (
                    cfg.get("_class_name")
                    or (cfg.get("architectures") or [None])[0]
                    or cfg.get("tokenizer_class")
                    or cfg.get("model_type")
                )
            except Exception:
                pass

        dtype = self._detect_dtype(source_path, config_json)

        db.insert(
            "component",
            {
                "component_type": component_type,
                "content_hash": content_hash,
                "storage_path": source_path,
                "size_bytes": size_bytes,
                "architecture": architecture,
                "config_json": config_json,
                "dtype": dtype,
            },
        )
        component_id = db.last_insert_rowid()

        return ComponentInfo(
            id=component_id,
            component_type=component_type,
            content_hash=content_hash,
            storage_path=source_path,
            size_bytes=size_bytes,
            architecture=architecture,
            config_json=config_json,
            dtype=dtype,
        )

    def get_model_components(self, model_id: int) -> dict[str, ComponentInfo]:
        """Return {component_type: ComponentInfo} for all components of a model."""
        db = self._db()
        rows = db.fetch_all(
            """
            SELECT c.id, c.component_type, c.content_hash, c.storage_path,
                   c.size_bytes, c.architecture, c.config_json, c.dtype
            FROM model_component mc
            JOIN component c ON mc.component_id = c.id
            WHERE mc.model_id = ?
            """,
            (model_id,),
        )
        result: dict[str, ComponentInfo] = {}
        for row in rows:
            info = ComponentInfo(
                id=row[0],
                component_type=row[1],
                content_hash=row[2],
                storage_path=row[3],
                size_bytes=row[4] or 0,
                architecture=row[5],
                config_json=row[6],
                dtype=row[7],
            )
            result[info.component_type] = info
        return result

    def register_model_components(self, model_id: int, components: dict[str, int]) -> None:
        """Map model → components in model_component table.

        Args:
            model_id: The model's database ID.
            components: {component_type: component_id} mapping.
        """
        db = self._db()
        for component_type, component_id in components.items():
            db.execute(
                "INSERT OR REPLACE INTO model_component (model_id, component_type, component_id) VALUES (?, ?, ?)",
                (model_id, component_type, component_id),
            )

    def find_compatible_components(self, transformer_architecture: str) -> dict[str, list[ComponentInfo]]:
        """Find existing vae/text_encoder/tokenizer compatible with a transformer architecture."""
        from frameartisan.modules.generation.component_compatibility import ARCHITECTURE_COMPATIBILITY

        compat = ARCHITECTURE_COMPATIBILITY.get(transformer_architecture)
        if compat is None:
            return {}

        db = self._db()
        result: dict[str, list[ComponentInfo]] = {}

        for comp_type in ("text_encoder", "vae", "tokenizer"):
            arch_list = compat.get(comp_type, [])
            if not arch_list:
                continue

            placeholders = ", ".join(["?"] * len(arch_list))
            rows = db.fetch_all(
                f"""
                SELECT id, component_type, content_hash, storage_path,
                       size_bytes, architecture, config_json, dtype
                FROM component
                WHERE component_type = ? AND architecture IN ({placeholders})
                """,
                (comp_type, *arch_list),
            )
            if rows:
                result[comp_type] = [
                    ComponentInfo(
                        id=r[0],
                        component_type=r[1],
                        content_hash=r[2],
                        storage_path=r[3],
                        size_bytes=r[4] or 0,
                        architecture=r[5],
                        config_json=r[6],
                        dtype=r[7],
                    )
                    for r in rows
                ]

        return result

    def get_component_by_hash(self, content_hash: str) -> ComponentInfo | None:
        """Lookup component by hash."""
        db = self._db()
        row = db.fetch_one(
            """
            SELECT id, component_type, content_hash, storage_path,
                   size_bytes, architecture, config_json, dtype
            FROM component
            WHERE content_hash = ?
            """,
            (content_hash,),
        )
        if row is None:
            return None
        return ComponentInfo(
            id=row[0],
            component_type=row[1],
            content_hash=row[2],
            storage_path=row[3],
            size_bytes=row[4] or 0,
            architecture=row[5],
            config_json=row[6],
            dtype=row[7],
        )

    def resolve_component_paths(self, model_filepath: str) -> dict[str, str]:
        """Return {component_type: storage_path} for a model looked up by its filepath."""
        db = self._db()
        rows = db.fetch_all(
            """
            SELECT c.component_type, c.storage_path
            FROM component c
            JOIN model_component mc ON mc.component_id = c.id
            JOIN model m ON mc.model_id = m.id
            WHERE m.filepath = ?
            """,
            (model_filepath,),
        )
        return {row[0]: row[1] for row in rows}

    def model_has_components(self, model_id: int) -> bool:
        """Check if a model has all 4 component mappings."""
        db = self._db()
        row = db.fetch_one(
            "SELECT COUNT(*) FROM model_component WHERE model_id = ?",
            (model_id,),
        )
        return row is not None and row[0] >= len(COMPONENT_TYPES)

    def compact_shared_components(self) -> dict[str, int]:
        """Move shared components to canonical _components/ storage and remove model-local duplicates.

        Returns dict with stats: {"moved": N, "deduplicated": N, "bytes_saved": N}
        """
        import shutil

        db = self._db()
        stats = {"moved": 0, "deduplicated": 0, "bytes_saved": 0}

        # Find components used by more than one model
        shared_rows = db.fetch_all(
            """
            SELECT c.id, c.component_type, c.content_hash, c.storage_path, c.size_bytes
            FROM component c
            JOIN model_component mc ON mc.component_id = c.id
            GROUP BY c.id
            HAVING COUNT(DISTINCT mc.model_id) > 1
            """
        )

        for comp_id, comp_type, content_hash, current_path, size_bytes in shared_rows:
            canonical_dir = os.path.join(self.components_base_dir, comp_type, content_hash)

            # Already in _components/?
            if os.path.realpath(current_path).startswith(os.path.realpath(self.components_base_dir)):
                # Still need to remove any model-local duplicates
                self._remove_local_duplicates(db, comp_id, comp_type)
                continue

            # Move the component to _components/
            if not os.path.exists(canonical_dir):
                os.makedirs(os.path.dirname(canonical_dir), exist_ok=True)
                if os.path.isdir(current_path) and not os.path.islink(current_path):
                    shutil.copytree(current_path, canonical_dir)
                    stats["moved"] += 1
                    logger.info("Compact: moved %s/%s to %s", comp_type, content_hash[:12], canonical_dir)
                else:
                    continue

            # Update DB to point to canonical location
            db.execute(
                "UPDATE component SET storage_path = ? WHERE id = ?",
                (canonical_dir, comp_id),
            )

            # Remove all model-local duplicates
            dedup_count = self._remove_local_duplicates(db, comp_id, comp_type)
            stats["deduplicated"] += dedup_count
            stats["bytes_saved"] += (size_bytes or 0) * max(0, dedup_count)

        return stats

    def _remove_local_duplicates(self, db: Database, comp_id: int, comp_type: str) -> int:
        """Remove model-local component directories that are now stored in canonical _components/ storage.

        Handles three cases:
        - Symlink (legacy): remove the symlink
        - Real directory (duplicate): delete the directory tree
        - Missing: no-op

        Returns the number of local copies removed.
        """
        import shutil

        rows = db.fetch_all(
            """
            SELECT m.filepath
            FROM model_component mc
            JOIN model m ON mc.model_id = m.id
            WHERE mc.component_id = ?
            """,
            (comp_id,),
        )

        removed_count = 0
        for (model_path,) in rows:
            local_dir = os.path.join(model_path, comp_type)

            if os.path.islink(local_dir):
                # Legacy symlink — remove it
                os.remove(local_dir)
                removed_count += 1
                logger.info("Compact: removed legacy symlink %s", local_dir)
            elif os.path.isdir(local_dir):
                # Real directory — duplicate data, remove it
                shutil.rmtree(local_dir)
                removed_count += 1
                logger.info("Compact: removed duplicate directory %s", local_dir)
            # Missing — nothing to do

        return removed_count

    def get_component_display_info(self, model_id: int) -> list[dict[str, str]]:
        """Return a list of {type, architecture, dtype_label} for display in the UI."""
        components = self.get_model_components(model_id)
        db = self._db()
        result = []
        for comp_type in COMPONENT_TYPES:
            info = components.get(comp_type)
            if info is None:
                continue
            if comp_type in ("tokenizer", "scheduler"):
                continue
            if info.dtype is None:
                info.dtype = self._detect_dtype(info.storage_path, info.config_json)
                if info.dtype:
                    db.execute("UPDATE component SET dtype = ? WHERE id = ?", (info.dtype, info.id))
            dtype_label = self._format_dtype_label(info.dtype, info.config_json)
            result.append(
                {
                    "type": comp_type,
                    "architecture": info.architecture or "",
                    "dtype_label": dtype_label,
                }
            )
        return result

    @staticmethod
    def _format_dtype_label(dtype: str | None, config_json: str | None) -> str:
        """Build a human-readable dtype/quantization label for display."""
        # Check config_json for quantization info (overrides raw dtype)
        if config_json:
            try:
                cfg = json.loads(config_json)
                quant_cfg = cfg.get("quantization_config")
                if quant_cfg:
                    method = quant_cfg.get("quant_method", "")
                    if method == "sdnq":
                        # SDNQ stores e.g. "int4", "int8" in weights_dtype
                        weights_dtype = quant_cfg.get("weights_dtype", "")
                        return f"sdnq {weights_dtype}" if weights_dtype else "sdnq"
                    if method in ("bitsandbytes", "bnb"):
                        bits = quant_cfg.get("load_in_4bit") and 4 or quant_cfg.get("load_in_8bit") and 8 or 4
                        return f"bnb {bits}-bit"
                    bits = quant_cfg.get("bits", quant_cfg.get("nbits", ""))
                    if bits:
                        return f"{method} {bits}-bit"
                    return method
            except Exception:
                pass

        return dtype or ""

    @staticmethod
    def _detect_dtype(source_path: str, config_json: str | None) -> str:
        """Detect dtype from config_json fields or safetensors header."""
        import glob as glob_mod
        import struct

        # Try config_json first: torch_dtype or dtype field
        if config_json:
            try:
                cfg = json.loads(config_json)
                torch_dtype = cfg.get("torch_dtype") or cfg.get("dtype") or ""
                if torch_dtype:
                    return str(torch_dtype).removeprefix("torch.")
            except Exception:
                pass

        # Fall back to safetensors header
        if source_path and os.path.isdir(source_path):
            files = glob_mod.glob(os.path.join(source_path, "*.safetensors"))
            if files:
                try:
                    with open(files[0], "rb") as f:
                        header_size = struct.unpack("<Q", f.read(8))[0]
                        header = json.loads(f.read(header_size))
                    for key, value in header.items():
                        if key != "__metadata__" and isinstance(value, dict):
                            dtype = value.get("dtype", "")
                            dtype_map = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}
                            return dtype_map.get(dtype, dtype.lower())
                except Exception:
                    pass

        return ""

    def get_component_variants(self, model_id: int, component_type: str) -> list[ComponentInfo]:
        """Return the default component plus any explicitly registered variants for this model."""
        components = self.get_model_components(model_id)
        current = components.get(component_type)

        db = self._db()
        rows = db.fetch_all(
            """
            SELECT c.id, c.component_type, c.content_hash, c.storage_path,
                   c.size_bytes, c.architecture, c.config_json, c.dtype
            FROM model_component_variant mcv
            JOIN component c ON mcv.component_id = c.id
            WHERE mcv.model_id = ? AND mcv.component_type = ?
            """,
            (model_id, component_type),
        )

        variant_components = [
            ComponentInfo(
                id=r[0], component_type=r[1], content_hash=r[2], storage_path=r[3],
                size_bytes=r[4] or 0, architecture=r[5], config_json=r[6], dtype=r[7],
            )
            for r in rows
        ]

        override_id = self.get_component_override(model_id, component_type)
        active_id = override_id if override_id is not None else (current.id if current else None)

        result = []
        if current:
            result.append(current)
        for v in variant_components:
            if current and v.id == current.id:
                continue
            result.append(v)

        if active_id and result and result[0].id != active_id:
            for i, v in enumerate(result):
                if v.id == active_id:
                    result.insert(0, result.pop(i))
                    break

        return result

    def add_component_variant(self, model_id: int, component_type: str, component_id: int) -> None:
        """Register a component as an available variant for a model's slot."""
        db = self._db()
        db.execute(
            "INSERT OR IGNORE INTO model_component_variant (model_id, component_type, component_id) VALUES (?, ?, ?)",
            (model_id, component_type, component_id),
        )

    def add_component_variant_to_sharing_models(
        self, model_id: int, component_type: str, component_id: int
    ) -> list[int]:
        """Register a variant for the given model AND all models sharing the same default component."""
        db = self._db()

        row = db.fetch_one(
            "SELECT component_id FROM model_component WHERE model_id = ? AND component_type = ?",
            (model_id, component_type),
        )
        if row is None:
            self.add_component_variant(model_id, component_type, component_id)
            return [model_id]

        default_component_id = row[0]

        rows = db.fetch_all(
            """
            SELECT mc.model_id FROM model_component mc
            JOIN model m ON mc.model_id = m.id
            WHERE mc.component_type = ? AND mc.component_id = ? AND m.deleted = 0
            """,
            (component_type, default_component_id),
        )

        affected_model_ids = [r[0] for r in rows]
        if model_id not in affected_model_ids:
            affected_model_ids.append(model_id)

        for mid in affected_model_ids:
            db.execute(
                "INSERT OR IGNORE INTO model_component_variant (model_id, component_type, component_id) VALUES (?, ?, ?)",
                (mid, component_type, component_id),
            )

        return affected_model_ids

    def get_compatible_model_ids(self, component_type: str, architecture: str) -> list[tuple[int, str]]:
        """Return (model_id, model_name) pairs for models compatible with a component."""
        from frameartisan.modules.generation.component_compatibility import ARCHITECTURE_COMPATIBILITY

        compatible_transformer_archs = []
        for trans_arch, compat in ARCHITECTURE_COMPATIBILITY.items():
            if component_type == "transformer":
                if trans_arch == architecture:
                    compatible_transformer_archs.append(trans_arch)
            else:
                if architecture in compat.get(component_type, []):
                    compatible_transformer_archs.append(trans_arch)

        if not compatible_transformer_archs:
            return []

        db = self._db()
        placeholders = ", ".join(["?"] * len(compatible_transformer_archs))
        rows = db.fetch_all(
            f"""
            SELECT DISTINCT m.id, m.name
            FROM model m
            JOIN model_component mc ON mc.model_id = m.id
            JOIN component c ON mc.component_id = c.id
            WHERE mc.component_type = 'transformer'
              AND c.architecture IN ({placeholders})
              AND m.deleted = 0
            ORDER BY m.name
            """,
            tuple(compatible_transformer_archs),
        )
        return [(r[0], r[1]) for r in rows]

    def get_component_override(self, model_id: int, component_type: str) -> int | None:
        """Get the override component_id for a model+component_type, or None."""
        db = self._db()
        row = db.fetch_one(
            "SELECT component_id FROM model_component_override WHERE model_id = ? AND component_type = ?",
            (model_id, component_type),
        )
        return row[0] if row else None

    def set_component_override(self, model_id: int, component_type: str, component_id: int) -> None:
        """Set an override component for a model slot."""
        db = self._db()
        db.execute(
            "INSERT OR REPLACE INTO model_component_override (model_id, component_type, component_id) VALUES (?, ?, ?)",
            (model_id, component_type, component_id),
        )

    def clear_component_override(self, model_id: int, component_type: str) -> None:
        """Remove the override for a model slot (reverts to default)."""
        db = self._db()
        db.execute(
            "DELETE FROM model_component_override WHERE model_id = ? AND component_type = ?",
            (model_id, component_type),
        )

    def resolve_model_components(self, model_id: int) -> dict[str, ComponentInfo]:
        """Get model components with overrides applied."""
        defaults = self.get_model_components(model_id)
        db = self._db()

        override_rows = db.fetch_all(
            "SELECT component_type, component_id FROM model_component_override WHERE model_id = ?",
            (model_id,),
        )

        if not override_rows:
            return defaults

        for comp_type, comp_id in override_rows:
            row = db.fetch_one(
                """
                SELECT id, component_type, content_hash, storage_path,
                       size_bytes, architecture, config_json, dtype
                FROM component WHERE id = ?
                """,
                (comp_id,),
            )
            if row:
                defaults[comp_type] = ComponentInfo(
                    id=row[0], component_type=row[1], content_hash=row[2], storage_path=row[3],
                    size_bytes=row[4] or 0, architecture=row[5], config_json=row[6], dtype=row[7],
                )

        return defaults

    def resolve_model_components_with_overrides(
        self, model_id: int, overrides: dict[str, int]
    ) -> dict[str, ComponentInfo]:
        """Get model components with explicit overrides applied (no DB lookup)."""
        defaults = self.get_model_components(model_id)
        db = self._db()

        for comp_type, comp_id in overrides.items():
            row = db.fetch_one(
                """
                SELECT id, component_type, content_hash, storage_path,
                       size_bytes, architecture, config_json, dtype
                FROM component WHERE id = ?
                """,
                (comp_id,),
            )
            if row:
                defaults[comp_type] = ComponentInfo(
                    id=row[0], component_type=row[1], content_hash=row[2], storage_path=row[3],
                    size_bytes=row[4] or 0, architecture=row[5], config_json=row[6], dtype=row[7],
                )

        return defaults

    def cleanup_after_registration(self, model_id: int, model_path: str) -> None:
        """Clean up model-local component copies that already exist elsewhere.

        Call this after registering a new model's components. For each component:
        - If canonical storage IS this model's directory, nothing to do (first model).
        - If canonical storage is in _components/, just remove the local copy.
        - If canonical storage is in another model's directory, move it to
          _components/ first, then remove all model-local copies.
        """
        import shutil

        components = self.get_model_components(model_id)
        components_base_real = os.path.realpath(self.components_base_dir)
        db = self._db()

        for comp_type, info in components.items():
            local_dir = os.path.join(model_path, comp_type)
            canonical_real = os.path.realpath(info.storage_path)
            local_real = os.path.realpath(local_dir)

            # Canonical storage IS this model's directory — first model with this component
            if canonical_real == local_real:
                continue

            # Canonical storage is already in _components/ — just remove local copy
            if canonical_real.startswith(components_base_real):
                if os.path.islink(local_dir):
                    os.remove(local_dir)
                    logger.info("Cleanup: removed legacy symlink %s", local_dir)
                elif os.path.isdir(local_dir):
                    shutil.rmtree(local_dir)
                    logger.info("Cleanup: removed duplicate %s (canonical in _components/)", local_dir)
                continue

            # Canonical storage is in another model's directory — move to _components/
            canonical_target = os.path.join(self.components_base_dir, comp_type, info.content_hash)
            if not os.path.exists(canonical_target):
                os.makedirs(os.path.dirname(canonical_target), exist_ok=True)
                if os.path.isdir(info.storage_path):
                    shutil.copytree(info.storage_path, canonical_target)
                else:
                    # Source gone — use local copy as the source instead
                    if os.path.isdir(local_dir):
                        shutil.copytree(local_dir, canonical_target)
                    else:
                        continue

            # Update DB to point to _components/
            db.execute(
                "UPDATE component SET storage_path = ? WHERE id = ?",
                (canonical_target, info.id),
            )
            logger.info("Cleanup: moved %s/%s to _components/", comp_type, info.content_hash[:12])

            # Remove all model-local copies (including the original model's)
            self._remove_local_duplicates(db, info.id, comp_type)
