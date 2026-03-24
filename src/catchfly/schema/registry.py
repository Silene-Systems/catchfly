"""Schema registry — version and track schemas across pipeline runs."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from catchfly._types import Schema

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """In-memory schema registry with optional persistence.

    Stores schemas by name with automatic versioning, tracks lineage,
    and provides diff between schema versions.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        """Initialize registry.

        Args:
            persist_path: Optional JSON file path for persistence.
                          Schemas are saved on each register() call.
        """
        self._store: dict[str, list[_SchemaEntry]] = {}
        self._persist_path = persist_path

        if persist_path:
            self._load_from_disk()

    def register(self, schema: Schema, name: str | None = None) -> str:
        """Register a schema, assigning a version.

        Args:
            schema: The schema to register.
            name: Schema name (auto-generated from lineage if None).

        Returns:
            Version string (e.g. "product_reviews:v3").
        """
        if name is None:
            name = self._auto_name(schema)

        if name not in self._store:
            self._store[name] = []

        version = len(self._store[name]) + 1
        entry = _SchemaEntry(
            schema=schema,
            version=version,
            timestamp=time.time(),
            name=name,
        )
        self._store[name].append(entry)

        version_str = f"{name}:v{version}"
        logger.info("SchemaRegistry: registered '%s'", version_str)

        if self._persist_path:
            self._save_to_disk()

        return version_str

    def get(self, name: str, version: int | None = None) -> Schema | None:
        """Retrieve a schema by name.

        Args:
            name: Schema name.
            version: Specific version (1-indexed). None = latest.

        Returns:
            Schema or None if not found.
        """
        entries = self._store.get(name)
        if not entries:
            return None

        if version is None:
            return entries[-1].schema

        if 1 <= version <= len(entries):
            return entries[version - 1].schema

        return None

    def list_schemas(self) -> list[dict[str, Any]]:
        """List all registered schemas with metadata."""
        result: list[dict[str, Any]] = []
        for name, entries in self._store.items():
            for entry in entries:
                result.append(
                    {
                        "name": name,
                        "version": entry.version,
                        "timestamp": entry.timestamp,
                        "lineage": entry.schema.lineage,
                        "num_fields": len(entry.schema.json_schema.get("properties", {})),
                    }
                )
        return result

    @staticmethod
    def diff(schema_a: Schema, schema_b: Schema) -> dict[str, Any]:
        """Compute field-level diff between two schemas.

        Returns:
            Dict with 'added', 'removed', and 'changed' fields.
        """
        props_a = set(schema_a.json_schema.get("properties", {}).keys())
        props_b = set(schema_b.json_schema.get("properties", {}).keys())

        added = props_b - props_a
        removed = props_a - props_b
        common = props_a & props_b

        changed: dict[str, dict[str, Any]] = {}
        for field in common:
            def_a = schema_a.json_schema["properties"][field]
            def_b = schema_b.json_schema["properties"][field]
            if def_a != def_b:
                changed[field] = {"before": def_a, "after": def_b}

        return {
            "added": sorted(added),
            "removed": sorted(removed),
            "changed": changed,
        }

    @staticmethod
    def _auto_name(schema: Schema) -> str:
        """Generate a name from schema lineage or properties."""
        if schema.lineage:
            return schema.lineage[0].split(":")[0].lower()
        props = schema.json_schema.get("properties", {})
        if props:
            return f"schema_{len(props)}fields"
        return "unnamed"

    def _save_to_disk(self) -> None:
        """Persist registry to JSON file."""
        if not self._persist_path:
            return

        data: list[dict[str, Any]] = []
        for name, entries in self._store.items():
            for entry in entries:
                data.append(
                    {
                        "name": name,
                        "version": entry.version,
                        "timestamp": entry.timestamp,
                        "json_schema": entry.schema.json_schema,
                        "field_metadata": entry.schema.field_metadata,
                        "lineage": entry.schema.lineage,
                    }
                )

        from pathlib import Path

        Path(self._persist_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_from_disk(self) -> None:
        """Load registry from JSON file."""
        if not self._persist_path:
            return

        from pathlib import Path

        path = Path(self._persist_path)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                schema = Schema(
                    model=None,
                    json_schema=item["json_schema"],
                    field_metadata=item.get("field_metadata", {}),
                    lineage=item.get("lineage", []),
                )
                name = item["name"]
                if name not in self._store:
                    self._store[name] = []
                self._store[name].append(
                    _SchemaEntry(
                        schema=schema,
                        version=item["version"],
                        timestamp=item["timestamp"],
                        name=name,
                    )
                )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "SchemaRegistry: failed to load from %s: %s",
                self._persist_path,
                exc,
            )


class _SchemaEntry:
    """Internal entry in the schema registry."""

    __slots__ = ("schema", "version", "timestamp", "name")

    def __init__(
        self,
        schema: Schema,
        version: int,
        timestamp: float,
        name: str,
    ) -> None:
        self.schema = schema
        self.version = version
        self.timestamp = timestamp
        self.name = name
