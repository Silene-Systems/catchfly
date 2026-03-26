"""Checkpoint support for pipeline resume."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from catchfly._types import Schema

logger = logging.getLogger(__name__)


class _Checkpoint:
    """Manages pipeline checkpoint state on disk.

    Files:
    - schema.json — discovered schema
    - records.jsonl — extracted records (one per line, append-safe)
    - state.json — set of processed document IDs
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def _schema_path(self) -> Path:
        return self._dir / "schema.json"

    @property
    def _records_path(self) -> Path:
        return self._dir / "records.jsonl"

    @property
    def _state_path(self) -> Path:
        return self._dir / "state.json"

    def save_schema(self, schema: Schema) -> None:
        """Persist discovered schema."""
        data = {
            "json_schema": schema.json_schema,
            "field_metadata": schema.field_metadata,
            "lineage": schema.lineage,
        }
        self._schema_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Checkpoint: saved schema to %s", self._schema_path)

    def load_schema(self) -> Schema | None:
        """Load schema from checkpoint, or None if not found."""
        if not self._schema_path.exists():
            return None

        try:
            data = json.loads(self._schema_path.read_text(encoding="utf-8"))
            from catchfly.schema.converters import json_schema_to_pydantic

            json_schema = data["json_schema"]
            model = None
            try:
                model = json_schema_to_pydantic(json_schema, "CheckpointSchema")
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(
                    "Checkpoint: could not reconstruct Pydantic model, "
                    "falling back to JSON Schema only: %s",
                    e,
                )

            return Schema(
                model=model,
                json_schema=json_schema,
                field_metadata=data.get("field_metadata", {}),
                lineage=data.get("lineage", []),
            )
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning("Checkpoint: failed to load schema: %s", e, exc_info=True)
            return None

    def append_record(self, record: Any) -> None:
        """Append a single record to the JSONL file."""
        if hasattr(record, "model_dump"):
            data = record.model_dump()
        elif isinstance(record, dict):
            data = record
        else:
            data = {"_raw": str(record)}

        with open(self._records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def load_records(self) -> list[dict[str, Any]]:
        """Load all records from checkpoint JSONL."""
        if not self._records_path.exists():
            return []

        records: list[dict[str, Any]] = []
        for line in self._records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def mark_processed(self, doc_id: str) -> None:
        """Mark a document ID as processed."""
        ids = self.load_processed_ids()
        ids.add(doc_id)
        self._state_path.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")

    def load_processed_ids(self) -> set[str]:
        """Load the set of already-processed document IDs."""
        if not self._state_path.exists():
            return set()

        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return set(data) if isinstance(data, list) else set()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Checkpoint: failed to load processed IDs: %s", e)
            return set()
