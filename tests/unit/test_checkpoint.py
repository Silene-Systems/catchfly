"""Tests for pipeline checkpoint/resume."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

from catchfly._types import Schema

if TYPE_CHECKING:
    from pathlib import Path
from catchfly.pipeline import _Checkpoint


class TestCheckpoint:
    def test_save_and_load_schema(self, tmp_path: Path) -> None:
        cp = _Checkpoint(tmp_path / "state")
        schema = Schema(
            model=None,
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            field_metadata={"name": {"description": "A name"}},
            lineage=["SinglePassDiscovery"],
        )

        cp.save_schema(schema)
        loaded = cp.load_schema()

        assert loaded is not None
        assert loaded.json_schema["properties"]["name"]["type"] == "string"
        assert loaded.field_metadata["name"]["description"] == "A name"
        assert loaded.lineage == ["SinglePassDiscovery"]

    def test_load_schema_nonexistent(self, tmp_path: Path) -> None:
        cp = _Checkpoint(tmp_path / "empty")
        assert cp.load_schema() is None

    def test_append_and_load_records(self, tmp_path: Path) -> None:
        cp = _Checkpoint(tmp_path / "state")

        class Item(BaseModel):
            name: str
            value: int

        cp.append_record(Item(name="a", value=1))
        cp.append_record(Item(name="b", value=2))
        cp.append_record({"name": "c", "value": 3})

        records = cp.load_records()
        assert len(records) == 3
        assert records[0]["name"] == "a"
        assert records[2]["value"] == 3

    def test_load_records_empty(self, tmp_path: Path) -> None:
        cp = _Checkpoint(tmp_path / "empty")
        assert cp.load_records() == []

    def test_mark_and_load_processed(self, tmp_path: Path) -> None:
        cp = _Checkpoint(tmp_path / "state")

        cp.mark_processed("doc1")
        cp.mark_processed("doc2")
        cp.mark_processed("doc1")  # duplicate

        ids = cp.load_processed_ids()
        assert ids == {"doc1", "doc2"}

    def test_load_processed_empty(self, tmp_path: Path) -> None:
        cp = _Checkpoint(tmp_path / "empty")
        assert cp.load_processed_ids() == set()

    def test_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        _Checkpoint(nested)
        assert nested.exists()

    def test_records_jsonl_is_append_safe(self, tmp_path: Path) -> None:
        """Multiple appends create valid JSONL."""
        cp = _Checkpoint(tmp_path / "state")
        cp.append_record({"x": 1})
        cp.append_record({"x": 2})

        # Read raw file — should be valid JSONL
        raw = (tmp_path / "state" / "records.jsonl").read_text()
        lines = [line for line in raw.strip().split("\n") if line]
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"x": 1}
        assert json.loads(lines[1]) == {"x": 2}

    def test_schema_roundtrip_with_pydantic_model(self, tmp_path: Path) -> None:
        """Schema with model class survives save/load (model rebuilt from json_schema)."""
        cp = _Checkpoint(tmp_path / "state")
        schema = Schema(
            model=None,
            json_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rating": {"type": "number"},
                },
                "required": ["title"],
            },
            lineage=["test"],
        )

        cp.save_schema(schema)
        loaded = cp.load_schema()

        assert loaded is not None
        # Pydantic model should be rebuilt from json_schema
        assert loaded.model is not None
        instance = loaded.model(title="test", rating=5.0)
        assert instance.title == "test"  # type: ignore[attr-defined]
