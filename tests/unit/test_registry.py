"""Tests for SchemaRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catchfly._types import Schema
from catchfly.schema.registry import SchemaRegistry

if TYPE_CHECKING:
    from pathlib import Path


def _make_schema(fields: list[str], lineage: str = "test") -> Schema:
    return Schema(
        model=None,
        json_schema={
            "type": "object",
            "properties": {f: {"type": "string"} for f in fields},
        },
        lineage=[lineage],
    )


class TestSchemaRegistry:
    def test_register_and_get(self) -> None:
        reg = SchemaRegistry()
        schema = _make_schema(["name", "age"])
        version = reg.register(schema, name="person")

        assert version == "person:v1"
        retrieved = reg.get("person")
        assert retrieved is not None
        assert "name" in retrieved.json_schema["properties"]

    def test_auto_versioning(self) -> None:
        reg = SchemaRegistry()
        v1 = reg.register(_make_schema(["name"]), name="person")
        v2 = reg.register(_make_schema(["name", "age"]), name="person")

        assert v1 == "person:v1"
        assert v2 == "person:v2"

    def test_get_specific_version(self) -> None:
        reg = SchemaRegistry()
        reg.register(_make_schema(["name"]), name="person")
        reg.register(_make_schema(["name", "age"]), name="person")

        v1 = reg.get("person", version=1)
        v2 = reg.get("person", version=2)

        assert v1 is not None
        assert len(v1.json_schema["properties"]) == 1
        assert v2 is not None
        assert len(v2.json_schema["properties"]) == 2

    def test_get_latest(self) -> None:
        reg = SchemaRegistry()
        reg.register(_make_schema(["a"]), name="test")
        reg.register(_make_schema(["a", "b"]), name="test")

        latest = reg.get("test")
        assert latest is not None
        assert len(latest.json_schema["properties"]) == 2

    def test_get_nonexistent(self) -> None:
        reg = SchemaRegistry()
        assert reg.get("nope") is None

    def test_get_invalid_version(self) -> None:
        reg = SchemaRegistry()
        reg.register(_make_schema(["a"]), name="test")
        assert reg.get("test", version=99) is None

    def test_list_schemas(self) -> None:
        reg = SchemaRegistry()
        reg.register(_make_schema(["a"]), name="alpha")
        reg.register(_make_schema(["a", "b"]), name="alpha")
        reg.register(_make_schema(["x"]), name="beta")

        listing = reg.list_schemas()
        assert len(listing) == 3
        assert listing[0]["name"] == "alpha"
        assert listing[0]["version"] == 1
        assert listing[2]["name"] == "beta"

    def test_diff_added(self) -> None:
        a = _make_schema(["name"])
        b = _make_schema(["name", "age"])
        diff = SchemaRegistry.diff(a, b)

        assert "age" in diff["added"]
        assert diff["removed"] == []
        assert diff["changed"] == {}

    def test_diff_removed(self) -> None:
        a = _make_schema(["name", "age"])
        b = _make_schema(["name"])
        diff = SchemaRegistry.diff(a, b)

        assert "age" in diff["removed"]
        assert diff["added"] == []

    def test_diff_changed(self) -> None:
        a = Schema(
            model=None,
            json_schema={
                "type": "object",
                "properties": {"rating": {"type": "string"}},
            },
        )
        b = Schema(
            model=None,
            json_schema={
                "type": "object",
                "properties": {"rating": {"type": "number"}},
            },
        )
        diff = SchemaRegistry.diff(a, b)

        assert "rating" in diff["changed"]
        assert diff["changed"]["rating"]["before"]["type"] == "string"
        assert diff["changed"]["rating"]["after"]["type"] == "number"

    def test_diff_identical(self) -> None:
        a = _make_schema(["name"])
        diff = SchemaRegistry.diff(a, a)
        assert diff == {"added": [], "removed": [], "changed": {}}

    def test_auto_name_from_lineage(self) -> None:
        schema = _make_schema(["a"], lineage="SinglePassDiscovery")
        name = SchemaRegistry._auto_name(schema)
        assert name == "singlepassdiscovery"

    def test_persistence(self, tmp_path: Path) -> None:
        path = str(tmp_path / "registry.json")

        # Save
        reg1 = SchemaRegistry(persist_path=path)
        reg1.register(_make_schema(["name", "age"]), name="person")
        reg1.register(_make_schema(["title"]), name="product")

        # Load fresh
        reg2 = SchemaRegistry(persist_path=path)
        assert reg2.get("person") is not None
        assert reg2.get("product") is not None
        assert len(reg2.list_schemas()) == 2

    def test_persistence_nonexistent_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "missing.json")
        reg = SchemaRegistry(persist_path=path)
        assert reg.list_schemas() == []
