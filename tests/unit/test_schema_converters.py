"""Tests for schema converters."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from catchfly.exceptions import SchemaError
from catchfly.schema.converters import (
    json_schema_to_pydantic,
    pydantic_to_json_schema,
    resolve_refs,
)


class SampleModel(BaseModel):
    name: str
    age: int
    score: float
    active: bool


class NestedModel(BaseModel):
    title: str
    tags: list[str]


class TestPydanticToJsonSchema:
    def test_basic(self) -> None:
        schema = pydantic_to_json_schema(SampleModel)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_with_list_field(self) -> None:
        schema = pydantic_to_json_schema(NestedModel)
        tags_prop = schema["properties"]["tags"]
        assert tags_prop["type"] == "array"


class TestJsonSchemaToPydantic:
    def test_roundtrip_basic(self) -> None:
        original_schema = pydantic_to_json_schema(SampleModel)
        model = json_schema_to_pydantic(original_schema, "Rebuilt")

        instance = model(name="Alice", age=30, score=9.5, active=True)
        assert instance.name == "Alice"  # type: ignore[attr-defined]
        assert instance.age == 30  # type: ignore[attr-defined]

    def test_optional_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        model = json_schema_to_pydantic(schema, "WithOptional")
        instance = model(name="Bob")
        assert instance.name == "Bob"  # type: ignore[attr-defined]
        assert instance.nickname is None  # type: ignore[attr-defined]

    def test_array_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        }
        model = json_schema_to_pydantic(schema, "WithArray")
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]  # type: ignore[attr-defined]

    def test_empty_properties_raises(self) -> None:
        with pytest.raises(SchemaError, match="no 'properties'"):
            json_schema_to_pydantic({"type": "object"})

    def test_nested_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
            "required": ["address"],
        }
        model = json_schema_to_pydantic(schema, "WithNested")
        instance = model(address={"city": "Warsaw"})
        assert instance.address.city == "Warsaw"  # type: ignore[attr-defined]

    def test_union_type_list(self) -> None:
        """Handle type as list, e.g. ["string", "null"] from Mistral."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
            },
            "required": ["name"],
        }
        model = json_schema_to_pydantic(schema, "WithUnion")
        instance = model(name="Alice", nickname="Ali")
        assert instance.name == "Alice"  # type: ignore[attr-defined]
        assert instance.nickname == "Ali"  # type: ignore[attr-defined]

        instance2 = model(name="Bob")
        assert instance2.age is None  # type: ignore[attr-defined]

    def test_anyof_type(self) -> None:
        """Handle anyOf construct."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
            },
            "required": ["value"],
        }
        model = json_schema_to_pydantic(schema, "WithAnyOf")
        instance = model(value="hello")
        assert instance.value == "hello"  # type: ignore[attr-defined]

    def test_required_field_accepts_none(self) -> None:
        """Required fields in discovered schemas should accept None.

        LLMs commonly return null for fields that don't appear in a
        particular document (e.g., patient_age when age is not mentioned).
        Required means 'present in most documents', not 'never null'.
        """
        schema = {
            "type": "object",
            "properties": {
                "patient_age": {"type": "integer"},
                "diagnosis": {"type": "string"},
            },
            "required": ["patient_age", "diagnosis"],
        }
        model = json_schema_to_pydantic(schema, "NullableRequired")

        # None should be accepted on required fields
        instance = model(patient_age=None, diagnosis=None)
        assert instance.patient_age is None  # type: ignore[attr-defined]
        assert instance.diagnosis is None  # type: ignore[attr-defined]

        # Normal values should still work
        instance2 = model(patient_age=42, diagnosis="pneumonia")
        assert instance2.patient_age == 42  # type: ignore[attr-defined]

    def test_required_field_still_required(self) -> None:
        """Required fields must be provided — omitting them should raise."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        model = json_schema_to_pydantic(schema, "StillRequired")

        # Omitting required field should raise
        with pytest.raises(Exception):  # noqa: B017 — ValidationError
            model()

        # Omitting optional field should be fine
        instance = model(name="Alice")
        assert instance.name == "Alice"  # type: ignore[attr-defined]
        assert instance.age is None  # type: ignore[attr-defined]

    def test_object_without_properties(self) -> None:
        """Object type without properties becomes dict."""
        schema = {
            "type": "object",
            "properties": {
                "metadata": {"type": "object"},
            },
            "required": ["metadata"],
        }
        model = json_schema_to_pydantic(schema, "WithFreeDict")
        instance = model(metadata={"key": "value"})
        assert instance.metadata == {"key": "value"}  # type: ignore[attr-defined]

    def test_json_schema_to_pydantic_with_defs(self) -> None:
        """json_schema_to_pydantic handles $ref/$defs transparently."""
        schema = {
            "$defs": {
                "Symptom": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "present": {"type": "boolean"},
                    },
                    "required": ["name", "present"],
                },
            },
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "symptoms": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Symptom"},
                },
            },
            "required": ["patient_id", "symptoms"],
        }
        model = json_schema_to_pydantic(schema, "Findings")
        instance = model(
            patient_id="P1",
            symptoms=[{"name": "splenomegaly", "present": True}],
        )
        assert instance.patient_id == "P1"  # type: ignore[attr-defined]
        assert instance.symptoms[0].name == "splenomegaly"  # type: ignore[attr-defined]
        assert instance.symptoms[0].present is True  # type: ignore[attr-defined]


class TestResolveRefs:
    def test_resolve_basic_ref(self) -> None:
        """$ref in array items is inlined, $defs stripped."""
        schema = {
            "$defs": {
                "Symptom": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "present": {"type": "boolean"},
                    },
                },
            },
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Symptom"},
                },
            },
        }
        resolved = resolve_refs(schema)
        assert "$defs" not in resolved
        assert "$ref" not in resolved["properties"]["symptoms"]["items"]
        items = resolved["properties"]["symptoms"]["items"]
        assert items["type"] == "object"
        assert "name" in items["properties"]
        assert "present" in items["properties"]

    def test_resolve_anyof_with_ref(self) -> None:
        """$ref inside anyOf is resolved."""
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            "type": "object",
            "properties": {
                "address": {
                    "anyOf": [
                        {"$ref": "#/$defs/Address"},
                        {"type": "null"},
                    ],
                },
            },
        }
        resolved = resolve_refs(schema)
        any_of = resolved["properties"]["address"]["anyOf"]
        assert any_of[0]["type"] == "object"
        assert "city" in any_of[0]["properties"]
        assert any_of[1]["type"] == "null"

    def test_resolve_no_refs_passthrough(self) -> None:
        """Schema without $ref passes through unchanged."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        resolved = resolve_refs(schema)
        assert resolved == schema

    def test_resolve_circular_ref(self) -> None:
        """Self-referential schema is truncated, not infinite loop."""
        schema = {
            "$defs": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/TreeNode"},
                        },
                    },
                },
            },
            "type": "object",
            "properties": {
                "root": {"$ref": "#/$defs/TreeNode"},
            },
        }
        resolved = resolve_refs(schema)
        # Should not raise — circular ref is truncated
        assert "$defs" not in resolved
        root = resolved["properties"]["root"]
        assert root["type"] == "object"
        assert "children" in root["properties"]

    def test_roundtrip_nested_pydantic(self) -> None:
        """Pydantic model with nested model survives JSON Schema roundtrip."""

        class Symptom(BaseModel):
            name: str
            present: bool

        class PatientFindings(BaseModel):
            patient_id: str
            symptoms: list[Symptom]

        schema_dict = pydantic_to_json_schema(PatientFindings)
        # Pydantic generates $ref/$defs for nested models
        assert "$defs" in schema_dict

        rebuilt = json_schema_to_pydantic(schema_dict, "Rebuilt")
        instance = rebuilt(
            patient_id="P1",
            symptoms=[
                {"name": "fever", "present": True},
                {"name": "splenomegaly", "present": False},
            ],
        )
        assert instance.patient_id == "P1"  # type: ignore[attr-defined]
        assert len(instance.symptoms) == 2  # type: ignore[attr-defined]
        assert instance.symptoms[0].name == "fever"  # type: ignore[attr-defined]
        assert instance.symptoms[0].present is True  # type: ignore[attr-defined]
        assert instance.symptoms[1].present is False  # type: ignore[attr-defined]

    def test_resolve_nested_ref_in_properties(self) -> None:
        """$ref used directly in a property (not inside array)."""
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                },
            },
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/$defs/Address"},
            },
        }
        resolved = resolve_refs(schema)
        addr = resolved["properties"]["address"]
        assert addr["type"] == "object"
        assert "city" in addr["properties"]
