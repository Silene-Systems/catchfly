"""Tests for catchfly.discovery.refine — schema refinement API."""

from __future__ import annotations

import json
from typing import Any

import pytest

from catchfly._types import Schema
from catchfly.discovery.refine import (
    SchemaFeedback,
    arefine_schema,
)
from catchfly.exceptions import DiscoveryError
from catchfly.providers.llm import LLMResponse


def _make_schema() -> Schema:
    json_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "rating": {"type": "number"},
            "author": {"type": "string"},
        },
        "required": ["title"],
    }
    return Schema(
        model=None,
        json_schema=json_schema,
        lineage=["SinglePassDiscovery"],
    )


class TestSchemaFeedback:
    def test_empty_feedback_is_empty(self) -> None:
        assert SchemaFeedback().is_empty() is True

    def test_feedback_with_add_not_empty(self) -> None:
        assert SchemaFeedback(add_fields={"x": {"type": "string"}}).is_empty() is False

    def test_feedback_with_instruction_not_empty(self) -> None:
        assert SchemaFeedback(instruction="remove metadata").is_empty() is False


class TestExplicitRefinement:
    """Deterministic refinement — no LLM calls."""

    async def test_add_fields(self) -> None:
        schema = _make_schema()
        feedback = SchemaFeedback(
            add_fields={"price": {"type": "number", "description": "Price"}},
        )
        refined = await arefine_schema(schema, feedback)
        assert "price" in refined.json_schema["properties"]
        assert "title" in refined.json_schema["properties"]
        assert refined.lineage[-1] == "refine:explicit"

    async def test_remove_fields(self) -> None:
        schema = _make_schema()
        feedback = SchemaFeedback(remove_fields=["author"])
        refined = await arefine_schema(schema, feedback)
        assert "author" not in refined.json_schema["properties"]
        assert "title" in refined.json_schema["properties"]

    async def test_modify_fields(self) -> None:
        schema = _make_schema()
        feedback = SchemaFeedback(
            modify_fields={
                "rating": {"type": "integer", "description": "Rating 1-5"}
            },
        )
        refined = await arefine_schema(schema, feedback)
        assert refined.json_schema["properties"]["rating"]["type"] == "integer"
        assert (
            refined.json_schema["properties"]["rating"]["description"] == "Rating 1-5"
        )

    async def test_combined_ops(self) -> None:
        schema = _make_schema()
        feedback = SchemaFeedback(
            add_fields={"price": {"type": "number"}},
            remove_fields=["author"],
            modify_fields={"rating": {"type": "integer"}},
        )
        refined = await arefine_schema(schema, feedback)
        props = refined.json_schema["properties"]
        assert "price" in props
        assert "author" not in props
        assert props["rating"]["type"] == "integer"

    async def test_empty_feedback_is_noop(self) -> None:
        schema = _make_schema()
        refined = await arefine_schema(schema, SchemaFeedback())
        assert refined.json_schema == schema.json_schema
        assert refined.lineage[-1] == "refine:noop"

    async def test_remove_all_triggers_invariant(self) -> None:
        """Trying to remove every field reverts via the empty-properties invariant."""
        schema = _make_schema()
        feedback = SchemaFeedback(remove_fields=["title", "rating", "author"])
        refined = await arefine_schema(schema, feedback)
        # _apply_changes reverted to input — all fields preserved
        assert "title" in refined.json_schema["properties"]
        assert "rating" in refined.json_schema["properties"]
        assert "author" in refined.json_schema["properties"]

    async def test_refined_schema_has_pydantic_model(self) -> None:
        """Refinement rebuilds the Pydantic model (hard-fail if broken)."""
        schema = _make_schema()
        refined = await arefine_schema(schema, SchemaFeedback(remove_fields=["author"]))
        assert refined.model is not None
        instance = refined.model(title="t", rating=4)
        assert instance.title == "t"  # type: ignore[attr-defined]

    async def test_refinement_does_not_mutate_input(self) -> None:
        schema = _make_schema()
        original_props = dict(schema.json_schema["properties"])
        await arefine_schema(schema, SchemaFeedback(remove_fields=["author"]))
        assert schema.json_schema["properties"] == original_props

    async def test_lineage_extended(self) -> None:
        schema = _make_schema()
        refined = await arefine_schema(
            schema, SchemaFeedback(add_fields={"x": {"type": "string"}})
        )
        assert refined.lineage[:-1] == schema.lineage
        assert refined.lineage[-1] == "refine:explicit"


class TestInstructionRefinement:
    """LLM-guided refinement via natural-language instruction."""

    async def test_instruction_mode_calls_llm(self) -> None:
        """Instruction populates the system prompt; LLM response is applied."""
        captured_prompts: list[str] = []

        class InstructionLLM:
            async def acomplete(
                self, messages: list[dict[str, str]], **kw: Any
            ) -> LLMResponse:
                captured_prompts.append(messages[-1]["content"])
                return LLMResponse(
                    content=json.dumps(
                        {
                            "add_fields": {},
                            "remove_fields": ["author"],
                            "modify_fields": {},
                            "rationale": "author is administrative",
                        }
                    ),
                    input_tokens=100,
                    output_tokens=50,
                )

            async def astructured_complete(
                self, messages: Any, output_schema: Any, **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        schema = _make_schema()
        refined = await arefine_schema(
            schema,
            SchemaFeedback(instruction="drop administrative fields"),
            client=InstructionLLM(),
        )
        assert "author" not in refined.json_schema["properties"]
        assert "title" in refined.json_schema["properties"]
        assert any("drop administrative fields" in p for p in captured_prompts)
        assert "refine:instruction" in refined.lineage

    async def test_instruction_plus_explicit_compose(self) -> None:
        """Instruction runs first, then explicit ops layer on top."""

        class InstructionLLM:
            async def acomplete(
                self, messages: list[dict[str, str]], **kw: Any
            ) -> LLMResponse:
                # LLM removes "author" per instruction
                return LLMResponse(
                    content=json.dumps(
                        {
                            "add_fields": {},
                            "remove_fields": ["author"],
                            "modify_fields": {},
                        }
                    ),
                    input_tokens=100,
                    output_tokens=50,
                )

            async def astructured_complete(
                self, messages: Any, output_schema: Any, **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        schema = _make_schema()
        feedback = SchemaFeedback(
            instruction="drop administrative fields",
            add_fields={"price": {"type": "number"}},
        )
        refined = await arefine_schema(schema, feedback, client=InstructionLLM())
        props = refined.json_schema["properties"]
        assert "author" not in props  # removed by instruction
        assert "price" in props  # added by explicit
        # Both steps appear in lineage
        assert "refine:instruction" in refined.lineage
        assert "refine:explicit" in refined.lineage

    async def test_instruction_llm_error_raises_discovery_error(self) -> None:
        """A failing LLM call surfaces as DiscoveryError."""

        class FailingLLM:
            async def acomplete(self, messages: Any, **kw: Any) -> LLMResponse:
                raise ValueError("network down")

            async def astructured_complete(
                self, messages: Any, output_schema: Any, **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        schema = _make_schema()
        with pytest.raises(DiscoveryError, match="LLM call failed"):
            await arefine_schema(
                schema,
                SchemaFeedback(instruction="drop everything"),
                client=FailingLLM(),
            )


class TestSyncWrapper:
    def test_refine_schema_sync(self) -> None:
        from catchfly.discovery.refine import refine_schema

        schema = _make_schema()
        refined = refine_schema(schema, SchemaFeedback(remove_fields=["author"]))
        assert "author" not in refined.json_schema["properties"]


class TestModuleExports:
    def test_package_exports_refine_symbols(self) -> None:
        from catchfly.discovery import (
            SchemaFeedback as FB,
        )
        from catchfly.discovery import (
            arefine_schema as fn_a,
        )
        from catchfly.discovery import (
            refine_schema as fn_s,
        )

        assert FB is SchemaFeedback
        assert callable(fn_a)
        assert callable(fn_s)
