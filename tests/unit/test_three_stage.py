"""Tests for ThreeStageDiscovery."""

from __future__ import annotations

import json
from typing import Any

import pytest

from catchfly._types import Document
from catchfly.discovery.three_stage import ThreeStageDiscovery
from catchfly.exceptions import DiscoveryError
from catchfly.providers.llm import LLMResponse


class MockThreeStageLLM:
    """Mock LLM that returns stage-appropriate responses."""

    def __init__(self) -> None:
        self._call_count = 0

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self._call_count += 1
        user_msg = messages[-1]["content"] if messages else ""
        sys_msg = messages[0]["content"] if messages else ""

        # Stage 1: initial schema (via SinglePassDiscovery)
        if "propose" in user_msg.lower() and "json schema" in user_msg.lower():
            return LLMResponse(
                content=json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "rating": {"type": "number"},
                            "category": {"type": "string"},
                        },
                        "required": ["title"],
                    }
                ),
                input_tokens=500,
                output_tokens=200,
            )

        # Refinement/expansion — detect by system prompt
        if "refinement" in sys_msg.lower() or "expansion" in sys_msg.lower():
            return LLMResponse(
                content=json.dumps(
                    {
                        "add_fields": {
                            "price": {"type": "number", "description": "Price in USD"},
                        },
                        "remove_fields": [],
                        "modify_fields": {},
                        "rationale": "Price appears in most documents",
                    }
                ),
                input_tokens=300,
                output_tokens=150,
            )

        # Extraction calls (default)
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Sample",
                    "rating": 4.5,
                    "category": "electronics",
                }
            ),
            input_tokens=200,
            output_tokens=100,
        )

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


def _make_docs(n: int = 20) -> list[Document]:
    return [
        Document(
            content=f"Product {i}: Great item, rating {i % 5 + 3}/5, price ${i * 10 + 99}",
            id=f"doc{i}",
        )
        for i in range(n)
    ]


def _patch(mock_llm: MockThreeStageLLM) -> tuple[Any, Any]:
    """Patch both SinglePass and ThreeStage to use mock."""
    import catchfly.discovery.single_pass as sp_mod
    import catchfly.discovery.three_stage as ts_mod

    orig_sp = sp_mod.OpenAICompatibleClient
    orig_ts = ts_mod.OpenAICompatibleClient
    sp_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    ts_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    return orig_sp, orig_ts


def _unpatch(orig_sp: Any, orig_ts: Any) -> None:
    import catchfly.discovery.single_pass as sp_mod
    import catchfly.discovery.three_stage as ts_mod

    sp_mod.OpenAICompatibleClient = orig_sp  # type: ignore[assignment]
    ts_mod.OpenAICompatibleClient = orig_ts  # type: ignore[assignment]


class TestThreeStageDiscovery:
    async def test_full_three_stages(self) -> None:
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(model="mock")

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = await discovery.adiscover(_make_docs(60), domain_hint="products")
        finally:
            _unpatch(orig_sp, orig_ts)

        assert schema.json_schema.get("properties") is not None
        # Stage 2+3 should have added "price" field
        assert "price" in schema.json_schema["properties"]
        assert "ThreeStageDiscovery:stage3" in schema.lineage
        # Report metadata should be in field_metadata
        report = schema.field_metadata.get("_discovery_report", {})
        assert report.get("stages_completed") == 3

    async def test_human_review_stops_at_stage2(self) -> None:
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(model="mock", human_review=True)

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = await discovery.adiscover(_make_docs(20))
        finally:
            _unpatch(orig_sp, orig_ts)

        assert "ThreeStageDiscovery:stage2" in schema.lineage
        report = schema.field_metadata.get("_discovery_report", {})
        assert report.get("stages_completed") == 2

    async def test_empty_docs_raises(self) -> None:
        discovery = ThreeStageDiscovery()
        with pytest.raises(DiscoveryError, match="No documents"):
            await discovery.adiscover([])

    async def test_few_docs_still_works(self) -> None:
        """Works even with fewer docs than stage sample sizes."""
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(
            model="mock",
            stage1_samples=3,
            stage2_samples=10,
            stage3_samples=50,
        )

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = await discovery.adiscover(_make_docs(5))
        finally:
            _unpatch(orig_sp, orig_ts)

        assert schema.json_schema.get("properties") is not None

    def test_sync_wrapper(self) -> None:
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(model="mock")

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = discovery.discover(_make_docs(10))
        finally:
            _unpatch(orig_sp, orig_ts)

        assert schema.json_schema.get("properties") is not None

    def test_compute_coverage(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string"},
                "rare": {"type": "string"},
            }
        }
        extracted = [
            {"name": "A", "rare": None},
            {"name": "B", "rare": None},
            {"name": "C", "rare": "val"},
        ]
        coverage = ThreeStageDiscovery._compute_coverage(schema, extracted)
        assert coverage["name"] == pytest.approx(1.0)
        assert coverage["rare"] == pytest.approx(1 / 3)

    def test_compute_coverage_empty(self) -> None:
        assert ThreeStageDiscovery._compute_coverage({}, []) == {}

    def test_apply_changes_add(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        changes = {
            "add_fields": {"age": {"type": "integer"}},
            "remove_fields": [],
            "modify_fields": {},
        }
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert "age" in result["properties"]
        assert "name" in result["properties"]

    def test_apply_changes_remove(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "junk": {"type": "string"},
            },
            "required": ["name", "junk"],
        }
        changes = {"remove_fields": ["junk"]}
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert "junk" not in result["properties"]
        assert "junk" not in result["required"]

    def test_apply_changes_modify(self) -> None:
        schema = {
            "type": "object",
            "properties": {"rating": {"type": "string"}},
        }
        changes = {"modify_fields": {"rating": {"type": "number"}}}
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert result["properties"]["rating"]["type"] == "number"

    def test_parse_changes_valid(self) -> None:
        content = json.dumps({"add_fields": {"x": {"type": "string"}}})
        result = ThreeStageDiscovery._parse_changes(content)
        assert "x" in result["add_fields"]

    def test_parse_changes_invalid_json(self) -> None:
        result = ThreeStageDiscovery._parse_changes("not json")
        assert result == {}

    def test_parse_changes_with_fences(self) -> None:
        inner = json.dumps({"add_fields": {"y": {"type": "integer"}}})
        content = f"```json\n{inner}\n```"
        result = ThreeStageDiscovery._parse_changes(content)
        assert "y" in result["add_fields"]

    def test_sample_fewer_than_n(self) -> None:
        docs = _make_docs(3)
        result = ThreeStageDiscovery._sample(docs, 10)
        assert len(result) == 3

    def test_sample_exact_n(self) -> None:
        docs = _make_docs(10)
        result = ThreeStageDiscovery._sample(docs, 5)
        assert len(result) == 5

    async def test_discovery_raises_on_empty_schema(self) -> None:
        """ThreeStageDiscovery should raise DiscoveryError if schema has no fields.

        Previously it silently returned an empty Schema with model=None,
        causing downstream extraction to crash with AttributeError.
        """

        class EmptySchemaLLM:
            async def acomplete(self, messages: list[dict[str, str]], **kw: Any) -> LLMResponse:
                return LLMResponse(
                    content=json.dumps({"type": "object", "properties": {}}),
                    input_tokens=100,
                    output_tokens=50,
                )

            async def astructured_complete(self, messages: Any, **kw: Any) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = EmptySchemaLLM()
        discovery = ThreeStageDiscovery(model="mock")
        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            with pytest.raises(DiscoveryError):
                await discovery.adiscover(_make_docs(10), domain_hint="test")
        finally:
            _unpatch(orig_sp, orig_ts)

    def test_build_schema_raises_on_empty_properties(self) -> None:
        """_build_schema raises DiscoveryError when properties is empty."""
        empty_schema: dict[str, Any] = {"type": "object", "properties": {}}
        with pytest.raises(DiscoveryError, match="no properties"):
            ThreeStageDiscovery._build_schema(empty_schema, {}, {"stages_completed": 1}, stage=1)

    def test_build_schema_raises_on_conversion_failure(self) -> None:
        """_build_schema raises DiscoveryError when Pydantic conversion fails."""
        bad_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": "not_a_dict"},
        }
        with pytest.raises(DiscoveryError, match="failed to build Pydantic model"):
            ThreeStageDiscovery._build_schema(bad_schema, {}, {"stages_completed": 2}, stage=2)

    # ------------------------------------------------------------------
    # Hardening: strictly-additive mode + empty-properties invariant
    # ------------------------------------------------------------------

    def test_apply_changes_strictly_additive_ignores_remove(self) -> None:
        """strictly_additive=True drops remove_fields from the change set."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
        }
        changes = {
            "add_fields": {"c": {"type": "integer"}},
            "remove_fields": ["a", "b"],
            "modify_fields": {},
        }
        result = ThreeStageDiscovery._apply_changes(
            schema, changes, strictly_additive=True
        )
        assert "a" in result["properties"]
        assert "b" in result["properties"]
        assert "c" in result["properties"]
        # required list should be unchanged (removals ignored)
        assert set(result["required"]) == {"a", "b"}

    def test_apply_changes_strictly_additive_ignores_modify(self) -> None:
        """strictly_additive=True drops modify_fields from the change set."""
        schema = {
            "type": "object",
            "properties": {"rating": {"type": "string"}},
        }
        changes = {
            "add_fields": {},
            "remove_fields": [],
            "modify_fields": {"rating": {"type": "number"}},
        }
        result = ThreeStageDiscovery._apply_changes(
            schema, changes, strictly_additive=True
        )
        # Type must NOT have changed
        assert result["properties"]["rating"]["type"] == "string"

    def test_apply_changes_reverts_when_emptied(self) -> None:
        """Invariant: refuse to reduce a non-empty schema to empty properties."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a"],
        }
        # LLM hallucinated: remove everything, add nothing
        changes = {"remove_fields": ["a", "b"]}
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert "a" in result["properties"]
        assert "b" in result["properties"]
        # required preserved as well
        assert "a" in result["required"]

    def test_apply_changes_revert_does_not_mutate_input(self) -> None:
        """The revert path returns a deep copy, never the input dict."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
        }
        changes = {"remove_fields": ["a"]}
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert result is not schema
        result["properties"]["a"]["type"] = "integer"
        # Original schema untouched
        assert schema["properties"]["a"]["type"] == "string"

    def test_apply_changes_partial_removal_allowed(self) -> None:
        """Sanity: removing SOME fields (not all) still works in default mode."""
        schema = {
            "type": "object",
            "properties": {
                "keep": {"type": "string"},
                "drop": {"type": "string"},
            },
        }
        changes = {"remove_fields": ["drop"]}
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert "keep" in result["properties"]
        assert "drop" not in result["properties"]

    def test_apply_changes_add_plus_remove_all_keeps_new(self) -> None:
        """add_fields saves the schema even when remove_fields clears everything."""
        schema = {
            "type": "object",
            "properties": {"old": {"type": "string"}},
        }
        changes = {
            "add_fields": {"new": {"type": "integer"}},
            "remove_fields": ["old"],
        }
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        # Final state has exactly {new}, not empty → invariant not triggered
        assert "new" in result["properties"]
        assert "old" not in result["properties"]

    def test_expansion_prompt_is_strictly_additive(self) -> None:
        """_EXPANSION_SYSTEM must not ask the LLM to remove or modify fields."""
        from catchfly.discovery.three_stage import _EXPANSION_SYSTEM

        lowered = _EXPANSION_SYSTEM.lower()
        # Must describe itself as additive
        assert "additive" in lowered
        # Must NOT instruct removal in its output shape
        assert "remove_fields" not in _EXPANSION_SYSTEM
        assert "modify_fields" not in _EXPANSION_SYSTEM
        assert "flag fields" not in lowered
        assert "for removal" not in lowered

    def test_min_samples_for_removal_default(self) -> None:
        """Default threshold is 5."""
        discovery = ThreeStageDiscovery()
        assert discovery.min_samples_for_removal == 5

    def test_focus_default_is_none(self) -> None:
        discovery = ThreeStageDiscovery()
        assert discovery.focus is None

    async def test_try_extraction_delegates_to_llm_direct(self) -> None:
        """_try_extraction uses LLMDirectExtraction under the hood.

        Verified by the shape of the extraction prompts reaching the LLM —
        LLMDirectExtraction uses 'JSON Schema to extract:' in its user
        prompt, which the old inline ``_try_extraction`` did not.
        """
        captured_user_prompts: list[str] = []

        class CapturingLLM:
            def __init__(self) -> None:
                self._call = 0

            async def acomplete(
                self, messages: list[dict[str, str]], **kw: Any
            ) -> LLMResponse:
                self._call += 1
                user_msg = messages[-1]["content"] if messages else ""
                sys_msg = messages[0]["content"] if messages else ""
                captured_user_prompts.append(user_msg)

                if "propose" in user_msg.lower() and "json schema" in user_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            }
                        ),
                        input_tokens=100,
                        output_tokens=50,
                    )
                if "refinement" in sys_msg.lower() or "expansion" in sys_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {"add_fields": {}, "remove_fields": [], "modify_fields": {}}
                        ),
                        input_tokens=50,
                        output_tokens=20,
                    )
                return LLMResponse(
                    content=json.dumps({"name": "Sample"}),
                    input_tokens=100,
                    output_tokens=50,
                )

            async def astructured_complete(
                self, messages: Any, output_schema: Any, **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = CapturingLLM()
        discovery = ThreeStageDiscovery(model="mock", human_review=True)

        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            await discovery.adiscover(_make_docs(10))
        finally:
            _unpatch(orig_sp, orig_ts)

        # At least one prompt should be an LLMDirectExtraction-style
        # extraction request (distinguishable by its header text).
        assert any(
            "JSON Schema to extract:" in p for p in captured_user_prompts
        ), "Expected LLMDirectExtraction-style extraction prompts"

    async def test_try_extraction_truncates_to_max_doc_chars(self) -> None:
        """Documents exceeding max_doc_chars are truncated before extraction."""
        captured_user_prompts: list[str] = []

        class CapturingLLM:
            async def acomplete(
                self, messages: list[dict[str, str]], **kw: Any
            ) -> LLMResponse:
                user_msg = messages[-1]["content"] if messages else ""
                sys_msg = messages[0]["content"] if messages else ""
                captured_user_prompts.append(user_msg)

                if "propose" in user_msg.lower() and "json schema" in user_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            }
                        ),
                        input_tokens=100,
                        output_tokens=50,
                    )
                if "refinement" in sys_msg.lower() or "expansion" in sys_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {"add_fields": {}, "remove_fields": [], "modify_fields": {}}
                        ),
                        input_tokens=50,
                        output_tokens=20,
                    )
                return LLMResponse(
                    content=json.dumps({"name": "ok"}),
                    input_tokens=50,
                    output_tokens=20,
                )

            async def astructured_complete(
                self, messages: Any, output_schema: Any, **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = CapturingLLM()
        discovery = ThreeStageDiscovery(
            model="mock",
            max_doc_chars=100,  # force truncation for test
            human_review=True,
        )
        long_docs = [
            Document(content="x" * 5000, id=f"long_doc_{i}") for i in range(6)
        ]

        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            await discovery.adiscover(long_docs)
        finally:
            _unpatch(orig_sp, orig_ts)

        extraction_prompts = [
            p for p in captured_user_prompts if "JSON Schema to extract:" in p
        ]
        assert extraction_prompts, "No extraction-style prompts captured"
        # Each extraction prompt should only contain ~100 'x' chars from the
        # document body. Counting 'x' runs is a safe lower bound regardless
        # of surrounding prompt boilerplate.
        for prompt in extraction_prompts:
            # Extract the doc body — it's wrapped between '---' markers
            assert "x" * 500 not in prompt  # full 5000-char doc NOT leaked
            assert "x" * 50 in prompt  # truncated slice still present

    async def test_focus_forwarded_to_stage1(self) -> None:
        """ThreeStageDiscovery.focus propagates into the stage 1 prompt."""
        captured_prompts: list[str] = []

        class CapturingLLM:
            async def acomplete(
                self, messages: list[dict[str, str]], **kw: Any
            ) -> LLMResponse:
                user_msg = messages[-1]["content"] if messages else ""
                captured_prompts.append(user_msg)
                # Always return a valid schema so discovery doesn't fail
                return LLMResponse(
                    content=json.dumps(
                        {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        }
                    ),
                    input_tokens=100,
                    output_tokens=50,
                )

            async def astructured_complete(
                self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = CapturingLLM()
        discovery = ThreeStageDiscovery(
            model="mock",
            focus="patient demographics only",
            human_review=True,  # Stop at stage 2 — less noise in captured prompts
        )

        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            await discovery.adiscover(_make_docs(10))
        finally:
            _unpatch(orig_sp, orig_ts)

        # Stage 1 goes through SinglePassDiscovery which calls _build_user_prompt
        # and includes the focus text. The first captured prompt is stage 1.
        assert any("Focus of extraction: patient demographics only" in p for p in captured_prompts)

    async def test_small_corpus_preserves_stage1_schema(self) -> None:
        """With 3 docs, stage 3 can't wipe out the schema even if LLM tries to."""

        class HostileExpansionLLM:
            """Returns remove_fields for EVERY existing field at every refinement call."""

            def __init__(self) -> None:
                self._call_count = 0

            async def acomplete(
                self, messages: list[dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                self._call_count += 1
                user_msg = messages[-1]["content"] if messages else ""
                sys_msg = messages[0]["content"] if messages else ""

                # Stage 1: initial schema from SinglePassDiscovery
                if "propose" in user_msg.lower() and "json schema" in user_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "rating": {"type": "number"},
                                    "category": {"type": "string"},
                                },
                                "required": ["title"],
                            }
                        ),
                        input_tokens=500,
                        output_tokens=200,
                    )

                # Refinement / expansion — hostile response: remove everything
                if "refinement" in sys_msg.lower() or "expansion" in sys_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "add_fields": {},
                                "remove_fields": [
                                    "title",
                                    "rating",
                                    "category",
                                ],
                                "modify_fields": {},
                                "rationale": "all fields have 0% coverage",
                            }
                        ),
                        input_tokens=300,
                        output_tokens=150,
                    )

                # Extraction call — returns empty (mimics small-N noise)
                return LLMResponse(
                    content=json.dumps({}),
                    input_tokens=200,
                    output_tokens=50,
                )

            async def astructured_complete(
                self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = HostileExpansionLLM()
        discovery = ThreeStageDiscovery(model="mock")

        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            # Small corpus: 3 docs. Stage 2 sample = 3 < min_samples_for_removal (5),
            # so stage 2 runs strictly-additive. Stage 3 is always strictly-additive.
            schema = await discovery.adiscover(_make_docs(3), domain_hint="products")
        finally:
            _unpatch(orig_sp, orig_ts)

        # Schema MUST still contain the stage-1 fields even though the LLM
        # tried to remove them at every subsequent stage.
        props = schema.json_schema.get("properties", {})
        assert "title" in props
        assert "rating" in props
        assert "category" in props
        # Stage-2 flag should be recorded in metadata
        report = schema.field_metadata.get("_discovery_report", {})
        assert report.get("stage2_strictly_additive") is True
        assert report.get("stages_completed") == 3

    async def test_large_corpus_still_allows_removal(self) -> None:
        """With enough docs, stage 2 can still remove redundant fields."""

        class RemovingStage2LLM:
            def __init__(self) -> None:
                self._call_count = 0

            async def acomplete(
                self, messages: list[dict[str, str]], **kw: Any
            ) -> LLMResponse:
                self._call_count += 1
                user_msg = messages[-1]["content"] if messages else ""
                sys_msg = messages[0]["content"] if messages else ""

                if "propose" in user_msg.lower() and "json schema" in user_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "junk": {"type": "string"},
                                },
                                "required": ["title"],
                            }
                        ),
                        input_tokens=500,
                        output_tokens=200,
                    )

                # Stage 2 refinement asks to remove the junk field
                if "refinement" in sys_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "add_fields": {},
                                "remove_fields": ["junk"],
                                "modify_fields": {},
                                "rationale": "junk field is redundant",
                            }
                        ),
                        input_tokens=300,
                        output_tokens=150,
                    )
                # Stage 3 expansion — no-op
                if "expansion" in sys_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {"add_fields": {}, "rationale": "nothing new"}
                        ),
                        input_tokens=300,
                        output_tokens=100,
                    )

                # Extraction
                return LLMResponse(
                    content=json.dumps({"title": "a", "junk": None}),
                    input_tokens=200,
                    output_tokens=100,
                )

            async def astructured_complete(
                self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = RemovingStage2LLM()
        discovery = ThreeStageDiscovery(model="mock")

        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            # 20 docs >> min_samples_for_removal (5) — stage 2 NOT strictly additive
            schema = await discovery.adiscover(_make_docs(20))
        finally:
            _unpatch(orig_sp, orig_ts)

        props = schema.json_schema.get("properties", {})
        assert "title" in props
        assert "junk" not in props  # removal permitted at large N
        report = schema.field_metadata.get("_discovery_report", {})
        assert report.get("stage2_strictly_additive") is False

    async def test_stage3_ignores_hostile_removals_at_large_n(self) -> None:
        """Even with 50+ docs, stage 3 ignores remove_fields (strictly additive by design)."""

        class HostileStage3LLM:
            async def acomplete(
                self, messages: list[dict[str, str]], **kw: Any
            ) -> LLMResponse:
                user_msg = messages[-1]["content"] if messages else ""
                sys_msg = messages[0]["content"] if messages else ""

                if "propose" in user_msg.lower() and "json schema" in user_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "rating": {"type": "number"},
                                },
                                "required": ["title"],
                            }
                        ),
                        input_tokens=500,
                        output_tokens=200,
                    )

                if "refinement" in sys_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {"add_fields": {}, "remove_fields": [], "modify_fields": {}}
                        ),
                        input_tokens=200,
                        output_tokens=50,
                    )

                # Stage 3: hostile — LLM ignores prompt and returns removals
                if "expansion" in sys_msg.lower():
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "add_fields": {"price": {"type": "number"}},
                                "remove_fields": ["title", "rating"],
                                "modify_fields": {},
                                "rationale": "pretend these are low-coverage",
                            }
                        ),
                        input_tokens=300,
                        output_tokens=150,
                    )

                return LLMResponse(
                    content=json.dumps({"title": "t", "rating": 3}),
                    input_tokens=200,
                    output_tokens=100,
                )

            async def astructured_complete(
                self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kw: Any
            ) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = HostileStage3LLM()
        discovery = ThreeStageDiscovery(model="mock")

        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            schema = await discovery.adiscover(_make_docs(60))
        finally:
            _unpatch(orig_sp, orig_ts)

        props = schema.json_schema.get("properties", {})
        # Stage-1 fields survive despite hostile stage-3 removals
        assert "title" in props
        assert "rating" in props
        # Legitimate additions pass through
        assert "price" in props
