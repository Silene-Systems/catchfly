"""Unit tests for catchfly.extraction._provenance helpers."""

from __future__ import annotations

from catchfly._types import Document, SourceSpan
from catchfly.extraction._provenance import (
    build_field_spans,
    locate_span,
    merge_field_spans,
    unwrap_record,
    wrap_schema_with_provenance,
)

# ---------------------------------------------------------------------------
# wrap_schema_with_provenance
# ---------------------------------------------------------------------------


class TestWrapSchemaWithProvenance:
    def test_wraps_flat_string_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"gender": {"type": "string"}},
            "required": ["gender"],
        }
        wrapped = wrap_schema_with_provenance(schema)

        gender = wrapped["properties"]["gender"]
        assert gender["type"] == "object"
        assert gender["properties"]["value"] == {"type": "string"}
        assert gender["properties"]["source_quotes"]["type"] == "array"
        assert gender["properties"]["source_quotes"]["items"] == {"type": "string"}
        assert gender["required"] == ["value", "source_quotes"]

    def test_wraps_array_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "symptoms": {"type": "array", "items": {"type": "string"}},
            },
        }
        wrapped = wrap_schema_with_provenance(schema)

        symptoms = wrapped["properties"]["symptoms"]
        # Original array schema is preserved inside "value"
        assert symptoms["properties"]["value"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        # source_quotes is still a flat array of strings (one per list item)
        assert symptoms["properties"]["source_quotes"]["type"] == "array"

    def test_wraps_nested_object_flat(self) -> None:
        """MVP: nested objects become part of value, not further decomposed."""
        schema = {
            "type": "object",
            "properties": {
                "liver_transaminase_levels": {
                    "type": "object",
                    "properties": {
                        "alt": {"type": "string"},
                        "ast": {"type": "string"},
                    },
                },
            },
        }
        wrapped = wrap_schema_with_provenance(schema)

        lt = wrapped["properties"]["liver_transaminase_levels"]
        # The entire nested object is preserved under "value" unchanged
        assert lt["properties"]["value"]["type"] == "object"
        assert lt["properties"]["value"]["properties"] == {
            "alt": {"type": "string"},
            "ast": {"type": "string"},
        }

    def test_preserves_required_list(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }
        wrapped = wrap_schema_with_provenance(schema)
        assert wrapped["required"] == ["a"]

    def test_idempotent(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        wrapped = wrap_schema_with_provenance(schema)
        twice = wrap_schema_with_provenance(wrapped)
        # Second pass returns the same dict — no double-wrapping
        assert twice is wrapped
        assert "value" in twice["properties"]["x"]["properties"]
        assert "value" not in twice["properties"]["x"]["properties"]["value"].get(
            "properties", {}
        )

    def test_non_object_schema_passthrough(self) -> None:
        schema = {"type": "string"}
        assert wrap_schema_with_provenance(schema) == schema


# ---------------------------------------------------------------------------
# unwrap_record
# ---------------------------------------------------------------------------


class TestUnwrapRecord:
    def test_basic_unwrap(self) -> None:
        raw = {
            "patient_id": {"value": "Patient 1", "source_quotes": ["## 2.1. Patient 1"]},
            "gender": {
                "value": "Male",
                "source_quotes": ["revealed a male karyotype"],
            },
        }
        clean, quotes = unwrap_record(raw)
        assert clean == {"patient_id": "Patient 1", "gender": "Male"}
        assert quotes == {
            "patient_id": ["## 2.1. Patient 1"],
            "gender": ["revealed a male karyotype"],
        }

    def test_list_field(self) -> None:
        raw = {
            "symptoms": {
                "value": ["hypotonia", "splenomegaly"],
                "source_quotes": ["mild truncal hypotonia", "asymptomatic splenomegaly"],
            },
        }
        clean, quotes = unwrap_record(raw)
        assert clean["symptoms"] == ["hypotonia", "splenomegaly"]
        assert quotes["symptoms"] == ["mild truncal hypotonia", "asymptomatic splenomegaly"]

    def test_empty_source_quotes(self) -> None:
        raw = {"inheritance": {"value": "autosomal recessive", "source_quotes": []}}
        clean, quotes = unwrap_record(raw)
        assert clean["inheritance"] == "autosomal recessive"
        assert quotes["inheritance"] == []

    def test_null_source_quotes_becomes_empty(self) -> None:
        raw = {"x": {"value": "y", "source_quotes": None}}
        clean, quotes = unwrap_record(raw)
        assert clean["x"] == "y"
        assert quotes["x"] == []

    def test_partial_wrapper_fallback(self) -> None:
        """When the LLM ignores the wrapper, the field passes through."""
        raw = {
            "good": {"value": "v", "source_quotes": ["q"]},
            "bad": "plain string, not a wrapper",
        }
        clean, quotes = unwrap_record(raw)
        assert clean["good"] == "v"
        assert clean["bad"] == "plain string, not a wrapper"
        assert quotes["good"] == ["q"]
        assert quotes["bad"] == []

    def test_filters_empty_quotes(self) -> None:
        raw = {"x": {"value": "v", "source_quotes": ["", "real quote", ""]}}
        _, quotes = unwrap_record(raw)
        assert quotes["x"] == ["real quote"]


# ---------------------------------------------------------------------------
# locate_span
# ---------------------------------------------------------------------------


class TestLocateSpan:
    def test_exact_match(self) -> None:
        doc = Document(content="The patient was a male infant.", id="d1")
        doc_id, start, end, conf = locate_span("male infant", [doc])
        assert doc_id == "d1"
        assert conf == "exact"
        assert doc.content[start:end] == "male infant"

    def test_fuzzy_whitespace(self) -> None:
        doc = Document(content="Birth weight:  2600 g  at  term.", id="d1")
        # LLM-returned quote collapses the double spaces
        doc_id, start, end, conf = locate_span("birth weight: 2600 g", [doc])
        assert conf == "fuzzy"
        assert doc_id == "d1"
        assert start is not None and end is not None
        # Fuzzy range should cover the original (messy) substring
        assert "2600 g" in doc.content[start:end]

    def test_unresolved(self) -> None:
        doc = Document(content="Completely different text.", id="d1")
        doc_id, start, end, conf = locate_span(
            "phrase that does not appear anywhere", [doc]
        )
        assert conf == "unresolved"
        assert doc_id is None
        assert start is None
        assert end is None

    def test_empty_quote_is_inferred(self) -> None:
        doc = Document(content="anything", id="d1")
        doc_id, start, end, conf = locate_span("", [doc])
        assert conf == "inferred"
        assert doc_id is None
        assert start is None
        assert end is None

    def test_whitespace_only_quote_is_inferred(self) -> None:
        doc = Document(content="anything", id="d1")
        _, _, _, conf = locate_span("   \n  ", [doc])
        assert conf == "inferred"

    def test_multi_document(self) -> None:
        doc1 = Document(content="First document about dogs.", id="d1")
        doc2 = Document(content="Second document about cats.", id="d2")
        doc_id, _, _, conf = locate_span("about cats", [doc1, doc2])
        assert doc_id == "d2"
        assert conf == "exact"

    def test_markdown_table_row(self) -> None:
        content = (
            "| Patient | Gender | Outcome |\n"
            "| --- | --- | --- |\n"
            "| Patient 4 | Male | Died at 2Mo |\n"
            "| Patient 5 | Female | Alive |\n"
        )
        doc = Document(content=content, id="paper")
        doc_id, start, end, conf = locate_span("Patient 4 | Male | Died at 2Mo", [doc])
        assert conf == "exact"
        assert doc_id == "paper"
        assert doc.content[start:end] == "Patient 4 | Male | Died at 2Mo"

    def test_fuzzy_smart_quotes(self) -> None:
        """NFKC normalization folds curly quotes to straight quotes."""
        doc = Document(content="The reviewer said \u201cgreat product\u201d today.", id="d")
        # LLM copied with straight quotes (common)
        _, _, _, conf = locate_span('"great product"', [doc])
        # Accept either exact or fuzzy — the key is that it resolves
        assert conf in ("exact", "fuzzy")

    def test_prefers_doc_id_over_source(self) -> None:
        doc = Document(content="hello world", id="my-id", source="some/path.txt")
        doc_id, _, _, _ = locate_span("hello", [doc])
        assert doc_id == "my-id"

    def test_falls_back_to_source_when_no_id(self) -> None:
        doc = Document(content="hello world", source="some/path.txt")
        doc_id, _, _, _ = locate_span("hello", [doc])
        assert doc_id == "some/path.txt"


# ---------------------------------------------------------------------------
# build_field_spans
# ---------------------------------------------------------------------------


class TestBuildFieldSpans:
    def test_list_field_preserves_order(self) -> None:
        content = (
            "Symptoms included mild truncal hypotonia at 3 months, "
            "asymptomatic splenomegaly noted at 6 months, "
            "and dysmetria and intention tremor by 12 months."
        )
        doc = Document(content=content, id="paper")
        quotes = {
            "symptoms": [
                "mild truncal hypotonia",
                "asymptomatic splenomegaly",
                "dysmetria and intention tremor",
            ],
        }
        spans = build_field_spans(quotes, [doc])
        assert len(spans["symptoms"]) == 3
        assert [s.quote for s in spans["symptoms"]] == quotes["symptoms"]
        for span in spans["symptoms"]:
            assert span.confidence == "exact"
            assert span.document_id == "paper"
            assert content[span.start : span.end] == span.quote

    def test_duplicate_quotes_preserved(self) -> None:
        doc = Document(content="fever and fever", id="d")
        spans = build_field_spans({"symptoms": ["fever", "fever"]}, [doc])
        assert len(spans["symptoms"]) == 2

    def test_unresolved_field(self) -> None:
        doc = Document(content="something", id="d")
        spans = build_field_spans({"x": ["nope"]}, [doc])
        assert len(spans["x"]) == 1
        assert spans["x"][0].confidence == "unresolved"
        assert spans["x"][0].start is None

    def test_empty_field_is_empty_list(self) -> None:
        doc = Document(content="anything", id="d")
        spans = build_field_spans({"inferred_field": []}, [doc])
        assert spans["inferred_field"] == []


# ---------------------------------------------------------------------------
# merge_field_spans
# ---------------------------------------------------------------------------


class TestMergeFieldSpans:
    def test_none_handling(self) -> None:
        a = {"x": [SourceSpan(quote="q1", document_id="d", start=0, end=2)]}
        assert merge_field_spans(a, None) == a
        assert merge_field_spans(None, a) == a
        assert merge_field_spans(None, None) is None

    def test_merges_disjoint_fields(self) -> None:
        a = {"x": [SourceSpan(quote="q1", document_id="d", start=0, end=2)]}
        b = {"y": [SourceSpan(quote="q2", document_id="d", start=3, end=5)]}
        merged = merge_field_spans(a, b)
        assert merged is not None
        assert set(merged.keys()) == {"x", "y"}

    def test_unions_same_field(self) -> None:
        a = {"x": [SourceSpan(quote="q1", document_id="d", start=0, end=2)]}
        b = {"x": [SourceSpan(quote="q2", document_id="d", start=3, end=5)]}
        merged = merge_field_spans(a, b)
        assert merged is not None
        assert len(merged["x"]) == 2
        assert {s.quote for s in merged["x"]} == {"q1", "q2"}

    def test_dedupes_identical_spans(self) -> None:
        span = SourceSpan(quote="q1", document_id="d", start=0, end=2)
        a = {"x": [span]}
        b = {"x": [SourceSpan(quote="q1", document_id="d", start=0, end=2)]}
        merged = merge_field_spans(a, b)
        assert merged is not None
        assert len(merged["x"]) == 1

    def test_does_not_mutate_inputs(self) -> None:
        a = {"x": [SourceSpan(quote="q1", document_id="d", start=0, end=2)]}
        b = {"x": [SourceSpan(quote="q2", document_id="d", start=3, end=5)]}
        a_copy = {"x": list(a["x"])}
        merge_field_spans(a, b)
        # merge_field_spans builds a fresh dict but copies lists — the
        # original input lists should not grow.
        assert len(a["x"]) == len(a_copy["x"])
