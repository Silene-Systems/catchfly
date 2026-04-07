"""Per-field provenance helpers for :class:`LLMDirectExtraction`.

This module implements the *quote-first, offset-second* provenance design:

1. The extraction schema is transformed so every top-level field is wrapped
   in ``{"value": <original>, "source_quotes": [str, ...]}``. The LLM is
   forced — via structured output — to copy a verbatim supporting excerpt
   alongside every extracted value.
2. After the LLM responds, the wrapper is stripped back to a clean record
   and the quotes are located in the source document with a deterministic
   matcher (exact → whitespace/punctuation-normalized fuzzy → unresolved).
3. The resulting :class:`SourceSpan` objects are attached to
   :attr:`RecordProvenance.field_spans`.

The matcher never trusts offsets produced by the LLM — models are poor at
counting characters but good at copying phrases. A 10–30 word excerpt is
nearly always uniquely locatable with plain :meth:`str.find`, which makes
the approach robust even for documents with many repeated values.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any

from catchfly._types import SourceSpan, SpanConfidence

if TYPE_CHECKING:
    from catchfly._types import Document


# ---------------------------------------------------------------------------
# Schema transformation
# ---------------------------------------------------------------------------

_PROVENANCE_WRAPPER_KEY = "__catchfly_provenance_wrapped__"


def wrap_schema_with_provenance(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap every top-level property in ``{value, source_quotes}``.

    Only the top-level properties of an object schema are wrapped (MVP).
    Nested objects are carried through unchanged as the ``value`` payload —
    a single quote then supports the whole nested block. This keeps the
    implementation tractable without blocking richer nesting in the future.

    The function is a no-op on schemas that are not object types (the
    original schema is returned untouched) and is idempotent: calling it on
    an already-wrapped schema returns the same schema.

    The returned schema carries a private marker
    (``__catchfly_provenance_wrapped__``) so the round-trip can be detected
    even if the caller reorders properties.
    """
    if not isinstance(schema, dict):
        return schema
    if schema.get(_PROVENANCE_WRAPPER_KEY):
        return schema
    if schema.get("type") != "object":
        return schema

    properties = schema.get("properties") or {}
    if not properties:
        return schema

    wrapped_properties: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        wrapped_properties[field_name] = {
            "type": "object",
            "description": (
                "Extracted value wrapped with its supporting quote(s) "
                "from the source document."
            ),
            "properties": {
                "value": field_schema,
                "source_quotes": {
                    "type": "array",
                    "description": (
                        "Verbatim excerpt(s) from the source document that "
                        "support this value. Include 10-30 words of "
                        "surrounding context so the excerpt is uniquely "
                        "locatable. For list-valued fields, return one "
                        "excerpt per list item in the same order. Return "
                        "an empty array if the value was inferred without "
                        "a directly supporting passage."
                    ),
                    "items": {"type": "string"},
                },
            },
            "required": ["value", "source_quotes"],
        }

    new_schema = dict(schema)
    new_schema["properties"] = wrapped_properties
    # Preserve the original top-level "required" list — those semantics still
    # apply to the *value* inside each wrapper. Strict-mode backends that
    # demand required-covers-all will accept this as-is because every wrapped
    # property is present by construction.
    new_schema[_PROVENANCE_WRAPPER_KEY] = True
    return new_schema


def unwrap_record(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Split a provenance-wrapped LLM response into (clean_record, quotes).

    Fields that don't conform to the wrapper shape are passed through into
    the clean record with an empty quote list — this keeps the extractor
    robust when a model partially ignores the wrapper instruction.
    """
    clean: dict[str, Any] = {}
    quotes: dict[str, list[str]] = {}
    for field_name, payload in raw.items():
        if (
            isinstance(payload, dict)
            and "value" in payload
            and "source_quotes" in payload
        ):
            clean[field_name] = payload["value"]
            raw_quotes = payload.get("source_quotes") or []
            if isinstance(raw_quotes, list):
                quotes[field_name] = [str(q) for q in raw_quotes if q]
            else:
                quotes[field_name] = []
        else:
            # Graceful fallback: LLM didn't apply the wrapper shape.
            clean[field_name] = payload
            quotes[field_name] = []
    return clean, quotes


# ---------------------------------------------------------------------------
# Quote → SourceSpan matcher
# ---------------------------------------------------------------------------

# Whitespace and leading/trailing punctuation are common sources of benign
# mismatch when an LLM paraphrases micro-formatting (extra spaces, trailing
# period, smart quotes from OCR). We normalize both sides before searching.
_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_PUNCT = " \t\n\r.,;:!?\"'`()[]{}<>—–-"


def _normalize(text: str) -> str:
    """Normalize text for fuzzy matching.

    Applies NFKC unicode normalization (folds OCR-style smart quotes and
    ligatures), collapses whitespace runs to a single space, and lowercases
    the result. Edge punctuation is *not* stripped here — trimming is done
    on the quote side before the search so original content indices stay
    meaningful for offset mapping.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.lower()


def _build_normalized_index(content: str) -> tuple[str, list[int]]:
    """Return (normalized_content, map) where map[i] is the original offset
    of normalized_content[i]. Used to translate a fuzzy match back to the
    original character offsets.
    """
    normalized_chars: list[str] = []
    offset_map: list[int] = []
    prev_ws = False
    for i, ch in enumerate(unicodedata.normalize("NFKC", content)):
        if ch.isspace():
            if prev_ws:
                continue
            normalized_chars.append(" ")
            offset_map.append(i)
            prev_ws = True
        else:
            normalized_chars.append(ch.lower())
            offset_map.append(i)
            prev_ws = False
    return "".join(normalized_chars), offset_map


def locate_span(
    quote: str,
    documents: list[Document],
) -> tuple[str | None, int | None, int | None, SpanConfidence]:
    """Locate a quote within the first document that contains it.

    Tries, in order:

    1. **Exact** substring match against each document's content.
    2. **Fuzzy** match after NFKC + whitespace normalization, mapping the
       normalized offset back to the original content range.
    3. **Unresolved** — returns ``(None, None, None, "unresolved")`` if
       no document contains the quote.

    Returns ``(document_id, start, end, confidence)``. ``document_id`` is
    the first of ``doc.id`` / ``str(doc.source)`` / ``None`` that is set.
    An empty quote is treated as an *inferred* value and returns
    ``(None, None, None, "inferred")``.
    """
    if not quote or not quote.strip():
        return None, None, None, "inferred"

    stripped = quote.strip(_EDGE_PUNCT)
    if not stripped:
        stripped = quote

    # Exact pass first — cheapest and most specific.
    for doc in documents:
        content = doc.content or ""
        idx = content.find(stripped)
        if idx != -1:
            return _doc_identifier(doc), idx, idx + len(stripped), "exact"

    # Fuzzy pass — normalize both sides.
    normalized_quote = _normalize(stripped)
    if normalized_quote:
        for doc in documents:
            content = doc.content or ""
            normalized_content, offset_map = _build_normalized_index(content)
            idx = normalized_content.find(normalized_quote)
            if idx == -1:
                continue
            start = offset_map[idx]
            end_idx = idx + len(normalized_quote) - 1
            end = (
                len(content)
                if end_idx >= len(offset_map)
                else offset_map[end_idx] + 1
            )
            return _doc_identifier(doc), start, end, "fuzzy"

    return None, None, None, "unresolved"


def _doc_identifier(doc: Document) -> str | None:
    """Stable identifier for a document, preferring ``id`` over ``source``."""
    if doc.id:
        return str(doc.id)
    if doc.source is not None:
        return str(doc.source)
    return None


def build_field_spans(
    quotes_by_field: dict[str, list[str]],
    documents: list[Document],
) -> dict[str, list[SourceSpan]]:
    """Turn raw quote lists into :class:`SourceSpan` objects.

    Each quote is located independently. Empty quote lists pass through as
    ``[]`` — callers interpret this as *field present, no supporting quote*.
    Duplicate quotes are preserved in their original order so list-valued
    fields retain element-level alignment with the extracted value list.
    """
    result: dict[str, list[SourceSpan]] = {}
    for field_name, quotes in quotes_by_field.items():
        spans: list[SourceSpan] = []
        for quote in quotes:
            doc_id, start, end, confidence = locate_span(quote, documents)
            spans.append(
                SourceSpan(
                    quote=quote,
                    document_id=doc_id,
                    start=start,
                    end=end,
                    confidence=confidence,
                )
            )
        result[field_name] = spans
    return result


# ---------------------------------------------------------------------------
# Merge helper (used when multi-record deduplication collapses duplicates)
# ---------------------------------------------------------------------------


def merge_field_spans(
    primary: dict[str, list[SourceSpan]] | None,
    secondary: dict[str, list[SourceSpan]] | None,
) -> dict[str, list[SourceSpan]] | None:
    """Merge per-field spans from two provenance dicts.

    Used when ``multi_record=True`` + ``deduplicate=True`` collapses two
    identical records extracted from different chunks: we want to preserve
    the union of supporting spans rather than arbitrarily discarding one
    copy. Duplicate spans (same quote + document + offsets) are dropped.
    """
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    merged: dict[str, list[SourceSpan]] = {k: list(v) for k, v in primary.items()}
    for field_name, spans in secondary.items():
        if field_name not in merged:
            merged[field_name] = list(spans)
            continue
        existing = merged[field_name]
        seen = {(s.quote, s.document_id, s.start, s.end) for s in existing}
        for span in spans:
            key = (span.quote, span.document_id, span.start, span.end)
            if key not in seen:
                existing.append(span)
                seen.add(key)
    return merged
