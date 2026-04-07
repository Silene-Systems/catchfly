"""ThreeStageDiscovery — progressive schema refinement inspired by SCHEMA-MINER.

Stage 1: Initial schema from a few exemplar documents
Stage 2: Refinement against a curated sample — fix gaps, remove redundancies
Stage 3: Expansion against a larger sample — add recurring fields, flag rare ones
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from pydantic import BaseModel, PrivateAttr

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._parsing import strip_markdown_fences
from catchfly._types import Document, Schema
from catchfly.discovery.single_pass import SinglePassDiscovery
from catchfly.exceptions import DiscoveryError, ProviderError, SchemaError
from catchfly.providers.llm import OpenAICompatibleClient

logger = logging.getLogger(__name__)

_REFINEMENT_SYSTEM = """\
You are a schema refinement assistant. Given a JSON Schema and extraction \
results from sample documents, identify problems and propose improvements.

Analyze:
- Missing fields: data present in documents but not captured by the schema
- Redundant fields: fields that are always empty or duplicate other fields
- Type mismatches: fields where extracted values don't match the declared type
- Naming issues: fields with unclear or inconsistent names

Output ONLY valid JSON with this structure:
{
  "add_fields": {"field_name": {"type": "...", "description": "..."}},
  "remove_fields": ["field_to_remove"],
  "modify_fields": {"field_name": {"type": "new_type", "description": "..."}},
  "rationale": "Brief explanation of changes"
}\
"""

_CHANGES_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "add_fields": {
            "type": "object",
            "description": "Field name → JSON Schema field definition",
        },
        "remove_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of fields to remove from the schema",
        },
        "modify_fields": {
            "type": "object",
            "description": (
                "Field name → partial JSON Schema update "
                "(merged into existing field)"
            ),
        },
        "rationale": {
            "type": "string",
            "description": "Brief explanation of the proposed changes",
        },
    },
}
"""Output schema for refinement / expansion / refine-instruction calls.

Kept permissive (no ``required``, no ``additionalProperties: false``) so
the LLM can return any subset of the operations — including an empty
change set — without triggering strict-mode violations across the
various compatible backends (OpenAI, Anthropic, Ollama, vLLM)."""


_EXPANSION_SYSTEM = """\
You are a schema expansion assistant. Given a JSON Schema and extraction \
results from a large sample, propose NEW fields to add to the schema.

Rules:
- Add fields whose data appears recurrently across documents (roughly >20% \
of the sample) but isn't yet captured by the schema
- Focus on fields that represent real, recurring information — ignore \
one-off or document-specific details
- Do NOT remove, rename, or modify existing fields — expansion is strictly \
additive. Low-coverage fields are handled elsewhere
- If no additions are warranted, return an empty add_fields object

Output ONLY valid JSON with this structure:
{
  "add_fields": {"field_name": {"type": "...", "description": "..."}},
  "rationale": "Brief explanation of additions"
}\
"""


class ThreeStageDiscovery(BaseModel):
    """Discover schema through 3 progressive refinement stages.

    Stage 1: LLM proposes initial schema from 2-3 exemplar documents
    Stage 2: Schema refined against 5-10 docs (fix gaps, remove redundancies)
    Stage 3: Schema expanded against 50-100 docs (add recurring, flag rare)
    """

    model: str = DEFAULT_MODEL
    stage1_samples: int = 3
    """Stage 1 uses few docs (2-3) to avoid over-fitting the initial schema."""
    stage2_samples: int = 10
    """Stage 2 uses a moderate sample to catch gaps without excessive LLM cost."""
    stage3_samples: int = 50
    """Stage 3 uses a larger sample to validate coverage at near-corpus scale."""
    min_samples_for_removal: int = 5
    """Minimum number of documents required before coverage-based field
    removal is permitted. When the stage 2 sample is smaller than this,
    refinement runs in strictly-additive mode — the LLM can propose
    additions but ``remove_fields`` / ``modify_fields`` are ignored. At
    N < 5 coverage statistics are too noisy (each field is effectively
    0% or 100%) to justify deleting work that stage 1 produced."""
    max_doc_chars: int = 30000
    """Per-document truncation limit (~7500 tokens). See
    :attr:`SinglePassDiscovery.max_doc_chars` for the rationale behind
    the default."""
    max_fields: int | None = None
    """Maximum number of fields in the discovered schema. None = no limit."""
    suggested_fields: list[str] | None = None
    """Field names to include in the schema. LLM may add others if found in documents."""
    focus: str | None = None
    """User-facing intent that narrows what the schema should capture.

    See :attr:`SinglePassDiscovery.focus` for the full rationale. The
    focus is forwarded to stage 1 (initial schema) — stages 2 and 3
    inherit the resulting schema. Leaving ``focus`` unset preserves the
    prior behaviour of proposing a broad, catch-all schema."""
    human_review: bool = False
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.3
    """Low temperature for deterministic schema proposals."""
    client: Any | None = None
    """Pre-configured LLM client. If ``None``, a default client is created
    from ``model``, ``base_url``, and ``api_key``."""

    _usage_callback: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> Any:
        """Return the injected client or create a default one."""
        if self.client is not None:
            return self.client
        return OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            usage_callback=self._usage_callback,
        )

    async def adiscover(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Schema:
        """Run 3-stage discovery asynchronously."""
        if not documents:
            raise DiscoveryError("No documents provided for schema discovery.")

        client = self._get_client()

        # --- Stage 1: Initial schema ---
        logger.info(
            "ThreeStageDiscovery: Stage 1 — initial schema from %d docs",
            self.stage1_samples,
        )
        schema = await self._stage1_initial(documents, domain_hint)
        json_schema = schema.json_schema
        report_meta: dict[str, Any] = {"stages_completed": 1}

        # --- Stage 2: Refinement ---
        stage2_docs = self._sample(documents, self.stage2_samples)
        logger.info("ThreeStageDiscovery: Stage 2 — refining against %d docs", len(stage2_docs))

        extracted = await self._try_extraction(client, json_schema, stage2_docs)
        coverage = self._compute_coverage(json_schema, extracted)

        refinement = await self._get_refinement(
            client, json_schema, extracted, coverage, _REFINEMENT_SYSTEM
        )
        # At small sample sizes, coverage statistics are too noisy to justify
        # field removal — a single missing doc gives 0% coverage even for a
        # perfectly valid field. Fall back to strictly-additive refinement.
        stage2_strictly_additive = len(stage2_docs) < self.min_samples_for_removal
        if stage2_strictly_additive:
            logger.info(
                "ThreeStageDiscovery: Stage 2 running in strictly-additive mode "
                "(sample %d < min_samples_for_removal %d)",
                len(stage2_docs),
                self.min_samples_for_removal,
            )
        json_schema = self._apply_changes(
            json_schema, refinement, strictly_additive=stage2_strictly_additive
        )
        report_meta["stages_completed"] = 2
        report_meta["stage2_changes"] = refinement
        report_meta["stage2_strictly_additive"] = stage2_strictly_additive

        if self.human_review:
            logger.info(
                "ThreeStageDiscovery: human_review=True — returning after Stage 2 "
                "for user inspection"
            )
            return self._build_schema(json_schema, coverage, report_meta, stage=2)

        # --- Stage 3: Expansion ---
        stage3_docs = self._sample(documents, self.stage3_samples)
        logger.info("ThreeStageDiscovery: Stage 3 — expanding against %d docs", len(stage3_docs))

        extracted = await self._try_extraction(client, json_schema, stage3_docs)
        coverage = self._compute_coverage(json_schema, extracted)

        expansion = await self._get_refinement(
            client, json_schema, extracted, coverage, _EXPANSION_SYSTEM
        )
        # Stage 3 is strictly additive by design — the _EXPANSION_SYSTEM
        # prompt only asks for new fields. The flag defends against LLMs
        # that ignore the instruction and return remove_fields anyway.
        json_schema = self._apply_changes(
            json_schema, expansion, strictly_additive=True
        )
        report_meta["stages_completed"] = 3
        report_meta["stage3_changes"] = expansion

        # Recompute final coverage
        coverage = self._compute_coverage(json_schema, extracted)

        return self._build_schema(json_schema, coverage, report_meta, stage=3)

    def discover(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Schema:
        """Run 3-stage discovery synchronously."""
        return run_sync(self.adiscover(documents, domain_hint=domain_hint, **kwargs))

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _stage1_initial(
        self,
        documents: list[Document],
        domain_hint: str | None,
    ) -> Schema:
        """Stage 1: Generate initial schema via SinglePassDiscovery."""
        sp = SinglePassDiscovery(
            model=self.model,
            num_samples=self.stage1_samples,
            max_fields=self.max_fields,
            suggested_fields=self.suggested_fields,
            focus=self.focus,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        return await sp.adiscover(documents, domain_hint=domain_hint)

    async def _try_extraction(
        self,
        client: OpenAICompatibleClient,
        json_schema: dict[str, Any],
        documents: list[Document],
    ) -> list[dict[str, Any]]:
        """Run extraction on the current schema via :class:`LLMDirectExtraction`.

        Dogfoods the same extractor users will run after discovery, so
        coverage statistics computed here reflect real extraction
        behaviour rather than a bespoke discovery-only prompt. Documents
        are truncated to ``max_doc_chars`` (mirroring the previous
        behaviour) and chunking is disabled so each document maps to a
        single extraction record, preserving the one-doc-one-dict
        contract that ``_compute_coverage`` depends on.
        """
        # Local import avoids the cycle: llm_direct.py does not import
        # from discovery, but discovery/__init__.py imports this module,
        # so the top-level import would be eager during package load.
        from catchfly.extraction.llm_direct import LLMDirectExtraction
        from catchfly.schema.converters import json_schema_to_pydantic

        # Truncate documents up-front — discovery only ever "sees" the
        # first ``max_doc_chars`` characters, matching the old behaviour.
        truncated_docs = [
            Document(
                content=doc.content[: self.max_doc_chars],
                id=doc.id or f"_discovery_doc_{i}",
                source=doc.source,
                metadata=dict(doc.metadata),
            )
            for i, doc in enumerate(documents)
        ]

        # Build Pydantic model once up-front so a malformed intermediate
        # schema (e.g. a stage-2 change that introduced an unsupported
        # type) surfaces cleanly instead of hiding inside the extractor.
        try:
            schema_model = json_schema_to_pydantic(
                json_schema, name="DiscoveryProbe"
            )
        except (SchemaError, ValueError, TypeError) as e:
            logger.debug(
                "ThreeStageDiscovery: intermediate schema not realisable "
                "as Pydantic model, skipping extraction probe: %s",
                e,
            )
            return [{} for _ in documents]

        extractor = LLMDirectExtraction(
            model=self.model,
            chunk_size=0,  # truncation already limited size; no chunking
            max_retries=1,
            on_error="collect",  # keep discovery alive on per-doc failures
            client=client,
        )

        try:
            result = await extractor.aextract(schema_model, truncated_docs)
        except (ProviderError, ValueError, KeyError, TypeError) as e:
            logger.debug(
                "ThreeStageDiscovery: dogfooded extraction probe failed: %s",
                e,
                exc_info=True,
            )
            return [{} for _ in documents]

        # LLMDirectExtraction appends records in completion order under
        # asyncio.gather, so we can't assume positional alignment with
        # ``truncated_docs``. Match back via the provenance's
        # source_document (which mirrors the same str(source or id)
        # fallback used when producing the RecordProvenance).
        def _doc_key(doc: Document) -> str:
            return str(doc.source or doc.id or "unknown")

        records_by_key: dict[str, dict[str, Any]] = {}
        for rec, prov in zip(result.records, result.provenance, strict=False):
            payload = rec.model_dump() if hasattr(rec, "model_dump") else {}
            records_by_key[prov.source_document] = payload

        return [records_by_key.get(_doc_key(doc), {}) for doc in truncated_docs]

    async def _get_refinement(
        self,
        client: OpenAICompatibleClient,
        json_schema: dict[str, Any],
        extracted: list[dict[str, Any]],
        coverage: dict[str, float],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Ask LLM to propose schema changes with a fixed output shape.

        Uses ``astructured_complete`` with :data:`_CHANGES_OUTPUT_SCHEMA`
        so providers that support ``response_format=json_schema`` (or
        tool calling) return a guaranteed-shape payload — eliminating
        the JSONDecodeError / shape-drift retries that the old
        plain-prompt path used to swallow silently.
        """
        user_prompt = self._build_refinement_prompt(json_schema, extracted, coverage)

        try:
            response = await client.astructured_complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                output_schema=_CHANGES_OUTPUT_SCHEMA,
                schema_name="schema_changes",
                temperature=self.temperature,
            )
            return self._parse_changes(response.content)
        except (ProviderError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("ThreeStageDiscovery: failed to parse refinement: %s", e, exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(documents: list[Document], n: int) -> list[Document]:
        if len(documents) <= n:
            return documents
        return random.sample(documents, n)

    @staticmethod
    def _compute_coverage(
        json_schema: dict[str, Any],
        extracted: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute field coverage: fraction of docs where field is non-null."""
        properties = json_schema.get("properties", {})
        if not properties or not extracted:
            return {}

        total = len(extracted)
        coverage: dict[str, float] = {}
        for field_name in properties:
            present = sum(1 for rec in extracted if rec.get(field_name) is not None)
            coverage[field_name] = present / total if total > 0 else 0.0
        return coverage

    @staticmethod
    def _build_refinement_prompt(
        json_schema: dict[str, Any],
        extracted: list[dict[str, Any]],
        coverage: dict[str, float],
    ) -> str:
        parts: list[str] = []
        parts.append(f"Current schema:\n```json\n{json.dumps(json_schema, indent=2)}\n```\n")

        parts.append("Field coverage (% of documents with non-null value):")
        for field, cov in sorted(coverage.items(), key=lambda x: x[1]):
            marker = " ⚠️ LOW" if cov < 0.2 else ""
            parts.append(f"  - {field}: {cov:.0%}{marker}")
        parts.append("")

        if extracted:
            parts.append(f"Sample extracted records ({min(len(extracted), 3)} shown):")
            for i, rec in enumerate(extracted[:3]):
                parts.append(f"  Record {i + 1}: {json.dumps(rec, default=str)}")
            parts.append("")

        parts.append("Propose schema changes based on the analysis above.")
        return "\n".join(parts)

    @staticmethod
    def _parse_changes(content: str) -> dict[str, Any]:
        """Parse schema change proposals from LLM response."""
        text = strip_markdown_fences(content)

        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            logger.warning("ThreeStageDiscovery: could not parse schema changes as JSON")
            return {}

    @staticmethod
    def _apply_changes(
        json_schema: dict[str, Any],
        changes: dict[str, Any],
        *,
        strictly_additive: bool = False,
    ) -> dict[str, Any]:
        """Apply add/remove/modify changes to a JSON Schema.

        Args:
            json_schema: The schema to modify. Not mutated — a deep copy is
                always returned.
            changes: Proposed changes with keys ``add_fields``,
                ``remove_fields``, and ``modify_fields``.
            strictly_additive: When ``True``, ``remove_fields`` and
                ``modify_fields`` are ignored regardless of what the LLM
                returned. Used by the expansion stage and by small-sample
                runs where coverage-based removal would be unsafe.

        Invariant: the result always contains at least as many properties
        as the input, or — if the changes would reduce properties to an
        empty set — the input schema is returned unchanged with a warning
        logged. This prevents an over-eager LLM (or a noisy small-sample
        coverage signal) from wiping out the discovered schema.
        """
        schema = json.loads(json.dumps(json_schema))  # deep copy
        properties = schema.setdefault("properties", {})
        required = set(schema.get("required", []))
        input_had_properties = bool(json_schema.get("properties"))

        # Add new fields
        for name, field_def in changes.get("add_fields", {}).items():
            if isinstance(field_def, dict) and name not in properties:
                properties[name] = field_def
                logger.debug("ThreeStageDiscovery: added field '%s'", name)

        if not strictly_additive:
            # Remove fields
            for name in changes.get("remove_fields", []):
                if name in properties:
                    del properties[name]
                    required.discard(name)
                    logger.debug("ThreeStageDiscovery: removed field '%s'", name)

            # Modify fields
            for name, field_def in changes.get("modify_fields", {}).items():
                if isinstance(field_def, dict) and name in properties:
                    properties[name].update(field_def)
                    logger.debug("ThreeStageDiscovery: modified field '%s'", name)
        elif changes.get("remove_fields") or changes.get("modify_fields"):
            logger.info(
                "ThreeStageDiscovery: ignoring remove_fields/modify_fields "
                "from LLM response (strictly_additive=True)"
            )

        # Invariant: never reduce a non-empty schema to an empty schema.
        # This guards against an over-eager LLM marking every field for
        # removal based on noisy coverage statistics from a small sample.
        if input_had_properties and not properties:
            logger.warning(
                "ThreeStageDiscovery: proposed changes would leave the "
                "schema with zero properties; keeping previous schema "
                "unchanged. This usually indicates noisy coverage "
                "statistics at small sample sizes."
            )
            reverted: dict[str, Any] = json.loads(json.dumps(json_schema))
            return reverted

        schema["required"] = sorted(required)
        result: dict[str, Any] = schema
        return result

    @staticmethod
    def _build_schema(
        json_schema: dict[str, Any],
        coverage: dict[str, float],
        report_meta: dict[str, Any],
        stage: int,
    ) -> Schema:
        """Build final Schema with discovery report in metadata."""
        from catchfly.schema.converters import json_schema_to_pydantic

        properties = json_schema.get("properties", {})
        if not properties:
            raise DiscoveryError(
                f"ThreeStageDiscovery produced a schema with no properties "
                f"after stage {stage}. Try providing `suggested_fields` or "
                f"a `domain_hint` to guide discovery."
            )

        try:
            model = json_schema_to_pydantic(json_schema, "DiscoveredSchema")
        except (SchemaError, ValueError, TypeError) as e:
            raise DiscoveryError(
                f"ThreeStageDiscovery: failed to build Pydantic model after "
                f"stage {stage}. The discovered schema may contain "
                f"unsupported types: {e}"
            ) from e

        lineage = [f"ThreeStageDiscovery:stage{stage}"]

        return Schema(
            model=model,
            json_schema=json_schema,
            field_metadata={
                "_discovery_report": {
                    "field_coverage": coverage,
                    **report_meta,
                },
            },
            lineage=lineage,
        )
