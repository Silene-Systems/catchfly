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

_EXPANSION_SYSTEM = """\
You are a schema expansion assistant. Given a JSON Schema, extraction results \
from a large sample, and field coverage statistics, decide which new fields \
to add and which rare fields to remove.

Rules:
- Add fields that appear in >20% of documents but aren't in the schema
- Flag fields with <5% coverage for removal
- Keep fields that are important even if rare (use judgment)

Output ONLY valid JSON with the same structure as before:
{
  "add_fields": {"field_name": {"type": "...", "description": "..."}},
  "remove_fields": ["field_to_remove"],
  "modify_fields": {},
  "rationale": "Brief explanation"
}\
"""


class ThreeStageDiscovery(BaseModel):
    """Discover schema through 3 progressive refinement stages.

    Stage 1: LLM proposes initial schema from 2-3 exemplar documents
    Stage 2: Schema refined against 5-10 docs (fix gaps, remove redundancies)
    Stage 3: Schema expanded against 50-100 docs (add recurring, flag rare)
    """

    model: str = "gpt-5.4-mini"
    stage1_samples: int = 3
    """Stage 1 uses few docs (2-3) to avoid over-fitting the initial schema."""
    stage2_samples: int = 10
    """Stage 2 uses a moderate sample to catch gaps without excessive LLM cost."""
    stage3_samples: int = 50
    """Stage 3 uses a larger sample to validate coverage at near-corpus scale."""
    max_doc_chars: int = 3000
    """Truncation limit per document — fits ~750 tokens, balancing context vs cost."""
    max_fields: int | None = None
    """Maximum number of fields in the discovered schema. None = no limit."""
    suggested_fields: list[str] | None = None
    """Field names to include in the schema. LLM may add others if found in documents."""
    human_review: bool = False
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.3
    """Low temperature for deterministic schema proposals."""

    _usage_callback: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

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

        client = OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            usage_callback=self._usage_callback,
        )

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
        json_schema = self._apply_changes(json_schema, refinement)
        report_meta["stages_completed"] = 2
        report_meta["stage2_changes"] = refinement

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
        json_schema = self._apply_changes(json_schema, expansion)
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
        """Attempt extraction on documents using current schema."""
        schema_str = json.dumps(json_schema, indent=2)
        results: list[dict[str, Any]] = []

        for doc in documents:
            prompt = (
                f"Extract data from this document using the schema below.\n\n"
                f"Schema:\n```json\n{schema_str}\n```\n\n"
                f"Document:\n---\n{doc.content[:self.max_doc_chars]}\n---\n\n"
                f"Output ONLY the extracted JSON."
            )
            try:
                sys_msg = "Extract structured data. Output only JSON."
                response = await client.acomplete(
                    [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                data = json.loads(response.content)
                results.append(data if isinstance(data, dict) else {})
            except json.JSONDecodeError:
                logger.debug(
                    "ThreeStageDiscovery: JSON parse error for doc '%s'",
                    doc.id or "(no id)",
                )
                results.append({})
            except (ProviderError, ValueError, KeyError, TypeError) as e:
                logger.debug(
                    "ThreeStageDiscovery: extraction failed for doc '%s': %s",
                    doc.id or "(no id)",
                    e,
                    exc_info=True,
                )
                results.append({})

        return results

    async def _get_refinement(
        self,
        client: OpenAICompatibleClient,
        json_schema: dict[str, Any],
        extracted: list[dict[str, Any]],
        coverage: dict[str, float],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Ask LLM to propose schema changes."""
        user_prompt = self._build_refinement_prompt(json_schema, extracted, coverage)

        try:
            response = await client.acomplete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

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
    ) -> dict[str, Any]:
        """Apply add/remove/modify changes to a JSON Schema."""
        schema = json.loads(json.dumps(json_schema))  # deep copy
        properties = schema.setdefault("properties", {})
        required = set(schema.get("required", []))

        # Add new fields
        for name, field_def in changes.get("add_fields", {}).items():
            if isinstance(field_def, dict) and name not in properties:
                properties[name] = field_def
                logger.debug("ThreeStageDiscovery: added field '%s'", name)

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

        model = None
        try:
            model = json_schema_to_pydantic(json_schema, "DiscoveredSchema")
        except (SchemaError, ValueError, TypeError) as e:
            logger.warning(
                "ThreeStageDiscovery: could not build Pydantic model: %s",
                e,
                exc_info=True,
            )

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
