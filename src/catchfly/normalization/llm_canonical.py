"""LLMCanonicalization — normalize values using LLM with map-reduce for scale."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from pydantic import BaseModel

from catchfly._compat import run_sync
from catchfly._types import NormalizationResult
from catchfly.exceptions import NormalizationError
from catchfly.providers.llm import OpenAICompatibleClient

logger = logging.getLogger(__name__)

_HIERARCHICAL_MERGE_PROMPT = """\
You are a data consolidation assistant. You are given a list of canonical group \
names for the field "{field}". Some of these names may refer to the same concept \
using different wording.

Rules:
- Merge names that refer to the same underlying concept into one group
- Choose the best (most standard/formal) name as the surviving canonical
- Do NOT merge names that represent genuinely different concepts
- Names that have no synonyms should appear as single-member groups
- Output ONLY valid JSON with the structure:
{{
  "groups": [
    {{
      "canonical": "the surviving name",
      "members": ["name1", "name2", ...],
      "rationale": "why these are the same concept"
    }}
  ]
}}\
"""

_CANONICALIZATION_PROMPT = """\
You are a data normalization assistant. Given a list of values for the field \
"{field}", group synonyms together and assign a canonical (standard) name to \
each group.

Rules:
- Group values that refer to the same concept
- Choose the most common or most formal variant as canonical name
- Every input value must appear in exactly one group
- Output ONLY valid JSON with the structure:
{{
  "groups": [
    {{
      "canonical": "the standard name",
      "members": ["variant1", "variant2", ...],
      "rationale": "why these are grouped"
    }}
  ]
}}\
"""


def _format_schema_context(field_metadata: dict[str, Any] | None) -> str:
    """Format field_metadata into a schema context block, or empty string."""
    if not field_metadata:
        return ""

    context_parts: list[str] = []
    if desc := field_metadata.get("description"):
        context_parts.append(f"Field description: {desc}")
    if examples := field_metadata.get("examples"):
        examples_str = ", ".join(str(e) for e in examples[:10])
        context_parts.append(f"Example canonical values: {examples_str}")
    if synonyms := field_metadata.get("synonyms"):
        synonyms_str = ", ".join(str(s) for s in synonyms[:10])
        context_parts.append(f"Field aliases: {synonyms_str}")
    if constraints := field_metadata.get("constraints"):
        context_parts.append(f"Constraints: {constraints}")

    if not context_parts:
        return ""

    return "\n\nSchema context for this field:\n" + "\n".join(context_parts)


def _build_system_prompt(field: str, field_metadata: dict[str, Any] | None = None) -> str:
    """Build canonicalization system prompt, optionally enriched with schema metadata."""
    return _CANONICALIZATION_PROMPT.format(field=field) + _format_schema_context(field_metadata)


def _build_hierarchical_system_prompt(
    field: str, field_metadata: dict[str, Any] | None = None
) -> str:
    """Build hierarchical merge system prompt with optional schema metadata."""
    return _HIERARCHICAL_MERGE_PROMPT.format(field=field) + _format_schema_context(field_metadata)


def _build_batch_prompt(values: list[str], field: str) -> str:
    values_str = "\n".join(f"  - {v}" for v in values)
    return f'Values for field "{field}":\n{values_str}\n\nGroup these into canonical forms.'


def _build_hierarchical_user_prompt(groups: list[dict[str, Any]]) -> str:
    """Format canonical names with member counts for hierarchical merge."""
    lines = [f'  - "{g["canonical"]}" ({len(g["members"])} members)' for g in groups]
    return (
        "Canonical groups to consolidate:\n"
        + "\n".join(lines)
        + "\n\nMerge names that refer to the same concept."
    )


class LLMCanonicalization(BaseModel):
    """Normalize values by asking the LLM to group synonyms.

    For small sets (≤ max_values_per_prompt), uses a single LLM call.
    For larger sets, uses map-reduce: split into batches, canonicalize
    each independently, string-merge identical canonicals, then optionally
    run a hierarchical LLM merge to consolidate semantically similar groups.
    """

    model: str = "gpt-5.4-mini"
    batch_size: int = 50
    max_values_per_prompt: int = 200
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    hierarchical_merge: bool = True
    hierarchical_merge_rounds: int = 1

    model_config = {"arbitrary_types_allowed": True}

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values asynchronously."""
        if not values:
            return NormalizationResult(mapping={})

        value_counts = Counter(values)
        unique_values = list(value_counts.keys())

        if len(unique_values) == 1:
            v = unique_values[0]
            return NormalizationResult(
                mapping={v: v},
                clusters={v: [v]},
                metadata={"strategy": "llm_canonicalization", "field": context_field},
            )

        field_metadata: dict[str, Any] | None = kwargs.get("field_metadata")

        client = OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            usage_callback=getattr(self, "_usage_callback", None),
        )

        logger.info(
            "LLMCanonicalization: normalizing %d values (%d unique) for '%s'",
            len(values),
            len(unique_values),
            context_field,
        )

        if len(unique_values) <= self.max_values_per_prompt:
            groups = await self._canonicalize_batch(
                client, unique_values, context_field, field_metadata
            )
        else:
            groups = await self._map_reduce(
                client, unique_values, context_field, field_metadata
            )

        return self._build_result(groups, value_counts, context_field)

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values synchronously."""
        return run_sync(self.anormalize(values, context_field=context_field, **kwargs))

    async def _canonicalize_batch(
        self,
        client: OpenAICompatibleClient,
        values: list[str],
        field: str,
        field_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Single-call canonicalization for a batch of values."""
        system_prompt = _build_system_prompt(field, field_metadata)
        user_prompt = _build_batch_prompt(values, field)

        response = await client.acomplete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )

        return self._parse_groups(response.content, values)

    async def _map_reduce(
        self,
        client: OpenAICompatibleClient,
        values: list[str],
        field: str,
        field_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Map-reduce canonicalization for large value sets."""
        # Map: split into batches and canonicalize each
        all_groups: list[dict[str, Any]] = []
        for i in range(0, len(values), self.batch_size):
            batch = values[i : i + self.batch_size]
            logger.debug(
                "LLMCanonicalization: map batch %d-%d of %d",
                i,
                min(i + self.batch_size, len(values)),
                len(values),
            )
            batch_groups = await self._canonicalize_batch(
                client, batch, field, field_metadata
            )
            all_groups.extend(batch_groups)

        # Reduce: merge groups with same/similar canonical names (string-based)
        merged = self._merge_groups(all_groups)

        # Hierarchical merge: second LLM pass to consolidate semantically similar groups
        if self.hierarchical_merge and len(merged) > 1:
            pre_hierarchical = len(merged)
            merged = await self._hierarchical_merge(
                client, merged, field, field_metadata
            )
            logger.info(
                "LLMCanonicalization: hierarchical merge %d → %d groups",
                pre_hierarchical,
                len(merged),
            )

        logger.info(
            "LLMCanonicalization: map-reduce %d batches → %d groups → %d final",
            (len(values) + self.batch_size - 1) // self.batch_size,
            len(all_groups),
            len(merged),
        )
        return merged

    async def _hierarchical_merge(
        self,
        client: OpenAICompatibleClient,
        groups: list[dict[str, Any]],
        field: str,
        field_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Second LLM pass to merge semantically similar canonical groups."""
        for round_idx in range(self.hierarchical_merge_rounds):
            canonical_names = [g["canonical"] for g in groups]
            if len(canonical_names) <= 1:
                break

            system_prompt = _build_hierarchical_system_prompt(field, field_metadata)

            # Batch canonical names if needed
            if len(canonical_names) <= self.max_values_per_prompt:
                user_prompt = _build_hierarchical_user_prompt(groups)
                response = await client.acomplete(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                merge_instructions = self._parse_groups(response.content, canonical_names)
            else:
                all_merge_groups: list[dict[str, Any]] = []
                for i in range(0, len(groups), self.batch_size):
                    batch = groups[i : i + self.batch_size]
                    batch_names = [g["canonical"] for g in batch]
                    user_prompt = _build_hierarchical_user_prompt(batch)
                    response = await client.acomplete(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                    )
                    all_merge_groups.extend(
                        self._parse_groups(response.content, batch_names)
                    )
                merge_instructions = self._merge_groups(all_merge_groups)

            new_groups = self._apply_hierarchical_merge(groups, merge_instructions)

            if len(new_groups) >= len(groups):
                logger.debug(
                    "LLMCanonicalization: hierarchical merge round %d — no reduction, stopping",
                    round_idx,
                )
                break

            logger.debug(
                "LLMCanonicalization: hierarchical merge round %d: %d → %d groups",
                round_idx,
                len(groups),
                len(new_groups),
            )
            groups = new_groups

        return groups

    @staticmethod
    def _apply_hierarchical_merge(
        original_groups: list[dict[str, Any]],
        merge_instructions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply hierarchical merge instructions to original groups.

        merge_instructions: groups where 'members' are canonical names to consolidate.
        """
        lookup: dict[str, dict[str, Any]] = {
            g["canonical"]: g for g in original_groups
        }
        consumed: set[str] = set()
        result: list[dict[str, Any]] = []

        for instruction in merge_instructions:
            surviving_canonical = instruction["canonical"]
            old_canonicals = instruction.get("members", [])
            rationale = instruction.get("rationale", "")

            # Collect all raw members from the original groups being merged
            merged_members: list[str] = []
            merged_rationales: list[str] = []
            for old_name in old_canonicals:
                if old_name in lookup and old_name not in consumed:
                    orig = lookup[old_name]
                    merged_members.extend(orig["members"])
                    if orig.get("rationale"):
                        merged_rationales.append(orig["rationale"])
                    consumed.add(old_name)
                elif old_name not in lookup:
                    logger.warning(
                        "Hierarchical merge: canonical '%s' not found in originals, skipping",
                        old_name,
                    )

            if merged_members:
                rationale_parts = [rationale] + merged_rationales
                combined_rationale = "; ".join(filter(None, rationale_parts))
                result.append({
                    "canonical": surviving_canonical,
                    "members": merged_members,
                    "rationale": combined_rationale,
                })

        # Keep any original groups that weren't consumed by any merge instruction
        for g in original_groups:
            if g["canonical"] not in consumed:
                result.append(g)

        return result

    def _parse_groups(
        self,
        content: str,
        expected_values: list[str],
    ) -> list[dict[str, Any]]:
        """Parse canonicalization groups from LLM response."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise NormalizationError(
                f"LLM response is not valid JSON: {e}\nResponse: {text[:500]}"
            ) from e

        groups: list[dict[str, Any]] = []
        raw_groups = data.get("groups", [])
        if not isinstance(raw_groups, list):
            raise NormalizationError(f"Expected 'groups' array, got {type(raw_groups).__name__}")

        for group in raw_groups:
            canonical = group.get("canonical", "")
            members = group.get("members", [])
            rationale = group.get("rationale", "")

            if canonical and members:
                groups.append(
                    {
                        "canonical": str(canonical),
                        "members": [str(m) for m in members],
                        "rationale": str(rationale),
                    }
                )

        # Check coverage — all expected values should be in some group
        grouped_values = {m for g in groups for m in g["members"]}
        missing = set(expected_values) - grouped_values
        if missing:
            # Add missing values as singletons
            for v in missing:
                groups.append(
                    {
                        "canonical": v,
                        "members": [v],
                        "rationale": "ungrouped",
                    }
                )

        return groups

    @staticmethod
    def _merge_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge groups with identical or very similar canonical names."""
        merged: dict[str, dict[str, Any]] = {}

        for group in groups:
            canonical = group["canonical"]
            canonical_lower = canonical.lower().strip()

            # Find existing group with same canonical (case-insensitive)
            match_key: str | None = None
            for key in merged:
                if key.lower().strip() == canonical_lower:
                    match_key = key
                    break

            if match_key is not None:
                # Merge into existing
                existing = merged[match_key]
                for member in group["members"]:
                    if member not in existing["members"]:
                        existing["members"].append(member)
                if group.get("rationale"):
                    existing["rationale"] += f"; {group['rationale']}"
            else:
                merged[canonical] = {
                    "canonical": canonical,
                    "members": list(group["members"]),
                    "rationale": group.get("rationale", ""),
                }

        return list(merged.values())

    @staticmethod
    def _build_result(
        groups: list[dict[str, Any]],
        value_counts: Counter[str],
        context_field: str,
    ) -> NormalizationResult:
        """Build NormalizationResult from grouped canonicals."""
        mapping: dict[str, str] = {}
        clusters: dict[str, list[str]] = {}
        explanations: dict[str, str] = {}

        for group in groups:
            canonical = group["canonical"]
            members = group["members"]
            rationale = group.get("rationale", "")
            clusters[canonical] = members

            for member in members:
                mapping[member] = canonical
                if member != canonical:
                    explanations[member] = (
                        f"grouped with '{canonical}' ({len(members)} members): {rationale}"
                    )
                else:
                    explanations[member] = f"canonical form ({len(members)} members): {rationale}"

        n_groups = len(groups)
        logger.info(
            "LLMCanonicalization: %d unique values → %d groups for '%s'",
            len(mapping),
            n_groups,
            context_field,
        )

        return NormalizationResult(
            mapping=mapping,
            clusters=clusters,
            metadata={
                "strategy": "llm_canonicalization",
                "field": context_field,
                "n_groups": n_groups,
                "explanations": explanations,
            },
        )
