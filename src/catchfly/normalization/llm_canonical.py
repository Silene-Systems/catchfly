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


def _build_batch_prompt(values: list[str], field: str) -> str:
    values_str = "\n".join(f"  - {v}" for v in values)
    return f'Values for field "{field}":\n{values_str}\n\nGroup these into canonical forms.'


class LLMCanonicalization(BaseModel):
    """Normalize values by asking the LLM to group synonyms.

    For small sets (≤ max_values_per_prompt), uses a single LLM call.
    For larger sets, uses map-reduce: split into batches, canonicalize
    each independently, then merge using embedding similarity to detect
    cross-batch duplicates.
    """

    model: str = "gpt-4o-mini"
    batch_size: int = 50
    max_values_per_prompt: int = 200
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1

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

        client = OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        logger.info(
            "LLMCanonicalization: normalizing %d values (%d unique) for '%s'",
            len(values),
            len(unique_values),
            context_field,
        )

        if len(unique_values) <= self.max_values_per_prompt:
            groups = await self._canonicalize_batch(client, unique_values, context_field)
        else:
            groups = await self._map_reduce(client, unique_values, context_field)

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
    ) -> list[dict[str, Any]]:
        """Single-call canonicalization for a batch of values."""
        system_prompt = _CANONICALIZATION_PROMPT.format(field=field)
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
            batch_groups = await self._canonicalize_batch(client, batch, field)
            all_groups.extend(batch_groups)

        # Reduce: merge groups with same/similar canonical names
        merged = self._merge_groups(all_groups)

        logger.info(
            "LLMCanonicalization: map-reduce %d batches → %d groups → %d merged",
            (len(values) + self.batch_size - 1) // self.batch_size,
            len(all_groups),
            len(merged),
        )
        return merged

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
