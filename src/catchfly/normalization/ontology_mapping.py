"""OntologyMapping — normalize values by mapping to ontology terms."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, PrivateAttr

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL
from catchfly._parsing import strip_markdown_fences
from catchfly._types import NormalizationResult
from catchfly.exceptions import NormalizationError, ProviderError
from catchfly.ontology.csv_json import CSVSource, JSONSource
from catchfly.ontology.hpo import HPOSource
from catchfly.ontology.index import OntologyIndex

if TYPE_CHECKING:
    from catchfly.ontology.types import OntologyEntry, OntologySource
    from catchfly.providers.llm import OpenAICompatibleClient

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = """\
You are a biomedical ontology mapping assistant. Given a clinical term \
and candidate ontology matches, select the single best match.

Rules:
- Select the candidate that best represents the input term
- If none of the candidates match, set selected_id to null
- Output ONLY valid JSON with the structure:
{{"selected_id": "ONTO:1234", "confidence": 0.95, "rationale": "why this match"}}\
"""


_AUGMENT_SYSTEM = """\
You are a biomedical terminology assistant. For each input term, \
generate {n} alternative phrasings that might appear in medical ontologies.

Include formal medical terms, common abbreviations, lay terms, and synonyms. \
Do NOT include the original term. Output ONLY valid JSON:
{{"phrasings": {{"<term1>": ["alt1", "alt2", ...], "<term2>": ["alt1", ...]}}}}\
"""


@dataclass
class _OntologyMatch:
    """Internal: result of matching a value to an ontology entry."""

    entry: OntologyEntry
    confidence: float
    rationale: str = ""


class OntologyMapping(BaseModel):
    """Normalize values by mapping to ontology terms.

    Process:
    1. Load ontology entries from the specified source
    2. Build an embedding index over all term names and synonyms
    3. For each input value, find top-K nearest ontology terms
    4. Optionally rerank candidates via LLM to select the best match

    The mapping result stores canonical names as strings (backward
    compatible). Ontology IDs and confidence scores are available
    in ``result.metadata["per_value"]``.
    """

    ontology: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranking_model: str | None = DEFAULT_MODEL
    top_k: int = 5
    """5 nearest neighbors provides sufficient recall without excessive reranking cost."""
    confidence_threshold: float = 0.0
    """0.0 = accept all matches; increase to filter low-confidence mappings."""
    reranking_concurrency: int = 30
    """30 concurrent LLM calls balances throughput vs rate-limit pressure."""
    cache_dir: str | None = None
    base_url: str | None = None
    api_key: str | None = None

    augment_queries: bool = False
    """When True, generate alternative phrasings per value via LLM before
    embedding search. Improves recall by +10-20pp on biomedical benchmarks
    (LLMAEL/GRF, ACL 2025).  Requires an extra LLM call per batch."""
    augmentation_skip_threshold: float = 0.95
    """Skip augmentation for values whose top-1 NN score exceeds this
    threshold (already high-confidence, no need for extra phrasings)."""
    augmentation_n_phrasings: int = 5
    """Number of alternative phrasings to generate per value."""
    augmentation_batch_size: int = 30
    """Number of values to augment per LLM call."""

    client: Any | None = None
    """Pre-configured LLM client for reranking."""

    embedding_client: Any | None = None
    """Pre-configured embedding client."""

    _usage_callback: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> Any:
        """Return the injected LLM client or create a default one."""
        if self.client is not None:
            return self.client
        from catchfly.providers.llm import OpenAICompatibleClient

        return OpenAICompatibleClient(
            model=self.reranking_model,
            base_url=self.base_url,
            api_key=self.api_key,
            usage_callback=self._usage_callback,
        )

    def _get_embedding_client(self) -> Any:
        """Return the injected embedding client or create a default one."""
        if self.embedding_client is not None:
            return self.embedding_client
        from catchfly.providers.embeddings import OpenAIEmbeddingClient

        return OpenAIEmbeddingClient(
            model=self.embedding_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values asynchronously by mapping to ontology terms."""
        if not values:
            return NormalizationResult(mapping={})

        unique_values = list(dict.fromkeys(values))

        # 1. Load ontology
        source = self._resolve_source()
        entries = source.load()
        if not entries:
            raise NormalizationError(
                f"Ontology '{self.ontology}' loaded 0 entries. "
                f"Check that the ontology path exists and contains valid data."
            )

        # 2. Build embedding index
        embed_client = self._get_embedding_client()
        cache_path = self._resolve_cache_path()
        index = OntologyIndex(entries, embed_client, cache_path=cache_path)
        await index.build()

        # 3. Embed input values
        logger.info(
            "OntologyMapping: mapping %d values for '%s' against %d entries",
            len(unique_values),
            context_field,
            len(entries),
        )
        query_embeddings = await embed_client.aembed(unique_values)

        # 4. Nearest-neighbor search
        nn_results = index.search(query_embeddings, top_k=self.top_k)

        # 4b. RAG augmentation: generate phrasings, search, merge
        if self.augment_queries and self.reranking_model:
            llm_client_for_augment = self._get_client()
            nn_results = await self._augment_and_merge(
                llm_client_for_augment,
                embed_client,
                index,
                unique_values,
                nn_results,
                context_field,
            )

        # 5. LLM reranking (optional)
        if self.reranking_model:
            llm_client = self._get_client()
            matches = await self._rerank_batch(
                llm_client, unique_values, nn_results, context_field
            )
        else:
            matches = [
                _OntologyMatch(entry=cands[0][0], confidence=cands[0][1])
                if cands
                else None
                for cands in nn_results
            ]

        # 6. Build result
        return self._build_result(
            unique_values, matches, nn_results, context_field
        )

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values synchronously."""
        return run_sync(
            self.anormalize(values, context_field=context_field, **kwargs)
        )

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------

    def _resolve_source(self) -> OntologySource:
        """Resolve the ontology config string to a loader."""
        ont = self.ontology.strip()
        if ont.lower() == "hpo":
            return HPOSource()

        path = Path(ont)
        suffix = path.suffix.lower()
        if suffix == ".obo":
            return HPOSource(path)
        if suffix == ".csv":
            return CSVSource(path)
        if suffix == ".json":
            return JSONSource(path)

        raise NormalizationError(
            f"Cannot resolve ontology '{ont}'. "
            "Use 'hpo', or a path to .obo, .csv, or .json file."
        )

    def _resolve_cache_path(self) -> Path | None:
        """Determine cache file path, or None to disable caching."""
        if self.cache_dir is not None:
            if not self.cache_dir:
                return None  # explicit empty string = disable
            return Path(self.cache_dir) / "ontology_index.json"

        # Default: alongside the ontology file
        ont = self.ontology.strip()
        if ont.lower() == "hpo":
            return None  # downloaded ontology — no stable path
        path = Path(ont)
        if path.exists():
            return path.with_suffix(path.suffix + ".catchfly_index.json")
        return None

    # ------------------------------------------------------------------
    # LLM reranking
    # ------------------------------------------------------------------

    async def _rerank_batch(
        self,
        client: OpenAICompatibleClient,
        values: list[str],
        nn_results: list[list[tuple[OntologyEntry, float]]],
        context_field: str,
    ) -> list[_OntologyMatch | None]:
        """Rerank NN candidates for each value via LLM."""
        sem = asyncio.Semaphore(self.reranking_concurrency)

        # Use indexed tasks so we can return results in original order
        async def _indexed(
            idx: int, val: str, cands: list[tuple[OntologyEntry, float]]
        ) -> tuple[int, _OntologyMatch | None]:
            match = await self._rerank_one(client, sem, val, cands, context_field)
            return idx, match

        tasks = [
            _indexed(i, val, cands)
            for i, (val, cands) in enumerate(zip(values, nn_results, strict=True))
        ]

        # Stream results with progress bar if tqdm is available
        results: list[_OntologyMatch | None] = [None] * len(values)
        try:
            from tqdm.asyncio import tqdm_asyncio  # type: ignore[import-untyped]

            for coro in tqdm_asyncio.as_completed(
                tasks, desc="HPO reranking", total=len(tasks)
            ):
                idx, match = await coro
                results[idx] = match
        except ImportError:
            for coro in asyncio.as_completed(tasks):
                idx, match = await coro
                results[idx] = match

        return results

    async def _rerank_one(
        self,
        client: OpenAICompatibleClient,
        sem: asyncio.Semaphore,
        value: str,
        candidates: list[tuple[OntologyEntry, float]],
        context_field: str,
    ) -> _OntologyMatch | None:
        """Rerank candidates for a single value."""
        if not candidates:
            return None

        # Build candidate list for prompt
        lines: list[str] = []
        entry_lookup: dict[str, OntologyEntry] = {}
        for i, (entry, score) in enumerate(candidates, 1):
            syns = ", ".join(entry.synonyms[:5]) if entry.synonyms else "none"
            lines.append(
                f"{i}. {entry.name} ({entry.id}) — score: {score:.3f}"
                f"\n   Synonyms: {syns}"
            )
            entry_lookup[entry.id] = entry

        user_prompt = (
            f'Field: "{context_field}"\n'
            f'Input term: "{value}"\n\n'
            f"Candidates:\n" + "\n".join(lines)
        )

        async with sem:
            try:
                response = await client.acomplete(
                    [
                        {"role": "system", "content": _RERANK_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                return self._parse_rerank_response(
                    response.content, entry_lookup, candidates
                )
            except (ProviderError, json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    "OntologyMapping: reranking failed for '%s', using NN top-1: %s",
                    value,
                    e,
                    exc_info=True,
                )
                return _OntologyMatch(
                    entry=candidates[0][0], confidence=candidates[0][1]
                )

    @staticmethod
    def _parse_rerank_response(
        content: str,
        entry_lookup: dict[str, OntologyEntry],
        candidates: list[tuple[OntologyEntry, float]],
    ) -> _OntologyMatch | None:
        """Parse LLM reranking response."""
        text = strip_markdown_fences(content)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback to top-1
            if candidates:
                return _OntologyMatch(
                    entry=candidates[0][0], confidence=candidates[0][1]
                )
            return None

        selected_id = data.get("selected_id")
        confidence = float(data.get("confidence", 0.0))
        rationale = str(data.get("rationale", ""))

        if selected_id and selected_id in entry_lookup:
            return _OntologyMatch(
                entry=entry_lookup[selected_id],
                confidence=confidence,
                rationale=rationale,
            )

        # selected_id is null or unknown — no match
        if candidates:
            return _OntologyMatch(
                entry=candidates[0][0],
                confidence=0.0,
                rationale=rationale or "LLM found no match",
            )
        return None

    # ------------------------------------------------------------------
    # RAG augmentation
    # ------------------------------------------------------------------

    async def _augment_and_merge(
        self,
        client: Any,
        embed_client: Any,
        index: OntologyIndex,
        values: list[str],
        nn_results: list[list[tuple[OntologyEntry, float]]],
        context_field: str,
    ) -> list[list[tuple[OntologyEntry, float]]]:
        """Generate phrasings for low-confidence values, search, merge."""
        # Identify values needing augmentation
        to_augment: list[str] = []
        augment_indices: list[int] = []
        for i, (value, candidates) in enumerate(
            zip(values, nn_results, strict=True)
        ):
            top_score = candidates[0][1] if candidates else 0.0
            if top_score < self.augmentation_skip_threshold:
                to_augment.append(value)
                augment_indices.append(i)

        if not to_augment:
            logger.debug("OntologyMapping: all values above skip threshold, no augmentation")
            return nn_results

        logger.info(
            "OntologyMapping: augmenting %d/%d values for '%s'",
            len(to_augment),
            len(values),
            context_field,
        )

        # Generate phrasings via LLM
        phrasings_per_value = await self._generate_phrasings(
            client, to_augment, context_field
        )

        # Flatten phrasings and embed them
        flat_phrasings: list[str] = []
        phrasing_to_value_idx: list[int] = []
        for local_idx, phrases in enumerate(phrasings_per_value):
            for p in phrases:
                flat_phrasings.append(p)
                phrasing_to_value_idx.append(local_idx)

        if not flat_phrasings:
            return nn_results

        phrasing_embeddings = await embed_client.aembed(flat_phrasings)
        phrasing_nn = index.search(phrasing_embeddings, top_k=self.top_k)

        # Merge phrasing results into original nn_results
        merged = [list(r) for r in nn_results]
        for flat_idx, local_idx in enumerate(phrasing_to_value_idx):
            global_idx = augment_indices[local_idx]
            merged[global_idx] = self._merge_candidates(
                merged[global_idx], phrasing_nn[flat_idx], self.top_k
            )

        return merged

    async def _generate_phrasings(
        self,
        client: Any,
        values: list[str],
        context_field: str,
    ) -> list[list[str]]:
        """Generate alternative phrasings in batches via LLM."""
        all_phrasings: list[list[str]] = [[] for _ in values]
        value_to_idx = {v: i for i, v in enumerate(values)}

        system_prompt = _AUGMENT_SYSTEM.format(n=self.augmentation_n_phrasings)

        for batch_start in range(0, len(values), self.augmentation_batch_size):
            batch = values[batch_start : batch_start + self.augmentation_batch_size]
            user_prompt = (
                f'Field: "{context_field}"\n'
                f"Terms to rephrase:\n"
                + "\n".join(f"  - {v}" for v in batch)
            )

            try:
                response = await client.acomplete(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                text = strip_markdown_fences(response.content)
                data = json.loads(text)
                phrasings_dict = data.get("phrasings", {})

                for term, phrases in phrasings_dict.items():
                    if term in value_to_idx and isinstance(phrases, list):
                        all_phrasings[value_to_idx[term]] = [
                            str(p) for p in phrases[: self.augmentation_n_phrasings]
                        ]
            except (ProviderError, json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    "OntologyMapping: augmentation failed for batch at %d: %s",
                    batch_start,
                    e,
                )
                # Graceful fallback: no phrasings for this batch

        return all_phrasings

    @staticmethod
    def _merge_candidates(
        original: list[tuple[OntologyEntry, float]],
        extra: list[tuple[OntologyEntry, float]],
        top_k: int,
    ) -> list[tuple[OntologyEntry, float]]:
        """Merge two candidate lists, dedup by entry ID, keep top-k by score."""
        seen: dict[str, tuple[OntologyEntry, float]] = {}
        for entry, score in original:
            if entry.id not in seen or score > seen[entry.id][1]:
                seen[entry.id] = (entry, score)
        for entry, score in extra:
            if entry.id not in seen or score > seen[entry.id][1]:
                seen[entry.id] = (entry, score)
        merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        values: list[str],
        matches: list[_OntologyMatch | None],
        nn_results: list[list[tuple[OntologyEntry, float]]],
        context_field: str,
    ) -> NormalizationResult:
        """Build NormalizationResult from ontology matches."""
        mapping: dict[str, str] = {}
        per_value: dict[str, dict[str, Any]] = {}
        explanations: dict[str, str] = {}

        for value, match, candidates in zip(values, matches, nn_results, strict=True):
            if match and match.confidence >= 0:
                mapping[value] = match.entry.name
                per_value[value] = {
                    "ontology_id": match.entry.id,
                    "confidence": match.confidence,
                    "synonyms": list(match.entry.synonyms),
                }
                explanations[value] = (
                    f"→ {match.entry.name} ({match.entry.id}), "
                    f"confidence={match.confidence:.3f}"
                )
                if match.rationale:
                    explanations[value] += f": {match.rationale}"
            else:
                # No match — identity mapping
                mapping[value] = value
                best = candidates[0] if candidates else None
                per_value[value] = {
                    "ontology_id": None,
                    "confidence": best[1] if best else 0.0,
                }
                if best:
                    explanations[value] = (
                        f"no match (best candidate: {best[0].name} "
                        f"({best[0].id}) at {best[1]:.3f})"
                    )
                else:
                    explanations[value] = "no candidates found"

        n_mapped = sum(1 for v in per_value.values() if v["ontology_id"])
        logger.info(
            "OntologyMapping: %d/%d values mapped for '%s'",
            n_mapped,
            len(values),
            context_field,
        )

        return NormalizationResult(
            mapping=mapping,
            clusters=None,
            metadata={
                "strategy": "ontology_mapping",
                "field": context_field,
                "per_value": per_value,
                "explanations": explanations,
                "n_mapped": n_mapped,
            },
        )
