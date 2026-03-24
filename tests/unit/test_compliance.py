"""Compliance tests — verify all strategy implementations meet Protocol contracts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel

from catchfly._types import Document, ExtractionResult, NormalizationResult, Schema
from catchfly.discovery.single_pass import SinglePassDiscovery
from catchfly.extraction.llm_direct import LLMDirectExtraction
from catchfly.normalization.dictionary import DictionaryNormalization
from catchfly.normalization.embedding_cluster import EmbeddingClustering
from catchfly.normalization.kllmeans import KLLMeansClustering
from catchfly.normalization.llm_canonical import LLMCanonicalization
from catchfly.providers.llm import LLMResponse

# ---------------------------------------------------------------------------
# Helpers: mock clients
# ---------------------------------------------------------------------------


class _MockLLMForCanonicalization:
    """Returns grouping JSON for LLMCanonicalization."""

    def __init__(self, values: list[str]) -> None:
        self._values = values

    async def acomplete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        groups = [
            {
                "canonical": v,
                "members": [v],
                "rationale": "identity",
            }
            for v in self._values
        ]
        return LLMResponse(
            content=json.dumps({"groups": groups}),
            input_tokens=100,
            output_tokens=50,
        )

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class _MockLLMForDiscovery:
    """Returns a valid JSON Schema for SinglePassDiscovery."""

    async def acomplete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["name"],
        }
        return LLMResponse(
            content=json.dumps(schema),
            input_tokens=200,
            output_tokens=100,
            model="mock",
        )

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class _MockLLMForExtraction:
    """Returns JSON matching a given schema for LLMDirectExtraction."""

    def __init__(self, response_data: dict[str, Any]) -> None:
        self._data = response_data

    async def acomplete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(self._data),
            input_tokens=200,
            output_tokens=100,
            model="mock",
        )

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class _MockKLLMeansLLM:
    """Returns first member as canonical name for KLLMeansClustering."""

    async def acomplete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        user_msg = messages[-1]["content"] if messages else ""
        lines = [
            ln.strip("- ").strip() for ln in user_msg.split("\n") if ln.startswith("-")
        ]
        name = lines[0] if lines else "cluster"
        return LLMResponse(content=name, input_tokens=50, output_tokens=10)

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class _MockKLLMeansEmbedder:
    """Deterministic embeddings for KLLMeansClustering compliance tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(self.dim)]


def _make_clustered_embeddings(
    groups: list[list[str]], dim: int = 8
) -> dict[str, list[float]]:
    """Create embeddings where texts in the same group are close together."""
    embeddings: dict[str, list[float]] = {}
    for group_idx, group in enumerate(groups):
        center = np.zeros(dim)
        center[group_idx % dim] = 1.0
        for text in group:
            vec = center + np.random.default_rng(hash(text) % 2**31).normal(0, 0.05, dim)
            embeddings[text] = vec.tolist()
    return embeddings


@contextmanager
def _patch_llm_canonical(mock_llm: Any) -> Iterator[None]:
    """Temporarily replace OpenAICompatibleClient in llm_canonical."""
    import catchfly.normalization.llm_canonical as mod

    original = mod.OpenAICompatibleClient
    mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    try:
        yield
    finally:
        mod.OpenAICompatibleClient = original  # type: ignore[assignment]


@contextmanager
def _patch_discovery(mock_llm: Any) -> Iterator[None]:
    """Temporarily replace OpenAICompatibleClient in single_pass."""
    import catchfly.discovery.single_pass as mod

    original = mod.OpenAICompatibleClient
    mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    try:
        yield
    finally:
        mod.OpenAICompatibleClient = original  # type: ignore[assignment]


@contextmanager
def _patch_extraction(mock_llm: Any) -> Iterator[None]:
    """Temporarily replace OpenAICompatibleClient in llm_direct."""
    import catchfly.extraction.llm_direct as mod

    original = mod.OpenAICompatibleClient
    mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    try:
        yield
    finally:
        mod.OpenAICompatibleClient = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Compliance check functions
# ---------------------------------------------------------------------------


async def check_normalizer_compliance(normalizer: Any, values: list[str]) -> None:
    """Verify a normalizer meets the NormalizationStrategy Protocol contract.

    Checks:
    - Returns NormalizationResult
    - mapping has entry for every input value (unique)
    - mapping values are all strings
    - Works with empty input (returns empty mapping)
    - Works with single value
    - Works with duplicates
    """
    # ---- Empty input ----
    empty_result = await normalizer.anormalize([], context_field="test")
    assert isinstance(empty_result, NormalizationResult), (
        "Empty input must return NormalizationResult"
    )
    assert empty_result.mapping == {}, "Empty input must produce empty mapping"

    # ---- Single value ----
    single_result = await normalizer.anormalize(["alpha"], context_field="test")
    assert isinstance(single_result, NormalizationResult), (
        "Single input must return NormalizationResult"
    )
    assert "alpha" in single_result.mapping, "Single value must appear in mapping"
    assert isinstance(single_result.mapping["alpha"], str), (
        "Mapping values must be strings"
    )

    # ---- Normal multi-value input ----
    result = await normalizer.anormalize(values, context_field="test")
    assert isinstance(result, NormalizationResult), "Must return NormalizationResult"

    unique_values = set(values)
    for v in unique_values:
        assert v in result.mapping, f"Value '{v}' missing from mapping"
        assert isinstance(result.mapping[v], str), f"Mapping['{v}'] must be a string"

    # ---- Duplicates ----
    duplicated = values + values
    dup_result = await normalizer.anormalize(duplicated, context_field="test")
    assert isinstance(dup_result, NormalizationResult), (
        "Duplicates must return NormalizationResult"
    )
    for v in unique_values:
        assert v in dup_result.mapping, f"Value '{v}' missing from duplicate mapping"


async def check_discovery_compliance(
    strategy: Any, documents: list[Document]
) -> None:
    """Verify a discovery strategy meets the DiscoveryStrategy Protocol contract.

    Checks:
    - Returns Schema
    - Schema has json_schema dict
    - Schema has model (can be None)
    """
    schema = await strategy.adiscover(documents)
    assert isinstance(schema, Schema), "Must return Schema"
    assert isinstance(schema.json_schema, dict), "json_schema must be a dict"
    assert "properties" in schema.json_schema, "json_schema must have 'properties'"
    # model can be None or a BaseModel subclass
    if schema.model is not None:
        assert issubclass(schema.model, BaseModel), "model must be a BaseModel subclass"


async def check_extraction_compliance(
    strategy: Any,
    schema_class: type[BaseModel],
    documents: list[Document],
) -> None:
    """Verify an extraction strategy meets the ExtractionStrategy Protocol contract.

    Checks:
    - Returns ExtractionResult
    - records is a list
    - errors is a list
    """
    result = await strategy.aextract(schema_class, documents)
    assert isinstance(result, ExtractionResult), "Must return ExtractionResult"
    assert isinstance(result.records, list), "records must be a list"
    assert isinstance(result.errors, list), "errors must be a list"

    # Empty input
    empty_result = await strategy.aextract(schema_class, [])
    assert isinstance(empty_result, ExtractionResult), "Empty must return ExtractionResult"
    assert empty_result.records == [], "Empty input must produce empty records"


# ---------------------------------------------------------------------------
# Test schema / documents used across compliance tests
# ---------------------------------------------------------------------------


class SimpleRecord(BaseModel):
    name: str
    score: int


SAMPLE_DOCS = [
    Document(content="Alice scored 95 on the test.", id="doc1", source="src1.txt"),
    Document(content="Bob scored 88 on the test.", id="doc2", source="src2.txt"),
]

NORMALIZATION_VALUES = ["NYC", "New York", "ny", "Los Angeles", "LA"]


# ---------------------------------------------------------------------------
# Parametrized compliance tests — normalizers
# ---------------------------------------------------------------------------


class TestNormalizerCompliance:
    """Run check_normalizer_compliance on all concrete normalizer implementations."""

    async def test_dictionary_normalization(self) -> None:
        normalizer = DictionaryNormalization(
            mapping={"NYC": "New York", "ny": "New York", "LA": "Los Angeles"},
            case_insensitive=False,
            passthrough_unmapped=True,
        )
        await check_normalizer_compliance(normalizer, NORMALIZATION_VALUES)

    async def test_embedding_clustering(self) -> None:
        groups = [["NYC", "New York", "ny"], ["Los Angeles", "LA"]]
        mock_embeddings = _make_clustered_embeddings(groups)

        async def mock_embed(texts: list[str]) -> list[list[float]]:
            return [mock_embeddings.get(t, [0.0] * 8) for t in texts]

        normalizer = EmbeddingClustering(
            clustering_algorithm="agglomerative",
            similarity_threshold=0.5,
            reduce_dimensions=False,
        )

        with patch.object(normalizer, "_get_embeddings", side_effect=mock_embed):
            await check_normalizer_compliance(normalizer, NORMALIZATION_VALUES)

    async def test_llm_canonicalization(self) -> None:
        mock_llm = _MockLLMForCanonicalization(
            list(set(NORMALIZATION_VALUES))
        )
        normalizer = LLMCanonicalization(model="mock")

        with _patch_llm_canonical(mock_llm):
            await check_normalizer_compliance(normalizer, NORMALIZATION_VALUES)

    async def test_kllmeans_clustering(self) -> None:
        mock_llm = _MockKLLMeansLLM()
        mock_embedder = _MockKLLMeansEmbedder()

        normalizer = KLLMeansClustering(
            num_clusters=2,
            num_iterations=3,
            summarize_every=100,  # disable LLM summaries during compliance check
        )

        with (
            patch(
                "catchfly.normalization.kllmeans.OpenAIEmbeddingClient",
                return_value=mock_embedder,
            ),
            patch(
                "catchfly.normalization.kllmeans.OpenAICompatibleClient",
                return_value=mock_llm,
            ),
        ):
            await check_normalizer_compliance(normalizer, NORMALIZATION_VALUES)


# ---------------------------------------------------------------------------
# Parametrized compliance tests — discovery
# ---------------------------------------------------------------------------


class TestDiscoveryCompliance:
    """Run check_discovery_compliance on all concrete discovery implementations."""

    async def test_single_pass_discovery(self) -> None:
        mock_llm = _MockLLMForDiscovery()
        strategy = SinglePassDiscovery(model="mock")

        with _patch_discovery(mock_llm):
            await check_discovery_compliance(strategy, SAMPLE_DOCS)


# ---------------------------------------------------------------------------
# Parametrized compliance tests — extraction
# ---------------------------------------------------------------------------


class TestExtractionCompliance:
    """Run check_extraction_compliance on all concrete extraction implementations."""

    async def test_llm_direct_extraction(self) -> None:
        mock_llm = _MockLLMForExtraction({"name": "Alice", "score": 95})
        extractor = LLMDirectExtraction(model="mock", max_retries=0)

        with _patch_extraction(mock_llm):
            await check_extraction_compliance(extractor, SimpleRecord, SAMPLE_DOCS)
