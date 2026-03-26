"""Tests for OntologyMapping normalization strategy."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from catchfly.exceptions import NormalizationError
from catchfly.normalization.ontology_mapping import OntologyMapping
from catchfly.ontology.types import OntologyEntry
from catchfly.providers.llm import LLMResponse

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockOntologyEmbedder:
    """Embedder with controlled vectors for ontology + query matching."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = 8) -> None:
        self.vectors = vectors
        self.dim = dim
        self.model = "mock-embedder"

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(t, self._hash_vec(t)) for t in texts]

    def _hash_vec(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(self.dim)]


class MockRerankingLLM:
    """Mock LLM that selects the first candidate."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self._responses = responses
        self.call_count = 0

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        if self._responses:
            data = self._responses[self.call_count % len(self._responses)]
        else:
            # Default: parse first candidate ID from user message
            user_msg = messages[-1]["content"]
            # Extract first ontology ID from the prompt
            import re
            match = re.search(r"\(([A-Z]+:\d+)\)", user_msg)
            selected_id = match.group(1) if match else None
            data = {
                "selected_id": selected_id,
                "confidence": 0.95,
                "rationale": "best match",
            }
        self.call_count += 1
        return LLMResponse(
            content=json.dumps(data), input_tokens=100, output_tokens=50
        )


SAMPLE_ENTRIES = [
    OntologyEntry(id="HP:001", name="Seizure", synonyms=("Seizures", "Epileptic seizure")),
    OntologyEntry(id="HP:002", name="Ataxia", synonyms=("Incoordination",)),
    OntologyEntry(id="HP:003", name="Fever", synonyms=("Pyrexia",)),
]

# Vectors: each entry along a different axis for clean separation
_VECTORS: dict[str, list[float]] = {
    # Seizure cluster
    "Seizure": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Seizures": [0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Epileptic seizure": [0.98, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Ataxia cluster
    "Ataxia": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Incoordination": [0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Fever cluster
    "Fever": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Pyrexia": [0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Query vectors
    "seizures": [0.97, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "unsteady gait": [0.05, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "high temperature": [0.0, 0.05, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0],
    "completely unknown": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


@contextmanager
def _patch_source(
    entries: list[OntologyEntry] | None = None,
) -> Iterator[None]:
    """Patch ontology source resolution to return sample entries."""
    import catchfly.normalization.ontology_mapping as mod

    _entries = entries or SAMPLE_ENTRIES
    orig_resolve = mod.OntologyMapping._resolve_source

    def _mock_resolve(self: Any) -> Any:
        class _MockSource:
            def load(self) -> list[OntologyEntry]:
                return list(_entries)
        return _MockSource()

    mod.OntologyMapping._resolve_source = _mock_resolve  # type: ignore[assignment]

    try:
        yield
    finally:
        mod.OntologyMapping._resolve_source = orig_resolve  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOntologyMapping:
    async def test_empty_values(self) -> None:
        normalizer = OntologyMapping(ontology="hpo")
        result = await normalizer.anormalize([])
        assert result.mapping == {}

    async def test_exact_match_no_reranking(self) -> None:
        """Value matching an ontology term name maps correctly without LLM."""
        embedder = MockOntologyEmbedder(_VECTORS)
        normalizer = OntologyMapping(
            ontology="hpo", reranking_model=None, embedding_client=embedder,
        )

        with _patch_source():
            result = await normalizer.anormalize(["seizures"], context_field="phenotype")

        assert result.mapping["seizures"] == "Seizure"
        assert result.metadata["per_value"]["seizures"]["ontology_id"] == "HP:001"

    async def test_multiple_values(self) -> None:
        embedder = MockOntologyEmbedder(_VECTORS)
        normalizer = OntologyMapping(
            ontology="hpo", reranking_model=None, embedding_client=embedder,
        )

        with _patch_source():
            result = await normalizer.anormalize(
                ["seizures", "unsteady gait", "high temperature"],
                context_field="phenotype",
            )

        assert result.mapping["seizures"] == "Seizure"
        assert result.mapping["unsteady gait"] == "Ataxia"
        assert result.mapping["high temperature"] == "Fever"

    async def test_with_reranking(self) -> None:
        embedder = MockOntologyEmbedder(_VECTORS)
        mock_llm = MockRerankingLLM()
        normalizer = OntologyMapping(
            ontology="hpo", reranking_model="mock",
            embedding_client=embedder, client=mock_llm,
        )

        with _patch_source():
            result = await normalizer.anormalize(["seizures"], context_field="phenotype")

        assert mock_llm.call_count == 1
        assert result.mapping["seizures"] == "Seizure"
        per_val = result.metadata["per_value"]["seizures"]
        assert per_val["ontology_id"] == "HP:001"
        assert per_val["confidence"] == 0.95

    async def test_explain_includes_ontology_id(self) -> None:
        embedder = MockOntologyEmbedder(_VECTORS)
        normalizer = OntologyMapping(
            ontology="hpo", reranking_model=None, embedding_client=embedder,
        )

        with _patch_source():
            result = await normalizer.anormalize(["seizures"], context_field="phenotype")

        explanation = result.explain("seizures")
        assert "HP:001" in explanation
        assert "Seizure" in explanation

    async def test_reranking_override(self) -> None:
        """LLM reranking can pick a different candidate than NN top-1."""
        embedder = MockOntologyEmbedder(_VECTORS)
        # Force LLM to select Ataxia instead of Seizure
        mock_llm = MockRerankingLLM(
            responses=[{"selected_id": "HP:002", "confidence": 0.8, "rationale": "override"}]
        )
        normalizer = OntologyMapping(
            ontology="hpo", reranking_model="mock",
            embedding_client=embedder, client=mock_llm,
        )

        with _patch_source():
            result = await normalizer.anormalize(["seizures"], context_field="phenotype")

        assert result.mapping["seizures"] == "Ataxia"
        assert result.metadata["per_value"]["seizures"]["ontology_id"] == "HP:002"

    async def test_sync_wrapper(self) -> None:
        embedder = MockOntologyEmbedder(_VECTORS)
        normalizer = OntologyMapping(
            ontology="hpo", reranking_model=None, embedding_client=embedder,
        )

        with _patch_source():
            result = normalizer.normalize(["seizures"], context_field="phenotype")

        assert result.mapping["seizures"] == "Seizure"

    async def test_n_mapped_in_metadata(self) -> None:
        embedder = MockOntologyEmbedder(_VECTORS)
        normalizer = OntologyMapping(
            ontology="hpo", reranking_model=None, embedding_client=embedder,
        )

        with _patch_source():
            result = await normalizer.anormalize(
                ["seizures", "high temperature"], context_field="phenotype"
            )

        assert result.metadata["n_mapped"] == 2


class TestResolveSource:
    def test_hpo_keyword(self) -> None:
        normalizer = OntologyMapping(ontology="hpo")
        source = normalizer._resolve_source()
        assert source.__class__.__name__ == "HPOSource"

    def test_obo_file(self, tmp_path: Path) -> None:
        obo = tmp_path / "hp.obo"
        obo.touch()
        normalizer = OntologyMapping(ontology=str(obo))
        source = normalizer._resolve_source()
        assert source.__class__.__name__ == "HPOSource"

    def test_csv_file(self, tmp_path: Path) -> None:
        csv = tmp_path / "custom.csv"
        csv.touch()
        normalizer = OntologyMapping(ontology=str(csv))
        source = normalizer._resolve_source()
        assert source.__class__.__name__ == "CSVSource"

    def test_json_file(self, tmp_path: Path) -> None:
        jf = tmp_path / "custom.json"
        jf.touch()
        normalizer = OntologyMapping(ontology=str(jf))
        source = normalizer._resolve_source()
        assert source.__class__.__name__ == "JSONSource"

    def test_unknown_raises(self) -> None:
        normalizer = OntologyMapping(ontology="unknown_thing")
        with pytest.raises(NormalizationError, match="Cannot resolve"):
            normalizer._resolve_source()


class TestParseRerankResponse:
    def _candidates(self) -> list[tuple[OntologyEntry, float]]:
        return [
            (OntologyEntry(id="HP:001", name="Seizure"), 0.9),
            (OntologyEntry(id="HP:002", name="Ataxia"), 0.5),
        ]

    def _lookup(self) -> dict[str, OntologyEntry]:
        return {e.id: e for e, _ in self._candidates()}

    def test_valid_response(self) -> None:
        content = json.dumps({"selected_id": "HP:001", "confidence": 0.95, "rationale": "exact"})
        match = OntologyMapping._parse_rerank_response(content, self._lookup(), self._candidates())
        assert match is not None
        assert match.entry.id == "HP:001"
        assert match.confidence == 0.95

    def test_null_selected_id(self) -> None:
        content = json.dumps({"selected_id": None, "confidence": 0.0, "rationale": "no match"})
        match = OntologyMapping._parse_rerank_response(content, self._lookup(), self._candidates())
        assert match is not None
        assert match.confidence == 0.0

    def test_invalid_json_falls_back(self) -> None:
        match = OntologyMapping._parse_rerank_response(
            "not json", self._lookup(), self._candidates()
        )
        assert match is not None
        assert match.entry.id == "HP:001"  # fallback to top-1

    def test_with_markdown_fences(self) -> None:
        inner = json.dumps({"selected_id": "HP:002", "confidence": 0.8, "rationale": "ok"})
        content = f"```json\n{inner}\n```"
        match = OntologyMapping._parse_rerank_response(content, self._lookup(), self._candidates())
        assert match is not None
        assert match.entry.id == "HP:002"
