"""Value normalization strategies."""

from catchfly.normalization.base import NormalizationStrategy
from catchfly.normalization.embedding_cluster import EmbeddingClustering
from catchfly.normalization.kllmeans import KLLMeansClustering
from catchfly.normalization.llm_canonical import LLMCanonicalization
from catchfly.normalization.ontology_mapping import OntologyMapping

__all__ = [
    "EmbeddingClustering",
    "KLLMeansClustering",
    "LLMCanonicalization",
    "NormalizationStrategy",
    "OntologyMapping",
]
