"""Value normalization strategies."""

from catchfly.normalization.base import NormalizationStrategy
from catchfly.normalization.cascade import CascadeNormalization
from catchfly.normalization.composite import CompositeNormalization
from catchfly.normalization.dictionary import DictionaryNormalization
from catchfly.normalization.embedding_cluster import EmbeddingClustering
from catchfly.normalization.kllmeans import KLLMeansClustering
from catchfly.normalization.llm_canonical import LLMCanonicalization
from catchfly.normalization.ontology_mapping import OntologyMapping

__all__ = [
    "CascadeNormalization",
    "CompositeNormalization",
    "DictionaryNormalization",
    "EmbeddingClustering",
    "KLLMeansClustering",
    "LLMCanonicalization",
    "NormalizationStrategy",
    "OntologyMapping",
]


def __dir__() -> list[str]:
    return __all__
