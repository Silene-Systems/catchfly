"""LLM and embedding provider abstractions."""

from catchfly.providers.embeddings import EmbeddingClient, OpenAIEmbeddingClient
from catchfly.providers.llm import LLMClient, LLMResponse, OpenAICompatibleClient

__all__ = [
    "EmbeddingClient",
    "LLMClient",
    "LLMResponse",
    "OpenAICompatibleClient",
    "OpenAIEmbeddingClient",
]


def __dir__() -> list[str]:
    return __all__
