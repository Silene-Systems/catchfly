"""Shared test fixtures for catchfly."""

from __future__ import annotations

from typing import Any

from catchfly.providers.llm import LLMResponse


class MockLLMClient:
    """Mock LLM client that returns pre-configured responses.

    Supports both acomplete() and astructured_complete() — the
    structured variant returns the same canned content.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock response"]
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                **kwargs,
            }
        )
        content = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        return LLMResponse(
            content=content,
            input_tokens=100,
            output_tokens=50,
            model=model or "mock-model",
        )

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        *,
        schema_name: str = "output",
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Structured output — returns same canned responses as acomplete."""
        return await self.acomplete(messages, model=model, temperature=temperature, **kwargs)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from catchfly._compat import run_sync

        return run_sync(self.acomplete(messages, model=model, temperature=temperature, **kwargs))

    def structured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        *,
        schema_name: str = "output",
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from catchfly._compat import run_sync

        return run_sync(
            self.astructured_complete(
                messages,
                output_schema,
                schema_name=schema_name,
                model=model,
                temperature=temperature,
                **kwargs,
            )
        )


class MockEmbeddingClient:
    """Mock embedding client that returns deterministic vectors."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._deterministic_vector(t) for t in texts]

    def embed(self, texts: list[str]) -> list[list[float]]:
        from catchfly._compat import run_sync

        return run_sync(self.aembed(texts))

    def _deterministic_vector(self, text: str) -> list[float]:
        """Generate a deterministic vector from text hash."""
        h = hash(text)
        return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(self.dimensions)]
