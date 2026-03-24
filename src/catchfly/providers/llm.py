"""LLM client abstraction over OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Protocol, runtime_checkable

from catchfly.exceptions import ProviderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM chat completion clients."""

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        *,
        schema_name: str = "output",
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def structured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        *,
        schema_name: str = "output",
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------


class LLMResponse:
    """Normalized response from an LLM completion call."""

    __slots__ = ("content", "input_tokens", "output_tokens", "model", "raw")

    def __init__(
        self,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        raw: Any = None,
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.raw = raw


# ---------------------------------------------------------------------------
# OpenAI-compatible implementation
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES = 3
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_INITIAL_BACKOFF = 1.0


class OpenAICompatibleClient:
    """LLM client for any OpenAI-compatible API endpoint.

    Works with OpenAI, Azure OpenAI, Anthropic, Mistral, Ollama, vLLM,
    LMStudio, llama.cpp server, and other OpenAI-compatible providers.

    Structured output uses a cascade of strategies (inspired by Pydantic-AI):
    1. Tool calling (most universal — works with all major providers)
    2. json_schema response_format (OpenAI, Anthropic)
    3. json_object response_format (OpenAI, some local models)
    4. Prompt-only fallback (always works)
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = 120.0,
        usage_callback: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_retries = max_retries
        self.timeout = timeout
        self.usage_callback = usage_callback

    def _make_async_client(self) -> Any:
        """Create a fresh AsyncOpenAI client.

        A new client is created per call to avoid 'Event loop is closed'
        errors when the client outlives the asyncio.run() scope.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAICompatibleClient. "
                "Install it with: pip install catchfly[openai]"
            ) from e

        kwargs: dict[str, Any] = {
            "api_key": self.api_key or "not-needed",
            "timeout": self.timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return openai.AsyncOpenAI(**kwargs)

    # ------------------------------------------------------------------
    # Plain text completion
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request asynchronously."""
        return await self._call(messages, model=model, temperature=temperature, **kwargs)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request synchronously."""
        from catchfly._compat import run_sync

        return run_sync(self.acomplete(messages, model=model, temperature=temperature, **kwargs))

    # ------------------------------------------------------------------
    # Structured output (cascading strategies)
    # ------------------------------------------------------------------

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
        """Get structured JSON output from the LLM.

        Tries strategies in order of reliability:
        1. Tool calling — most universal, works across all major providers
        2. json_schema response_format — OpenAI, Anthropic
        3. json_object response_format — OpenAI, some local models
        4. Prompt-only — always works (least reliable)
        """
        effective_model = model or self.model

        # Strategy 1: Tool calling
        try:
            return await self._structured_via_tools(
                messages, output_schema, schema_name, effective_model, temperature, **kwargs
            )
        except ProviderError as e:
            logger.debug("Tool calling failed for %s, trying json_schema: %s", effective_model, e)

        # Strategy 2: json_schema response_format
        try:
            return await self._structured_via_json_schema(
                messages, output_schema, schema_name, effective_model, temperature, **kwargs
            )
        except ProviderError as e:
            logger.debug(
                "json_schema format failed for %s, trying json_object: %s", effective_model, e
            )

        # Strategy 3: json_object response_format
        try:
            return await self._structured_via_json_object(
                messages, effective_model, temperature, **kwargs
            )
        except ProviderError as e:
            logger.debug(
                "json_object format failed for %s, falling back to prompt: %s",
                effective_model,
                e,
            )

        # Strategy 4: Plain prompting (no response_format)
        logger.warning(
            "All structured output strategies failed for %s, using prompt-only fallback",
            effective_model,
        )
        return await self._call(messages, model=effective_model, temperature=temperature, **kwargs)

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
        """Get structured JSON output synchronously."""
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

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    async def _structured_via_tools(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        name: str,
        model: str,
        temperature: float | None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Use tool calling to get structured output."""
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Output the extracted {name} data as structured JSON.",
                "parameters": schema,
            },
        }

        response = await self._call(
            messages,
            model=model,
            temperature=temperature,
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": name}},
            **kwargs,
        )

        # Extract content from tool call arguments
        raw = response.raw
        if raw and hasattr(raw, "choices") and raw.choices:
            message = raw.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_call = message.tool_calls[0]
                if hasattr(tool_call, "function") and tool_call.function:
                    arguments = tool_call.function.arguments
                    # Validate it's parseable JSON
                    json.loads(arguments)
                    return LLMResponse(
                        content=arguments,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        model=response.model,
                        raw=raw,
                    )

        raise ProviderError("Tool calling did not return tool_calls in response")

    async def _structured_via_json_schema(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        name: str,
        model: str,
        temperature: float | None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Use response_format with json_schema type."""
        return await self._call(
            messages,
            model=model,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema},
            },
            **kwargs,
        )

    async def _structured_via_json_object(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Use response_format with json_object type."""
        return await self._call(
            messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core call with retry logic
    # ------------------------------------------------------------------

    async def _call(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Low-level completion call with retry logic."""
        effective_model = model or self.model
        async_client = self._make_async_client()

        call_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            **kwargs,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.monotonic()
                logger.debug(
                    "LLM call attempt=%d model=%s messages=%d",
                    attempt + 1,
                    effective_model,
                    len(messages),
                )

                response = await async_client.chat.completions.create(**call_kwargs)
                elapsed_ms = (time.monotonic() - start) * 1000

                content = response.choices[0].message.content or ""
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                logger.debug(
                    "LLM response model=%s tokens_in=%d tokens_out=%d latency=%.0fms",
                    effective_model,
                    input_tokens,
                    output_tokens,
                    elapsed_ms,
                )

                if self.usage_callback is not None:
                    self.usage_callback(effective_model, input_tokens, output_tokens, elapsed_ms)

                await async_client.close()
                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=effective_model,
                    raw=response,
                )

            except Exception as e:
                last_error = e
                if not self._is_retryable(e) or attempt >= self.max_retries:
                    break

                backoff = _INITIAL_BACKOFF * (2**attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    backoff,
                    e,
                )
                await asyncio.sleep(backoff)

        await async_client.close()
        raise ProviderError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is worth retrying."""
        error_str = str(error).lower()
        if "rate" in error_str and "limit" in error_str:
            return True

        status_code = getattr(error, "status_code", None)
        return bool(status_code and status_code in _RETRY_STATUS_CODES)
