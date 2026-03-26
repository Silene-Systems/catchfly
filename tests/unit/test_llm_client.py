"""Tests for LLM client — provider routing and initialization."""

from __future__ import annotations

import pytest

from catchfly.providers.llm import OpenAICompatibleClient, _resolve_provider


class TestOpenAICompatibleClient:
    def test_init_defaults(self) -> None:
        client = OpenAICompatibleClient()
        assert client.model == "gpt-5.4-mini"
        assert client.base_url is None
        assert client.max_retries == 3

    def test_init_custom(self) -> None:
        client = OpenAICompatibleClient(
            model="qwen3.5",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
        )
        assert client.model == "qwen3.5"
        assert client.base_url == "http://localhost:11434/v1"
        assert client.api_key == "test-key"

    def test_init_anthropic_prefix(self) -> None:
        """model='anthropic/claude-...' strips prefix and routes correctly."""
        mp = pytest.MonkeyPatch()
        mp.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        try:
            client = OpenAICompatibleClient(model="anthropic/claude-haiku-4-5-20251001")
            assert client.model == "claude-haiku-4-5-20251001"
            assert client.base_url == "https://api.anthropic.com/v1/"
            assert client.api_key == "sk-ant-test"
        finally:
            mp.undo()

    def test_init_explicit_overrides_provider(self) -> None:
        """Explicit base_url/api_key always wins over provider routing."""
        client = OpenAICompatibleClient(
            model="anthropic/claude-haiku-4-5-20251001",
            base_url="http://my-proxy/v1",
            api_key="my-key",
        )
        assert client.model == "claude-haiku-4-5-20251001"
        assert client.base_url == "http://my-proxy/v1"
        assert client.api_key == "my-key"

    def test_init_unknown_prefix_passthrough(self) -> None:
        """Unknown prefix like 'mycompany/foo' is left as-is."""
        client = OpenAICompatibleClient(model="mycompany/foo", api_key="k")
        assert client.model == "mycompany/foo"
        assert client.api_key == "k"


class TestResolveProvider:
    def test_anthropic(self) -> None:
        mp = pytest.MonkeyPatch()
        mp.setenv("ANTHROPIC_API_KEY", "ant-key")
        try:
            model, url, key = _resolve_provider("anthropic/claude-3", None, None)
            assert model == "claude-3"
            assert url == "https://api.anthropic.com/v1/"
            assert key == "ant-key"
        finally:
            mp.undo()

    def test_groq(self) -> None:
        mp = pytest.MonkeyPatch()
        mp.setenv("GROQ_API_KEY", "groq-key")
        try:
            model, url, key = _resolve_provider("groq/llama-3", None, None)
            assert model == "llama-3"
            assert url == "https://api.groq.com/openai/v1/"
            assert key == "groq-key"
        finally:
            mp.undo()

    def test_no_prefix(self) -> None:
        model, url, key = _resolve_provider("gpt-4o", None, "my-key")
        assert model == "gpt-4o"
        assert url is None
        assert key == "my-key"

    def test_explicit_overrides(self) -> None:
        model, url, key = _resolve_provider("anthropic/claude-3", "http://proxy", "explicit-key")
        assert model == "claude-3"
        assert url == "http://proxy"
        assert key == "explicit-key"
