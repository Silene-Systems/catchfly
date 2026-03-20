"""Tests for LLM client."""

from __future__ import annotations

from catchfly.providers.llm import LLMResponse, OpenAICompatibleClient

# Import mock from conftest
from tests.conftest import MockLLMClient


class TestLLMResponse:
    def test_create(self) -> None:
        resp = LLMResponse(content="hello", input_tokens=10, output_tokens=5, model="gpt-4o")
        assert resp.content == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.model == "gpt-4o"

    def test_defaults(self) -> None:
        resp = LLMResponse(content="hi")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.model == ""
        assert resp.raw is None


class TestMockLLMClient:
    async def test_acomplete(self) -> None:
        client = MockLLMClient(responses=["response 1", "response 2"])
        resp1 = await client.acomplete([{"role": "user", "content": "hi"}])
        resp2 = await client.acomplete([{"role": "user", "content": "hello"}])

        assert resp1.content == "response 1"
        assert resp2.content == "response 2"
        assert len(client.calls) == 2

    async def test_cycles_responses(self) -> None:
        client = MockLLMClient(responses=["only one"])
        resp1 = await client.acomplete([{"role": "user", "content": "a"}])
        resp2 = await client.acomplete([{"role": "user", "content": "b"}])
        assert resp1.content == "only one"
        assert resp2.content == "only one"

    async def test_records_call_params(self) -> None:
        client = MockLLMClient()
        await client.acomplete(
            [{"role": "user", "content": "test"}],
            model="gpt-5",
            temperature=0.5,
        )
        assert client.calls[0]["model"] == "gpt-5"
        assert client.calls[0]["temperature"] == 0.5


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
