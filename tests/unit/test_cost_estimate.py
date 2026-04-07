"""Unit tests for cost-estimation helpers."""

from __future__ import annotations

from catchfly.extraction._tokenize import (
    _heuristic_tokens,
    count_messages_tokens,
    count_tokens,
)
from catchfly.telemetry.tracker import (
    estimate_llm_cost,
    is_model_priced,
)


class TestHeuristicTokens:
    def test_empty_string(self) -> None:
        assert _heuristic_tokens("") == 0

    def test_four_chars_is_one_token(self) -> None:
        assert _heuristic_tokens("abcd") == 1

    def test_rounds_up(self) -> None:
        # 5 chars → ceil(5/4) == 2
        assert _heuristic_tokens("abcde") == 2

    def test_longer_string(self) -> None:
        text = "a" * 400
        assert _heuristic_tokens(text) == 100


class TestCountTokens:
    def test_returns_tuple(self) -> None:
        tokens, backend = count_tokens("hello world", "gpt-5.4-mini")
        assert isinstance(tokens, int)
        assert tokens > 0
        assert backend.startswith(("tiktoken:", "heuristic:"))

    def test_empty_is_zero(self) -> None:
        tokens, _ = count_tokens("", "gpt-5.4-mini")
        assert tokens == 0

    def test_unknown_model_falls_back(self) -> None:
        """Unknown model names still produce a count (via cl100k or heuristic)."""
        tokens, backend = count_tokens("hello", "not-a-real-model")
        assert tokens > 0
        assert backend.startswith(("tiktoken:", "heuristic:"))


class TestCountMessagesTokens:
    def test_per_message_overhead(self) -> None:
        """Each message contributes ~4 tokens of formatting overhead."""
        empty = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        tokens, _ = count_messages_tokens(empty, "gpt-5.4-mini")
        # At minimum 2 messages × 4 overhead = 8 + (role string tokens)
        assert tokens >= 8

    def test_scales_with_content(self) -> None:
        short = [{"role": "user", "content": "hi"}]
        long = [{"role": "user", "content": "hi " * 500}]
        short_tokens, _ = count_messages_tokens(short, "gpt-5.4-mini")
        long_tokens, _ = count_messages_tokens(long, "gpt-5.4-mini")
        assert long_tokens > short_tokens * 10


class TestEstimateLLMCost:
    def test_known_model(self) -> None:
        # gpt-5.4-mini: $0.15 / 1M input, $0.60 / 1M output
        cost = estimate_llm_cost("gpt-5.4-mini", 1_000_000, 1_000_000)
        assert cost == 0.15 + 0.60

    def test_unknown_model_returns_zero(self) -> None:
        assert estimate_llm_cost("unknown-model-xyz", 1000, 1000) == 0.0

    def test_custom_pricing_override(self) -> None:
        cost = estimate_llm_cost(
            "my-model",
            1_000_000,
            1_000_000,
            pricing={"my-model": (1.0, 2.0)},
        )
        assert cost == 3.0

    def test_override_merges_with_defaults(self) -> None:
        """Override adds to the table without removing defaults."""
        cost = estimate_llm_cost(
            "gpt-5.4-mini",
            1_000_000,
            0,
            pricing={"my-model": (99.0, 99.0)},
        )
        assert cost == 0.15  # gpt-5.4-mini defaults unchanged

    def test_zero_tokens_zero_cost(self) -> None:
        assert estimate_llm_cost("gpt-5.4-mini", 0, 0) == 0.0


class TestIsModelPriced:
    def test_known_model_true(self) -> None:
        assert is_model_priced("gpt-5.4-mini")
        assert is_model_priced("claude-sonnet-4-6")

    def test_unknown_model_false(self) -> None:
        assert not is_model_priced("not-a-real-model")
