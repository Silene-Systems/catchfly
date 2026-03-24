"""Cost and usage tracking for LLM calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from catchfly._types import UsageReport
from catchfly.exceptions import BudgetExceededError

logger = logging.getLogger(__name__)

# Approximate cost per 1M tokens (input, output) for common models.
# Users can override via cost_per_1m_tokens parameter.
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4": (2.50, 10.00),
    "gpt-5.4-mini": (0.15, 0.60),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "mistral-small-latest": (0.10, 0.30),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


@dataclass
class UsageRecord:
    """A single LLM call record."""

    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


class UsageTracker:
    """Tracks LLM usage across pipeline stages."""

    def __init__(
        self,
        max_cost_usd: float | None = None,
        cost_per_1m_tokens: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._records: list[UsageRecord] = []
        self.max_cost_usd = max_cost_usd
        self._pricing = {**_DEFAULT_PRICING, **(cost_per_1m_tokens or {})}

    def record(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float | None = None,
    ) -> None:
        """Record a single LLM call."""
        if cost_usd is None:
            cost_usd = self._estimate_cost(model, input_tokens, output_tokens)

        rec = UsageRecord(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        self._records.append(rec)

        logger.debug(
            "Usage: stage=%s model=%s in=%d out=%d cost=$%.4f",
            stage,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        )

        # Check budget after recording
        if self.max_cost_usd is not None:
            total = self.total_cost()
            if total > self.max_cost_usd:
                raise BudgetExceededError(spent=total, limit=self.max_cost_usd)

    def make_callback(self, stage: str) -> Any:
        """Create a usage callback for a specific pipeline stage.

        Returns a callable(model, input_tokens, output_tokens, latency_ms)
        that records usage to this tracker.
        """

        def _cb(
            model: str,
            input_tokens: int,
            output_tokens: int,
            latency_ms: float,
        ) -> None:
            self.record(stage, model, input_tokens, output_tokens, latency_ms)

        return _cb

    def total_cost(self) -> float:
        """Total cost across all recorded calls."""
        return sum(r.cost_usd for r in self._records)

    def total_input_tokens(self) -> int:
        """Total input tokens across all calls."""
        return sum(r.input_tokens for r in self._records)

    def total_output_tokens(self) -> int:
        """Total output tokens across all calls."""
        return sum(r.output_tokens for r in self._records)

    def report(self) -> UsageReport:
        """Generate a usage report with per-stage breakdown."""
        breakdown: dict[str, Any] = {}
        for rec in self._records:
            if rec.stage not in breakdown:
                breakdown[rec.stage] = {
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                    "latency_ms": 0.0,
                }
            stage_data = breakdown[rec.stage]
            stage_data["cost_usd"] += rec.cost_usd
            stage_data["input_tokens"] += rec.input_tokens
            stage_data["output_tokens"] += rec.output_tokens
            stage_data["calls"] += 1
            stage_data["latency_ms"] += rec.latency_ms

        return UsageReport(
            total_cost_usd=self.total_cost(),
            total_input_tokens=self.total_input_tokens(),
            total_output_tokens=self.total_output_tokens(),
            total_latency_ms=sum(r.latency_ms for r in self._records),
            breakdown=breakdown,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export all records as a dict for JSON serialization."""
        return {
            "total_cost_usd": self.total_cost(),
            "total_input_tokens": self.total_input_tokens(),
            "total_output_tokens": self.total_output_tokens(),
            "records": [
                {
                    "stage": r.stage,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                }
                for r in self._records
            ],
        }

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model pricing table."""
        pricing = self._pricing.get(model)
        if pricing is None:
            # Unknown model — return 0, user can override via cost_usd param
            return 0.0

        input_cost_per_1m, output_cost_per_1m = pricing
        return (input_tokens * input_cost_per_1m + output_tokens * output_cost_per_1m) / 1_000_000
