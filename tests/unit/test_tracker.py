"""Tests for usage tracker."""

from __future__ import annotations

import pytest

from catchfly.exceptions import BudgetExceededError
from catchfly.telemetry.tracker import UsageTracker


class TestUsageTracker:
    def test_record_and_total(self) -> None:
        tracker = UsageTracker()
        tracker.record("discovery", "gpt-5.4-mini", 1000, 500, 200.0, cost_usd=0.01)
        tracker.record("extraction", "gpt-5.4-mini", 2000, 1000, 300.0, cost_usd=0.02)

        assert tracker.total_cost() == pytest.approx(0.03)
        assert tracker.total_input_tokens() == 3000
        assert tracker.total_output_tokens() == 1500

    def test_report_breakdown(self) -> None:
        tracker = UsageTracker()
        tracker.record("discovery", "gpt-5.4-mini", 1000, 500, 200.0, cost_usd=0.01)
        tracker.record("extraction", "gpt-5.4-mini", 2000, 1000, 300.0, cost_usd=0.02)

        report = tracker.report()
        assert report.total_cost_usd == pytest.approx(0.03)
        assert "discovery" in report.breakdown
        assert "extraction" in report.breakdown
        assert report.breakdown["discovery"]["calls"] == 1

    def test_budget_exceeded(self) -> None:
        tracker = UsageTracker(max_cost_usd=0.05)
        tracker.record("discovery", "gpt-5.4-mini", 1000, 500, 200.0, cost_usd=0.03)

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record("extraction", "gpt-5.4-mini", 2000, 1000, 300.0, cost_usd=0.03)

        assert exc_info.value.spent == pytest.approx(0.06)
        assert exc_info.value.limit == 0.05

    def test_no_budget_no_error(self) -> None:
        tracker = UsageTracker()  # no limit
        tracker.record("discovery", "gpt-5.4-mini", 1000, 500, 200.0, cost_usd=100.0)
        # Should not raise

    def test_auto_cost_estimation(self) -> None:
        tracker = UsageTracker()
        tracker.record("discovery", "gpt-5.4-mini", 1_000_000, 0, 200.0)
        # gpt-5.4-mini: $0.15 per 1M input tokens
        assert tracker.total_cost() == pytest.approx(0.15)

    def test_unknown_model_zero_cost(self) -> None:
        tracker = UsageTracker()
        tracker.record("discovery", "unknown-model", 1000, 500, 200.0)
        assert tracker.total_cost() == 0.0

    def test_to_dict(self) -> None:
        tracker = UsageTracker()
        tracker.record("discovery", "gpt-5.4-mini", 1000, 500, 200.0, cost_usd=0.01)
        data = tracker.to_dict()
        assert data["total_cost_usd"] == pytest.approx(0.01)
        assert len(data["records"]) == 1
        assert data["records"][0]["stage"] == "discovery"

    def test_make_callback(self) -> None:
        tracker = UsageTracker()
        cb = tracker.make_callback("extraction")
        cb("gpt-5.4-mini", 500, 200, 100.0)
        cb("gpt-5.4-mini", 300, 100, 50.0)
        assert tracker.total_input_tokens() == 800
        assert tracker.total_output_tokens() == 300
        report = tracker.report()
        assert report.breakdown["extraction"]["calls"] == 2

    def test_make_callback_budget_exceeded(self) -> None:
        tracker = UsageTracker(max_cost_usd=0.001)
        cb = tracker.make_callback("discovery")
        with pytest.raises(BudgetExceededError):
            cb("gpt-5.4", 100_000, 50_000, 500.0)
