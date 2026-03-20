"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest

from catchfly.exceptions import (
    BudgetExceededError,
    CatchflyError,
    DiscoveryError,
    ExtractionError,
    NormalizationError,
    ProviderError,
    SchemaError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_catchfly_error(self) -> None:
        errors = [
            ProviderError("test"),
            DiscoveryError("test"),
            ExtractionError("test"),
            NormalizationError("test"),
            SchemaError("test"),
            BudgetExceededError(spent=10.0, limit=5.0),
        ]
        for error in errors:
            assert isinstance(error, CatchflyError)

    def test_budget_exceeded_message(self) -> None:
        error = BudgetExceededError(spent=12.50, limit=10.00)
        assert error.spent == 12.50
        assert error.limit == 10.00
        assert "$12.50" in str(error)
        assert "$10.00" in str(error)

    def test_catch_all(self) -> None:
        with pytest.raises(CatchflyError):
            raise ProviderError("connection failed")
