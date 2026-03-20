"""Catchfly exception hierarchy.

All catchfly exceptions inherit from CatchflyError, enabling
users to catch any library error with a single except clause.
"""


class CatchflyError(Exception):
    """Base exception for all catchfly errors."""


class ProviderError(CatchflyError):
    """Error communicating with an LLM or embedding provider."""


class DiscoveryError(CatchflyError):
    """Error during schema discovery."""


class ExtractionError(CatchflyError):
    """Error during data extraction."""


class NormalizationError(CatchflyError):
    """Error during value normalization."""


class SchemaError(CatchflyError):
    """Error in schema conversion or validation."""


class BudgetExceededError(CatchflyError):
    """Pipeline cost exceeded the configured max_cost_usd limit."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: spent ${spent:.2f}, limit ${limit:.2f}")
