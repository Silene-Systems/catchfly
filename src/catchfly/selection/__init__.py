"""Field selection strategies."""

from catchfly.selection.base import FieldSelector
from catchfly.selection.llm import LLMFieldSelector
from catchfly.selection.statistical import StatisticalFieldSelector

__all__ = [
    "FieldSelector",
    "LLMFieldSelector",
    "StatisticalFieldSelector",
]


def __dir__() -> list[str]:
    return __all__
