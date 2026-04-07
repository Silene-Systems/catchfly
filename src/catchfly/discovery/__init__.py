"""Schema discovery strategies."""

from catchfly.discovery.base import DiscoveryReport, DiscoveryStrategy
from catchfly.discovery.optimizer import SchemaOptimizer
from catchfly.discovery.refine import (
    SchemaFeedback,
    arefine_schema,
    refine_schema,
)
from catchfly.discovery.single_pass import SinglePassDiscovery
from catchfly.discovery.three_stage import ThreeStageDiscovery

__all__ = [
    "DiscoveryReport",
    "DiscoveryStrategy",
    "SchemaFeedback",
    "SchemaOptimizer",
    "SinglePassDiscovery",
    "ThreeStageDiscovery",
    "arefine_schema",
    "refine_schema",
]


def __dir__() -> list[str]:
    return __all__
