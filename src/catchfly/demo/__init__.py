"""Built-in demo datasets for quick experimentation."""

from __future__ import annotations

import json
import logging
from importlib import resources
from typing import Any

from catchfly._types import Document

logger = logging.getLogger(__name__)

__all__ = ["load_samples"]


def __dir__() -> list[str]:
    return __all__


_AVAILABLE_DATASETS = ("product_reviews", "support_tickets", "case_reports")


def load_samples(name: str = "product_reviews") -> list[Document]:
    """Load a built-in demo dataset.

    Available datasets:
    - "product_reviews" — electronics reviews with inconsistent attributes
    - "support_tickets" — SaaS support tickets with messy categories
    - "case_reports" — short biomedical case report excerpts

    Returns a list of Document objects (~10 documents each).
    """
    if name not in _AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(_AVAILABLE_DATASETS)}")

    data_file = f"{name}.json"
    try:
        ref = resources.files("catchfly.demo").joinpath(data_file)
        raw = ref.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Demo dataset '{name}' not found. Expected file: catchfly/demo/{data_file}"
        ) from e

    items: list[dict[str, Any]] = json.loads(raw)
    documents = [
        Document(
            content=item["content"],
            id=item.get("id", f"{name}_{i}"),
            source=f"demo:{name}",
            metadata=item.get("metadata", {}),
        )
        for i, item in enumerate(items)
    ]

    logger.info("Loaded demo dataset '%s': %d documents", name, len(documents))
    return documents
