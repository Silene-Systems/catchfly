"""Tests for demo datasets."""

from __future__ import annotations

import pytest

from catchfly._types import Document
from catchfly.demo import load_samples


class TestLoadSamples:
    @pytest.mark.parametrize("name", ["product_reviews", "support_tickets", "case_reports"])
    def test_load_dataset(self, name: str) -> None:
        docs = load_samples(name)
        assert len(docs) == 10
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.content for d in docs)
        assert all(d.source == f"demo:{name}" for d in docs)

    def test_default_is_product_reviews(self) -> None:
        docs = load_samples()
        assert docs[0].source == "demo:product_reviews"

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_samples("nonexistent")

    def test_documents_have_ids(self) -> None:
        docs = load_samples("product_reviews")
        assert all(d.id is not None for d in docs)
        assert docs[0].id == "review_001"
