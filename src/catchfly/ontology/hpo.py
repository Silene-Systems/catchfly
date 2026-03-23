"""HPO (Human Phenotype Ontology) loader via pronto."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from catchfly.ontology.types import OntologyEntry

logger = logging.getLogger(__name__)


def _import_pronto() -> Any:
    try:
        import pronto

        return pronto
    except ImportError as e:
        raise ImportError(
            "pronto is required for HPO ontology loading. "
            "Install it with: pip install catchfly[medical]"
        ) from e


class HPOSource:
    """Load HPO terms from an OBO file.

    If *path* is ``None``, pronto downloads ``hp.obo`` from OBO Foundry
    on first use.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None

    def load(self) -> list[OntologyEntry]:
        pronto = _import_pronto()

        if self.path:
            ont = pronto.Ontology(str(self.path))
        else:
            ont = pronto.Ontology.from_obo_library("hp.obo")

        entries: list[OntologyEntry] = []
        for term in ont.terms():
            if term.obsolete:
                continue
            synonyms = tuple(syn.description for syn in term.synonyms)
            entries.append(
                OntologyEntry(id=term.id, name=term.name, synonyms=synonyms)
            )

        logger.info("HPOSource: loaded %d terms", len(entries))
        return entries
