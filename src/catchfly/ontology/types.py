"""Core ontology data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class OntologyEntry:
    """A single term in an ontology.

    Attributes:
        id: Ontology identifier (e.g., "HP:0001250").
        name: Preferred label (e.g., "Seizure").
        synonyms: Alternative labels (e.g., ("Seizures", "Epileptic seizure")).
    """

    id: str
    name: str
    synonyms: tuple[str, ...] = ()

    @property
    def all_texts(self) -> list[str]:
        """Name followed by all synonyms — used for embedding."""
        return [self.name, *self.synonyms]


@runtime_checkable
class OntologySource(Protocol):
    """Protocol for ontology loaders."""

    def load(self) -> list[OntologyEntry]:
        """Load ontology entries from the source."""
        ...
