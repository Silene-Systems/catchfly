"""Ontology loaders and embedding index for OntologyMapping normalization."""

from catchfly.ontology.csv_json import CSVSource, JSONSource
from catchfly.ontology.hpo import HPOSource
from catchfly.ontology.index import OntologyIndex
from catchfly.ontology.types import OntologyEntry, OntologySource

__all__ = [
    "CSVSource",
    "HPOSource",
    "JSONSource",
    "OntologyEntry",
    "OntologyIndex",
    "OntologySource",
]
