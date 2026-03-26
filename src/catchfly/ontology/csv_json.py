"""Custom ontology loaders for CSV and JSON files."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from catchfly.ontology.types import OntologyEntry

logger = logging.getLogger(__name__)


class CSVSource:
    """Load ontology entries from a CSV file.

    Expected columns: ``id``, ``name``, and optionally ``synonyms``
    (semicolon-separated).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[OntologyEntry]:
        entries: list[OntologyEntry] = []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file is empty: {self.path}")
            missing = {"id", "name"} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV file {self.path} missing required columns: {sorted(missing)}"
                )
            for row in reader:
                raw_synonyms = row.get("synonyms", "")
                synonyms = tuple(s.strip() for s in raw_synonyms.split(";") if s.strip())
                entries.append(OntologyEntry(id=row["id"], name=row["name"], synonyms=synonyms))

        logger.info("CSVSource: loaded %d entries from %s", len(entries), self.path)
        return entries


class JSONSource:
    """Load ontology entries from a JSON file.

    Expected format: list of ``{"id": "...", "name": "...", "synonyms": [...]}``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[OntologyEntry]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(
                f"JSON file {self.path} must contain a list of objects, got {type(data).__name__}"
            )
        entries: list[OntologyEntry] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"JSON entry {i} in {self.path} must be an object, got {type(item).__name__}"
                )
            for key in ("id", "name"):
                if key not in item:
                    raise ValueError(
                        f"JSON entry {i} in {self.path} missing required key: '{key}'"
                    )
            entries.append(
                OntologyEntry(
                    id=item["id"],
                    name=item["name"],
                    synonyms=tuple(item.get("synonyms", [])),
                )
            )

        logger.info("JSONSource: loaded %d entries from %s", len(entries), self.path)
        return entries
