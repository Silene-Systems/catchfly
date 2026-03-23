"""Tests for ontology types and loaders."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from catchfly.ontology.csv_json import CSVSource, JSONSource
from catchfly.ontology.types import OntologyEntry

if TYPE_CHECKING:
    from pathlib import Path


class TestOntologyEntry:
    def test_frozen(self) -> None:
        entry = OntologyEntry(id="HP:0001250", name="Seizure", synonyms=("Seizures",))
        with pytest.raises(AttributeError):
            entry.name = "other"  # type: ignore[misc]

    def test_all_texts(self) -> None:
        entry = OntologyEntry(
            id="HP:0001250", name="Seizure", synonyms=("Seizures", "Epileptic seizure")
        )
        assert entry.all_texts == ["Seizure", "Seizures", "Epileptic seizure"]

    def test_all_texts_no_synonyms(self) -> None:
        entry = OntologyEntry(id="HP:0001250", name="Seizure")
        assert entry.all_texts == ["Seizure"]

    def test_default_synonyms(self) -> None:
        entry = OntologyEntry(id="X:1", name="Test")
        assert entry.synonyms == ()


class TestCSVSource:
    def test_loads_entries(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ontology.csv"
        csv_file.write_text(
            "id,name,synonyms\n"
            "HP:0001250,Seizure,Seizures;Epileptic seizure\n"
            "HP:0000708,Ataxia,Incoordination\n",
            encoding="utf-8",
        )
        source = CSVSource(csv_file)
        entries = source.load()

        assert len(entries) == 2
        assert entries[0].id == "HP:0001250"
        assert entries[0].name == "Seizure"
        assert entries[0].synonyms == ("Seizures", "Epileptic seizure")

    def test_missing_synonyms_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ontology.csv"
        csv_file.write_text("id,name\nHP:0001250,Seizure\n", encoding="utf-8")
        entries = CSVSource(csv_file).load()
        assert entries[0].synonyms == ()

    def test_empty_synonyms(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ontology.csv"
        csv_file.write_text("id,name,synonyms\nHP:1,Test,\n", encoding="utf-8")
        entries = CSVSource(csv_file).load()
        assert entries[0].synonyms == ()


class TestJSONSource:
    def test_loads_entries(self, tmp_path: Path) -> None:
        json_file = tmp_path / "ontology.json"
        data = [
            {"id": "HP:0001250", "name": "Seizure", "synonyms": ["Seizures"]},
            {"id": "HP:0000708", "name": "Ataxia"},
        ]
        json_file.write_text(json.dumps(data), encoding="utf-8")
        entries = JSONSource(json_file).load()

        assert len(entries) == 2
        assert entries[0].synonyms == ("Seizures",)
        assert entries[1].synonyms == ()
