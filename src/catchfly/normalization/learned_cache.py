"""Self-learning dictionary cache for normalization mappings."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from catchfly._types import NormalizationResult

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


class LearnedDictionaryCache(BaseModel):
    """Persist normalization mappings for reuse across runs.

    Stores per-field mappings with confidence and provenance.  On load,
    creates :class:`DictionaryNormalization` instances from entries above
    ``min_confidence``.

    Cache format (JSON)::

        {
            "version": 1,
            "created": "2026-03-26T12:00:00Z",
            "updated": "2026-03-26T13:00:00Z",
            "field_mappings": {
                "phenotype": {
                    "seizures": {
                        "canonical": "Seizure",
                        "confidence": 0.95,
                        "source": "ontology_mapping"
                    }
                }
            }
        }
    """

    path: str
    min_confidence: float = 0.80
    """Only cache/load mappings above this confidence threshold."""

    def save(
        self,
        normalizations: dict[str, NormalizationResult],
    ) -> None:
        """Merge new normalization mappings into the cache file.

        Higher-confidence entries win on conflict (upsert semantics).

        Args:
            normalizations: Field name → NormalizationResult from a
                pipeline run.
        """
        from pathlib import Path

        cache_path = Path(self.path)
        existing = self._load_raw(cache_path)

        field_mappings: dict[str, dict[str, Any]] = existing.get(
            "field_mappings", {}
        )

        for field, result in normalizations.items():
            strategy = result.metadata.get("strategy", "unknown")
            per_value = result.metadata.get("per_value", {})
            field_entries = field_mappings.setdefault(field, {})

            for raw, canonical in result.mapping.items():
                if raw == canonical:
                    continue

                conf = per_value.get(raw, {}).get("confidence", 1.0)
                if conf < self.min_confidence:
                    continue

                # Keep higher confidence
                old = field_entries.get(raw)
                if old and old.get("confidence", 0.0) >= conf:
                    continue

                field_entries[raw] = {
                    "canonical": canonical,
                    "confidence": conf,
                    "source": strategy,
                }

        now = datetime.now(tz=timezone.utc).isoformat()
        data = {
            "version": _CACHE_VERSION,
            "created": existing.get("created", now),
            "updated": now,
            "field_mappings": field_mappings,
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        total = sum(len(v) for v in field_mappings.values())
        logger.info("LearnedDictionaryCache: saved %d mappings to %s", total, self.path)

    def load_dictionary(self, field: str) -> Any:
        """Load cached mappings for a field as a DictionaryNormalization.

        Returns ``None`` if no cached entries exist for the field.
        """
        from pathlib import Path

        from catchfly.normalization.dictionary import DictionaryNormalization

        data = self._load_raw(Path(self.path))
        field_entries = data.get("field_mappings", {}).get(field, {})
        if not field_entries:
            return None

        mapping: dict[str, str] = {}
        for raw, entry in field_entries.items():
            if entry.get("confidence", 0.0) >= self.min_confidence:
                mapping[raw] = entry["canonical"]

        if not mapping:
            return None

        return DictionaryNormalization(
            mapping=mapping,
            case_insensitive=True,
            passthrough_unmapped=True,
        )

    def load_all(self) -> dict[str, Any]:
        """Load cached mappings for all fields.

        Returns:
            Dict of field name → DictionaryNormalization. Empty fields
            are omitted.
        """
        from pathlib import Path

        data = self._load_raw(Path(self.path))
        result: dict[str, Any] = {}
        for field in data.get("field_mappings", {}):
            dn = self.load_dictionary(field)
            if dn is not None:
                result[field] = dn
        return result

    @staticmethod
    def _load_raw(cache_path: Any) -> dict[str, Any]:
        """Load and validate the cache file, returning raw data."""
        from pathlib import Path

        path = Path(cache_path)
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("LearnedDictionaryCache: corrupt cache at %s", cache_path)
            return {}

        if data.get("version") != _CACHE_VERSION:
            logger.info("LearnedDictionaryCache: version mismatch, ignoring cache")
            return {}

        return dict(data)
