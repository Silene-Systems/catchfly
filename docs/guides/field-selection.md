# Field Selection

Field selection is the decision layer between extraction and normalization — it determines **which fields** should be normalized.

## The Problem

When using schema discovery, you don't know field names in advance. The `normalize_fields` parameter requires either explicit field names (catch-22 with discovery) or `"all"` (too aggressive — normalizes free-text fields like `description` or `summary`). Field selection provides an intelligent middle ground.

## LLMFieldSelector

The default in `Pipeline.quick()`. Asks the LLM to classify each string field as normalizable or not, given the schema and sample values. Costs ~$0.001 (one short LLM call).

```python
from catchfly.selection import LLMFieldSelector

selector = LLMFieldSelector(model="gpt-5.4-mini")
fields = selector.select(schema=discovered_schema, records=extracted_records)
# e.g. ["brand", "category", "color"]  — free-text fields excluded
```

## StatisticalFieldSelector

Zero LLM cost — uses statistical heuristics on extracted values:

1. **Type filter** — only string and array-of-string fields
2. **Cardinality ratio** — `unique_values / total_records < 0.5` → repetitive, good candidate
3. **Value length** — average < 50 chars → likely categorical
4. **Name exclusion** — skip `description`, `text`, `summary`, `id`, `url`, etc.
5. **Minimum unique values** — skip fields with < 3 unique values

```python
from catchfly.selection import StatisticalFieldSelector

selector = StatisticalFieldSelector(
    max_cardinality_ratio=0.5,
    max_avg_length=50,
    exclude_patterns=["description", "summary", "author"],
    min_unique_values=3,
)
fields = selector.select(schema=schema, records=records)
```

## Pipeline Integration

| `normalize_fields` | `field_selector` | Behavior |
|---|---|---|
| `["brand", "color"]` | any | Explicit override — selector ignored |
| `"all"` | any | All string fields normalized |
| `None` | `LLMFieldSelector()` | Selector decides automatically |
| `None` | `None` | Skip normalization |

### Progressive disclosure

```python
from catchfly import Pipeline
from catchfly.selection import LLMFieldSelector, StatisticalFieldSelector

# Level 0: Zero-config — LLMFieldSelector included by default
pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(docs, domain_hint="Product reviews")

# Level 1: Swap selector
pipeline = Pipeline(
    discovery=...,
    extraction=...,
    normalization=...,
    field_selector=StatisticalFieldSelector(max_cardinality_ratio=0.3),
)

# Level 2: Explicit override — bypass selector entirely
results = pipeline.run(docs, normalize_fields=["brand", "category"])
```

## Custom Field Selector

Implement the `FieldSelector` protocol:

```python
from catchfly.selection import FieldSelector
from catchfly._types import Schema
from typing import Any

class DomainFieldSelector:
    """Always normalize these biomedical fields."""

    NORMALIZE = {"phenotype", "gene", "mutation", "medication"}

    def select(self, schema: Schema, records: list[Any], **kwargs: Any) -> list[str]:
        return [f for f in schema.json_schema.get("properties", {}) if f in self.NORMALIZE]

    async def aselect(self, schema: Schema, records: list[Any], **kwargs: Any) -> list[str]:
        return self.select(schema, records)
```
