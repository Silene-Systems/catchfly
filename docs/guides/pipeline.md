# Pipeline

The `Pipeline` orchestrates discovery → extraction → field selection → normalization in one call.

## Quick Start

```python
from catchfly import Pipeline

pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(
    documents=docs,
    domain_hint="Electronics product reviews",
)
```

`Pipeline.quick()` creates: **SinglePassDiscovery + LLMDirectExtraction + LLMFieldSelector + LLMCanonicalization**. Fields to normalize are auto-selected.

## Custom Pipeline

Choose your own strategies:

```python
from catchfly import Pipeline
from catchfly.discovery import ThreeStageDiscovery
from catchfly.extraction import LLMDirectExtraction
from catchfly.normalization import LLMCanonicalization
from catchfly.selection import StatisticalFieldSelector

pipeline = Pipeline(
    discovery=ThreeStageDiscovery(model="gpt-5.4-mini"),
    extraction=LLMDirectExtraction(model="gpt-5.4-mini"),
    normalization=LLMCanonicalization(model="gpt-5.4-mini"),
    field_selector=StatisticalFieldSelector(),
)
```

## Per-Field Normalization

Route different fields to different strategies using a dict:

```python
from catchfly.normalization import OntologyMapping, LLMCanonicalization, DictionaryNormalization

pipeline = Pipeline(
    discovery=...,
    extraction=...,
    normalization={
        "phenotype": OntologyMapping(ontology="hpo"),
        "medication": DictionaryNormalization(mapping=drug_dict),
        "category": LLMCanonicalization(model="gpt-5.4-mini"),
    },
)
```

## Document Loading

Pass `Document` objects, or use glob patterns to load files:

```python
from catchfly import Document

# Document objects
docs = [Document(content="...", id="doc1")]

# Glob patterns — auto-resolved to Documents
results = pipeline.run(
    documents=["data/*.txt", "reports/**/*.md"],
    domain_hint="...",
)
```

## Schema Callback

Inspect or modify the discovered schema before extraction:

```python
def review_schema(schema):
    print(f"Discovered {len(schema.json_schema['properties'])} fields")
    return schema  # return modified schema, or None to keep as-is

results = pipeline.run(
    documents=docs,
    domain_hint="...",
    on_schema_ready=review_schema,
)
```

## Normalize Fields Override

```python
# Auto-select via field_selector (default in Pipeline.quick())
results = pipeline.run(docs)

# Explicit field list — bypasses selector
results = pipeline.run(docs, normalize_fields=["category", "brand"])

# Normalize all string/array-of-string fields
results = pipeline.run(docs, normalize_fields="all")
```

## Cost Control

```python
# Estimate cost before running
estimate = pipeline.estimate_cost(documents=docs)
print(estimate)  # {"discovery": 0.01, "extraction": 0.15, ...}

# Set a hard budget limit
results = pipeline.run(documents=docs, max_cost_usd=20.0)
# Halts gracefully if budget exceeded, returns partial results
```

## Progress Tracking

```python
pipeline = Pipeline.quick(model="gpt-5.4-mini", verbose=True)   # tqdm progress bars
pipeline = Pipeline.quick(model="gpt-5.4-mini", verbose=False)  # silent (default)
```

## Checkpoint & Resume

For large corpora (1000+ documents), enable checkpoint to resume after interruption:

```python
results = pipeline.run(
    documents=large_corpus,
    checkpoint_dir="./catchfly_state/",
)
# If interrupted, re-run the same command — already-processed docs are skipped
```

Checkpoint files:
- `schema.json` — discovered schema
- `records.jsonl` — extracted records (append-safe)
- `state.json` — processed document IDs

## Skip Discovery

Bring your own Pydantic schema:

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    amount: float
    date: str

results = pipeline.run(documents=docs, schema=Invoice)
```

## Error Handling

```python
pipeline = Pipeline.quick(model="gpt-5.4-mini", on_error="collect")
results = pipeline.run(docs, domain_hint="...")

for doc, error in results.errors:
    print(f"Failed: {doc.id} — {error}")
```

Options: `"raise"` (default), `"skip"`, `"collect"`.

## Results

```python
results.schema              # Schema object (Pydantic model + JSON Schema)
results.records             # list of extracted Pydantic model instances
results.normalizations      # dict[field_name, NormalizationResult]
results.errors              # list[(Document, Exception)]
results.report              # UsageReport (cost, tokens, latency)
results.report.cost_usd     # total cost in USD

results.to_dataframe()      # pandas DataFrame
results.to_csv("out.csv")
results.to_parquet("out.parquet")
results.to_json("out.json")
```
