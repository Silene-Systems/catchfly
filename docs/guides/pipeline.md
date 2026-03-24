# Pipeline

The `Pipeline` orchestrates discovery → extraction → normalization in one call.

## Quick Start

```python
from catchfly import Pipeline

pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(
    documents=docs,
    domain_hint="Electronics product reviews",
    normalize_fields=["category", "brand"],
)
```

`Pipeline.quick()` creates: **SinglePassDiscovery + LLMDirectExtraction + LLMCanonicalization**.

## Custom Pipeline

Choose your own strategies:

```python
from catchfly import Pipeline
from catchfly.discovery import ThreeStageDiscovery
from catchfly.extraction import LLMDirectExtraction
from catchfly.normalization import LLMCanonicalization

pipeline = Pipeline(
    discovery=ThreeStageDiscovery(model="gpt-5.4-mini"),
    extraction=LLMDirectExtraction(model="gpt-5.4-mini"),
    normalization=LLMCanonicalization(model="gpt-5.4-mini"),
)
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

## Results

```python
results.schema              # Schema object (Pydantic model + JSON Schema)
results.records             # list of extracted Pydantic model instances
results.normalizations      # dict[field_name, NormalizationResult]
results.errors              # list[(Document, Exception)]
results.report              # UsageReport (cost, tokens, latency)

results.to_dataframe()      # pandas DataFrame
results.to_csv("out.csv")
results.to_parquet("out.parquet")
results.to_json("out.json")
```
