# Quick Start

## 1. Install

```bash
pip install catchfly[openai,clustering,export]
```

## 2. Run the Demo

Catchfly ships with built-in demo datasets so you can try it immediately:

```python
from catchfly import Pipeline
from catchfly.demo import load_samples

# Load 10 sample product reviews
docs = load_samples("product_reviews")

# Create pipeline with sensible defaults
pipeline = Pipeline.quick(model="gpt-5.4-mini")

# Discover schema → extract → auto-select fields → normalize
results = pipeline.run(
    documents=docs,
    domain_hint="Electronics product reviews",
)

# Explore results
print(results.schema.json_schema)    # Discovered schema
print(results.to_dataframe())        # Extracted data as DataFrame
print(results.report)                # Cost & usage stats
```

`Pipeline.quick()` uses **SinglePassDiscovery + LLMDirectExtraction + LLMFieldSelector + LLMCanonicalization** — the recommended zero-config default. The field selector automatically identifies categorical fields worth normalizing.

## 3. Available Demo Datasets

```python
load_samples("product_reviews")   # Electronics reviews
load_samples("support_tickets")   # SaaS support tickets
load_samples("case_reports")      # Biomedical case reports
```

## 4. Bring Your Own Documents

```python
from catchfly import Document, Pipeline

docs = [
    Document(content="Your text here...", id="doc1"),
    Document(content="Another document...", id="doc2"),
]

pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(docs, domain_hint="Your domain description")
```

You can also pass glob patterns:

```python
results = pipeline.run(
    documents=["data/*.txt", "reports/**/*.md"],
    domain_hint="Your domain description",
)
```

## 5. Bring Your Own Schema

Skip discovery and use a Pydantic model you define:

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    amount: float
    date: str
    items: list[str]

results = pipeline.run(docs, schema=Invoice)
```

## 6. Override Field Selection

By default `Pipeline.quick()` auto-selects fields to normalize. You can override this:

```python
# Explicit fields — bypasses auto-selection
results = pipeline.run(docs, normalize_fields=["category", "brand"])

# Normalize all string fields
results = pipeline.run(docs, normalize_fields="all")
```

## Next Steps

- [Schema Discovery Guide](../guides/discovery.md) — SinglePass, ThreeStage, Optimizer
- [Data Extraction Guide](../guides/extraction.md) — Chunking strategies, error handling
- [Field Selection Guide](../guides/field-selection.md) — Auto-detect normalizable fields
- [Normalization Guide](../guides/normalization.md) — Cascade, OntologyMapping, LLMCanonicalization
- [Pipeline Guide](../guides/pipeline.md) — Cost control, checkpoint, callbacks
