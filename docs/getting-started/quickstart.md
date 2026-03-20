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

# Discover schema → extract → normalize
results = pipeline.run(
    documents=docs,
    domain_hint="Electronics product reviews",
    normalize_fields=["pros"],
)

# Explore results
print(results.schema.json_schema)    # Discovered schema
print(results.to_dataframe())        # Extracted data as DataFrame
print(results.report)                # Cost & usage stats
```

`Pipeline.quick()` uses **SinglePassDiscovery + LLMDirectExtraction + EmbeddingClustering** — the lightest, fastest combination.

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

## Next Steps

- [Schema Discovery Guide](../guides/discovery.md) — SinglePass, ThreeStage, Optimizer
- [Normalization Guide](../guides/normalization.md) — Embedding, LLM, kLLMmeans
- [Pipeline Guide](../guides/pipeline.md) — Cost control, checkpoint, error handling
