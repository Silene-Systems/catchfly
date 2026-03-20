# catchfly

*Many strategies, one pipeline — from unstructured text to structured data.*

**Catchfly** automates the pipeline of **schema discovery → structured extraction → normalization** from unstructured text at scale. Interchangeable strategies at each stage let you go from raw documents to clean, normalized, structured data with minimal effort.

## Quick Start

```bash
pip install catchfly[openai,clustering]
```

```python
from catchfly import Pipeline
from catchfly.demo import load_samples

# Load built-in demo data (10 product reviews)
docs = load_samples("product_reviews")

# One line to create a full pipeline
pipeline = Pipeline.quick(model="gpt-5.4-mini")

# Discover schema → extract records → normalize values
results = pipeline.run(
    documents=docs,
    domain_hint="Electronics product reviews",
    normalize_fields=["pros"],
)

print(results.schema)            # Discovered Pydantic model
print(results.to_dataframe())    # Extracted + normalized data
print(results.report)            # Cost & usage stats
```

## Local Models (Ollama)

```python
pipeline = Pipeline.quick(
    model="qwen3.5",
    base_url="http://localhost:11434/v1",
)
```

## Modular Usage

Each stage works independently:

```python
# Discovery only
from catchfly.discovery.single_pass import SinglePassDiscovery
discovery = SinglePassDiscovery(model="gpt-5.4-mini")
schema = discovery.discover(documents=docs, domain_hint="...")

# Extraction only (bring your own schema)
from catchfly.extraction.llm_direct import LLMDirectExtraction
extractor = LLMDirectExtraction(model="gpt-5.4-mini")
records = extractor.extract(schema=MyModel, documents=docs)

# Normalization only (bring your own data)
from catchfly.normalization.embedding_cluster import EmbeddingClustering
normalizer = EmbeddingClustering(embedding_model="text-embedding-3-small")
mapping = normalizer.normalize(values=["NYC", "New York", "NY"], context_field="city")
```

## Async Support

All strategies provide async methods — async-first, sync-friendly:

```python
# Async
results = await pipeline.arun(documents=docs, domain_hint="...")

# Sync (works in notebooks too)
results = pipeline.run(documents=docs, domain_hint="...")
```

## Installation

```bash
pip install catchfly                        # Core only (~5 MB)
pip install catchfly[openai]                # + OpenAI SDK
pip install catchfly[clustering]            # + scikit-learn, numpy, umap
pip install catchfly[export]                # + pandas, pyarrow
pip install catchfly[all]                   # Everything
```

Or with uv:

```bash
uv add catchfly[openai,clustering]
```

## Features

- **Schema Discovery** — LLM proposes a Pydantic schema from sample documents
- **Structured Extraction** — LLM extracts data per-document with retries and validation
- **Normalization** — Cluster and canonicalize messy values (embedding + HDBSCAN)
- **Async-first** — All operations support async with sync wrappers
- **LLM-agnostic** — Works with any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM)
- **Lightweight core** — Only pydantic + httpx; heavy deps are optional
- **Production-ready** — Error handling, cost tracking, provenance, export to DataFrame/CSV/Parquet

## Requirements

- Python 3.10+
- An OpenAI-compatible LLM endpoint

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Links

- [Documentation](https://catchfly.dev)
- [Product Requirements](catchfly_prd.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)
