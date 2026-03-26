<p align="center">
  <img src="docs/assets/logo.webp" alt="catchfly" width="200">
</p>

<p align="center">
  <em>Catch the structured data.</em>
</p>

<p align="center">
  <a href="https://catchfly.dev">Documentation</a> &bull;
  <a href="https://github.com/silene-systems/catchfly">GitHub</a> &bull;
  <a href="https://pypi.org/project/catchfly/">PyPI</a>
</p>

---

**Catchfly** automates **schema discovery → structured extraction → normalization** from unstructured text at scale. Interchangeable strategies at each stage let you go from raw documents to clean, normalized data with minimal effort.

## Quick Start

```bash
pip install catchfly[openai,clustering]
```

```python
from catchfly import Pipeline
from catchfly.demo import load_samples

docs = load_samples("product_reviews")

pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(
    documents=docs,
    domain_hint="Electronics product reviews",
    normalize_fields=["pros"],
)

results.to_dataframe()  # → pandas DataFrame
```

## Strategies at a Glance

| Stage | Strategy | Description |
|---|---|---|
| **Discovery** | `SinglePassDiscovery` | One LLM call → JSON Schema from sample docs |
| | `ThreeStageDiscovery` | 3-stage progressive refinement (initial → refine → expand) |
| | `SchemaOptimizer` | PARSE-style iterative field enrichment (descriptions, examples, synonyms) |
| **Extraction** | `LLMDirectExtraction` | Per-document extraction with tool calling, retries, chunking |
| **Normalization** | `CascadeNormalization` | Chain strategies with confidence-based routing + self-learning |
| | `OntologyMapping` | Embed → NN search → LLM rerank against HPO/custom ontologies |
| | `LLMCanonicalization` | LLM groups synonyms, map-reduce for large sets (>200 values) |
| | `EmbeddingClustering` | Embed → HDBSCAN/agglomerative → canonical selection |
| **Infrastructure** | `SchemaRegistry` | Version, diff, and persist schemas across runs |

## Biomedical Normalization

Map clinical terms to ontology entries using local SapBERT embeddings (zero API cost) with optional LLM reranking:

```python
from catchfly.normalization import CascadeNormalization, OntologyMapping
from catchfly.providers import SentenceTransformerEmbeddingClient

# SapBERT embeddings — 0.802 Acc@1 on BC5CDR, beats OpenAI embeddings
embed_client = SentenceTransformerEmbeddingClient()  # default: SapBERT

normalizer = OntologyMapping(
    ontology="hpo",
    embedding_client=embed_client,
    augment_queries=True,  # LLM generates alternative phrasings (+10-20pp recall)
)
result = await normalizer.anormalize(
    ["seizures", "high temperature", "low muscle tone"],
    context_field="phenotype",
)
# result.mapping: {"seizures": "Seizure", "high temperature": "Fever", ...}

# Self-learning cascade — learns from results, cheaper on re-runs
cascade = CascadeNormalization.default(
    dictionary={"ALT": "Alanine aminotransferase"},
    ontology="hpo",
    use_confidence=True,  # confidence-based routing between steps
)
result = await cascade.anormalize(values, context_field="phenotype")
cascade.learn(result)  # next run resolves known mappings instantly ($0)
```

Requires: `pip install catchfly[embeddings,medical]`

## Local Models (Ollama)

```python
pipeline = Pipeline.quick(
    model="qwen3.5",
    base_url="http://localhost:11434/v1",
)
```

Works with any OpenAI-compatible endpoint: Ollama, vLLM, LMStudio, llama.cpp.

## Modular Usage

Each stage works independently — use one, two, or all three:

```python
# Discovery
from catchfly.discovery.single_pass import SinglePassDiscovery
schema = SinglePassDiscovery(model="gpt-5.4-mini").discover(docs, domain_hint="...")

# Extraction (bring your own schema)
from catchfly.extraction.llm_direct import LLMDirectExtraction
records = LLMDirectExtraction(model="gpt-5.4-mini").extract(schema=MyModel, documents=docs)

# Normalization (bring your own data)
from catchfly.normalization.embedding_cluster import EmbeddingClustering
mapping = EmbeddingClustering().normalize(values=["NYC", "New York", "NY"], context_field="city")
```

## Schema Optimizer (PARSE-style)

Iteratively enrich field descriptions for better extraction and normalization:

```python
from catchfly.discovery.optimizer import SchemaOptimizer

optimizer = SchemaOptimizer(model="gpt-5.4-mini", num_iterations=3)
enriched = optimizer.optimize(schema=MyModel, test_documents=docs[:10])
# enriched.field_metadata has descriptions, examples, synonyms per field
```

## kLLMmeans with Schema-Seeded Warmstart

The core novel contribution — bridge schema optimization and normalization:

```python
from catchfly.normalization.kllmeans import KLLMeansClustering

normalizer = KLLMeansClustering(
    num_clusters=5,
    seed_from_schema=True,         # use enriched field descriptions as initial centroids
    summarize_every=3,             # LLM generates textual centroids every 3 iterations
)
result = normalizer.normalize(
    values=messy_values,
    context_field="medication",
    field_metadata=enriched.field_metadata["medication"],
)
```

## Production Features

```python
# Cost control
results = pipeline.run(documents=docs, max_cost_usd=20.0)

# Checkpoint/resume (for 1000+ documents)
results = pipeline.run(documents=large_corpus, checkpoint_dir="./state/")

# Error handling
extractor = LLMDirectExtraction(model="gpt-5.4-mini", on_error="collect")
results = extractor.extract(schema=MyModel, documents=docs)
print(results.errors)  # failed documents collected, not raised

# Export
results.to_dataframe()
results.to_csv("output.csv")
results.to_parquet("output.parquet")
```

## Async Support

All strategies are async-first with sync wrappers (Jupyter-safe):

```python
# Async
results = await pipeline.arun(documents=docs)

# Sync (auto-detects running event loop in notebooks)
results = pipeline.run(documents=docs)
```

## Installation

```bash
pip install catchfly                  # Core only (~5 MB)
pip install catchfly[openai]          # + OpenAI SDK
pip install catchfly[embeddings]      # + sentence-transformers (SapBERT, local)
pip install catchfly[clustering]      # + scikit-learn, numpy, umap
pip install catchfly[export]          # + pandas, pyarrow
pip install catchfly[medical]         # + ontology loaders (HPO)
pip install catchfly[all]             # Everything
```

Or with uv:

```bash
uv add catchfly[openai,clustering,export]
```

## Architecture

```
catchfly
├── discovery/
│   ├── SinglePassDiscovery        # 1-shot schema from samples
│   ├── ThreeStageDiscovery        # Progressive 3-stage refinement
│   └── SchemaOptimizer            # PARSE-style field enrichment
├── extraction/
│   └── LLMDirectExtraction        # Tool calling + retry + chunking
├── normalization/
│   ├── CascadeNormalization       # Chain strategies, confidence routing, learn()
│   ├── OntologyMapping            # Embed → NN → LLM rerank (RAG augmentation)
│   ├── LLMCanonicalization        # LLM synonym grouping (map-reduce)
│   ├── LearnedDictionaryCache     # Persist mappings for reuse across runs
│   └── EmbeddingClustering        # Embed → cluster → canonicalize
├── providers/
│   ├── OpenAICompatibleClient     # Any OpenAI-compatible LLM endpoint
│   ├── OpenAIEmbeddingClient      # API embeddings with caching
│   └── SentenceTransformerEmbeddingClient  # Local embeddings (SapBERT)
├── schema/
│   ├── SchemaRegistry             # Version + diff + persist
│   └── converters                 # JSON Schema ↔ Pydantic roundtrip
└── Pipeline                       # Orchestrator: quick(), run(), arun()
```

## Requirements

- Python 3.10+
- An OpenAI-compatible LLM endpoint (OpenAI, Anthropic, Mistral, Ollama, vLLM)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Links

- [Documentation](https://catchfly.dev)
- [GitHub](https://github.com/silene-systems/catchfly)
- [PyPI](https://pypi.org/project/catchfly/)
