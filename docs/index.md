<p align="center">
  <img src="assets/logo.webp" alt="catchfly" width="200">
</p>

<p align="center">
  <em>Catch the structured data.</em>
</p>

**Catchfly** automates **schema discovery → structured extraction → normalization** from unstructured text at scale. Interchangeable strategies at each stage let you go from raw documents to clean, normalized data with minimal effort.

## Why Catchfly?

Extracting structured data from unstructured text with LLMs is solved at the unit level. But the **end-to-end workflow** — discovering what to extract, extracting consistently across thousands of documents, and normalizing the messy output — remains fragmented.

Catchfly connects all stages:

```
Documents → [Discovery] → Schema → [Extraction] → Records → [Field Selection] → [Normalization] → Clean Data
```

## Quick Example

```python
from catchfly import Pipeline
from catchfly.demo import load_samples

docs = load_samples("product_reviews")
pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(
    documents=docs,
    domain_hint="Electronics product reviews",
)
results.to_dataframe()
```

`Pipeline.quick()` auto-discovers a schema, extracts records, selects categorical fields via `LLMFieldSelector`, and normalizes them — zero config needed.

## Strategies at a Glance

| Stage | Strategy | Best For |
|---|---|---|
| **Discovery** | `SinglePassDiscovery` | Quick schema from a few samples |
| | `ThreeStageDiscovery` | Careful progressive refinement |
| | `SchemaOptimizer` | Enriching field descriptions for better extraction |
| **Extraction** | `LLMDirectExtraction` | Per-document structured extraction |
| | `TokenChunking`, `SentenceChunking`, `SemanticChunking` | Pluggable chunking via chonkie |
| **Field Selection** | `LLMFieldSelector` | Auto-detect normalizable fields (default in `quick()`) |
| | `StatisticalFieldSelector` | Zero-cost heuristic field detection |
| **Normalization** | `CascadeNormalization` | Chain strategies: Dictionary → LLM → Ontology |
| | `OntologyMapping` | Map to HPO/ICD-10/custom ontologies |
| | `LLMCanonicalization` | General-purpose, schema-aware, hierarchical merge |
| | `DictionaryNormalization` | Zero-cost exact or case-insensitive lookup |
| | `CompositeNormalization` | Route different fields to different strategies |
| | `EmbeddingClustering` | Fast, no LLM needed (after embedding) |
| | `KLLMeansClustering` | Surface-form deduplication |

## Next Steps

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Schema Discovery Guide](guides/discovery.md)
- [Field Selection Guide](guides/field-selection.md)
