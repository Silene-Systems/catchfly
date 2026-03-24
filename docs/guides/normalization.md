# Normalization

Catchfly offers four normalization strategies for cleaning up messy extracted values.

## OntologyMapping

Map values to standardized ontology terms (HPO, ICD-10, custom). Uses embedding nearest-neighbor search with optional LLM reranking.

```python
from catchfly.normalization import OntologyMapping

normalizer = OntologyMapping(
    ontology="hpo",                        # or path to .obo, .csv, .json
    embedding_model="text-embedding-3-small",
    reranking_model="gpt-5.4-mini",        # None to skip LLM reranking
    top_k=5,                               # candidates per value
)

result = normalizer.normalize(
    values=["seizures", "epileptic fits", "high temperature"],
    context_field="phenotype",
)

print(result.mapping)
# {"seizures": "Seizure", "epileptic fits": "Seizure", "high temperature": "Fever"}

# Rich metadata with ontology IDs and confidence scores
print(result.metadata["per_value"]["seizures"])
# {"ontology_id": "HP:0001250", "confidence": 0.95, "synonyms": ["Seizures", ...]}

print(result.explain("seizures"))
# → Seizure (HP:0001250), confidence=0.950: best match
```

### Supported ontology formats

| Format | Usage |
|--------|-------|
| **HPO** (Human Phenotype Ontology) | `ontology="hpo"` or path to `.obo` file |
| **Custom CSV** | `ontology="my_terms.csv"` — columns: `id`, `name`, `synonyms` (semicolon-separated) |
| **Custom JSON** | `ontology="my_terms.json"` — list of `{"id", "name", "synonyms"}` |

### Embedding index caching

The first run embeds all ontology terms (~50k texts for HPO). Results are cached to disk automatically alongside the ontology file. Subsequent runs load from cache instantly.

### Without LLM reranking

For faster, cheaper normalization, disable the LLM reranking step:

```python
normalizer = OntologyMapping(
    ontology="hpo",
    reranking_model=None,  # pure embedding nearest-neighbor
)
```

**Best for:** Biomedical normalization, mapping to standardized ontologies, when you need ontology IDs.

**Install:** `pip install catchfly[medical]` (adds `pronto` + `numpy`)

---

## LLMCanonicalization

LLM groups synonyms and assigns canonical names. Map-reduce for large sets with hierarchical merge for cross-batch consolidation.

```python
from catchfly.normalization import LLMCanonicalization

normalizer = LLMCanonicalization(
    model="gpt-5.4-mini",
    max_values_per_prompt=200,  # single call below this
    batch_size=50,              # batch size for map-reduce
)

result = normalizer.normalize(
    values=["NYC", "New York", "NY", "Big Apple"],
    context_field="city",
)

print(result.mapping)     # {"NYC": "New York", "NY": "New York", ...}
print(result.clusters)    # {"New York": ["NYC", "NY", "Big Apple", ...]}
```

### Hierarchical merge

For large value sets (>200 unique values), LLMCanonicalization uses map-reduce: split into batches, canonicalize each, then merge. By default, a **hierarchical merge** step runs a second LLM pass to consolidate semantically similar groups across batches (e.g., "Wedding Rings" + "Rings & Bands" → "Rings").

```python
normalizer = LLMCanonicalization(
    model="gpt-5.4-mini",
    hierarchical_merge=True,        # enabled by default
    hierarchical_merge_rounds=1,    # number of consolidation rounds
)
```

Set `hierarchical_merge=False` to use only case-insensitive string matching (faster, less accurate).

### Schema-aware prompting

When `field_metadata` from `SchemaOptimizer` is provided, it enriches the LLM prompt with field descriptions, examples, and constraints:

```python
from catchfly.discovery import SchemaOptimizer

# Enrich schema
optimizer = SchemaOptimizer(model="gpt-5.4-mini", num_iterations=3)
enriched = optimizer.optimize(schema=my_schema, test_documents=docs)

# Normalize with schema context
result = normalizer.normalize(
    values=extracted_values,
    context_field="product_type",
    field_metadata=enriched.field_metadata["product_type"],
)
```

The pipeline passes `field_metadata` automatically when both `SchemaOptimizer` and `LLMCanonicalization` are configured.

**Best for:** General-purpose normalization, high quality, when you need rationale.

---

## EmbeddingClustering

Embed values → cluster → pick canonical name per cluster. No LLM needed during clustering.

```python
from catchfly.normalization import EmbeddingClustering

normalizer = EmbeddingClustering(
    embedding_model="text-embedding-3-small",
    clustering_algorithm="agglomerative",  # or "hdbscan"
    similarity_threshold=0.7,
    reduce_dimensions=True,    # UMAP for high-dim embeddings
)

result = normalizer.normalize(
    values=["NYC", "New York", "new york city", "LA", "Los Angeles"],
    context_field="city",
)

print(result.mapping)    # {"NYC": "New York", "new york city": "New York", ...}
print(result.explain("NYC"))  # Human-readable explanation
```

**Best for:** Fast normalization, large value sets, when LLM cost is a concern.

---

## KLLMeansClustering

k-means in embedding space with periodic LLM-generated textual centroids.

```python
from catchfly.normalization import KLLMeansClustering

normalizer = KLLMeansClustering(
    num_clusters=5,           # or None for auto-detection via silhouette
    embedding_model="text-embedding-3-small",
    summarization_model="gpt-5.4-mini",
    num_iterations=10,
    summarize_every=3,        # LLM summary every 3 iterations
)

result = normalizer.normalize(values=messy_values, context_field="medication")
```

**Best for:** Surface-form deduplication (e.g., brand name variants) where embeddings separate variants well. For semantic normalization, use LLMCanonicalization instead.

---

## Choosing a Strategy

| Criterion | OntologyMapping | LLMCanonicalization | EmbeddingClustering | KLLMeansClustering |
|---|---|---|---|---|
| **Use case** | Map to ontology | Group synonyms | Cluster variants | Cluster with LLM centroids |
| **LLM cost** | Medium (reranking) | Medium | Low (embedding only) | Medium-High |
| **Quality** | High (ontology-grounded) | High | Good | Good |
| **Ontology IDs** | Yes | No | No | No |
| **Schema context** | No | Yes | No | No |
| **Scale** | Any | Small-medium (<1k) | Any | Any |
| **Best domain** | Biomedical | General | General | Surface-form |
