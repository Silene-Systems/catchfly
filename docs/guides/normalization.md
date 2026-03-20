# Normalization

Catchfly offers three normalization strategies for cleaning up messy extracted values.

## EmbeddingClustering

Embed values → cluster → pick canonical name per cluster. No LLM needed during clustering.

```python
from catchfly.normalization.embedding_cluster import EmbeddingClustering

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

## LLMCanonicalization

LLM groups synonyms and assigns canonical names. Map-reduce for large sets.

```python
from catchfly.normalization.llm_canonical import LLMCanonicalization

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

For **>200 unique values**, automatically splits into batches (map), canonicalizes each, and merges (reduce) with cross-batch duplicate detection.

**Best for:** High quality, small-medium value sets, when you need rationale.

## KLLMeansClustering

k-means in embedding space with periodic LLM-generated textual centroids. The **core novel contribution** of Catchfly.

```python
from catchfly.normalization.kllmeans import KLLMeansClustering

normalizer = KLLMeansClustering(
    num_clusters=5,           # or None for auto-detection via silhouette
    embedding_model="text-embedding-3-small",
    summarization_model="gpt-5.4-mini",
    num_iterations=10,
    summarize_every=3,        # LLM summary every 3 iterations
)

result = normalizer.normalize(values=messy_values, context_field="medication")
```

### Schema-Seeded Warmstart

The key innovation: use enriched field descriptions from `SchemaOptimizer` as initial centroids:

```python
from catchfly.discovery.optimizer import SchemaOptimizer
from catchfly.normalization.kllmeans import KLLMeansClustering

# Step 1: Enrich schema
optimizer = SchemaOptimizer(model="gpt-5.4-mini", num_iterations=3)
enriched = optimizer.optimize(schema=my_schema, test_documents=docs)

# Step 2: Normalize with schema-seeded centroids
normalizer = KLLMeansClustering(
    num_clusters=5,
    seed_from_schema=True,
)
result = normalizer.normalize(
    values=extracted_values,
    context_field="medication",
    field_metadata=enriched.field_metadata["medication"],
)
```

This bridges schema optimization and normalization — the enriched descriptions guide clustering from the start.

**Best for:** Large value sets, interpretable clusters, when schema metadata is available.

## Choosing a Strategy

| Criterion | EmbeddingClustering | LLMCanonicalization | KLLMeansClustering |
|---|---|---|---|
| **LLM cost** | Low (embedding only) | Medium | Medium-High |
| **Quality** | Good | High | High |
| **Interpretability** | Medium | High (rationale) | High (textual centroids) |
| **Scale** | Any | Small-medium (<1k) | Any |
| **Requires k** | No (HDBSCAN) | No | Yes (or auto) |
| **Schema integration** | No | No | Yes (seed_from_schema) |
