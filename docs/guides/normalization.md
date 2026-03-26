# Normalization

Catchfly offers multiple normalization strategies for cleaning up messy extracted values — from zero-cost dictionary lookup to ontology-grounded mapping.

## CascadeNormalization

Chain multiple strategies sequentially. Each step receives only the values unmapped by previous steps. Recommended for production.

```python
from catchfly.normalization import CascadeNormalization

# Default cascade: Dictionary → LLM → Ontology
cascade = CascadeNormalization.default(
    dictionary={"ALT": "Alanine aminotransferase", "AST": "Aspartate aminotransferase"},
    model="gpt-5.4-mini",
    ontology="hpo",
)

result = cascade.normalize(values=extracted_values, context_field="phenotype")
```

### Confidence-based routing

By default, CascadeNormalization uses identity checks (mapped = canonical differs from raw). With confidence thresholds, low-confidence mappings flow to the next step instead of being accepted:

```python
cascade = CascadeNormalization.default(
    dictionary=known_terms,
    model="gpt-5.4-mini",
    ontology="hpo",
    use_confidence=True,  # Dictionary=1.0, LLM=0.80, Ontology=0.90
)
```

Or set per-step thresholds manually:

```python
cascade = CascadeNormalization(
    steps=[dict_step, llm_step, ontology_step],
    confidence_thresholds=[1.0, 0.80, 0.90],
)
```

### Self-learning

After a normalization run, teach the cascade to remember high-confidence mappings:

```python
result = cascade.normalize(values=extracted_values, context_field="phenotype")

# Prepend learned mappings as a dictionary step for next run
cascade.learn(result)  # known mappings now resolve instantly ($0)
```

For persistent learning across runs, use `LearnedDictionaryCache`:

```python
from catchfly.normalization import LearnedDictionaryCache

cache = LearnedDictionaryCache(path="./cache/normalizations.json", min_confidence=0.80)

# After normalization
cache.save({"phenotype": result})

# On next run — load cached mappings
cached_dict = cache.load_dictionary("phenotype")
if cached_dict:
    cascade.steps.insert(0, cached_dict)
```

### Custom cascade

```python
from catchfly.normalization import (
    CascadeNormalization,
    DictionaryNormalization,
    LLMCanonicalization,
    OntologyMapping,
)

cascade = CascadeNormalization(steps=[
    DictionaryNormalization(mapping=known_terms, case_insensitive=True),
    LLMCanonicalization(model="gpt-5.4-mini"),
    OntologyMapping(ontology="path/to/ontology.obo"),
])

result = cascade.normalize(values=messy_values, context_field="diagnosis")
# result.metadata has per-step stats (strategy name, mapped count, remaining count)
```

**Best for:** Production pipelines that need maximum coverage at controlled cost.

---

## DictionaryNormalization

Zero-cost normalization using a static dictionary. Exact or case-insensitive matching.

```python
from catchfly.normalization import DictionaryNormalization

normalizer = DictionaryNormalization(
    mapping={"NYC": "New York", "LA": "Los Angeles", "SF": "San Francisco"},
    case_insensitive=True,
)

result = normalizer.normalize(values=["nyc", "NYC", "Chicago"], context_field="city")
# {"nyc": "New York", "NYC": "New York", "Chicago": "Chicago"}
```

**Best for:** Known abbreviations, acronyms, domain-specific mappings. First step in a cascade.

---

## OntologyMapping

Map values to standardized ontology terms (HPO, ICD-10, custom). Uses embedding nearest-neighbor search with optional LLM reranking.

```python
from catchfly.normalization import OntologyMapping

normalizer = OntologyMapping(
    ontology="hpo",                        # or path to .obo, .csv, .json
    embedding_model="text-embedding-3-small",
    reranking_model="gpt-5.4-mini",        # None to skip LLM reranking
    top_k=5,                               # candidates per value
    augment_queries=True,                  # RAG: generate alternative phrasings
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

### Local biomedical embeddings (SapBERT)

For zero-cost biomedical embeddings that outperform OpenAI models:

```python
from catchfly.providers import SentenceTransformerEmbeddingClient

# SapBERT: 0.802 Acc@1 on BC5CDR with reranking (beats text-embedding-3-large)
embed_client = SentenceTransformerEmbeddingClient()  # default: SapBERT

normalizer = OntologyMapping(
    ontology="hpo",
    embedding_client=embed_client,  # inject local client
    reranking_model="gpt-5.4-mini",
)
```

Requires: `pip install catchfly[embeddings]` (adds `sentence-transformers`)

### RAG-augmented queries

When `augment_queries=True`, the LLM generates alternative phrasings per value before embedding search. This improves recall by +10-20pp on biomedical benchmarks:

```python
normalizer = OntologyMapping(
    ontology="hpo",
    augment_queries=True,
    augmentation_skip_threshold=0.95,  # skip for high-confidence matches
    augmentation_n_phrasings=5,
)
```

### Without LLM reranking

For faster, cheaper normalization, disable the LLM reranking step:

```python
normalizer = OntologyMapping(
    ontology="hpo",
    reranking_model=None,  # pure embedding nearest-neighbor
)
```

**Best for:** Biomedical normalization, mapping to standardized ontologies, when you need ontology IDs.

**Install:** `pip install catchfly[medical]` (adds `pronto` + `numpy`). For local embeddings: `pip install catchfly[embeddings,medical]`

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

## CompositeNormalization

Route different fields to different normalization strategies:

```python
from catchfly import Pipeline
from catchfly.normalization import (
    CompositeNormalization,
    LLMCanonicalization,
    OntologyMapping,
    DictionaryNormalization,
)

# Option 1: Via Pipeline dict syntax
pipeline = Pipeline(
    discovery=...,
    extraction=...,
    normalization={
        "phenotype": OntologyMapping(ontology="hpo"),
        "medication": DictionaryNormalization(mapping=drug_dict),
        "category": LLMCanonicalization(model="gpt-5.4-mini"),
    },
)

# Option 2: Explicit CompositeNormalization with default fallback
composite = CompositeNormalization(
    field_strategies={
        "phenotype": OntologyMapping(ontology="hpo"),
        "gene": DictionaryNormalization(mapping=gene_dict),
    },
    default=LLMCanonicalization(model="gpt-5.4-mini"),
)
```

**Best for:** Multi-domain extraction where different fields need different normalization approaches.

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

| Criterion | CascadeNormalization | OntologyMapping | LLMCanonicalization | DictionaryNormalization | EmbeddingClustering | KLLMeansClustering |
|---|---|---|---|---|---|---|
| **Use case** | Production pipeline | Map to ontology | Group synonyms | Known mappings | Cluster variants | Surface-form dedup |
| **LLM cost** | Varies | Medium (reranking) | Medium | Zero | Low (embedding) | Medium-High |
| **Quality** | Highest | High (ontology-grounded) | High | Exact only | Good | Good |
| **Ontology IDs** | If ontology step | Yes | No | No | No | No |
| **Schema context** | If LLM step | No | Yes | No | No | No |
| **Scale** | Any | Any | Small-medium (<1k) | Any | Any | Any |
| **Best domain** | General | Biomedical | General | Any | General | Surface-form |
