# Catchfly — Product Requirements Document

**Version:** 2.3
**Date:** March 23, 2026
**Author:** Adrian Michalski

---

## 1. Vision

Catchfly is a Python library for AI/ML engineers and researchers that automates the pipeline of **schema discovery → structured extraction → normalization** from unstructured text at scale. It provides interchangeable strategies at each stage, enabling users to go from raw documents (PDFs, emails, support tickets, research papers, legal filings) to clean, normalized, structured data with minimal manual effort.

**Tagline:** *Catch the structured data.*

The name references *Silene* (Catchfly) — a genus with many species, reflecting the library's design of many interchangeable strategies under a unified interface.

---

## 2. Problem Statement

Extracting structured data from unstructured text with LLMs is a solved problem at the unit level (one document, one schema, one extraction). But the end-to-end workflow — discovering what to extract, extracting it consistently across thousands of documents, and normalizing the messy output — remains fragmented. Researchers currently stitch together ad-hoc scripts combining Instructor/Pydantic, custom prompts, and manual post-processing.

Specific pain points:

- **Schema discovery is manual.** Users must inspect documents and hand-craft Pydantic models before they can start extracting.
- **Extraction output is inconsistent.** The same concept appears as "ALT", "ALAT", "Alanine aminotransferase" across documents — or "New York", "NYC", "NY" in another domain.
- **Normalization is an afterthought.** No library integrates normalization as a first-class step in the extraction pipeline.
- **No unified pipeline exists.** EVAPORATE does schema+extraction, PARSE does schema optimization — but nothing connects discovery, extraction, and normalization into one coherent pipeline.

---

## 3. Target Users

**Primary:** AI/ML engineers and NLP researchers who need to extract structured data from document corpora (10–100k documents). They are comfortable with Python, Pydantic, and LLM APIs.

**Secondary:** Domain researchers (biomedical, legal, financial, e-commerce) who work with publication corpora, case files, product catalogs, or support tickets and need structured datasets for analysis. They may have limited ML background but understand their domain deeply.

**Example use cases:**
- Biomedical researcher extracting patient data from clinical case reports across thousands of publications
- Legal analyst structuring contract clauses from a corpus of agreements
- E-commerce team normalizing product attributes ("colour", "color", "clr") across supplier catalogs
- Support team categorizing and normalizing ticket labels across years of historical data
- Financial analyst extracting key metrics from earnings call transcripts

---

## 4. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                             catchfly                                  │
│                                                                       │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Discovery  │→│ Extraction  │→│ Field Select  │→│Normalization  │  │
│  │            │  │            │  │              │  │              │  │
│  │ Strategies:│  │ Strategies:│  │ Strategies:  │  │ Strategies:  │  │
│  │ •SinglePass│  │ •LLMDirect │  │ •Statistical │  │ •OntologyMap │  │
│  │ •ThreeStage│  │ •CodeSynth │  │ •LLM-based   │  │ •LLMCanonicl │  │
│  │ •PARSE-opt │  │            │  │              │  │ •Embedding   │  │
│  │            │  │            │  │              │  │ •kLLMmeans   │  │
│  └────────────┘  └────────────┘  └──────────────┘  └──────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐│
│  │                        Shared Layer                               ││
│  │     LLM Client · Embedding Models · Schema Registry · Telemetry  ││
│  └───────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────────┘
```

Each stage is independent — users can use discovery alone, extraction alone, or normalization alone. The pipeline connects them seamlessly when used together. Field Selection is optional: users can bypass it with explicit `normalize_fields` lists.

---

## 5. Module Specifications

### 5.1 Schema Discovery

**Purpose:** Given a corpus of documents (or a sample), automatically discover or propose a structured schema for extraction.

#### Strategy A: SinglePassDiscovery

- Input: N sample documents (default: 5) + optional domain hint (e.g., "product listings from electronics catalogs")
- Process: Single LLM call with representative samples → proposes Pydantic model
- Output: JSON Schema / Pydantic model class
- Config: `model`, `num_samples`, `domain_hint`, `temperature`
- Sync: `discovery.discover(documents, domain_hint)`
- Async: `await discovery.adiscover(documents, domain_hint)`
- Inspired by: LlamaExtract `infer_schema()`, Simon Willison's LLM schemas

#### Strategy B: ThreeStageDiscovery

- **Stage 1 — Initial schema:** LLM generates schema from domain description or 2-3 exemplar documents
- **Stage 2 — Refinement:** Schema is validated against a small curated sample (5-10 docs). LLM identifies missing fields, redundant fields, type mismatches. Human can intervene.
- **Stage 3 — Expansion:** Schema is tested against a larger uncurated sample (50-100 docs). New recurring fields are proposed; rare fields are flagged for removal.
- Output: Refined JSON Schema / Pydantic model + discovery report (field coverage stats, confidence scores)
- Config: `model`, `stage1_samples`, `stage2_samples`, `stage3_samples`, `human_review` (bool)
- Sync: `discovery.discover(documents, domain_hint)`
- Async: `await discovery.adiscover(documents, domain_hint)`
- Inspired by: SCHEMA-MINER 3-stage workflow

#### Schema Optimizer (PARSE-style)

- Input: Any schema (from Strategy A, B, or user-provided) + a small extraction test set (10-20 docs)
- Process: Iteratively enriches field descriptions, adds examples, constraints, and synonyms based on extraction error analysis. Runs N optimization iterations.
- Output: Enriched schema with rich field descriptions, examples, synonyms, and constraints
- Config: `model`, `num_iterations` (default: 5), `test_docs`
- Sync: `optimizer.optimize(schema, test_documents)`
- Async: `await optimizer.aoptimize(schema, test_documents)`
- Inspired by: PARSE ARCHITECT component
- **Key design decision:** Enriched descriptions are stored as `field_metadata` and exported alongside the schema. This metadata is consumed by the normalization module as context — providing LLM-based normalizers with domain knowledge about each field (what values to expect, how to group them, what canonical forms exist).

### 5.2 Extraction

**Purpose:** Given a schema and documents, extract structured data.

#### Strategy: LLMDirectExtraction

- Input: Schema (Pydantic model or JSON Schema) + documents
- Process: Each document (or chunk) is processed by LLM with structured output (tool use / JSON mode)
- Output: List of extracted records as Pydantic model instances
- Features:
  - Automatic chunking for long documents (configurable chunk size, overlap)
  - Batch processing with progress tracking (tqdm)
  - Retry logic with validation feedback (à la Instructor)
  - Provenance tracking: source document, chunk index, confidence, character offsets in source text
  - Document-level error handling (`on_error="skip"` / `"raise"` / `"collect"`)
  - Checkpoint/resume for long-running jobs (state persisted to `.catchfly_state.json`)
- Config: `model`, `chunk_size`, `max_retries`, `batch_size`, `parallel_workers`, `on_error`
- Sync: `extractor.extract(schema, documents)`
- Async: `await extractor.aextract(schema, documents)`
- Built on: Instructor / native structured output APIs

#### Strategy: CodeSynthExtraction (future / v2)

- LLM generates Python extraction functions from sample docs
- Functions are applied at scale without LLM inference per document
- Inspired by: EVAPORATE-Code+
- **Not in scope for v1** — noted for roadmap

### 5.3 Normalization

**Purpose:** Given extracted records with inconsistent values, normalize them to canonical forms.

Normalization is the hardest stage — different value types need different approaches. Surface-form variants ("HP" / "Hewlett-Packard") are handled well by LLM reasoning. Semantic mappings ("wedding band" → "Rings/Bands") require domain knowledge. Domain-specific terms ("ALT" → "Alanine aminotransferase") need ontology grounding. Strategies are ordered by recommended usage.

> **Lessons from benchmarking (v0.5):** Embedding-based clustering (k-means in vector space) works well for surface-form variants but poorly for semantic normalization. LLM-based reasoning outperforms embedding clustering on semantic tasks. Schema context (field descriptions, examples) improves LLM normalization when provided as prompt context. See `paper/FIRST_RESULTS.md` for full experimental evidence.

> **Industry research findings (v1.1):** Analysis of normalization approaches across major libraries (LangChain, LlamaIndex, Docling, Unstructured, MedCAT, scispaCy) and recent papers (EDC EMNLP 2024, RAGnorm 2024, xMEN JAMIA 2024, Brinkmann et al. 2024, PMC 2025 multistage biomedical normalization) confirms:
> 1. No major LLM framework has built-in normalization cascades — this is catchfly's niche.
> 2. Embedding clustering alone achieves ~40% Omega-F on biomedical normalization (PMC 2025); not viable as standalone step.
> 3. Dominant 2024-2025 pattern: dictionary/embedding **retrieval** → LLM verification/reranking (not embedding clustering).
> 4. Catchfly's OntologyMapping (embedding retrieval + LLM reranking) matches state-of-the-art architecture (RAGnorm, xMEN).
> 5. ~~Specialized biomedical embeddings (PubMedBERT, BioBERT) underperform general models (OpenAI text-embedding-3-large) for normalization (PMC 2025 benchmarking: 20-52% vs 69.4%).~~ **Corrected (v1.1, 2026-03-26):** Our own benchmark on BC5CDR Disease (4363 mentions, 13320 CTD MEDIC terms) shows **SapBERT outperforms OpenAI text-embedding-3-large** on entity normalization retrieval: Acc@1 0.787 vs 0.772, Acc@5 0.898 vs 0.863. With LLM reranking (gpt-4.1-mini): 0.802 vs 0.793. SapBERT is the recommended default for biomedical use — zero API cost, best accuracy. Full results: `benchmarks/` (branch `medbench`).
> 6. LLM-only normalization (GPT-4 zero-shot) achieves F1=91% on product attribute normalization (Brinkmann 2024) — embedding pre-filtering adds negligible value at small-to-medium scale.
>
> These findings informed the decision to exclude embedding clustering from the default CascadeNormalization pipeline and to keep it only as a standalone strategy for cost-sensitive offline use cases.

#### Strategy 1: OntologyMapping (P0) ✅

- Input: Unique values of a field + ontology reference (HPO, UMLS, ICD-10, SNOMED, or custom)
- Process:
  1. Embed extracted values with domain-appropriate model (SapBERT for biomedical, general embeddings otherwise)
  2. Nearest-neighbor search in ontology embedding index → top-K candidates per value
  3. LLM reranking: LLM selects best candidate given value + context + candidates
- Output: Mapping `{raw_value: (ontology_id, canonical_name, confidence)}`
- Config: `ontology`, `embedding_model`, `top_k`, `reranking_model`, `confidence_threshold`, `augment_queries` (bool, default False)
- **RAG augmentation (v1.1.1):** When `augment_queries=True`, LLM generates alternative phrasings per value before search. Improves recall +10-20pp on biomedical benchmarks (LLMAEL/GRF, ACL 2025). High-confidence values skip augmentation (`augmentation_skip_threshold=0.95`).
- **Homonym disambiguation (v1.1.1):** OntologyIndex detects entries with identical names but different IDs and appends disambiguating context (synonyms, ID) to index texts. Zero extra LLM calls.
- Ontology loaders included for: UMLS (via QuickUMLS or API), HPO (via OBO file), ICD-10, custom CSV/JSON
- Sync: `normalizer.normalize(values, context_field)`
- Async: `await normalizer.anormalize(values, context_field)`
- Inspired by: CLINES normalization, AutoBioKG, Generative Relevance Feedback
- **Critical for biomedical use case** (Silene Systems) — maps extracted values to standardized ontology terms

#### Strategy 2: LLMCanonicalization (recommended default)

- Input: Unique values of a field (batched if > threshold) + optional `field_metadata` from SchemaOptimizer
- Process: LLM groups synonyms and assigns canonical names. For large sets (>200 unique values), uses map-reduce: split into batches → LLM canonicalizes each batch → hierarchical merge consolidates cross-batch groups.
- **Schema-aware prompting:** When `field_metadata` is provided (from SchemaOptimizer), field descriptions, examples, and constraints are included in the system prompt. This gives the LLM domain context for better grouping decisions. Benchmarks show +1.5% NMI improvement on semantic normalization tasks with schema context.
- **Hierarchical merge:** ✅ Second LLM pass on canonical names to merge semantically similar groups across batches (e.g., "Wedding Rings" + "Rings & Bands"). Configurable via `hierarchical_merge=True` (default) and `hierarchical_merge_rounds`.
- Output: Mapping `{raw_value: canonical_value}` + grouping rationale
- Config: `model`, `batch_size`, `max_values_per_prompt`, `field_metadata`
- Sync: `normalizer.normalize(values, context_field)`
- Async: `await normalizer.anormalize(values, context_field)`
- Inspired by: EDC Canonicalize phase, LLM map-reduce pattern

#### Strategy 3: EmbeddingClustering

- Input: Unique values of a field + embedding model
- Process: Embed all values → optional dimensionality reduction (UMAP) → cluster (HDBSCAN or agglomerative) → select/generate canonical name per cluster
- Output: Mapping `{raw_value: canonical_value}` + cluster metadata
- Config: `embedding_model`, `similarity_threshold`, `clustering_algorithm`, `min_cluster_size`, `reduce_dimensions` (bool, default: True for >256d embeddings)
- Canonical name selection: most frequent value in cluster, or LLM-generated from cluster members
- Sync: `normalizer.normalize(values, context_field)`
- Async: `await normalizer.anormalize(values, context_field)`
- Inspired by: SemHash, standard entity deduplication
- **Best for:** Large-scale surface-form deduplication where LLM calls are too expensive. Fast and deterministic.

#### Strategy 4: KLLMeansClustering

- Input: Unique values of a field + optional field descriptions for centroid initialization
- Process: k-LLMmeans algorithm — k-means in embedding space with periodic LLM-generated textual summary centroids. Optionally seeds initial centroids from schema field metadata.
- Output: Clusters with canonical names (centroid summaries) + mapping `{raw_value: cluster_label}`
- Config: `num_clusters` (or auto-detect via silhouette), `embedding_model`, `summarization_model`, `num_iterations`, `summarize_every`, `seed_from_schema` (bool)
- Sync: `normalizer.normalize(values, context_field)`
- Async: `await normalizer.anormalize(values, context_field)`
- Inspired by: k-LLMmeans (Diaz-Rodriguez 2026)
- **Note:** Benchmarking (v0.5) showed schema-seeded initialization provides marginal improvement over k-means++ on IE normalization tasks. Effective for surface-form clustering (NMI>0.98 on Brand) but limited on semantic normalization. For most use cases, LLMCanonicalization is recommended instead. **Deprecation candidate for v2** — LLMCanonicalization matches surface-form performance (0.980 vs 0.983 NMI, statistically insignificant) while being simpler and more capable on semantic tasks.

#### Strategy 5: CascadeNormalization (recommended for production) ✅

- Input: Unique values of a field
- Process: Chain multiple strategies sequentially. Each step receives only the values unmapped by previous steps. Mappings merge across steps.
- Default cascade: `DictionaryNormalization → LLMCanonicalization → OntologyMapping`
- Config: `steps` (list of NormalizationStrategy instances), `confidence_thresholds` (optional per-step routing), or use `CascadeNormalization.default(dictionary=..., model=..., ontology=..., use_confidence=True)`
- Output: Merged mapping `{raw_value: canonical_value}` + per-step statistics in metadata
- **Confidence-based routing (v1.1.1):** When `confidence_thresholds` is set, values are considered "resolved" only when mapped with confidence ≥ threshold. Low-confidence mappings flow to subsequent steps. All strategies now emit `per_value` confidence metadata. Defaults: Dictionary=1.0, LLM=0.80, Ontology=0.90.
- **Self-learning (v1.1.1):** `cascade.learn(result)` prepends/merges high-confidence mappings as a dictionary step. `LearnedDictionaryCache` persists mappings to JSON for reuse across runs. Expected savings: -80-95% normalization cost on repeated corpora.
- Why not embedding clustering in cascade: Our benchmarks (v0.5) + industry consensus (EDC EMNLP 2024, RAGnorm 2024, xMEN JAMIA 2024, PMC 2025) show LLM reasoning subsumes embedding clustering benefits. Dictionary pre-filter handles trivial surface-form variants at zero cost. Embedding clustering remains available as standalone strategy for offline/cost-sensitive use cases.
- Inspired by: MedCAT (dictionary detection → embedding disambiguation), xMEN (TF-IDF + SapBERT → cross-encoder reranking), RAGnorm (embedding retrieval + LLM generation)

```python
from catchfly.normalization import CascadeNormalization

# Quick setup
cascade = CascadeNormalization.default(
    dictionary={"ALT": "Alanine aminotransferase", "AST": "Aspartate aminotransferase"},
    model="gpt-5.4-mini",
    ontology="hpo",
)
result = await cascade.anormalize(values, context_field="phenotype")

# Custom cascade
cascade = CascadeNormalization(steps=[
    DictionaryNormalization(mapping=known_terms, case_insensitive=True),
    LLMCanonicalization(model="gpt-5.4-mini"),
    OntologyMapping(ontology="path/to/ontology.obo"),
])
```

### 5.4 Field Selection

**Purpose:** Given a discovered schema and extracted records, automatically determine which fields should be normalized. This is the missing decision layer between schema discovery (which identifies fields) and normalization (which acts on them).

**Problem statement:** When schema discovery is used, the user doesn't know field names in advance. The current `normalize_fields` parameter requires either explicit field names (catch-22 with discovery) or `"all"` (too aggressive — normalizes free-text fields like `description`, `product_name`, `summary`). There is no intelligent middle ground.

**Design rationale:** Field selection is a distinct concern from both discovery and normalization. Following the library's "compose small units" and "consistency above all" principles, it gets its own Protocol — identical in shape to `DiscoveryStrategy`, `ExtractionStrategy`, and `NormalizationStrategy`. This avoids overloading the discovery prompt (discovery discovers schema — single responsibility) or bloating Pipeline with inline heuristics.

#### Protocol: FieldSelector

```python
@runtime_checkable
class FieldSelector(Protocol):
    """Decides which discovered fields should be normalized."""

    def select(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]: ...

    async def aselect(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]: ...
```

- Input: discovered `Schema` (with `json_schema` + `field_metadata`) + extracted records
- Output: list of field names to normalize
- Follows same async-first, sync-friendly pattern as all other strategies
- Sync: `selector.select(schema, records)`
- Async: `await selector.aselect(schema, records)`

#### Strategy A: StatisticalFieldSelector

Zero LLM cost — analyzes actual extracted values using statistical heuristics:

1. **Type filter:** Only consider string fields and array-of-string fields (from `json_schema.properties`)
2. **Cardinality ratio:** `unique_values / total_records` — fields with ratio < 0.5 are repetitive → good normalization candidates
3. **Value length:** Average string length < 50 chars → likely categorical, not free-text
4. **Name exclusion patterns:** Skip fields matching configurable patterns: `description`, `text`, `content`, `summary`, `note`, `comment`, `id`, `url`, `email`, `name`, `title` (covers common free-text and identifier fields)
5. **Minimum unique values:** Skip fields with fewer than 3 unique values (nothing meaningful to normalize)

- Config: `max_cardinality_ratio` (default: 0.5), `max_avg_length` (default: 50), `exclude_patterns` (list[str], configurable), `min_unique_values` (default: 3)
- No dependencies beyond core — pure Python analysis of in-memory values
- Suitable as the default for `Pipeline.quick()` (zero additional cost)

#### Strategy B: LLMFieldSelector

Asks the LLM to classify each string field as normalizable or not, given the schema and a sample of extracted values. More accurate on ambiguous fields but costs one additional LLM call.

- Input: schema + sampled values per field
- Process: Single LLM call with field names, types, descriptions, and example values → LLM returns list of fields that would benefit from normalization
- Config: `model`, `base_url`, `api_key`, `num_sample_values` (default: 20 per field)
- Cost: ~$0.001 (one short LLM call)

#### Pipeline Integration

```python
class Pipeline:
    def __init__(
        self,
        discovery=None,
        extraction=None,
        normalization=None,
        field_selector=None,  # NEW — FieldSelector instance
        *,
        verbose=False,
    ): ...
```

**Behavior in `run()`/`arun()`:**

| `normalize_fields` | `field_selector` | Behavior |
|---|---|---|
| `["pros", "brand"]` | any | Explicit override — selector ignored |
| `"all"` | any | All string fields (existing behavior, backwards-compatible) |
| `None` | `LLMFieldSelector()` | Selector analyzes records and decides |
| `None` | `None` | Skip normalization (backwards-compatible) |

`Pipeline.quick()` sets `field_selector=LLMFieldSelector()` by default, enabling zero-config normalization for the discovery workflow. The LLM call costs ~$0.001 and produces more accurate results than statistical heuristics — particularly for array-of-string fields where cardinality metrics are misleading.

**Progressive Disclosure:**

- **Level 0:** `Pipeline.quick(model="gpt-5.4-mini")` → auto-selects fields via `LLMFieldSelector`
- **Level 1:** `Pipeline(..., field_selector=LLMFieldSelector(model="gpt-5.4-mini"))` → swap selector
- **Level 2:** `Pipeline(..., field_selector=StatisticalFieldSelector(max_cardinality_ratio=0.3, exclude_patterns=["author"]))` → configure selector
- **Level 3:** Implement `FieldSelector` Protocol with custom domain logic
- **Override:** `pipeline.run(docs, normalize_fields=["pros"])` → bypass selector entirely

```python
# Zero-config — discovery + extraction + auto field selection + normalization
pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(docs, domain_hint="Electronics product reviews")
# Pipeline discovers schema, extracts records, auto-selects categorical fields, normalizes them

# Explicit override still works
results = pipeline.run(docs, normalize_fields=["pros", "brand"])
```

---

## 6. Shared Layer

### 6.1 LLM Client

Thin Protocol-based abstraction over OpenAI-compatible chat completion APIs. Catchfly does **not** bundle or depend on a routing layer like litellm. Instead, it talks to any OpenAI-compatible endpoint — the user points it at their provider of choice.

- **Default backend:** `openai` Python SDK, which works with OpenAI, Azure OpenAI, and any OpenAI-compatible server (Ollama, vLLM, LMStudio, llama.cpp server, etc.)
- **Configuration:** Users provide a `base_url` + `api_key` (or environment variables). For local models: `base_url="http://localhost:11434/v1"` (Ollama default).
- **Features:** retry logic, rate limiting, cost tracking, token counting
- **All strategies accept `model` parameter** as string — the model name is passed directly to the endpoint.
- **Custom backends:** Users can implement the `LLMClient` Protocol to integrate any provider not compatible with the OpenAI API format.

```python
from catchfly.providers import LLMClient

# OpenAI (default)
client = LLMClient(model="gpt-5.4")

# Local Ollama
client = LLMClient(model="qwen3.5", base_url="http://localhost:11434/v1")

# Custom endpoint
client = LLMClient(model="my-model", base_url="https://my-api.com/v1", api_key="...")

# Anthropic via OpenAI-compatible proxy, or implement custom backend
```

### 6.2 Embedding Model Abstraction ✅

- `EmbeddingClient` Protocol: `aembed(texts) -> list[list[float]]` + `embed(texts) -> list[list[float]]`
- `OpenAIEmbeddingClient` ✅ — OpenAI-compatible API (OpenAI, Ollama, vLLM). Default: `text-embedding-3-small`. In-memory cache with LRU eviction, batching.
- `SentenceTransformerEmbeddingClient` ✅ (v1.1.1) — Local inference via sentence-transformers. Default: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (768-dim, biomedical entity linking). Lazy model loading, GPU auto-detect (CUDA/MPS/CPU), `asyncio.to_thread()` for non-blocking async, in-memory cache. Zero API cost.
- **Benchmark validation (2026-03-26):** SapBERT beats OpenAI text-embedding-3-large by 1.5pp Acc@1 on BC5CDR disease normalization at zero cost. SapBERT + LLM rerank = 0.802 Acc@1 (best pipeline).
- Any object implementing the `EmbeddingClient` protocol works (duck-typed)

### 6.3 Schema Registry

- Stores discovered/refined schemas with versioning
- Tracks schema lineage (which discovery strategy, which optimizer, which iteration)
- Exports to: Pydantic model, JSON Schema, TypedDict
- Imports from: Pydantic model, JSON Schema, dict

### 6.4 Cost & Telemetry

- Tracks LLM calls, tokens consumed, latency per stage
- Reports total pipeline cost estimate
- **Cost limits:** `max_cost_usd` parameter on pipeline and per-stage — pipeline halts gracefully when budget is exceeded
- **Cost estimation:** `pipeline.estimate_cost(documents)` returns approximate cost before execution
- Exportable as JSON/CSV for analysis

---

## 7. User-Facing API Design

### 7.1 Quick Start (zero-config)

`Pipeline.quick()` creates a pipeline with sensible defaults: **SinglePassDiscovery** + **LLMDirectExtraction** + **LLMFieldSelector** + **LLMCanonicalization** — the most effective combination that covers the full pipeline using a single LLM provider. The field selector automatically identifies which discovered fields are categorical and worth normalizing — no `normalize_fields` parameter needed.

```python
from catchfly import Pipeline

# Sensible defaults — one model, one line
pipeline = Pipeline.quick(model="gpt-5.4-mini")

results = pipeline.run(
    documents=["path/to/docs/*.txt"],
    domain_hint="Product reviews from an electronics retailer",
)
# Pipeline discovers schema, extracts records, auto-selects categorical fields, normalizes them

# Inspect results
results.to_dataframe()           # → pandas DataFrame
results.to_csv("output.csv")     # → CSV export
results.schema                   # → discovered Pydantic model
results.report                   # → cost, time, coverage stats
```

### 7.2 Built-in Demo (first 5 minutes)

Catchfly ships with a small demo dataset so new users can run a full pipeline immediately without sourcing their own data.

```python
from catchfly import Pipeline
from catchfly.demo import load_samples

# 10 sample documents (product reviews, support tickets, short case reports)
docs = load_samples("product_reviews")  # or "support_tickets", "case_reports"

pipeline = Pipeline.quick(model="gpt-5.4-mini")
results = pipeline.run(documents=docs, domain_hint="Electronics product reviews")

print(results.schema)            # discovered schema
print(results.to_dataframe())    # extracted + normalized data
print(results.report.cost_usd)   # e.g. ~$0.03 for demo
```

Available demo datasets (~10 documents each, bundled in the package):
- `"product_reviews"` — electronics retailer reviews with inconsistent attribute names
- `"support_tickets"` — SaaS support tickets with messy category labels
- `"case_reports"` — short biomedical case report excerpts

### 7.3 Full Pipeline (explicit strategies)

```python
from catchfly import Pipeline
from catchfly.discovery import ThreeStageDiscovery
from catchfly.extraction import LLMDirectExtraction
from catchfly.normalization import LLMCanonicalization

pipeline = Pipeline(
    discovery=ThreeStageDiscovery(model="gpt-5.4"),
    extraction=LLMDirectExtraction(model="gpt-5.4"),
    normalization=LLMCanonicalization(model="gpt-5.4-mini"),
)

# Sync
results = pipeline.run(
    documents=["path/to/docs/*.txt"],
    domain_hint="Support tickets from a SaaS product",
    normalize_fields=["category", "priority", "product_area"],
    max_cost_usd=50.0,
)

# Async
results = await pipeline.arun(
    documents=["path/to/docs/*.txt"],
    domain_hint="Support tickets from a SaaS product",
    normalize_fields=["category", "priority", "product_area"],
)

# results.schema         → discovered Pydantic model
# results.records        → list of extracted records
# results.normalizations → field → {raw: canonical} mappings
# results.errors         → list of per-document errors (if on_error="collect")
# results.report         → cost, time, coverage stats
```

### 7.4 Local Models (Ollama / vLLM)

```python
from catchfly import Pipeline

# Point at local Ollama instance — no API keys needed
pipeline = Pipeline.quick(
    model="qwen3.5",
    base_url="http://localhost:11434/v1",
)

results = pipeline.run(
    documents=my_docs,
    domain_hint="Invoices from European suppliers",
)
```

### 7.5 Modular Usage (each stage independent)

```python
# Discovery only
from catchfly.discovery import SinglePassDiscovery

discovery = SinglePassDiscovery(model="gpt-5.4")
schema = discovery.discover(documents=sample_docs, domain_hint="...")

# Extraction only (bring your own schema)
from catchfly.extraction import LLMDirectExtraction

extractor = LLMDirectExtraction(model="gpt-5.4")
records = extractor.extract(schema=MyPydanticModel, documents=all_docs)

# Async extraction
records = await extractor.aextract(schema=MyPydanticModel, documents=all_docs)

# Normalization only (bring your own data)
from catchfly.normalization import LLMCanonicalization

normalizer = LLMCanonicalization(model="gpt-5.4-mini")
mapping = normalizer.normalize(values=unique_values, context_field="product_category")
```

### 7.6 Schema Optimizer

```python
from catchfly.discovery import SchemaOptimizer

optimizer = SchemaOptimizer(model="gpt-5.4", num_iterations=5)
enriched_schema = optimizer.optimize(
    schema=base_schema,
    test_documents=sample_docs[:20],
)
# enriched_schema has rich field descriptions usable as context for LLM normalization
```

---

## 8. Production Features

### 8.1 Checkpoint & Resume

Long-running pipelines (1000+ documents) persist progress to a state file. If the process is interrupted, calling `pipeline.run()` with the same output directory resumes from the last checkpoint.

```python
results = pipeline.run(
    documents=large_corpus,
    checkpoint_dir="./catchfly_state/",  # enables checkpoint/resume
)
```

### 8.2 Error Handling

Document-level error handling prevents one bad document from killing the entire pipeline.

```python
extractor = LLMDirectExtraction(model="gpt-5.4", on_error="collect")
results = extractor.extract(schema=MyModel, documents=all_docs)

results.records   # successfully extracted records
results.errors    # list of (document, error) pairs for failed documents
```

Options: `on_error="raise"` (default — fail fast), `"skip"` (silent skip), `"collect"` (continue and collect errors).

### 8.3 Cost Control

```python
# Estimate before running
estimate = pipeline.estimate_cost(documents=my_docs)
print(estimate)  # EstimatedCost(total_usd=12.50, breakdown={discovery: 0.30, extraction: 11.80, normalization: 0.40})

# Set hard limit
results = pipeline.run(documents=my_docs, max_cost_usd=20.0)
# Pipeline halts gracefully when budget is exceeded, returning partial results
```

### 8.4 Progress Tracking

Built-in tqdm progress bars for all batch operations. Configurable via `verbose` parameter.

```python
pipeline = Pipeline.quick(model="gpt-5.4-mini", verbose=True)   # tqdm progress bars
pipeline = Pipeline.quick(model="gpt-5.4-mini", verbose=False)  # silent
```

### 8.5 Export Formats

```python
results.to_dataframe()                    # → pandas DataFrame
results.to_csv("output.csv")             # → CSV
results.to_parquet("output.parquet")     # → Parquet
results.to_json("output.json")           # → JSON
results.normalizations["category"].explain("NYC")  # → normalization audit trail
```

---

## 9. Technical Requirements

### 9.1 Dependencies

**Core (lightweight — ~5 MB install):**
- Python 3.10+
- `pydantic >= 2.0, < 3.0`
- `httpx >= 0.25`

**Optional extras:**

```toml
[project.optional-dependencies]
openai = ["openai>=1.50"]
instructor = ["instructor>=1.5"]
embeddings = ["sentence-transformers>=3.0"]
clustering = ["scikit-learn>=1.3", "numpy>=1.24", "umap-learn>=0.5"]
medical = ["pronto>=2.5"]
pdf = ["pymupdf>=1.24"]
docling = ["docling>=2.0"]
export = ["pandas>=2.0", "pyarrow>=15.0"]
benchmark = ["datasets>=2.14", "tabulate>=0.9"]
all = ["catchfly[openai,instructor,embeddings,clustering,medical,pdf,export]"]
```

Install profiles:
- `pip install catchfly` — core only, minimal footprint
- `pip install catchfly[openai]` — add OpenAI SDK for API-based LLM/embeddings
- `pip install catchfly[clustering]` — add scikit-learn for embedding-based normalization
- `pip install catchfly[medical]` — add ontology loaders (HPO, UMLS)
- `pip install catchfly[all]` — everything

**Key decisions:**
- **No litellm.** Catchfly uses the `openai` SDK to talk to any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, Azure, etc.). Users who need multi-provider routing can use litellm externally.
- **No sentence-transformers in core.** PyTorch is 5+ GB. Local embeddings are optional; API-based embeddings (OpenAI, Ollama) work out of the box.
- **HDBSCAN from scikit-learn.** Since scikit-learn 1.3, `sklearn.cluster.HDBSCAN` is built-in — no separate `hdbscan` package needed.

### 9.2 Package Structure

```
src/catchfly/
├── __init__.py
├── py.typed                    # PEP 561 marker for type checkers
├── _types.py                   # Shared type aliases and TypeVars
├── exceptions.py               # CatchflyError hierarchy
├── pipeline.py                 # Pipeline orchestrator
├── demo/
│   ├── __init__.py             # load_samples() function
│   ├── product_reviews.json    # ~10 sample product reviews
│   ├── support_tickets.json    # ~10 sample support tickets
│   └── case_reports.json       # ~10 sample case report excerpts
├── discovery/
│   ├── __init__.py
│   ├── base.py                 # DiscoveryStrategy protocol
│   ├── single_pass.py          # SinglePassDiscovery
│   ├── three_stage.py          # ThreeStageDiscovery
│   └── optimizer.py            # SchemaOptimizer (PARSE-style)
├── extraction/
│   ├── __init__.py
│   ├── base.py                 # ExtractionStrategy protocol
│   ├── llm_direct.py           # LLMDirectExtraction
│   └── chunking.py             # Document chunking strategies
├── normalization/
│   ├── __init__.py
│   ├── base.py                 # NormalizationStrategy protocol
│   ├── cascade.py              # CascadeNormalization (confidence routing, learn())
│   ├── dictionary.py           # DictionaryNormalization (per-value confidence)
│   ├── embedding_cluster.py    # EmbeddingClustering
│   ├── learned_cache.py        # LearnedDictionaryCache (self-learning)
│   ├── llm_canonical.py        # LLMCanonicalization (per-value confidence)
│   ├── ontology_mapping.py     # OntologyMapping (RAG augmentation, homonyms)
│   └── kllmeans.py             # KLLMeansClustering
├── selection/
│   ├── __init__.py             # FieldSelector Protocol re-export
│   ├── base.py                 # FieldSelector Protocol
│   ├── statistical.py          # StatisticalFieldSelector
│   └── llm.py                  # LLMFieldSelector
├── providers/
│   ├── __init__.py
│   ├── llm.py                  # LLMClient protocol + OpenAI-compatible impl
│   └── embeddings.py           # Embedding provider abstraction
├── schema/
│   ├── __init__.py
│   ├── registry.py             # Schema versioning & storage
│   └── converters.py           # JSON Schema ↔ Pydantic ↔ dict
└── telemetry/
    ├── __init__.py
    └── tracker.py              # Cost & usage tracking

benchmarks/                        # Outside src/ — not shipped in package
├── datasets/                      # Bank77, CLINC, GoEmo, MASSIVE loaders
├── metrics/                       # NMI, ARI, ACC, Wilcoxon, Bonferroni
├── harness/                       # PRD benchmark runner (seeded vs cold-start)
├── sici/                          # SICI paper experiments (strategies, convergence, etc.)
├── results/                       # JSON persistence, Markdown/LaTeX reporting
└── test_*.py                      # 20 benchmark tests

scripts/
    run_benchmark.py               # CLI for live benchmark runs

paper/
    sici_paper.md                  # EMNLP publication template
```

### 9.3 Design Principles

- **Strategy pattern everywhere.** Each stage defines a Protocol (abstract interface) with domain-specific method names (`discover`, `extract`, `select`, `normalize`). New strategies are added by implementing the protocol. No inheritance chains.
- **Pydantic-native.** Schemas are Pydantic models throughout. Input validation, serialization, and LLM structured output all use the same models. Strategy configuration uses Pydantic `BaseModel` for validation.
- **Async-first, sync-friendly.** Core logic is implemented as async (`adiscover`, `aextract`, `anormalize`). Sync wrappers (`discover`, `extract`, `normalize`) detect whether an event loop is already running (e.g., Jupyter notebooks) and handle it gracefully — via `nest_asyncio` patching or `anyio.from_thread.run()` — instead of bare `asyncio.run()`. Pipeline supports both `run()` and `arun()`.
- **Lightweight core, opt-in heavyweights.** Core install is `pydantic` + `httpx` only. Heavy dependencies (PyTorch, scikit-learn, pandas) are imported lazily and only when the user's chosen strategy requires them.
- **Stateless strategies, stateful pipeline.** Each strategy is a pure function (input → output). The pipeline object holds state (schema registry, telemetry, intermediate results, checkpoints).
- **LLM-agnostic via OpenAI compatibility.** Catchfly talks to any OpenAI-compatible endpoint. Users bring their own LLM — cloud API, local Ollama, self-hosted vLLM.
- **Explicit error handling.** Custom exception hierarchy (`CatchflyError` → `DiscoveryError`, `ExtractionError`, `NormalizationError`, `ProviderError`). Document-level error collection, not silent failures.
- **Structured logging.** All modules use `logging.getLogger(__name__)` with sensible levels: `DEBUG` for individual LLM calls, `INFO` for stage progress, `WARNING` for retries/fallbacks, `ERROR` for document failures. No `print()` statements. Users can integrate with `structlog` or any standard logging handler.

### 9.4 Core Types

Defined in `_types.py` — these are the domain types used across all protocols and strategies:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel

# --- Document ---
@dataclass
class Document:
    """A single document for processing."""
    content: str                           # the text content
    id: str | None = None                  # optional identifier
    source: str | Path | None = None       # file path, URL, or other provenance
    metadata: dict[str, Any] = field(default_factory=dict)

# --- Schema ---
@dataclass
class Schema:
    """A discovered or user-provided schema."""
    model: type[BaseModel]                 # the Pydantic model class
    json_schema: dict[str, Any]            # JSON Schema representation
    field_metadata: dict[str, Any] = field(default_factory=dict)  # enriched descriptions from optimizer
    lineage: list[str] = field(default_factory=list)              # e.g. ["SinglePassDiscovery", "SchemaOptimizer:iter3"]

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# --- Results ---
@dataclass
class ExtractionResult(Generic[SchemaT]):
    records: list[SchemaT]
    errors: list[tuple[Document, Exception]]
    provenance: list[RecordProvenance]

@dataclass
class RecordProvenance:
    source_document: str
    chunk_index: int | None
    char_start: int | None
    char_end: int | None
    confidence: float | None

@dataclass
class NormalizationResult:
    mapping: dict[str, str]                # {raw_value: canonical_value}
    clusters: dict[str, list[str]] | None  # {canonical: [raw_values]}
    metadata: dict[str, Any]               # strategy-specific (rationale, scores, etc.)

    def explain(self, value: str) -> str:
        """Return human-readable explanation of why value was mapped to its canonical form."""
        ...
```

### 9.5 Protocols

Each stage defines a Protocol with typed, domain-specific signatures using the core types above:

```python
from typing import Protocol

class DiscoveryStrategy(Protocol):
    def discover(self, documents: list[Document], **kwargs) -> Schema: ...
    async def adiscover(self, documents: list[Document], **kwargs) -> Schema: ...

class ExtractionStrategy(Protocol):
    def extract(self, schema: type[SchemaT], documents: list[Document], **kwargs) -> ExtractionResult[SchemaT]: ...
    async def aextract(self, schema: type[SchemaT], documents: list[Document], **kwargs) -> ExtractionResult[SchemaT]: ...

class FieldSelector(Protocol):
    def select(self, schema: Schema, records: list[Any], **kwargs) -> list[str]: ...
    async def aselect(self, schema: Schema, records: list[Any], **kwargs) -> list[str]: ...

class NormalizationStrategy(Protocol):
    def normalize(self, values: list[str], context_field: str, **kwargs) -> NormalizationResult: ...
    async def anormalize(self, values: list[str], context_field: str, **kwargs) -> NormalizationResult: ...
```

---

## 10. Development Infrastructure

### 10.1 Testing Strategy

```
tests/
├── conftest.py                  # Shared fixtures, MockLLMClient, MockEmbedder
├── unit/
│   ├── test_schema_converters.py
│   ├── test_chunking.py
│   ├── test_types.py
│   ├── test_embedding_cluster.py
│   ├── test_llm_canonical.py
│   └── test_kllmeans.py
├── integration/
│   ├── test_discovery_pipeline.py
│   ├── test_extraction_pipeline.py
│   └── test_full_pipeline.py
├── fixtures/
│   ├── sample_documents/        # Small test documents
│   ├── schemas/                 # Example schemas
│   └── llm_responses/           # Recorded LLM responses (cassettes)
└── benchmarks/                  # Optional performance tests
```

**Mocking LLM calls:** A `MockLLMClient` implementing the `LLMClient` Protocol returns pre-recorded responses from JSON fixture files. This is the primary testing strategy — unit tests never hit real APIs.

```python
@pytest.fixture
def mock_llm():
    return MockLLMClient(responses_dir="tests/fixtures/llm_responses/")
```

**Test markers:**
- `@pytest.mark.llm` — requires real LLM API (skipped by default, run with `pytest -m llm`)
- `@pytest.mark.slow` — long-running tests (benchmarks, large datasets)

**Property-based testing:** Hypothesis for schema converters (roundtrip: Pydantic → JSON Schema → Pydantic) and normalization invariants (every input value maps to exactly one canonical value).

**Snapshot testing:** `syrupy` for comparing discovered schemas against reference outputs — catches regressions in discovery strategies.

### 10.2 CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  lint:    # ruff check + ruff format --check
  typecheck:  # mypy --strict (with pydantic plugin)
  test:
    matrix:
      python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - pip install -e ".[openai,clustering,export]"
      - pytest --cov=catchfly --cov-report=xml --cov-fail-under=70 -m "not llm and not slow"
  integration:  # weekly, with real LLM API keys from secrets
    steps:
      - pytest -m llm --timeout=300
```

### 10.3 Dev Tooling

Configured in `pyproject.toml`:

- **Ruff** — linting + formatting (replaces flake8, isort, black)
- **mypy** — strict mode with `pydantic.mypy` plugin
- **pytest** — with `pytest-asyncio`, `pytest-cov`, `syrupy`
- **pre-commit** — runs ruff + mypy on staged files

### 10.4 Contributing

A `CONTRIBUTING.md` in the repository root covers:
- How to set up the dev environment (`uv sync` or `pip install -e ".[all]"` + dev group)
- How to add a new strategy (implement the Protocol, add tests with MockLLMClient, add snapshot fixture)
- Code style expectations (ruff handles formatting, mypy strict must pass)
- PR process (CI must be green, one approval required)

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "syrupy>=4.0",
    "hypothesis>=6.100",
    "mypy>=1.10",
    "ruff>=0.5",
    "pre-commit>=3.8",
]
docs = [
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.26",
]
```

---

## 11. Implementation Milestones

| Version | Focus | Key deliverables | Status |
|---------|-------|-------------------|--------|
| v0.1 | MVP | Core types, LLM client, SinglePassDiscovery, LLMDirectExtraction, EmbeddingClustering, Pipeline.quick(), demo datasets | ✅ |
| v0.3 | Schema optimization | SchemaOptimizer (PARSE-style), LLMCanonicalization with map-reduce, cost tracking, checkpoint/resume, export | ✅ |
| v0.5 | Full discovery + benchmarks | ThreeStageDiscovery, KLLMeansClustering, SchemaRegistry, benchmark framework. SICI experiments → negative results (see `paper/FIRST_RESULTS.md`) | ✅ |
| v0.8 | Ontology mapping | OntologyMapping (HPO + custom), schema-aware LLM normalization (`field_metadata`), hierarchical merge | ✅ |
| v0.8.1 | Production hardening | Re-exports, error handling, magic numbers → params, bounded cache, input validation, 234 tests/83% coverage, 16/16 live e2e | ✅ |
| v1.0 | Production release | All success criteria met, 269 tests, PyPI publication, mkdocs documentation | ✅ |
| v1.1 | CascadeNormalization + chunking | CascadeNormalization (Dictionary → LLM → Ontology), SentenceChunking (chonkie), cookbook notebooks | ✅ |
| v1.1 | Field Selection | `FieldSelector` Protocol, `LLMFieldSelector` (default in `Pipeline.quick()`), `StatisticalFieldSelector`, pipeline integration, notebooks on RareArena | ✅ |
| v1.1.1 | Biomedical Normalization SOTA | `SentenceTransformerEmbeddingClient` (SapBERT default), confidence-based cascade routing, RAG-augmented normalization (`augment_queries`), self-learning dictionary cache (`LearnedDictionaryCache`), homonym disambiguation in OntologyIndex, `NormalizationResult.to_dictionary()`, `CascadeNormalization.learn()`. 396 tests. | ✅ |

---

## 12. Benchmark Datasets

The following datasets will be used for testing and validation:

| Dataset | Domain | Size | Tests | Source | Priority |
|---------|--------|------|-------|--------|----------|
| CaseReportBench | Medical (IEM) | ~100 case reports | Discovery + Extraction | PMC 2025 | P0 |
| MedMentions | Biomedical | 4,392 abstracts | OntologyMapping (UMLS) | PubMed | P0 |
| CORAL | Oncology | 40 reports | Full pipeline | CLINES paper | P1 |
| WDC-PAVE | E-commerce (products) | ~4.5k values | LLM normalization (per-attribute) | Brinkmann 2024 | P1 |
| SWDE | Web (8 domains) | 124k pages | Discovery + Extraction | Hao 2011 | P2 |
| Bank77 | Banking intents | ~13k utterances | Text clustering (supplementary) | Casanueva 2020 | P3 |
| CLINC | Dialog intents | ~23k utterances | Text clustering (supplementary) | Larson 2019 | P3 |

---

## 13. Success Criteria

The library is considered ready for release when:

1. **All four normalization strategies produce correct output** on at least 2 datasets each. ✅ *Verified via `scripts/test_live.py` — all 4 strategies (EmbeddingClustering, LLMCanonicalization, OntologyMapping, KLLMeansClustering) produce correct output on city/animal/phenotype datasets. 16/16 PASS.*
2. **End-to-end pipeline runs without errors** on at least one medical and one non-medical dataset. ✅ *Pipeline runs on product_reviews (non-medical), support_tickets (non-medical), and case_reports (medical) — 0 errors on all three.*
3. **OntologyMapping maps biomedical terms** to UMLS/HPO with ≥80% accuracy on MedMentions or CORAL. ✅ *OntologyMapping maps 5/5 (100%) phenotype terms to correct HPO entries (seizures→Seizure, high temperature→Fever, headache→Headache, feeling sick→Nausea, low muscle tone→Hypotonia). Full MedMentions benchmark deferred to post-v1.*
4. **Documentation covers** installation, quickstart, each strategy's usage, and at least 2 worked examples from different domains. ✅ *mkdocs site: installation, quickstart, 4 guides (discovery, extraction, normalization, pipeline), 6 API reference pages.*
5. **Package installable** via `pip install catchfly` with core dependencies only; optional extras via `pip install catchfly[clustering]`, `pip install catchfly[medical]`, `pip install catchfly[all]`. ✅ *`uv build` produces `catchfly-0.8.1-py3-none-any.whl`. Core deps: pydantic + httpx only.*
6. **Test coverage** ≥ 70% for core modules. ✅ *83% overall (234 tests). Providers at 35% by design — unit tests use mocks, real API tested via `scripts/test_live.py`.*
7. **Both sync and async APIs work** for all strategies and the pipeline orchestrator. ✅ *All strategies implement both `method()` and `amethod()`. Pipeline has `run()` and `arun()`. Verified in unit and live tests.*

---

## 13.3 Benchmark Framework & SICI Experiments (v0.5, archived)

Benchmark framework in `benchmarks/` (not shipped): metrics (NMI/ARI/ACC/Wilcoxon), dataset loaders (Bank77, CLINC, GoEmo, MASSIVE), harness, 20 tests. SICI experiments (schema-seeded k-LLMmeans initialization) concluded with **negative results** — LLM reasoning outperforms embedding clustering; schema context is valuable as prompt context, not embedding centroids. Full results: `paper/FIRST_RESULTS.md`. Code preserved on `benchmarks` branch.

---

## 14. Out of Scope for v1

- **CodeSynthExtraction** (EVAPORATE-Code+ style) — deferred to v2
- **Streaming/online normalization** — incremental normalization for incoming data deferred
- **GUI/web interface** — Python API only
- **Fine-tuning** — all strategies use inference-only LLMs, no training
- **Multi-modal extraction** (images, tables from PDFs) — text-only in v1, multi-modal in v2
- **Built-in PDF parsing** — v1 expects pre-extracted text; optional integration via `pip install catchfly[pdf]`
- **Multi-provider routing** — no built-in litellm; users who need it can use litellm externally as their OpenAI-compatible proxy
- **CLI interface** — Python API only in v1; CLI deferred to v1.1+

---

## 15. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM API costs during benchmarking | High | Use small models (gpt-5.4-mini, claude-haiku-4-5) for development; local models (Ollama) for iteration; full models only for final benchmark runs. Cost limits (`max_cost_usd`) prevent runaway spending. |
| Schema-seeded warmstart shows no improvement | **Materialized** | Benchmarks (v0.5) confirmed negligible delta. Pivoted to schema-aware LLM normalization — schema context is passed as prompt context to LLMCanonicalization instead of embedding centroids. |
| Ontology loaders (UMLS) complex to integrate | Medium | Start with HPO (simple OBO file); UMLS via QuickUMLS which handles complexity |
| Pydantic schema generation from LLM is unreliable | Medium | Validate generated schemas; fallback to JSON Schema dict if Pydantic codegen fails. SchemaOptimizer iteratively improves reliability. |
| OpenAI SDK compatibility across providers | Medium | Test against Ollama, vLLM, and OpenAI. Abstract provider-specific quirks in `LLMClient` implementation. |
| Dependency conflicts with user's environment | Medium | Keep core deps minimal (pydantic + httpx). Heavy deps are optional. Test against Python 3.10-3.13. |

---

## 16. Future Roadmap (v2+)

- CodeSynthExtraction (EVAPORATE-Code+ pattern)
- Multi-modal extraction (tables, figures from PDFs via vision models)
- Human-in-the-loop UI for schema refinement (Stage 2 of ThreeStageDiscovery)
- Fine-tuned small models for extraction (distillation from large model extractions)
- Integration with knowledge graph construction (EDC-style triple generation)
- Evaluation framework: built-in benchmarking against ground-truth datasets
- CLI interface (`catchfly discover --docs ./data/*.txt --model gpt-5.4-mini`)
- KLLMeansClustering deprecation (v2) — superseded by LLMCanonicalization on both surface-form and semantic tasks
- Confidence calibration for extraction and normalization scores
- Schema-driven feedback loop: normalization results feed back to SchemaOptimizer for iterative improvement
