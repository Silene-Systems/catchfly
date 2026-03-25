# Changelog

All notable changes to catchfly are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — v1.1

### Added
- **Pluggable chunking strategies** — `ChunkingStrategy` protocol with `FixedSizeChunking` (built-in) and chonkie-backed `TokenChunking`, `SentenceChunking`, `RecursiveChunking`, `SemanticChunking`. Install via `pip install catchfly[chunking]`.
- **`CascadeNormalization`** — chain normalization strategies sequentially (Dictionary → LLM → Ontology). Each step receives only unmapped values from previous steps. `CascadeNormalization.default()` factory for recommended cascade.
- **Field Selection** — `FieldSelector` protocol with `LLMFieldSelector` (LLM-based, ~$0.001) and `StatisticalFieldSelector` (zero-cost heuristics). `Pipeline.quick()` now includes `LLMFieldSelector` by default — auto-normalizes without `normalize_fields`.
- **Cookbook** — 5 Jupyter notebooks: Quick Start, Rare Disease, Product Catalog, Custom Schema, Local Models.
- `chunking` and `semantic-chunking` optional dependencies in `pyproject.toml`.
- `LLMDirectExtraction.chunking_strategy` parameter for pluggable chunking.
- `Pipeline.__init__()` accepts `field_selector` parameter.

### Changed
- `Pipeline.quick()` now uses `LLMFieldSelector` by default — zero-config normalization for discovery workflows.
- Documentation updated: new Field Selection guide, chunking strategies in extraction guide, CascadeNormalization/DictionaryNormalization/CompositeNormalization in normalization guide, updated pipeline guide with callbacks/globs/verbose.

## [1.0.0] — 2026-03-24

### Added
- **`DictionaryNormalization`** — new zero-cost normalization strategy using static dictionary lookup with exact or case-insensitive matching.
- **`CompositeNormalization`** — route different fields to different normalization strategies. Pipeline now accepts `normalization={"field": strategy}` dict syntax.
- **`normalize_fields="all"`** — auto-detect string/array-of-string fields from schema for normalization.
- **`on_schema_ready` callback** — optional callback in `Pipeline.arun()`/`run()` invoked after discovery, before extraction, allowing schema inspection or modification.
- **Document glob loader** — `Pipeline.arun()`/`run()` now accepts `list[str]` glob patterns (e.g. `["data/*.txt"]`) that auto-resolve to Documents. New `catchfly.loaders` module with `load_glob()` and `resolve_documents()`.
- **`verbose` parameter** — `Pipeline.__init__()` and `Pipeline.quick()` accept `verbose=True` for tqdm progress bars during normalization.
- **`UsageReport.cost_usd`** — property alias for `total_cost_usd` (PRD compatibility).

### Fixed
- **UsageTracker now connected to strategies** (critical bug) — `usage_callback` injection into `OpenAICompatibleClient`, wired through all 7 strategies via Pipeline. `max_cost_usd` budget enforcement and `result.report` cost tracking now work correctly.
- **`RecordProvenance.confidence` populated** — retry-based heuristic (`1.0` on first attempt, decreasing by `0.3` per retry). Was previously always `None`.

### Changed
- Version bumped to 1.0.0. Development status upgraded to Beta.
- All PRD §13 success criteria met. 269 tests, ruff clean.

## [0.8.1] — 2026-03-24

### Fixed
- `Pipeline.quick()` now uses `LLMCanonicalization` (the recommended default) instead of `EmbeddingClustering`, matching PRD specification.
- Silent `except Exception` blocks across discovery, normalization, and schema modules now log warnings with context (document ID, cluster index, exception info).
- `assert` statements in `OntologyIndex` cache methods replaced with proper `raise ValueError` (safe under `python -O`).
- `SchemaRegistry._load_from_disk()` catches specific exceptions (`JSONDecodeError`, `KeyError`, `TypeError`) instead of bare `Exception`.

### Added
- Public re-exports in all subpackage `__init__.py` files — `from catchfly.discovery import ThreeStageDiscovery` now works as documented.
- `on_error` parameter on `Pipeline.quick()` — forwarded to `LLMDirectExtraction`.
- Configurable parameters replacing hardcoded magic numbers:
  - `SinglePassDiscovery.max_doc_chars` (default: 3000)
  - `ThreeStageDiscovery.max_doc_chars` (default: 3000)
  - `SchemaOptimizer.max_docs_per_iteration` (default: 10)
  - `SchemaOptimizer.low_coverage_threshold` (default: 0.8)
  - `SchemaOptimizer.max_doc_chars` (default: 3000)
  - `KLLMeansClustering.max_members_in_prompt` (default: 50)
- Bounded embedding cache in `OpenAIEmbeddingClient` — `max_cache_size` parameter (default: 10,000) with FIFO eviction.
- Input validation in `CSVSource` and `JSONSource` ontology loaders — missing columns/keys now raise `ValueError` with clear messages.

### Removed
- Unused `verbose` parameter from `Pipeline.__init__()` and `Pipeline.quick()` (was dead code — never referenced in pipeline logic).
- Unused `embedding_model` parameter from `Pipeline.quick()` (not needed by `LLMCanonicalization`).

## [0.8.0] — 2026-03-23

### Added
- `normalization/ontology_mapping.py` — `OntologyMapping` strategy: embed + nearest-neighbor + LLM reranking against ontology terms.
- `ontology/` package — `OntologyEntry`, `OntologyIndex` (embedding search + disk cache), `HPOSource`, `CSVSource`, `JSONSource`.
- `LLMCanonicalization` — `field_metadata` support for schema-aware prompting; hierarchical merge (LLM-based cross-batch consolidation).
- `KLLMeansClustering` — removed SICI (schema-seeded initialization) after negative benchmark results; k-means++ only.

## [0.5.0] — 2026-03

### Added
- `discovery/three_stage.py` — `ThreeStageDiscovery` with 3-stage progressive refinement.
- `normalization/kllmeans.py` — `KLLMeansClustering` with k-means++ and LLM textual centroids.
- `schema/registry.py` — schema versioning, lineage tracking, and persistence.
- `benchmarks/` framework — metrics (NMI/ARI/ACC/Wilcoxon), dataset loaders (Bank77, CLINC, GoEmo, MASSIVE, WDC-PAVE), harness, SICI experiments.

## [0.3.0] — 2026-03

### Added
- `discovery/optimizer.py` — `SchemaOptimizer` (PARSE-style iterative enrichment).
- `normalization/llm_canonical.py` — `LLMCanonicalization` with map-reduce for large value sets.
- `telemetry/tracker.py` — cost tracking, `estimate_cost()`, `max_cost_usd` budget limits.
- Checkpoint/resume support in `Pipeline`.
- Export: `to_dataframe()`, `to_csv()`, `to_parquet()`, `to_json()`.
- `on_error` modes (`"raise"`, `"skip"`, `"collect"`) in `LLMDirectExtraction`.

## [0.1.0] — 2026-02

### Added
- Initial release — `Pipeline.quick()` works end-to-end.
- `_types.py`, `exceptions.py`, `py.typed` — core types and exception hierarchy.
- `providers/llm.py` — `LLMClient` Protocol + `OpenAICompatibleClient`.
- `providers/embeddings.py` — API-based embeddings with caching.
- `schema/converters.py` — JSON Schema <-> Pydantic model conversion.
- `discovery/single_pass.py` — `SinglePassDiscovery`.
- `extraction/llm_direct.py` — `LLMDirectExtraction` with chunking and retries.
- `normalization/embedding_cluster.py` — `EmbeddingClustering` (HDBSCAN/agglomerative).
- `pipeline.py` — Pipeline orchestrator with `quick()`, `run()`, `arun()`.
- `demo/` — bundled sample datasets (product reviews, support tickets, case reports).
