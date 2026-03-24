# Changelog

All notable changes to catchfly are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

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
