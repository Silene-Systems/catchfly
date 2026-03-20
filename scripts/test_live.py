"""Catchfly live integration tests with real LLM providers.

Usage:
    # Fill in scripts/.env with API keys, then:
    uv run python scripts/test_live.py              # run all tests
    uv run python scripts/test_live.py discovery     # run only discovery tests
    uv run python scripts/test_live.py normalization # run only normalization tests

Available test suites:
    discovery     — SinglePassDiscovery + ThreeStageDiscovery
    optimizer     — SchemaOptimizer (PARSE-style enrichment)
    extraction    — LLMDirectExtraction (tool calling across providers)
    normalization — EmbeddingClustering + LLMCanonicalization
    kllmeans      — KLLMeansClustering (with and without schema seed)
    pipeline      — Full pipeline end-to-end
    registry      — SchemaRegistry round-trip
    all           — Everything (default)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------


def _load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_live")


# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------


@dataclass
class Provider:
    name: str
    model: str
    base_url: str | None
    api_key_env: str
    embedding_model: str | None = None

    @property
    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


OPENAI = Provider("OpenAI", "gpt-5.4-mini", None, "OPENAI_API_KEY", "text-embedding-3-small")
ANTHROPIC = Provider(
    "Anthropic", "claude-haiku-4-5-20251001", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"
)
MISTRAL = Provider(
    "Mistral", "mistral-small-latest", "https://api.mistral.ai/v1/", "MISTRAL_API_KEY"
)
ALL_PROVIDERS = [OPENAI, ANTHROPIC, MISTRAL]


def first_available(*providers: Provider) -> Provider | None:
    return next((p for p in providers if p.available), None)


# ---------------------------------------------------------------------------
# Test results tracking
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    passed: bool
    elapsed_s: float
    detail: str = ""


@dataclass
class TestSuite:
    name: str
    results: list[TestResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, elapsed: float, detail: str = "") -> None:
        self.results.append(TestResult(name, passed, elapsed, detail))
        icon = "PASS" if passed else "FAIL"
        logger.info("[%s] %s (%.1fs)%s", icon, name, elapsed, f" — {detail}" if detail else "")

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def timed(fn: Any) -> tuple[Any, float]:
    """Run fn() and return (result, elapsed_seconds)."""
    start = time.monotonic()
    try:
        result = fn()
        return result, time.monotonic() - start
    except Exception as e:
        return e, time.monotonic() - start


# ---------------------------------------------------------------------------
# TEST SUITES
# ---------------------------------------------------------------------------


def test_discovery(suite: TestSuite) -> None:
    """Test SinglePassDiscovery + ThreeStageDiscovery across providers."""
    from catchfly.demo import load_samples
    from catchfly.discovery.single_pass import SinglePassDiscovery
    from catchfly.discovery.three_stage import ThreeStageDiscovery

    docs = load_samples("product_reviews")[:3]

    for provider in ALL_PROVIDERS:
        if not provider.available:
            suite.add(f"SinglePass:{provider.name}", True, 0, "SKIPPED — no key")
            continue

        def run_sp(p: Provider = provider) -> Any:
            sp = SinglePassDiscovery(model=p.model, base_url=p.base_url, api_key=p.api_key)
            return sp.discover(docs, domain_hint="Electronics product reviews")

        result, elapsed = timed(run_sp)
        if isinstance(result, Exception):
            suite.add(f"SinglePass:{provider.name}", False, elapsed, str(result))
        else:
            n_fields = len(result.json_schema.get("properties", {}))
            suite.add(f"SinglePass:{provider.name}", n_fields > 0, elapsed, f"{n_fields} fields")

    # ThreeStageDiscovery (only with first available — expensive)
    p = first_available(*ALL_PROVIDERS)
    if p:
        docs_large = load_samples("product_reviews")

        def run_ts() -> Any:
            ts = ThreeStageDiscovery(
                model=p.model,
                base_url=p.base_url,
                api_key=p.api_key,
                stage1_samples=2,
                stage2_samples=3,
                stage3_samples=5,
            )
            return ts.discover(docs_large, domain_hint="Electronics reviews")

        result, elapsed = timed(run_ts)
        if isinstance(result, Exception):
            suite.add(f"ThreeStage:{p.name}", False, elapsed, str(result))
        else:
            n_fields = len(result.json_schema.get("properties", {}))
            report = result.field_metadata.get("_discovery_report", {})
            stages = report.get("stages_completed", "?")
            suite.add(
                f"ThreeStage:{p.name}",
                n_fields > 0,
                elapsed,
                f"{n_fields} fields, {stages} stages",
            )


def test_optimizer(suite: TestSuite) -> None:
    """Test SchemaOptimizer with real LLM."""
    from pydantic import BaseModel

    from catchfly.demo import load_samples
    from catchfly.discovery.optimizer import SchemaOptimizer

    p = first_available(*ALL_PROVIDERS)
    if not p:
        suite.add("Optimizer", True, 0, "SKIPPED — no keys")
        return

    class ReviewSchema(BaseModel):
        product_name: str
        rating: float
        price: float | None = None

    docs = load_samples("product_reviews")[:5]

    def run() -> Any:
        opt = SchemaOptimizer(
            model=p.model, base_url=p.base_url, api_key=p.api_key, num_iterations=2
        )
        return opt.optimize(ReviewSchema, docs)

    result, elapsed = timed(run)
    if isinstance(result, Exception):
        suite.add(f"Optimizer:{p.name}", False, elapsed, str(result))
    else:
        n_enriched = len(result.field_metadata)
        has_desc = any(
            "description" in v for v in result.field_metadata.values() if isinstance(v, dict)
        )
        suite.add(
            f"Optimizer:{p.name}",
            n_enriched > 0,
            elapsed,
            f"{n_enriched} fields enriched, has_desc={has_desc}",
        )


def test_extraction(suite: TestSuite) -> None:
    """Test LLMDirectExtraction with tool calling across providers."""
    from pydantic import BaseModel

    from catchfly.demo import load_samples
    from catchfly.extraction.llm_direct import LLMDirectExtraction

    class Review(BaseModel):
        product_name: str
        rating: float
        pros: list[str] = []

    docs = load_samples("product_reviews")[:2]

    for provider in ALL_PROVIDERS:
        if not provider.available:
            suite.add(f"Extraction:{provider.name}", True, 0, "SKIPPED")
            continue

        def run(p: Provider = provider) -> Any:
            ext = LLMDirectExtraction(
                model=p.model, base_url=p.base_url, api_key=p.api_key, max_retries=1
            )
            return ext.extract(Review, docs)

        result, elapsed = timed(run)
        if isinstance(result, Exception):
            suite.add(f"Extraction:{provider.name}", False, elapsed, str(result))
        else:
            suite.add(
                f"Extraction:{provider.name}",
                len(result.records) == 2,
                elapsed,
                f"{len(result.records)} records, {len(result.errors)} errors",
            )


def test_normalization(suite: TestSuite) -> None:
    """Test EmbeddingClustering + LLMCanonicalization."""
    p = first_available(OPENAI)  # needs embeddings
    if not p:
        suite.add("EmbeddingClustering", True, 0, "SKIPPED — no OPENAI_API_KEY")
        suite.add("LLMCanonicalization", True, 0, "SKIPPED")
        return

    # --- EmbeddingClustering ---
    from catchfly.normalization.embedding_cluster import EmbeddingClustering

    values = [
        "NYC",
        "New York",
        "new york city",
        "Los Angeles",
        "LA",
        "L.A.",
        "San Francisco",
        "SF",
    ]

    def run_ec() -> Any:
        ec = EmbeddingClustering(
            embedding_model="text-embedding-3-small",
            api_key=p.api_key,
            clustering_algorithm="agglomerative",
            similarity_threshold=0.7,
            reduce_dimensions=False,
        )
        return ec.normalize(values, context_field="city")

    result, elapsed = timed(run_ec)
    if isinstance(result, Exception):
        suite.add("EmbeddingClustering", False, elapsed, str(result))
    else:
        n_canonical = len(set(result.mapping.values()))
        mapping_str = ", ".join(f"{k}→{v}" for k, v in sorted(result.mapping.items()))
        suite.add(
            "EmbeddingClustering",
            n_canonical < len(values),
            elapsed,
            f"{n_canonical} canonical: {mapping_str}",
        )

    # --- LLMCanonicalization ---
    from catchfly.normalization.llm_canonical import LLMCanonicalization

    llm_p = first_available(*ALL_PROVIDERS)
    if not llm_p:
        suite.add("LLMCanonicalization", True, 0, "SKIPPED")
        return

    def run_lc() -> Any:
        lc = LLMCanonicalization(model=llm_p.model, base_url=llm_p.base_url, api_key=llm_p.api_key)
        return lc.normalize(values, context_field="city")

    result, elapsed = timed(run_lc)
    if isinstance(result, Exception):
        suite.add(f"LLMCanonicalization:{llm_p.name}", False, elapsed, str(result))
    else:
        n_canonical = len(set(result.mapping.values()))
        suite.add(
            f"LLMCanonicalization:{llm_p.name}",
            n_canonical < len(values),
            elapsed,
            f"{n_canonical} canonical",
        )


def test_kllmeans(suite: TestSuite) -> None:
    """Test KLLMeansClustering — cold start and schema-seeded."""
    p = first_available(OPENAI)
    if not p:
        suite.add("kLLMmeans", True, 0, "SKIPPED — no OPENAI_API_KEY")
        return

    from catchfly.normalization.kllmeans import KLLMeansClustering

    values = ["cat", "cats", "kitten", "dog", "dogs", "puppy", "fish", "goldfish"]

    # --- Cold start ---
    def run_cold() -> Any:
        kl = KLLMeansClustering(
            num_clusters=3,
            embedding_model="text-embedding-3-small",
            summarization_model=p.model,
            api_key=p.api_key,
            num_iterations=5,
            summarize_every=2,
        )
        return kl.normalize(values, context_field="animal")

    result, elapsed = timed(run_cold)
    if isinstance(result, Exception):
        suite.add("kLLMmeans:cold", False, elapsed, str(result))
    else:
        n_clusters = result.metadata.get("k", "?")
        cat_canonical = result.mapping.get("cat", "?")
        dog_canonical = result.mapping.get("dog", "?")
        suite.add(
            "kLLMmeans:cold",
            cat_canonical != dog_canonical,
            elapsed,
            f"k={n_clusters}, cat→{cat_canonical}, dog→{dog_canonical}",
        )

    # --- Schema-seeded ---
    def run_seeded() -> Any:
        kl = KLLMeansClustering(
            num_clusters=3,
            embedding_model="text-embedding-3-small",
            summarization_model=p.model,
            api_key=p.api_key,
            seed_from_schema=True,
            num_iterations=5,
            summarize_every=2,
        )
        metadata = {
            "description": "Type of pet animal",
            "examples": ["cat", "dog", "fish"],
            "synonyms": ["feline", "canine", "aquatic"],
        }
        return kl.normalize(values, context_field="animal", field_metadata=metadata)

    result, elapsed = timed(run_seeded)
    if isinstance(result, Exception):
        suite.add("kLLMmeans:seeded", False, elapsed, str(result))
    else:
        n_clusters = result.metadata.get("k", "?")
        suite.add("kLLMmeans:seeded", True, elapsed, f"k={n_clusters}, schema-seeded init OK")


def test_pipeline(suite: TestSuite) -> None:
    """Test full pipeline end-to-end with normalization."""
    p = first_available(OPENAI)
    if not p:
        suite.add("Pipeline:full", True, 0, "SKIPPED — no OPENAI_API_KEY")
        return

    from catchfly import Pipeline
    from catchfly.demo import load_samples

    docs = load_samples("support_tickets")[:3]

    def run() -> Any:
        pipeline = Pipeline.quick(model=p.model, base_url=p.base_url, api_key=p.api_key)
        return pipeline.run(
            docs,
            domain_hint="SaaS support tickets",
            normalize_fields=["category", "priority"],
        )

    result, elapsed = timed(run)
    if isinstance(result, Exception):
        suite.add(f"Pipeline:full:{p.name}", False, elapsed, str(result))
    else:
        n_records = len(result.records)
        n_norm = len(result.normalizations)
        cols = list(result.to_dataframe().columns) if n_records else []
        suite.add(
            f"Pipeline:full:{p.name}",
            n_records > 0,
            elapsed,
            f"{n_records} records, {n_norm} normalized fields, cols={cols[:5]}",
        )


def test_registry(suite: TestSuite) -> None:
    """Test SchemaRegistry (no LLM needed)."""
    import tempfile

    from catchfly._types import Schema
    from catchfly.schema.registry import SchemaRegistry

    def run() -> Any:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reg.json")
            reg = SchemaRegistry(persist_path=path)

            s1 = Schema(
                model=None,
                json_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                lineage=["test"],
            )
            s2 = Schema(
                model=None,
                json_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                },
                lineage=["test"],
            )

            v1 = reg.register(s1, "person")
            v2 = reg.register(s2, "person")

            diff = SchemaRegistry.diff(s1, s2)

            # Reload from disk
            reg2 = SchemaRegistry(persist_path=path)
            loaded = reg2.get("person")

            return {
                "v1": v1,
                "v2": v2,
                "diff_added": diff["added"],
                "persisted": loaded is not None,
                "n_schemas": len(reg2.list_schemas()),
            }

    result, elapsed = timed(run)
    if isinstance(result, Exception):
        suite.add("Registry", False, elapsed, str(result))
    else:
        suite.add(
            "Registry",
            result["persisted"] and result["n_schemas"] == 2,
            elapsed,
            f"{result['v1']}, {result['v2']}, diff added={result['diff_added']}, persisted={result['persisted']}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUITES = {
    "discovery": test_discovery,
    "optimizer": test_optimizer,
    "extraction": test_extraction,
    "normalization": test_normalization,
    "kllmeans": test_kllmeans,
    "pipeline": test_pipeline,
    "registry": test_registry,
}


def main() -> None:
    requested = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("=" * 70)
    print("Catchfly — Live Integration Tests")
    print("=" * 70)

    for p in ALL_PROVIDERS:
        status = f"set ({len(p.api_key)} chars)" if p.available else "NOT SET"
        print(f"  {p.name:12s} ({p.api_key_env}): {status}")
    print()

    if requested == "all":
        suites_to_run = list(SUITES.keys())
    elif requested in SUITES:
        suites_to_run = [requested]
    else:
        print(f"Unknown suite: '{requested}'. Available: {', '.join(SUITES.keys())}, all")
        sys.exit(1)

    all_results = TestSuite("all")

    for suite_name in suites_to_run:
        print(f"--- {suite_name.upper()} {'—' * (60 - len(suite_name))}")
        suite = TestSuite(suite_name)
        try:
            SUITES[suite_name](suite)
        except Exception as e:
            suite.add(f"{suite_name}:CRASH", False, 0, str(e))
            logger.error("Suite '%s' crashed: %s", suite_name, e, exc_info=True)
        all_results.results.extend(suite.results)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in all_results.results:
        icon = "PASS" if r.passed else "FAIL"
        print(f"  [{icon}] {r.name} ({r.elapsed_s:.1f}s)")
    print(f"\n  {all_results.passed} passed, {all_results.failed} failed")
    print(f"  Total: {sum(r.elapsed_s for r in all_results.results):.1f}s")

    sys.exit(0 if all_results.failed == 0 else 1)


if __name__ == "__main__":
    main()
