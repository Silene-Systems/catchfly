"""Live integration test with real LLM providers.

Usage:
    # Fill in scripts/.env with your API keys, then:
    uv run python scripts/test_live.py

Tests the full pipeline (discovery → extraction → normalization)
against OpenAI, Anthropic, and Mistral APIs.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _load_env() -> None:
    """Load .env file from the same directory as this script."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Don't override existing env vars
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_live")


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str | None
    api_key_env: str
    embedding_model: str | None = None


PROVIDERS = [
    ProviderConfig(
        name="OpenAI",
        model="gpt-5.4-mini",
        base_url=None,  # default
        api_key_env="OPENAI_API_KEY",
        embedding_model="text-embedding-3-small",
    ),
    ProviderConfig(
        name="Anthropic",
        model="claude-haiku-4-5-20251001",
        base_url="https://api.anthropic.com/v1/",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    ProviderConfig(
        name="Mistral",
        model="mistral-small-latest",
        base_url="https://api.mistral.ai/v1/",
        api_key_env="MISTRAL_API_KEY",
    ),
]


def test_discovery_and_extraction(config: ProviderConfig) -> bool:
    """Test schema discovery + extraction with a given provider."""
    from catchfly import Pipeline
    from catchfly.demo import load_samples

    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        logger.warning("[%s] Skipped — %s not set", config.name, config.api_key_env)
        return False

    docs = load_samples("product_reviews")[:2]

    logger.info("[%s] Testing discovery + extraction with model=%s", config.name, config.model)
    start = time.monotonic()

    try:
        pipeline = Pipeline.quick(
            model=config.model,
            base_url=config.base_url,
            api_key=api_key,
        )
        results = pipeline.run(
            documents=docs,
            domain_hint="Electronics product reviews with ratings and prices",
        )

        elapsed = time.monotonic() - start

        # Verify discovery
        assert results.schema is not None, "Schema is None — discovery failed"
        props = results.schema.json_schema.get("properties", {})
        assert len(props) > 0, "Schema has no properties"
        logger.info(
            "[%s] Discovery OK — %d fields: %s", config.name, len(props), list(props.keys())
        )

        # Verify extraction
        assert len(results.records) > 0, "No records extracted"
        logger.info("[%s] Extraction OK — %d records", config.name, len(results.records))

        # Show a sample record
        record = results.records[0]
        if hasattr(record, "model_dump"):
            logger.info("[%s] Sample record: %s", config.name, record.model_dump())
        else:
            logger.info("[%s] Sample record: %s", config.name, record)

        # Verify export
        df = results.to_dataframe()
        logger.info("[%s] DataFrame: %d rows, columns=%s", config.name, len(df), list(df.columns))

        logger.info("[%s] PASSED (%.1fs)", config.name, elapsed)
        return True

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("[%s] FAILED (%.1fs): %s", config.name, elapsed, e, exc_info=True)
        return False


def test_normalization(embedding_provider: ProviderConfig) -> bool:
    """Test normalization with real embeddings (needs OpenAI or compatible endpoint)."""
    api_key = os.environ.get(embedding_provider.api_key_env, "")
    if not api_key:
        logger.warning("[Normalization] Skipped — %s not set", embedding_provider.api_key_env)
        return False

    if not embedding_provider.embedding_model:
        logger.warning(
            "[Normalization] Skipped — no embedding model for %s", embedding_provider.name
        )
        return False

    from catchfly.normalization.embedding_cluster import EmbeddingClustering

    logger.info(
        "[Normalization] Testing with %s embeddings (%s)",
        embedding_provider.name,
        embedding_provider.embedding_model,
    )
    start = time.monotonic()

    try:
        normalizer = EmbeddingClustering(
            embedding_model=embedding_provider.embedding_model,
            base_url=embedding_provider.base_url,
            api_key=api_key,
            clustering_algorithm="agglomerative",
            similarity_threshold=0.7,
            reduce_dimensions=False,
        )

        values = [
            "New York",
            "NYC",
            "NY",
            "new york city",
            "Los Angeles",
            "LA",
            "L.A.",
            "los angeles",
            "San Francisco",
            "SF",
            "San Fran",
        ]

        result = normalizer.normalize(values, context_field="city")
        elapsed = time.monotonic() - start

        # Verify all values mapped
        assert len(result.mapping) == len(set(values)), (
            f"Expected {len(set(values))} mappings, got {len(result.mapping)}"
        )

        # Print mapping
        logger.info("[Normalization] Results:")
        for raw, canonical in sorted(result.mapping.items()):
            logger.info("  %s → %s", raw, canonical)

        # Check that at least some NYC variants clustered together
        ny_canonical = result.mapping.get("NYC")
        if ny_canonical and result.mapping.get("NY") == ny_canonical:
            logger.info("[Normalization] NYC/NY clustered correctly → %s", ny_canonical)
        else:
            logger.warning("[Normalization] NYC/NY did NOT cluster together")

        # Test explain
        explanation = result.explain("NYC")
        logger.info("[Normalization] explain('NYC'): %s", explanation)

        logger.info("[Normalization] PASSED (%.1fs)", elapsed)
        return True

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("[Normalization] FAILED (%.1fs): %s", elapsed, e, exc_info=True)
        return False


def test_full_pipeline_with_normalization(config: ProviderConfig) -> bool:
    """Test full pipeline including normalization (needs OpenAI for embeddings)."""
    api_key = os.environ.get(config.api_key_env, "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        logger.warning("[Full-%s] Skipped — %s not set", config.name, config.api_key_env)
        return False

    if not openai_key and not config.embedding_model:
        logger.warning("[Full-%s] Skipped — needs OPENAI_API_KEY for embeddings", config.name)
        return False

    from catchfly import Pipeline
    from catchfly.demo import load_samples
    from catchfly.discovery.single_pass import SinglePassDiscovery
    from catchfly.extraction.llm_direct import LLMDirectExtraction
    from catchfly.normalization.embedding_cluster import EmbeddingClustering

    logger.info("[Full-%s] Testing full pipeline with normalization", config.name)
    start = time.monotonic()

    try:
        pipeline = Pipeline(
            discovery=SinglePassDiscovery(
                model=config.model,
                base_url=config.base_url,
                api_key=api_key,
            ),
            extraction=LLMDirectExtraction(
                model=config.model,
                base_url=config.base_url,
                api_key=api_key,
            ),
            normalization=EmbeddingClustering(
                embedding_model="text-embedding-3-small",
                api_key=openai_key or api_key,
                reduce_dimensions=False,
                clustering_algorithm="agglomerative",
                similarity_threshold=0.7,
            ),
        )

        docs = load_samples("support_tickets")[:3]
        results = pipeline.run(
            documents=docs,
            domain_hint="SaaS support tickets with categories and priorities",
            normalize_fields=["category", "priority"],
        )

        elapsed = time.monotonic() - start

        assert results.schema is not None
        assert len(results.records) > 0
        logger.info("[Full-%s] %d records extracted", config.name, len(results.records))

        for field, norm_result in results.normalizations.items():
            logger.info(
                "[Full-%s] Normalized '%s': %d unique → %d canonical",
                config.name,
                field,
                len(norm_result.mapping),
                len(set(norm_result.mapping.values())),
            )

        df = results.to_dataframe()
        logger.info("[Full-%s] DataFrame: %d rows, %s", config.name, len(df), list(df.columns))
        logger.info("[Full-%s] PASSED (%.1fs)", config.name, elapsed)
        return True

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("[Full-%s] FAILED (%.1fs): %s", config.name, elapsed, e, exc_info=True)
        return False


def main() -> None:
    print("=" * 60)
    print("Catchfly v0.1.0 — Live Integration Tests")
    print("=" * 60)
    print()

    # Show which API keys are available
    for p in PROVIDERS:
        key = os.environ.get(p.api_key_env, "")
        status = f"set ({len(key)} chars)" if key else "NOT SET"
        print(f"  {p.name:12s} ({p.api_key_env}): {status}")
    print()

    results: dict[str, bool | None] = {}

    # --- Test 1: Discovery + Extraction per provider ---
    print("-" * 60)
    print("TEST 1: Discovery + Extraction")
    print("-" * 60)
    for config in PROVIDERS:
        key = f"discovery+extraction:{config.name}"
        results[key] = test_discovery_and_extraction(config)
        print()

    # --- Test 2: Normalization with embeddings ---
    print("-" * 60)
    print("TEST 2: Normalization (embedding clustering)")
    print("-" * 60)
    # Use first provider that has an embedding model configured
    embedding_provider = next((p for p in PROVIDERS if p.embedding_model), None)
    if embedding_provider:
        results["normalization"] = test_normalization(embedding_provider)
    else:
        logger.warning("No embedding provider available")
        results["normalization"] = False
    print()

    # --- Test 3: Full pipeline with normalization ---
    print("-" * 60)
    print("TEST 3: Full pipeline (discovery + extraction + normalization)")
    print("-" * 60)
    # Test with the first available provider
    for config in PROVIDERS:
        if os.environ.get(config.api_key_env):
            key = f"full_pipeline:{config.name}"
            results[key] = test_full_pipeline_with_normalization(config)
            print()
            break  # Only test one to save costs

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    for name, result in results.items():
        icon = "PASS" if result else "FAIL" if result is False else "SKIP"
        print(f"  [{icon}] {name}")
    print(f"\n  {passed} passed, {failed} failed, {len(results) - passed - failed} skipped")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
