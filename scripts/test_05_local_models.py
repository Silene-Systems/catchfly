"""Test script for 05_local_models — Ollama pipeline.

Requires: ollama running locally with `ollama pull qwen3.5`
"""

from __future__ import annotations

from catchfly import Pipeline
from catchfly.demo import load_samples
from catchfly.discovery import SinglePassDiscovery
from catchfly.extraction import LLMDirectExtraction
from catchfly.normalization import LLMCanonicalization

OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "qwen3.5"

# --- Load demo data ---
docs = load_samples("case_reports")
print(f"Loaded {len(docs)} case reports\n")

# --- Pipeline.quick ---
print("=" * 60)
print("PIPELINE.QUICK (auto everything)")
print("=" * 60)
pipeline = Pipeline.quick(model=MODEL, base_url=OLLAMA_URL)

results = pipeline.run(
    docs[:3],
    domain_hint="Rare disease clinical case reports",
)

print(f"Auto-normalized fields: {list(results.normalizations.keys())}")
for record in results.records:
    print(f"  {record.diagnosis}")

# --- Modular: Discovery ---
print(f"\n{'=' * 60}")
print("MODULAR: Discovery")
print("=" * 60)
discovery = SinglePassDiscovery(model=MODEL, base_url=OLLAMA_URL)
schema = discovery.discover(docs[:3], domain_hint="Medical case reports")

print("Discovered fields:")
for name in schema.json_schema["properties"]:
    print(f"  - {name}")

# --- Modular: Extraction ---
print(f"\n{'=' * 60}")
print("MODULAR: Extraction")
print("=" * 60)
extractor = LLMDirectExtraction(model=MODEL, base_url=OLLAMA_URL, max_retries=2, on_error="skip")
result = extractor.extract(schema.model, docs[:3])
print(f"Extracted {len(result.records)} records")

# --- Modular: Normalization ---
print(f"\n{'=' * 60}")
print("MODULAR: Normalization")
print("=" * 60)
normalizer = LLMCanonicalization(model=MODEL, base_url=OLLAMA_URL)
norm = normalizer.normalize(
    ["seizures", "epileptic fits", "convulsions", "hepatomegaly", "liver enlargement"],
    context_field="phenotypes",
)
for raw_val, canonical in norm.mapping.items():
    print(f"  {raw_val} -> {canonical}")

print("\nDone. All stages ran locally via Ollama.")
