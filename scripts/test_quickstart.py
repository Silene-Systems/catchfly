"""Test script — full pipeline on real Amazon reviews from HuggingFace.

Discovery → Extraction → LLMFieldSelector → Normalization on 50 reviews.
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset

from catchfly import Document, Pipeline
from catchfly.discovery import SinglePassDiscovery
from catchfly.extraction import LLMDirectExtraction
from catchfly.normalization import LLMCanonicalization
from catchfly.selection import LLMFieldSelector

# --- Load real data from HuggingFace ---
# RareArena — rare disease case reports from PMC, mapped to Orphanet
# This is catchfly's core use case: unstructured clinical text → structured data
print("Downloading RareArena (rare disease case reports) from HuggingFace...")
ds = load_dataset("THUMedInfo/RareArena", data_files="RDC.json", split="train", streaming=True)

N_DOCS = 50
docs: list[Document] = []
for row in ds:
    case = row.get("case_report", "") or ""
    tests = row.get("test_results", "") or ""
    if len(case) < 100:
        continue

    # Combine case report + test results into a single clinical document
    full_text = case
    if tests:
        full_text += f"\n\nTest results: {tests}"

    docs.append(Document(
        content=full_text,
        id=row.get("_id", f"case-{len(docs)}"),
    ))
    if len(docs) >= N_DOCS:
        break

print(f"Loaded {len(docs)} reviews (avg {sum(len(d.content) for d in docs) // len(docs)} chars)\n")
print("Sample document:")
print(docs[0].content[:300])
print("...\n")

# --- Build pipeline with LLMFieldSelector ---
MODEL = "gpt-4.1-mini"

pipeline = Pipeline(
    discovery=SinglePassDiscovery(model=MODEL, max_fields=8),
    extraction=LLMDirectExtraction(model=MODEL, on_error="skip"),
    normalization=LLMCanonicalization(model=MODEL),
    field_selector=LLMFieldSelector(model=MODEL),
)

# --- Run WITHOUT normalize_fields — field selector decides ---
print("Running pipeline (discovery → extraction → field selection → normalization)...\n")
results = pipeline.run(
    docs,
    domain_hint="Rare disease clinical case reports with symptoms, diagnosis, and treatment",
)

# --- 1. Schema ---
print("=" * 60)
print("DISCOVERED SCHEMA")
print("=" * 60)
print(json.dumps(results.schema.json_schema, indent=2))

# --- 2. Records ---
print(f"\n{'=' * 60}")
print(f"EXTRACTED RECORDS ({len(results.records)} total, {len(results.errors)} errors)")
print("=" * 60)
for i, record in enumerate(results.records[:5]):
    print(f"\n  Record {i + 1}:")
    if hasattr(record, "model_dump"):
        for k, v in record.model_dump().items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:80] + "..."
            print(f"    {k}: {val_str}")
    else:
        print(f"    {record}")

# --- 3. Auto-selected fields ---
print(f"\n{'=' * 60}")
print("FIELD SELECTION (LLMFieldSelector)")
print("=" * 60)
selected = list(results.normalizations.keys())
print(f"  Auto-selected: {selected}")

# Show what was NOT selected (and why it makes sense)
all_fields = list(results.schema.json_schema.get("properties", {}).keys())
skipped = [f for f in all_fields if f not in selected]
print(f"  Skipped: {skipped}")

# --- 4. Normalization results ---
print(f"\n{'=' * 60}")
print(f"NORMALIZATION ({len(results.normalizations)} fields)")
print("=" * 60)
for field_name, norm in results.normalizations.items():
    n_groups = len(norm.clusters) if norm.clusters else 0
    n_values = len(norm.mapping) if norm.mapping else 0
    n_multi = sum(1 for members in (norm.clusters or {}).values() if len(members) > 1)
    print(f"\n  '{field_name}' — {n_values} values → {n_groups} groups ({n_multi} with merges):")
    for canonical, members in sorted(
        (norm.clusters or {}).items(),
        key=lambda x: -len(x[1]),
    )[:15]:
        if len(members) > 1:
            shown = members[:5]
            extra = f" + {len(members) - 5} more" if len(members) > 5 else ""
            print(f"    {canonical}: {shown}{extra}")

# --- 5. Cost ---
print(f"\n{'=' * 60}")
print("COST")
print("=" * 60)
print(f"  Total cost: ${results.report.total_cost_usd:.4f}")
print(f"  Total tokens: {results.report.total_input_tokens + results.report.total_output_tokens:,}")
