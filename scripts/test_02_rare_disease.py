"""Test script for 02_rare_disease — ThreeStageDiscovery + SchemaOptimizer + CascadeNormalization + HPO."""

from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import hf_hub_download

from catchfly import Document

# --- Step 1: Load RareArena ---
print("Downloading RareArena case reports...")
path = hf_hub_download(repo_id="THUMedInfo/RareArena", filename="RDC.json", repo_type="dataset")
raw = [json.loads(line) for line in open(path) if line.strip()]

docs: list[Document] = []
for row in raw:
    case = row.get("case_report", "") or ""
    tests = row.get("test_results", "") or ""
    if len(case) < 100:
        continue
    full_text = case + (f"\n\nTest results: {tests}" if tests else "")
    docs.append(Document(content=full_text, id=row.get("_id", f"case-{len(docs)}")))
    if len(docs) >= 10:
        break

print(f"{len(docs)} case reports loaded (avg {sum(len(d.content) for d in docs) // len(docs)} chars)\n")

MODEL = "gpt-4.1-mini"

# --- Step 2: ThreeStageDiscovery ---
print("=" * 60)
print("STEP 2: Discovery")
print("=" * 60)
from catchfly.discovery import SinglePassDiscovery

discovery = SinglePassDiscovery(
    model=MODEL,
    max_fields=8,
    suggested_fields=["symptoms", "diagnosis", "treatment"],
)
schema = discovery.discover(docs, domain_hint="Rare disease clinical case reports")

print("Discovered fields:")
for name, prop in schema.json_schema["properties"].items():
    print(f"  {name}: {prop.get('type', 'object')}")

# --- Step 3: SchemaOptimizer ---
print(f"\n{'=' * 60}")
print("STEP 3: SchemaOptimizer")
print("=" * 60)
from catchfly.discovery import SchemaOptimizer

optimizer = SchemaOptimizer(model=MODEL, num_iterations=2)
enriched = optimizer.optimize(schema, test_documents=docs[:10])

for field_name in list(enriched.field_metadata.keys())[:4]:
    meta = enriched.field_metadata[field_name]
    print(f"\n  {field_name}:")
    print(f"    Description: {meta.get('description', '(none)')}")
    print(f"    Examples: {meta.get('examples', [])}")

# --- Step 4: Extract ---
print(f"\n{'=' * 60}")
print("STEP 4: Extraction")
print("=" * 60)
from catchfly.extraction import LLMDirectExtraction

extractor = LLMDirectExtraction(
    model=MODEL,
    on_error="collect",
)
result = extractor.extract(enriched.model, docs)

print(f"Extracted {len(result.records)} records, {len(result.errors)} errors\n")
for record in result.records[:3]:
    if hasattr(record, "model_dump"):
        data = record.model_dump()
        for k, v in data.items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:80] + "..."
            print(f"    {k}: {val_str}")
        print()

# --- Step 5: CascadeNormalization + HPO ---
print(f"\n{'=' * 60}")
print("STEP 5: CascadeNormalization + HPO")
print("=" * 60)
from catchfly.normalization import CascadeNormalization

cascade = CascadeNormalization.default(
    dictionary={
        "ALT": "Alanine aminotransferase",
        "AST": "Aspartate aminotransferase",
        "CK": "Creatine kinase",
    },
    model=MODEL,
    ontology="hpo",
)

# Collect values from any string or array-of-string field that looks clinical
all_phenotypes: list[str] = []
candidate_fields = ["symptoms", "phenotypes", "presenting_complaints", "clinical_presentation"]
used_field = None
for field in candidate_fields:
    for record in result.records:
        vals = getattr(record, field, None)
        if vals and isinstance(vals, list):
            all_phenotypes.extend(vals)
            used_field = field
        elif vals and isinstance(vals, str) and len(vals) > 10:
            # Split comma-separated symptoms from free-text fields
            all_phenotypes.append(vals)
            used_field = field
    if all_phenotypes:
        break

if used_field:
    print(f"Using field '{used_field}'")
else:
    print("WARNING: No symptom/phenotype field found in discovered schema")

print(f"Normalizing {len(all_phenotypes)} phenotype values ({len(set(all_phenotypes))} unique)")
norm_result = cascade.normalize(all_phenotypes, context_field="symptoms")

# Cascade steps
print("\nCascade steps:")
for step in norm_result.metadata.get("steps", []):
    print(f"  Step {step['step']} ({step['strategy']}): "
          f"mapped {step['mapped_count']}, remaining {step['remaining_count']}")

# Sample HPO mappings
print("\nSample mappings:")
sample_values = ["seizures", "hepatosplenomegaly", "dyspnea",
                 "muscle weakness", "hearing loss", "fever", "abdominal pain"]
for value in sample_values:
    if value in norm_result.mapping:
        print(f"  {norm_result.explain(value)}")

# Clusters with multiple members
if norm_result.clusters:
    print(f"\nClusters with merged variants:")
    for canonical, members in sorted(norm_result.clusters.items()):
        if len(members) > 1:
            print(f"  {canonical}: {members[:5]}")
else:
    print("\nNo clusters (too few values to normalize)")

print("\nDone.")
