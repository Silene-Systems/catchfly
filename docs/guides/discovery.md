# Schema Discovery

Catchfly offers three discovery strategies, from simple to sophisticated.

## SinglePassDiscovery

One LLM call proposes a JSON Schema from sample documents.

```python
from catchfly.discovery.single_pass import SinglePassDiscovery

discovery = SinglePassDiscovery(
    model="gpt-5.4-mini",
    num_samples=5,          # how many docs to sample
    temperature=0.7,
)

schema = discovery.discover(
    documents=docs,
    domain_hint="Clinical case reports about rare diseases",
)

print(schema.json_schema)   # JSON Schema dict
print(schema.model)         # Pydantic model class (or None if codegen failed)
```

**Best for:** Quick exploration, small corpora, prototyping.

## ThreeStageDiscovery

Progressive refinement inspired by SCHEMA-MINER:

1. **Stage 1 — Initial:** Schema from 2-3 exemplar documents
2. **Stage 2 — Refinement:** Validate against 5-10 docs, fix gaps and redundancies
3. **Stage 3 — Expansion:** Test against 50+ docs, add recurring fields, flag rare ones

```python
from catchfly.discovery.three_stage import ThreeStageDiscovery

discovery = ThreeStageDiscovery(
    model="gpt-5.4-mini",
    stage1_samples=3,
    stage2_samples=10,
    stage3_samples=50,
    human_review=False,   # set True to stop after Stage 2 for inspection
)

schema = discovery.discover(docs, domain_hint="Support tickets")

# Check the discovery report
report = schema.field_metadata.get("_discovery_report", {})
print(report["field_coverage"])    # {field: coverage_fraction}
print(report["stages_completed"])  # 3
```

**Best for:** Production schemas, large/heterogeneous corpora.

## SchemaOptimizer (PARSE-style)

Iteratively enriches field descriptions by analyzing extraction errors:

```python
from catchfly.discovery.optimizer import SchemaOptimizer

optimizer = SchemaOptimizer(
    model="gpt-5.4-mini",
    num_iterations=5,
)

enriched = optimizer.optimize(
    schema=my_schema,              # Schema or Pydantic model
    test_documents=docs[:20],
)

# Each field now has rich metadata
for field, meta in enriched.field_metadata.items():
    print(f"{field}: {meta.get('description')}")
    print(f"  examples: {meta.get('examples')}")
    print(f"  synonyms: {meta.get('synonyms')}")
```

The enriched `field_metadata` can be passed to `KLLMeansClustering` as seed centroids — this is the bridge between discovery and normalization.

**Best for:** Improving extraction quality, preparing for kLLMmeans normalization.
