<p align="center">
  <h1 align="center">catchfly</h1>
  <p align="center"><strong>From chaos to categories.</strong></p>
  <p align="center">Schema discovery + term normalization for unstructured text data.</p>
  <p align="center">
    <a href="https://pypi.org/project/catchfly/"><img src="https://img.shields.io/pypi/v/catchfly?color=6B2D8B" alt="PyPI"></a>
    <a href="https://github.com/silene-systems/catchfly/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/catchfly" alt="Python"></a>
  </p>
</p>

---

> *Named after [Silene](https://en.wikipedia.org/wiki/Silene) (catchfly) — plants that secrete a sticky substance to capture insects. Catchfly captures scattered terms and groups them into canonical categories.*
>
> *Part of the [Silene Systems](https://silene.systems) ecosystem for rare disease research.*

## The problem

You extracted 3,000 mentions from 200 scientific papers. Or an LLM classified 3 million emails into 4,000 labels. Now you have:

- **No schema** — you don't know what categories should exist
- **Duplicates everywhere** — "miglustat" and "Zavesca" are the same drug
- **Ambiguous boundaries** — is "cognitive decline" the same as "cognitive impairment"?
- **Related but distinct entities** — "ALT" and "AST" are similar but clinically different

No existing tool does both **schema discovery** and **term normalization** as a composable Python library.

## What catchfly does

**Two operations, one pipeline:**

**Discover** — *"What categories exist in my data?"*

```python
from catchfly import Resolver

resolver = Resolver(embed_provider="gemini", llm_provider="openai/gpt-5.4")

schema = resolver.discover(
    mentions=["miglustat", "splenomegaly", "ALT: 120", "ataxia", ...],
    contexts={"miglustat": ["Patient received miglustat 200mg daily"], ...},
)

# schema.categories → ["Treatments", "Symptoms", "Lab Values", ...]
# schema.examples  → {"Treatments": ["miglustat", "arimoclomol"], ...}

schema.rename("Lab Values", "Laboratory Findings")
schema.merge("Symptoms", "Clinical Signs")
```

**Normalize** — *"Which terms belong where, and which are synonyms?"*

```python
result = resolver.normalize(
    mentions=all_mentions,
    schema=schema,
    contexts=contexts,
)

# result.groups    → [NormGroup(canonical="miglustat", members=["Zavesca", "NB-DNJ"])]
# result.ambiguous → [AmbiguousPair("cognitive decline", "cognitive impairment")]
```

**Or all-in-one:**

```python
result = resolver.resolve(mentions, contexts=contexts)
```

## Self-improving prompts

Give catchfly 20–50 labeled examples and it optimizes its own prompts for your domain. Inspired by [GEPA](https://github.com/gepa-ai/gepa), implemented from scratch with zero external dependencies.

```python
optimized = resolver.optimize(
    mentions=mentions,
    contexts=contexts,
    ground_truth={"miglustat": "miglustat", "Zavesca": "miglustat", "ALT": "ALT", "AST": "AST"},
    iterations=20,  # ~$1–2, ~15 min
)

result = optimized.resolve(new_mentions)  # 20–30% better accuracy on your domain
```

## How it works

Four components, composable as a pipeline:

| Component | What it does | Cost |
|---|---|---|
| **EmbeddingSimilarity** | Pre-filter: finds candidate pairs via cosine similarity | Embedding API only |
| **LLMGrouper** | Core engine: LLM analyzes clusters, proposes categories/synonyms with 4-way relation typing (synonym / hierarchy / related / distinct) | LLM API |
| **LLMClassifier** | Propagation: assigns remaining mentions to approved categories via few-shot classification | LLM API |
| **UserSeeded** | User guidance: seed mappings override and anchor discovery + normalization | Embedding only |

## Tiered quality

| Tier | What runs | Cost / 1K mentions | Accuracy | Best for |
|---|---|---|---|---|
| **Tier 1** (free) | Embedding only | $0 (local models) | 60–70% | Exploration |
| **Tier 2** (standard) | Embedding + LLM | $0.15–0.50 | 82–90% | Production |
| **Tier 3** (optimized) | Tier 2 + optimize() | $1–3 one-time | 90–95%+ | Systematic reviews |

## Use cases

- **Medical literature** — normalize symptoms, drugs, genetic variants from systematic reviews
- **Email categorization** — collapse 4,000 LLM-generated labels into 150 clean groups
- **E-commerce tags** — build taxonomy from 50,000 user-generated product tags
- **Any domain** — if you have messy text labels, catchfly cleans them up

## Provider support

```python
# Embeddings
Resolver(embed_provider="gemini")       # Google Gemini Embedding 2
Resolver(embed_provider="openai")        # OpenAI text-embedding-3
Resolver(embed_provider="local")         # sentence-transformers (offline, free)
Resolver(embed_provider=my_function)     # any callable

# LLM
Resolver(llm_provider="openai/gpt-5.4")
Resolver(llm_provider="gemini/flash")
Resolver(llm_provider=my_function)       # any callable
```

## Evaluation

```python
ground_truth = {"miglustat": "miglustat", "Zavesca": "miglustat", "ALT": "ALT", "AST": "AST"}
metrics = resolver.evaluate(result, ground_truth)
# → precision, recall, false_merges, missed_merges, category_accuracy
```

## Installation

```bash
pip install catchfly
```

> **v0.0.1 is a name reservation.** Active development underway. First functional release (v0.1.0) expected Q2 2026.

## Part of the Silene ecosystem

| Project | What it is |
|---|---|
| **[Silene Systems](https://silene.systems)** | Computational phenotyping platform for rare diseases |
| **Campion** | Agentic batch literature extraction platform (SaaS) |
| **catchfly** | Schema discovery + term normalization library (open source, this repo) |

> *[Silene tomentosa](https://en.wikipedia.org/wiki/Silene_tomentosa) (Gibraltar campion) is one of the rarest plants in the world — thought extinct in 1992, rediscovered in 1994 on the Rock of Gibraltar. Like rare diseases, it hides in plain sight, waiting for someone to look carefully enough.*

## License

Apache-2.0

## Citation

If you use catchfly in academic work:

```bibtex
@software{catchfly2026,
  author = {Michalski, Adrian},
  title = {catchfly: Schema discovery and category normalization for unstructured text data},
  year = {2026},
  url = {https://github.com/silene-systems/catchfly},
  license = {Apache-2.0}
}
``