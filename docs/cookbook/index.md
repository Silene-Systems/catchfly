# Cookbook

Practical, runnable Jupyter notebooks showing catchfly in action. Each notebook is self-contained — start wherever fits your use case.

| # | Notebook | What you'll learn | Difficulty | Est. API cost |
|---|----------|-------------------|------------|---------------|
| 01 | [Quick Start](https://github.com/silene-systems/catchfly/blob/main/notebooks/01_quickstart.ipynb) | 5 lines to structured data | Beginner | ~$0.01 |
| 02 | [Rare Disease Case Reports](https://github.com/silene-systems/catchfly/blob/main/notebooks/02_rare_disease.ipynb) | Full pipeline with HPO ontology coding | Intermediate | ~$0.05 |
| 03 | [Product Catalog](https://github.com/silene-systems/catchfly/blob/main/notebooks/03_product_catalog.ipynb) | Cost control, checkpoints, per-field normalization | Intermediate | ~$0.03 |
| 04 | [Custom Schema](https://github.com/silene-systems/catchfly/blob/main/notebooks/04_custom_schema.ipynb) | Skip discovery, Pydantic models, JSON Schema | Beginner | ~$0.02 |
| 05 | [Local Models](https://github.com/silene-systems/catchfly/blob/main/notebooks/05_local_models.ipynb) | Ollama, zero API cost, data stays local | Beginner | $0.00 |

## Run locally

```bash
pip install "catchfly[all]" jupyter
cd notebooks/
jupyter notebook
```

All notebooks use catchfly's bundled demo datasets — no external data downloads required.

## What makes catchfly different

Most LLM extraction libraries stop at "LLM outputs JSON." Catchfly goes further:

- **Schema discovery** — the LLM figures out *what* to extract, not just *how*
- **Normalization** — extracted values like "ALT", "ALAT", and "Alanine aminotransferase" are mapped to a single canonical form
- **Ontology grounding** — phenotypes map to HPO codes, diagnoses to ICD-10, with confidence scores
- **Cascade pipeline** — Dictionary, LLM, and Ontology normalization chain together with fallback

No other open-source library offers this end-to-end pipeline.
