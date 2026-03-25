# Installation

## With pip

```bash
pip install catchfly                  # Core only (~5 MB)
pip install catchfly[openai]          # + OpenAI SDK
pip install catchfly[clustering]      # + scikit-learn, numpy, umap
pip install catchfly[export]          # + pandas, pyarrow
pip install catchfly[medical]         # + ontology loaders
pip install catchfly[chunking]        # + chonkie (token/sentence/recursive)
pip install catchfly[semantic-chunking]  # + chonkie with semantic splitting
pip install catchfly[all]             # Everything
```

## With uv

```bash
uv add catchfly[openai,clustering,export]
```

## What's in each extra?

| Extra | Packages | Used By |
|---|---|---|
| `openai` | `openai` SDK | LLM calls, embeddings (API-based) |
| `clustering` | `scikit-learn`, `numpy`, `umap-learn` | EmbeddingClustering, KLLMeansClustering |
| `export` | `pandas`, `pyarrow` | `results.to_dataframe()`, `.to_parquet()` |
| `medical` | `pronto`, `numpy` | OntologyMapping (HPO, custom ontologies) |
| `chunking` | `chonkie` | TokenChunking, SentenceChunking, RecursiveChunking |
| `semantic-chunking` | `chonkie[semantic]` | SemanticChunking (embedding-based splitting) |
| `pdf` | `pymupdf` | PDF document loading |

## Requirements

- **Python 3.10+**
- **An OpenAI-compatible LLM endpoint** — OpenAI, Anthropic, Mistral, Ollama, vLLM, or any provider with an OpenAI-compatible API

## Local Models

Catchfly works with local models via Ollama or vLLM — no API key needed:

```python
from catchfly import Pipeline

pipeline = Pipeline.quick(
    model="qwen3.5",
    base_url="http://localhost:11434/v1",
)
```
