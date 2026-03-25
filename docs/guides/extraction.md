# Data Extraction

## LLMDirectExtraction

Extracts structured data from documents using LLM with tool calling.

```python
from catchfly.extraction import LLMDirectExtraction
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    features: list[str]

extractor = LLMDirectExtraction(
    model="gpt-5.4-mini",
    chunk_size=4000,     # characters per chunk
    chunk_overlap=200,   # overlap between chunks
    max_retries=3,       # retry on validation error
    batch_size=10,       # concurrent requests
    on_error="collect",  # "raise", "skip", or "collect"
)

result = extractor.extract(schema=Product, documents=docs)

result.records      # list[Product]
result.errors       # list[(Document, Exception)] if on_error="collect"
result.provenance   # list[RecordProvenance] with char offsets
```

## Structured Output Strategy

Catchfly uses a cascading strategy for structured output (inspired by Pydantic-AI):

1. **Tool calling** — most universal, works with all major providers
2. **json_schema response_format** — OpenAI, Anthropic
3. **json_object response_format** — OpenAI, local models
4. **Prompt-only fallback** — always works

This means extraction works with OpenAI, Anthropic, Mistral, Ollama, and vLLM without configuration changes.

## Chunking Strategies

Long documents are split into chunks before extraction. Catchfly provides a `ChunkingStrategy` protocol with multiple implementations.

### Built-in: Fixed-size chunking

The default chunking splits by character count with overlap:

```python
extractor = LLMDirectExtraction(
    model="gpt-5.4-mini",
    chunk_size=4000,
    chunk_overlap=200,
)
```

### Chonkie-backed strategies

For smarter chunking, install the `chunking` extra and use chonkie-backed strategies:

```bash
pip install catchfly[chunking]           # token, sentence, recursive
pip install catchfly[semantic-chunking]  # + semantic (embedding-based)
```

```python
from catchfly.extraction import (
    LLMDirectExtraction,
    TokenChunking,
    SentenceChunking,
    RecursiveChunking,
    SemanticChunking,
)

# Token-based chunking (fixed token count + overlap)
extractor = LLMDirectExtraction(
    model="gpt-5.4-mini",
    chunking_strategy=TokenChunking(chunk_size=512, chunk_overlap=64),
)

# Sentence-boundary aware
extractor = LLMDirectExtraction(
    model="gpt-5.4-mini",
    chunking_strategy=SentenceChunking(chunk_size=512),
)

# Hierarchical delimiter rules (paragraphs → sentences → words)
extractor = LLMDirectExtraction(
    model="gpt-5.4-mini",
    chunking_strategy=RecursiveChunking(chunk_size=512),
)

# Embedding-based semantic splitting (requires semantic-chunking extra)
extractor = LLMDirectExtraction(
    model="gpt-5.4-mini",
    chunking_strategy=SemanticChunking(chunk_size=512),
)
```

All chunking strategies preserve character offsets in provenance metadata.

### Choosing a chunking strategy

| Strategy | Best For | Cost |
|---|---|---|
| Fixed-size (default) | Simple documents, consistent length | Zero |
| `TokenChunking` | Token-budget-aware splitting | Zero |
| `SentenceChunking` | Preserving sentence boundaries | Zero |
| `RecursiveChunking` | Structured documents (paragraphs, sections) | Zero |
| `SemanticChunking` | Topic-coherent chunks, research papers | Embedding calls |

## Error Handling

```python
# Fail fast (default)
extractor = LLMDirectExtraction(on_error="raise")

# Skip bad documents silently
extractor = LLMDirectExtraction(on_error="skip")

# Collect errors and continue
extractor = LLMDirectExtraction(on_error="collect")
result = extractor.extract(schema=Product, documents=docs)
for doc, error in result.errors:
    print(f"Failed: {doc.id} — {error}")
```

## Provenance Tracking

Each extracted record tracks its source:

```python
for prov in result.provenance:
    print(f"Source: {prov.source_document}")
    print(f"Chunk: {prov.chunk_index}")
    print(f"Chars: {prov.char_start}-{prov.char_end}")
    print(f"Confidence: {prov.confidence}")
```

Confidence is based on retry count: `1.0` on first attempt, decreasing by `0.3` per retry.
