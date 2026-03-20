# Data Extraction

## LLMDirectExtraction

Extracts structured data from documents using LLM with tool calling.

```python
from catchfly.extraction.llm_direct import LLMDirectExtraction
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
```
