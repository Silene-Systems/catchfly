# Contributing to Catchfly

## Setup

```bash
git clone https://github.com/silene-systems/catchfly.git
cd catchfly
uv sync --group dev --extra openai --extra clustering --extra medical --extra export
```

Or with pip:

```bash
pip install -e ".[openai,clustering,medical,export]"
pip install pytest pytest-asyncio pytest-cov mypy ruff
```

## Running Tests

```bash
# Unit tests (no API calls)
uv run pytest -m "not llm and not slow"

# With coverage
uv run pytest --cov=catchfly --cov-report=term-missing -m "not llm and not slow"

# Tests that hit real LLM APIs (requires OPENAI_API_KEY)
uv run pytest -m llm --timeout=300
```

## Code Quality

```bash
uv run ruff check .          # Lint
uv run ruff format --check . # Format check
uv run ruff format .         # Auto-format
uv run mypy src/             # Type check (strict mode)
```

All three must pass before merging. CI runs these automatically.

## Adding a New Strategy

1. **Implement the Protocol** — see `discovery/base.py`, `extraction/base.py`, or `normalization/base.py` for the interface your strategy must satisfy.
2. **Add both sync and async methods** — implement `async def a<method>()` first, then add a sync wrapper using `run_sync()`.
3. **Use Pydantic BaseModel for config** — strategy parameters should be Pydantic fields with sensible defaults.
4. **Add tests with MockLLMClient** — unit tests must not call real APIs. Use the fixtures from `tests/conftest.py`.
5. **Re-export from `__init__.py`** — add your strategy to the relevant subpackage's `__init__.py` and `__all__`.
6. **Add documentation** — update the relevant guide in `docs/guides/` and add API docs in `docs/api/`.

## Project Structure

```
src/catchfly/
├── discovery/     # Schema discovery strategies
├── extraction/    # Data extraction strategies
├── normalization/ # Value normalization strategies
├── providers/     # LLM and embedding clients
├── schema/        # Schema converters and registry
├── ontology/      # Ontology loaders and index
├── telemetry/     # Cost and usage tracking
├── demo/          # Built-in demo datasets
├── pipeline.py    # Pipeline orchestrator
├── _types.py      # Core domain types
├── _compat.py     # Async/sync compatibility
└── exceptions.py  # Exception hierarchy
```

## PR Process

- CI must be green (lint + typecheck + tests on Python 3.10-3.13)
- One approval required
- Keep PRs focused — one feature or fix per PR
