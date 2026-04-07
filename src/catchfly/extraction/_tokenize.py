"""Token counting helpers used by dry-run cost estimation.

Prefers :mod:`tiktoken` when installed (accurate for OpenAI-compatible
models), falls back to a conservative ``chars / 4`` heuristic so core
catchfly retains no hard dependency on a tokenizer.
"""

from __future__ import annotations

from collections.abc import Callable

EncodeFn = Callable[[str], list[int]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def count_tokens(text: str, model: str) -> tuple[int, str]:
    """Return ``(token_count, backend)`` for *text* under *model*.

    The returned backend string is either ``"tiktoken:<encoding>"`` or
    ``"heuristic:chars-over-4"`` so callers can record which method was
    used in downstream reports.
    """
    encoder = _get_tiktoken_encoder(model)
    if encoder is not None:
        encoding_name, encode = encoder
        return len(encode(text)), f"tiktoken:{encoding_name}"
    return _heuristic_tokens(text), "heuristic:chars-over-4"


def count_messages_tokens(
    messages: list[dict[str, str]], model: str
) -> tuple[int, str]:
    """Count tokens across a list of chat messages.

    Adds a fixed per-message overhead (4 tokens) to approximate OpenAI
    chat formatting. The overhead is constant across backends so that
    the heuristic and tiktoken paths agree on the accounting.
    """
    total = 0
    backend = "heuristic:chars-over-4"
    for msg in messages:
        content = msg.get("content", "") or ""
        role = msg.get("role", "") or ""
        count, backend = count_tokens(content + role, model)
        total += count + 4  # ~4 tokens per message for formatting
    return total, backend


def _heuristic_tokens(text: str) -> int:
    """Conservative token estimate: ``ceil(len(text) / 4)``.

    Modern BPE tokenizers average ~4 characters per token on English
    prose. This overestimates slightly for code and underestimates for
    CJK, which is acceptable for a dry-run upper bound.
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


# ---------------------------------------------------------------------------
# tiktoken integration (optional)
# ---------------------------------------------------------------------------


def _get_tiktoken_encoder(model: str) -> tuple[str, EncodeFn] | None:
    """Return ``(encoding_name, encode_fn)`` or ``None`` if tiktoken is
    unavailable or the model is not mappable to a known encoding.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None

    # tiktoken.encoding_for_model handles the OpenAI family; for
    # anything else we fall back to cl100k_base which is the closest
    # approximation for modern GPT-style models.
    try:
        enc = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except (KeyError, ValueError):
            return None

    encoding_name: str = getattr(enc, "name", "unknown")
    encode_fn: EncodeFn = enc.encode
    return encoding_name, encode_fn
