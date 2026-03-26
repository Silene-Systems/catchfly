"""Shared parsing utilities for LLM response handling."""

from __future__ import annotations


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses.

    Handles fenced blocks like ``json``, ``python``, and bare triple-backtick markers.
    Returns the inner content with leading/trailing whitespace removed.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return text
