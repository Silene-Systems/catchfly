"""Convenience loaders for creating Document objects from files."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

from catchfly._types import Document

logger = logging.getLogger(__name__)


def load_glob(pattern: str, *, encoding: str = "utf-8") -> list[Document]:
    """Load documents from files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g. ``"data/*.txt"``, ``"docs/**/*.md"``).
        encoding: Text encoding for reading files.

    Returns:
        List of Document objects with content read from each file.
    """
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        logger.warning("load_glob: no files matched pattern '%s'", pattern)
        return []

    documents: list[Document] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding=encoding)
            documents.append(
                Document(content=content, id=path.name, source=str(path))
            )
        except Exception:
            logger.warning("load_glob: failed to read '%s'", p, exc_info=True)

    logger.info(
        "load_glob: loaded %d documents from pattern '%s'",
        len(documents),
        pattern,
    )
    return documents


def resolve_documents(
    documents: list[Document] | list[str],
) -> list[Document]:
    """Resolve a list of Documents or glob pattern strings to Documents.

    If the list contains strings, each is treated as a glob pattern
    and expanded via :func:`load_glob`.
    """
    if not documents:
        return []

    if isinstance(documents[0], str):
        resolved: list[Document] = []
        for pattern in documents:
            resolved.extend(load_glob(str(pattern)))
        return resolved

    return documents  # type: ignore[return-value]
