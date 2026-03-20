"""Async/sync compatibility layer.

Provides run_sync() that works in both regular scripts and
environments with a running event loop (Jupyter notebooks).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Detects whether an event loop is already running (e.g. Jupyter)
    and handles it gracefully instead of raising RuntimeError.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — standard case (scripts, CLI)
        return asyncio.run(coro)

    # Running loop detected (Jupyter, async REPL)
    # Try nest_asyncio first (lightweight, widely used)
    try:
        import nest_asyncio  # type: ignore[import-not-found]

        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    except ImportError:
        pass

    # Fallback: run in a new thread to avoid blocking the existing loop
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()
