"""Tests for async/sync compatibility layer."""

from __future__ import annotations

import asyncio

from catchfly._compat import run_sync


class TestRunSync:
    def test_basic_coroutine(self) -> None:
        async def add(a: int, b: int) -> int:
            return a + b

        result = run_sync(add(2, 3))
        assert result == 5

    def test_async_with_await(self) -> None:
        async def slow_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        result = run_sync(slow_add(10, 20))
        assert result == 30
