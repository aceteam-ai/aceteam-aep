"""Helpers for running coroutines from synchronous call sites."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

import anyio

T = TypeVar("T")


def run_coro_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` from synchronous code whether or not an event loop is running.

    ``asyncio.run()`` cannot be used when a loop is already running (e.g. Jupyter,
    ASGI). In that case we complete the coroutine on a worker thread using
    ``anyio.run()`` with a fresh loop. (``anyio.from_thread`` is for the opposite
    direction: call into a *running* loop from another thread.)

    Note: asyncio objects bound to a *different* loop (rare) may misbehave when
    session state was created on the main loop but sync calls are bridged this way;
    prefer async SDK clients in async contexts when possible.
    """

    async def _runner() -> T:
        return await coro

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(_runner)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(anyio.run, _runner).result()


__all__ = ["run_coro_sync"]
