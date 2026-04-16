from __future__ import annotations

import asyncio
import threading
import uuid
from functools import cached_property
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from programasweights.runtime_llamacpp import PawFunction


# compile_and_load is synchronous; use a thread lock so prepare() is safe under
# asyncio.run() (no running loop) and under create_task() (running loop).
paw_lock = threading.Lock()


class AsyncPawFunction:
    """
    A virtual programasweights PawFunction that begins compiling asynchronously
    when initialized.
    Compilation runs in a worker thread so the asyncio event loop can serve
    other traffic while compile/download/load runs.
    Calls to the function await preparation then run inference; inference itself
    is synchronous and may still occupy the event loop for each call.
    The entire compilation and loading process requires mutex to prevent
    overwhelming the compilation service.
    """

    def __init__(self, spec: str):
        self._spec = spec
        self._fn: PawFunction | None = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.prepare())
        else:
            asyncio.create_task(self.prepare())

    async def prepare(self) -> PawFunction:
        if self._fn is not None:
            return self._fn

        def _compile_and_load() -> None:
            with paw_lock:
                if self._fn is None:
                    import programasweights as paw

                    self._fn = paw.compile_and_load(self._spec)

        await asyncio.to_thread(_compile_and_load)
        assert self._fn is not None
        return self._fn

    async def __call__(self, text: str) -> str:
        fn = await self.prepare()
        return fn(text)


class CustomPolicy(BaseModel):
    """Custom policy for the safety detector that can evaluate arbitrary rules
    defined in natural language.

    Usage:

    ```python
    >>> foo_policy = CustomPolicy(
    ...     name="Foo",
    ...     rule='The text must contain the word "foo".',
    ... )
    ```

    By default, this class compiles a PawFunction to evaluate the policy lazily
    the first time this class is called:

    ```python
    >>> await foo_policy("foo bar")
    Compiling...
    Compiled: 123abc
    Downloading program 123abc...
    Waiting for program to be ready...
    Loading interpreter...
    Ready.
    True
    >>> await foo_policy("boo far")
    False
    ```

    To reduce first-call latency, use .eager() to begin compiling in the
    background in a resource-safe manner:

    ```python
    >>> foo_policy.eager()
    Compiling...
    Compiled: 123abc
    Downloading program 123abc...
    Loading interpreter...
    Ready.
    >>> await foo_policy("foo bar")
    True
    >>> await foo_policy("boo far")
    True
    ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    rule: str
    enabled: bool = True

    @cached_property
    def _paw(self) -> AsyncPawFunction:
        """
        The virtual PawFunction that will be used to evaluate the policy.
        This function will begin compiling from the first time this cached
        property is accessed.
        """
        return AsyncPawFunction(
            "Determine whether the text follows the rule:\n\n"
            + self.rule
            + '\n\nYour answer must be a single character, either "Y" or "N".'
        )

    def eager(self) -> Self:
        """
        Begin eagerly compiling the PawFunction in the background.
        """
        _ = self._paw
        return self

    async def __call__(self, text: str) -> bool:
        return (await self._paw(text)).strip().upper().startswith("Y")


__all__ = [
    "CustomPolicy",
]
