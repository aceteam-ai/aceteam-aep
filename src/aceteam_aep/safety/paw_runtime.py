"""Owned runtime for PAW bundles, with the MIT programasweights bundle layout.

Only local files are read. Compilation, bundle downloads, model downloads, and
the native llama.cpp binding belong to later slices. A backend protocol lets this
slice prove shared model and context ownership with fakes.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# programasweights runtime_llamacpp.INPUT_PLACEHOLDER in the locked SDK.
INPUT_PLACEHOLDER = "{INPUT_PLACEHOLDER}"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class InvalidExpertOutput(ValueError):
    """The interpreter did not produce a Y or N first token."""


@dataclass(frozen=True)
class PawBundle:
    directory: Path
    program_id: str
    interpreter: str
    spec_sha256: str
    prefix_text: str
    suffix_text: str
    adapter_path: Path
    prefix_cache_path: Path

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        program_id: str | None = None,
        interpreter: str | None = None,
        spec_sha256: str | None = None,
    ) -> PawBundle:
        root = Path(directory).resolve()
        required = (root / "meta.json", root / "prompt_template.txt", root / "adapter.gguf")
        for path in required:
            if not path.is_file():
                raise ValueError(f"PAW bundle is missing {path.name}")
        meta = json.loads(required[0].read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("PAW bundle metadata must be an object")

        def required_field(key: str, supplied: str | None) -> str:
            embedded = meta.get(key)
            if embedded is not None and (not isinstance(embedded, str) or not embedded):
                raise ValueError(f"PAW bundle metadata has invalid {key}")
            if embedded is not None and supplied is not None and embedded != supplied:
                raise ValueError(f"PAW bundle metadata conflicts with supplied {key}")
            value = embedded if embedded is not None else supplied
            if not isinstance(value, str) or not value:
                raise ValueError(f"PAW bundle metadata is missing {key}")
            return value

        program_id = required_field("program_id", program_id)
        interpreter = required_field("interpreter", interpreter)
        spec_sha256 = required_field("spec_sha256", spec_sha256)
        if not _SHA256.fullmatch(spec_sha256):
            raise ValueError("PAW bundle spec_sha256 must be a SHA-256 hex digest")
        template = required[1].read_text(encoding="utf-8")
        if template.count(INPUT_PLACEHOLDER) != 1:
            raise ValueError("PAW prompt template needs one input placeholder")
        prefix, suffix = template.split(INPUT_PLACEHOLDER)
        return cls(
            directory=root,
            program_id=program_id,
            interpreter=interpreter,
            spec_sha256=spec_sha256.lower(),
            prefix_text=prefix,
            suffix_text=suffix,
            adapter_path=required[2],
            prefix_cache_path=root / "prefix_kv_cache.bin",
        )


@dataclass(frozen=True)
class ExpertRaw:
    token: str
    logit_y: float
    logit_n: float
    z: float


class InterpreterBackend(Protocol):
    """Small llama.cpp boundary. A context owns one adapter and one prefix."""

    def load_model(self, path: Path) -> object: ...
    def new_context(self, model: object, n_ctx: int) -> object: ...
    def bind_adapter(self, model: object, context: object, adapter: Path) -> None: ...
    def prepare_prefix(self, context: object, prefix: str, cache: Path) -> None: ...
    def infer(self, context: object, text_with_suffix: str) -> tuple[str, float, float]: ...
    def free_context(self, context: object) -> None: ...


@dataclass
class _Resident:
    context: object | None
    busy: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class SharedInterpreter:
    """One process-wide base model with bounded, adapter-specific contexts."""

    _models: dict[tuple[int, Path], Future[object]] = {}
    _backend_refs: dict[int, InterpreterBackend] = {}
    _models_lock = threading.Lock()

    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        *,
        backend: InterpreterBackend,
        max_resident_experts: int = 8,
        n_ctx: int = 2048,
    ) -> None:
        if not model_id:
            raise ValueError("interpreter model id is required")
        if max_resident_experts < 1 or n_ctx < 1:
            raise ValueError("resident limit and context size must be positive")
        self.model_path = Path(model_path).resolve()
        self.model_id = model_id
        self.backend = backend
        self.max_resident_experts = max_resident_experts
        self.n_ctx = n_ctx
        key = (id(backend), self.model_path)
        with self._models_lock:
            self._backend_refs[id(backend)] = backend
            pending = self._models.get(key)
            owns_load = pending is None
            if pending is None:
                pending = Future()
                self._models[key] = pending
        if owns_load:
            try:
                pending.set_result(backend.load_model(self.model_path))
            except BaseException as exc:
                pending.set_exception(exc)
                with self._models_lock:
                    if self._models.get(key) is pending:
                        del self._models[key]
                raise
        self._model = pending.result()
        self._bundles: dict[str, PawBundle] = {}
        self._resident: OrderedDict[str, _Resident] = OrderedDict()
        self._condition = threading.Condition()
        self._closing = False
        self._closed = False

    def add_expert(self, expert_id: str, bundle: PawBundle) -> None:
        if not expert_id:
            raise ValueError("expert id is required")
        if bundle.interpreter != self.model_id:
            raise ValueError("PAW bundle interpreter does not match the loaded model")
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("interpreter is closed")
            if expert_id in self._bundles:
                raise ValueError("duplicate expert id")
            self._bundles[expert_id] = bundle

    def _acquire(self, expert_id: str) -> _Resident:
        old_context: object | None = None
        with self._condition:
            while True:
                if self._closing or self._closed:
                    raise RuntimeError("interpreter is closed")
                bundle = self._bundles.get(expert_id)
                if bundle is None:
                    raise KeyError(expert_id)
                resident = self._resident.get(expert_id)
                if resident is not None:
                    if resident.busy:
                        self._condition.wait()
                        continue
                    resident.busy = True
                    self._resident.move_to_end(expert_id)
                    return resident
                if len(self._resident) >= self.max_resident_experts:
                    idle = next(
                        ((key, value) for key, value in self._resident.items() if not value.busy),
                        None,
                    )
                    if idle is None:
                        self._condition.wait()
                        continue
                    old_id, old = idle
                    del self._resident[old_id]
                    old_context = old.context
                # Reserve capacity while the slow backend work runs outside
                # the Condition, so idle experts remain available.
                resident = _Resident(context=None, busy=True)
                self._resident[expert_id] = resident
                break

        try:
            if old_context is not None:
                self.backend.free_context(old_context)
            context = self.backend.new_context(self._model, self.n_ctx)
            try:
                self.backend.bind_adapter(self._model, context, bundle.adapter_path)
                self.backend.prepare_prefix(context, bundle.prefix_text, bundle.prefix_cache_path)
            except BaseException:
                self.backend.free_context(context)
                raise
        except BaseException:
            with self._condition:
                if self._resident.get(expert_id) is resident:
                    del self._resident[expert_id]
                self._condition.notify_all()
            raise
        with self._condition:
            resident.context = context
            self._condition.notify_all()
        return resident

    def score(self, expert_id: str, text: str) -> ExpertRaw:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("score must run in a worker thread, not on an event loop")
        resident = self._acquire(expert_id)
        try:
            bundle = self._bundles[expert_id]
            assert resident.context is not None
            with resident.lock:
                token, logit_y, logit_n = self.backend.infer(
                    resident.context, text + bundle.suffix_text
                )
            if token not in {"Y", "N"}:
                raise InvalidExpertOutput("expert first token must be Y or N")
            if not math.isfinite(logit_y) or not math.isfinite(logit_n):
                raise InvalidExpertOutput("expert logits must be finite")
            return ExpertRaw(token, logit_y, logit_n, logit_y - logit_n)
        finally:
            with self._condition:
                resident.busy = False
                self._condition.notify_all()

    async def score_async(self, expert_id: str, text: str) -> ExpertRaw:
        return await asyncio.to_thread(self.score, expert_id, text)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._condition.wait()
                return
            self._closing = True
            self._condition.notify_all()
            while any(value.busy for value in self._resident.values()):
                self._condition.wait()
            contexts = [value.context for value in self._resident.values()]
            self._resident.clear()
            self._closed = True
            self._condition.notify_all()
        for context in contexts:
            assert context is not None
            self.backend.free_context(context)
