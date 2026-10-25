"""Owned runtime for PAW bundles, with the MIT programasweights bundle layout.

Only local files are read. Compilation, bundle downloads, and model downloads are
outside this module. The llama.cpp binding is imported lazily when a native backend
is constructed; tests inject a fake backend and never load a model.
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib
import json
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_PLACEHOLDERS = ("{{input}}", "{input}")
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
    def load(cls, directory: str | Path) -> PawBundle:
        root = Path(directory).resolve()
        required = (root / "meta.json", root / "prompt_template.txt", root / "adapter.gguf")
        for path in required:
            if not path.is_file():
                raise ValueError(f"PAW bundle is missing {path.name}")
        meta = json.loads(required[0].read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("PAW bundle metadata must be an object")
        for key in ("program_id", "interpreter", "spec_sha256"):
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f"PAW bundle metadata is missing {key}")
        if not _SHA256.fullmatch(meta["spec_sha256"]):
            raise ValueError("PAW bundle spec_sha256 must be a SHA-256 hex digest")
        template = required[1].read_text(encoding="utf-8")
        placeholder = next((value for value in _PLACEHOLDERS if value in template), None)
        if placeholder is None or template.count(placeholder) != 1:
            raise ValueError("PAW prompt template needs one input placeholder")
        prefix, suffix = template.split(placeholder)
        return cls(
            directory=root,
            program_id=meta["program_id"],
            interpreter=meta["interpreter"],
            spec_sha256=meta["spec_sha256"].lower(),
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
class _NativeContext:
    model: object
    pointer: object
    adapter: object | None = None
    prefix_tokens: tuple[int, ...] = ()
    prefix_cache: Path | None = None


class LlamaCppBackend:
    """Low-level llama.cpp adapter. No PAW SDK runtime objects are constructed."""

    def __init__(self) -> None:
        try:
            self.llama = importlib.import_module("llama_cpp")
        except ImportError as exc:
            raise RuntimeError("Install the custom-policies extra for llama.cpp inference") from exc
        self.llama.llama_backend_init()

    def load_model(self, path: Path) -> object:
        if not path.is_file():
            raise FileNotFoundError(path)
        llama = self.llama
        loader = getattr(llama, "llama_model_load_from_file", None)
        if loader is None:
            loader = llama.llama_load_model_from_file
        model = loader(str(path).encode(), llama.llama_model_default_params())
        if not model:
            raise RuntimeError("llama.cpp could not load the interpreter model")
        return model

    def new_context(self, model: object, n_ctx: int) -> object:
        params = self.llama.llama_context_default_params()
        params.n_ctx = n_ctx
        pointer = self.llama.llama_init_from_model(model, params)
        if not pointer:
            raise RuntimeError("llama.cpp could not initialize an expert context")
        return _NativeContext(model=model, pointer=pointer)

    def bind_adapter(self, model: object, context: object, adapter: Path) -> None:
        native = self._context(context)
        handle = self.llama.llama_adapter_lora_init(model, str(adapter).encode())
        if not handle:
            raise RuntimeError("llama.cpp could not load the expert adapter")
        adapters = ctypes.pointer(handle)
        scales = (ctypes.c_float * 1)(1.0)
        if self.llama.llama_set_adapters_lora(native.pointer, adapters, scales, 1) != 0:
            self.llama.llama_adapter_lora_free(handle)
            raise RuntimeError("llama.cpp could not bind the expert adapter")
        native.adapter = handle

    def prepare_prefix(self, context: object, prefix: str, cache: Path) -> None:
        native = self._context(context)
        native.prefix_tokens = tuple(self._tokenize(native.model, prefix, add_special=True))
        native.prefix_cache = cache
        self._reset_to_prefix(native)

    def infer(self, context: object, text_with_suffix: str) -> tuple[str, float, float]:
        native = self._context(context)
        self._reset_to_prefix(native)
        tokens = self._tokenize(native.model, text_with_suffix, add_special=False)
        if not tokens:
            raise InvalidExpertOutput("expert input produced no tokens")
        n_ctx = self.llama.llama_n_ctx(native.pointer)
        if len(native.prefix_tokens) + len(tokens) >= n_ctx:
            raise ValueError("expert input exceeds the interpreter context")
        self._decode(native.pointer, tokens, start_pos=len(native.prefix_tokens))
        y = self._single_token(native.model, "Y")
        n = self._single_token(native.model, "N")
        logits = self.llama.llama_get_logits_ith(native.pointer, -1)
        if not logits:
            raise RuntimeError("llama.cpp returned no first-token logits")
        size = self.llama.llama_n_vocab(native.model)
        sampled = max(range(size), key=lambda token_id: float(logits[token_id]))
        token = "Y" if sampled == y else "N" if sampled == n else "?"
        return token, float(logits[y]), float(logits[n])

    def free_context(self, context: object) -> None:
        native = self._context(context)
        if native.adapter is not None:
            self.llama.llama_adapter_lora_free(native.adapter)
        self.llama.llama_free(native.pointer)

    @staticmethod
    def _context(context: object) -> _NativeContext:
        if not isinstance(context, _NativeContext):
            raise TypeError("invalid native interpreter context")
        return context

    def _tokenize(self, model: object, text: str, *, add_special: bool) -> list[int]:
        encoded = text.encode("utf-8")
        capacity = max(len(encoded) + 8, 8)
        tokens = (ctypes.c_int32 * capacity)()
        count = self.llama.llama_tokenize(
            model, encoded, len(encoded), tokens, capacity, add_special, True
        )
        if count < 0:
            capacity = -count
            tokens = (ctypes.c_int32 * capacity)()
            count = self.llama.llama_tokenize(
                model, encoded, len(encoded), tokens, capacity, add_special, True
            )
        if count < 0:
            raise RuntimeError("llama.cpp tokenization failed")
        return list(tokens[:count])

    def _single_token(self, model: object, value: str) -> int:
        tokens = self._tokenize(model, value, add_special=False)
        if len(tokens) != 1:
            raise RuntimeError(f"interpreter requires a single token for {value}")
        return tokens[0]

    def _decode(
        self, pointer: object, tokens: list[int] | tuple[int, ...], *, start_pos: int
    ) -> None:
        for offset in range(0, len(tokens), 512):
            chunk = tokens[offset : offset + 512]
            batch = self.llama.llama_batch_init(len(chunk), 0, 1)
            try:
                batch.n_tokens = len(chunk)
                for index, token in enumerate(chunk):
                    batch.token[index] = token
                    batch.pos[index] = start_pos + offset + index
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = 0
                    batch.logits[index] = index == len(chunk) - 1
                if self.llama.llama_decode(pointer, batch) != 0:
                    raise RuntimeError("llama.cpp decode failed")
            finally:
                self.llama.llama_batch_free(batch)

    def _reset_to_prefix(self, native: _NativeContext) -> None:
        cache = native.prefix_cache
        if cache is not None and cache.is_file():
            capacity = max(self.llama.llama_n_ctx(native.pointer), 1)
            tokens = (ctypes.c_int32 * capacity)()
            count = ctypes.c_size_t()
            loaded = self.llama.llama_state_load_file(
                native.pointer, str(cache).encode(), tokens, capacity, ctypes.byref(count)
            )
            if loaded and tuple(tokens[: count.value]) == native.prefix_tokens:
                return
        self.llama.llama_kv_cache_clear(native.pointer)
        self._decode(native.pointer, native.prefix_tokens, start_pos=0)
        if cache is not None and not cache.exists() and native.prefix_tokens:
            tokens = (ctypes.c_int32 * len(native.prefix_tokens))(*native.prefix_tokens)
            self.llama.llama_state_save_file(
                native.pointer, str(cache).encode(), tokens, len(native.prefix_tokens)
            )


@dataclass
class _Resident:
    context: object
    busy: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class SharedInterpreter:
    """One process-wide base model with bounded, adapter-specific contexts."""

    _models: dict[tuple[int, Path], object] = {}
    _backend_refs: dict[int, InterpreterBackend] = {}
    _models_lock = threading.Lock()
    _default_backend: LlamaCppBackend | None = None

    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        *,
        backend: InterpreterBackend | None = None,
        max_resident_experts: int = 8,
        n_ctx: int = 2048,
    ) -> None:
        if not model_id:
            raise ValueError("interpreter model id is required")
        if max_resident_experts < 1 or n_ctx < 1:
            raise ValueError("resident limit and context size must be positive")
        self.model_path = Path(model_path).resolve()
        self.model_id = model_id
        if backend is None:
            with self._models_lock:
                if self._default_backend is None:
                    type(self)._default_backend = LlamaCppBackend()
                backend = self._default_backend
        assert backend is not None
        self.backend = backend
        self.max_resident_experts = max_resident_experts
        self.n_ctx = n_ctx
        key = (id(backend), self.model_path)
        with self._models_lock:
            self._backend_refs[id(backend)] = backend
            if key not in self._models:
                self._models[key] = backend.load_model(self.model_path)
            self._model = self._models[key]
        self._bundles: dict[str, PawBundle] = {}
        self._resident: OrderedDict[str, _Resident] = OrderedDict()
        self._condition = threading.Condition()
        self._closed = False

    def add_expert(self, expert_id: str, bundle: PawBundle) -> None:
        if not expert_id:
            raise ValueError("expert id is required")
        if bundle.interpreter != self.model_id:
            raise ValueError("PAW bundle interpreter does not match the loaded model")
        with self._condition:
            if self._closed:
                raise RuntimeError("interpreter is closed")
            if expert_id in self._bundles:
                raise ValueError("duplicate expert id")
            self._bundles[expert_id] = bundle

    def _acquire(self, expert_id: str) -> _Resident:
        with self._condition:
            while True:
                if self._closed:
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
                    self.backend.free_context(old.context)
                context = self.backend.new_context(self._model, self.n_ctx)
                try:
                    self.backend.bind_adapter(self._model, context, bundle.adapter_path)
                    self.backend.prepare_prefix(
                        context, bundle.prefix_text, bundle.prefix_cache_path
                    )
                except BaseException:
                    self.backend.free_context(context)
                    raise
                resident = _Resident(context=context, busy=True)
                self._resident[expert_id] = resident
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
            while any(value.busy for value in self._resident.values()):
                self._condition.wait()
            self._closed = True
            for resident in self._resident.values():
                self.backend.free_context(resident.context)
            self._resident.clear()
            self._condition.notify_all()
