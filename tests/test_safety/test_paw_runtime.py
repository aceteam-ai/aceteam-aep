"""Shared PAW inference is exercised with local bundles and a fake backend only."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aceteam_aep.safety.paw_runtime import (
    InvalidExpertOutput,
    PawBundle,
    SharedInterpreter,
)


class FakeBackend:
    def __init__(self) -> None:
        self.models: list[Path] = []
        self.contexts: list[dict] = []
        self.binds: list[tuple[int, Path]] = []
        self.freed: list[int] = []
        self.prefix_loads: list[Path] = []
        self.calls: list[tuple[int, int, str]] = []
        self.token = "Y"
        self.logits = (0.123456789123, -0.876543210987)
        self.cache: set[Path] = set()

    def load_model(self, path: Path) -> object:
        self.models.append(path)
        return {"path": path}

    def new_context(self, model: object, n_ctx: int) -> object:
        context = {"model": model, "n_ctx": n_ctx, "id": len(self.contexts)}
        self.contexts.append(context)
        return context

    def bind_adapter(self, model: object, context: object, adapter: Path) -> None:
        assert isinstance(context, dict)
        assert context["model"] is model
        context["adapter"] = adapter
        self.binds.append((context["id"], adapter))

    def prepare_prefix(self, context: object, prefix: str, cache: Path) -> None:
        assert isinstance(context, dict)
        assert "adapter" in context
        context["prefix"] = prefix
        if cache in self.cache:
            self.prefix_loads.append(cache)
        self.cache.add(cache)

    def infer(self, context: object, text_with_suffix: str) -> tuple[str, float, float]:
        assert isinstance(context, dict)
        self.calls.append((context["id"], threading.get_ident(), text_with_suffix))
        return self.token, *self.logits

    def free_context(self, context: object) -> None:
        assert isinstance(context, dict)
        self.freed.append(context["id"])


def bundle(tmp_path: Path, name: str, *, interpreter: str = "base-v1") -> PawBundle:
    root = tmp_path / name
    root.mkdir()
    (root / "meta.json").write_text(
        json.dumps(
            {
                "program_id": name,
                "interpreter": interpreter,
                "spec_sha256": "a" * 64,
            }
        )
    )
    (root / "prompt_template.txt").write_text("Check this: {input} Reply Y or N.")
    (root / "adapter.gguf").write_bytes(b"fake adapter")
    return PawBundle.load(root)


def runtime(tmp_path: Path, backend: FakeBackend, *, limit: int = 8) -> SharedInterpreter:
    return SharedInterpreter(
        tmp_path / "base.gguf", "base-v1", backend=backend, max_resident_experts=limit
    )


def test_bundle_requires_files_metadata_and_one_placeholder(tmp_path: Path) -> None:
    loaded = bundle(tmp_path, "first")
    assert loaded.program_id == "first"
    assert loaded.interpreter == "base-v1"
    assert loaded.spec_sha256 == "a" * 64
    assert (loaded.prefix_text, loaded.suffix_text) == ("Check this: ", " Reply Y or N.")
    (loaded.directory / "adapter.gguf").unlink()
    with pytest.raises(ValueError, match="missing adapter.gguf"):
        PawBundle.load(loaded.directory)
    (loaded.directory / "adapter.gguf").write_bytes(b"fake adapter")
    (loaded.directory / "prompt_template.txt").write_text("No placeholder")
    with pytest.raises(ValueError, match="one input placeholder"):
        PawBundle.load(loaded.directory)


def test_shared_model_loaded_once_for_three_bundles(tmp_path: Path) -> None:
    backend = FakeBackend()
    shared = runtime(tmp_path, backend)
    for name in ("one", "two", "three"):
        shared.add_expert(name, bundle(tmp_path, name))
        shared.score(name, "sample")
    assert len(backend.models) == 1
    assert len(backend.contexts) == 3
    shared.close()


def test_base_model_registry_is_shared_across_runtime_instances(tmp_path: Path) -> None:
    backend = FakeBackend()
    first = runtime(tmp_path, backend)
    second = runtime(tmp_path, backend)
    first.add_expert("one", bundle(tmp_path, "one"))
    second.add_expert("two", bundle(tmp_path, "two"))
    first.score("one", "sample")
    second.score("two", "sample")
    assert len(backend.models) == 1
    assert backend.contexts[0]["model"] is backend.contexts[1]["model"]
    first.close()
    second.close()


def test_adapter_bound_per_context_only(tmp_path: Path) -> None:
    backend = FakeBackend()
    shared = runtime(tmp_path, backend)
    for name in ("one", "two", "three"):
        shared.add_expert(name, bundle(tmp_path, name))
        shared.score(name, name)
    assert len({context_id for context_id, _ in backend.binds}) == 3
    assert [path.parent.name for _, path in backend.binds] == ["one", "two", "three"]
    assert [context["adapter"].parent.name for context in backend.contexts] == [
        "one",
        "two",
        "three",
    ]
    shared.close()


def test_interpreter_mismatch_refused(tmp_path: Path) -> None:
    backend = FakeBackend()
    shared = runtime(tmp_path, backend)
    with pytest.raises(ValueError, match="does not match"):
        shared.add_expert("wrong", bundle(tmp_path, "wrong", interpreter="other"))
    assert backend.contexts == []
    shared.close()


def test_non_yn_first_token_raises(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.token = "maybe"
    shared = runtime(tmp_path, backend)
    shared.add_expert("one", bundle(tmp_path, "one"))
    with pytest.raises(InvalidExpertOutput, match="Y or N"):
        shared.score("one", "sample")
    shared.close()


def test_eviction_readmit_same_z(tmp_path: Path) -> None:
    backend = FakeBackend()
    shared = runtime(tmp_path, backend, limit=2)
    for name in ("one", "two", "three"):
        shared.add_expert(name, bundle(tmp_path, name))
    first = shared.score("one", "same input")
    shared.score("two", "sample")
    shared.score("three", "sample")
    second = shared.score("one", "same input")
    assert first.z == second.z
    assert backend.freed == [0, 1]
    assert backend.prefix_loads == [tmp_path / "one" / "prefix_kv_cache.bin"]
    assert len(backend.contexts) == 4
    shared.close()


def test_one_busy_expert_does_not_block_another_context(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBackend(FakeBackend):
        def infer(self, context: object, text_with_suffix: str) -> tuple[str, float, float]:
            assert isinstance(context, dict)
            if context["adapter"].parent.name == "one":
                started.set()
                assert release.wait(timeout=5)
            return super().infer(context, text_with_suffix)

    backend = BlockingBackend()
    shared = runtime(tmp_path, backend, limit=2)
    shared.add_expert("one", bundle(tmp_path, "one"))
    shared.add_expert("two", bundle(tmp_path, "two"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(shared.score, "one", "first")
        try:
            assert started.wait(timeout=2)
            second = pool.submit(shared.score, "two", "second")
            assert second.result(timeout=2).token == "Y"
        finally:
            release.set()
        assert first.result(timeout=2).token == "Y"
    shared.close()


def test_score_runs_off_event_loop(tmp_path: Path) -> None:
    backend = FakeBackend()
    shared = runtime(tmp_path, backend)
    shared.add_expert("one", bundle(tmp_path, "one"))

    async def exercise() -> None:
        loop_thread = threading.get_ident()
        await shared.score_async("one", "sample")
        assert backend.calls[0][1] != loop_thread
        with pytest.raises(RuntimeError, match="worker thread"):
            shared.score("one", "sample")
        assert len(backend.calls) == 1

    with ThreadPoolExecutor(max_workers=1) as pool:
        loop = asyncio.new_event_loop()
        try:
            loop.set_default_executor(pool)
            loop.run_until_complete(exercise())
        finally:
            loop.close()
    shared.close()


def test_logit_diff_unrounded(tmp_path: Path) -> None:
    backend = FakeBackend()
    shared = runtime(tmp_path, backend)
    shared.add_expert("one", bundle(tmp_path, "one"))
    raw = shared.score("one", "input")
    assert raw.logit_y == 0.123456789123
    assert raw.logit_n == -0.876543210987
    assert raw.z == backend.logits[0] - backend.logits[1]
    assert raw.z != round(raw.z, 4)
    assert backend.calls[0][2] == "input Reply Y or N."
    shared.close()
