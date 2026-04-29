from __future__ import annotations

import asyncio
import logging
import math
import threading
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, Field

from .base import SafetyDetector, SafetySignal

if TYPE_CHECKING:
    from programasweights.runtime_llamacpp import PawFunction

log = logging.getLogger(__name__)


@dataclass
class PawResult:
    """Result from a PAW policy check with optional confidence."""

    compliant: bool
    p_compliant: float | None = None
    chunk_results: list[tuple[bool, float | None]] | None = None


def _extract_yn_probability(fn: PawFunction) -> float | None:
    """Extract P(Y) vs P(N) from llama.cpp logits after eval.

    Must be called after fn._llm.eval() and before sample().
    Returns the softmax probability of the Y token, or None on failure.
    """
    try:
        import ctypes

        import llama_cpp as _llama

        ctx = fn._llm.ctx
        n_vocab = fn._llm.n_vocab()

        logits_ptr = _llama.llama_get_logits_ith(ctx, -1)
        if not logits_ptr:
            return None

        y_ids = fn._llm.tokenize(b"Y", add_bos=False, special=False)
        n_ids = fn._llm.tokenize(b"N", add_bos=False, special=False)
        if not y_ids or not n_ids:
            return None

        logit_y = float(logits_ptr[y_ids[0]])
        logit_n = float(logits_ptr[n_ids[0]])

        max_l = max(logit_y, logit_n)
        exp_y = math.exp(logit_y - max_l)
        exp_n = math.exp(logit_n - max_l)
        return exp_y / (exp_y + exp_n)
    except Exception:
        log.debug("Failed to extract Y/N logprobs from PAW", exc_info=True)
        return None


def _paw_call_with_logprobs(
    fn: PawFunction, text: str
) -> tuple[str, float | None]:
    """Replicate PawFunction.__call__ with logit extraction before sampling."""
    fn._llm.n_tokens = fn._n_prefix

    input_with_suffix = text + fn._suffix_text
    input_tokens = fn._llm.tokenize(
        input_with_suffix.encode("utf-8"),
        add_bos=False,
        special=True,
    )

    tokens_used = fn._n_prefix + len(input_tokens)
    remaining = fn._n_ctx - tokens_used
    if remaining <= 0:
        raise ValueError(
            f"Input too long: {tokens_used} tokens used "
            f"(prefix={fn._n_prefix}, input={len(input_tokens)}), "
            f"context window={fn._n_ctx}."
        )

    fn._llm.eval(input_tokens)

    p_compliant = _extract_yn_probability(fn)

    output_tokens: list[int] = []
    for _ in range(remaining):
        token = fn._llm.sample(temp=0)
        if token == fn._llm.token_eos():
            break
        output_tokens.append(token)
        fn._llm.eval([token])

    output_bytes = fn._llm.detokenize(output_tokens)
    return output_bytes.decode("utf-8", errors="replace").strip(), p_compliant

CustomPolicyAppliesTo = Literal["input", "output", "both"]
CustomPolicySeverity = Literal["low", "medium", "high"]

# Llama backends enforce a small context window; chunk user text so each PAW
# call stays within limits (see programasweights ValueError on token overflow).
_CUSTOM_POLICY_CHUNK_CHARS = 4096


def _iter_policy_text_chunks(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into fixed-size slices for model calls; empty → one slice."""
    if not text:
        return [""]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _retrieve_eager_task_exception(task: asyncio.Task[object]) -> None:
    """Eager ``create_task``; retrieve the result so asyncio doesn't warn."""

    if task.cancelled():
        return
    task.exception()


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
        self._prepare_task: asyncio.Task[PawFunction] | None = None
        self._prepare_lock = asyncio.Lock()
        self._eager_task: asyncio.Task[PawFunction] | None = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.prepare())
        else:
            # Single entry point: prepare() creates the compile task and awaits it.
            self._eager_task = asyncio.create_task(self.prepare())
            self._eager_task.add_done_callback(_retrieve_eager_task_exception)

    async def _prepare_impl(self) -> PawFunction:
        def _compile_and_load() -> None:
            with paw_lock:
                if self._fn is None:
                    import programasweights as paw

                    self._fn = paw.compile_and_load(self._spec)

        await asyncio.to_thread(_compile_and_load)
        assert self._fn is not None
        return self._fn

    async def prepare(self) -> PawFunction:
        # Fast path when compilation already finished.
        if self._fn is not None:
            return self._fn
        # Exactly one in-flight task for this instance: concurrent awaiters share
        # it instead of each scheduling asyncio.to_thread via _prepare_impl.
        async with self._prepare_lock:
            # Double check after acquiring the lock.
            if self._fn is not None:
                return self._fn
            if self._prepare_task is None:
                self._prepare_task = asyncio.create_task(self._prepare_impl())
        assert self._prepare_task is not None
        try:
            return await self._prepare_task
        except BaseException:
            async with self._prepare_lock:
                self._prepare_task = None
            raise

    async def __call__(self, text: str) -> str:
        fn = await self.prepare()
        return fn(text)

    async def call_with_confidence(self, text: str) -> tuple[str, float | None]:
        """Run PAW inference and extract Y/N token probability.

        Returns (output_text, p_compliant) where p_compliant is the softmax
        probability of the Y token, or None if logit extraction failed.
        Falls back to normal __call__ if logprob extraction is unavailable.
        """
        fn = await self.prepare()
        try:
            return _paw_call_with_logprobs(fn, text)
        except Exception:
            log.debug("Logprob path failed, falling back to standard call", exc_info=True)
            output = fn(text)
            return output, None


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
    False
    ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    rule: str
    enabled: bool = True
    applies_to: CustomPolicyAppliesTo = "both"
    """Whether the rule is evaluated on user messages, assistant output, or both."""

    severity: CustomPolicySeverity = "high"
    """Maps to :attr:`SafetySignal.severity` when this policy is violated."""

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
            + "\n\nYour answer must be a single character, either "
            + '"Y" (yes, the text complies with the rule) '
            + 'or "N" (no, the text violates the rule).'
        )

    def eager(self) -> Self:
        """
        Begin eagerly compiling the PawFunction in the background, if enabled.

        For disabled policies, this method is a no-op.
        """
        if self.enabled:
            _ = self._paw
        return self

    async def __call__(self, text: str) -> bool:
        for piece in _iter_policy_text_chunks(text, _CUSTOM_POLICY_CHUNK_CHARS):
            raw = await self._paw(piece)
            if not raw.strip().upper().startswith("Y"):
                return False
        return True

    async def check_with_confidence(self, text: str) -> PawResult:
        """Evaluate policy with per-chunk confidence scores.

        Returns a PawResult with overall compliance and per-chunk P(compliant).
        """
        chunks = _iter_policy_text_chunks(text, _CUSTOM_POLICY_CHUNK_CHARS)
        chunk_results: list[tuple[bool, float | None]] = []
        overall_compliant = True
        min_p: float | None = None

        for piece in chunks:
            raw, p_compliant = await self._paw.call_with_confidence(piece)
            is_compliant = raw.strip().upper().startswith("Y")
            chunk_results.append((is_compliant, p_compliant))
            if not is_compliant:
                overall_compliant = False
            if p_compliant is not None:
                min_p = p_compliant if min_p is None else min(min_p, p_compliant)

        return PawResult(
            compliant=overall_compliant,
            p_compliant=min_p,
            chunk_results=chunk_results if len(chunks) > 1 else None,
        )


def default_custom_policies() -> tuple[CustomPolicy, ...]:
    """Starter custom policies (all disabled until explicitly enabled).

    IDs are deterministic so dashboards and automation can refer to them
    stably across process restarts.
    """

    def _stable_id(slug: str) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"https://aceteam-aep.invalid/custom-policy/{slug}",
            )
        )

    return (
        CustomPolicy(
            id=_stable_id("english-only"),
            name="English only",
            rule=(
                "Messages must be primarily in English. "
                "Empty or whitespace-only text is compliant. "
                "This rule is violated if the substantive content of a message is not in English."
            ),
            enabled=False,
            applies_to="both",
            severity="low",
        ),
        CustomPolicy(
            id=_stable_id("ssn"),
            name="Social Security Number",
            rule=(
                "Must not contain a U.S. Social Security Number (SSN) in the format XXX-XX-XXXX. "
                "Empty or whitespace-only text is compliant. "
                "This rule is violated if at least 1 SSN is found."
            ),
            enabled=False,
            applies_to="input",
            severity="high",
        ),
    )


class CustomPolicyStore:
    """Authoritative collection of custom policies keyed by id.

    Shared by the proxy HTTP API and :class:`CustomSafetyDetector` so CRUD and
    safety checks use the same data without passing raw dicts around.
    """

    def __init__(self, initial: Iterable[CustomPolicy] = ()) -> None:
        self._by_id: dict[str, CustomPolicy] = {}
        for policy in initial:
            if policy.id in self._by_id:
                raise ValueError(f"Policy ID {policy.id} already exists")
            self._by_id[policy.id] = policy

    def upsert(self, policy: CustomPolicy) -> None:
        self._by_id[policy.id] = policy

    def delete(self, policy_id: str) -> None:
        del self._by_id[policy_id]

    def get(self, policy_id: str) -> CustomPolicy | None:
        return self._by_id.get(policy_id)

    def all(self) -> list[CustomPolicy]:
        return list(self._by_id.values())


class CustomSafetyDetector(SafetyDetector):
    """Custom safety detector that can evaluate arbitrary rules defined in
    natural language.

    Policies can be added and removed dynamically via the :class:`CustomPolicyStore`.
    """

    name = "custom_safety"

    def __init__(self, store: CustomPolicyStore) -> None:
        self._store = store

    @property
    def store(self) -> CustomPolicyStore:
        return self._store

    async def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> Sequence[SafetySignal]:
        def _sources_for_policy(p: CustomPolicy) -> tuple[tuple[str, str], ...]:
            if p.applies_to == "input":
                return (("input", input_text),)
            if p.applies_to == "output":
                return (("output", output_text),)
            return (("output", output_text), ("input", input_text))

        async def _check_one(policy: CustomPolicy) -> Sequence[SafetySignal]:
            signals: list[SafetySignal] = []
            try:
                for source, text in _sources_for_policy(policy):
                    result = await policy.check_with_confidence(text)
                    if not result.compliant:
                        p_unsafe = (
                            round(1.0 - result.p_compliant, 4)
                            if result.p_compliant is not None
                            else None
                        )
                        detail = f"{source} violates {policy.name}"
                        if result.p_compliant is not None:
                            detail += f" (p_unsafe={p_unsafe})"
                        if result.chunk_results and len(result.chunk_results) > 1:
                            failing = [
                                i
                                for i, (ok, _) in enumerate(result.chunk_results)
                                if not ok
                            ]
                            detail += f" [chunks {failing}]"
                        signals.append(
                            SafetySignal(
                                signal_type="custom_safety",
                                severity=policy.severity,
                                call_id=call_id,
                                detail=detail,
                                score=p_unsafe,
                            )
                        )
            except Exception:
                log.warning("Custom safety check failed for %s", policy.name, exc_info=True)
            return signals

        signals: list[SafetySignal] = []

        chunks = await asyncio.gather(
            *(_check_one(policy) for policy in self._store.all() if policy.enabled)
        )
        for chunk in chunks:
            signals.extend(chunk)
        return signals


__all__ = [
    "CustomPolicy",
    "CustomPolicyAppliesTo",
    "CustomPolicySeverity",
    "CustomPolicyStore",
    "CustomSafetyDetector",
    "PawResult",
    "default_custom_policies",
]
