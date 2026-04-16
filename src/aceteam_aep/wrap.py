"""AEP wrapper — intercept existing SDK clients without changing your code.

Usage::

    import openai
    from aceteam_aep import wrap

    client = wrap(openai.OpenAI())

    # Use exactly as before — AEP tracks everything
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )

    # AEP data on the wrapped client
    print(client.aep.cost_usd)          # total spend so far
    print(client.aep.get_cost_tree())   # per-call breakdown
    print(client.aep.get_spans())       # execution trace
    print(client.aep.safety_signals)    # T&S flags raised this session
    print(client.aep.enforcement)       # latest PASS/FLAG/BLOCK decision

Supports:
- openai.OpenAI / openai.AsyncOpenAI
- anthropic.Anthropic / anthropic.AsyncAnthropic
- Any object with a .chat.completions.create() method (OpenAI-compatible)
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .costs import CostTracker
from .enforcement import (
    EnforcementDecision,
    EnforcementPolicy,
    build_detectors_from_policy,
    evaluate,
)
from .safety.base import DetectorRegistry, SafetySignal
from .safety.cost_anomaly import CostAnomalyDetector
from .spans import Span, SpanTracker
from .types import Usage
from .utils.async_bridge import run_coro_sync

log = logging.getLogger(__name__)


def _count_by(items: list[Any], attr: str) -> dict[str, int]:
    """Count items by attribute value."""
    counts: dict[str, int] = {}
    for item in items:
        key = getattr(item, attr, "") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _default_detectors() -> list[Any]:
    """Build the default detector list. Model-based detectors only if available."""
    from .safety.agent_threat import AgentThreatDetector

    detectors: list[Any] = [CostAnomalyDetector(), AgentThreatDetector()]
    try:
        from .safety.pii import PiiDetector

        detectors.append(PiiDetector())
    except Exception:
        pass
    try:
        from .safety.content import ContentSafetyDetector

        detectors.append(ContentSafetyDetector())
    except Exception:
        pass
    return detectors


class AepPreflightBlock(Exception):
    """Raised when pre-flight safety checks block a request."""

    def __init__(self, decision: EnforcementDecision, signals: Sequence[SafetySignal]) -> None:
        self.decision = decision
        self.signals = signals
        super().__init__(f"AEP pre-flight blocked: {decision.reason}")


def _snippet(text: str, n: int = 5) -> str:
    """First and last *n* lines, with a separator if truncated."""
    lines = text.strip().splitlines()
    if len(lines) <= n * 2:
        return text.strip()
    head = "\n".join(lines[:n])
    tail = "\n".join(lines[-n:])
    return f"{head}\n  ... ({len(lines) - n * 2} lines omitted) ...\n{tail}"


@dataclass
class AepSource:
    """A data source referenced during agent execution."""

    uri: str  # e.g. "file:///data/report.pdf", "https://api.example.com/v1/data"
    label: str = ""  # human-readable description
    kind: str = ""  # "file", "api", "database", "web", "tool"
    call_id: str = ""  # which LLM call used this source


@dataclass
class AepSession:
    """AEP session state attached to a wrapped client as ``client.aep``."""

    entity: str
    _cost_tracker: CostTracker = field(default_factory=CostTracker)
    _span_tracker: SpanTracker = field(default_factory=SpanTracker)
    _registry: DetectorRegistry = field(default_factory=DetectorRegistry)
    _safety_signals: list[SafetySignal] = field(default_factory=list)
    _call_costs: list[Decimal] = field(default_factory=list)
    _policy: EnforcementPolicy = field(default_factory=EnforcementPolicy)
    _last_enforcement: EnforcementDecision = field(
        default_factory=lambda: EnforcementDecision(action="pass")
    )
    _call_count: int = 0
    _verbose: bool = False
    _sources: list[AepSource] = field(default_factory=list)

    @property
    def cost_usd(self) -> Decimal:
        """Total compute cost for this session."""
        nodes = self._cost_tracker.get_cost_tree()
        return sum((n.compute_cost for n in nodes), Decimal("0"))

    def get_cost_tree(self) -> list[Any]:
        """Return the full hierarchical cost tree."""
        return self._cost_tracker.get_cost_tree()

    def get_spans(self) -> list[Span]:
        """Return all recorded spans."""
        return self._span_tracker.get_spans()

    @property
    def safety_signals(self) -> list[SafetySignal]:
        """All T&S flags raised this session."""
        return list(self._safety_signals)

    @property
    def enforcement(self) -> EnforcementDecision:
        """Latest enforcement decision."""
        return self._last_enforcement

    @property
    def call_count(self) -> int:
        """Number of LLM calls recorded."""
        return self._call_count

    # --- Provenance ---

    def add_source(
        self,
        uri: str,
        *,
        label: str = "",
        kind: str = "",
        call_id: str = "",
    ) -> None:
        """Record a data source used during agent execution.

        Call this whenever your agent reads a file, queries an API,
        accesses a database, or uses any external data source.

        Args:
            uri: Source identifier (file path, URL, database query, etc.)
            label: Human-readable description
            kind: Source type ("file", "api", "database", "web", "tool")
            call_id: Associate with a specific LLM call (optional)
        """
        self._sources.append(AepSource(uri=uri, label=label, kind=kind, call_id=call_id))

    @property
    def sources(self) -> list[AepSource]:
        """All data sources recorded this session."""
        return list(self._sources)

    def get_citations(self, call_id: str = "") -> list[AepSource]:
        """Get sources associated with a specific call, or all if no call_id."""
        if not call_id:
            return list(self._sources)
        return [s for s in self._sources if s.call_id == call_id]

    @property
    def provenance_summary(self) -> dict[str, Any]:
        """Summary of provenance data for this session."""
        return {
            "total_sources": len(self._sources),
            "sources_by_kind": _count_by(self._sources, "kind"),
            "sources": [
                {"uri": s.uri, "label": s.label, "kind": s.kind, "call_id": s.call_id}
                for s in self._sources
            ],
        }

    def print_summary(self) -> None:
        """Print a colored CLI summary of this session."""
        from .cli import format_session_summary

        print(
            format_session_summary(
                cost=self.cost_usd,
                signals=self._safety_signals,
                call_count=self._call_count,
                policy=self._policy,
            )
        )

    def serve_dashboard(self, port: int = 8899) -> None:
        """Start a local web dashboard (blocking). Opens http://localhost:{port}."""
        from .dashboard import serve

        serve(self, port=port)

    async def preflight_check(self, *, input_text: str, call_id: str) -> None:
        """Run detectors on input text before the API call.

        Raises ``AepPreflightBlock`` if any detector returns a HIGH severity
        signal that the enforcement policy would block.
        """
        if not input_text:
            return

        signals = await self._registry.run_all(
            input_text=input_text,
            output_text="",
            call_id=call_id,
        )
        if signals:
            decision = evaluate(signals, self._policy)
            if decision.action == "block":
                self._safety_signals.extend(signals)
                self._last_enforcement = decision
                self._call_count += 1
                raise AepPreflightBlock(decision, signals)

    def _log_verbose(
        self,
        *,
        call_id: str,
        input_text: str,
        output_text: str,
        signals: Sequence[SafetySignal],
        enforcement: EnforcementDecision,
    ) -> None:
        """Print input/output snippets and detector results when verbose."""
        _DIM = "\033[2m"
        _CYAN = "\033[36m"
        _RESET = "\033[0m"
        print(f"\n{_DIM}{'─' * 60}{_RESET}")
        print(f"{_CYAN}[AEP] call_id={call_id}{_RESET}")
        print(f"{_DIM}INPUT  >{_RESET}")
        for line in _snippet(input_text).splitlines():
            print(f"  {_DIM}{line}{_RESET}")
        print(f"{_DIM}OUTPUT >{_RESET}")
        for line in _snippet(output_text).splitlines():
            print(f"  {_DIM}{line}{_RESET}")
        print(f"{_DIM}ENFORCEMENT > {enforcement.action.upper()}{_RESET}")
        if signals:
            for s in signals:
                print(f"  {_DIM}[{s.severity.upper()}] {s.signal_type}: {s.detail}{_RESET}")
        else:
            print(f"  {_DIM}(no signals){_RESET}")
        print(f"{_DIM}{'─' * 60}{_RESET}")

    async def _record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_id: str,
        *,
        output_text: str = "",
        input_text: str = "",
    ) -> None:
        """Record cost + span for one LLM call and run T&S checks."""
        self._call_count += 1

        span = self._span_tracker.start_span(
            executor_type="llm",
            executor_id=model,
            metadata={
                "call_id": call_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        cost_node = self._cost_tracker.record_llm_cost(
            span_id=span.span_id,
            model=model,
            usage=usage,
        )
        self._span_tracker.end_span(span.span_id)
        self._call_costs.append(cost_node.compute_cost)

        # Run all registered detectors
        signals = await self._registry.run_all(
            input_text=input_text,
            output_text=output_text,
            call_id=call_id,
            call_cost=cost_node.compute_cost,
        )
        self._safety_signals.extend(signals)

        # Evaluate enforcement on the new signals
        self._last_enforcement = evaluate(signals, self._policy)

        if self._verbose:
            self._log_verbose(
                call_id=call_id,
                input_text=input_text,
                output_text=output_text,
                signals=signals,
                enforcement=self._last_enforcement,
            )


# ---------------------------------------------------------------------------
# OpenAI wrapper
# ---------------------------------------------------------------------------


def _wrap_openai(client: Any, session: AepSession) -> Any:
    """Intercept client.chat.completions.create on an OpenAI-like client."""
    original_create = client.chat.completions.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex[:8]

        # Pre-flight: check input before sending to LLM
        input_text = _extract_input_text(kwargs.get("messages", []))
        run_coro_sync(session.preflight_check(input_text=input_text, call_id=call_id))

        result = original_create(*args, **kwargs)

        try:
            usage = result.usage
            model = result.model or kwargs.get("model", "unknown")
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            output_text = ""
            for choice in getattr(result, "choices", []):
                msg = getattr(choice, "message", None)
                if msg and hasattr(msg, "content") and msg.content:
                    output_text += msg.content
            input_text = _extract_input_text(kwargs.get("messages", []))
            run_coro_sync(
                session._record_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    call_id=call_id,
                    output_text=output_text,
                    input_text=input_text,
                )
            )
        except Exception:
            pass  # Never break user code due to AEP instrumentation

        return result

    client.chat.completions.create = patched_create
    client.aep = session
    return client


def _wrap_openai_async(client: Any, session: AepSession) -> Any:
    """Intercept async client.chat.completions.create on an AsyncOpenAI-like client."""
    original_create = client.chat.completions.create

    async def patched_create(*args: Any, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex[:8]

        # Pre-flight: check input before sending to LLM
        input_text = _extract_input_text(kwargs.get("messages", []))
        await session.preflight_check(input_text=input_text, call_id=call_id)

        result = await original_create(*args, **kwargs)

        try:
            usage = result.usage
            model = result.model or kwargs.get("model", "unknown")
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            output_text = ""
            for choice in getattr(result, "choices", []):
                msg = getattr(choice, "message", None)
                if msg and hasattr(msg, "content") and msg.content:
                    output_text += msg.content
            input_text = _extract_input_text(kwargs.get("messages", []))
            await session._record_call(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_id=call_id,
                output_text=output_text,
                input_text=input_text,
            )
        except Exception:
            pass

        return result

    client.chat.completions.create = patched_create
    client.aep = session
    return client


# ---------------------------------------------------------------------------
# Anthropic wrapper
# ---------------------------------------------------------------------------


def _wrap_anthropic(client: Any, session: AepSession) -> Any:
    """Intercept client.messages.create on an Anthropic client."""
    original_create = client.messages.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex[:8]

        # Pre-flight: check input before sending to LLM
        input_text = _extract_input_text(kwargs.get("messages", []))
        run_coro_sync(session.preflight_check(input_text=input_text, call_id=call_id))

        result = original_create(*args, **kwargs)

        try:
            usage = result.usage
            model = result.model or kwargs.get("model", "unknown")
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            output_text = ""
            for block in getattr(result, "content", []):
                if hasattr(block, "text"):
                    output_text += block.text
            input_text = _extract_input_text(kwargs.get("messages", []))
            run_coro_sync(
                session._record_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    call_id=call_id,
                    output_text=output_text,
                    input_text=input_text,
                )
            )
        except Exception:
            pass

        return result

    client.messages.create = patched_create
    client.aep = session
    return client


def _wrap_anthropic_async(client: Any, session: AepSession) -> Any:
    """Intercept async client.messages.create on an AsyncAnthropic client."""
    original_create = client.messages.create

    async def patched_create(*args: Any, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex[:8]

        # Pre-flight: check input before sending to LLM
        input_text = _extract_input_text(kwargs.get("messages", []))
        await session.preflight_check(input_text=input_text, call_id=call_id)

        result = await original_create(*args, **kwargs)

        try:
            usage = result.usage
            model = result.model or kwargs.get("model", "unknown")
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            output_text = ""
            for block in getattr(result, "content", []):
                if hasattr(block, "text"):
                    output_text += block.text
            input_text = _extract_input_text(kwargs.get("messages", []))
            await session._record_call(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_id=call_id,
                output_text=output_text,
                input_text=input_text,
            )
        except Exception:
            pass

        return result

    client.messages.create = patched_create
    client.aep = session
    return client


def _is_async_client(client: Any) -> bool:
    """Check if a client is async (has coroutine .create methods)."""
    import asyncio

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return asyncio.iscoroutinefunction(client.chat.completions.create)
    if hasattr(client, "messages"):
        return asyncio.iscoroutinefunction(client.messages.create)
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wrap(
    client: Any,
    *,
    entity: str = "default",
    detectors: list[Any] | None = None,
    policy: EnforcementPolicy | dict[str, Any] | str | None = None,
    verbose: bool = False,
) -> Any:
    """Wrap any OpenAI or Anthropic client with AEP accountability + T&S.

    Args:
        client: An ``openai.OpenAI()``, ``anthropic.Anthropic()``, or any
                OpenAI-compatible client instance.
        entity: Logical entity name for cost attribution (org ID, user ID, etc.).
        detectors: Custom list of SafetyDetector instances. If None, uses defaults
                   (PII, content safety, cost anomaly). When ``policy`` has detector
                   configs and ``detectors`` is None, detectors are built from policy.
        policy: Enforcement policy. Accepts ``EnforcementPolicy``, a dict, a YAML
                file path, or None for defaults.
        verbose: Print input/output snippets and detector results for each call.
                 Also enabled by setting the ``AEP_LOG=1`` environment variable.

    Returns:
        The same client object, mutated in-place with AEP instrumentation.
        Access AEP data via ``client.aep``.

    Example::

        import openai
        from aceteam_aep import wrap

        client = wrap(openai.OpenAI(), policy={
            "default_action": "flag",
            "detectors": {
                "pii": {"action": "block", "threshold": 0.8},
                "cost_anomaly": {"action": "pass", "multiplier": 10},
            },
        })
    """
    resolved_policy = EnforcementPolicy.from_config(policy)

    session = AepSession(
        entity=entity,
        _policy=resolved_policy,
        _verbose=verbose or os.environ.get("AEP_LOG", "") in ("1", "true", "yes"),
    )

    # Register detectors: explicit > policy-derived > defaults
    if detectors is not None:
        resolved_detectors = detectors
    elif resolved_policy.overrides:
        resolved_detectors = build_detectors_from_policy(resolved_policy)
    else:
        resolved_detectors = _default_detectors()
    for det in resolved_detectors:
        session._registry.add(det)

    client_type = type(client).__name__
    is_async = _is_async_client(client)

    # Detect Anthropic by duck-typing: has .messages but not .chat
    if hasattr(client, "messages") and not hasattr(client, "chat"):
        wrapper = _wrap_anthropic_async if is_async else _wrap_anthropic
        return wrapper(client, session)

    # OpenAI or OpenAI-compatible: has .chat.completions
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        wrapper = _wrap_openai_async if is_async else _wrap_openai
        return wrapper(client, session)

    raise TypeError(
        f"Cannot wrap client of type '{client_type}'. "
        "Supported: openai.OpenAI, anthropic.Anthropic, or any OpenAI-compatible client."
    )


def _extract_input_text(messages: list[Any]) -> str:
    """Extract text content from a messages list for T&S analysis."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


__all__ = ["AepPreflightBlock", "AepSession", "SafetySignal", "wrap"]
