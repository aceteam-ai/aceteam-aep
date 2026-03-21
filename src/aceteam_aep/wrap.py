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

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .costs import CostTracker
from .enforcement import EnforcementDecision, EnforcementPolicy, evaluate
from .safety.base import DetectorRegistry, SafetySignal
from .safety.cost_anomaly import CostAnomalyDetector
from .spans import Span, SpanTracker
from .types import Usage


def _default_detectors() -> list[Any]:
    """Build the default detector list. Model-based detectors only if available."""
    detectors: list[Any] = [CostAnomalyDetector()]
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

    def _record_call(
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
        signals = self._registry.run_all(
            input_text=input_text,
            output_text=output_text,
            call_id=call_id,
            call_cost=cost_node.compute_cost,
        )
        self._safety_signals.extend(signals)

        # Evaluate enforcement on the new signals
        self._last_enforcement = evaluate(signals, self._policy)


# ---------------------------------------------------------------------------
# OpenAI wrapper
# ---------------------------------------------------------------------------


def _wrap_openai(client: Any, session: AepSession) -> Any:
    """Intercept client.chat.completions.create on an OpenAI-like client."""
    original_create = client.chat.completions.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex[:8]
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
            session._record_call(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_id=call_id,
                output_text=output_text,
                input_text=input_text,
            )
        except Exception:
            pass  # Never break user code due to AEP instrumentation

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
            session._record_call(
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wrap(
    client: Any,
    *,
    entity: str = "default",
    detectors: list[Any] | None = None,
    policy: EnforcementPolicy | None = None,
) -> Any:
    """Wrap any OpenAI or Anthropic client with AEP accountability + T&S.

    Args:
        client: An ``openai.OpenAI()``, ``anthropic.Anthropic()``, or any
                OpenAI-compatible client instance.
        entity: Logical entity name for cost attribution (org ID, user ID, etc.).
        detectors: Custom list of SafetyDetector instances. If None, uses defaults
                   (PII, content safety, cost anomaly).
        policy: Custom EnforcementPolicy. If None, uses default (block on high,
                flag on medium).

    Returns:
        The same client object, mutated in-place with AEP instrumentation.
        Access AEP data via ``client.aep``.

    Example::

        import openai
        from aceteam_aep import wrap

        client = wrap(openai.OpenAI(), entity="org:acme")
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(f"Cost: ${client.aep.cost_usd}")
        print(f"Safety: {client.aep.enforcement.action}")
    """
    session = AepSession(
        entity=entity,
        _policy=policy or EnforcementPolicy(),
    )

    # Register detectors
    for det in detectors or _default_detectors():
        session._registry.add(det)

    client_type = type(client).__name__

    # Detect Anthropic by duck-typing: has .messages but not .chat
    if hasattr(client, "messages") and not hasattr(client, "chat"):
        return _wrap_anthropic(client, session)

    # OpenAI or OpenAI-compatible: has .chat.completions
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return _wrap_openai(client, session)

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


__all__ = ["AepSession", "SafetySignal", "wrap"]
