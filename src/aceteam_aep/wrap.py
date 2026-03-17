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

Supports:
- openai.OpenAI / openai.AsyncOpenAI
- anthropic.Anthropic / anthropic.AsyncAnthropic
- Any object with a .chat.completions.create() method (OpenAI-compatible)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from .costs import CostTracker
from .spans import Span, SpanTracker
from .types import Usage

# ---------------------------------------------------------------------------
# T&S signal detection — lightweight, regex-based, no external deps
# ---------------------------------------------------------------------------

# Patterns that may indicate PII in model output
_PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),           # SSN
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),                  # credit card
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
]

# Simple heuristics for content classification
_HARMFUL_KEYWORDS = frozenset([
    "how to make a bomb", "synthesis of", "step by step hack",
    "bypass security", "exploit vulnerability", "malware payload",
])

_SENSITIVE_TOPICS = frozenset([
    "suicide", "self-harm", "child abuse", "csam",
    "terrorism", "extremism",
])


@dataclass
class SafetySignal:
    """A single T&S flag raised during a session."""

    # signal_type: "pii_in_output", "harmful_content", "sensitive_topic", "cost_anomaly"
    signal_type: str
    # severity: "low", "medium", "high"
    severity: str
    call_id: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AepSession:
    """AEP session state attached to a wrapped client as ``client.aep``."""

    entity: str
    _cost_tracker: CostTracker = field(default_factory=CostTracker)
    _span_tracker: SpanTracker = field(default_factory=SpanTracker)
    _safety_signals: list[SafetySignal] = field(default_factory=list)
    _call_costs: list[Decimal] = field(default_factory=list)

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
        # Span first so we have span_id for cost attribution
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

        # T&S checks
        self._check_pii(output_text, call_id)
        self._check_content(input_text + " " + output_text, call_id)
        self._check_cost_anomaly(cost_node.compute_cost, call_id)

    def _check_pii(self, text: str, call_id: str) -> None:
        for pattern in _PII_PATTERNS:
            if pattern.search(text):
                self._safety_signals.append(SafetySignal(
                    signal_type="pii_in_output",
                    severity="high",
                    call_id=call_id,
                    detail="Potential PII pattern detected in model output",
                ))
                break  # one signal per call is enough

    def _check_content(self, text: str, call_id: str) -> None:
        lower = text.lower()
        for kw in _HARMFUL_KEYWORDS:
            if kw in lower:
                self._safety_signals.append(SafetySignal(
                    signal_type="harmful_content",
                    severity="high",
                    call_id=call_id,
                    detail=f"Potentially harmful keyword detected: '{kw}'",
                ))
                return
        for topic in _SENSITIVE_TOPICS:
            if topic in lower:
                self._safety_signals.append(SafetySignal(
                    signal_type="sensitive_topic",
                    severity="medium",
                    call_id=call_id,
                    detail=f"Sensitive topic detected: '{topic}'",
                ))
                return

    def _check_cost_anomaly(self, cost: Decimal, call_id: str) -> None:
        """Flag calls that cost 5x more than session average."""
        if len(self._call_costs) < 3:
            return
        avg = sum(self._call_costs[:-1], Decimal("0")) / len(self._call_costs[:-1])
        if avg > 0 and cost > avg * 5:
            self._safety_signals.append(SafetySignal(
                signal_type="cost_anomaly",
                severity="medium",
                call_id=call_id,
                detail=f"Call cost ${cost:.6f} is >5x session average ${avg:.6f}",
            ))


# ---------------------------------------------------------------------------
# OpenAI wrapper
# ---------------------------------------------------------------------------

def _wrap_openai(client: Any, session: AepSession) -> Any:
    """Intercept client.chat.completions.create on an OpenAI-like client."""
    original_create = client.chat.completions.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex[:8]
        result = original_create(*args, **kwargs)

        # Extract usage and text from response
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

def wrap(client: Any, *, entity: str = "default") -> Any:
    """Wrap any OpenAI or Anthropic client with AEP accountability + T&S.

    Args:
        client: An ``openai.OpenAI()``, ``anthropic.Anthropic()``, or any
                OpenAI-compatible client instance.
        entity: Logical entity name for cost attribution (org ID, user ID, etc.).

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
        print(f"Safety signals: {client.aep.safety_signals}")
    """
    session = AepSession(entity=entity)
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


__all__ = ["wrap", "AepSession", "SafetySignal"]
