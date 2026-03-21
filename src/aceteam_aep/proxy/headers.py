"""AEP header protocol — parse and emit X-AEP-* headers on the proxy.

Headers are the bridge between Layer 1 (wire-level proxy) and Layer 2 (application-level
protocol). Any client can send AEP headers to inject governance context without using
the Python SDK.

Spec:
    X-AEP-Entity: org:acme                     # Who is calling
    X-AEP-Classification: confidential          # Sensitivity level
    X-AEP-Consent: training=no,sharing=org      # Consent directives
    X-AEP-Budget: 5.00                          # Max budget for this call (USD)
    X-AEP-Trace-ID: abc123                      # Link to parent execution envelope
    X-AEP-Sources: doc:contract-123,url:...     # Declare sources in context
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass
class AepRequestContext:
    """Parsed AEP context from request headers."""

    entity: str = "default"
    classification: str = "public"  # public, internal, confidential, restricted
    consent: dict[str, bool] = field(default_factory=dict)
    budget: Decimal | None = None
    trace_id: str | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def has_governance(self) -> bool:
        return self.classification != "public" or bool(self.consent)


_CLASSIFICATION_LEVELS = ("public", "internal", "confidential", "restricted")
_CLASSIFICATION_RANK = {level: i for i, level in enumerate(_CLASSIFICATION_LEVELS)}


def parse_aep_headers(headers: dict[str, str] | Any) -> AepRequestContext:
    """Parse X-AEP-* headers from a request into an AepRequestContext."""
    get = headers.get if isinstance(headers, dict) else getattr(headers, "get", lambda k, d=None: d)

    ctx = AepRequestContext()

    entity = get("x-aep-entity", None)
    if entity:
        ctx.entity = entity

    classification = get("x-aep-classification", None)
    if classification and classification.lower() in _CLASSIFICATION_LEVELS:
        ctx.classification = classification.lower()

    consent_raw = get("x-aep-consent", None)
    if consent_raw:
        for pair in consent_raw.split(","):
            pair = pair.strip()
            if "=" in pair:
                key, val = pair.split("=", 1)
                ctx.consent[key.strip()] = val.strip().lower() not in ("no", "false", "0")

    budget_raw = get("x-aep-budget", None)
    if budget_raw:
        with contextlib.suppress(InvalidOperation):
            ctx.budget = Decimal(budget_raw.strip())

    trace_id = get("x-aep-trace-id", None)
    if trace_id:
        ctx.trace_id = trace_id

    sources_raw = get("x-aep-sources", None)
    if sources_raw:
        ctx.sources = [s.strip() for s in sources_raw.split(",") if s.strip()]

    return ctx


def build_response_headers(
    *,
    cost: Decimal,
    enforcement: str,
    call_id: str,
    classification: str = "public",
    flag_reason: str = "",
    trace_id: str | None = None,
) -> dict[str, str]:
    """Build X-AEP-* response headers."""
    headers = {
        "X-AEP-Cost": str(cost),
        "X-AEP-Enforcement": enforcement,
        "X-AEP-Call-ID": call_id,
        "X-AEP-Classification": classification,
    }
    if flag_reason:
        headers["X-AEP-Flag-Reason"] = flag_reason
    if trace_id:
        headers["X-AEP-Trace-ID"] = trace_id
    return headers


def strip_aep_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove X-AEP-* headers before forwarding to upstream API."""
    return {k: v for k, v in headers.items() if not k.lower().startswith("x-aep-")}


def classification_rank(level: str) -> int:
    """Get numeric rank of a classification level (higher = more sensitive)."""
    return _CLASSIFICATION_RANK.get(level.lower(), 0)


__all__ = [
    "AepRequestContext",
    "build_response_headers",
    "classification_rank",
    "parse_aep_headers",
    "strip_aep_headers",
]
