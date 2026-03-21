"""CLI pretty-print for AEP session summaries."""

from __future__ import annotations

from decimal import Decimal

from .enforcement import EnforcementPolicy, evaluate
from .safety.base import SafetySignal

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_ACTION_COLORS = {"pass": _GREEN, "flag": _YELLOW, "block": _RED}
_SEV_COLORS = {"high": _RED, "medium": _YELLOW, "low": _CYAN}


def format_session_summary(
    *,
    cost: Decimal,
    signals: list[SafetySignal],
    call_count: int,
    policy: EnforcementPolicy | None = None,
) -> str:
    """Format a colored CLI summary of an AEP session."""
    policy = policy or EnforcementPolicy()
    decision = evaluate(signals, policy)
    color = _ACTION_COLORS.get(decision.action, "")

    lines = [
        "",
        f"{_DIM}{'─' * 50}{_RESET}",
        f"  {_BOLD}AEP Session Summary{_RESET}",
        f"{_DIM}{'─' * 50}{_RESET}",
        f"  Calls:  {call_count}",
        f"  Cost:   ${cost:.6f}",
        f"  Safety: {color}{_BOLD}{decision.action.upper()}{_RESET}",
    ]
    if signals:
        lines.append(f"  Signals ({len(signals)}):")
        for s in signals:
            sev_color = _SEV_COLORS.get(s.severity, "")
            lines.append(
                f"    [{sev_color}{s.severity.upper()}{_RESET}] "
                f"{s.signal_type}: {s.detail}"
            )
    lines.append(f"{_DIM}{'─' * 50}{_RESET}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["format_session_summary"]
