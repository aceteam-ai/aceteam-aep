"""Cost anomaly detector — flags calls that spike above session average."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from .base import SafetyDetector, SafetySignal


class CostAnomalyDetector(SafetyDetector):
    """Flags calls whose cost exceeds a multiple of the session average."""

    name = "cost_anomaly"

    def __init__(self, min_calls: int = 3, multiplier: int = 5) -> None:
        if min_calls <= 0:
            raise ValueError("min_calls must be greater than 0")
        self._min_calls = min_calls
        self._multiplier = multiplier
        self._history: list[Decimal] = []

    def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        call_cost: Decimal = Decimal("0"),
        **kwargs,
    ) -> Sequence[SafetySignal]:
        signals: list[SafetySignal] = []
        self._history.append(call_cost)
        if len(self._history) <= self._min_calls:
            return signals
        prior = self._history[:-1]
        avg = sum(prior, Decimal("0")) / len(prior)
        if avg > 0 and call_cost > avg * self._multiplier:
            cost_ratio = float(call_cost / avg)
            message = (
                f"Cost ${call_cost:.6f} is over {self._multiplier} times the average (${avg:.6f})"
            )
            signals.append(
                SafetySignal(
                    signal_type="cost_anomaly",
                    severity="medium",
                    call_id=call_id,
                    detail=message,
                    score=cost_ratio,
                )
            )
        return signals


__all__ = [
    "CostAnomalyDetector",
]
