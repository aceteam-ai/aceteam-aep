"""Cost anomaly detector — flags calls that spike above session average."""

from __future__ import annotations

from decimal import Decimal

from .base import SafetySignal


class CostAnomalyDetector:
    """Flags calls whose cost exceeds a multiple of the session average."""

    name = "cost_anomaly"

    def __init__(self, min_calls: int = 3, multiplier: int = 5) -> None:
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
        **kwargs: object,
    ) -> list[SafetySignal]:
        signals: list[SafetySignal] = []
        self._history.append(call_cost)
        if len(self._history) <= self._min_calls:
            return signals
        prior = self._history[:-1]
        avg = sum(prior, Decimal("0")) / len(prior)
        if avg > 0 and call_cost > avg * self._multiplier:
            signals.append(
                SafetySignal(
                    signal_type="cost_anomaly",
                    severity="medium",
                    call_id=call_id,
                    detail=f"Cost ${call_cost:.6f} is >{self._multiplier}x session avg ${avg:.6f}",
                )
            )
        return signals


__all__ = ["CostAnomalyDetector"]
