"""AEP budget enforcement with pessimistic reservation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""

    def __init__(self, budget_total: Decimal, spent: Decimal, reserved: Decimal) -> None:
        self.budget_total = budget_total
        self.spent = spent
        self.reserved = reserved
        remaining = budget_total - spent - reserved
        super().__init__(
            f"Budget exceeded: total={budget_total}, spent={spent}, "
            f"reserved={reserved}, remaining={remaining}"
        )


@dataclass
class BudgetState:
    """Runtime budget state."""

    total: Decimal
    spent: Decimal
    reserved: Decimal
    currency: str = "USD"

    def remaining(self) -> Decimal:
        return self.total - self.spent - self.reserved

    def can_reserve(self, amount: Decimal) -> bool:
        return self.remaining() >= amount


@dataclass
class ReservationToken:
    """Token representing a budget reservation."""

    id: str
    amount: Decimal


class BudgetEnforcer:
    """Pessimistic budget enforcement.

    Before each LLM call, reserve an estimated cost. After the call,
    settle with the actual cost (release reservation, add actual).
    """

    def __init__(self, total: Decimal | str, currency: str = "USD") -> None:
        self._state = BudgetState(
            total=Decimal(total),
            spent=Decimal("0"),
            reserved=Decimal("0"),
            currency=currency,
        )
        self._reservations: dict[str, Decimal] = {}

    @property
    def state(self) -> BudgetState:
        return self._state

    def reserve(self, estimated_cost: Decimal | str) -> ReservationToken:
        """Reserve budget for an upcoming operation.

        Raises BudgetExceededError if insufficient budget.
        """
        amount = Decimal(estimated_cost)
        if not self._state.can_reserve(amount):
            raise BudgetExceededError(self._state.total, self._state.spent, self._state.reserved)
        token_id = uuid.uuid4().hex
        self._state.reserved += amount
        self._reservations[token_id] = amount
        return ReservationToken(id=token_id, amount=amount)

    def settle(self, token: ReservationToken, actual_cost: Decimal | str) -> None:
        """Settle a reservation with the actual cost."""
        actual = Decimal(actual_cost)
        reserved = self._reservations.pop(token.id, token.amount)
        self._state.reserved -= reserved
        self._state.spent += actual

    def check(self) -> bool:
        """Check if budget is still available. Raises if exceeded."""
        if self._state.remaining() < 0:
            raise BudgetExceededError(self._state.total, self._state.spent, self._state.reserved)
        return True


__all__ = ["BudgetEnforcer", "BudgetExceededError", "BudgetState", "ReservationToken"]
