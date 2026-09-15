"""AEP budget enforcement with pessimistic reservation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


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


class BudgetValidationError(ValueError):
    """Raised when a reservation or settlement amount is not valid."""


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

    @staticmethod
    def _validated_amount(amount: Decimal | str, *, operation: str) -> Decimal:
        """Return a non-negative, finite amount suitable for accounting.

        Validation happens before callers mutate reservation or spending state.
        Zero remains valid so failed calls can release their reservation without
        inventing a cost.
        """
        try:
            value = Decimal(amount)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise BudgetValidationError(f"Invalid {operation} amount: {amount!r}") from exc
        if not value.is_finite() or value < 0:
            raise BudgetValidationError(
                f"{operation.capitalize()} amount must be finite and non-negative: {amount!r}"
            )
        return value

    def reserve(self, estimated_cost: Decimal | str) -> ReservationToken:
        """Reserve budget for an upcoming operation.

        Raises BudgetExceededError if insufficient budget.
        """
        amount = self._validated_amount(estimated_cost, operation="reservation")
        if not self._state.can_reserve(amount):
            raise BudgetExceededError(self._state.total, self._state.spent, self._state.reserved)
        token_id = uuid.uuid4().hex
        self._state.reserved += amount
        self._reservations[token_id] = amount
        return ReservationToken(id=token_id, amount=amount)

    def settle(self, token: ReservationToken, actual_cost: Decimal | str) -> None:
        """Settle a reservation exactly once with its actual incurred cost.

        The recorded reservation, rather than the caller-owned token amount,
        is released. An actual cost may exceed its estimate: it is still
        recorded honestly, and :meth:`check` reports the resulting budget
        exceedance through ``BudgetExceededError``.
        """
        actual = self._validated_amount(actual_cost, operation="settlement")
        reserved = self._reservations.get(token.id)
        if reserved is None:
            raise BudgetValidationError("Unknown or already-settled reservation token")
        del self._reservations[token.id]
        self._state.reserved -= reserved
        self._state.spent += actual

    def check(self) -> bool:
        """Check if budget is still available. Raises if exceeded."""
        if self._state.remaining() < 0:
            raise BudgetExceededError(self._state.total, self._state.spent, self._state.reserved)
        return True


__all__ = [
    "BudgetEnforcer",
    "BudgetExceededError",
    "BudgetState",
    "BudgetValidationError",
    "ReservationToken",
]
