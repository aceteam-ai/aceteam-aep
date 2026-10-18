"""Tests for budget enforcement."""

from decimal import Decimal

import pytest

from aceteam_aep.budget import BudgetEnforcer, BudgetExceededError, BudgetValidationError


def test_basic_reservation():
    enforcer = BudgetEnforcer(total="10.00")
    token = enforcer.reserve("1.00")

    assert enforcer.state.remaining() == Decimal("9.00")
    assert token.amount == Decimal("1.00")


def test_settle_with_actual_cost():
    enforcer = BudgetEnforcer(total="10.00")
    token = enforcer.reserve("2.00")

    # Actual cost less than reservation
    enforcer.settle(token, "1.50")

    assert enforcer.state.spent == Decimal("1.50")
    assert enforcer.state.reserved == Decimal("0")
    assert enforcer.state.remaining() == Decimal("8.50")


def test_budget_exceeded():
    enforcer = BudgetEnforcer(total="1.00")

    with pytest.raises(BudgetExceededError):
        enforcer.reserve("2.00")


def test_budget_exceeded_after_spending():
    enforcer = BudgetEnforcer(total="5.00")
    token = enforcer.reserve("3.00")
    enforcer.settle(token, "3.00")

    with pytest.raises(BudgetExceededError):
        enforcer.reserve("3.00")


def test_multiple_reservations():
    enforcer = BudgetEnforcer(total="10.00")
    t1 = enforcer.reserve("3.00")
    t2 = enforcer.reserve("3.00")

    assert enforcer.state.reserved == Decimal("6.00")
    assert enforcer.state.remaining() == Decimal("4.00")

    enforcer.settle(t1, "2.50")
    assert enforcer.state.reserved == Decimal("3.00")
    assert enforcer.state.spent == Decimal("2.50")

    enforcer.settle(t2, "3.00")
    assert enforcer.state.reserved == Decimal("0")
    assert enforcer.state.spent == Decimal("5.50")


def test_check_ok():
    enforcer = BudgetEnforcer(total="10.00")
    assert enforcer.check() is True


def test_state_currency():
    enforcer = BudgetEnforcer(total="100.00", currency="EUR")
    assert enforcer.state.currency == "EUR"


def test_string_amounts():
    enforcer = BudgetEnforcer(total="10.00")
    token = enforcer.reserve("1.50")
    enforcer.settle(token, "1.25")
    assert enforcer.state.spent == Decimal("1.25")


def test_settlement_is_single_use_and_does_not_mutate_state_on_repeat():
    """The internal token-existence guard is required for this regression.

    Without it, a second settlement would debit the same reservation again.
    """
    enforcer = BudgetEnforcer(total="10.00")
    token = enforcer.reserve("4.00")
    enforcer.settle(token, "4.00")
    before = (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining())

    with pytest.raises(BudgetValidationError, match="Unknown or already-settled"):
        enforcer.settle(token, "4.00")

    assert (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining()) == before


def test_unknown_or_foreign_token_does_not_mutate_state():
    enforcer = BudgetEnforcer(total="10.00")
    foreign_token = BudgetEnforcer(total="10.00").reserve("3.00")
    before = (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining())

    with pytest.raises(BudgetValidationError, match="Unknown or already-settled"):
        enforcer.settle(foreign_token, "1.00")

    assert (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining()) == before


def test_settlement_uses_recorded_reservation_not_mutable_token_amount():
    enforcer = BudgetEnforcer(total="10.00")
    token = enforcer.reserve("2.00")
    token.amount = Decimal("9.00")

    enforcer.settle(token, "1.00")

    assert enforcer.state.reserved == Decimal("0")
    assert enforcer.state.spent == Decimal("1.00")
    assert enforcer.state.remaining() == Decimal("9.00")


@pytest.mark.parametrize("amount", ["-0.01", "NaN", "Infinity", "-Infinity"])
def test_invalid_reservation_amount_does_not_mutate_state(amount):
    enforcer = BudgetEnforcer(total="10.00")
    before = (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining())

    with pytest.raises(BudgetValidationError):
        enforcer.reserve(amount)

    assert (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining()) == before


@pytest.mark.parametrize("amount", ["-0.01", "NaN", "Infinity", "-Infinity"])
def test_invalid_settlement_amount_does_not_mutate_state(amount):
    enforcer = BudgetEnforcer(total="10.00")
    token = enforcer.reserve("2.00")
    before = (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining())

    with pytest.raises(BudgetValidationError):
        enforcer.settle(token, amount)

    assert (enforcer.state.spent, enforcer.state.reserved, enforcer.state.remaining()) == before

    enforcer.settle(token, "1.00")
    assert enforcer.state.spent == Decimal("1.00")
    assert enforcer.state.reserved == Decimal("0")


def test_underestimated_actual_cost_is_recorded_before_budget_exceeded_report():
    enforcer = BudgetEnforcer(total="5.00")
    token = enforcer.reserve("3.00")

    enforcer.settle(token, "6.00")

    assert enforcer.state.spent == Decimal("6.00")
    assert enforcer.state.reserved == Decimal("0")
    with pytest.raises(BudgetExceededError):
        enforcer.check()
