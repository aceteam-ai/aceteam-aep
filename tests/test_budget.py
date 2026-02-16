"""Tests for budget enforcement."""

from decimal import Decimal

import pytest

from aceteam_aep.budget import BudgetEnforcer, BudgetExceededError


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
