"""E2E test: DSPy through AEP proxy."""

from __future__ import annotations

import os

import pytest

from .conftest import get_proxy_state

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def require_dspy():
    try:
        import dspy  # noqa: F401
    except ImportError:
        pytest.skip("dspy not installed")


def test_dspy_through_proxy(aep_proxy, require_dspy):
    """DSPy call routes through AEP proxy via OPENAI_API_BASE."""
    port, base_url = aep_proxy

    import dspy

    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_base=f"{base_url}/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    result = predict(question="What is the capital of France?")

    assert result.answer
    assert "paris" in result.answer.lower()

    state = get_proxy_state(base_url)
    assert state["calls"] >= 1
    assert float(state["cost"]) > 0
