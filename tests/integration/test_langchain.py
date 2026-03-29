"""E2E test: LangChain through AEP proxy."""

from __future__ import annotations

import os

import pytest

from .conftest import get_proxy_state

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def require_langchain():
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
    except ImportError:
        pytest.skip("langchain-openai not installed")


def test_langchain_through_proxy(aep_proxy, require_langchain):
    """LangChain ChatOpenAI routes through AEP proxy."""
    port, base_url = aep_proxy

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_base=f"{base_url}/v1",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    result = llm.invoke("What is the capital of France? Answer in one word.")

    assert result.content
    assert "paris" in result.content.lower()

    state = get_proxy_state(base_url)
    assert state["calls"] >= 1
    assert float(state["cost"]) > 0
