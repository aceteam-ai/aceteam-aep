"""E2E test: CrewAI through AEP proxy."""

from __future__ import annotations

import os

import pytest

from .conftest import get_proxy_state

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def require_crewai():
    try:
        import crewai  # noqa: F401
    except ImportError:
        pytest.skip("crewai not installed")


def test_crewai_through_proxy(aep_proxy, require_crewai):
    """CrewAI agent call routes through AEP proxy."""
    port, base_url = aep_proxy

    # CrewAI reads OPENAI_API_BASE or uses litellm under the hood
    os.environ["OPENAI_API_BASE"] = f"{base_url}/v1"

    from crewai import Agent, Crew, Task

    researcher = Agent(
        role="Researcher",
        goal="Answer simple questions accurately",
        backstory="You are a helpful research assistant.",
        llm="gpt-4o-mini",
        verbose=False,
    )

    task = Task(
        description="What is the capital of France? Answer in one word.",
        expected_output="A single word: the capital city name.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    result = crew.kickoff()

    assert result
    assert "paris" in str(result).lower()

    state = get_proxy_state(base_url)
    assert state["calls"] >= 1
    assert float(state["cost"]) > 0

    # Clean up
    del os.environ["OPENAI_API_BASE"]
