"""Tests for global SDK instrumentation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aceteam_aep.instrument import instrument, uninstrument
from aceteam_aep.safety.cost_anomaly import CostAnomalyDetector


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    """Ensure instrumentation is cleaned up after each test."""
    yield  # type: ignore[misc]
    uninstrument()


def test_instrument_patches_openai() -> None:
    """After instrument(), new OpenAI clients should have .aep."""
    import openai

    original_init = openai.OpenAI.__init__
    instrument(detectors=[CostAnomalyDetector()])

    # __init__ should be patched
    assert openai.OpenAI.__init__ is not original_init

    # Cleanup
    uninstrument()
    assert openai.OpenAI.__init__ is original_init


def test_instrument_patches_anthropic() -> None:
    """After instrument(), Anthropic class should be patched."""
    import anthropic

    original_init = anthropic.Anthropic.__init__
    instrument(detectors=[CostAnomalyDetector()])

    assert anthropic.Anthropic.__init__ is not original_init

    uninstrument()
    assert anthropic.Anthropic.__init__ is original_init


def test_double_instrument_warns() -> None:
    """Calling instrument() twice should warn and skip."""
    instrument(detectors=[CostAnomalyDetector()])
    with pytest.warns(match="") if False else patch("aceteam_aep.instrument.log") as mock_log:
        instrument(detectors=[CostAnomalyDetector()])
        mock_log.warning.assert_called_once()


def test_uninstrument_is_safe_when_not_instrumented() -> None:
    """uninstrument() should not crash when not instrumented."""
    uninstrument()  # should not raise
