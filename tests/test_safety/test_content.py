"""Tests for content safety detector."""

from __future__ import annotations

import pytest

from aceteam_aep.safety.content import ContentSafetyDetector


@pytest.fixture(scope="module")
def detector() -> ContentSafetyDetector:
    return ContentSafetyDetector()


async def test_flags_toxic_content(detector: ContentSafetyDetector) -> None:
    signals = await detector.check(
        input_text="",
        output_text="I will kill you and your entire family you worthless piece of trash",
        call_id="t1",
    )
    assert any(s.signal_type == "content_safety" for s in signals)


async def test_clean_content_passes(detector: ContentSafetyDetector) -> None:
    signals = await detector.check(
        input_text="",
        output_text="The quarterly earnings report shows 12% growth in revenue.",
        call_id="t2",
    )
    assert not any(s.signal_type == "content_safety" for s in signals)


async def test_unavailable_model_returns_empty() -> None:
    """If model fails to load, detector silently disables — no crash."""
    det = ContentSafetyDetector(model_name="nonexistent/will-fail")
    signals = await det.check(
        input_text="",
        output_text="anything",
        call_id="t3",
    )
    assert signals == []
