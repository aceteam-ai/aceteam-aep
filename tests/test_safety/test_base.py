"""Tests for the SafetyDetector protocol and DetectorRegistry."""

from __future__ import annotations

from collections.abc import Sequence

from aceteam_aep.safety.base import DetectorRegistry, SafetyDetector, SafetySignal


class FakeDetector(SafetyDetector):
    name = "fake"

    async def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> Sequence[SafetySignal]:
        if "bad" in output_text:
            return [
                SafetySignal(
                    signal_type="test", severity="high", call_id=call_id, detail="found bad"
                )
            ]
        return []


async def test_detector_registry_runs_all_detectors() -> None:
    reg = DetectorRegistry()
    reg.add(FakeDetector())
    signals = await reg.run_all(
        input_text="",
        output_text="something bad",
        call_id="abc",
    )
    assert len(signals) == 1
    assert signals[0].signal_type == "test"
    assert signals[0].detector == "fake"


async def test_detector_registry_empty_on_clean_input() -> None:
    reg = DetectorRegistry()
    reg.add(FakeDetector())
    signals = await reg.run_all(
        input_text="",
        output_text="all good",
        call_id="abc",
    )
    assert signals == []


async def test_detector_registry_catches_detector_errors() -> None:
    class BrokenDetector(SafetyDetector):
        name = "broken"

        async def check(self, **kwargs) -> Sequence[SafetySignal]:
            raise RuntimeError("boom")

    reg = DetectorRegistry()
    reg.add(BrokenDetector())
    signals = await reg.run_all(
        input_text="",
        output_text="test",
        call_id="abc",
    )
    assert signals == []


async def test_multiple_detectors_combine_signals() -> None:
    class SecondDetector(SafetyDetector):
        name = "second"

        async def check(
            self,
            *,
            input_text: str,
            output_text: str,
            call_id: str,
            **kwargs,
        ) -> Sequence[SafetySignal]:
            if "bad" in input_text:
                return [
                    SafetySignal(
                        signal_type="input_check",
                        severity="medium",
                        call_id=call_id,
                        detail="bad input",
                    )
                ]
            return []

    reg = DetectorRegistry()
    reg.add(FakeDetector())
    reg.add(SecondDetector())
    signals = await reg.run_all(input_text="bad input", output_text="bad output", call_id="x")
    assert len(signals) == 2
    types = {s.signal_type for s in signals}
    assert types == {"test", "input_check"}


def test_safety_signal_has_timestamp() -> None:
    sig = SafetySignal(signal_type="t", severity="low", call_id="x", detail="d")
    assert sig.timestamp  # auto-generated, not empty
