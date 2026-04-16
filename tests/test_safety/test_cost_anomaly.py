"""Tests for cost anomaly detector."""

from decimal import Decimal

from aceteam_aep.safety.cost_anomaly import CostAnomalyDetector


def test_no_anomaly_with_few_calls() -> None:
    det = CostAnomalyDetector()
    signals = det.check(input_text="", output_text="", call_id="1", call_cost=Decimal("1.00"))
    assert signals == []


def test_detects_spike_after_baseline() -> None:
    det = CostAnomalyDetector(min_calls=3, multiplier=5)
    for i in range(3):
        det.check(input_text="", output_text="", call_id=str(i), call_cost=Decimal("0.01"))
    signals = det.check(input_text="", output_text="", call_id="spike", call_cost=Decimal("1.00"))
    assert any(s.signal_type == "cost_anomaly" for s in signals)


def test_no_false_positive_on_normal_variation() -> None:
    det = CostAnomalyDetector(min_calls=3, multiplier=5)
    for i in range(3):
        det.check(input_text="", output_text="", call_id=str(i), call_cost=Decimal("0.01"))
    signals = det.check(input_text="", output_text="", call_id="normal", call_cost=Decimal("0.03"))
    assert not any(s.signal_type == "cost_anomaly" for s in signals)


def test_zero_average_does_not_crash() -> None:
    det = CostAnomalyDetector(min_calls=2, multiplier=5)
    for i in range(3):
        det.check(input_text="", output_text="", call_id=str(i), call_cost=Decimal("0"))
    signals = det.check(input_text="", output_text="", call_id="x", call_cost=Decimal("1.00"))
    # avg is 0, so condition `avg > 0` is False — no signal, no crash
    assert signals == []
