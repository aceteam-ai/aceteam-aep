"""PAW confidence plumbing keeps legacy layer verdicts until calibration is wired."""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import PropertyMock, patch

import pytest

from aceteam_aep.safety.custom import (
    CustomPolicy,
    CustomPolicyStore,
    CustomSafetyDetector,
    PawResult,
    _extract_yn_logit_diff,
    _paw_call_with_logprobs,
)
from aceteam_aep.safety.pipeline import PawLayer, paw_layer_p_safe


def test_raw_logit_gap_survives_softmax_saturation(monkeypatch: pytest.MonkeyPatch) -> None:
    llama = types.ModuleType("llama_cpp")
    llama.llama_get_logits_ith = lambda _context, _index: [50.0, 0.0]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", llama)

    class FakeLlama:
        ctx = object()
        n_tokens = 0
        samples = 0

        def tokenize(self, value, *, add_bos, special):
            if value == b"Y":
                return [0]
            if value == b"N":
                return [1]
            return [2]

        def eval(self, tokens):
            self.n_tokens += len(tokens)

        def sample(self, *, temp):
            self.samples += 1
            return 0 if self.samples == 1 else 99

        def token_eos(self):
            return 99

        def detokenize(self, tokens):
            return b"Y" if tokens == [0] else b""

    fn = types.SimpleNamespace(_llm=FakeLlama(), _n_prefix=2, _suffix_text="", _n_ctx=10)
    assert _extract_yn_logit_diff(fn) == 50.0
    output, probability, gap = _paw_call_with_logprobs(fn, "sample")
    assert output == "Y"
    assert probability == 1.0
    assert gap == 50.0


@pytest.mark.asyncio
async def test_policy_carries_per_chunk_raw_gaps() -> None:
    class FakePaw:
        def __init__(self):
            self.calls = 0

        async def call_with_logit_diff(self, text):
            self.calls += 1
            return ("Y", 0.99, 4.6) if self.calls == 1 else ("Y", 0.8, 1.2)

    fake = FakePaw()
    policy = CustomPolicy(name="test", rule="test")
    with patch.object(CustomPolicy, "_paw", new_callable=PropertyMock, return_value=fake):
        result = await policy.check_with_confidence("x" * 4097)
    assert result.compliant
    assert result.p_compliant == 0.8
    assert result.logit_diff == 1.2
    assert result.chunk_logit_diffs == [4.6, 1.2]
    assert result.chunk_results == [(True, 0.99), (True, 0.8)]


@pytest.mark.asyncio
async def test_detailed_check_keeps_both_branches_and_unrounded_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CustomPolicy(name="test", rule="test", applies_to="both")
    seen: list[str] = []

    async def fake_check(_self: CustomPolicy, text: str) -> PawResult:
        seen.append(text)
        return PawResult(compliant=text == "clean", p_compliant=0.99995, logit_diff=4.6)

    monkeypatch.setattr(CustomPolicy, "check_with_confidence", fake_check)
    detector = CustomSafetyDetector(CustomPolicyStore([policy]))
    signals, details = await detector.check_detailed(
        input_text="unsafe", output_text="clean", call_id="call"
    )
    assert seen == ["clean", "unsafe"]
    assert [(entry.policy_id, entry.source) for entry in details] == [
        (policy.id, "output"),
        (policy.id, "input"),
    ]
    assert [entry.result.compliant for entry in details] == [True, False]
    assert len(signals) == 1
    assert signals[0].score == pytest.approx(5e-5)
    assert "p_unsafe=0.0" in signals[0].detail

    layer = PawLayer(detector)
    legacy = await layer.score(
        input_text="unsafe", output_text="clean", call_id="call", prior_results=[]
    )
    assert legacy.p_safe == 1.0
    assert legacy.signals[0].score == pytest.approx(5e-5)


def test_paw_layer_p_safe_uses_raw_gap_on_every_policy_and_chunk() -> None:
    def identity(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    clean = PawResult(compliant=True, p_compliant=0.5, logit_diff=4.6)
    assert paw_layer_p_safe([clean], identity) == pytest.approx(0.99, abs=1e-3)
    second = PawResult(compliant=True, logit_diff=1.0)
    assert paw_layer_p_safe([clean, second], identity) == pytest.approx(identity(1.0))
    chunks = PawResult(compliant=True, logit_diff=4.6, chunk_logit_diffs=[4.6, -1.0])
    assert paw_layer_p_safe([chunks], identity) == pytest.approx(identity(-1.0))
    assert paw_layer_p_safe([PawResult(compliant=True, logit_diff=None)], identity) is None
    assert (
        paw_layer_p_safe([PawResult(compliant=True, chunk_logit_diffs=[4.6, None])], identity)
        is None
    )
    assert paw_layer_p_safe([PawResult(compliant=True, chunk_logit_diffs=[])], identity) is None
    assert paw_layer_p_safe([], identity) == 1.0


@pytest.mark.asyncio
async def test_missing_chunk_gap_marks_result_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePaw:
        calls = 0

        async def call_with_logit_diff(self, text):
            self.calls += 1
            return ("Y", 0.99, 4.6) if self.calls == 1 else ("Y", None, None)

    policy = CustomPolicy(name="test", rule="test")
    with patch.object(CustomPolicy, "_paw", new_callable=PropertyMock, return_value=FakePaw()):
        result = await policy.check_with_confidence("x" * 4097)
    assert result.logit_diff is None
    assert result.chunk_logit_diffs == [4.6, None]
    assert result.p_compliant == 0.99
