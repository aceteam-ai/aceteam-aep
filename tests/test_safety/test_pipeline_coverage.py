"""Pipeline mode must retain every registered detector's legacy coverage."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.responses import StreamingResponse

from aceteam_aep.enforcement import (
    EnforcementPolicy,
    evaluate_pipeline_with_residual,
)
from aceteam_aep.proxy.app import ProxyState, create_proxy_app
from aceteam_aep.proxy.streaming import handle_streaming_request
from aceteam_aep.safety.agent_threat import AgentThreatDetector
from aceteam_aep.safety.base import SafetySignal
from aceteam_aep.safety.content import ContentSafetyDetector
from aceteam_aep.safety.cost_anomaly import CostAnomalyDetector
from aceteam_aep.safety.custom import CustomPolicyStore, CustomSafetyDetector
from aceteam_aep.safety.ferpa import FerpaDetector
from aceteam_aep.safety.pipeline import PipelineResult, SafetyPipeline
from aceteam_aep.safety.secrets import SecretDetector
from aceteam_aep.safety.trust_engine import TrustEngineDetector

EXAMPLE_KEY = "AKIAIOSFODNN7EXAMPLE"
REGEX_ONLY = {"enabled": True, "layers": [{"name": "regex"}]}


class NoopCustomDetector(CustomSafetyDetector):
    def __init__(self) -> None:
        super().__init__(CustomPolicyStore())

    async def check(self, **kwargs) -> list[SafetySignal]:
        return []


def _response_text(text: str) -> dict:
    return {
        "id": "chatcmpl-residual",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def _post(
    *,
    detectors: list,
    policy: dict,
    input_text: str = "hello",
    output_text: str = "ok",
) -> tuple[httpx.Response, AsyncMock]:
    app = create_proxy_app(detectors=detectors, policy=policy, dashboard=False)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    upstream = AsyncMock()
    upstream.__aenter__ = AsyncMock(return_value=upstream)
    upstream.__aexit__ = AsyncMock(return_value=False)
    upstream.request = AsyncMock(
        return_value=httpx.Response(
            200,
            json=_response_text(output_text),
            headers={"content-type": "application/json"},
        )
    )
    async with client:
        with patch("aceteam_aep.proxy.app.httpx.AsyncClient", return_value=upstream):
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": input_text}]},
                headers={"Authorization": "Bearer sk-test"},
            )
    return response, upstream.request


def test_residual_registry_partitions_by_identity_and_serializes_names():
    secret = SecretDetector()
    ferpa = FerpaDetector()
    cost = CostAnomalyDetector()
    content = ContentSafetyDetector()
    trust = TrustEngineDetector(judge_service_url="https://judge.example.test")
    threat = AgentThreatDetector()
    custom = NoopCustomDetector()
    detectors = [secret, ferpa, cost, content, trust, threat, custom]

    state = ProxyState(detectors=detectors, policy={"pipeline": {"enabled": True}})
    assert state.pipeline is not None
    assert state.residual_registry is not None
    assert [d.name for d in state.residual_registry._detectors] == [
        "secret_leak",
        "ferpa",
        "cost_anomaly",
    ]
    assert {id(d) for d in state.pipeline.covered_detectors} == {
        id(content),
        id(trust),
        id(threat),
        id(custom),
    }
    assert state.to_dict()["pipeline"]["residual_detectors"] == [
        "secret_leak",
        "ferpa",
        "cost_anomaly",
    ]

    paw_only = ProxyState(
        detectors=detectors, policy={"pipeline": {"enabled": True, "layers": [{"name": "paw"}]}}
    )
    assert paw_only.residual_registry is not None
    assert threat in paw_only.residual_registry._detectors
    assert content in paw_only.residual_registry._detectors
    assert trust in paw_only.residual_registry._detectors

    regex_only = ProxyState(detectors=detectors, policy={"pipeline": REGEX_ONLY})
    assert regex_only.residual_registry is not None
    assert custom in regex_only.residual_registry._detectors

    legacy = ProxyState(detectors=detectors)
    assert legacy.pipeline is None
    assert legacy.residual_registry is None


def test_unknown_layer_without_detector_property_covers_nothing():
    class UnknownLayer:
        name = "unknown"
        prior_p_safe = 0.9

        async def score(self, **kwargs):
            raise AssertionError("not evaluated in this test")

    pipeline = SafetyPipeline(layers=[UnknownLayer()])
    assert pipeline.covered_detectors == ()


def test_residual_verdict_uses_legacy_cost_score_and_stricter_action():
    policy = EnforcementPolicy()
    cost = SafetySignal(
        signal_type="cost_anomaly",
        severity="medium",
        call_id="cost",
        detail="cost spike",
        score=6.0,
    )
    flag = evaluate_pipeline_with_residual(
        PipelineResult(p_safe=0.9, verdict="pass"), [cost], policy
    )
    assert flag.action == "flag"
    assert flag.signals == [cost]
    assert "cost_anomaly" in flag.reason

    unavailable = PipelineResult(
        p_safe=None,
        verdict="block",
        evaluation_unavailable_reason="required evaluation unavailable: paw",
    )
    assert evaluate_pipeline_with_residual(unavailable, [], policy).action == "block"

    high = SafetySignal(signal_type="third_party", severity="high", call_id="high", detail="hit")
    assert (
        evaluate_pipeline_with_residual(
            PipelineResult(p_safe=0.9, verdict="pass"), [high], policy
        ).action
        == "block"
    )


@pytest.mark.parametrize("pipeline_enabled", [False, True])
async def test_pipeline_enabled_secret_in_user_input_blocks(pipeline_enabled):
    policy = {"pipeline": {"enabled": True}} if pipeline_enabled else {}
    response, upstream = await _post(
        detectors=[SecretDetector(), AgentThreatDetector(), NoopCustomDetector()],
        policy=policy,
        input_text=f"Please inspect {EXAMPLE_KEY}",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "safety_block"
    assert "request blocked" in response.json()["error"]["message"].lower()
    upstream.assert_not_awaited()


async def test_pipeline_enabled_secret_in_output_blocks():
    response, upstream = await _post(
        detectors=[SecretDetector(), AgentThreatDetector(), NoopCustomDetector()],
        policy={"pipeline": {"enabled": True}},
        output_text=f"The example credential is {EXAMPLE_KEY}",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "safety_block"
    upstream.assert_awaited_once()


async def test_pipeline_enabled_runs_third_party_detector_exactly_once():
    class ThirdPartyDetector:
        name = "third_party"

        def __init__(self) -> None:
            self.calls = 0

        async def check(self, **kwargs) -> list[SafetySignal]:
            self.calls += 1
            return [
                SafetySignal(
                    signal_type="third_party",
                    severity="high",
                    call_id=kwargs["call_id"],
                    detail="third party hit",
                )
            ]

    spy = ThirdPartyDetector()
    response, upstream = await _post(
        detectors=[spy, AgentThreatDetector(), NoopCustomDetector()],
        policy={"pipeline": REGEX_ONLY},
    )
    assert response.status_code == 400
    assert "request blocked" in response.json()["error"]["message"].lower()
    upstream.assert_not_awaited()
    assert spy.calls == 1


async def test_pipeline_enabled_residual_honors_detector_override():
    response, upstream = await _post(
        detectors=[SecretDetector(), AgentThreatDetector(), NoopCustomDetector()],
        policy={
            "detectors": {"secret_leak": {"action": "flag"}},
            "pipeline": REGEX_ONLY,
        },
        input_text=f"Review example key {EXAMPLE_KEY}",
    )
    assert response.status_code == 200
    assert response.headers["X-AEP-Enforcement"] == "flag"
    upstream.assert_awaited_once()


async def test_pipeline_enabled_ferpa_medium_flags_not_blocks():
    response, upstream = await _post(
        detectors=[FerpaDetector(), AgentThreatDetector(), NoopCustomDetector()],
        policy={"pipeline": REGEX_ONLY},
        input_text="Please export the class roster",
    )
    assert response.status_code == 200
    assert response.headers["X-AEP-Enforcement"] == "flag"
    upstream.assert_awaited_once()


@pytest.mark.parametrize("pipeline_enabled", [False, True])
async def test_pipeline_enabled_cost_anomaly_runs_once_per_phase(pipeline_enabled):
    detector = CostAnomalyDetector()
    policy = {"pipeline": REGEX_ONLY} if pipeline_enabled else {}
    response, upstream = await _post(
        detectors=[detector, AgentThreatDetector(), NoopCustomDetector()],
        policy=policy,
    )
    assert response.status_code == 200
    upstream.assert_awaited_once()
    assert len(detector._history) == 2


async def test_pipeline_enabled_covered_detector_not_run_twice():
    class CountingThreat(AgentThreatDetector):
        def __init__(self) -> None:
            self.calls = 0

        async def check(self, **kwargs) -> list[SafetySignal]:
            self.calls += 1
            return []

    detector = CountingThreat()
    response, upstream = await _post(
        detectors=[detector, NoopCustomDetector()],
        policy={"pipeline": {"enabled": True}},
    )
    assert response.status_code == 200
    upstream.assert_awaited_once()
    assert detector.calls == 2


async def test_residual_detector_exception_keeps_legacy_fail_open_behavior():
    class FailingDetector:
        name = "third_party"

        async def check(self, **kwargs) -> list[SafetySignal]:
            raise RuntimeError("optional detector unavailable")

    response, upstream = await _post(
        detectors=[FailingDetector(), AgentThreatDetector(), NoopCustomDetector()],
        policy={"pipeline": REGEX_ONLY},
    )
    assert response.status_code == 200
    assert response.headers["X-AEP-Enforcement"] == "pass"
    upstream.assert_awaited_once()


async def test_pipeline_enabled_streaming_secret_emits_block_event(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "upstream.test"
        delta = {"model": "gpt-4o", "choices": [{"delta": {"content": EXAMPLE_KEY}}]}
        return httpx.Response(
            200,
            content=f"data: {json.dumps(delta)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    state = ProxyState(
        detectors=[SecretDetector(), AgentThreatDetector(), NoopCustomDetector()],
        policy={"pipeline": {"enabled": True}},
    )
    assert state.pipeline is not None
    assert state.residual_registry is not None
    response = await handle_streaming_request(
        target_url="https://upstream.test/v1/chat/completions",
        body_bytes=b'{"stream": true}',
        headers={"Content-Type": "application/json"},
        call_id="stream-secret",
        input_text="hello",
        registry=state.registry,
        residual_registry=state.residual_registry,
        policy=state.policy,
        pipeline=state.pipeline,
    )
    assert isinstance(response, StreamingResponse)
    body = "".join(
        [
            chunk if isinstance(chunk, str) else bytes(chunk).decode()
            async for chunk in response.body_iterator
        ]
    )
    assert '"aep_safety_block": true' in body
    assert body.index('"aep_safety_block": true') < body.index("data: [DONE]")
