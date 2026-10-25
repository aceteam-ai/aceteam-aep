"""Secret matches must stay out of client metadata and blocked responses."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
from starlette.testclient import TestClient

from aceteam_aep.attestation import generate_keypair
from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetyDetector, SafetySignal
from aceteam_aep.safety.secrets import SecretDetector

_SECRET = "AKIA" + "IOSFODNN7EXAMPLE"


def _assert_no_secret_fragment(value: str) -> None:
    assert all(_SECRET[i : i + 4] not in value for i in range(len(_SECRET) - 3))


def _response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        },
    )


def _request(client: TestClient, content: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": content}]},
        headers={"Authorization": "Bearer sk-test"},
    )


def test_input_secret_block_has_no_secret_in_body_or_headers() -> None:
    client = TestClient(create_proxy_app(detectors=[SecretDetector()], dashboard=False))
    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as upstream:
        response = _request(client, f"Here is {_SECRET}")
    assert response.status_code == 400
    upstream.assert_not_called()
    _assert_no_secret_fragment(response.text)
    _assert_no_secret_fragment(str(dict(response.headers)))


def test_output_secret_block_has_no_secret_in_body_or_headers() -> None:
    client = TestClient(create_proxy_app(detectors=[SecretDetector()], dashboard=False))
    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as upstream_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(return_value=_response(f"The key is {_SECRET}"))
        upstream_cls.return_value = upstream
        response = _request(client, "hello")
    assert response.status_code == 400
    _assert_no_secret_fragment(response.text)
    _assert_no_secret_fragment(str(dict(response.headers)))


def test_upstream_error_echoing_secret_is_suppressed() -> None:
    client = TestClient(create_proxy_app(detectors=[SecretDetector()], dashboard=False))
    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as upstream_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(return_value=httpx.Response(400, text=f"invalid {_SECRET}"))
        upstream_cls.return_value = upstream
        response = _request(client, "hello")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "safety_block"
    _assert_no_secret_fragment(response.text)
    _assert_no_secret_fragment(str(dict(response.headers)))


def test_signed_flag_header_excludes_signal_detail() -> None:
    class DetailDetector(SafetyDetector):
        name = "detail_test"

        async def check(self, *, call_id: str, **_kwargs) -> list[SafetySignal]:
            return [
                SafetySignal(
                    signal_type="detail_test", severity="medium", call_id=call_id, detail=_SECRET
                )
            ]

    private_key, _ = generate_keypair()
    client = TestClient(
        create_proxy_app(detectors=[DetailDetector()], dashboard=False, sign_key=private_key)
    )
    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as upstream_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(return_value=_response("safe output"))
        upstream_cls.return_value = upstream
        response = _request(client, "hello")
    assert response.status_code == 200
    assert "x-aep-signals" in response.headers
    assert all("detail" not in signal for signal in json.loads(response.headers["x-aep-signals"]))
    _assert_no_secret_fragment(str(dict(response.headers)))


async def test_secret_signal_detail_has_no_secret_fragment() -> None:
    signals = await SecretDetector().check(
        input_text=f"token {_SECRET}", output_text=f"reply {_SECRET}", call_id="call"
    )
    assert len(signals) == 2
    for signal in signals:
        _assert_no_secret_fragment(signal.detail)
