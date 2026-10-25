"""Private PAW compiler boundary. Every transport is stubbed; no hosted calls."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import programasweights as paw
import programasweights.client as paw_client
import programasweights.config as paw_config
import pytest

from aceteam_aep.safety.custom import (
    AsyncPawFunction,
    CustomPolicy,
    PrivateCompileUnavailableError,
    require_private_compile_endpoint,
)


@pytest.fixture(autouse=True)
def _isolated_paw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def refuse_socket(_self: socket.socket, _address: object) -> None:
        raise AssertionError("real network access is forbidden in PAW tests")

    monkeypatch.setattr(socket.socket, "connect", refuse_socket)
    monkeypatch.setattr(paw_config, "_CONFIG_FILE", tmp_path / "paw-config.json")
    monkeypatch.setenv("PAW_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PAW_API_URL", raising=False)
    monkeypatch.delenv("PAW_API_KEY", raising=False)


def _stub_compiler(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    posts: list[tuple[str, dict]] = []

    def fake_post(url: str, *, json: dict, **_kwargs: object) -> httpx.Response:
        posts.append((url, json))
        return httpx.Response(
            200,
            json={"program_id": "0123456789abcdef", "status": "compiled"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(paw_client.httpx, "post", fake_post)
    monkeypatch.setattr(paw, "function", lambda *_args, **_kwargs: lambda _text: "Y")
    return posts


async def test_compile_sends_public_false_to_private_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts = _stub_compiler(monkeypatch)
    monkeypatch.setenv("PAW_API_URL", "http://paw-compile.test:9321/")
    fn = AsyncPawFunction("tenant policy text")
    await fn.prepare()
    assert posts == [
        (
            "http://paw-compile.test:9321/api/v1/compile",
            {"spec": "tenant policy text", "public": False},
        )
    ]


async def test_refuses_hosted_default_without_request(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = _stub_compiler(monkeypatch)
    with pytest.raises(PrivateCompileUnavailableError):
        await AsyncPawFunction("private rule").prepare()
    assert posts == []


@pytest.mark.parametrize(
    "url",
    (
        "https://programasweights.com",
        "https://programasweights.com/",
        "https://www.programasweights.com",
        "https://api.programasweights.com.",
    ),
)
async def test_refuses_explicit_hosted_urls(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    posts = _stub_compiler(monkeypatch)
    monkeypatch.setenv("PAW_API_URL", url)
    with pytest.raises(PrivateCompileUnavailableError):
        await AsyncPawFunction("private rule").prepare()
    assert posts == []


async def test_refuses_hosted_url_from_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posts = _stub_compiler(monkeypatch)
    (tmp_path / "paw-config.json").write_text(
        json.dumps({"api_url": "https://www.programasweights.com"})
    )
    with pytest.raises(PrivateCompileUnavailableError):
        await AsyncPawFunction("private rule").prepare()
    assert posts == []


@pytest.mark.parametrize("url", ("", "not-a-url", "http://", "ftp://paw-compile.test"))
def test_refuses_missing_or_malformed_urls(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setattr(paw, "get_api_url", lambda: url)
    with pytest.raises(PrivateCompileUnavailableError):
        require_private_compile_endpoint()


async def test_public_cannot_be_set_through_policy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts = _stub_compiler(monkeypatch)
    monkeypatch.setenv("PAW_API_URL", "http://paw-compile.test")
    policy = CustomPolicy.model_validate({"name": "Private", "rule": "tenant text", "public": True})
    assert "public" not in policy.model_dump()
    await policy._paw.prepare()
    assert len(posts) == 1
    assert posts[0][1]["public"] is False


def test_guard_treats_missing_sdk_as_unavailable() -> None:
    with (
        patch.dict(sys.modules, {"programasweights": None}),
        pytest.raises(PrivateCompileUnavailableError, match="custom-policies extra"),
    ):
        require_private_compile_endpoint()


async def test_private_endpoint_error_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAW_API_URL", "http://paw-compile.test")
    calls: list[str] = []

    def fail_post(url: str, **_kwargs: object) -> httpx.Response:
        calls.append(url)
        raise httpx.ConnectError("private compiler unavailable")

    monkeypatch.setattr(paw_client.httpx, "post", fail_post)
    with pytest.raises(httpx.ConnectError):
        await AsyncPawFunction("tenant rule").prepare()
    assert calls == ["http://paw-compile.test/api/v1/compile"]
