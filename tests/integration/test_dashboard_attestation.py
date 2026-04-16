"""E2E test: dashboard shows attestation and Trust Engine data."""

from __future__ import annotations

import os
import socket
import subprocess
import time

import httpx

pytestmark = __import__("pytest").mark.integration


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_dashboard_attestation_data(tmp_path):
    """Signed proxy populates attestation in dashboard state API."""
    if not os.environ.get("OPENAI_API_KEY"):
        __import__("pytest").skip("OPENAI_API_KEY not set")

    # 1. Generate keypair
    key_dir = tmp_path / "keys"
    subprocess.run(
        ["uv", "run", "aceteam-aep", "keygen", "--output", str(key_dir)],
        check=True,
        capture_output=True,
    )

    # 2. Start signed proxy
    port = _find_free_port()
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "aceteam-aep",
            "proxy",
            "--port",
            str(port),
            "--sign-key",
            str(key_dir / "aep.key"),
            "--signer-id",
            "proxy:dashboard-test",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        base_url = f"http://localhost:{port}"
        for _ in range(30):
            try:
                r = httpx.get(f"{base_url}/dashboard/api/state", timeout=1)
                if r.status_code == 200:
                    break
            except httpx.ConnectError:
                time.sleep(0.5)
        else:
            proc.kill()
            raise RuntimeError("Proxy failed to start")

        api_key = os.environ["OPENAI_API_KEY"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 3. Check state before any calls — attestation should be enabled but empty
        state = httpx.get(f"{base_url}/dashboard/api/state", timeout=5).json()
        assert state["attestation"]["enabled"] is True
        assert state["attestation"]["signer_id"] == "proxy:dashboard-test"
        assert state["attestation"]["chain_height"] == 0
        assert state["attestation"]["latest_hash"] is None

        # 4. Make a call
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hi"}],
            },
            timeout=30,
        )
        assert r.status_code == 200

        # 5. Check state after call — chain should have 1 entry
        state = httpx.get(f"{base_url}/dashboard/api/state", timeout=5).json()
        assert state["attestation"]["chain_height"] == 1
        assert state["attestation"]["latest_hash"] is not None
        assert state["attestation"]["latest_hash"].startswith("sha256:")

        # 6. Verify response headers have attestation
        assert "x-aep-verdict-hash" in r.headers
        assert "x-aep-signature" in r.headers
        assert r.headers["x-aep-signer-id"] == "proxy:dashboard-test"
        assert r.headers["x-aep-chain-height"] == "0"

        # 7. Make second call — chain grows
        r2 = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
            timeout=30,
        )
        assert r2.status_code == 200
        assert r2.headers["x-aep-chain-height"] == "1"

        state = httpx.get(f"{base_url}/dashboard/api/state", timeout=5).json()
        assert state["attestation"]["chain_height"] == 2

        # 8. Dashboard HTML loads
        dash = httpx.get(f"{base_url}/dashboard/", timeout=5)
        assert dash.status_code == 200
        assert "attestation-section" in dash.text
        assert "Attestation (Signed Verdicts)" in dash.text

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_dashboard_no_attestation_without_signing(tmp_path):
    """Proxy without --sign-key has attestation=null in state."""
    if not os.environ.get("OPENAI_API_KEY"):
        __import__("pytest").skip("OPENAI_API_KEY not set")

    port = _find_free_port()
    proc = subprocess.Popen(
        ["uv", "run", "aceteam-aep", "proxy", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        base_url = f"http://localhost:{port}"
        for _ in range(30):
            try:
                r = httpx.get(f"{base_url}/dashboard/api/state", timeout=1)
                if r.status_code == 200:
                    break
            except httpx.ConnectError:
                time.sleep(0.5)
        else:
            proc.kill()
            raise RuntimeError("Proxy failed to start")

        state = httpx.get(f"{base_url}/dashboard/api/state", timeout=5).json()
        assert state["attestation"] is None

    finally:
        proc.terminate()
        proc.wait(timeout=5)
