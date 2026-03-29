"""E2E test: signed verdicts through AEP proxy."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time

import httpx

from aceteam_aep.attestation import AepPublicKey, verify_verdict

pytestmark = __import__("pytest").mark.integration


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_signed_proxy_end_to_end(tmp_path):
    """Full flow: keygen → proxy with signing → calls → verify chain."""
    if not os.environ.get("OPENAI_API_KEY"):
        __import__("pytest").skip("OPENAI_API_KEY not set")

    # 1. Generate keypair
    key_dir = tmp_path / "keys"
    subprocess.run(
        ["uv", "run", "aceteam-aep", "keygen", "--output", str(key_dir)],
        check=True,
        capture_output=True,
    )
    assert (key_dir / "aep.key").exists()
    assert (key_dir / "aep.pub").exists()

    # 2. Start proxy with signing
    port = _find_free_port()
    proc = subprocess.Popen(
        [
            "uv", "run", "aceteam-aep", "proxy",
            "--port", str(port),
            "--sign-key", str(key_dir / "aep.key"),
            "--signer-id", "proxy:integration-test",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait for proxy
        base_url = f"http://localhost:{port}"
        for _ in range(30):
            try:
                r = httpx.get(f"{base_url}/aep/api/state", timeout=1)
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

        # 3. Make 3 calls
        collected_chain = []
        for i in range(3):
            r = httpx.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": f"Say the number {i}"}],
                },
                timeout=30,
            )
            assert r.status_code == 200

            # Check attestation headers present
            assert "x-aep-verdict-hash" in r.headers
            assert "x-aep-signature" in r.headers
            assert r.headers["x-aep-signer-id"] == "proxy:integration-test"
            assert r.headers["x-aep-chain-height"] == str(i)
            assert "x-aep-chain-hash" in r.headers

            # Collect chain entry for verification
            collected_chain.append({
                "call_id": r.headers["x-aep-call-id"],
                "action": r.headers["x-aep-enforcement"],
                "verdict_hash": r.headers["x-aep-verdict-hash"],
                "signature": r.headers["x-aep-signature"],
                "chain_height": int(r.headers["x-aep-chain-height"]),
                "chain_hash": r.headers["x-aep-chain-hash"],
            })

        # 4. Verify chain
        pub_key = AepPublicKey.load(key_dir / "aep.pub")

        # Verify individual signatures
        for entry in collected_chain:
            assert verify_verdict(
                entry["verdict_hash"],
                entry["signature"],
                pub_key,
            ), f"Signature invalid at height {entry['chain_height']}"

        # 5. Write chain to JSONL and verify via CLI
        chain_file = tmp_path / "audit.jsonl"
        with open(chain_file, "w") as f:
            for entry in collected_chain:
                f.write(json.dumps(entry) + "\n")

        result = subprocess.run(
            [
                "uv", "run", "aceteam-aep", "verify",
                "--pub-key", str(key_dir / "aep.pub"),
                "--chain", str(chain_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Chain VALID" in result.stdout

    finally:
        proc.terminate()
        proc.wait(timeout=5)
