"""Tests for AEP attestation — signed verdicts and Merkle chains."""

from __future__ import annotations

import json

from aceteam_aep.attestation import (
    AepPrivateKey,
    AttestationEngine,
    canonical_json,
    generate_keypair,
    verdict_hash,
    verify_chain,
    verify_verdict,
)


class TestCanonicalJson:
    def test_sorted_keys(self):
        data = {"z": 1, "a": 2, "m": 3}
        result = json.loads(canonical_json(data))
        assert list(result.keys()) == ["a", "m", "z"]

    def test_omits_none(self):
        data = {"a": 1, "b": None, "c": 3}
        result = json.loads(canonical_json(data))
        assert "b" not in result

    def test_minimal_encoding(self):
        data = {"key": "value"}
        raw = canonical_json(data)
        assert b" " not in raw  # no whitespace
        assert raw == b'{"key":"value"}'

    def test_nested_sorting(self):
        data = {"outer": {"z": 1, "a": 2}}
        result = json.loads(canonical_json(data))
        assert list(result["outer"].keys()) == ["a", "z"]

    def test_deterministic(self):
        data = {"b": 2, "a": 1, "c": [3, 2, 1]}
        assert canonical_json(data) == canonical_json(data)


class TestVerdictHash:
    def test_produces_sha256_prefix(self):
        h = verdict_hash(
            call_id="test",
            action="pass",
            signals=[],
            timestamp="2026-03-29T00:00:00Z",
        )
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # prefix + 64 hex chars

    def test_deterministic(self):
        kwargs = {
            "call_id": "abc",
            "action": "block",
            "signals": [{"signal_type": "pii", "severity": "high"}],
            "timestamp": "2026-03-29T00:00:00Z",
            "confidence": 0.73,
        }
        assert verdict_hash(**kwargs) == verdict_hash(**kwargs)

    def test_different_inputs_different_hashes(self):
        base = {
            "call_id": "abc",
            "action": "pass",
            "signals": [],
            "timestamp": "2026-03-29T00:00:00Z",
        }
        h1 = verdict_hash(**base)
        h2 = verdict_hash(**{**base, "action": "block"})
        assert h1 != h2


class TestKeypair:
    def test_generate_and_sign(self):
        private, public = generate_keypair()
        data = b"hello world"
        sig = private.sign(data)
        assert public.verify(sig, data)

    def test_wrong_data_fails_verification(self):
        private, public = generate_keypair()
        sig = private.sign(b"hello")
        assert not public.verify(sig, b"wrong")

    def test_save_and_load(self, tmp_path):
        private, public = generate_keypair()
        private.save(tmp_path / "test.key")
        public.save(tmp_path / "test.pub")

        loaded_private = AepPrivateKey.load(tmp_path / "test.key")
        loaded_public = loaded_private.public_key

        data = b"test data"
        sig = loaded_private.sign(data)
        assert loaded_public.verify(sig, data)

    def test_public_key_to_base64(self):
        _, public = generate_keypair()
        b64 = public.to_base64()
        assert len(b64) == 44  # 32 bytes base64 encoded


class TestAttestationEngine:
    def test_sign_verdict_returns_headers(self):
        private, _ = generate_keypair()
        engine = AttestationEngine(_private_key=private, signer_id="proxy:test")

        headers = engine.sign_verdict(
            call_id="test-001",
            action="pass",
            signals=[],
        )

        assert "X-AEP-Verdict-Hash" in headers
        assert "X-AEP-Signature" in headers
        assert headers["X-AEP-Signer-Id"] == "proxy:test"
        assert headers["X-AEP-Chain-Height"] == "0"
        assert "X-AEP-Chain-Hash" in headers

    def test_chain_height_increments(self):
        private, _ = generate_keypair()
        engine = AttestationEngine(_private_key=private)

        h1 = engine.sign_verdict(call_id="1", action="pass", signals=[])
        h2 = engine.sign_verdict(call_id="2", action="pass", signals=[])
        h3 = engine.sign_verdict(call_id="3", action="block", signals=[{"t": "pii"}])

        assert h1["X-AEP-Chain-Height"] == "0"
        assert h2["X-AEP-Chain-Height"] == "1"
        assert h3["X-AEP-Chain-Height"] == "2"
        assert engine.chain_height == 3

    def test_chain_hashes_differ(self):
        private, _ = generate_keypair()
        engine = AttestationEngine(_private_key=private)

        h1 = engine.sign_verdict(call_id="1", action="pass", signals=[])
        h2 = engine.sign_verdict(call_id="2", action="pass", signals=[])

        assert h1["X-AEP-Chain-Hash"] != h2["X-AEP-Chain-Hash"]

    def test_signatures_verify(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)

        headers = engine.sign_verdict(
            call_id="verify-me",
            action="block",
            signals=[{"signal_type": "agent_threat", "severity": "high"}],
            confidence=0.42,
        )

        assert verify_verdict(
            headers["X-AEP-Verdict-Hash"],
            headers["X-AEP-Signature"],
            public,
        )


class TestChainVerification:
    def test_valid_chain_verifies(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private, signer_id="proxy:test")

        for i in range(5):
            engine.sign_verdict(
                call_id=f"call-{i}",
                action="pass" if i % 2 == 0 else "flag",
                signals=[{"type": "cost_anomaly"}] if i % 2 else [],
            )

        assert verify_chain(engine.chain, public)

    def test_tampered_chain_hash_fails(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)

        for i in range(3):
            engine.sign_verdict(call_id=f"call-{i}", action="pass", signals=[])

        chain = engine.chain
        # Tamper with middle chain hash — breaks the chain
        chain[1]["chain_hash"] = "sha256:" + "a" * 64

        assert not verify_chain(chain, public)

    def test_tampered_signature_fails(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)

        for i in range(3):
            engine.sign_verdict(call_id=f"call-{i}", action="pass", signals=[])

        chain = engine.chain
        # Tamper with signature
        chain[1]["signature"] = "ed25519:" + "b" * 128

        assert not verify_chain(chain, public)

    def test_empty_chain_verifies(self):
        _, public = generate_keypair()
        assert verify_chain([], public)

    def test_single_entry_chain_verifies(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        engine.sign_verdict(call_id="only", action="pass", signals=[])
        assert verify_chain(engine.chain, public)

    def test_wrong_key_fails(self):
        private1, _ = generate_keypair()
        _, public2 = generate_keypair()
        engine = AttestationEngine(_private_key=private1)
        engine.sign_verdict(call_id="test", action="pass", signals=[])
        assert not verify_chain(engine.chain, public2)
