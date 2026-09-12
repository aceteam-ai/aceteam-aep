"""Tests for AEP attestation — signed verdicts and audit chains."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from aceteam_aep.attestation import (
    AepPrivateKey,
    AttestationEngine,
    _genesis_link,  # internal, used to pin the fixture
    build_statement,
    canonical_json,
    generate_keypair,
    statement_digest,
    verdict_hash,
    verify_chain,
    verify_chain_report,
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
        """Legacy chain_hash/signature are cosmetic only in format_version 2 —
        tampering them alone does not affect the strict guarantee, since
        verify_chain never trusts them. The authenticated equivalent is
        prev_link; tampering that breaks verification."""
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)

        for i in range(3):
            engine.sign_verdict(call_id=f"call-{i}", action="pass", signals=[])

        chain = engine.chain
        # Legacy field alone: no effect on the authenticated result.
        chain[1]["chain_hash"] = "sha256:" + "a" * 64
        assert verify_chain(chain, public)

        # Authenticated link: breaks verification.
        chain[1]["prev_link"] = "sha256:" + "a" * 64
        assert not verify_chain(chain, public)

    def test_tampered_signature_fails(self):
        """Legacy signature (over the narrow verdict_hash) is cosmetic only
        in format_version 2. The authenticated equivalent is
        statement_signature; tampering that breaks verification."""
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)

        for i in range(3):
            engine.sign_verdict(call_id=f"call-{i}", action="pass", signals=[])

        chain = engine.chain
        # Legacy field alone: no effect on the authenticated result.
        chain[1]["signature"] = "ed25519:" + "b" * 128
        assert verify_chain(chain, public)

        # Authenticated signature: breaks verification.
        chain[1]["statement_signature"] = "ed25519:" + "b" * 128
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


class TestCanonicalBytesFixture:
    """Pin the exact canonical bytes and digest for one statement.

    A drift in field ordering, separators, null-omission, or the genesis
    link formula should be caught here, in one place, rather than only as
    a downstream digest mismatch.
    """

    def test_pinned_canonical_bytes_and_digest(self):
        genesis = _genesis_link("fixture-exec-0000")
        assert genesis == (
            "sha256:6f438792dce3f647fd6bbcaf1d4feb062954b491fd188ae9632e8cced1c6d95a"
        )

        statement = build_statement(
            signer_id="proxy:fixture",
            execution_id="fixture-exec-0000",
            call_id="call-fixture",
            action="block",
            signals=[{"signal_type": "pii", "severity": "high"}],
            timestamp="2026-03-29T00:00:00+00:00",
            sequence=0,
            prev_link=genesis,
            confidence=0.73,
        )
        raw = canonical_json(statement)
        assert raw == (
            b'{"action":"block","call_id":"call-fixture","confidence":0.73,'
            b'"execution_id":"fixture-exec-0000","format_version":2,'
            b'"prev_link":"sha256:6f438792dce3f647fd6bbcaf1d4feb062954b491fd188ae9632e8cced1c6d95a",'
            b'"schema":"aceteam.aep.verdict-statement","sequence":0,'
            b'"signals":[{"severity":"high","signal_type":"pii"}],'
            b'"signer_id":"proxy:fixture","timestamp":"2026-03-29T00:00:00+00:00"}'
        )
        assert statement_digest(statement) == (
            "sha256:1d24a2f995f5fa2ffc8d52cdb7ef182b338723d61c40baccb5d2dd334c65ce2a"
        )

    def test_null_commitments_and_confidence_omitted_from_bytes(self):
        statement = build_statement(
            signer_id="s",
            execution_id="e",
            call_id="c",
            action="pass",
            signals=[],
            timestamp="t",
            sequence=0,
            prev_link="sha256:" + "0" * 64,
        )
        raw = canonical_json(statement)
        assert b"confidence" not in raw
        assert b"input_commitment" not in raw
        assert b"output_commitment" not in raw
        assert b"policy_commitment" not in raw


class TestNonFiniteAndMalformedRejected:
    def test_nan_confidence_rejected_at_sign_time(self):
        private, _ = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        with pytest.raises(ValueError):
            engine.sign_verdict(call_id="x", action="pass", signals=[], confidence=float("nan"))

    def test_infinite_confidence_rejected_at_sign_time(self):
        private, _ = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        with pytest.raises(ValueError):
            engine.sign_verdict(call_id="x", action="pass", signals=[], confidence=float("inf"))

    def test_bool_sequence_rejected(self):
        with pytest.raises(ValueError):
            build_statement(
                signer_id="s",
                execution_id="e",
                call_id="c",
                action="pass",
                signals=[],
                timestamp="t",
                sequence=True,  # bool is not a valid sequence, even though isinstance(True, int)
                prev_link="sha256:" + "0" * 64,
            )

    def test_malformed_entry_fails_closed_without_raising(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        engine.sign_verdict(call_id="only", action="pass", signals=[])
        chain = engine.chain
        chain[0]["statement_signature"] = "ed25519:not-hex-zz"
        result = verify_chain_report(chain, public)
        assert result.valid is False
        assert result.status == "invalid"

    def test_missing_required_field_fails_closed(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        engine.sign_verdict(call_id="only", action="pass", signals=[])
        chain = engine.chain
        del chain[0]["sequence"]
        result = verify_chain_report(chain, public)
        assert result.valid is False
        assert result.status == "legacy_unsupported"


class TestLegacyChainsReportedUnsupported:
    """format_version 1 entries (the pre-#9657 shape) must not be silently
    upgraded to the strict full-chain guarantee."""

    def test_hand_built_legacy_v1_entry_is_unsupported(self):
        private, public = generate_keypair()
        v_hash = verdict_hash(
            call_id="legacy-1", action="pass", signals=[], timestamp="2026-01-01T00:00:00Z"
        )
        sig = private.sign(bytes.fromhex(v_hash.removeprefix("sha256:")))
        v_hash_bytes = bytes.fromhex(v_hash.removeprefix("sha256:"))
        legacy_chain_hash = f"sha256:{hashlib.sha256(b'aep-genesis' + v_hash_bytes).hexdigest()}"
        legacy_entry = {
            "call_id": "legacy-1",
            "action": "pass",
            "timestamp": "2026-01-01T00:00:00Z",
            "verdict_hash": v_hash,
            "signature": f"ed25519:{sig.hex()}",
            "chain_height": 0,
            "chain_hash": legacy_chain_hash,
            "prev_chain_hash": "",
        }
        result = verify_chain_report([legacy_entry], public)
        assert result.valid is False
        assert result.status == "legacy_unsupported"
        # Legacy entries can still satisfy the narrow single-verdict primitive.
        assert verify_verdict(legacy_entry["verdict_hash"], legacy_entry["signature"], public)

    def test_mixed_legacy_and_v2_chain_is_unsupported(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        engine.sign_verdict(call_id="v2-entry", action="pass", signals=[])
        chain = engine.chain
        del chain[0]["format_version"]  # simulate a legacy entry mixed in
        result = verify_chain_report(chain, public)
        assert result.status == "legacy_unsupported"


class TestAppendixAAttackReproductions:
    """Reproduce both #9652 Appendix A attacks against the format_version 2
    chain and show they now fail closed."""

    def test_attack_1_changed_displayed_action_now_rejected(self):
        """Original bug: flipping entry[1]['action'] from 'block' to 'pass'
        still verified True, because the exported chain didn't carry enough
        of the originally-signed fields to recompute the verdict hash."""
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private, signer_id="proxy:local-review")
        for call, action in [("a", "pass"), ("b", "block"), ("c", "pass")]:
            engine.sign_verdict(call_id=call, action=action, signals=[])

        original = engine.chain
        assert verify_chain(original, public) is True

        changed = copy.deepcopy(original)
        changed[1]["action"] = "pass"  # attacker flips a displayed BLOCK to PASS

        result = verify_chain_report(changed, public)
        assert result.valid is False
        assert result.status == "invalid"
        assert result.failed_index == 1
        assert "signature" in result.reason or "malformed" in result.reason

    def test_attack_2_omitted_middle_entry_now_rejected(self):
        """Original bug: dropping the middle entry and recomputing the
        unsigned chain_hash/prev_chain_hash/chain_height links (no signing
        key needed) still verified True."""
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private, signer_id="proxy:local-review")
        for call, action in [("a", "pass"), ("b", "block"), ("c", "pass")]:
            engine.sign_verdict(call_id=call, action=action, signals=[])

        original = engine.chain
        omitted = copy.deepcopy([original[0], original[2]])

        # Attacker recomputes the *legacy, unsigned* links exactly as
        # #9652 Appendix A does — this alone used to be sufficient.
        previous = ""
        for i, row in enumerate(omitted):
            row["chain_height"] = i
            row["prev_chain_hash"] = previous
            raw = bytes.fromhex(row["verdict_hash"].removeprefix("sha256:"))
            prefix = b"aep-genesis" if i == 0 else bytes.fromhex(previous.removeprefix("sha256:"))
            row["chain_hash"] = "sha256:" + hashlib.sha256(prefix + raw).hexdigest()
            previous = row["chain_hash"]

        result = verify_chain_report(omitted, public)
        assert result.valid is False
        # sequence (still 0, 2 — untouched, since the attacker only forged
        # the legacy unsigned fields) is caught before any crypto check.
        assert result.status == "invalid"
        assert result.failed_index == 1
        assert "sequence" in result.reason

    def test_attack_2_variant_renumbered_sequence_still_rejected(self):
        """Stronger variant: the attacker also renumbers the authenticated
        ``sequence``/``prev_link`` fields to match the new position. This
        passes the structural checks but the statement signature — signed
        over the *original* sequence/prev_link — no longer matches."""
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private, signer_id="proxy:local-review")
        for call, action in [("a", "pass"), ("b", "block"), ("c", "pass")]:
            engine.sign_verdict(call_id=call, action=action, signals=[])

        original = engine.chain
        omitted = copy.deepcopy([original[0], original[2]])
        omitted[1]["sequence"] = 1
        omitted[1]["prev_link"] = omitted[0]["statement_digest"]

        result = verify_chain_report(omitted, public)
        assert result.valid is False
        assert result.status == "invalid"
        assert result.failed_index == 1
        assert "signature" in result.reason

    def test_attack_2_reordered_entries_rejected(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        for call in ["a", "b", "c"]:
            engine.sign_verdict(call_id=call, action="pass", signals=[])

        chain = engine.chain
        reordered = [chain[0], chain[2], chain[1]]
        result = verify_chain_report(reordered, public)
        assert result.valid is False


class TestMutationIsolation:
    """One mutation per bound field: each must independently break
    verification, and only that field's mutation should."""

    def _signed_pair(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private, signer_id="proxy:mutation-test")
        engine.sign_verdict(
            call_id="call-0",
            action="pass",
            signals=[{"signal_type": "pii", "severity": "low"}],
            confidence=0.73,
        )
        engine.sign_verdict(call_id="call-1", action="block", signals=[], confidence=0.1)
        return engine, public

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("call_id", "attacker-call-id"),
            ("timestamp", "1999-01-01T00:00:00Z"),
            ("signer_id", "proxy:attacker"),
            ("execution_id", "attacker-session"),
        ],
    )
    def test_mutating_bound_field_breaks_verification(self, field, value):
        engine, public = self._signed_pair()
        chain = engine.chain
        assert verify_chain(chain, public)
        chain[1][field] = value
        assert not verify_chain(chain, public)

    def test_mutating_signals_breaks_verification(self):
        engine, public = self._signed_pair()
        chain = engine.chain
        chain[1]["signals"] = [{"signal_type": "pii", "severity": "high"}]
        assert not verify_chain(chain, public)

    def test_mutating_confidence_breaks_verification(self):
        engine, public = self._signed_pair()
        chain = engine.chain
        chain[1]["confidence"] = 0.99
        assert not verify_chain(chain, public)

    def test_confidence_present_to_none_breaks_verification(self):
        engine, public = self._signed_pair()
        chain = engine.chain
        chain[0]["confidence"] = None
        assert not verify_chain(chain, public)

    def test_mutating_commitment_breaks_verification(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        engine.sign_verdict(
            call_id="call-0",
            action="pass",
            signals=[],
            output_commitment="sha256:" + "1" * 64,
        )
        chain = engine.chain
        assert verify_chain(chain, public)
        chain[0]["output_commitment"] = "sha256:" + "2" * 64
        assert not verify_chain(chain, public)

    def test_commitments_absent_by_default_unbound(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        engine.sign_verdict(call_id="call-0", action="pass", signals=[])
        entry = engine.chain[0]
        assert "output_commitment" not in entry or entry.get("output_commitment") is None
        assert verify_chain(engine.chain, public)


class TestCrossSessionSplicing:
    def test_two_sessions_same_key_cannot_be_spliced(self):
        private, public = generate_keypair()
        session_a = AttestationEngine(_private_key=private, signer_id="proxy:same-key")
        session_b = AttestationEngine(_private_key=private, signer_id="proxy:same-key")

        session_a.sign_verdict(call_id="a-0", action="pass", signals=[])
        session_b.sign_verdict(call_id="b-0", action="pass", signals=[])
        session_a.sign_verdict(call_id="a-1", action="pass", signals=[])

        assert session_a.execution_id != session_b.execution_id

        spliced = [session_a.chain[0], session_b.chain[0]]
        result = verify_chain_report(spliced, public)
        assert result.valid is False
        assert result.status == "invalid"
        assert "execution" in result.reason or "session" in result.reason

    def test_expected_execution_id_pins_session(self):
        private, public = generate_keypair()
        engine = AttestationEngine(_private_key=private)
        engine.sign_verdict(call_id="only", action="pass", signals=[])

        assert verify_chain(engine.chain, public, expected_execution_id=engine.execution_id)
        assert not verify_chain(engine.chain, public, expected_execution_id="someone-elses-session")

    def test_new_engine_sessions_have_distinct_identity(self):
        private, _ = generate_keypair()
        ids = {AttestationEngine(_private_key=private).execution_id for _ in range(20)}
        assert len(ids) == 20
