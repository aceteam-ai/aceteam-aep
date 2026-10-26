"""Version 3 statements bind a structured evaluation without changing v2."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from aceteam_aep.attestation import (
    STATEMENT_VERSION,
    STATEMENT_VERSION_V3,
    AttestationEngine,
    build_statement,
    canonical_json,
    entry_from_headers,
    generate_keypair,
    verify_chain_report,
)
from aceteam_aep.enforcement import (
    DetectorPolicy,
    EnforcementPolicy,
    PipelinePolicy,
    policy_commitment,
)
from aceteam_aep.safety.custom import CustomPolicy, CustomPolicyStore


def evaluation() -> dict:
    return {
        "mode": "pipeline",
        "decided_by": "pipeline:unavailable",
        "phases": [
            {
                "phase": "input",
                "action": "block",
                "pipeline": {
                    "layers": [
                        {
                            "name": "rules",
                            "status": "evaluated",
                            "p_safe": 0.8,
                            "reason_code": None,
                        },
                        {
                            "name": "judge",
                            "status": "unavailable",
                            "p_safe": None,
                            "reason_code": "evaluator_unavailable",
                        },
                    ],
                    "p_safe": None,
                    "calibrated": False,
                    "pass_above": 0.8,
                    "block_below": 0.2,
                    "unavailable": ["judge"],
                },
                "residual": [],
            }
        ],
    }


def signer():
    private, public = generate_keypair()
    return AttestationEngine(_private_key=private, signer_id="proxy:test"), public


def test_v3_evaluation_is_signed():
    engine, public = signer()
    headers = engine.sign_verdict(
        call_id="call-1", action="block", signals=[], evaluation=evaluation()
    )
    assert headers["X-AEP-Format-Version"] == str(STATEMENT_VERSION_V3)
    assert engine.chain[0]["evaluation"] == evaluation()
    assert verify_chain_report(engine.chain, public).status == "valid"

    changed = copy.deepcopy(engine.chain)
    changed[0]["evaluation"]["decided_by"] = "pipeline:aggregate"
    assert verify_chain_report(changed, public).status == "invalid"


def test_v3_layer_status_and_removal_tamper_detected():
    engine, public = signer()
    engine.sign_verdict(call_id="call-1", action="block", signals=[], evaluation=evaluation())
    changed = copy.deepcopy(engine.chain)
    changed[0]["evaluation"]["phases"][0]["pipeline"]["layers"][1]["status"] = "evaluated"
    assert verify_chain_report(changed, public).status == "invalid"
    changed = copy.deepcopy(engine.chain)
    changed[0]["evaluation"]["phases"][0]["pipeline"]["layers"].pop()
    assert verify_chain_report(changed, public).status == "invalid"
    changed = copy.deepcopy(engine.chain)
    del changed[0]["evaluation"]
    assert verify_chain_report(changed, public).status == "invalid"


def test_v3_signal_projection_omits_free_text_detail():
    engine, public = signer()
    headers = engine.sign_verdict(
        call_id="call-1",
        action="block",
        signals=[
            {
                "signal_type": "secret_leak",
                "detector": "secret_scan",
                "severity": "high",
                "score": 0.9,
                "detail": "private finding text",
            }
        ],
        evaluation=evaluation(),
    )
    assert "detail" not in headers["X-AEP-Signals"]
    assert "private finding text" not in json.dumps(engine.chain)
    assert verify_chain_report(engine.chain, public).status == "valid"


def test_v2_chain_still_verifies_with_unchanged_default_headers():
    engine, public = signer()
    headers = engine.sign_verdict(call_id="call-1", action="pass", signals=[])
    assert headers["X-AEP-Format-Version"] == str(STATEMENT_VERSION)
    assert "X-AEP-Evaluation" not in headers
    assert not any("Commitment" in name for name in headers)
    assert "evaluation" not in engine.chain[0]
    assert verify_chain_report(engine.chain, public).status == "valid"


def test_mixed_version_chain_invalid():
    engine, public = signer()
    engine.sign_verdict(call_id="call-1", action="pass", signals=[])
    # A signer keeps one format version for its lifetime, without advancing
    # either chain after a rejected call.
    with pytest.raises(ValueError, match="cannot mix"):
        engine.sign_verdict(call_id="call-2", action="block", signals=[], evaluation=evaluation())
    assert engine.chain_height == 1
    assert verify_chain_report(engine.chain, public).status == "valid"

    # A verifier must still reject a mixed chain assembled from two signers
    # sharing a key and execution identity, even with otherwise valid links.
    second = AttestationEngine(
        _private_key=engine._private_key,
        signer_id=engine.signer_id,
        execution_id=engine.execution_id,
    )
    second._chain_height = engine.chain_height
    second._prev_chain_hash = engine.chain[-1]["chain_hash"]
    second._prev_statement_digest = engine.chain[-1]["statement_digest"]
    second.sign_verdict(call_id="call-2", action="block", signals=[], evaluation=evaluation())
    assert verify_chain_report(engine.chain + second.chain, public).status == "invalid"


def test_v3_engine_rejects_v2_after_first_statement():
    engine, public = signer()
    engine.sign_verdict(call_id="call-1", action="block", signals=[], evaluation=evaluation())
    with pytest.raises(ValueError, match="cannot mix"):
        engine.sign_verdict(call_id="call-2", action="pass", signals=[])
    assert engine.chain_height == 1
    assert verify_chain_report(engine.chain, public).status == "valid"


def test_entry_from_headers_roundtrip_v3():
    engine, public = signer()
    headers = engine.sign_verdict(
        call_id="call-1",
        action="block",
        signals=[],
        evaluation=evaluation(),
        input_commitment="sha256:" + "1" * 64,
        output_commitment="sha256:" + "2" * 64,
        policy_commitment="sha256:" + "3" * 64,
    )
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "block"})
    for name in ("Evaluation", "Input-Commitment", "Output-Commitment", "Policy-Commitment"):
        assert f"X-AEP-{name}" in headers
    entry = entry_from_headers({key.lower(): value for key, value in headers.items()})
    assert entry["evaluation"] == evaluation()
    assert entry["policy_commitment"] == "sha256:" + "3" * 64
    assert verify_chain_report([entry], public).status == "valid"


def test_entry_from_headers_rejects_case_collisions():
    engine, _ = signer()
    headers = engine.sign_verdict(
        call_id="call-1", action="block", signals=[], evaluation=evaluation()
    )
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "block"})
    headers["x-aep-evaluation"] = headers["X-AEP-Evaluation"]
    with pytest.raises(ValueError, match="duplicate attestation header"):
        entry_from_headers(headers)


@pytest.mark.parametrize("header", ["X-AEP-Evaluation", "X-AEP-Signals"])
def test_entry_from_headers_rejects_duplicate_json_keys(header):
    engine, _ = signer()
    headers = engine.sign_verdict(
        call_id="call-1", action="block", signals=[], evaluation=evaluation()
    )
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "block"})
    if header == "X-AEP-Evaluation":
        headers[header] = headers[header].replace(
            '"mode":"pipeline"', '"mode":"pipeline","mode":"pipeline"'
        )
    else:
        headers[header] = '[{"detector":"rules","detector":"rules"}]'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        entry_from_headers(headers)


def test_entry_from_headers_requires_canonical_evaluation_bytes():
    engine, public = signer()
    headers = engine.sign_verdict(
        call_id="call-1", action="block", signals=[], evaluation=evaluation()
    )
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "block"})
    headers["X-AEP-Evaluation"] = json.dumps(json.loads(headers["X-AEP-Evaluation"]))
    with pytest.raises(ValueError, match="canonical JSON"):
        entry_from_headers(headers)

    altered = evaluation()
    altered["decided_by"] = "pipeline:aggregate"
    headers["X-AEP-Evaluation"] = canonical_json(altered).decode("utf-8")
    assert verify_chain_report([entry_from_headers(headers)], public).status == "invalid"


@pytest.mark.parametrize("mutation", ["missing", "number", "json", "shape"])
def test_entry_from_headers_uses_value_error_for_malformed_headers(mutation):
    engine, _ = signer()
    headers = engine.sign_verdict(
        call_id="call-1", action="block", signals=[], evaluation=evaluation()
    )
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "block"})
    if mutation == "missing":
        del headers["X-AEP-Statement-Digest"]
    elif mutation == "number":
        headers["X-AEP-Chain-Height"] = "not an integer"
    elif mutation == "json":
        headers["X-AEP-Evaluation"] = "{"
    else:
        headers["X-AEP-Evaluation"] = '{"phases":null}'
    with pytest.raises(ValueError):
        entry_from_headers(headers)


def test_v2_rejects_unsigned_evaluation_in_entry_and_headers():
    engine, public = signer()
    headers = engine.sign_verdict(call_id="call-1", action="pass", signals=[])
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "pass"})
    with pytest.raises(ValueError, match="unsigned evaluation"):
        entry_from_headers({**headers, "X-AEP-Evaluation": "{}"})
    tampered = copy.deepcopy(engine.chain)
    tampered[0]["evaluation"] = evaluation()
    assert verify_chain_report(tampered, public).status == "invalid"


def test_v2_commitment_headers_roundtrip_when_supplied():
    engine, public = signer()
    headers = engine.sign_verdict(
        call_id="call-1",
        action="pass",
        signals=[],
        input_commitment="sha256:" + "1" * 64,
    )
    assert headers["X-AEP-Input-Commitment"] == "sha256:" + "1" * 64
    assert "X-AEP-Evaluation" not in headers
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "pass"})
    assert verify_chain_report([entry_from_headers(headers)], public).status == "valid"


def test_v3_null_signal_score_roundtrips_through_canonical_headers():
    engine, public = signer()
    headers = engine.sign_verdict(
        call_id="call-1",
        action="block",
        signals=[
            {
                "signal_type": "policy_hit",
                "detector": "rules",
                "severity": "high",
                "score": None,
            }
        ],
        evaluation=evaluation(),
    )
    assert json.loads(headers["X-AEP-Signals"])[0]["score"] is None
    headers.update({"X-AEP-Call-Id": "call-1", "X-AEP-Enforcement": "block"})
    entry = entry_from_headers(headers)
    assert entry["signals"][0]["score"] is None
    assert verify_chain_report([entry], public).status == "valid"

    # An HTTP archive may canonicalize JSON and omit nullable object members.
    canonicalized = dict(headers)
    signal_without_nulls = {
        key: value
        for key, value in json.loads(headers["X-AEP-Signals"])[0].items()
        if value is not None
    }
    canonicalized["X-AEP-Signals"] = json.dumps([signal_without_nulls])
    assert verify_chain_report([entry_from_headers(canonicalized)], public).status == "valid"


def test_evaluation_rejects_unknown_nonfinite_and_free_text():
    for mutate in (
        lambda e: e.update({"extra": "not signed"}),
        lambda e: e["phases"][0]["pipeline"].update({"p_safe": float("nan")}),
        lambda e: e["phases"][0]["pipeline"].update({"p_safe": 10**1000}),
        lambda e: e["phases"][0]["pipeline"]["layers"][1].update(
            {"reason_code": "raw exception text"}
        ),
    ):
        record = evaluation()
        mutate(record)
        with pytest.raises((ValueError, TypeError)):
            build_statement(
                signer_id="proxy:test",
                execution_id="run",
                call_id="call",
                action="block",
                signals=[],
                timestamp="2026-01-01T00:00:00Z",
                sequence=0,
                prev_link="sha256:" + "0" * 64,
                evaluation=record,
            )


def test_evaluation_rejects_unavailable_list_mismatch():
    record = evaluation()
    record["phases"][0]["pipeline"]["unavailable"] = []
    with pytest.raises(ValueError, match="unavailable must list"):
        build_statement(
            signer_id="proxy:test",
            execution_id="run",
            call_id="call",
            action="block",
            signals=[],
            timestamp="2026-01-01T00:00:00Z",
            sequence=0,
            prev_link="sha256:" + "0" * 64,
            evaluation=record,
        )


def test_policy_commitment_stable_and_tracks_effective_changes():
    base = EnforcementPolicy()
    ordered = EnforcementPolicy(
        block_on=frozenset(reversed(sorted(base.block_on))),
        flag_on=base.flag_on,
        allow_types=base.allow_types,
        default_action=base.default_action,
        overrides={"b": DetectorPolicy(action="flag"), "a": DetectorPolicy(action="block")},
        pipeline=PipelinePolicy(
            enabled=True, pass_below=0.3, block_above=0.7, layers=[{"name": "rules"}]
        ),
    )
    reordered = replace(ordered, overrides=dict(reversed(list(ordered.overrides.items()))))
    assert policy_commitment(ordered, []) == policy_commitment(reordered, [])
    assert policy_commitment(ordered, []) != policy_commitment(
        replace(ordered, pipeline=replace(ordered.pipeline, block_above=0.8)), []
    )
    assert policy_commitment(ordered, []) != policy_commitment(
        replace(ordered, overrides={**ordered.overrides, "a": DetectorPolicy(action="flag")}), []
    )


def test_policy_commitment_normalizes_numeric_thresholds():
    integers = EnforcementPolicy(
        overrides={"rules": DetectorPolicy(action="block", threshold=0)},
        pipeline=PipelinePolicy(pass_below=0, block_above=1),
    )
    floats = EnforcementPolicy(
        overrides={"rules": DetectorPolicy(action="block", threshold=0.0)},
        pipeline=PipelinePolicy(pass_below=0.0, block_above=1.0),
    )
    assert policy_commitment(integers) == policy_commitment(floats)


def test_policy_commitment_excludes_rule_text_and_disabled_rules(monkeypatch):
    import aceteam_aep.enforcement as enforcement

    seen = []
    original = enforcement.canonical_json

    def capture(payload):
        seen.append(payload)
        return original(payload)

    monkeypatch.setattr(enforcement, "canonical_json", capture)
    active = CustomPolicy(id="active", name="Active", rule="private rule alpha", enabled=True)
    disabled = CustomPolicy(id="disabled", name="Disabled", rule="private rule beta", enabled=False)
    before = policy_commitment(EnforcementPolicy(), [active, disabled])
    assert "private rule alpha" not in json.dumps(seen)
    assert "private rule beta" not in json.dumps(seen)
    disabled.rule = "different disabled rule"
    assert policy_commitment(EnforcementPolicy(), [active, disabled]) == before
    active.rule = "different active rule"
    assert policy_commitment(EnforcementPolicy(), [active, disabled]) != before


def test_policy_commitment_reads_the_current_custom_policy_store():
    store = CustomPolicyStore(
        [CustomPolicy(id="active", name="Active", rule="first rule", enabled=True)]
    )
    before = policy_commitment(EnforcementPolicy(), store)
    store.upsert(CustomPolicy(id="active", name="Active", rule="second rule", enabled=True))
    assert policy_commitment(EnforcementPolicy(), store) != before


def test_cli_verify_accepts_v3_jsonl(tmp_path):
    engine, public = signer()
    engine.sign_verdict(call_id="call-1", action="block", signals=[], evaluation=evaluation())
    chain_path = tmp_path / "chain.jsonl"
    chain_path.write_text("\n".join(json.dumps(entry) for entry in engine.chain) + "\n")
    key_path = tmp_path / "public.pem"
    public.save(key_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aceteam_aep.proxy.cli",
            "verify",
            "--pub-key",
            str(key_path),
            "--chain",
            str(chain_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Chain VALID" in result.stdout
    assert "pipeline:unavailable" in result.stdout
