"""The public repository's labeled fixtures are synthetic and offline."""

from __future__ import annotations

import json
import socket
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from aceteam_aep.safety.agent_threat import AgentThreatDetector
from aceteam_aep.safety.eval.__main__ import main
from aceteam_aep.safety.eval.dataset import (
    assign_split,
    load_dataset,
    load_policies,
    rows_by_source,
    verification_summary,
    verify_rows,
    write_rows,
)
from aceteam_aep.safety.ferpa import FerpaDetector
from aceteam_aep.safety.pii import PiiDetector
from aceteam_aep.safety.secrets import SecretDetector
from scripts.safety_eval.generate_fixtures import POLICIES, generate_rows

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "tests/fixtures/safety_eval/policies.json"
LABELED = ROOT / "tests/fixtures/safety_eval/labeled.jsonl"


@pytest.fixture(autouse=True)
def forbid_socket_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def _fixture():
    policies = load_policies(CATALOG)
    return policies, load_dataset(LABELED, policies)


def _write_mutation(tmp_path: Path, rows: list[dict]) -> Path:
    output = tmp_path / "mutated.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output


def test_schema_catalog_counts_and_regeneration() -> None:
    policies, rows = _fixture()
    assert len(policies) == 5
    assert len(rows) == 400
    assert list(policies) == [policy["id"] for policy in POLICIES]
    assert [row.to_dict() for row in rows] == generate_rows()
    counts = Counter((row.category, row.kind) for row in rows)
    for category in {policy.category for policy in policies.values()}:
        assert counts[(category, "violation")] == 40
        assert counts[(category, "hard_negative")] == 20
        assert counts[(category, "easy_negative")] == 20
    assert set(rows_by_source(rows)) == {"synthetic_v1"}
    assert all(not row.verified for row in rows)
    summary = verification_summary(rows)
    assert summary.test_rows > 0
    assert summary.unverified_test_rows == summary.test_rows
    assert summary.drop_rate == 0


def test_splits_reproduce_and_hold_out_whole_category(tmp_path: Path) -> None:
    policies, rows = _fixture()
    assert all(row.split == assign_split(row.id, row.category, row.source) for row in rows)
    by_category: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_category[row.category].add(row.split)
    assert by_category["ethics"] == {"test_xd"}
    assert all(
        "test_xd" not in splits for category, splits in by_category.items() if category != "ethics"
    )
    assert assign_split("public-example", "security", "deepset_prompt_injections") == "test_xd"
    assert assign_split("training-example", "security", "FuzzyBench_training") == "dev"
    main(["splits", "--labeled", str(LABELED), "--check"])
    data = [row.to_dict() for row in rows]
    data[0]["split"] = "test_xd" if data[0]["split"] != "test_xd" else "dev"
    mutated = _write_mutation(tmp_path, data)
    with pytest.raises(SystemExit, match="split column differs"):
        main(["splits", "--labeled", str(mutated), "--check"])
    with pytest.raises(ValueError, match="split differs"):
        load_dataset(mutated, policies)


def test_loader_rejects_bad_ids_catalog_links_and_fuzzybench_test_source(tmp_path: Path) -> None:
    policies, rows = _fixture()
    data = [row.to_dict() for row in rows[:2]]
    data[1]["id"] = data[0]["id"]
    with pytest.raises(ValueError, match="duplicate labeled row id"):
        load_dataset(_write_mutation(tmp_path, data), policies)
    data[1]["id"] = rows[1].id
    data[1]["policy_id"] = "missing_policy"
    with pytest.raises(ValueError, match="catalog policy"):
        load_dataset(_write_mutation(tmp_path, data), policies)
    data = [rows[0].to_dict()]
    data[0]["source"] = "FuzzyBench_training"
    data[0]["split"] = "test_xd"
    with pytest.raises(ValueError, match="FuzzyBench source is forbidden"):
        load_dataset(_write_mutation(tmp_path, data), policies)


@pytest.mark.asyncio
async def test_paw_rows_evade_regex_and_residual_detectors() -> None:
    _, rows = _fixture()
    pii = PiiDetector()
    pii._load_attempted = True
    pii._fallback = True
    detectors = [pii, AgentThreatDetector(scan_input=True), SecretDetector(), FerpaDetector()]
    for row in rows:
        for detector in detectors:
            signals = await detector.check(
                input_text=row.text if row.applies_to == "input" else "",
                output_text=row.text if row.applies_to == "output" else "",
                call_id=row.id,
            )
            assert not signals, f"{row.id} triggered {detector.name}"


def test_fake_data_guard_rejects_nonreserved_contact_values(tmp_path: Path) -> None:
    policies, rows = _fixture()
    for bad in ("user@outside.invalid", "212-555-9955", "202-555-9955"):
        data = [rows[0].to_dict()]
        data[0]["text"] = f"Synthetic example {bad}"
        with pytest.raises(ValueError, match="nonreserved email|nonfictional phone"):
            load_dataset(_write_mutation(tmp_path, data), policies)
    safe = [rows[0].to_dict()]
    safe[0]["text"] = "Synthetic example user@example.com and 212-555-0188"
    assert len(load_dataset(_write_mutation(tmp_path, safe), policies)) == 1


class FakeVerifier:
    def __init__(self, disagree: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.disagree = disagree

    def verify(self, *, model_id, policy, text, applies_to):
        self.calls.append((model_id, policy.id))
        return int(model_id == "model-b" and self.disagree)


def test_verify_fake_client_writes_votes_and_drops_disagreement(tmp_path: Path) -> None:
    policies, rows = _fixture()
    row = next(row for row in rows if row.split == "test_id" and row.label == 0)
    verifier = FakeVerifier(disagree=True)
    voted = verify_rows(
        [row],
        policies,
        verifier,
        model_a="model-a",
        family_a="family-a",
        model_b="model-b",
        family_b="family-b",
        generator_family="generator-family",
    )
    assert verifier.calls == [("model-a", row.policy_id), ("model-b", row.policy_id)]
    output = tmp_path / "votes.jsonl"
    write_rows(output, voted)
    assert load_dataset(output, policies) == []
    summary = verification_summary(load_dataset(output, policies, include_disagreements=True))
    assert summary.disagreements_dropped == 1
    assert summary.drop_rate == 1.0
    with pytest.raises(ValueError, match="families must differ"):
        verify_rows(
            [row],
            policies,
            verifier,
            model_a="model-a",
            family_a="generator-family",
            model_b="model-b",
            family_b="family-b",
            generator_family="generator-family",
        )


def test_verify_subcommand_uses_only_explicit_fake_adapter(tmp_path: Path, monkeypatch) -> None:
    policies, rows = _fixture()
    source = tmp_path / "one.jsonl"
    write_rows(source, [next(row for row in rows if row.split == "test_id" and row.label == 0)])
    fake = types.ModuleType("fake_safety_eval_verifier")
    client = FakeVerifier()
    fake.__dict__["client"] = client
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    output = tmp_path / "verified.jsonl"
    main(
        [
            "verify",
            "--catalog",
            str(CATALOG),
            "--labeled",
            str(source),
            "--output",
            str(output),
            "--client",
            "fake_safety_eval_verifier:client",
            "--model-a",
            "model-a",
            "--family-a",
            "family-a",
            "--model-b",
            "model-b",
            "--family-b",
            "family-b",
            "--generator-family",
            "generator-family",
        ]
    )
    verified = load_dataset(output, policies, verified_only=True)
    assert len(verified) == 1 and verified[0].verified
    assert len(client.calls) == 2
