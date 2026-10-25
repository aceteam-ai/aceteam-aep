"""Validate labeled evaluation rows without loading models or contacting a service."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SPLITS = {"dev", "test_id", "test_xd"}
KINDS = {"violation", "hard_negative", "easy_negative"}
HOLDOUT_CATEGORY = "ethics"
ROW_FIELDS = {
    "id",
    "text",
    "applies_to",
    "policy_id",
    "label",
    "category",
    "kind",
    "source",
    "split",
    "votes",
}
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\b(?:\+1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")


def assign_split(row_id: str, category: str, source: str) -> str:
    if "fuzzybench" in source.lower():
        return "dev"
    if category == HOLDOUT_CATEGORY or source != "synthetic_v1":
        return "test_xd"
    digest = hashlib.sha256(row_id.encode()).digest()
    return "dev" if digest[0] < 128 else "test_id"


@dataclass(frozen=True)
class Policy:
    id: str
    category: str
    rule: str
    severity: str
    applies_to: str


@dataclass(frozen=True)
class LabeledRow:
    id: str
    text: str
    applies_to: str
    policy_id: str
    label: int
    category: str
    kind: str
    source: str
    split: str
    votes: tuple[dict[str, object], ...]

    @property
    def verified(self) -> bool:
        return (
            len(self.votes) == 2
            and self.votes[0]["label"] == self.votes[1]["label"] == self.label
            and self.votes[0]["family"] != self.votes[1]["family"]
        )

    @property
    def disagreed(self) -> bool:
        return len(self.votes) == 2 and self.votes[0]["label"] != self.votes[1]["label"]

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) if field != "votes" else list(self.votes)
            for field in ROW_FIELDS
        }


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def load_policies(path: str | Path) -> dict[str, Policy]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("policy catalog must be a list")
    policies: dict[str, Policy] = {}
    for item in data:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "category",
            "rule",
            "severity",
            "applies_to",
        }:
            raise ValueError("invalid policy catalog entry")
        if not all(isinstance(value, str) and value for value in item.values()):
            raise ValueError("policy catalog fields must be nonempty strings")
        if item["severity"] not in {"high", "medium", "low"} or item["applies_to"] not in {
            "input",
            "output",
        }:
            raise ValueError("invalid policy severity or side")
        policy = Policy(**item)
        if policy.id in policies:
            raise ValueError("duplicate policy id")
        policies[policy.id] = policy
    return policies


def _validate_fake_data(text: str) -> None:
    for email in EMAIL.finditer(text):
        domain = email.group(1).lower() + "." + email.group(0).rsplit(".", 1)[-1].lower()
        if domain not in {"example.com", "example.org", "example.net"} and not domain.endswith(
            (".example.com", ".example.org", ".example.net")
        ):
            raise ValueError("nonreserved email in synthetic fixture")
    for phone in PHONE.finditer(text):
        digits = "".join(char for char in phone.group() if char.isdigit())
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if not (len(digits) == 10 and digits[3:6] == "555" and digits[6:8] == "01"):
            raise ValueError("nonfictional phone in synthetic fixture")


def _validate_votes(votes: object) -> tuple[dict[str, object], ...]:
    if not isinstance(votes, list) or len(votes) not in (0, 2):
        raise ValueError("votes must be empty or contain two verifier votes")
    for vote in votes:
        if not isinstance(vote, dict) or set(vote) != {"model_id", "family", "label"}:
            raise ValueError("invalid verifier vote")
        if not isinstance(vote["model_id"], str) or not vote["model_id"]:
            raise ValueError("verifier model id is required")
        if not isinstance(vote["family"], str) or not vote["family"]:
            raise ValueError("verifier family is required")
        if type(vote["label"]) is not int or vote["label"] not in (0, 1):
            raise ValueError("verifier label must be 0 or 1")
    if len(votes) == 2 and (
        votes[0]["family"] == votes[1]["family"] or votes[0]["model_id"] == votes[1]["model_id"]
    ):
        raise ValueError("verifier models must use distinct families")
    return tuple(votes)


def load_dataset(
    labeled_path: str | Path,
    policies: dict[str, Policy],
    *,
    verified_only: bool = False,
    include_disagreements: bool = False,
) -> list[LabeledRow]:
    rows = []
    seen = set()
    for line_number, line in enumerate(Path(labeled_path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != ROW_FIELDS:
            raise ValueError(f"invalid labeled row schema at line {line_number}")
        if not all(
            isinstance(item[field], str) and item[field]
            for field in ROW_FIELDS - {"label", "votes"}
        ):
            raise ValueError(f"invalid labeled row text field at line {line_number}")
        if type(item["label"]) is not int or item["label"] not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if item["id"] in seen:
            raise ValueError("duplicate labeled row id")
        seen.add(item["id"])
        policy = policies.get(item["policy_id"])
        if (
            policy is None
            or item["category"] != policy.category
            or item["applies_to"] != policy.applies_to
        ):
            raise ValueError("labeled row does not match a catalog policy")
        if item["kind"] not in KINDS or item["label"] != int(item["kind"] == "violation"):
            raise ValueError("labeled row kind and label disagree")
        if "fuzzybench" in item["source"].lower() and item["split"] != "dev":
            raise ValueError("FuzzyBench source is forbidden in test splits")
        if item["split"] not in SPLITS or item["split"] != assign_split(
            item["id"], item["category"], item["source"]
        ):
            raise ValueError("labeled row split differs from deterministic assignment")
        if item["source"] == "synthetic_v1":
            _validate_fake_data(item["text"])
        item["votes"] = _validate_votes(item["votes"])
        row = LabeledRow(**item)
        if row.disagreed and row.split != "dev" and not include_disagreements:
            continue
        if verified_only and row.split != "dev" and not row.verified:
            continue
        rows.append(row)
    return rows


@dataclass(frozen=True)
class VerificationSummary:
    test_rows: int
    unverified_test_rows: int
    disagreements_dropped: int
    drop_rate: float


def verification_summary(rows: Sequence[LabeledRow]) -> VerificationSummary:
    """Summarize the full validated view before test disagreements are removed."""
    test = [row for row in rows if row.split in {"test_id", "test_xd"}]
    dropped = sum(row.disagreed for row in test)
    return VerificationSummary(
        test_rows=len(test),
        unverified_test_rows=sum(not row.verified and not row.disagreed for row in test),
        disagreements_dropped=dropped,
        drop_rate=dropped / len(test) if test else 0.0,
    )


def rows_by_source(rows: Iterable[LabeledRow]) -> dict[str, list[LabeledRow]]:
    grouped: dict[str, list[LabeledRow]] = defaultdict(list)
    for row in rows:
        grouped[row.source].append(row)
    return dict(grouped)


class VerifierClient(Protocol):
    def verify(self, *, model_id: str, policy: Policy, text: str, applies_to: str) -> int: ...


def verify_rows(
    rows: Sequence[LabeledRow],
    policies: dict[str, Policy],
    client: VerifierClient,
    *,
    model_a: str,
    family_a: str,
    model_b: str,
    family_b: str,
    generator_family: str,
) -> list[LabeledRow]:
    if len({family_a, family_b, generator_family}) != 3 or model_a == model_b:
        raise ValueError("verifier families must differ from each other and the generator")
    verified = []
    for row in rows:
        policy = policies[row.policy_id]
        votes = []
        for model_id, family in ((model_a, family_a), (model_b, family_b)):
            label = client.verify(
                model_id=model_id, policy=policy, text=row.text, applies_to=row.applies_to
            )
            if type(label) is not int or label not in (0, 1):
                raise ValueError("verifier returned an invalid label")
            votes.append({"model_id": model_id, "family": family, "label": label})
        verified.append(LabeledRow(**{**row.to_dict(), "votes": tuple(votes)}))
    return verified


def write_rows(path: str | Path, rows: Sequence[LabeledRow]) -> None:
    Path(path).write_text("".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows))
