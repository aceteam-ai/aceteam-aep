"""Tests for AEP header parsing and emission."""

from decimal import Decimal

from aceteam_aep.proxy.headers import (
    AepRequestContext,
    build_response_headers,
    classification_rank,
    parse_aep_headers,
    strip_aep_headers,
)


def test_parse_empty_headers() -> None:
    ctx = parse_aep_headers({})
    assert ctx.entity == "default"
    assert ctx.classification == "public"
    assert ctx.consent == {}
    assert ctx.budget is None


def test_parse_full_headers() -> None:
    ctx = parse_aep_headers({
        "x-aep-entity": "org:acme",
        "x-aep-classification": "confidential",
        "x-aep-consent": "training=no,sharing=yes",
        "x-aep-budget": "5.00",
        "x-aep-trace-id": "trace-abc",
        "x-aep-sources": "doc:contract-123, url:https://example.com",
    })
    assert ctx.entity == "org:acme"
    assert ctx.classification == "confidential"
    assert ctx.consent == {"training": False, "sharing": True}
    assert ctx.budget == Decimal("5.00")
    assert ctx.trace_id == "trace-abc"
    assert len(ctx.sources) == 2


def test_parse_invalid_budget_ignored() -> None:
    ctx = parse_aep_headers({"x-aep-budget": "not-a-number"})
    assert ctx.budget is None


def test_parse_invalid_classification_ignored() -> None:
    ctx = parse_aep_headers({"x-aep-classification": "top-secret"})
    assert ctx.classification == "public"


def test_has_governance() -> None:
    assert not AepRequestContext().has_governance
    assert AepRequestContext(classification="confidential").has_governance
    assert AepRequestContext(consent={"training": False}).has_governance


def test_strip_aep_headers() -> None:
    headers = {
        "Authorization": "Bearer sk-test",
        "Content-Type": "application/json",
        "X-AEP-Entity": "org:acme",
        "x-aep-classification": "confidential",
    }
    stripped = strip_aep_headers(headers)
    assert "Authorization" in stripped
    assert "Content-Type" in stripped
    assert "X-AEP-Entity" not in stripped
    assert "x-aep-classification" not in stripped


def test_build_response_headers() -> None:
    headers = build_response_headers(
        cost=Decimal("0.0042"),
        enforcement="flag",
        call_id="abc123",
        classification="confidential",
        flag_reason="PII detected",
        trace_id="trace-xyz",
    )
    assert headers["X-AEP-Cost"] == "0.0042"
    assert headers["X-AEP-Enforcement"] == "flag"
    assert headers["X-AEP-Classification"] == "confidential"
    assert headers["X-AEP-Flag-Reason"] == "PII detected"
    assert headers["X-AEP-Trace-ID"] == "trace-xyz"


def test_classification_rank_ordering() -> None:
    assert classification_rank("public") < classification_rank("internal")
    assert classification_rank("internal") < classification_rank("confidential")
    assert classification_rank("confidential") < classification_rank("restricted")
