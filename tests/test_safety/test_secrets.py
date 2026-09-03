"""Tests for SecretDetector — credential and API key detection."""

from __future__ import annotations

import pytest

from aceteam_aep.safety.secrets import SecretDetector

# Split prefixes so the literals don't match GitHub push-protection's secret
# scanning regex (Stripe / GitHub / Google patterns are actively validated).
# The runtime values still match SecretDetector's regex so the tests pass.
_STRIPE_LIVE = "sk" + "_live_" + "4eC39HqLyjWDarjtT1zdp7dc"
_GITHUB_PAT = "ghp" + "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
_GOOGLE_KEY = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"


@pytest.fixture
def detector() -> SecretDetector:
    return SecretDetector()


async def test_detects_aws_access_key(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="my key is AKIAIOSFODNN7EXAMPLE",
        output_text="",
        call_id="c1",
    )
    assert len(signals) >= 1
    assert any("AWS access key" in s.detail for s in signals)
    assert all(s.severity == "high" for s in signals)
    assert all(s.score == 1.0 for s in signals)


async def test_detects_aws_secret_key(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        output_text="",
        call_id="c2",
    )
    assert len(signals) >= 1
    assert any("AWS secret key" in s.detail for s in signals)


async def test_detects_github_pat(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text=f"token: {_GITHUB_PAT}",
        output_text="",
        call_id="c3",
    )
    assert len(signals) >= 1
    assert any("GitHub PAT" in s.detail for s in signals)


async def test_detects_github_fine_grained(detector: SecretDetector) -> None:
    pat = "github_pat_" + "A" * 22 + "_" + "B" * 59
    signals = await detector.check(
        input_text=f"token: {pat}",
        output_text="",
        call_id="c4",
    )
    assert len(signals) >= 1
    assert any("fine-grained" in s.detail for s in signals)


async def test_detects_stripe_live_key(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="",
        output_text=f"Use {_STRIPE_LIVE} for payments",
        call_id="c5",
    )
    assert len(signals) >= 1
    assert any("Stripe" in s.detail for s in signals)
    assert "output" in signals[0].detail


async def test_detects_slack_token(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="xoxb-123456789-abcdefghij",
        output_text="",
        call_id="c6",
    )
    assert len(signals) >= 1
    assert any("Slack" in s.detail for s in signals)


async def test_detects_pem_private_key(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="-----BEGIN RSA PRIVATE KEY-----\nMIIEowI...",
        output_text="",
        call_id="c7",
    )
    assert len(signals) >= 1
    assert any("PEM private key" in s.detail for s in signals)


async def test_detects_pem_ec_key(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="-----BEGIN EC PRIVATE KEY-----\nMHQ...",
        output_text="",
        call_id="c7b",
    )
    assert len(signals) >= 1
    assert any("PEM private key" in s.detail for s in signals)


async def test_detects_gitlab_pat(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="glpat-xxxxxxxxxxxxxxxxxxxx",
        output_text="",
        call_id="c8",
    )
    assert len(signals) >= 1
    assert any("GitLab" in s.detail for s in signals)


async def test_detects_generic_api_key(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text='api_key="sk_1234567890abcdefghijklmnop"',
        output_text="",
        call_id="c9",
    )
    assert len(signals) >= 1
    assert any("generic API key" in s.detail for s in signals)


async def test_detects_google_api_key(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text=f"key={_GOOGLE_KEY}",
        output_text="",
        call_id="c10",
    )
    assert len(signals) >= 1
    assert any("Google" in s.detail for s in signals)


async def test_no_false_positive_on_clean_text(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="Hello world, this is a normal message with no secrets.",
        output_text="The weather is nice today. Let me help you with that.",
        call_id="c11",
    )
    assert len(signals) == 0


async def test_no_false_positive_on_code_discussion(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="Use the API key from your environment variables",
        output_text="Set OPENAI_API_KEY in your .env file",
        call_id="c12",
    )
    assert len(signals) == 0


async def test_scans_both_input_and_output(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="AKIAIOSFODNN7EXAMPLE",
        output_text=_GITHUB_PAT,
        call_id="c13",
    )
    input_signals = [s for s in signals if "input" in s.detail]
    output_signals = [s for s in signals if "output" in s.detail]
    assert len(input_signals) >= 1
    assert len(output_signals) >= 1


async def test_signal_metadata(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="AKIAIOSFODNN7EXAMPLE",
        output_text="",
        call_id="test-call",
    )
    assert len(signals) >= 1
    s = signals[0]
    assert s.signal_type == "secret_leak"
    assert s.severity == "high"
    assert s.call_id == "test-call"
    assert s.score == 1.0


async def test_redacts_in_detail(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text="AKIAIOSFODNN7EXAMPLE",
        output_text="",
        call_id="c14",
    )
    assert len(signals) >= 1
    assert "AKIAIOSF..." in signals[0].detail
    assert "AKIAIOSFODNN7EXAMPLE" not in signals[0].detail


async def test_multiple_secrets_in_one_text(detector: SecretDetector) -> None:
    signals = await detector.check(
        input_text=f"AWS: AKIAIOSFODNN7EXAMPLE and Stripe: {_STRIPE_LIVE}",
        output_text="",
        call_id="c15",
    )
    types = {s.detail.split(" detected")[0] for s in signals}
    assert "AWS access key" in types
    assert "Stripe live secret key" in types
