"""Secret/credential detector — regex-based detection of leaked API keys and tokens.

Catches AWS keys, GitHub PATs, Stripe keys, Slack tokens, PEM private keys,
and generic API key patterns in both agent input and output traffic.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from .base import SafetyDetector, SafetySignal

log = logging.getLogger(__name__)

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (
        re.compile(r"(?:aws_secret_access_key|AWS_SECRET)\s*[:=]\s*['\"]?([0-9a-zA-Z/+=]{40})", re.IGNORECASE),
        "AWS secret key",
    ),
    (re.compile(r"ghp_[0-9a-zA-Z]{36}"), "GitHub PAT"),
    (re.compile(r"github_pat_[0-9a-zA-Z]{22}_[0-9a-zA-Z]{59}"), "GitHub fine-grained PAT"),
    (re.compile(r"gho_[0-9a-zA-Z]{36}"), "GitHub OAuth token"),
    (re.compile(r"ghu_[0-9a-zA-Z]{36}"), "GitHub user-to-server token"),
    (re.compile(r"ghs_[0-9a-zA-Z]{36}"), "GitHub server-to-server token"),
    (re.compile(r"glpat-[0-9a-zA-Z\-]{20,}"), "GitLab PAT"),
    (re.compile(r"sk_live_[0-9a-zA-Z]{24,}"), "Stripe live secret key"),
    (re.compile(r"pk_live_[0-9a-zA-Z]{24,}"), "Stripe live publishable key"),
    (re.compile(r"rk_live_[0-9a-zA-Z]{24,}"), "Stripe restricted key"),
    (re.compile(r"xox[bpors]-[0-9a-zA-Z\-]{10,}"), "Slack token"),
    (re.compile(r"-----BEGIN\s*(?:RSA|EC|DSA|OPENSSH|ENCRYPTED)?\s*PRIVATE KEY-----"), "PEM private key"),
    (re.compile(r"sk-[0-9a-zA-Z]{20,}T3BlbkFJ[0-9a-zA-Z]{20,}"), "OpenAI API key"),
    (re.compile(r"sk-ant-api\d{2}-[0-9a-zA-Z\-]{80,}"), "Anthropic API key"),
    (re.compile(r"AIzaSy[0-9a-zA-Z\-_]{33}"), "Google API key"),
    (re.compile(r"sq0atp-[0-9a-zA-Z\-]{22,}"), "Square access token"),
    (re.compile(r"sqOatp-[0-9a-zA-Z\-]{22,}"), "Square OAuth token"),
    (re.compile(r"EZAK[0-9a-zA-Z\-]{54,}"), "EasyPost API key"),
    (
        re.compile(r"(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9\-._]{20,})", re.IGNORECASE),
        "generic API key/token",
    ),
]


class SecretDetector(SafetyDetector):
    """Detect leaked credentials and API keys in agent traffic.

    Scans both input and output text for known secret patterns.
    All matches produce high-severity signals with score=1.0.
    """

    name = "secret_leak"

    async def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs: object,
    ) -> Sequence[SafetySignal]:
        signals: list[SafetySignal] = []
        for source, text in [("input", input_text), ("output", output_text)]:
            if not text:
                continue
            for pattern, desc in _SECRET_PATTERNS:
                for match in pattern.finditer(text):
                    start, end = match.span()
                    redacted = text[start : min(start + 8, end)] + "..."
                    signals.append(
                        SafetySignal(
                            signal_type="secret_leak",
                            severity="high",
                            call_id=call_id,
                            detail=f"{desc} detected in {source} [{start}:{end}] ({redacted})",
                            score=1.0,
                        )
                    )
        return signals


__all__ = ["SecretDetector"]
