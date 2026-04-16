"""Agent threat detector — regex-based detection of malicious agent behavior.

Detects common attack patterns: port scanning, reverse shells, subprocess
execution, socket connections, localhost probing, credential file access,
and destructive commands.

NOTE: This is a regex-based detector — pattern matching on known dangerous
strings. Easy to bypass with rephrasing or obfuscation. The production
version will use the Trust Engine's ensemble-of-judges approach
(LLM-as-judge + classifier fusion) for intent-level detection.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Literal

from .base import SafetyDetector, SafetySignal

logger = logging.getLogger(__name__)

_THREAT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnmap\b", re.IGNORECASE), "port scanning (nmap)"),
    (re.compile(r"\bnetcat\b|\bnc\s+-", re.IGNORECASE), "reverse shell / netcat"),
    (re.compile(r"\bssh\s+.*-p\b", re.IGNORECASE), "SSH brute force"),
    (re.compile(r"socket\.connect\(", re.IGNORECASE), "raw socket connection"),
    (
        re.compile(r"subprocess\.(run|call|Popen)\(", re.IGNORECASE),
        "subprocess execution",
    ),
    (re.compile(r"os\.(system|popen)\(", re.IGNORECASE), "OS command execution"),
    (
        re.compile(r"\bcurl\b.*\blocalhost\b|\bwget\b.*\blocalhost\b", re.IGNORECASE),
        "localhost probing",
    ),
    (re.compile(r"for\s+port\s+in\s+range\(", re.IGNORECASE), "port scan loop"),
    (
        re.compile(r"0\.0\.0\.0|127\.0\.0\.1:\d{4,5}", re.IGNORECASE),
        "internal service targeting",
    ),
    (
        re.compile(r"\b/etc/passwd\b|\b/etc/shadow\b", re.IGNORECASE),
        "credential file access",
    ),
    (re.compile(r"rm\s+-rf\s+/", re.IGNORECASE), "destructive command"),
]


class AgentThreatDetector(SafetyDetector):
    """Detect when an AI agent attempts network attacks or system exploitation.

    Scans both input and output text for known dangerous patterns such as
    port scanning, reverse shells, subprocess execution, and credential access.

    NOTE: This is a regex-based demo detector. The production version will use
    the Trust Engine's ensemble-of-judges approach (LLM-as-judge + classifier
    fusion) for intent-level detection.
    """

    name = "agent_threat"

    def __init__(
        self,
        *,
        scan_input: bool = False,
        scan_output: bool = True,
    ) -> None:
        if not (scan_input or scan_output):
            logger.warning(
                f"{self.__class__.__name__}: scan_input and scan_output are both False, "
                "so this detector will not detect anything"
            )
            scan_input = True
            scan_output = True
        self._scan_input = scan_input
        self._scan_output = scan_output
        self._patterns = _THREAT_PATTERNS

    async def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs: object,
    ) -> Sequence[SafetySignal]:
        signals: list[SafetySignal] = []
        texts: list[tuple[str, Literal["input", "output"]]] = []
        if self._scan_input:
            texts.append((input_text, "input"))
        if self._scan_output:
            texts.append((output_text, "output"))
        for text, source in texts:
            if not text:
                continue
            for pattern, desc in self._patterns:
                if pattern.search(text):
                    signals.append(
                        SafetySignal(
                            signal_type="agent_threat",
                            severity="high",
                            call_id=call_id,
                            detail=f"{desc} detected in {source}",
                            score=1.0,
                        )
                    )
        return signals


__all__ = ["AgentThreatDetector"]
