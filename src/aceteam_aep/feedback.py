"""Signal feedback loop — learn from CISO/operator verdicts to tune thresholds.

When a human reviews a flagged signal and marks it as "confirmed" (true
positive) or "dismissed" (false positive), this module records the verdict
and adjusts detector thresholds accordingly.

The feedback loop works at the detector level:

1. **Record**: operator verdicts on individual signals
2. **Analyze**: compute false positive rate per detector
3. **Recommend**: suggest threshold adjustments
4. **Apply**: generate an updated policy YAML

Feedback is stored in a simple append-only JSONL file for auditability.
Each line is a verdict::

    {"signal_type": "pii", "score": 0.62, "verdict": "dismissed", "ts": "..."}
    {"signal_type": "pii", "score": 0.91, "verdict": "confirmed", "ts": "..."}

From these verdicts, the system can recommend: "PII threshold should be
0.75 — currently 0.5, which causes 60% false positives below 0.75."

Usage::

    from aceteam_aep.feedback import FeedbackStore, recommend_thresholds

    store = FeedbackStore("feedback.jsonl")
    store.record("pii", score=0.62, verdict="dismissed")
    store.record("pii", score=0.91, verdict="confirmed")

    recs = recommend_thresholds(store)
    # {"pii": ThresholdRecommendation(current=0.5, suggested=0.75, ...)}

    # Apply recommendations to generate updated policy
    updated_yaml = apply_recommendations(recs, "policies/finance.yaml")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    """A single operator verdict on a safety signal."""

    signal_type: str
    score: float | None
    verdict: str  # "confirmed" | "dismissed"
    detail: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "score": self.score,
            "verdict": self.verdict,
            "detail": self.detail,
            "ts": self.timestamp,
        }


class FeedbackStore:
    """Append-only JSONL store for signal verdicts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(
        self,
        signal_type: str,
        *,
        score: float | None = None,
        verdict: str,
        detail: str = "",
    ) -> Verdict:
        """Record an operator verdict."""
        if verdict not in ("confirmed", "dismissed"):
            raise ValueError(f"verdict must be 'confirmed' or 'dismissed', got '{verdict}'")

        v = Verdict(
            signal_type=signal_type,
            score=score,
            verdict=verdict,
            detail=detail,
            timestamp=datetime.now(UTC).isoformat(),
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(v.to_dict()) + "\n")

        log.info("Recorded %s verdict for %s (score=%s)", verdict, signal_type, score)
        return v

    def load(self) -> list[Verdict]:
        """Load all verdicts from the store."""
        if not self.path.exists():
            return []

        verdicts: list[Verdict] = []
        for line in self.path.read_text().strip().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            verdicts.append(
                Verdict(
                    signal_type=d["signal_type"],
                    score=d.get("score"),
                    verdict=d["verdict"],
                    detail=d.get("detail", ""),
                    timestamp=d.get("ts", ""),
                )
            )
        return verdicts

    def verdicts_for(self, signal_type: str) -> list[Verdict]:
        """Get all verdicts for a specific detector."""
        return [v for v in self.load() if v.signal_type == signal_type]


@dataclass
class ThresholdRecommendation:
    """Suggested threshold adjustment for a detector."""

    signal_type: str
    current_threshold: float | None
    suggested_threshold: float | None
    total_verdicts: int
    confirmed: int
    dismissed: int
    false_positive_rate: float  # dismissed / total
    reason: str = ""


@dataclass
class FeedbackSummary:
    """Analysis of feedback data for all detectors."""

    recommendations: dict[str, ThresholdRecommendation] = field(default_factory=dict)
    total_verdicts: int = 0


def recommend_thresholds(
    store: FeedbackStore,
    *,
    current_thresholds: dict[str, float | None] | None = None,
    min_verdicts: int = 5,
) -> FeedbackSummary:
    """Analyze verdicts and recommend threshold adjustments.

    Needs at least ``min_verdicts`` per detector to make a recommendation.
    Uses the score distribution of dismissed (false positive) signals to
    find a threshold that would have filtered them out.

    Args:
        store: feedback store to analyze
        current_thresholds: current thresholds by detector name
        min_verdicts: minimum verdicts needed before recommending
    """
    current = current_thresholds or {}
    all_verdicts = store.load()

    # Group by signal_type
    by_type: dict[str, list[Verdict]] = {}
    for v in all_verdicts:
        by_type.setdefault(v.signal_type, []).append(v)

    summary = FeedbackSummary(total_verdicts=len(all_verdicts))

    for sig_type, verdicts in by_type.items():
        confirmed = [v for v in verdicts if v.verdict == "confirmed"]
        dismissed = [v for v in verdicts if v.verdict == "dismissed"]
        total = len(verdicts)
        fp_rate = len(dismissed) / total if total > 0 else 0.0

        current_thresh = current.get(sig_type)

        if total < min_verdicts:
            summary.recommendations[sig_type] = ThresholdRecommendation(
                signal_type=sig_type,
                current_threshold=current_thresh,
                suggested_threshold=None,
                total_verdicts=total,
                confirmed=len(confirmed),
                dismissed=len(dismissed),
                false_positive_rate=fp_rate,
                reason=f"Need {min_verdicts - total} more verdicts to recommend",
            )
            continue

        # Find the threshold that separates most false positives from true positives.
        # Strategy: set threshold to the 90th percentile of dismissed scores.
        # This filters out ~90% of false positives while keeping true positives.
        dismissed_scores = sorted([v.score for v in dismissed if v.score is not None])
        confirmed_scores = sorted([v.score for v in confirmed if v.score is not None])

        suggested: float | None = None
        reason = ""

        if dismissed_scores:
            # 90th percentile of false positive scores
            idx = min(int(len(dismissed_scores) * 0.9), len(dismissed_scores) - 1)
            suggested = round(dismissed_scores[idx], 2)

            # Don't suggest a threshold that would block too many true positives
            if confirmed_scores:
                min_confirmed = confirmed_scores[0]
                if suggested >= min_confirmed:
                    # Back off to just below the lowest confirmed score
                    suggested = round(min_confirmed - 0.01, 2)

            if current_thresh is not None and suggested <= current_thresh:
                suggested = None
                reason = "Current threshold is already optimal"
            else:
                would_filter = sum(1 for s in dismissed_scores if s < suggested)
                reason = (
                    f"Raising threshold from {current_thresh} to {suggested} "
                    f"would filter {would_filter}/{len(dismissed_scores)} false positives "
                    f"({fp_rate:.0%} FP rate)"
                )
        elif fp_rate == 0:
            reason = "No false positives — threshold is working well"
        else:
            reason = "Dismissed signals have no scores — cannot compute threshold"

        summary.recommendations[sig_type] = ThresholdRecommendation(
            signal_type=sig_type,
            current_threshold=current_thresh,
            suggested_threshold=suggested,
            total_verdicts=total,
            confirmed=len(confirmed),
            dismissed=len(dismissed),
            false_positive_rate=fp_rate,
            reason=reason,
        )

    return summary


def apply_recommendations(
    summary: FeedbackSummary,
    policy_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Apply threshold recommendations to a policy YAML file.

    Returns the updated YAML string. Writes to ``output_path`` if provided,
    otherwise returns without writing.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyyaml is required. Install with: pip install aceteam-aep[yaml]"
        ) from None

    data = yaml.safe_load(Path(policy_path).read_text())
    detectors = data.get("detectors", {})

    changes: list[str] = []
    for sig_type, rec in summary.recommendations.items():
        if rec.suggested_threshold is None:
            continue
        if sig_type in detectors:
            old = detectors[sig_type].get("threshold")
            detectors[sig_type]["threshold"] = rec.suggested_threshold
            changes.append(f"{sig_type}: {old} -> {rec.suggested_threshold}")
        else:
            detectors[sig_type] = {"threshold": rec.suggested_threshold}
            changes.append(f"{sig_type}: (new) -> {rec.suggested_threshold}")

    data["detectors"] = detectors

    # Add feedback metadata as a comment
    result = yaml.dump(data, default_flow_style=False, sort_keys=False)
    header = (
        f"# Updated by AEP feedback loop ({datetime.now(UTC).strftime('%Y-%m-%d')})\n"
        f"# Changes: {', '.join(changes) if changes else 'none'}\n"
    )
    result = header + result

    if output_path:
        Path(output_path).write_text(result)
        log.info("Wrote updated policy to %s", output_path)

    return result


__all__ = [
    "FeedbackStore",
    "FeedbackSummary",
    "ThresholdRecommendation",
    "Verdict",
    "apply_recommendations",
    "recommend_thresholds",
]
