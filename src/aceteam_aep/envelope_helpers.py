"""Helper functions for extracting summary data from AEP envelopes."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from aceteam_aep.costs import CostNode
from aceteam_aep.spans import Span


def sum_cost_tree(nodes: list[CostNode] | None) -> Decimal:
    """Sum compute_cost + value_added_fee + platform_fee across all nodes."""
    if not nodes:
        return Decimal("0")
    total = Decimal("0")
    for node in nodes:
        total += node.compute_cost + node.value_added_fee + node.platform_fee
    return total


def compute_duration(spans: list[Span] | None) -> int:
    """Find root span and return duration in milliseconds."""
    if not spans:
        return 0
    root = next((s for s in spans if s.parent_span_id is None), None)
    if not root or not root.started_at or not root.ended_at:
        return 0
    start = datetime.fromisoformat(root.started_at)
    end = datetime.fromisoformat(root.ended_at)
    return int((end - start).total_seconds() * 1000)


def extract_primary_model(nodes: list[CostNode] | None) -> str | None:
    """Return model name of the first llm_tokens cost node."""
    if not nodes:
        return None
    for node in nodes:
        if node.category == "llm_tokens" and node.metadata:
            model = node.metadata.get("model")
            if model:
                return str(model)
    return None


__all__ = ["compute_duration", "extract_primary_model", "sum_cost_tree"]
