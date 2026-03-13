"""Workflow-oriented envelope builder composing SpanTracker + CostTracker."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from aceteam_aep.budget import BudgetEnforcer
from aceteam_aep.costs import CostNode, CostTracker
from aceteam_aep.envelope import Citation, ExecutionEnvelope, ExecutionError
from aceteam_aep.pricing import DefaultPricingProvider, PricingProvider
from aceteam_aep.spans import SpanTracker
from aceteam_aep.types import Usage


@dataclass
class NodeRecord:
    """Generic record for crash recovery reconstruction."""

    node_id: str
    node_type: str
    status: str  # "SUCCESS", "ERROR", "YIELDED"
    started_at: str | None
    finished_at: str | None
    output: dict | None
    error: str | None
    model_name: str | None
    provider: str | None
    input_tokens: int
    output_tokens: int
    cost: Decimal


class EnvelopeBuilder:
    """Composes SpanTracker + CostTracker into a workflow envelope."""

    def __init__(
        self,
        execution_id: str,
        org_id: str,
        pricing: PricingProvider | None = None,
        budget: BudgetEnforcer | None = None,
    ) -> None:
        self._execution_id = execution_id
        self._org_id = org_id
        self._span_tracker = SpanTracker()
        self._cost_tracker = CostTracker(
            entity=f"org:{org_id}",
            pricing=pricing or DefaultPricingProvider(),
        )
        self._budget = budget
        self._citations: list[Citation] = []
        self._errors: list[ExecutionError] = []
        self._root_span_id: str | None = None
        self._finished = False

    def start(self) -> str:
        """Record execution start. Creates root span. Returns root span_id."""
        span = self._span_tracker.start_span(
            executor_type="workflow",
            executor_id=self._execution_id,
        )
        self._root_span_id = span.span_id
        return span.span_id

    def begin_node(self, node_id: str, node_type: str) -> str:
        """Creates child span under root. Returns span_id."""
        if self._finished:
            raise RuntimeError("Cannot begin_node after finish()")
        span = self._span_tracker.start_span(
            executor_type=node_type,
            executor_id=node_id,
            parent_span_id=self._root_span_id,
        )
        return span.span_id

    def end_node(
        self,
        span_id: str,
        *,
        citations: list[Citation] | None = None,
        cost: CostNode | None = None,
    ) -> None:
        """Closes span, collects citations and costs."""
        self._span_tracker.end_span(span_id, status="OK")
        if citations:
            self._citations.extend(citations)
        if cost:
            self._cost_tracker.add_node(cost)

    def end_node_error(self, span_id: str, error: ExecutionError) -> None:
        """Marks span as ERROR, records error."""
        self._span_tracker.end_span(span_id, status="ERROR")
        self._errors.append(error)

    def record_llm_cost(
        self,
        span_id: str,
        model: str,
        usage: Usage,
        parent_cost_id: str | None = None,
    ) -> CostNode:
        """Delegates to CostTracker.record_llm_cost()."""
        return self._cost_tracker.record_llm_cost(
            span_id=span_id,
            model=model,
            usage=usage,
            parent_cost_id=parent_cost_id,
        )

    def finish(
        self,
        status: Literal["success", "partial", "failure"] = "success",
    ) -> ExecutionEnvelope:
        """Closes root span, assembles envelope from tracker state."""
        if self._finished:
            raise RuntimeError("EnvelopeBuilder.finish() already called")
        self._finished = True
        if self._root_span_id:
            root_status = "ERROR" if self._errors else "OK"
            self._span_tracker.end_span(self._root_span_id, status=root_status)
        return ExecutionEnvelope(
            execution_id=self._execution_id,
            status=status,
            cost_tree=self._cost_tracker.get_cost_tree() or None,
            spans=self._span_tracker.get_spans() or None,
            citations=self._citations or None,
            errors=self._errors or None,
            budget_state=self._budget.state if self._budget else None,
        )

    @classmethod
    def reconstruct(
        cls,
        execution_id: str,
        org_id: str,
        node_records: list[NodeRecord],
    ) -> ExecutionEnvelope:
        """Rebuild envelope from persisted node records (crash recovery)."""
        builder = cls(execution_id=execution_id, org_id=org_id)
        builder.start()
        has_errors = False
        for record in node_records:
            span_id = builder.begin_node(record.node_id, record.node_type)
            if record.status == "ERROR":
                has_errors = True
                builder.end_node_error(
                    span_id,
                    ExecutionError(
                        code="NODE_ERROR",
                        message=record.error or "Unknown error",
                        node_id=record.node_id,
                    ),
                )
            else:
                cost_node = None
                if record.cost > 0 and record.model_name:
                    cost_node = CostNode(
                        id=f"cost-{record.node_id}",
                        parent_id=None,
                        entity=f"org:{org_id}",
                        category="llm_tokens",
                        compute_cost=record.cost,
                    )
                builder.end_node(span_id, cost=cost_node)
        status = "partial" if has_errors else "success"
        return builder.finish(status=status)


__all__ = ["EnvelopeBuilder", "NodeRecord"]
