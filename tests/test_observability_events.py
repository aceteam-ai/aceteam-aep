"""Tests for observability event Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aceteam_aep.observability import FlaggedCall, ObservabilityEvent


class TestObservabilityEvent:
    def test_auto_generated_id_and_timestamp(self):
        event = ObservabilityEvent(session_id="sess-1", type="call_start")
        assert len(event.id) == 32  # uuid4 hex
        assert event.id.isalnum()
        assert "T" in event.timestamp  # ISO 8601
        assert event.timestamp.endswith("+00:00")

    def test_unique_ids_per_instance(self):
        e1 = ObservabilityEvent(session_id="s", type="call_start")
        e2 = ObservabilityEvent(session_id="s", type="call_start")
        assert e1.id != e2.id

    def test_invalid_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ObservabilityEvent(session_id="sess-1", type="unknown_type")  # type: ignore[arg-type]

    def test_valid_types(self):
        for t in ("call_start", "call_end", "safety_signal", "enforcement", "cost"):
            event = ObservabilityEvent(session_id="s", type=t)  # type: ignore[arg-type]
            assert event.type == t

    def test_serialization_via_model_dump(self):
        event = ObservabilityEvent(
            session_id="sess-abc",
            type="call_end",
            call_id="call-1",
            model="gpt-4o",
            provider="openai",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.005,
            latency_ms=320.5,
        )
        d = event.model_dump()
        assert d["session_id"] == "sess-abc"
        assert d["type"] == "call_end"
        assert d["model"] == "gpt-4o"
        assert d["tokens_in"] == 100
        assert d["cost_usd"] == 0.005
        assert d["latency_ms"] == 320.5
        assert d["call_id"] == "call-1"

    def test_event_with_metadata_dict(self):
        meta = {"source": "proxy", "version": 2, "tags": ["test"]}
        event = ObservabilityEvent(session_id="s", type="safety_signal", metadata=meta)
        assert event.metadata == meta
        d = event.model_dump()
        assert d["metadata"] == meta

    def test_optional_fields_default_to_none(self):
        event = ObservabilityEvent(session_id="s", type="cost")
        assert event.call_id is None
        assert event.model is None
        assert event.provider is None
        assert event.tokens_in is None
        assert event.tokens_out is None
        assert event.cost_usd is None
        assert event.latency_ms is None
        assert event.action is None
        assert event.detector is None
        assert event.severity is None
        assert event.reason is None
        assert event.metadata is None

    def test_action_valid_values(self):
        for action in ("pass", "flag", "block"):
            event = ObservabilityEvent(
                session_id="s",
                type="enforcement",
                action=action,  # type: ignore[arg-type]
            )
            assert event.action == action

    def test_invalid_action_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ObservabilityEvent(
                session_id="s",
                type="enforcement",
                action="allow",  # type: ignore[arg-type]
            )


class TestFlaggedCall:
    def test_creation_with_defaults(self):
        fc = FlaggedCall(call_id="call-xyz", session_id="sess-1")
        assert len(fc.id) == 32
        assert fc.id.isalnum()
        assert "T" in fc.timestamp
        assert fc.action == "flag"
        assert fc.input_messages == []
        assert fc.output_text is None
        assert fc.verdict is None

    def test_unique_ids_per_instance(self):
        fc1 = FlaggedCall(call_id="c", session_id="s")
        fc2 = FlaggedCall(call_id="c", session_id="s")
        assert fc1.id != fc2.id

    def test_verdict_update_via_model_copy(self):
        fc = FlaggedCall(
            call_id="call-1",
            session_id="sess-1",
            action="flag",
            detector="secret_detector",
            severity="high",
            reason="API key detected",
        )
        reviewed = fc.model_copy(
            update={
                "verdict": "confirmed",
                "verdict_by": "user@example.com",
                "verdict_at": "2026-04-30T12:00:00+00:00",
                "verdict_note": "True positive, key revoked",
            }
        )
        assert reviewed.verdict == "confirmed"
        assert reviewed.verdict_by == "user@example.com"
        assert reviewed.verdict_note == "True positive, key revoked"
        # Original unchanged
        assert fc.verdict is None

    def test_with_output_text_none_blocked_before_llm(self):
        fc = FlaggedCall(
            call_id="call-blocked",
            session_id="sess-2",
            action="block",
            output_text=None,
            detector="pii_detector",
            reason="PII in prompt",
        )
        assert fc.action == "block"
        assert fc.output_text is None

    def test_with_input_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is my SSN?"},
        ]
        fc = FlaggedCall(call_id="c", session_id="s", input_messages=messages)
        assert fc.input_messages == messages
        assert len(fc.input_messages) == 2

    def test_serialization_via_model_dump(self):
        fc = FlaggedCall(
            call_id="call-99",
            session_id="sess-99",
            action="block",
            model="claude-3-5-sonnet",
            detector="secret_detector",
            severity="critical",
            reason="Private key leaked",
            input_messages=[{"role": "user", "content": "here is my key: sk-..."}],
        )
        d = fc.model_dump()
        assert d["call_id"] == "call-99"
        assert d["action"] == "block"
        assert d["model"] == "claude-3-5-sonnet"
        assert d["severity"] == "critical"
        assert len(d["input_messages"]) == 1

    def test_invalid_action_raises_validation_error(self):
        with pytest.raises(ValidationError):
            FlaggedCall(
                call_id="c",
                session_id="s",
                action="allow",  # type: ignore[arg-type]
            )

    def test_invalid_verdict_raises_validation_error(self):
        with pytest.raises(ValidationError):
            FlaggedCall(
                call_id="c",
                session_id="s",
                verdict="accepted",  # type: ignore[arg-type]
            )

    def test_metadata_stored_correctly(self):
        meta = {"org_id": "org-123", "policy": "strict"}
        fc = FlaggedCall(call_id="c", session_id="s", metadata=meta)
        assert fc.metadata == meta
