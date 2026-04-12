"""Tests for AEP Redis event publisher."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from aceteam_aep.proxy.redis_publisher import (
    build_event,
    publish_event,
)


def test_build_event_llm_call():
    event = build_event(
        instance_id="inst-001",
        event_type="llm_call",
        action="pass",
        message="GPT-4o call: 150 in, 80 out",
        cost_usd=0.003,
        model="gpt-4o",
        tokens_in=150,
        tokens_out=80,
    )
    assert event["instance_id"] == "inst-001"
    assert event["type"] == "llm_call"
    assert event["action"] == "pass"
    assert event["cost_usd"] == 0.003
    assert "id" in event
    assert "timestamp" in event


def test_build_event_safety_block():
    event = build_event(
        instance_id="inst-001",
        event_type="safety_block",
        action="block",
        message="Content filter blocked: agent threat detected",
        detector="agent_threat",
        severity="high",
    )
    assert event["type"] == "safety_block"
    assert event["action"] == "block"
    assert event["detector"] == "agent_threat"
    assert event["cost_usd"] == 0


@pytest.mark.asyncio
async def test_publish_event_xadd_and_publish():
    mock_redis = AsyncMock()
    with patch(
        "aceteam_aep.proxy.redis_publisher._get_redis",
        return_value=mock_redis,
    ):
        event = build_event(
            instance_id="inst-001",
            event_type="llm_call",
            action="pass",
            message="test call",
        )
        await publish_event(event)

        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "instance:inst-001:events"

        mock_redis.publish.assert_called_once()
        pub_args = mock_redis.publish.call_args
        assert pub_args[0][0] == "instance:inst-001:live"


@pytest.mark.asyncio
async def test_publish_event_no_redis_url():
    with patch(
        "aceteam_aep.proxy.redis_publisher._get_redis",
        return_value=None,
    ):
        event = build_event(
            instance_id="inst-001",
            event_type="llm_call",
            action="pass",
            message="test",
        )
        await publish_event(event)  # Should not raise


@pytest.mark.asyncio
async def test_publish_event_payload_is_valid_json():
    mock_redis = AsyncMock()
    with patch(
        "aceteam_aep.proxy.redis_publisher._get_redis",
        return_value=mock_redis,
    ):
        event = build_event(
            instance_id="inst-002",
            event_type="safety_block",
            action="block",
            message="blocked by policy",
            detector="pii",
            severity="medium",
            cost_usd=0.001,
            model="gpt-4o-mini",
            tokens_in=50,
            tokens_out=0,
        )
        await publish_event(event)

        xadd_call = mock_redis.xadd.call_args
        payload_str = xadd_call[0][1]["event"]
        parsed = json.loads(payload_str)
        assert parsed["instance_id"] == "inst-002"
        assert parsed["type"] == "safety_block"
        assert parsed["detector"] == "pii"


@pytest.mark.asyncio
async def test_publish_event_redis_error_does_not_raise():
    mock_redis = AsyncMock()
    mock_redis.xadd.side_effect = Exception("Redis connection lost")
    with patch(
        "aceteam_aep.proxy.redis_publisher._get_redis",
        return_value=mock_redis,
    ):
        event = build_event(
            instance_id="inst-001",
            event_type="llm_call",
            action="pass",
            message="test",
        )
        await publish_event(event)  # Should swallow the exception


def test_build_event_generates_unique_ids():
    e1 = build_event(instance_id="x", event_type="llm_call", action="pass", message="a")
    e2 = build_event(instance_id="x", event_type="llm_call", action="pass", message="a")
    assert e1["id"] != e2["id"]
