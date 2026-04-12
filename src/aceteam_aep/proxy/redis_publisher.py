"""Publish AEP per-request events to Redis for platform consumption.

When AEP_REDIS_URL is set, each request/response cycle publishes an event to:
  - XADD instance:{id}:events  (persistent stream for flush worker)
  - PUBLISH instance:{id}:live  (ephemeral for real-time dashboard)

When AEP_REDIS_URL is absent, all functions are no-ops.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

_redis_client: Any | None = None
_redis_initialized = False


class AepEvent(TypedDict, total=False):
    id: str
    instance_id: str
    type: str
    action: str
    message: str
    cost_usd: float
    model: str | None
    tokens_in: int | None
    tokens_out: int | None
    detector: str | None
    severity: str | None
    timestamp: str


def build_event(
    *,
    instance_id: str,
    event_type: str,
    action: str,
    message: str,
    cost_usd: float = 0,
    model: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    detector: str | None = None,
    severity: str | None = None,
) -> AepEvent:
    return AepEvent(
        id=str(uuid.uuid4()),
        instance_id=instance_id,
        type=event_type,
        action=action,
        message=message,
        cost_usd=cost_usd,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        detector=detector,
        severity=severity,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _get_redis() -> Any | None:
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client

    _redis_initialized = True
    redis_url = os.environ.get("AEP_REDIS_URL")
    if not redis_url:
        return None

    try:
        import redis.asyncio as aioredis

        _redis_client = aioredis.from_url(redis_url, decode_responses=True)
        logger.info("AEP Redis publisher connected to %s", redis_url)
    except Exception:
        logger.warning("Failed to connect to Redis at %s", redis_url, exc_info=True)
        _redis_client = None

    return _redis_client


async def publish_event(event: AepEvent) -> None:
    client = _get_redis()
    if client is None:
        return

    instance_id = event.get("instance_id", "unknown")
    stream_key = f"instance:{instance_id}:events"
    channel_key = f"instance:{instance_id}:live"
    payload = json.dumps(event)

    try:
        await client.xadd(stream_key, {"event": payload}, maxlen=1000)
        await client.publish(channel_key, payload)
    except Exception:
        logger.warning("Failed to publish AEP event to Redis", exc_info=True)
