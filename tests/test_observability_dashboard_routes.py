from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aceteam_aep.observability.events import FlaggedCall
from aceteam_aep.observability.store import SqliteEventStore


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEventStore]:
    s = SqliteEventStore(str(tmp_path / "test.db"))
    await s.initialize()
    await s.record_flagged_call(
        FlaggedCall(
            call_id="c1",
            session_id="sess_1",
            action="flag",
            detector="pii",
            severity="medium",
            reason="PII detected",
            model="gpt-4o",
            input_messages=[{"role": "user", "content": "test"}],
            output_text="response with PII",
        )
    )
    yield s
    await s.close()


@pytest.fixture
def app(store: SqliteEventStore):
    from aceteam_aep.proxy.app import create_proxy_app

    return create_proxy_app(
        target_base_url="https://api.openai.com",
        dashboard=True,
        event_store=store,
    )


def test_list_incidents(app, store):
    client = TestClient(app)
    resp = client.get("/dashboard/api/incidents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["incidents"]) == 1
    assert data["incidents"][0]["call_id"] == "c1"


def test_get_incident_detail(app, store):
    client = TestClient(app)
    resp = client.get("/dashboard/api/incidents/c1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["flagged_call"]["call_id"] == "c1"
    assert "events" in data


def test_get_incident_not_found(app, store):
    client = TestClient(app)
    resp = client.get("/dashboard/api/incidents/nonexistent")
    assert resp.status_code == 404


def test_update_verdict(app, store):
    client = TestClient(app)
    resp = client.patch(
        "/dashboard/api/incidents/c1/verdict",
        json={"verdict": "confirmed", "verdict_by": "user_1", "verdict_note": "real PII"},
    )
    assert resp.status_code == 200
    resp2 = client.get("/dashboard/api/incidents/c1")
    assert resp2.json()["flagged_call"]["verdict"] == "confirmed"


def test_export_incident(app, store):
    client = TestClient(app)
    resp = client.get("/dashboard/api/incidents/c1/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"


def test_timeline_route(app, store):
    client = TestClient(app)
    resp = client.get("/dashboard/api/timeline")
    assert resp.status_code == 200
    assert "spans" in resp.json()


def test_traffic_route(app, store):
    client = TestClient(app)
    resp = client.get("/dashboard/api/traffic")
    assert resp.status_code == 200
    assert "total_calls" in resp.json()


def test_topology_route(app, store):
    client = TestClient(app)
    resp = client.get("/dashboard/api/topology")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data
    assert "agents" in data
