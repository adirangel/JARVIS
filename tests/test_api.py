"""Tests for JARVIS API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "jarvis"


def test_system(client):
    r = client.get("/api/system")
    assert r.status_code == 200
    data = r.json()
    assert "cpu_percent" in data
    assert "ram" in data
    assert "disk" in data
    assert "used_gb" in data["ram"]
    assert "total_gb" in data["ram"]


def test_weather(client):
    r = client.get("/api/weather")
    assert r.status_code == 200
    data = r.json()
    assert "temp_c" in data
    assert "location" in data
    assert "condition" in data


def test_uptime(client):
    r = client.get("/api/uptime")
    assert r.status_code == 200
    data = r.json()
    assert "uptime_formatted" in data
    assert "session" in data
    assert "commands" in data


def test_conversation_get(client):
    r = client.get("/api/conversation")
    assert r.status_code == 200
    data = r.json()
    assert "messages" in data
    assert isinstance(data["messages"], list)


def test_conversation_post_empty(client):
    r = client.post("/api/conversation", json={"text": ""})
    assert r.status_code == 400  # empty message rejected


def test_voice_status(client):
    r = client.get("/api/voice/status")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "message" in data
