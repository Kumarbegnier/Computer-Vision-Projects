from pathlib import Path
import sys

from fastapi.testclient import TestClient


CODE_DIR = Path(__file__).resolve().parents[1] / "Basketball Dribble Analysis" / "Code"
sys.path.insert(0, str(CODE_DIR))
import api  # noqa: E402


def test_health_has_trace_headers() -> None:
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


def test_ready_endpoint_shape() -> None:
    client = TestClient(api.app)
    response = client.get("/ready")
    assert response.status_code in {200, 503}
    payload = response.json()
    assert "status" in payload
    assert "checks" in payload
    assert "default_video_exists" in payload["checks"]


def test_rate_limit_on_non_public_routes(monkeypatch) -> None:
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 1)
    api._rate_limit_buckets.clear()
    client = TestClient(api.app)

    first = client.post("/analyze", json={})
    assert first.status_code in {200, 400, 404, 422}

    second = client.post("/analyze", json={})
    assert second.status_code == 429


def test_optional_api_key_auth(monkeypatch) -> None:
    monkeypatch.setattr(api, "ENABLE_API_KEY_AUTH", True)
    monkeypatch.setattr(api, "API_KEY", "top-secret")
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 100)
    api._rate_limit_buckets.clear()
    client = TestClient(api.app)

    unauthorized = client.post("/analyze", json={})
    assert unauthorized.status_code == 401

    authorized = client.post("/analyze", json={}, headers={"x-api-key": "top-secret"})
    assert authorized.status_code != 401
