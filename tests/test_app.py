from fastapi.testclient import TestClient

from extraction_api.app import Ticket, create_app


def test_idempotency_and_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_API_KEY", "demo")
    calls = []

    def fake(text):
        calls.append(text)
        return Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)

    client = TestClient(create_app(fake, str(tmp_path / "extract.db")))
    headers = {"idempotency-key": "request-123", "x-api-key": "demo"}
    payload = {"text": "Cannot log into my account"}
    assert client.post("/extract", headers=headers, json=payload).status_code == 200
    assert client.post("/extract", headers=headers, json=payload).status_code == 200
    assert len(calls) == 1
    assert client.post("/extract", headers=headers, json={"text": "Different request body"}).status_code == 409
    assert client.post("/extract", headers={"idempotency-key": "another-key"}, json=payload).status_code == 401
