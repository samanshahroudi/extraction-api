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


def test_failed_extraction_can_retry_same_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    calls = []

    def flaky(text):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("temporary provider failure")
        return Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)

    client = TestClient(create_app(flaky, str(tmp_path / "extract.db")))
    headers = {"idempotency-key": "retry-request"}
    payload = {"text": "Cannot log into my account"}
    failed = client.post("/extract", headers=headers, json=payload)
    assert failed.status_code == 503
    assert failed.json() == {"detail": "extraction unavailable"}
    recovered = client.post("/extract", headers=headers, json=payload)
    assert recovered.status_code == 200
    cached = client.post("/extract", headers=headers, json=payload)
    assert cached.json() == recovered.json()
    assert cached.status_code == 200
    assert len(calls) == 2


def test_blank_text_is_rejected_before_extraction(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    calls = []

    def fake(text):
        calls.append(text)
        return Ticket(category="other", urgency=1, summary="No useful content", needs_human=True)

    client = TestClient(create_app(fake, str(tmp_path / "extract.db")))
    headers = {"idempotency-key": "blank-request"}
    for text in [" " * 10, " \t\n" * 4]:
        assert client.post("/extract", headers=headers, json={"text": text}).status_code == 422
    assert calls == []
    # Validation must not reserve the key; preserve meaningful surrounding whitespace.
    text = "  Cannot log into my account  "
    assert client.post("/extract", headers=headers, json={"text": text}).status_code == 200
    assert calls == [text]
