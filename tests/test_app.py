import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APITimeoutError, RateLimitError

from extraction_api.app import Ticket, create_app, extract_live


def test_database_contention_is_retryable_without_extraction(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    path = tmp_path / "extract.db"
    ticket = Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)
    extractor = Mock(return_value=ticket)
    client = TestClient(create_app(extractor, str(path)), raise_server_exceptions=False)
    connect = sqlite3.connect
    # Exercise a real SQLite lock without waiting for the production timeout.
    monkeypatch.setattr("extraction_api.app.sqlite3.connect",
                        lambda *args, **kwargs: connect(*args, **{**kwargs, "timeout": 0}))
    headers = {"idempotency-key": "contended-request"}
    payload = {"text": "Cannot log into my account"}
    with connect(path) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        response = client.post("/extract", headers=headers, json=payload)
        assert response.status_code == 503
        assert response.json() == {"detail": "database temporarily busy"}
        assert response.headers["retry-after"] == "1"
        extractor.assert_not_called()
    assert client.post("/extract", headers=headers, json=payload).json() == ticket.model_dump()
    assert client.post("/extract", headers=headers, json=payload).json() == ticket.model_dump()
    extractor.assert_called_once_with(payload["text"])


@pytest.mark.parametrize("summary", [" " * 5, " \t\n  "])
def test_ticket_summary_requires_content(summary):
    with pytest.raises(ValueError, match="summary cannot be blank"):
        Ticket(category="other", urgency=1, summary=summary, needs_human=True)


def test_ticket_summary_preserves_meaningful_whitespace():
    ticket = Ticket(category="technical", urgency=3,
                    summary="  Cannot log in  ", needs_human=True)
    assert ticket.summary == "  Cannot log in  "


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


def test_concurrent_duplicate_requests_extract_and_persist_once(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    calls = []
    ticket = Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)

    def fake(text):
        calls.append(text)
        return ticket

    path = tmp_path / "extract.db"
    client = TestClient(create_app(fake, str(path)))
    ready = Barrier(2)
    payload = {"text": "Cannot log into my account"}

    def request():
        ready.wait(timeout=5)
        return client.post("/extract", headers={"idempotency-key": "concurrent-request"},
                           json=payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request) for _ in range(2)]
        responses = [future.result(timeout=15) for future in futures]
    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json() for response in responses] == [ticket.model_dump()] * 2
    assert calls == [payload["text"]]
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT key,result FROM requests").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "concurrent-request"
    assert Ticket.model_validate_json(rows[0][1]) == ticket


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


def test_blank_idempotency_key_is_rejected_before_extraction(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    calls = []

    def fake(text):
        calls.append(text)
        return Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)

    client = TestClient(create_app(fake, str(tmp_path / "extract.db")))
    payload = {"text": "Cannot log into my account"}
    for key in [" " * 8, "\t" * 8]:
        response = client.post("/extract", headers={"idempotency-key": key}, json=payload)
        assert response.status_code == 422
    assert calls == []
    assert client.post("/extract", headers={"idempotency-key": "request-123"},
                       json=payload).status_code == 200
    assert calls == [payload["text"]]


@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError, RateLimitError])
@pytest.mark.parametrize("recover", [False, True])
def test_live_extraction_retries_are_bounded(monkeypatch, error_type, recover):
    request = httpx.Request("POST", "https://provider.example.test/responses")
    if error_type is RateLimitError:
        error = error_type("rate limited", response=httpx.Response(429, request=request), body=None)
    else:
        error = error_type(request=request)
    ticket = Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)
    client = Mock()
    client.responses.parse.side_effect = [error, error, SimpleNamespace(output_parsed=ticket)
                                         if recover else error]
    factory = Mock(return_value=client)
    delays = []
    monkeypatch.setattr("extraction_api.app.OpenAI", factory)
    monkeypatch.setattr("extraction_api.app.time.sleep", delays.append)
    if recover:
        assert extract_live("Cannot log into my account") == ticket
    else:
        with pytest.raises(RuntimeError, match="temporarily unavailable") as exc:
            extract_live("Cannot log into my account")
        assert exc.value.__cause__ is error
    factory.assert_called_once_with(timeout=15, max_retries=0)
    assert client.responses.parse.call_count == 3
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize("outcome", [ValueError("invalid provider output"),
                                     SimpleNamespace(output_parsed=None)])
def test_live_extraction_does_not_retry_invalid_output(monkeypatch, outcome):
    client = Mock()
    client.responses.parse.side_effect = [outcome]
    monkeypatch.setattr("extraction_api.app.OpenAI", Mock(return_value=client))
    delays = []
    monkeypatch.setattr("extraction_api.app.time.sleep", delays.append)
    with pytest.raises(ValueError):
        extract_live("Cannot log into my account")
    assert client.responses.parse.call_count == 1
    assert delays == []


def test_cached_extraction_survives_app_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    path = str(tmp_path / "extract.db")
    ticket = Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)
    headers = {"idempotency-key": "durable-request"}
    payload = {"text": "Cannot log into my account"}
    with TestClient(create_app(lambda text: ticket, path)) as client:
        assert client.post("/extract", headers=headers, json=payload).json() == ticket.model_dump()

    def unavailable(text):
        pytest.fail("persisted requests must not call the extractor after restart")

    with TestClient(create_app(unavailable, path)) as restarted:
        cached = restarted.post("/extract", headers=headers, json=payload)
        assert cached.status_code == 200
        assert cached.json() == ticket.model_dump()
        conflict = restarted.post("/extract", headers=headers,
                                  json={"text": "Different request body"})
        assert conflict.status_code == 409


def test_invalid_extractor_result_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    path = tmp_path / "extract.db"
    valid = Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)
    invalid = valid.model_copy(update={"urgency": 99})
    extractor = Mock(side_effect=[invalid, valid])
    client = TestClient(create_app(extractor, str(path)), raise_server_exceptions=False)
    headers = {"idempotency-key": "invalid-output"}
    payload = {"text": "Cannot log into my account"}
    response = client.post("/extract", headers=headers, json=payload)
    assert response.status_code == 503
    assert response.json() == {"detail": "extraction unavailable"}
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    assert client.post("/extract", headers=headers, json=payload).json() == valid.model_dump()
    assert client.post("/extract", headers=headers, json=payload).json() == valid.model_dump()
    assert extractor.call_count == 2


@pytest.mark.parametrize("supplied", [None, "wrong-key", "demo-key-extra", "demo-ke"])
def test_invalid_api_key_does_not_extract_or_reserve_key(tmp_path, monkeypatch, supplied):
    monkeypatch.setenv("PORTFOLIO_API_KEY", "demo-key")
    ticket = Ticket(category="other", urgency=1, summary="Support requested", needs_human=True)
    extractor = Mock(return_value=ticket)
    path = tmp_path / "extract.db"
    client = TestClient(create_app(extractor, str(path)))
    headers = {"idempotency-key": "auth-request"}
    if supplied is not None:
        headers["x-api-key"] = supplied
    payload = {"text": "Please help with my account"}
    assert client.post("/extract", headers=headers, json=payload).status_code == 401
    extractor.assert_not_called()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    headers["x-api-key"] = "demo-key"
    assert client.post("/extract", headers=headers, json=payload).json() == ticket.model_dump()
    extractor.assert_called_once()


def test_commit_contention_is_retryable_and_rolls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    path = tmp_path / "extract.db"
    ticket = Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)
    extractor = Mock(return_value=ticket)
    client = TestClient(create_app(extractor, str(path)), raise_server_exceptions=False)
    connect = sqlite3.connect
    monkeypatch.setattr("extraction_api.app.sqlite3.connect",
                        lambda *args, **kwargs: connect(*args, **{**kwargs, "timeout": 0}))
    headers = {"idempotency-key": "commit-contention"}
    payload = {"text": "Cannot log into my account"}
    with connect(path) as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM requests").fetchall()
        response = client.post("/extract", headers=headers, json=payload)
        assert response.status_code == 503
        assert response.json() == {"detail": "database temporarily busy"}
        assert response.headers["retry-after"] == "1"
        extractor.assert_called_once()
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    assert client.post("/extract", headers=headers, json=payload).json() == ticket.model_dump()
    assert client.post("/extract", headers=headers, json=payload).json() == ticket.model_dump()
    assert extractor.call_count == 2


def test_connections_close_after_success_cache_conflict_and_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTFOLIO_API_KEY", raising=False)
    connections = []
    connect = sqlite3.connect

    def track_connection(*args, **kwargs):
        # Retain worker-thread connections to inspect their lifetime from the test thread.
        connection = connect(*args, **{**kwargs, "check_same_thread": False})
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track_connection)
    ticket = Ticket(category="technical", urgency=3, summary="Cannot log in", needs_human=True)
    extractor = Mock(side_effect=[ticket, RuntimeError("provider unavailable")])
    client = TestClient(create_app(extractor, str(tmp_path / "extract.db")))
    payload = {"text": "Cannot log into my account"}
    headers = {"idempotency-key": "connection-test"}
    assert client.post("/extract", headers=headers, json=payload).status_code == 200
    assert client.post("/extract", headers=headers, json=payload).status_code == 200
    assert client.post("/extract", headers=headers,
                       json={"text": "Different request body"}).status_code == 409
    assert client.post("/extract", headers={"idempotency-key": "failed-request"},
                       json=payload).status_code == 503
    assert extractor.call_count == 2
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
