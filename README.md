# Support ticket extraction API

## The problem

Support teams receive free-form messages but downstream systems need a small, stable ticket schema. A demo that asks an LLM for JSON is easy; an API that handles malformed output, retries, duplicate requests, and client errors is the engineering work.

## Architecture and how it works

`POST /extract` validates the input, checks an idempotency key in SQLite, sends the text to the OpenAI Responses API with a Pydantic output schema, validates the result, and commits it. The transaction serializes requests with the same key. Transient network/rate failures get bounded exponential retries. A health endpoint stays independent of the model provider.

`HTTP request → validation → idempotency lookup → Responses API → Pydantic result → SQLite → response`

The model sees a system instruction that treats ticket text as data. It cannot select arbitrary tools. An optional local API key protects the endpoint. No request text is logged by this service.

## Concepts and choices

FastAPI supplies an explicit HTTP contract; Pydantic supplies schema enforcement; the official OpenAI SDK supplies structured output. SQLite makes the example runnable without infrastructure. The extractor is injected into `create_app`, so tests exercise the HTTP behavior without billing an API. The source lives in `app.py`; the API tests live in `tests/test_app.py`.

## Run and example

Install with `python -m pip install -e ".[dev]"`. Set `OPENAI_API_KEY` and optionally `PORTFOLIO_API_KEY`, then run:

```bash
python -m uvicorn extraction_api.app:app --port 8001
curl -X POST localhost:8001/extract -H 'content-type: application/json' \
  -H 'idempotency-key: example-001' -H 'x-api-key: change-me-for-local-demo' \
  -d '{"text":"I have been unable to log in since yesterday and need help."}'
```

Use the same key and body to see the cached result. Use the same key and different body to see HTTP 409. `PORTFOLIO_DB` controls the database location.

If SQLite cannot acquire the write lock or commit within its timeout, the API returns HTTP 503 with `Retry-After: 1`. Retry with the same key and body. Lock acquisition failures do not call the extractor; commit failures roll back the cached result and may require extraction again.

## Trade-offs, limitations, and next production steps

The database transaction remains open during the network request, which keeps the code simple but limits write concurrency. For higher throughput, reserve a request row first, execute outside the transaction, and use a lease plus recovery worker. API-key comparison should become a real authentication layer with per-tenant quotas. Add request IDs, metrics, redacted traces, explicit cost accounting, and migration tooling. Structured output validates shape; it does not prove the classification is correct. Build a labeled extraction set and track errors by category before release.

## Interview preparation

Explain why idempotency keys are tied to payload hashes, why only transient failures are retried, how schema validation differs from semantic correctness, and how you would prevent duplicate work across multiple replicas.

## Verify

Run `python -m pytest -q` and `python -m ruff check .` from this repository.
