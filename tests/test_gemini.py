import asyncio
import logging
import pytest
import sys
import os
import httpx
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from llm import Gemini
from llm.gemini import VertexAuthError, VertexTransientError


class FakeCredentials:
    valid = True
    token = "fake-token"


class RefreshableFakeCredentials:
    def __init__(self):
        self.valid = False
        self.token = "stale-token"
        self.refresh_call_count = 0

    def refresh(self, auth_req):
        self.refresh_call_count += 1
        self.token = "refreshed-token"
        self.valid = True


class FailingRefreshFakeCredentials:
    def __init__(self):
        self.valid = False
        self.token = "stale-token"

    def refresh(self, auth_req):
        raise RuntimeError("revoked service account key")


def make_response(status_code: int, body: dict | str = ""):
    response = AsyncMock()
    response.status_code = status_code
    if isinstance(body, dict):
        response.json = lambda: body
        response.text = str(body)
    else:
        response.text = body
    return response


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "3")
    monkeypatch.setenv("GEMINI_BACKOFF_MAX_SECONDS", "0.01")
    return Gemini()

def test_gemini_requires_project_configuration(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)

    with pytest.raises(ValueError, match="VERTEX_PROJECT"):
        Gemini()


def test_gemini_custom_parallelism(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_PARALLELISM", "50")
    gemini = Gemini()
    assert gemini.parallelism() == 50


def test_warns_when_fd_limit_too_low(monkeypatch, caplog):
    import resource
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_PARALLELISM", "300")
    monkeypatch.setattr(resource, "getrlimit", lambda res: (256, 4096))

    with caplog.at_level(logging.WARNING, logger="llm.gemini"):
        Gemini()

    assert any("file descriptor" in record.getMessage() for record in caplog.records)


def test_no_warning_when_fd_limit_sufficient(monkeypatch, caplog):
    import resource
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_PARALLELISM", "50")
    monkeypatch.setattr(resource, "getrlimit", lambda res: (4096, 4096))

    with caplog.at_level(logging.WARNING, logger="llm.gemini"):
        Gemini()

    assert not any("file descriptor" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_requests(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_PARALLELISM", "2")
    gemini = Gemini()

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fake_post(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return make_response(200, {"candidates": [{"content": {"parts": [{"text": "Paris"}]}}]})

    monkeypatch.setattr(gemini._Gemini__http_client, "post", fake_post)

    await asyncio.gather(*(
        gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )
        for _ in range(10)
    ))

    assert max_in_flight <= 2


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds(gemini, monkeypatch, caplog):
    responses = [
        make_response(429, "quota exceeded"),
        make_response(200, {"candidates": [{"content": {"parts": [{"text": "Paris"}]}}]}),
    ]
    post = AsyncMock(side_effect=responses)
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with caplog.at_level(logging.WARNING, logger="llm.gemini"):
        response = await gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )

    assert response.answer == "Paris"
    assert response.attempt_number == 2
    assert post.call_count == 2
    assert any(record.levelno == logging.WARNING and "retrying" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_no_retry_on_non_retryable_status(gemini, monkeypatch, caplog):
    post = AsyncMock(return_value=make_response(400, "bad request"))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with caplog.at_level(logging.ERROR, logger="llm.gemini"):
        with pytest.raises(RuntimeError) as exc_info:
            await gemini.ask_generic_question(
                system_prompt="You are a helpful assistant.",
                question="What is the capital of France?",
                temperature=0.0,
            )

    assert not isinstance(exc_info.value, VertexTransientError)
    assert exc_info.value.attempt_number == 1
    assert post.call_count == 1
    assert any(
        record.levelno == logging.ERROR and getattr(record, "status_code", None) == 400
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_401_raises_auth_error_not_generic_runtime_error(gemini, monkeypatch, caplog):
    post = AsyncMock(return_value=make_response(401, "invalid credentials"))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with caplog.at_level(logging.ERROR, logger="llm.gemini"):
        with pytest.raises(VertexAuthError) as exc_info:
            await gemini.ask_generic_question(
                system_prompt="You are a helpful assistant.",
                question="What is the capital of France?",
                temperature=0.0,
            )

    assert exc_info.value.status_code == 401
    assert post.call_count == 1
    assert any(
        record.levelno == logging.ERROR and getattr(record, "status_code", None) == 401
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_403_raises_auth_error_distinct_from_generic_bad_request(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(403, "permission denied"))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with pytest.raises(VertexAuthError) as exc_info:
        await gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )

    assert not isinstance(exc_info.value, VertexTransientError)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_retries_exhausted_raises(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(429, "quota exceeded"))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with pytest.raises(VertexTransientError) as exc_info:
        await gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )

    assert post.call_count == 3
    assert exc_info.value.attempt_number == 3


@pytest.mark.asyncio
async def test_request_deadline_cuts_retries_short(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "50")
    monkeypatch.setenv("GEMINI_BACKOFF_MAX_SECONDS", "0.05")
    monkeypatch.setenv("GEMINI_REQUEST_DEADLINE_SECONDS", "0.1")
    gemini = Gemini()

    post = AsyncMock(return_value=make_response(429, "quota exceeded"))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with pytest.raises(VertexTransientError):
        await gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )

    # The deadline should stop retries long before GEMINI_MAX_RETRIES is reached.
    assert post.call_count < 50


@pytest.mark.asyncio
async def test_finish_reason_stop_on_normal_response(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(
        200, {"candidates": [{"content": {"parts": [{"text": "Paris"}]}, "finishReason": "STOP"}]}
    ))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    response = await gemini.ask_generic_question(
        system_prompt="You are a helpful assistant.",
        question="What is the capital of France?",
        temperature=0.0,
    )

    assert response.answer == "Paris"
    assert response.finish_reason == "STOP"
    assert response.attempt_number == 1


@pytest.mark.asyncio
async def test_retries_on_transport_error_then_succeeds(gemini, monkeypatch, caplog):
    responses = [
        httpx.ConnectError("connection refused"),
        make_response(200, {"candidates": [{"content": {"parts": [{"text": "Paris"}]}}]}),
    ]
    post = AsyncMock(side_effect=responses)
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with caplog.at_level(logging.WARNING, logger="llm.gemini"):
        response = await gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )

    assert response.answer == "Paris"
    assert response.attempt_number == 2
    assert post.call_count == 2


@pytest.mark.asyncio
async def test_finish_reason_max_tokens_surfaces_truncation(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(
        200, {"candidates": [{"content": {"parts": [{"text": "Par"}]}, "finishReason": "MAX_TOKENS"}]}
    ))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    response = await gemini.ask_generic_question(
        system_prompt="You are a helpful assistant.",
        question="What is the capital of France?",
        temperature=0.0,
    )

    assert response.answer == "Par"
    assert response.finish_reason == "MAX_TOKENS"


@pytest.mark.asyncio
async def test_finish_reason_safety_block_with_empty_parts(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(
        200, {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
    ))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    response = await gemini.ask_generic_question(
        system_prompt="You are a helpful assistant.",
        question="What is the capital of France?",
        temperature=0.0,
    )

    assert response.answer == ""
    assert response.finish_reason == "SAFETY"


@pytest.mark.asyncio
async def test_finish_reason_prompt_blocked_no_candidates(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(
        200, {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
    ))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    response = await gemini.ask_generic_question(
        system_prompt="You are a helpful assistant.",
        question="What is the capital of France?",
        temperature=0.0,
    )

    assert response.answer == ""
    assert response.finish_reason == "SAFETY"


@pytest.mark.asyncio
async def test_aclose_closes_underlying_http_client(gemini, monkeypatch):
    aclose = AsyncMock()
    monkeypatch.setattr(gemini._Gemini__http_client, "aclose", aclose)

    await gemini.aclose()

    aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_http_client(gemini, monkeypatch):
    aclose = AsyncMock()
    monkeypatch.setattr(gemini._Gemini__http_client, "aclose", aclose)

    async with gemini as g:
        assert g is gemini

    aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_token_refreshes_off_thread_when_invalid(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (RefreshableFakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    gemini = Gemini()

    token = await gemini._get_token()

    assert token == "refreshed-token"
    assert gemini._Gemini__credentials.refresh_call_count == 1


@pytest.mark.asyncio
async def test_get_token_propagates_refresh_failure(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FailingRefreshFakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    gemini = Gemini()

    with pytest.raises(RuntimeError, match="revoked service account key"):
        await gemini._get_token()

    # The token lock must be released even after a failed refresh, so a
    # subsequent caller isn't left deadlocked waiting on it.
    assert not gemini._Gemini__token_lock.locked()


@pytest.mark.asyncio
async def test_get_token_concurrent_callers_share_single_refresh(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (RefreshableFakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    gemini = Gemini()

    real_refresh = gemini._Gemini__credentials.refresh

    def slow_refresh(auth_req):
        # Simulate a slow network call so concurrent callers pile up on the lock.
        import time
        time.sleep(0.05)
        real_refresh(auth_req)

    monkeypatch.setattr(gemini._Gemini__credentials, "refresh", slow_refresh)

    tokens = await asyncio.gather(*(gemini._get_token() for _ in range(10)))

    assert set(tokens) == {"refreshed-token"}
    assert gemini._Gemini__credentials.refresh_call_count == 1
