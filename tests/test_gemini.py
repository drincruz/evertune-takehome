import pytest
import sys
import os
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from llm import Gemini
from llm.gemini import VertexTransientError


class FakeCredentials:
    valid = True
    token = "fake-token"


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

def test_gemini_custom_parallelism(monkeypatch):
    monkeypatch.setattr("llm.gemini.google.auth.default", lambda scopes: (FakeCredentials(), None))
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_PARALLELISM", "50")
    gemini = Gemini()
    assert gemini.parallelism() == 50


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds(gemini, monkeypatch):
    responses = [
        make_response(429, "quota exceeded"),
        make_response(200, {"candidates": [{"content": {"parts": [{"text": "Paris"}]}}]}),
    ]
    post = AsyncMock(side_effect=responses)
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    response = await gemini.ask_generic_question(
        system_prompt="You are a helpful assistant.",
        question="What is the capital of France?",
        temperature=0.0,
    )

    assert response.answer == "Paris"
    assert post.call_count == 2


@pytest.mark.asyncio
async def test_no_retry_on_non_retryable_status(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(400, "bad request"))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with pytest.raises(RuntimeError) as exc_info:
        await gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )

    assert not isinstance(exc_info.value, VertexTransientError)
    assert post.call_count == 1


@pytest.mark.asyncio
async def test_retries_exhausted_raises(gemini, monkeypatch):
    post = AsyncMock(return_value=make_response(429, "quota exceeded"))
    monkeypatch.setattr(gemini._Gemini__http_client, "post", post)

    with pytest.raises(VertexTransientError):
        await gemini.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )

    assert post.call_count == 3
