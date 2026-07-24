import asyncio
import logging
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from llm import Together


def make_completion(content: str, prompt_tokens: int = 5, completion_tokens: int = 3):
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.fixture
def together(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "fake-key")
    monkeypatch.setenv("TOGETHER_MODEL", "fake-model")
    return Together()


def test_together_default_parallelism(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "fake-key")
    monkeypatch.delenv("TOGETHER_PARALLELISM", raising=False)
    together = Together()
    assert together.parallelism() == 100


def test_together_custom_parallelism(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "fake-key")
    monkeypatch.setenv("TOGETHER_PARALLELISM", "7")
    together = Together()
    assert together.parallelism() == 7


@pytest.mark.asyncio
async def test_ask_generic_question_returns_response(together, monkeypatch):
    create = AsyncMock(return_value=make_completion("Paris"))
    monkeypatch.setattr(together._Together__client.chat.completions, "create", create)

    response = await together.ask_generic_question(
        system_prompt="You are a helpful assistant.",
        question="What is the capital of France?",
        temperature=0.0,
    )

    assert response.answer == "Paris"
    assert response.input_tokens == 5
    assert response.output_tokens == 3
    assert response.attempt_number == 1
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_ask_generic_question_logs_and_reraises_on_failure(together, monkeypatch, caplog):
    create = AsyncMock(side_effect=RuntimeError("upstream failure"))
    monkeypatch.setattr(together._Together__client.chat.completions, "create", create)

    with caplog.at_level(logging.ERROR, logger="llm.together"):
        with pytest.raises(RuntimeError, match="upstream failure"):
            await together.ask_generic_question(
                system_prompt="You are a helpful assistant.",
                question="What is the capital of France?",
                temperature=0.0,
            )

    assert any(
        record.levelno == logging.ERROR and getattr(record, "attempt_number", None) == 1
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_requests(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "fake-key")
    monkeypatch.setenv("TOGETHER_MODEL", "fake-model")
    monkeypatch.setenv("TOGETHER_PARALLELISM", "2")
    together = Together()

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fake_create(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return make_completion("Paris")

    monkeypatch.setattr(together._Together__client.chat.completions, "create", fake_create)

    await asyncio.gather(*(
        together.ask_generic_question(
            system_prompt="You are a helpful assistant.",
            question="What is the capital of France?",
            temperature=0.0,
        )
        for _ in range(10)
    ))

    assert max_in_flight <= 2
