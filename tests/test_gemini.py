import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from llm import Gemini, LLM

@pytest.mark.asyncio
async def test_gemini_ask_generic_question():
    gemini = Gemini()
    assert gemini.parallelism() == 100

    response = await gemini.ask_generic_question(
        system_prompt="You are a helpful assistant.",
        question="What is the capital of France? Answer with only the city name.",
        temperature=0.0
    )

    assert isinstance(response, LLM.SimpleResponse)
    assert "Paris" in response.answer
    assert response.input_tokens > 0
    assert response.output_tokens > 0

@pytest.mark.asyncio
async def test_gemini_custom_parallelism(monkeypatch):
    monkeypatch.setenv("GEMINI_PARALLELISM", "50")
    gemini = Gemini()
    assert gemini.parallelism() == 50
