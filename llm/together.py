import asyncio
import logging
import os
from together import AsyncTogether
from together.types.chat.completion_create_params import MessageChatCompletionUserMessageParam, MessageChatCompletionSystemMessageParam

from llm import LLM

logger = logging.getLogger(__name__)

class Together(LLM):
    def __init__(self):
        self.__client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
        self.__model = os.getenv("TOGETHER_MODEL")
        self.__semaphore = asyncio.Semaphore(self.parallelism())

    def parallelism(self):
        return int(os.getenv("TOGETHER_PARALLELISM", "100"))

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float) -> LLM.SimpleResponse:
        async with self.__semaphore:
            try:
                response = await self.__client.chat.completions.create(
                    model=self.__model,
                    messages=[
                        MessageChatCompletionUserMessageParam(role="user", content=question),
                        MessageChatCompletionSystemMessageParam(role="system", content=system_prompt),
                    ],
                    logprobs=1,
                    temperature=temperature,
                )
            except Exception as exc:
                logger.error(
                    "Together AI request failed: %s",
                    exc,
                    exc_info=True,
                    extra={"status_code": getattr(exc, "status_code", None), "attempt_number": 1},
                )
                raise

            return LLM.SimpleResponse(
                answer=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )