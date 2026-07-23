import asyncio
import os
import httpx
import google.auth
import google.auth.transport.requests
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from llm import LLM

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

class VertexTransientError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"Vertex AI returned HTTP status {status_code}: {body}")
        self.status_code = status_code

class Gemini(LLM):
    def __init__(self, model_name: str | None = None, project: str | None = None, location: str | None = None):
        self.__model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.__project = project or os.getenv("VERTEX_PROJECT", "evertune-tests")
        self.__location = location or os.getenv("VERTEX_LOCATION", "us-central1")

        self.__credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.__url = (
            f"https://{self.__location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.__project}/locations/{self.__location}/"
            f"publishers/google/models/{self.__model}:generateContent"
        )
        self.__http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=200, max_connections=200),
            timeout=60.0
        )
        self.__retrying = AsyncRetrying(
            retry=retry_if_exception_type((VertexTransientError, httpx.TransportError)),
            stop=stop_after_attempt(int(os.getenv("GEMINI_MAX_RETRIES", "5"))),
            wait=wait_random_exponential(multiplier=1, max=float(os.getenv("GEMINI_BACKOFF_MAX_SECONDS", "20"))),
            reraise=True,
        )
        self.__token_lock = asyncio.Lock()

    def parallelism(self) -> int:
        return int(os.getenv("GEMINI_PARALLELISM", "100"))

    async def _get_token(self) -> str:
        if self.__credentials.valid:
            return self.__credentials.token
        async with self.__token_lock:
            if not self.__credentials.valid:
                auth_req = google.auth.transport.requests.Request()
                await asyncio.to_thread(self.__credentials.refresh, auth_req)
        return self.__credentials.token

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float) -> LLM.SimpleResponse:
        token = await self._get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": question}]
                }
            ],
            "generationConfig": {
                "temperature": temperature
            }
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        async def _send() -> httpx.Response:
            response = await self.__http_client.post(self.__url, headers=headers, json=payload)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise VertexTransientError(response.status_code, response.text)
            if response.status_code != 200:
                raise RuntimeError(f"Vertex AI returned HTTP status {response.status_code}: {response.text}")
            return response

        response: httpx.Response = await self.__retrying(_send)

        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            answer = ""
            finish_reason = data.get("promptFeedback", {}).get("blockReason")
        else:
            parts = candidates[0].get("content", {}).get("parts", [])
            answer = "".join(p.get("text", "") for p in parts)
            finish_reason = candidates[0].get("finishReason")

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        return LLM.SimpleResponse(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason
        )
