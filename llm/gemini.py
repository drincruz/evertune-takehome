import os
import httpx
import google.auth
import google.auth.transport.requests
from llm import LLM

class Gemini(LLM):
    def __init__(self, model_name: str = None, project: str = None, location: str = None):
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

    def parallelism(self) -> int:
        return int(os.getenv("GEMINI_PARALLELISM", "100"))

    def _get_token(self) -> str:
        if not self.__credentials.valid:
            auth_req = google.auth.transport.requests.Request()
            self.__credentials.refresh(auth_req)
        return self.__credentials.token

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float) -> LLM.SimpleResponse:
        token = self._get_token()
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

        response = await self.__http_client.post(self.__url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Vertex AI returned HTTP status {response.status_code}: {response.text}")

        data = response.json()
        
        candidates = data.get("candidates", [])
        if not candidates:
            answer = ""
        else:
            parts = candidates[0].get("content", {}).get("parts", [])
            answer = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        return LLM.SimpleResponse(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
