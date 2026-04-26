from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


@dataclass(slots=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_seconds: int = 60

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def check_ollama_available(self) -> tuple[bool, str | None]:
        req = Request(self._url("/api/tags"), method="GET")
        try:
            with urlopen(req, timeout=self.timeout_seconds):
                return True, None
        except (HTTPError, URLError, TimeoutError) as exc:
            return False, f"Ollama unavailable: {exc}"

    def generate(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = Request(
            self._url("/api/generate"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code == 404:
                raise OllamaError(
                    f"Model not found on Ollama server: '{self.model}'. Run: ollama pull {self.model}"
                ) from exc
            raise OllamaError(f"Ollama returned HTTP error: {exc}") from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama request timed out: {exc}") from exc
        except URLError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError("Malformed JSON response from Ollama.") from exc

        text = data.get("response")
        if not isinstance(text, str):
            raise OllamaError("Malformed Ollama response: missing 'response' string field.")
        return text.strip()
