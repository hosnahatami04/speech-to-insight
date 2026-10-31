from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    host: str = "http://localhost:11434"
    model_tag: str = "qwen2.5:3b-instruct"
    temperature: float = 0.0
    seed: int = 0
    timeout_s: float = 120.0
    retries: int = 1


class OllamaError(RuntimeError):
    """Raised when the Ollama server returns a non-success status or malformed JSON."""


class OllamaClient:
    """Minimal HTTP client for Ollama's /api/generate endpoint. Deterministic
    (temperature 0, fixed seed) and bounded by a per-call timeout. No framework
    (LangChain/LlamaIndex) by design — the request is small and auditable."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config or OllamaConfig()

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Call the model with ``format=json`` and return the parsed object.
        One retry on malformed JSON or transport errors."""

        import urllib.error
        import urllib.request

        payload = json.dumps(
            {
                "model": self._config.model_tag,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": self._config.temperature,
                    "seed": self._config.seed,
                },
            }
        ).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self._config.retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self._config.host}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._config.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                envelope = json.loads(body)
                text = envelope.get("response", "")
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise OllamaError(f"expected JSON object, got {type(parsed).__name__}")
                return parsed
            except (urllib.error.URLError, OSError, json.JSONDecodeError, OllamaError) as exc:
                last_error = exc
                if attempt >= self._config.retries:
                    raise OllamaError(
                        f"generate failed after {attempt + 1} attempts: {exc}"
                    ) from exc
        # Unreachable but keeps mypy happy about the return type.
        raise OllamaError(f"generate failed: {last_error}")
