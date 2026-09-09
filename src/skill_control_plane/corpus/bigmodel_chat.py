from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_BIGMODEL_CHAT_MODEL = "glm-5.2"

UrlopenFn = Callable[..., object]


def canonical_json_text(content: str) -> str:
    """Validate model output as one JSON object and return canonical JSON text."""
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json\n"):
                text = text[5:].strip()

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("BigModel completion must be one JSON object")
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True)
class BigModelChatClient:
    """Minimal OpenAI-compatible BigModel chat client for offline extraction."""

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1200
    reasoning_effort: str = "none"
    timeout: float = 120.0
    urlopen_fn: UrlopenFn = urlopen

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("BIGMODEL_API_KEY", "")
        self.base_url = (
            self.base_url
            or os.environ.get("BIGMODEL_BASE_URL")
            or DEFAULT_BIGMODEL_BASE_URL
        ).rstrip("/")
        self.model = (
            self.model
            or os.environ.get("BIGMODEL_CHAT_MODEL")
            or DEFAULT_BIGMODEL_CHAT_MODEL
        )
        if not self.api_key:
            raise RuntimeError("BIGMODEL_API_KEY is required")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.reasoning_effort not in {
            "max", "xhigh", "high", "medium", "low", "minimal", "none"
        }:
            raise ValueError("unsupported reasoning_effort")

    def __call__(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a deterministic information-extraction engine. "
                            "Return only the JSON object requested by the user prompt."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self.urlopen_fn(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(
                f"BigModel chat request failed: HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"BigModel chat request failed: {type(exc.reason).__name__}"
            ) from exc

        data = json.loads(raw)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("BigModel chat response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("BigModel chat response has no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            finish_reason = choices[0].get("finish_reason")
            reasoning = message.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = message.get("reasoning")
            reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
            has_tool_calls = bool(message.get("tool_calls"))
            raise RuntimeError(
                "BigModel chat response has no text content "
                f"(finish_reason={finish_reason!r}, "
                f"reasoning_chars={reasoning_chars}, "
                f"tool_calls={has_tool_calls})"
            )

        return canonical_json_text(content)
