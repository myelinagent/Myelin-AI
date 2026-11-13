"""Minimal chat-completions client for any OpenAI-compatible endpoint.

Standard library only, so the harness adds no dependencies. Configure with
environment variables:

* ``MYELIN_BENCH_BASE_URL`` — endpoint root (default ``https://api.openai.com/v1``).
* ``MYELIN_BENCH_API_KEY`` — bearer token; falls back to ``OPENAI_API_KEY``.
  With neither set, no ``Authorization`` header is sent.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLMError(RuntimeError):
    """The endpoint kept failing after every retry."""


class ChatClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("MYELIN_BENCH_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self._api_key = (
            api_key
            or os.environ.get("MYELIN_BENCH_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.0) -> Completion:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        # No key configured means the environment authenticates the call
        # itself (for example an egress proxy that injects credentials).
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        last_error = "no attempt made"
        for attempt in range(self._max_retries + 1):
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read())
                choice = payload["choices"][0]["message"]["content"] or ""
                usage = payload.get("usage") or {}
                return Completion(
                    text=choice.strip(),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    seconds=time.monotonic() - started,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in _RETRYABLE_STATUS:
                    break
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                # Network failure, timeout, or a malformed response body.
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self._max_retries:
                time.sleep(min(2.0**attempt, 30.0))
        raise LLMError(f"{self.model} failed after retries: {last_error}")
