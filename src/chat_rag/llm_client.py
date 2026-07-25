from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Protocol

import httpx
import openai
from openai import OpenAI


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    def complete(
        self, system: str, user: str, on_delta: Callable[[str], None] | None = None
    ) -> str: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        max_output_tokens: int = 15_000,
    ) -> None:
        self.model = model
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.Client(transport=transport, timeout=120),
        )

    def complete(
        self, system: str, user: str, on_delta: Callable[[str], None] | None = None
    ) -> str:
        for attempt in range(1, self.max_attempts + 1):
            emitted = False
            parts: list[str] = []
            try:
                if on_delta is not None:
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=self.max_output_tokens,
                        stream=True,
                    )
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content:
                            parts.append(content)
                            on_delta(content)
                            emitted = True
                    answer = "".join(parts)
                    if not answer:
                        raise LLMError("LLM returned an empty answer")
                    return answer
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self.max_output_tokens,
                )
                content = completion.choices[0].message.content
                if not content:
                    raise LLMError("LLM returned an empty answer")
                return content
            except openai.APIStatusError as error:
                status = error.status_code
                if status in {401, 403}:
                    raise LLMError("LLM authentication or permission failed") from None
                if status in {408, 409, 429} or status >= 500:
                    if attempt < self.max_attempts and not emitted:
                        self._backoff(attempt)
                        continue
                    raise LLMError(f"LLM remained unavailable (HTTP {status})") from None
                raise LLMError(f"LLM request was rejected (HTTP {status})") from None
            except openai.APIConnectionError:
                if emitted:
                    return "".join(parts)
                if attempt < self.max_attempts and not emitted:
                    self._backoff(attempt)
                    continue
                raise LLMError("LLM connection failed") from None
            except (IndexError, TypeError) as error:
                raise LLMError("LLM returned an invalid response") from error
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> None:
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        time.sleep(delay + random.uniform(0, delay / 4 if delay else 0))
