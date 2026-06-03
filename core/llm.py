"""
core/llm.py
Thin wrapper around the OpenAI client for GPT-4o / GPT-5.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Type, TypeVar, Dict
from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

MODEL = "gpt-4o"
_MAX_RETRIES = 3


def _backoff(attempt: int) -> None:
    """Exponential backoff with jitter."""
    time.sleep((2 ** attempt) + random.uniform(0, 1))


class LLMClient:
    def __init__(self, api_key: str, model: str = MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def _record_usage(self, usage) -> None:
        if usage:
            self._usage["prompt_tokens"] += usage.prompt_tokens
            self._usage["completion_tokens"] += usage.completion_tokens

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                self._record_usage(resp.usage)
                return resp.choices[0].message.content or ""
            except RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                _backoff(attempt)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < _MAX_RETRIES - 1:
                    _backoff(attempt)
                else:
                    raise
        return ""

    def structured(self, system: str, user: str, schema: Type[T], temperature: float = 0.2) -> T:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system + "\n\nRespond ONLY with valid JSON matching the schema."},
                        {"role": "user",   "content": user},
                    ],
                )
                self._record_usage(resp.usage)
                raw = resp.choices[0].message.content or "{}"
                data = json.loads(raw)
                return schema.model_validate(data)
            except RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                _backoff(attempt)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < _MAX_RETRIES - 1:
                    _backoff(attempt)
                else:
                    raise
        raise RuntimeError("LLM structured call failed after all retries")

    @property
    def total_tokens(self) -> int:
        return self._usage["prompt_tokens"] + self._usage["completion_tokens"]

    @property
    def usage_summary(self) -> dict:
        return {**self._usage, "total": self.total_tokens, "model": self.model}
