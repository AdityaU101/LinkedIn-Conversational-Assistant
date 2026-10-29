"""OpenAI-compatible chat-completions provider (Groq, OpenRouter, proprietary baselines)."""
from __future__ import annotations

from typing import Optional

import httpx

from common.config import env
from llm.providers.base import Provider, ProviderError, ProviderResponse


class OpenAICompatibleProvider(Provider):
    local = False

    def __init__(self, name: str, base_url: Optional[str], api_key_env: str, client: Optional[httpx.Client] = None,
                 extra_headers: Optional[dict] = None):
        self.name = name
        self.base_url = (base_url or "").rstrip("/")
        self.api_key_env = api_key_env
        self.client = client or httpx.Client()
        self.extra_headers = extra_headers or {}

    def available(self, model: Optional[str] = None) -> bool:
        # Key presence only: no network call, no cost. Use scripts/check_providers.py for a live check.
        return bool(self.base_url and env(self.api_key_env))

    def complete(self, messages, model, json_mode=False, temperature=0.0, max_tokens=1500, timeout=120):
        key = env(self.api_key_env)
        if not key:
            raise ProviderError(f"{self.name}: {self.api_key_env} not set")
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}", **self.extra_headers}
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            raise ProviderError(f"{self.name} timeout", retryable=True)
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} connection error: {type(e).__name__}", retryable=True)
        if r.status_code != 200:
            # never include request headers (API key) in errors
            raise ProviderError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}",
                                retryable=r.status_code in (408, 409, 429) or r.status_code >= 500)
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return ProviderResponse(text=(choice.get("message") or {}).get("content") or "",
                                input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                                raw={"id": data.get("id"), "model": data.get("model"),
                                     "cost": usage.get("cost")})  # OpenRouter reports cost when available
