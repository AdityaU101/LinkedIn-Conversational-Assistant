"""Ollama (local) provider — default for anything containing resume text."""
from __future__ import annotations

import time
from typing import Optional

import httpx

from llm.providers.base import Provider, ProviderError, ProviderResponse


class OllamaProvider(Provider):
    local = True

    def __init__(self, name: str, base_url: str, client: Optional[httpx.Client] = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client()
        self._tags: tuple[float, set[str]] | None = None

    def models(self) -> set[str]:
        if self._tags and time.time() - self._tags[0] < 30:
            return self._tags[1]
        try:
            r = self.client.get(f"{self.base_url}/api/tags", timeout=2.0)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
        except Exception:
            names = set()
        self._tags = (time.time(), names)
        return names

    def available(self, model: Optional[str] = None) -> bool:
        names = self.models()
        if not names:
            return False
        if model is None:
            return True
        return model in names or f"{model}:latest" in names

    def complete(self, messages, model, json_mode=False, temperature=0.0, max_tokens=1500, timeout=120):
        body = {"model": model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if json_mode:
            body["format"] = "json"
        try:
            r = self.client.post(f"{self.base_url}/api/chat", json=body, timeout=timeout)
        except httpx.TimeoutException as e:
            raise ProviderError(f"ollama timeout: {e}", retryable=True)
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama connection error: {type(e).__name__}", retryable=True)
        if r.status_code != 200:
            raise ProviderError(f"ollama HTTP {r.status_code}: {r.text[:200]}", retryable=r.status_code >= 500)
        data = r.json()
        return ProviderResponse(text=(data.get("message") or {}).get("content", ""),
                                input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"),
                                raw={"total_duration_ns": data.get("total_duration")})
