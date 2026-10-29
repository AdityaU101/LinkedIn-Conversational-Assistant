"""Provider interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ProviderResponse:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: dict = field(default_factory=dict)


class Provider:
    name: str = "base"
    local: bool = False

    def available(self, model: Optional[str] = None) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def complete(self, messages: list[dict], model: str, json_mode: bool = False, temperature: float = 0.0,
                 max_tokens: int = 1500, timeout: float = 120) -> ProviderResponse:  # pragma: no cover
        raise NotImplementedError
