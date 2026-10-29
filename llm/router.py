"""Provider-independent model router.

* Task -> ordered provider/model chain from configs/models.yaml.
* PII policy: tasks marked ``pii: true`` (resume text) are only routed to
  local providers unless ``privacy.allow_remote_pii`` is true.
* Retries transient errors, falls back along the chain, logs every call
  (provider, model, task, latency, tokens, cost, success, error, retries).
* ``complete_json`` validates output against a Pydantic model and performs at
  most one repair retry.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from common.config import env, get_model_config, get_settings
from llm import call_log
from llm.providers.base import Provider, ProviderError, ProviderResponse
from llm.providers.ollama_provider import OllamaProvider
from llm.providers.openai_compatible import OpenAICompatibleProvider

log = logging.getLogger(__name__)


class NoProviderAvailable(RuntimeError):
    pass


class LLMOutputError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    retries: int = 0
    data: Any = None
    parsed: Optional[BaseModel] = None
    repaired: bool = False
    call_ids: list[int] = field(default_factory=list)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (tolerates code fences / chatter)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        while start >= 0:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(t)):
                ch = t[i]
                if in_str:
                    esc = (ch == "\\") and not esc
                    if ch == '"' and not esc:
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = t.find(open_c, start + 1)
    raise LLMOutputError("No valid JSON found in model output")


class ModelRouter:
    def __init__(self, config: Optional[dict] = None, providers: Optional[dict[str, Provider]] = None,
                 log_calls: bool = True):
        self.config = config or get_model_config()
        self.gen = self.config.get("generation", {})
        self.providers: dict[str, Provider] = providers or self._build_providers()
        self.log_calls = log_calls
        self.allow_remote_pii = bool(get_settings().get("privacy.allow_remote_pii", False))

    def _build_providers(self) -> dict[str, Provider]:
        out: dict[str, Provider] = {}
        for name, p in self.config.get("providers", {}).items():
            base = env(p["base_url_env"]) if p.get("base_url_env") else None
            base = base or p.get("base_url") or p.get("default_base_url")
            if p["type"] == "ollama":
                out[name] = OllamaProvider(name, base or "http://localhost:11434")
            elif p["type"] == "openai_compatible":
                headers = {"HTTP-Referer": "https://localhost/linkedin-assistant", "X-Title": "LinkedIn Assistant"} \
                    if name == "openrouter" else None
                out[name] = OpenAICompatibleProvider(name, base, p["api_key_env"], extra_headers=headers)
        return out

    # ---------------------------------------------------------------- routing
    def task_config(self, task: str) -> dict:
        tasks = self.config.get("tasks", {})
        if task not in tasks:
            raise KeyError(f"Unknown LLM task {task!r}")
        return tasks[task]

    def permitted(self, task: str, provider: str) -> bool:
        pii = self.task_config(task).get("pii", False) if task in self.config.get("tasks", {}) else False
        prov = self.providers.get(provider)
        return bool(prov) and (prov.local or not pii or self.allow_remote_pii)

    def resolve(self, task: str, provider: Optional[str] = None, model: Optional[str] = None) -> list[tuple[str, str]]:
        if provider and model:
            if task in self.config.get("tasks", {}) and not self.permitted(task, provider):
                raise PermissionError(f"Task {task!r} contains personal data and may not use remote provider {provider!r}")
            return [(provider, model)]
        chain = [(c["provider"], c["model"]) for c in self.task_config(task)["chain"]]
        return [(p, m) for p, m in chain if self.permitted(task, p)]

    def available_chain(self, task: str) -> list[tuple[str, str]]:
        return [(p, m) for p, m in self.resolve(task) if self.providers[p].available(m)]

    def is_available(self, task: str) -> bool:
        try:
            return bool(self.available_chain(task))
        except KeyError:
            return False

    def status(self) -> dict:
        out = {"providers": {}, "tasks": {}}
        for name, p in self.providers.items():
            out["providers"][name] = {"local": p.local, "configured": p.available(None) if not p.local else None}
            if p.local:
                out["providers"][name]["configured"] = p.available(None)
        for task in self.config.get("tasks", {}):
            chain = self.resolve(task)
            avail = [f"{p}/{m}" for p, m in chain if self.providers[p].available(m)]
            out["tasks"][task] = {"chain": [f"{p}/{m}" for p, m in chain], "available": avail,
                                  "pii": self.task_config(task).get("pii", False)}
        return out

    def _cost(self, provider: str, model: str, resp: ProviderResponse) -> Optional[float]:
        if resp.raw.get("cost") is not None:
            return float(resp.raw["cost"])
        price = (self.config.get("pricing") or {}).get(f"{provider}/{model}")
        if not price or price.get("input") is None or price.get("output") is None:
            return None
        if resp.input_tokens is None or resp.output_tokens is None:
            return None
        return (resp.input_tokens * price["input"] + resp.output_tokens * price["output"]) / 1e6

    # ------------------------------------------------------------------ calls
    def complete(self, task: str, messages: list[dict], json_mode: bool = False, provider: Optional[str] = None,
                 model: Optional[str] = None, experiment: Optional[str] = None, **kw) -> LLMResult:
        chain = self.resolve(task, provider, model)
        if not chain:
            raise NoProviderAvailable(f"No permitted provider for task {task!r}")
        errors = []
        for prov_name, model_name in chain:
            prov = self.providers[prov_name]
            if not (provider and model) and not prov.available(model_name):
                errors.append(f"{prov_name}/{model_name}: unavailable")
                continue
            retries = 0
            max_retries = int(self.gen.get("max_retries", 2))
            while True:
                t0 = time.perf_counter()
                try:
                    resp = prov.complete(messages, model_name, json_mode=json_mode,
                                         temperature=kw.get("temperature", self.gen.get("temperature", 0.0)),
                                         max_tokens=kw.get("max_tokens", self.gen.get("max_tokens", 1500)),
                                         timeout=kw.get("timeout", self.gen.get("timeout_s", 120)))
                    latency = (time.perf_counter() - t0) * 1000
                    cost = self._cost(prov_name, model_name, resp)
                    cid = call_log.log_call(task, prov_name, model_name, latency, resp.input_tokens, resp.output_tokens,
                                            cost, True, None, retries, experiment) if self.log_calls else -1
                    return LLMResult(resp.text, prov_name, model_name, latency, resp.input_tokens, resp.output_tokens,
                                     cost, retries, call_ids=[cid])
                except ProviderError as e:
                    latency = (time.perf_counter() - t0) * 1000
                    if self.log_calls:
                        call_log.log_call(task, prov_name, model_name, latency, None, None, None, False, str(e),
                                          retries, experiment)
                    if e.retryable and retries < max_retries:
                        retries += 1
                        time.sleep(min(8.0, 0.5 * 2 ** retries))
                        continue
                    errors.append(f"{prov_name}/{model_name}: {e}")
                    break
        raise NoProviderAvailable("; ".join(errors) or f"No provider available for {task!r}")

    def complete_json(self, task: str, messages: list[dict], schema: Optional[Type[BaseModel]] = None,
                      repair: bool = True, **kw) -> LLMResult:
        res = self.complete(task, messages, json_mode=True, **kw)
        try:
            data = extract_json(res.text)
            parsed = schema.model_validate(data) if schema else None
            res.data, res.parsed = data, parsed
            return res
        except (LLMOutputError, ValidationError) as err:
            if not repair:
                raise LLMOutputError(f"Invalid structured output: {err}")
            fix = messages + [
                {"role": "assistant", "content": res.text[:6000]},
                {"role": "user", "content": "Your previous output was invalid: "
                 f"{str(err)[:800]}\nReturn ONLY corrected JSON that satisfies the schema. Do not add new facts."},
            ]
            res2 = self.complete(task, fix, json_mode=True, **{**kw, "provider": res.provider, "model": res.model})
            try:
                data = extract_json(res2.text)
                parsed = schema.model_validate(data) if schema else None
            except (LLMOutputError, ValidationError) as err2:
                raise LLMOutputError(f"Invalid structured output after repair: {err2}")
            res2.data, res2.parsed, res2.repaired = data, parsed, True
            res2.call_ids = res.call_ids + res2.call_ids
            res2.latency_ms += res.latency_ms
            res2.input_tokens = (res.input_tokens or 0) + (res2.input_tokens or 0)
            res2.output_tokens = (res.output_tokens or 0) + (res2.output_tokens or 0)
            if res.cost_usd is not None and res2.cost_usd is not None:
                res2.cost_usd += res.cost_usd
            return res2


@lru_cache(maxsize=1)
def get_router() -> ModelRouter:
    return ModelRouter()
