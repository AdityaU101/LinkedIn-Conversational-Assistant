"""Central configuration: YAML settings + environment variables.

Secrets are only ever read from the environment (optionally populated from a
local, git-ignored ``.env``). They are never logged or written to disk.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # optional: populate os.environ from .env for local development
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _load_yaml(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / name
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings:
    """Typed-ish view over configs/settings.yaml with env overrides."""

    def __init__(self, raw: dict[str, Any] | None = None):
        self.raw = raw if raw is not None else _load_yaml("settings.yaml")
        self.mode: str = os.environ.get("APP_MODE", self.raw.get("mode", "development"))
        if self.mode not in self.raw.get("modes", {}):
            raise ValueError(f"Unknown APP_MODE {self.mode!r}")

    # -- mode-dependent limits -------------------------------------------------
    @property
    def mode_cfg(self) -> dict[str, Any]:
        return self.raw["modes"][self.mode]

    @property
    def max_jobs(self) -> int | None:
        env = os.environ.get("MAX_JOBS")
        if env:
            return None if env.lower() in {"none", "null", "all"} else int(env)
        return self.mode_cfg.get("max_jobs")

    # -- paths -----------------------------------------------------------------
    def path(self, key: str) -> Path:
        override = os.environ.get(f"LA_PATH_{key.upper()}")
        p = Path(override) if override else PROJECT_ROOT / self.raw["paths"][key]
        return p

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_model_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable; empty strings count as unset."""
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def has_env(*names: str) -> bool:
    return all(env(n) for n in names)
