"""Application settings, read from environment variables.

`Settings.from_env()` first loads a `.env` file (via python-dotenv) without
overriding variables that are already set, so real environment variables, such as
those injected by Cloud Run, always win over the file. Provider API keys
(OPENAI_API_KEY, ANTHROPIC_API_KEY, ...) are read by LiteLLM straight from the
environment, so loading `.env` first is what makes them available.

Environment variables:
    PROMPTS_DIR            directory holding the prompt YAML files
    ACTIVE_PROMPTS         JSON object: role -> prompt id (merged over defaults)
    PROMPT_CANARY          JSON object: role -> {prompt id: weight}

    MODEL_PRIMARY | MODEL_STRONG | MODEL_SMALL
                           LiteLLM model id for each alias (required by the gateway)
    MODEL_<ALIAS>_FALLBACK optional model id on a second provider
    TIMEOUT_<ALIAS>_S      per-call timeout in seconds (defaults below)
    LLM_MAX_RETRIES        retries per deployment after the first attempt (default 2)
    BACKOFF_BASE_S         first backoff delay (default 0.5)
    BACKOFF_CAP_S          maximum backoff delay (default 8)
    MAX_RETRY_AFTER_S      longest Retry-After we are willing to wait (default 10)
    BREAKER_FAILURES       consecutive failed calls before the breaker opens (default 5)
    BREAKER_COOLDOWN_S     seconds before a half-open probe is allowed (default 30)

    SYSTEM1_ENABLED        true/false: use the fast path at all (default true)
    SYSTEM1_MODEL_PATH     joblib artifact from scripts/train_system1.py
    SYSTEM1_TAU            override the confidence threshold stored in the artifact
    SYSTEM1_MIN_SIM        minimum nearest-neighbour cosine similarity (default 0.80)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, TypeVar

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOTENV = REPO_ROOT / ".env"

ALIASES = ("primary", "strong", "small")

DEFAULT_ACTIVE_PROMPTS: dict[str, str] = {
    "triage": "triage_v2",  # single-call baseline (no System 1); used in eval comparisons
    "system2": "system2_v1",  # strong model, escalated tickets
    "fastpath": "fastpath_v1",  # small model, System 1 already decided the intent
    "repair": "repair_v1",
}

DEFAULT_TIMEOUTS_S: dict[str, float] = {"small": 10.0, "primary": 20.0, "strong": 30.0}

T = TypeVar("T", int, float)


def _json_env(env: Mapping[str, str], name: str) -> dict:
    raw = env.get(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _num_env(env: Mapping[str, str], name: str, default: T, cast: Callable[[str], T]) -> T:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class RouterSettings:
    """System 1 / System 2 routing settings."""

    system1_enabled: bool = True
    model_path: Path = REPO_ROOT / "models" / "system1_v1.joblib"
    tau_override: float | None = None  # None -> use the tau saved in the model artifact
    min_similarity: float = 0.80

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "RouterSettings":
        """Read routing settings from `env`."""
        tau: float | None = None
        if (env.get("SYSTEM1_TAU") or "").strip():
            tau = _num_env(env, "SYSTEM1_TAU", 0.0, float)
            if not 0.0 < tau <= 1.0:
                raise ValueError(f"SYSTEM1_TAU must be in (0, 1], got {tau}")
        min_sim = _num_env(env, "SYSTEM1_MIN_SIM", 0.80, float)
        if min_sim > 1.0:
            raise ValueError(f"SYSTEM1_MIN_SIM must be <= 1, got {min_sim}")
        return cls(
            system1_enabled=_bool_env(env, "SYSTEM1_ENABLED", True),
            model_path=Path(env.get("SYSTEM1_MODEL_PATH") or REPO_ROOT / "models" / "system1_v1.joblib"),
            tau_override=tau,
            min_similarity=min_sim,
        )


@dataclass(frozen=True)
class GatewaySettings:
    """LLM gateway settings: model ids, timeouts, retry policy, circuit breaker."""

    models: Mapping[str, str] = field(default_factory=dict)  # alias -> LiteLLM model id
    fallback_models: Mapping[str, str] = field(default_factory=dict)  # alias -> 2nd provider id
    timeout_s: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS_S))
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0
    max_retry_after_s: float = 10.0
    breaker_failures: int = 5
    breaker_cooldown_s: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "GatewaySettings":
        """Read gateway settings from `env`."""
        models, fallbacks, timeouts = {}, {}, dict(DEFAULT_TIMEOUTS_S)
        for alias in ALIASES:
            name = alias.upper()
            if env.get(f"MODEL_{name}"):
                models[alias] = env[f"MODEL_{name}"].strip()
            if env.get(f"MODEL_{name}_FALLBACK"):
                fallbacks[alias] = env[f"MODEL_{name}_FALLBACK"].strip()
            timeouts[alias] = _num_env(env, f"TIMEOUT_{name}_S", timeouts[alias], float)
        return cls(
            models=models,
            fallback_models=fallbacks,
            timeout_s=timeouts,
            max_retries=_num_env(env, "LLM_MAX_RETRIES", 2, int),
            backoff_base_s=_num_env(env, "BACKOFF_BASE_S", 0.5, float),
            backoff_cap_s=_num_env(env, "BACKOFF_CAP_S", 8.0, float),
            max_retry_after_s=_num_env(env, "MAX_RETRY_AFTER_S", 10.0, float),
            breaker_failures=max(1, _num_env(env, "BREAKER_FAILURES", 5, int)),
            breaker_cooldown_s=_num_env(env, "BREAKER_COOLDOWN_S", 30.0, float),
        )


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    prompts_dir: Path = REPO_ROOT / "prompts" / "triage"
    active_prompts: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ACTIVE_PROMPTS))
    prompt_canary: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    gateway: GatewaySettings = field(default_factory=GatewaySettings)
    router: RouterSettings = field(default_factory=RouterSettings)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, dotenv_path: Path | None = None
    ) -> "Settings":
        """Build settings from the environment.

        When `env` is None the real process environment is used, after loading
        `.env` (or `dotenv_path`) with `override=False`. Passing an explicit `env`
        mapping skips dotenv entirely, which keeps tests hermetic.
        """
        if env is None:
            load_dotenv(dotenv_path or DEFAULT_DOTENV, override=False)
            env = os.environ
        return cls(
            prompts_dir=Path(env.get("PROMPTS_DIR", str(REPO_ROOT / "prompts" / "triage"))),
            active_prompts={**DEFAULT_ACTIVE_PROMPTS, **_json_env(env, "ACTIVE_PROMPTS")},
            prompt_canary=_json_env(env, "PROMPT_CANARY"),
            gateway=GatewaySettings.from_env(env),
            router=RouterSettings.from_env(env),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings.from_env()