"""LLM gateway: one place that talks to model providers.

Aliases (`primary`, `strong`, `small`) map to LiteLLM model ids from settings, so
code and prompts never hard-code a model name. Each alias may also have a fallback
deployment on a second provider, registered in the Router as `<alias>-fallback`.

Resilience policy:
  * Every call has a timeout (per alias).
  * Retry ONLY on 429, 5xx/529, timeouts and connection errors; never on 4xx.
  * Exponential backoff with jitter; a Retry-After header is honoured, and if it is
    longer than `max_retry_after_s` we stop waiting and go to the fallback instead.
  * After retries are exhausted (or the breaker is open) the fallback deployment is
    tried once through the same policy.
  * A per-deployment circuit breaker opens after N consecutive FAILED CALLS (a call
    that exhausted its retries), then lets a single probe through after a cooldown.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, Callable, Mapping, Sequence

from app.config import ALIASES, GatewaySettings
from app.schemas import ErrorCode

log = logging.getLogger(__name__)

FALLBACK_SUFFIX = "-fallback"
TIMEOUT_GRACE_S = 1.0  # hard asyncio cap = provider timeout + grace
RESERVED_PARAMS = frozenset({"model", "messages", "timeout", "num_retries", "stream"})

_TIMEOUT_NAMES = frozenset(
    {"Timeout", "APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout", "WriteTimeout"}
)
_CONNECTION_NAMES = frozenset(
    {"APIConnectionError", "ConnectError", "ConnectionError", "RemoteProtocolError", "ReadError"}
)


# ------------------------------------------------------------------------- errors
class GatewayConfigError(ValueError):
    """Raised at startup when gateway settings are incomplete."""


class GatewayError(Exception):
    """A failed LLM call, already mapped to an API error code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        retry_after_s: float | None = None,
        status_code: int | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.status_code = status_code
        self.attempts = attempts


@dataclass(frozen=True)
class Failure:
    """Classification of an exception raised by a provider call."""

    code: ErrorCode
    retryable: bool
    retry_after_s: float | None = None
    status_code: int | None = None


def parse_retry_after(value: Any) -> float | None:
    """Parse a Retry-After header (seconds or HTTP-date) into seconds, or None."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(text).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _retry_after_from(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if not headers:
        return None

    try:
        if isinstance(headers, Mapping):
            for k, v in headers.items():
                if str(k).lower() == "retry-after":
                    return parse_retry_after(v)
        elif hasattr(headers, "items"):
            for k, v in headers.items():
                if str(k).lower() == "retry-after":
                    return parse_retry_after(v)
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def classify_exception(exc: BaseException) -> Failure:
    """Decide whether an exception is retryable and which error code it maps to."""
    names = {c.__name__ for c in type(exc).__mro__}
    raw_status = getattr(exc, "status_code", None)
    status = raw_status if isinstance(raw_status, int) else None
    retry_after = _retry_after_from(exc)

    def fail(code: ErrorCode, retryable: bool) -> Failure:
        return Failure(code, retryable, retry_after if retryable else None, status)

    if names & _TIMEOUT_NAMES or status in (408, 504):
        return fail(ErrorCode.TIMEOUT, True)
    if status == 429 or "RateLimitError" in names:
        return fail(ErrorCode.RATE_LIMIT, True)
    if status is not None and status >= 500:
        return fail(ErrorCode.PROVIDER_ERROR, True)
    if names & _CONNECTION_NAMES:
        return fail(ErrorCode.PROVIDER_ERROR, True)
    return fail(ErrorCode.PROVIDER_ERROR, False)


# --------------------------------------------------------------- circuit breaker
class BreakerState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Open after `threshold` consecutive failures; probe once after `cooldown_s`."""

    def __init__(
        self, threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_started_at: float | None = None

    @property
    def state(self) -> BreakerState:
        """Current state."""
        return self._state

    def allow(self) -> bool:
        """Return True if a call may proceed (and reserve the probe slot if half-open)."""
        now = self._clock()
        if self._state is BreakerState.CLOSED:
            return True
        if self._state is BreakerState.OPEN:
            if now - self._opened_at < self._cooldown:
                return False
            self._state = BreakerState.HALF_OPEN
            self._probe_started_at = now
            return True

        # HALF_OPEN: Allow one probe at a time.
        if self._probe_started_at is not None and now - self._probe_started_at < self._cooldown:
            return False

        self._probe_started_at = now
        return True

    def record_success(self) -> None:
        """A call succeeded (or the provider answered with a non-retryable error)."""
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._probe_started_at = None

    def record_failure(self) -> None:
        """A call failed after exhausting its retries."""
        now = self._clock()
        # If state is HALF_OPEN or probe window expired, re-open breaker directly.
        if self._state is BreakerState.HALF_OPEN or (
            self._probe_started_at is not None and now - self._probe_started_at >= self._cooldown
        ):
            self._open()
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._open()

    def _open(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._probe_started_at = None


# ------------------------------------------------------------------------ results
@dataclass(frozen=True)
class LLMResult:
    """Successful completion plus metadata for metrics/logging."""

    text: str
    alias: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float | None
    fallback_used: bool
    attempts: int


def _litellm_cost(response: Any) -> float | None:
    """Cost of a response via LiteLLM, or None if unknown."""
    try:
        hidden = getattr(response, "_hidden_params", None) or {}
        if hidden.get("response_cost") is not None:
            return float(hidden["response_cost"])
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        return None


def build_router(settings: GatewaySettings) -> Any:
    """Create the LiteLLM Router with one entry per alias and per fallback."""
    from litellm import Router

    model_list = [
        {"model_name": alias, "litellm_params": {"model": model_id}}
        for alias, model_id in settings.models.items()
    ] + [
        {"model_name": alias + FALLBACK_SUFFIX, "litellm_params": {"model": model_id}}
        for alias, model_id in settings.fallback_models.items()
    ]
    return Router(model_list=model_list, num_retries=0, disable_cooldowns=True)


# ------------------------------------------------------------------------ gateway
class Gateway:
    """Resilient async LLM client."""

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        router: Any | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        cost_fn: Callable[[Any], float | None] = _litellm_cost,
    ) -> None:
        missing = [a for a in ALIASES if a not in settings.models]
        if missing:
            raise GatewayConfigError(
                "Missing model ids for aliases "
                f"{missing}; set " + ", ".join(f"MODEL_{a.upper()}" for a in missing)
            )
        self._s = settings
        self._router = router if router is not None else build_router(settings)
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._cost_fn = cost_fn

        self._model_ids: dict[str, str] = dict(settings.models)
        self._model_ids.update(
            {alias + FALLBACK_SUFFIX: mid for alias, mid in settings.fallback_models.items()}
        )
        self._breakers = {
            key: CircuitBreaker(settings.breaker_failures, settings.breaker_cooldown_s, clock)
            for key in self._model_ids
        }

    def breaker_states(self) -> dict[str, str]:
        """Breaker state per deployment."""
        return {key: br.state.value for key, br in self._breakers.items()}

    async def complete(
        self,
        alias: str,
        messages: Sequence[Mapping[str, str]],
        *,
        params: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> LLMResult:
        if alias not in self._s.models:
            raise ValueError(f"Unknown model alias '{alias}'")
        params = dict(params or {})
        clash = RESERVED_PARAMS & params.keys()
        if clash:
            raise ValueError(f"params may not set {sorted(clash)}")

        start = self._clock()
        deployments = [(alias, False)]
        if alias in self._s.fallback_models:
            deployments.append((alias + FALLBACK_SUFFIX, True))

        real_failure: GatewayError | None = None
        open_failure: GatewayError | None = None
        total_attempts = 0

        for key, is_fallback in deployments:
            breaker = self._breakers[key]
            if not breaker.allow():
                log.warning(
                    "circuit open, skipping deployment",
                    extra={"deployment": key, "request_id": request_id},
                )
                open_failure = GatewayError(
                    ErrorCode.PROVIDER_ERROR, f"circuit open for {key}", retryable=True
                )
                continue
            max_attempts = (
                1 if breaker.state is BreakerState.HALF_OPEN else 1 + self._s.max_retries
            )
            try:
                response, attempts = await self._attempt_loop(
                    key, alias, messages, params, max_attempts, request_id
                )
            except GatewayError as exc:
                total_attempts += exc.attempts
                if exc.retryable:
                    breaker.record_failure()
                    real_failure = exc
                    continue
                breaker.record_success()
                exc.attempts = total_attempts
                raise
            total_attempts += attempts
            breaker.record_success()
            return self._build_result(
                response, alias, key, is_fallback, total_attempts, self._clock() - start
            )

        final = real_failure or open_failure
        assert final is not None
        final.attempts = total_attempts
        raise final

    async def _attempt_loop(
        self,
        key: str,
        alias: str,
        messages: Sequence[Mapping[str, str]],
        params: dict[str, Any],
        max_attempts: int,
        request_id: str | None,
    ) -> tuple[Any, int]:
        timeout = self._s.timeout_s[alias]
        for attempt in range(1, max_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self._router.acompletion(
                        model=key, messages=list(messages), timeout=timeout, **params
                    ),
                    timeout=timeout + TIMEOUT_GRACE_S,
                )
                return response, attempt
            except Exception as exc:
                failure = classify_exception(exc)
                description = f"{type(exc).__name__} (status {failure.status_code})"
                error = GatewayError(
                    failure.code,
                    f"{key}: {description}",
                    retryable=failure.retryable,
                    retry_after_s=failure.retry_after_s,
                    status_code=failure.status_code,
                    attempts=attempt,
                )
                if not failure.retryable:
                    log.error(
                        "non-retryable provider error",
                        extra={"deployment": key, "request_id": request_id},
                        exc_info=True,
                    )
                    raise error from exc
                delay = self._delay(attempt, failure.retry_after_s)
                if attempt == max_attempts or delay is None:
                    raise error from exc
                log.warning(
                    "retrying provider call",
                    extra={
                        "deployment": key,
                        "attempt": attempt,
                        "delay_s": round(delay, 3),
                        "code": failure.code.value,
                        "request_id": request_id,
                    },
                )
                await self._sleep(delay)
        raise AssertionError("unreachable")

    def _delay(self, attempt: int, retry_after_s: float | None) -> float | None:
        backoff = min(self._s.backoff_cap_s, self._s.backoff_base_s * 2 ** (attempt - 1))
        delay = backoff / 2 + self._rng.uniform(0, backoff / 2)
        if retry_after_s is not None:
            if retry_after_s > self._s.max_retry_after_s:
                return None
            return max(delay, retry_after_s)
        return delay

    def _build_result(
        self,
        response: Any,
        alias: str,
        key: str,
        is_fallback: bool,
        attempts: int,
        latency_s: float,
    ) -> LLMResult:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR, f"{key}: malformed provider response", attempts=attempts
            ) from exc
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=content or "",
            alias=alias,
            model=self._model_ids[key],
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=latency_s,
            cost_usd=self._cost_fn(response),
            fallback_used=is_fallback,
            attempts=attempts,
        )