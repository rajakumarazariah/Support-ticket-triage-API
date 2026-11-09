"""Tests for app/gateway.py. Uses a scripted fake router (no network)."""
import asyncio
import time
from email.utils import formatdate
from types import SimpleNamespace

import pytest

import app.gateway as gw
from app.config import GatewaySettings
from app.gateway import (
    BreakerState,
    CircuitBreaker,
    Gateway,
    GatewayConfigError,
    GatewayError,
    classify_exception,
    parse_retry_after,
)
from app.schemas import ErrorCode

MODELS = {"primary": "prov-a/primary-m", "strong": "prov-a/strong-m", "small": "prov-a/small-m"}
FALLBACKS = {"primary": "prov-b/primary-m", "strong": "prov-b/strong-m", "small": "prov-b/small-m"}


# ---- helpers ----------------------------------------------------------------
def make_exc(name="Exc", status=None, headers=None):
    exc = type(name, (Exception,), {})("boom with secret prompt text")
    if status is not None:
        exc.status_code = status
    if headers is not None:
        exc.response = SimpleNamespace(headers=headers)
    return exc


def ok(text="hello", pt=10, ct=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


class FakeRouter:
    """Returns scripted outcomes per deployment key; records every call."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    async def acompletion(self, model, messages, timeout, **kw):
        self.calls.append((model, timeout, kw))
        outcome = self.script[model].pop(0)
        if callable(outcome):
            return await outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FixedRng:
    def uniform(self, a, b):
        return b  # deterministic: delay == full backoff


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_gateway(script, *, fallback=True, retries=2, threshold=5, cooldown=30.0, clock=None, **kw):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    settings = GatewaySettings(
        models=MODELS,
        fallback_models=FALLBACKS if fallback else {},
        timeout_s={"small": 5, "primary": 5, "strong": 5},
        max_retries=retries,
        backoff_base_s=0.5,
        backoff_cap_s=8.0,
        max_retry_after_s=10.0,
        breaker_failures=threshold,
        breaker_cooldown_s=cooldown,
    )
    router = FakeRouter(script)
    g = Gateway(settings, router=router, sleep=fake_sleep, rng=FixedRng(),
                clock=clock or time.monotonic, cost_fn=lambda r: 0.001, **kw)
    return g, router, sleeps


MSG = [{"role": "user", "content": "hi"}]


# ---- classification ---------------------------------------------------------
@pytest.mark.parametrize(
    "name,status,code,retryable",
    [
        ("RateLimitError", 429, ErrorCode.RATE_LIMIT, True),
        ("Exc", 429, ErrorCode.RATE_LIMIT, True),
        ("InternalServerError", 500, ErrorCode.PROVIDER_ERROR, True),
        ("ServiceUnavailableError", 503, ErrorCode.PROVIDER_ERROR, True),
        ("Exc", 529, ErrorCode.PROVIDER_ERROR, True),
        ("Timeout", 408, ErrorCode.TIMEOUT, True),
        ("APITimeoutError", None, ErrorCode.TIMEOUT, True),
        ("TimeoutError", None, ErrorCode.TIMEOUT, True),
        ("Exc", 504, ErrorCode.TIMEOUT, True),
        ("APIConnectionError", None, ErrorCode.PROVIDER_ERROR, True),
        ("BadRequestError", 400, ErrorCode.PROVIDER_ERROR, False),
        ("AuthenticationError", 401, ErrorCode.PROVIDER_ERROR, False),
        ("NotFoundError", 404, ErrorCode.PROVIDER_ERROR, False),
        ("KeyError", None, ErrorCode.PROVIDER_ERROR, False),
    ],
)
def test_classify(name, status, code, retryable):
    f = classify_exception(make_exc(name, status))
    assert (f.code, f.retryable) == (code, retryable)


def test_asyncio_timeout_is_timeout():
    assert classify_exception(asyncio.TimeoutError()).code is ErrorCode.TIMEOUT


def test_retry_after_parsing():
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after(" 1.5 ") == 1.5
    assert 0 < parse_retry_after(formatdate(time.time() + 30, usegmt=True)) <= 30
    assert parse_retry_after("garbage") is None and parse_retry_after(None) is None
    f = classify_exception(make_exc("RateLimitError", 429, {"Retry-After": "3"}))
    assert f.retry_after_s == 3.0
    # not reported for non-retryable errors
    assert classify_exception(make_exc("BadRequestError", 400, {"retry-after": "3"})).retry_after_s is None


# ---- circuit breaker unit ---------------------------------------------------
def test_breaker_lifecycle():
    clock = Clock()
    br = CircuitBreaker(2, 30, clock)
    assert br.allow()
    br.record_failure()
    assert br.state is BreakerState.CLOSED
    br.record_failure()
    assert br.state is BreakerState.OPEN and not br.allow()
    clock.t += 29
    assert not br.allow()
    clock.t += 2
    assert br.allow() and br.state is BreakerState.HALF_OPEN
    assert not br.allow()  # only one probe in flight
    br.record_failure()  # failed probe re-opens immediately
    assert br.state is BreakerState.OPEN and not br.allow()
    clock.t += 31
    assert br.allow()
    br.record_success()
    assert br.state is BreakerState.CLOSED and br.allow()


def test_breaker_success_resets_consecutive_count_and_lost_probe_recovers():
    clock = Clock()
    br = CircuitBreaker(3, 10, clock)
    br.record_failure(); br.record_failure(); br.record_success(); br.record_failure()
    assert br.state is BreakerState.CLOSED  # not consecutive
    for _ in range(2):
        br.record_failure()
    assert br.state is BreakerState.OPEN
    clock.t += 11
    assert br.allow()  # probe starts and is "lost" (never reports)
    assert not br.allow()
    clock.t += 11
    assert br.allow()  # a new probe is allowed after another cooldown


# ---- happy path -------------------------------------------------------------
async def test_success_returns_text_usage_cost_latency():
    g, router, sleeps = make_gateway({"small": [ok("hi!", 12, 7)]})
    res = await g.complete("small", MSG, params={"temperature": 0, "max_tokens": 50})
    assert (res.text, res.input_tokens, res.output_tokens) == ("hi!", 12, 7)
    assert res.model == "prov-a/small-m" and res.alias == "small"
    assert res.cost_usd == 0.001 and res.fallback_used is False and res.attempts == 1
    assert res.latency_s >= 0 and sleeps == []
    model, timeout, kw = router.calls[0]
    assert model == "small" and timeout == 5 and kw == {"temperature": 0, "max_tokens": 50}


async def test_none_content_becomes_empty_string_and_missing_usage_is_zero():
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    g, _, _ = make_gateway({"small": [resp]})
    res = await g.complete("small", MSG)
    assert res.text == "" and res.input_tokens == 0 and res.output_tokens == 0


async def test_malformed_response_is_non_retryable_provider_error():
    g, _, _ = make_gateway({"small": [SimpleNamespace(choices=[])]})
    with pytest.raises(GatewayError) as ei:
        await g.complete("small", MSG)
    assert ei.value.code is ErrorCode.PROVIDER_ERROR and not ei.value.retryable


# ---- retries ----------------------------------------------------------------
async def test_retries_429_honouring_retry_after_then_succeeds():
    g, router, sleeps = make_gateway(
        {"primary": [make_exc("RateLimitError", 429, {"retry-after": "2"}), ok()]}
    )
    res = await g.complete("primary", MSG)
    assert res.attempts == 2 and not res.fallback_used
    assert sleeps == [2.0]  # retry-after (2s) beats the 0.5s backoff


async def test_backoff_grows_exponentially_and_is_capped():
    g, _, sleeps = make_gateway(
        {"primary": [make_exc("InternalServerError", 500)] * 6 + [ok()]},
        retries=6, fallback=False,
    )
    await g.complete("primary", MSG)
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]


async def test_timeout_is_retried():
    g, router, _ = make_gateway({"small": [asyncio.TimeoutError(), ok()]})
    res = await g.complete("small", MSG)
    assert res.attempts == 2


async def test_hanging_call_is_cut_off_by_timeout(monkeypatch):
    monkeypatch.setattr(gw, "TIMEOUT_GRACE_S", 0.0)

    async def hang():
        await asyncio.sleep(10)

    g, _, _ = make_gateway({"small": [hang]}, retries=0, fallback=False)
    g._s = GatewaySettings(**{**g._s.__dict__, "timeout_s": {"small": 0.05, "primary": 5, "strong": 5}})
    with pytest.raises(GatewayError) as ei:
        await g.complete("small", MSG)
    assert ei.value.code is ErrorCode.TIMEOUT and ei.value.retryable


async def test_400_is_never_retried_and_never_falls_back():
    g, router, sleeps = make_gateway({"primary": [make_exc("BadRequestError", 400)]})
    with pytest.raises(GatewayError) as ei:
        await g.complete("primary", MSG)
    assert ei.value.code is ErrorCode.PROVIDER_ERROR and not ei.value.retryable
    assert len(router.calls) == 1 and sleeps == []
    assert g.breaker_states()["primary"] == "closed"


async def test_error_message_does_not_leak_provider_text():
    g, _, _ = make_gateway({"primary": [make_exc("BadRequestError", 400)]})
    with pytest.raises(GatewayError) as ei:
        await g.complete("primary", MSG)
    assert "secret prompt text" not in ei.value.message


# ---- fallback ---------------------------------------------------------------
async def test_falls_back_after_retries_exhausted():
    err = make_exc("InternalServerError", 500)
    g, router, _ = make_gateway({"primary": [err, err, err], "primary-fallback": [ok("from b")]})
    res = await g.complete("primary", MSG)
    assert res.fallback_used and res.text == "from b" and res.model == "prov-b/primary-m"
    assert res.attempts == 4
    assert [c[0] for c in router.calls] == ["primary"] * 3 + ["primary-fallback"]


async def test_long_retry_after_skips_waiting_and_uses_fallback():
    g, router, sleeps = make_gateway(
        {"primary": [make_exc("RateLimitError", 429, {"retry-after": "120"})],
         "primary-fallback": [ok()]}
    )
    res = await g.complete("primary", MSG)
    assert res.fallback_used and sleeps == [] and len(router.calls) == 2


async def test_both_providers_down_raises_last_real_failure():
    e500 = make_exc("InternalServerError", 500)
    g, _, _ = make_gateway(
        {"primary": [e500] * 3, "primary-fallback": [make_exc("RateLimitError", 429)] * 3}
    )
    with pytest.raises(GatewayError) as ei:
        await g.complete("primary", MSG)
    assert ei.value.code is ErrorCode.RATE_LIMIT and ei.value.attempts == 6


async def test_no_fallback_configured_raises_after_retries():
    g, router, _ = make_gateway({"small": [make_exc("Timeout", 408)] * 3}, fallback=False)
    with pytest.raises(GatewayError) as ei:
        await g.complete("small", MSG)
    assert ei.value.code is ErrorCode.TIMEOUT and len(router.calls) == 3


# ---- breaker integration ----------------------------------------------------
async def test_breaker_opens_then_half_open_probe_single_attempt_then_closes():
    clock = Clock()
    e500 = make_exc("InternalServerError", 500)
    script = {"small": [e500, e500, e500, e500, e500, e500, e500, ok("recovered")]}
    g, router, _ = make_gateway(script, fallback=False, retries=2, threshold=2, cooldown=30, clock=clock)

    for _ in range(2):  # two failed calls (3 attempts each) open the breaker
        with pytest.raises(GatewayError):
            await g.complete("small", MSG)
    assert g.breaker_states()["small"] == "open" and len(router.calls) == 6

    with pytest.raises(GatewayError) as ei:  # fails fast, router untouched
        await g.complete("small", MSG)
    assert "circuit open" in ei.value.message and len(router.calls) == 6

    clock.t += 31  # half-open: exactly ONE attempt even though retries=2
    with pytest.raises(GatewayError):
        await g.complete("small", MSG)
    assert len(router.calls) == 7 and g.breaker_states()["small"] == "open"

    clock.t += 31
    res = await g.complete("small", MSG)
    assert res.text == "recovered" and g.breaker_states()["small"] == "closed"


async def test_open_primary_breaker_routes_straight_to_fallback():
    clock = Clock()
    e500 = make_exc("InternalServerError", 500)
    script = {"primary": [e500] * 3, "primary-fallback": [ok("b1"), ok("b2")]}
    g, router, _ = make_gateway(script, retries=2, threshold=1, clock=clock)
    first = await g.complete("primary", MSG)  # primary fails -> breaker opens -> fallback
    second = await g.complete("primary", MSG)  # primary skipped entirely
    assert first.fallback_used and second.fallback_used
    assert [c[0] for c in router.calls].count("primary") == 3


# ---- validation -------------------------------------------------------------
async def test_unknown_alias_and_reserved_params():
    g, _, _ = make_gateway({})
    with pytest.raises(ValueError, match="Unknown model alias"):
        await g.complete("nope", MSG)
    with pytest.raises(ValueError, match="may not set"):
        await g.complete("small", MSG, params={"timeout": 1})


def test_missing_model_config_fails_at_startup():
    with pytest.raises(GatewayConfigError, match="MODEL_STRONG"):
        Gateway(GatewaySettings(models={"primary": "a", "small": "b"}), router=object())


# ---- real LiteLLM Router wiring (offline via mock_response) -----------------
async def test_real_router_alias_wiring_with_mock_response():
    pytest.importorskip("litellm")
    settings = GatewaySettings(
        models={"primary": "openai/gpt-x", "strong": "openai/gpt-y", "small": "openai/gpt-z"},
        fallback_models={"primary": "openai/gpt-w"},
    )
    g = Gateway(settings)
    res = await g.complete("small", MSG, params={"mock_response": "mocked!", "max_tokens": 20})
    assert res.text == "mocked!" and res.model == "openai/gpt-z" and not res.fallback_used