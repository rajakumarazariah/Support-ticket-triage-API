"""Route each ticket to System 1 (fast path) or System 2 (strong LLM).

Rules, in priority order:
  1. Guardrails flagged risk keywords      -> System 2 (System 1 isn't even run)
  2. System 1 disabled / not loaded        -> System 2
  3. System 1 raised an error              -> System 2 (degrade, don't fail the request)
  4. confidence >= tau AND similarity >= min_similarity -> System 1
  5. otherwise                             -> System 2

Every decision carries a machine-readable reason for metrics, traces and debugging.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Sequence

from app.schemas import HandledBy
from app.system1 import System1, System1Prediction

log = logging.getLogger(__name__)


class Reason(StrEnum):
    """Why a ticket went where it did."""

    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"
    LOW_SIMILARITY = "low_similarity"
    LOW_CONFIDENCE_AND_SIMILARITY = "low_confidence_and_similarity"
    RISK_KEYWORD = "risk_keyword"
    SYSTEM1_DISABLED = "system1_disabled"
    SYSTEM1_UNAVAILABLE = "system1_unavailable"
    SYSTEM1_ERROR = "system1_error"


@dataclass(frozen=True)
class RoutingDecision:
    """Outcome of routing one ticket."""

    route: HandledBy
    reason: Reason
    prediction: System1Prediction | None = None  # None when System 1 was skipped or failed
    tau: float | None = None
    min_similarity: float | None = None
    risk_flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for logs and span attributes (no ticket text)."""
        p = self.prediction
        return {
            "route": self.route.value,
            "reason": self.reason.value,
            "tau": self.tau,
            "min_similarity": self.min_similarity,
            "confidence": p.confidence if p else None,
            "similarity": p.similarity if p else None,
            "system1_intent": p.intent.value if p else None,
            "risk_flags": list(self.risk_flags),
        }


class TriageRouter:
    """Decides which system handles a ticket."""

    def __init__(
        self,
        system1: System1 | None,
        *,
        min_similarity: float,
        tau: float | None = None,
        enabled: bool = True,
    ) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [0, 1]")
        self._system1 = system1
        self._enabled = enabled
        self._min_similarity = min_similarity
        self._tau = tau if tau is not None else (system1.tau if system1 else None)
        if self._tau is not None and not 0.0 < self._tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")

    @classmethod
    def from_settings(cls, settings: Any, embed_fn: Any = None) -> "TriageRouter":
        """Build from `app.config.Settings`, loading System 1 once.

        If System 1 is enabled but its model file is missing, this RAISES: a silent
        fallback to System 2 would hide a bad deploy as a cost spike. Disable it
        explicitly with SYSTEM1_ENABLED=false to run without the fast path.
        """
        cfg = settings.router
        system1 = System1.load(cfg.model_path, embed_fn=embed_fn) if cfg.system1_enabled else None
        return cls(
            system1,
            min_similarity=cfg.min_similarity,
            tau=cfg.tau_override,
            enabled=cfg.system1_enabled,
        )

    @property
    def tau(self) -> float | None:
        """Active confidence threshold."""
        return self._tau

    def route(self, text: str, risk_flags: Sequence[str] = ()) -> RoutingDecision:
        """Route one ticket. `risk_flags` are the guardrails' matched risk keywords."""
        flags = tuple(risk_flags)
        base = {"tau": self._tau, "min_similarity": self._min_similarity, "risk_flags": flags}

        if flags:
            return RoutingDecision(HandledBy.SYSTEM2, Reason.RISK_KEYWORD, **base)
        if not self._enabled:
            return RoutingDecision(HandledBy.SYSTEM2, Reason.SYSTEM1_DISABLED, **base)
        if self._system1 is None:
            return RoutingDecision(HandledBy.SYSTEM2, Reason.SYSTEM1_UNAVAILABLE, **base)

        try:
            pred = self._system1.predict(text)
        except Exception:  # noqa: BLE001 - any System 1 failure must degrade, not 500
            log.exception("System 1 prediction failed; routing to System 2")
            return RoutingDecision(HandledBy.SYSTEM2, Reason.SYSTEM1_ERROR, **base)

        confident = pred.confidence >= self._tau  # type: ignore[operator]
        similar = pred.similarity >= self._min_similarity
        if confident and similar:
            return RoutingDecision(HandledBy.SYSTEM1, Reason.HIGH_CONFIDENCE, pred, **base)
        if not confident and not similar:
            reason = Reason.LOW_CONFIDENCE_AND_SIMILARITY
        elif not confident:
            reason = Reason.LOW_CONFIDENCE
        else:
            reason = Reason.LOW_SIMILARITY
        return RoutingDecision(HandledBy.SYSTEM2, reason, pred, **base)

    async def aroute(self, text: str, risk_flags: Sequence[str] = ()) -> RoutingDecision:
        """Async wrapper: runs the CPU-bound embedding in a worker thread."""
        return await asyncio.to_thread(self.route, text, risk_flags)