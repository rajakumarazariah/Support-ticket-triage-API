"""Tests for app/router.py."""
import pytest

from app.config import RouterSettings, Settings
from app.router import Reason, TriageRouter
from app.schemas import Category, HandledBy, Intent
from app.system1 import System1, System1Prediction


def pred(conf, sim, intent=Intent.TRACK_ORDER):
    return System1Prediction(intent, Category.ORDER, conf, sim, intent)


class StubS1:
    def __init__(self, prediction=None, tau=0.9, error=None):
        self.prediction, self.tau, self.error, self.calls = prediction, tau, error, 0

    def predict(self, text):
        self.calls += 1
        if self.error:
            raise self.error
        return self.prediction


def make(prediction=None, tau=0.9, min_sim=0.8, **kw):
    s1 = StubS1(prediction, tau, kw.pop("error", None))
    return TriageRouter(s1, min_similarity=min_sim, **kw), s1


def test_confident_and_similar_goes_to_system1():
    r, _ = make(pred(0.97, 0.95))
    d = r.route("where is my order")
    assert d.route is HandledBy.SYSTEM1 and d.reason is Reason.HIGH_CONFIDENCE
    assert d.prediction.intent is Intent.TRACK_ORDER
    assert d.tau == 0.9 and d.min_similarity == 0.8


@pytest.mark.parametrize(
    "conf,sim,reason",
    [
        (0.50, 0.95, Reason.LOW_CONFIDENCE),
        (0.97, 0.50, Reason.LOW_SIMILARITY),
        (0.50, 0.50, Reason.LOW_CONFIDENCE_AND_SIMILARITY),
    ],
)
def test_below_threshold_goes_to_system2_with_reason(conf, sim, reason):
    d = make(pred(conf, sim))[0].route("x")
    assert d.route is HandledBy.SYSTEM2 and d.reason is reason
    assert d.prediction is not None  # kept for logging


def test_thresholds_are_inclusive():
    assert make(pred(0.9, 0.8))[0].route("x").route is HandledBy.SYSTEM1
    assert make(pred(0.8999, 0.8))[0].route("x").route is HandledBy.SYSTEM2
    assert make(pred(0.9, 0.7999))[0].route("x").route is HandledBy.SYSTEM2


def test_nan_confidence_is_never_system1():
    assert make(pred(float("nan"), 0.99))[0].route("x").route is HandledBy.SYSTEM2


def test_risk_flags_force_system2_without_running_system1():
    r, s1 = make(pred(0.99, 0.99))
    d = r.route("I will sue you", risk_flags=["sue", "lawyer"])
    assert d.route is HandledBy.SYSTEM2 and d.reason is Reason.RISK_KEYWORD
    assert d.risk_flags == ("sue", "lawyer") and d.prediction is None
    assert s1.calls == 0


def test_disabled_unavailable_and_error_degrade_to_system2():
    r, s1 = make(pred(0.99, 0.99), enabled=False)
    assert r.route("x").reason is Reason.SYSTEM1_DISABLED and s1.calls == 0
    r = TriageRouter(None, min_similarity=0.8)
    assert r.route("x").reason is Reason.SYSTEM1_UNAVAILABLE and r.tau is None
    r, _ = make(error=RuntimeError("boom"))
    d = r.route("x")
    assert d.route is HandledBy.SYSTEM2 and d.reason is Reason.SYSTEM1_ERROR


def test_tau_override_beats_artifact_tau():
    r, _ = make(pred(0.85, 0.95), tau=0.9)
    assert r.route("x").route is HandledBy.SYSTEM2
    r2 = TriageRouter(StubS1(pred(0.85, 0.95), tau=0.9), min_similarity=0.8, tau=0.8)
    assert r2.tau == 0.8 and r2.route("x").route is HandledBy.SYSTEM1


def test_invalid_router_config():
    with pytest.raises(ValueError):
        TriageRouter(StubS1(), min_similarity=1.5)
    with pytest.raises(ValueError):
        TriageRouter(StubS1(), min_similarity=0.8, tau=0)


def test_as_dict_has_no_ticket_text_and_is_flat():
    d = make(pred(0.97, 0.95))[0].route("secret ticket text").as_dict()
    assert d["route"] == "system1" and d["reason"] == "high_confidence"
    assert d["system1_intent"] == "track_order"
    assert "secret ticket text" not in str(d)
    assert make(None)[0].route("x", ["sue"]).as_dict()["confidence"] is None


async def test_aroute_matches_route():
    r, _ = make(pred(0.97, 0.95))
    assert (await r.aroute("x")).route is HandledBy.SYSTEM1


# ---- with the real System1 class and artifact --------------------------------
def test_end_to_end_with_real_system1(system1_model):
    path, embed = system1_model
    r = TriageRouter(System1.load(path, embed_fn=embed), min_similarity=0.7)
    assert r.route("get_refund fresh text").route is HandledBy.SYSTEM1
    d = r.route("ood weird text")
    assert d.route is HandledBy.SYSTEM2 and "similarity" in d.reason.value


def test_from_settings(system1_model, tmp_path):
    path, embed = system1_model
    s = Settings(router=RouterSettings(model_path=path, min_similarity=0.7, tau_override=0.6))
    r = TriageRouter.from_settings(s, embed_fn=embed)
    assert r.tau == 0.6 and r.route("track_order hi").route is HandledBy.SYSTEM1

    off = TriageRouter.from_settings(Settings(router=RouterSettings(system1_enabled=False)))
    assert off.route("x").reason is Reason.SYSTEM1_DISABLED

    missing = Settings(router=RouterSettings(model_path=tmp_path / "nope.joblib"))
    with pytest.raises(FileNotFoundError):  # fail fast, don't silently hit System 2
        TriageRouter.from_settings(missing, embed_fn=embed)