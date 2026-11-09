"""Tests for app/system1.py."""
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from app.schemas import INTENT_TO_CATEGORY, Category, Intent
from app.system1 import System1, System1Error


@pytest.fixture(scope="module")
def s1(system1_model):
    path, embed = system1_model
    return System1.load(path, embed_fn=embed)


def test_predict_in_distribution(s1):
    for intent in ("track_order", "get_refund", "cancel_order", "payment_issue"):
        p = s1.predict(f"{intent} brand new unseen text")
        assert p.intent is Intent(intent)
        assert p.category is INTENT_TO_CATEGORY[Intent(intent)]
        assert isinstance(p.category, Category)
        assert 0.5 < p.confidence <= 1.0
        assert p.similarity > 0.7
        assert p.nn_intent is Intent(intent)


def test_out_of_distribution_has_low_similarity(s1):
    p = s1.predict("ood something totally different")
    in_dist = s1.predict("track_order brand new unseen text")
    assert p.similarity < 0.55 < 0.7 < in_dist.similarity  # clear gap for a min_sim between
    assert 0.0 <= p.confidence <= 1.0


def test_tau_and_metadata_loaded(s1):
    assert 0.0 < s1.tau <= 1.0
    assert s1.metadata["embedding_dim"] == 16
    assert s1._train.dtype == np.float32  # float16 on disk, float32 in memory


def test_load_runs_one_warmup_prediction(system1_model):
    path, embed = system1_model
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return embed(texts)

    System1.load(path, embed_fn=counting)
    assert calls == [["warm up"]]
    calls.clear()
    System1.load(path, embed_fn=counting, warmup=False)
    assert calls == []


def test_missing_file_and_bad_artifacts(tmp_path, system1_model):
    with pytest.raises(FileNotFoundError):
        System1.load(tmp_path / "nope.joblib", embed_fn=lambda t: None)
    path, embed = system1_model
    art = joblib.load(path)
    broken = {k: v for k, v in art.items() if k != "train_labels"}
    joblib.dump(broken, tmp_path / "a.joblib")
    with pytest.raises(System1Error, match="train_labels"):
        System1.load(tmp_path / "a.joblib", embed_fn=embed)
    art["metadata"].pop("tau")
    joblib.dump(art, tmp_path / "b.joblib")
    with pytest.raises(System1Error, match="tau"):
        System1.load(tmp_path / "b.joblib", embed_fn=embed)


def _parts(system1_model):
    path, embed = system1_model
    art = joblib.load(path)
    return art, embed


def _build(art, embed, **meta_over):
    meta = {**art["metadata"], **meta_over}
    return System1(art["classifier"], art["train_embeddings"], art["train_labels"], meta, embed)


@pytest.mark.parametrize("tau", [0, 0.0, 1.5, -0.2, "x"])
def test_invalid_tau_rejected(system1_model, tau):
    art, embed = _parts(system1_model)
    with pytest.raises(System1Error, match="tau"):
        _build(art, embed, tau=tau)


def test_category_mismatch_with_schemas_rejected(system1_model):
    art, embed = _parts(system1_model)
    bad = {**art["metadata"]["intent_to_category"], "track_order": "REFUND"}
    with pytest.raises(System1Error, match="category mismatch"):
        _build(art, embed, intent_to_category=bad)


def test_unknown_class_rejected(system1_model):
    art, embed = _parts(system1_model)
    clf = SimpleNamespace(classes_=np.array(["track_order", "make_coffee"]))
    with pytest.raises(System1Error, match="make_coffee"):
        System1(clf, art["train_embeddings"], art["train_labels"], art["metadata"], embed)


def test_shape_mismatches_rejected(system1_model):
    art, embed = _parts(system1_model)
    with pytest.raises(System1Error, match="dim"):
        _build(art, embed, embedding_dim=384)
    with pytest.raises(System1Error, match="different lengths"):
        System1(art["classifier"], art["train_embeddings"], art["train_labels"][:-1],
                art["metadata"], embed)


def test_wrong_encoder_output_shape_raises(system1_model):
    art, _ = _parts(system1_model)
    s = _build(art, lambda t: np.zeros((1, 8), dtype=np.float32))
    with pytest.raises(System1Error, match="encoder returned"):
        s.predict("x")


def test_non_normalised_encoder_output_is_normalised(system1_model):
    art, embed = _parts(system1_model)
    s = _build(art, lambda t: embed(t) * 7.0)
    p = s.predict("track_order text")
    assert p.intent is Intent.TRACK_ORDER and p.similarity <= 1.0