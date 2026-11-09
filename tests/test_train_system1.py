"""Tests for scripts/train_system1.py using synthetic embeddings (no model download)."""
import json
import sys
import zlib
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_system1 as ts  # noqa: E402


def test_ece_perfect_and_overconfident():
    conf = np.array([1.0] * 10)
    assert ts.expected_calibration_error(conf, np.ones(10)) == pytest.approx(0.0)
    # always 100% confident but only half right -> ECE 0.5
    assert ts.expected_calibration_error(conf, np.array([1.0, 0.0] * 5)) == pytest.approx(0.5)


def test_choose_threshold_picks_lowest_meeting_target():
    conf = np.array([0.99, 0.95, 0.90, 0.80, 0.70, 0.60])
    correct = np.array([1, 1, 1, 1, 0, 0], dtype=float)
    # accuracy at t=0.80 is 4/4; at 0.70 is 4/5=0.8; target 0.95 -> tau 0.80
    assert ts.choose_threshold(conf, correct, 0.95) == pytest.approx(0.80)
    # target 0.8 allows going down to 0.70
    assert ts.choose_threshold(conf, correct, 0.80) == pytest.approx(0.70)


def test_choose_threshold_handles_ties():
    conf = np.array([0.9, 0.9, 0.9, 0.5])
    correct = np.array([1, 1, 0, 1], dtype=float)
    # accepting any 0.9 accepts all three (acc 2/3); target 0.9 is unreachable except
    # at no threshold -> error
    with pytest.raises(ValueError):
        ts.choose_threshold(conf, correct, 0.95)


def test_choose_threshold_unreachable_raises():
    with pytest.raises(ValueError):
        ts.choose_threshold(np.array([0.9, 0.8]), np.array([0.0, 0.0]), 0.5)


def _fake_embed(texts):
    """Deterministic class-clustered embeddings: text looks like 'intentK sample'."""
    out = []
    for t in texts:
        k = int(t.split()[0].replace("intent", ""))
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        center = np.zeros(16)
        center[k] = 1.0
        v = center + rng.normal(scale=0.35, size=16)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


def _write(path: Path, n_per: int, tag: str):
    rows = []
    for k in range(4):
        for i in range(n_per):
            rows.append(
                {
                    "id": f"{tag}-{k}-{i}",
                    "text": f"intent{k} sample {tag} {i}",
                    "category": "CAT_A" if k < 2 else "CAT_B",
                    "intent": f"intent_{k}",
                    "reference_reply": "ok",
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_end_to_end_saves_model_with_tau_and_plots(tmp_path):
    _write(tmp_path / "train.jsonl", 60, "tr")
    _write(tmp_path / "dev.jsonl", 40, "dv")
    out = tmp_path / "models" / "m.joblib"
    ts.main(
        [
            "--train", str(tmp_path / "train.jsonl"),
            "--dev", str(tmp_path / "dev.jsonl"),
            "--out", str(out),
            "--plots-dir", str(tmp_path / "plots"),
            "--target-accuracy", "0.9",
            "--cv", "3",
        ],
        embed_fn=_fake_embed,
    )
    art = joblib.load(out)
    meta = art["metadata"]
    assert 0.0 < meta["tau"] <= 1.0
    assert meta["metrics"]["dev_accuracy_at_tau"] >= 0.9
    assert meta["intent_to_category"]["intent_0"] == "CAT_A"
    assert art["train_embeddings"].shape == (240, 16)
    assert (tmp_path / "plots" / "system1_reliability.png").exists()
    assert (tmp_path / "plots" / "system1_coverage.png").exists()
    # classifier is usable and returns calibrated probabilities that sum to 1
    proba = art["classifier"].predict_proba(_fake_embed(["intent1 sample x 0"]))
    assert proba.sum() == pytest.approx(1.0)