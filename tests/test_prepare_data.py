"""Tests for scripts/prepare_data.py using synthetic data (no network)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_data as pd_mod  # noqa: E402


def _make_df(n_per_intent: int = 20) -> pd.DataFrame:
    rows = []
    intents = {"cancel_order": "ORDER", "track_order": "ORDER", "get_refund": "REFUND"}
    for intent, cat in intents.items():
        for i in range(n_per_intent):
            rows.append(
                {
                    "instruction": f"{intent} question number {i}",
                    "category": cat,
                    "intent": intent,
                    "response": f"reply for {intent} {i}",
                }
            )
    return pd.DataFrame(rows)


def test_clean_drops_duplicates_case_and_whitespace_insensitive():
    df = _make_df(3)
    dup = df.iloc[[0]].copy()
    dup["instruction"] = "  CANCEL_ORDER   question number 0 "
    out = pd_mod.clean(pd.concat([df, dup], ignore_index=True))
    assert len(out) == len(df)


def test_clean_drops_all_copies_of_conflicting_instruction():
    df = _make_df(3)
    conflict = pd.DataFrame(
        [
            {"instruction": "where is my money", "category": "REFUND", "intent": "get_refund", "response": "a"},
            {"instruction": "where is my money", "category": "ORDER", "intent": "track_order", "response": "b"},
        ]
    )
    out = pd_mod.clean(pd.concat([df, conflict], ignore_index=True))
    assert "where is my money" not in set(out["instruction"])


def test_check_intent_category_ok_and_failure():
    df = _make_df(2)
    assert pd_mod.check_intent_category(df)["get_refund"] == "REFUND"
    df.loc[0, "category"] = "REFUND"  # cancel_order now in two categories
    with pytest.raises(ValueError):
        pd_mod.check_intent_category(df)


def test_splits_are_disjoint_stratified_and_deterministic():
    df = pd_mod.clean(_make_df(40))
    a = pd_mod.make_splits(df, seed=42)
    b = pd_mod.make_splits(df, seed=42)
    sizes = {k: len(v) for k, v in a.items()}
    assert sum(sizes.values()) == len(df)
    assert abs(sizes["train"] / len(df) - 0.70) < 0.03
    ids = [set(v["instruction"]) for v in a.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    for part in a.values():  # every intent present in every split
        assert part["intent"].nunique() == 3
    assert list(a["dev"]["instruction"]) == list(b["dev"]["instruction"])


def test_make_splits_rejects_tiny_classes():
    df = _make_df(2)
    with pytest.raises(ValueError):
        pd_mod.make_splits(df)


def test_write_jsonl_roundtrip(tmp_path):
    df = pd_mod.clean(_make_df(3))
    recs = pd_mod.to_records(df, "dev")
    path = tmp_path / "x.jsonl"
    pd_mod.write_jsonl(recs, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(recs)
    assert set(pd.read_json(path, lines=True).columns) == set(pd_mod.OUTPUT_COLUMNS)