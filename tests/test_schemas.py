"""Tests for app/schemas.py."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    CATEGORY_TO_QUEUE,
    ERROR_HTTP_STATUS,
    INTENT_TO_CATEGORY,
    Category,
    ErrorCode,
    ErrorOut,
    FastpathLLMOut,
    Intent,
    LLMTriage,
    Queue,
    TicketIn,
    TriageOut,
    queue_for,
)


def _good_out(**over):
    base = dict(
        category="ORDER", intent="track_order", priority="medium", queue="orders",
        summary="Customer asks where order is.", draft_reply="Thanks for reaching out.",
        needs_human=False, confidence=0.93, handled_by="system1",
        prompt_version="fastpath_v1", model="small", git_sha="abc123",
    )
    base.update(over)
    return base


# ---- mappings ---------------------------------------------------------------
def test_enum_sizes_and_mapping_completeness():
    assert len(Category) == 11 and len(Intent) == 27
    assert set(INTENT_TO_CATEGORY) == set(Intent)
    assert set(INTENT_TO_CATEGORY.values()) == set(Category)  # every category used
    assert set(CATEGORY_TO_QUEUE) == set(Category)
    assert set(ERROR_HTTP_STATUS) == set(ErrorCode)


def test_mapping_matches_prepared_data_if_present():
    path = Path(__file__).resolve().parents[1] / "data" / "train.jsonl"
    if not path.exists():
        pytest.skip("data/train.jsonl not generated yet")
    pairs = {
        (r["intent"], r["category"])
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }
    expected = {
        (i.value, c.value)
        for i, c in INTENT_TO_CATEGORY.items()
    }

    print("\nExtra mappings in dataset:")
    print(*sorted(pairs - expected), sep="\n")

    print("\nMissing mappings in dataset:")
    print(*sorted(expected - pairs), sep="\n")

    assert pairs == expected

# ---- TicketIn ---------------------------------------------------------------
def test_ticket_in_strips_and_validates():
    assert TicketIn(text="  hello  ").text == "hello"
    with pytest.raises(ValidationError):
        TicketIn(text="   ")
    with pytest.raises(ValidationError):
        TicketIn(text="x" * 4001)
    assert TicketIn(text="x" * 4000)
    with pytest.raises(ValidationError):
        TicketIn(text="hi", client_id="bad id!")
    with pytest.raises(ValidationError):
        TicketIn(text="hi", unexpected="x")


# ---- LLM outputs ------------------------------------------------------------
def test_llm_triage_normalises_case_and_ignores_extras():
    out = LLMTriage.model_validate(
        dict(category="order", intent="Track_Order", priority="HIGH", summary="s",
             draft_reply="r", needs_human="false", reasoning="because", extra_key=1)
    )
    assert out.category is Category.ORDER and out.intent is Intent.TRACK_ORDER
    assert out.needs_human is False


def test_llm_triage_rejects_bad_enum_and_inconsistent_pair():
    base = dict(category="ORDER", intent="track_order", priority="low", summary="s",
                draft_reply="r", needs_human=False)
    with pytest.raises(ValidationError):
        LLMTriage.model_validate({**base, "intent": "make_coffee"})
    with pytest.raises(ValidationError):
        LLMTriage.model_validate({**base, "category": "REFUND"})
    with pytest.raises(ValidationError):
        LLMTriage.model_validate({**base, "priority": "critical"})


def test_fastpath_out_requires_fields():
    assert FastpathLLMOut.model_validate(
        dict(priority="low", summary="s", draft_reply="r", needs_human=False)
    )
    with pytest.raises(ValidationError):
        FastpathLLMOut.model_validate(dict(priority="low", summary="s"))


def test_json_schema_available_for_repair_prompt():
    schema = LLMTriage.model_json_schema()
    assert "intent" in schema["properties"] and "$defs" in schema


# ---- TriageOut / ErrorOut ---------------------------------------------------
def test_triage_out_valid_and_serialises_to_plain_strings():
    out = TriageOut(**_good_out())
    dumped = json.loads(out.model_dump_json())
    assert dumped["intent"] == "track_order" and dumped["queue"] == "orders"
    assert set(dumped) == {
        "category", "intent", "priority", "queue", "summary", "draft_reply", "needs_human",
        "confidence", "handled_by", "prompt_version", "model", "git_sha",
    }


@pytest.mark.parametrize(
    "override",
    [dict(confidence=1.2), dict(confidence=-0.1), dict(category="REFUND"),
     dict(queue="billing"), dict(handled_by="system3"), dict(extra=1)],
)
def test_triage_out_rejects_invalid(override):
    with pytest.raises(ValidationError):
        TriageOut(**_good_out(**override))


def test_queue_for_all_categories():
    assert all(isinstance(queue_for(c), Queue) for c in Category)


def test_error_out_status_codes():
    assert ErrorOut(error=ErrorCode.RATE_LIMIT, message="slow down", retry_after_s=2).http_status == 429
    assert ErrorOut(error="blocked_input", message="no").http_status == 400
    with pytest.raises(ValidationError):
        ErrorOut(error="made_up", message="x")