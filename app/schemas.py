"""Pydantic v2 schemas for the triage API.

Three layers:
  * API input:       TicketIn
  * LLM output:     LLMTriage (System 2) and FastpathLLMOut (System 1 + LLM writer).
                    Lenient about case and extra keys, strict about enums, so a
                    malformed answer fails validation and triggers the repair retry.
  * API output:     TriageOut (success) and ErrorOut (failure).

The enums mirror the Bitext dataset (11 categories, 27 intents). `queue` is derived
from the category in code, never asked from the LLM, so it cannot be wrong.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

MAX_TICKET_CHARS = 4000


# --------------------------------------------------------------------------- enums
class Category(StrEnum):
    """Top-level ticket category (matches Bitext `category`)."""

    ACCOUNT = "ACCOUNT"
    CANCEL = "CANCEL"
    CONTACT = "CONTACT"
    DELIVERY = "DELIVERY"
    FEEDBACK = "FEEDBACK"
    INVOICE = "INVOICE"
    ORDER = "ORDER"
    PAYMENT = "PAYMENT"
    REFUND = "REFUND"
    SHIPPING = "SHIPPING"
    SUBSCRIPTION = "SUBSCRIPTION"


class Intent(StrEnum):
    """Fine-grained customer intent (matches Bitext `intent`)."""

    CANCEL_ORDER = "cancel_order"
    CHANGE_ORDER = "change_order"
    CHANGE_SHIPPING_ADDRESS = "change_shipping_address"
    CHECK_CANCELLATION_FEE = "check_cancellation_fee"
    CHECK_INVOICE = "check_invoice"
    CHECK_PAYMENT_METHODS = "check_payment_methods"
    CHECK_REFUND_POLICY = "check_refund_policy"
    COMPLAINT = "complaint"
    CONTACT_CUSTOMER_SERVICE = "contact_customer_service"
    CONTACT_HUMAN_AGENT = "contact_human_agent"
    CREATE_ACCOUNT = "create_account"
    DELETE_ACCOUNT = "delete_account"
    DELIVERY_OPTIONS = "delivery_options"
    DELIVERY_PERIOD = "delivery_period"
    EDIT_ACCOUNT = "edit_account"
    GET_INVOICE = "get_invoice"
    GET_REFUND = "get_refund"
    NEWSLETTER_SUBSCRIPTION = "newsletter_subscription"
    PAYMENT_ISSUE = "payment_issue"
    PLACE_ORDER = "place_order"
    RECOVER_PASSWORD = "recover_password"
    REGISTRATION_PROBLEMS = "registration_problems"
    REVIEW = "review"
    SET_UP_SHIPPING_ADDRESS = "set_up_shipping_address"
    SWITCH_ACCOUNT = "switch_account"
    TRACK_ORDER = "track_order"
    TRACK_REFUND = "track_refund"


class Priority(StrEnum):
    """Urgency level, per the rubric in the v2 prompt."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Queue(StrEnum):
    """Human work queue a ticket is routed to."""

    ACCOUNT = "account"
    BILLING = "billing"
    ORDERS = "orders"
    SHIPPING = "shipping"
    GENERAL = "general"


class HandledBy(StrEnum):
    """Which path produced the answer."""

    SYSTEM1 = "system1"
    SYSTEM2 = "system2"


class ErrorCode(StrEnum):
    """Typed error codes returned to API clients."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    SCHEMA_FAIL = "schema_fail"
    BLOCKED_INPUT = "blocked_input"


# ------------------------------------------------------------------------ mappings
INTENT_TO_CATEGORY: dict[Intent, Category] = {
    Intent.CREATE_ACCOUNT: Category.ACCOUNT,
    Intent.DELETE_ACCOUNT: Category.ACCOUNT,
    Intent.EDIT_ACCOUNT: Category.ACCOUNT,
    Intent.SWITCH_ACCOUNT: Category.ACCOUNT,
    Intent.RECOVER_PASSWORD: Category.ACCOUNT,
    Intent.REGISTRATION_PROBLEMS: Category.ACCOUNT,
    Intent.CHECK_CANCELLATION_FEE: Category.CANCEL,
    Intent.CONTACT_CUSTOMER_SERVICE: Category.CONTACT,
    Intent.CONTACT_HUMAN_AGENT: Category.CONTACT,
    Intent.DELIVERY_OPTIONS: Category.DELIVERY,
    Intent.DELIVERY_PERIOD: Category.DELIVERY,
    Intent.COMPLAINT: Category.FEEDBACK,
    Intent.REVIEW: Category.FEEDBACK,
    Intent.CHECK_INVOICE: Category.INVOICE,
    Intent.GET_INVOICE: Category.INVOICE,
    Intent.NEWSLETTER_SUBSCRIPTION: Category.SUBSCRIPTION,
    Intent.CANCEL_ORDER: Category.ORDER,
    Intent.CHANGE_ORDER: Category.ORDER,
    Intent.PLACE_ORDER: Category.ORDER,
    Intent.TRACK_ORDER: Category.ORDER,
    Intent.CHECK_PAYMENT_METHODS: Category.PAYMENT,
    Intent.PAYMENT_ISSUE: Category.PAYMENT,
    Intent.CHECK_REFUND_POLICY: Category.REFUND,
    Intent.GET_REFUND: Category.REFUND,
    Intent.TRACK_REFUND: Category.REFUND,
    Intent.CHANGE_SHIPPING_ADDRESS: Category.SHIPPING,
    Intent.SET_UP_SHIPPING_ADDRESS: Category.SHIPPING,
}

CATEGORY_TO_QUEUE: dict[Category, Queue] = {
    Category.ACCOUNT: Queue.ACCOUNT,
    Category.CANCEL: Queue.BILLING,
    Category.CONTACT: Queue.GENERAL,
    Category.DELIVERY: Queue.SHIPPING,
    Category.FEEDBACK: Queue.GENERAL,
    Category.INVOICE: Queue.BILLING,
    Category.SUBSCRIPTION: Queue.GENERAL,
    Category.ORDER: Queue.ORDERS,
    Category.PAYMENT: Queue.BILLING,
    Category.REFUND: Queue.BILLING,
    Category.SHIPPING: Queue.SHIPPING,
}

ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.RATE_LIMIT: 429,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.PROVIDER_ERROR: 502,
    ErrorCode.SCHEMA_FAIL: 502,
    ErrorCode.BLOCKED_INPUT: 400,
}


def queue_for(category: Category) -> Queue:
    """Return the work queue for a category."""
    return CATEGORY_TO_QUEUE[category]


# ------------------------------------------------------------------- normalisation
def _lower(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def _upper(v: Any) -> Any:
    return v.strip().upper() if isinstance(v, str) else v


LenientCategory = Annotated[Category, BeforeValidator(_upper)]
LenientIntent = Annotated[Intent, BeforeValidator(_lower)]
LenientPriority = Annotated[Priority, BeforeValidator(_lower)]


def _check_intent_category(intent: Intent, category: Category) -> None:
    expected = INTENT_TO_CATEGORY[intent]
    if category != expected:
        raise ValueError(f"intent '{intent.value}' belongs to {expected.value}, not {category.value}")


# --------------------------------------------------------------------------- input
class TicketIn(BaseModel):
    """Incoming support ticket."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TICKET_CHARS, description="Raw ticket text")
    client_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.\-:]+$"
    )


# ------------------------------------------------------------------- LLM outputs
class LLMTriage(BaseModel):
    """What the System 2 prompt must return. `reasoning` is dropped before the API reply."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str | None = None
    category: LenientCategory
    intent: LenientIntent
    priority: LenientPriority
    summary: str = Field(min_length=1, max_length=300)
    draft_reply: str = Field(min_length=1, max_length=1000)
    needs_human: bool

    @model_validator(mode="after")
    def _consistent(self) -> "LLMTriage":
        _check_intent_category(self.intent, self.category)
        return self


class FastpathLLMOut(BaseModel):
    """What the fast-path prompt returns (System 1 already supplied category/intent)."""

    model_config = ConfigDict(extra="ignore")

    priority: LenientPriority
    summary: str = Field(min_length=1, max_length=300)
    draft_reply: str = Field(min_length=1, max_length=1000)
    needs_human: bool


# -------------------------------------------------------------------------- output
class TriageOut(BaseModel):
    """Successful triage response."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    intent: Intent
    priority: Priority
    queue: Queue
    summary: str = Field(min_length=1, max_length=300)
    draft_reply: str = Field(min_length=1, max_length=1000)
    needs_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    handled_by: HandledBy
    prompt_version: str
    model: str
    git_sha: str

    @model_validator(mode="after")
    def _consistent(self) -> "TriageOut":
        _check_intent_category(self.intent, self.category)
        if self.queue != queue_for(self.category):
            raise ValueError(f"queue '{self.queue.value}' does not match category {self.category.value}")
        return self


class ErrorOut(BaseModel):
    """Error response body."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorCode
    message: str
    request_id: str | None = None
    retry_after_s: float | None = Field(default=None, ge=0)

    @property
    def http_status(self) -> int:
        """HTTP status code for this error."""
        return ERROR_HTTP_STATUS[self.error]