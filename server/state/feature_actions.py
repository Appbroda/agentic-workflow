"""Durable, credential-free records for workflow-changing actions a person asked for.

The external-operation journal already makes each *side effect* idempotent and recoverable.
What it cannot answer is whether the thing somebody asked the platform to do -- retry this
repository, approve this repair, resume this feature -- actually happened. That answer used
to live in one nullable column on a chat message, claimed by a conditional UPDATE and
finished by another; an executor that died between the two left it reading ``executing``
with nothing in the system able to move it again.

These models are that answer made durable: a stable identity so a repeat is a replay rather
than a second effect, a lease so a dead executor is detectable, and enough evidence for
recovery to decide what happened without rerunning anything blindly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, JsonValue, field_validator

from state.enums import FeatureActionStatus
from state.models import StateModel

type NonEmptyString = Annotated[str, Field(min_length=1)]

# Every action type the platform will execute on a person's behalf. Declared here rather than
# inferred from a string so a stored proposal from an older build cannot name something this
# build does not recognise, and so authorization has a closed set to check against.
ACTION_TYPES = frozenset(
    {
        "ANSWER_CLARIFICATION",
        "RESUME_WORKFLOW",
        "RETRY_WORKSTREAM",
        "ANSWER_DESIGN_VERDICT",
        "CANCEL_WORKFLOW",
        "APPROVE_REPOSITORY_REPAIR",
        "REJECT_REPOSITORY_REPAIR",
        "APPROVE_CONTRACT_CHANGE",
        "REJECT_CONTRACT_CHANGE",
        "RETIRE_FEATURE",
        # Opening the pull requests a feature that did not land is holding. Named after the
        # decision rather than the button, like every other member here.
        "PUBLISH_FEATURE",
        # Asking for changes to a completed feature's published work. It re-opens the run on
        # `-V{n}` branches, opens replacement pull requests, and closes the superseded ones.
        "REVISE_FEATURE",
    }
)

# Where the request came from. Both go through the same durable record and the same
# authorization: chat is an interpretation layer, never a second way in.
ACTION_ORIGINS = frozenset({"chat", "rest"})


class FeatureAction(StateModel):
    """One durable workflow-changing action, its ownership and its outcome."""

    action_id: NonEmptyString
    feature_id: NonEmptyString
    repository_id: str | None = None
    action_type: NonEmptyString
    actor_id: NonEmptyString
    actor_display_name: str | None = None
    origin: NonEmptyString
    origin_message_id: int | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    input_fingerprint: NonEmptyString
    idempotency_key: NonEmptyString
    status: FeatureActionStatus
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    domain_result_committed_at: datetime | None = None
    pending_result_summary: str | None = None
    result_summary: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    external_operation_ids: list[str] = Field(default_factory=list)
    reconciled_by: str | None = None
    reconciliation_reason: str | None = None
    reconciled_at: datetime | None = None

    @field_validator("action_type")
    @classmethod
    def action_type_must_be_supported(cls, value: str) -> str:
        """Refuse to rehydrate an action this build does not know how to execute."""
        if value not in ACTION_TYPES:
            msg = f"unsupported action type: {value}"
            raise ValueError(msg)
        return value

    @field_validator("origin")
    @classmethod
    def origin_must_be_known(cls, value: str) -> str:
        """Keep the audit trail's provenance to a closed set."""
        if value not in ACTION_ORIGINS:
            msg = f"unknown action origin: {value}"
            raise ValueError(msg)
        return value

    @field_validator(
        "lease_expires_at",
        "heartbeat_at",
        "started_at",
        "completed_at",
        "domain_result_committed_at",
        "reconciled_at",
        "created_at",
    )
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime | None) -> datetime | None:
        """Keep lease expiry comparable across worker processes in different zones."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            msg = "action timestamps must be timezone-aware"
            raise ValueError(msg)
        return value

    @property
    def is_terminal(self) -> bool:
        """Return whether this action has an outcome a repeat should replay rather than redo."""
        from state.enums import TERMINAL_ACTION_STATUSES

        return self.status in TERMINAL_ACTION_STATUSES

    def lease_is_live(self, *, now: datetime) -> bool:
        """Return whether some executor still credibly owns this action.

        An action with no expiry is not owned. One whose expiry has passed is not owned
        either, whatever it still claims: that is precisely the crashed-executor case, and
        treating the claim as authoritative is what left actions stuck forever.
        """
        return self.lease_expires_at is not None and self.lease_expires_at > now


def normalize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload in a stable shape so the same intent hashes to the same identity.

    Only ordering and empty-value handling are normalized. Values are not coerced: a retry
    that buys two attempts and one that buys three are different intents and must fingerprint
    differently.
    """
    return {key: payload[key] for key in sorted(payload) if payload[key] is not None}


__all__ = [
    "ACTION_ORIGINS",
    "ACTION_TYPES",
    "FeatureAction",
    "normalize_action_payload",
]
