"""The one component that moves a feature from one status to another.

Before this module, ``status`` was assigned directly at forty-three sites across four modules
and nothing validated any of them. Nothing prevented ``completed -> running_child_workflows``;
nothing recorded why a status was written; and a status set in the middle of a merge could not
be told apart from one set by the step that owned the decision.

Three questions live here, and only here:

``is (from_status, to_status) legal?``
    Derived from what the code already does rather than from what ``FeatureWorkflowStatus``
    looks like it should allow. The table below records, per destination, the origins the
    platform actually writes it from.

``which fields may a transition write?``
    Exactly three: ``status``, ``transition_reason``, and -- when the caller names one --
    ``current_agent``. Not ``updated_at``: that is a mutation timestamp, written by whatever
    persists the snapshot, and a transition service that touched it would change the ordering
    of the features list as a side effect of being introduced.

``what reason is recorded?``
    A mandatory sentence, kept on the snapshot as ``transition_reason``. A status with no
    reason is what made the forty-three sites indistinguishable from each other in a durable
    record; the failure summary explains why a feature *stopped*, and this explains why it
    is where it is whether or not it stopped.

An attempted illegal transition is a platform defect, not a workflow outcome. It raises
``IllegalFeatureTransition``, which carries ``PLATFORM_DEFECT`` through the ordinary
``DiagnosedFailure`` contract from `25-`, so it is recorded as this codebase's fault and never
as a statement about the target repository. It is not coerced and not logged-and-continued:
either the table is wrong, in which case it is one line to correct with a reason attached, or
the caller is wrong, in which case silence would have hidden it.
"""

from __future__ import annotations

from typing import Any

from state.enums import FeatureWorkflowStatus
from state.failure_diagnosis import DiagnosedFailure, FeatureFailureClassification
from state.feature_models import FeatureWorkflowSnapshot


class IllegalFeatureTransition(DiagnosedFailure):
    """A status transition no path in this platform is supposed to be able to make."""

    def __init__(self, message: str, *, diagnostic: str) -> None:
        """Classify the attempt as this platform's defect and say what was attempted."""
        super().__init__(
            message,
            classification=FeatureFailureClassification.PLATFORM_DEFECT,
            diagnostics=[diagnostic],
        )


# The statuses a feature never leaves. `storage.feature_store._RETIRED_EQUIVALENT_STATUSES`
# is the same set, read by `retire` and by the queue's claim disposition; this is that rule
# expressed as a transition.
_RETIRED = frozenset(
    {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)

# The two cancelled statuses, separately from `completed`. A cancellation is finished being
# decided in the same sense, but it is finalised by a path that re-reads the snapshot after
# the workflow has already written one of these -- see `_CANCELLATION_ORIGINS`.
_CANCELLED = frozenset(
    {
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)

# Every status a claim may still act on -- everything a run passes through, plus the two
# stopped-and-answerable ones. `failed` and `failed_requires_human` are deliberately in here:
# a person resumes from them, and the platform itself re-enters publication from
# `failed_requires_human` when a repository's reviewed work is still unpublished, which is
# exactly what `feature_is_at_rest` exists to say.
#
# This is the origin set for the destinations the
# platform does not gate on status, and that is a deliberate statement rather than a
# shortcut: which step runs next is `next_step`'s answer, read from artifacts and child rows,
# and the status is a description of where the feature is rather than a gate on where it may
# go. Encoding a stricter order here would put a second, weaker copy of `next_step` in the
# middle of the write path -- the second-source-of-truth defect this plan exists to remove.
_RESUMABLE = frozenset(FeatureWorkflowStatus) - _RETIRED

# Cancellation is finalised by `FeatureStore._finish_cancellation`, which re-reads the
# snapshot *after* the workflow's own `_mark_cancelled` has already checkpointed `cancelled`.
# So a cancelled feature can be written back to `cancelling` while its external operations
# are still stopping. That is a real backwards move out of a retired status and it is
# recorded here rather than quietly permitted everywhere: only the cancellation family may
# do it, and only to another member of the cancellation family.
_CANCELLATION_ORIGINS = _RESUMABLE | _CANCELLED


# Every legal transition, keyed by destination. A status absent from a value set cannot reach
# that key; a status absent from the keys is never written by any path.
#
# Derived from the forty-three assignment sites this task replaced, not from the enum. Four
# destinations are written exactly one line after the status they follow, and those are the
# narrow entries -- they are the sequences a caller could actually get wrong.
_LEGAL_ORIGINS: dict[FeatureWorkflowStatus, frozenset[FeatureWorkflowStatus]] = {
    # Allocated at creation, never returned to. A feature that went back to `pending` would
    # be eligible for a fresh start claim with a run's worth of artifacts behind it.
    FeatureWorkflowStatus.PENDING: frozenset(),
    FeatureWorkflowStatus.ANALYZING_PRD: _RESUMABLE,
    FeatureWorkflowStatus.INSPECTING_REPOSITORIES: _RESUMABLE,
    # `completed` is additionally legal here, and only here among the retired statuses:
    # a feature revision (a person asking for changes to already-published work) re-opens
    # the run at planning. Only `revise_feature` makes that move, synchronously inside the
    # request, so no claim ever observes a completed feature mid-revision.
    FeatureWorkflowStatus.PLANNING: frozenset({*_RESUMABLE, FeatureWorkflowStatus.COMPLETED}),
    # Written by the planning step, at the point the contract and plan artifacts exist. The
    # planner sets `planning` itself two statements earlier, so nothing else can precede it.
    FeatureWorkflowStatus.CONTRACT_READY: frozenset({FeatureWorkflowStatus.PLANNING}),
    FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS: _RESUMABLE,
    FeatureWorkflowStatus.INTEGRATION_REVIEW: _RESUMABLE,
    # The integration verdict's own routing: findings exist, so the review ran, so the review
    # step set `integration_review` before deciding this.
    FeatureWorkflowStatus.CHANGES_REQUESTED: frozenset({FeatureWorkflowStatus.INTEGRATION_REVIEW}),
    FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS: _RESUMABLE,
    # Publication announces itself and then starts, on consecutive lines.
    FeatureWorkflowStatus.CREATING_PULL_REQUESTS: frozenset(
        {FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS}
    ),
    # A completion artifact is a claim that every pull request exists and was read back from
    # the provider. Only the path that verified that may make it.
    FeatureWorkflowStatus.COMPLETED: frozenset({FeatureWorkflowStatus.CREATING_PULL_REQUESTS}),
    # No path writes `failed`; every stop this platform records is `failed_requires_human`,
    # because every one of them is answerable by somebody. It stays in the enum because
    # durable rows carry it, and it stays unreachable here so that a path that starts writing
    # it has to say, in this table, what it means by the distinction.
    FeatureWorkflowStatus.FAILED: frozenset(),
    FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN: _RESUMABLE,
    FeatureWorkflowStatus.WAITING_FOR_HUMAN: _RESUMABLE,
    FeatureWorkflowStatus.CANCELLING: _CANCELLATION_ORIGINS,
    FeatureWorkflowStatus.CANCELLED: _CANCELLATION_ORIGINS,
    # Only the finalisation path knows whether external side effects were retained, and it
    # reaches that answer from `cancelling`.
    FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS: frozenset(
        {FeatureWorkflowStatus.CANCELLING, *_CANCELLED}
    ),
}


def feature_transition_is_legal(
    from_status: FeatureWorkflowStatus, to_status: FeatureWorkflowStatus
) -> bool:
    """Report whether the platform is allowed to move a feature between these two statuses.

    Restating a status is always allowed. A feature that fails twice, is cancelled twice, or
    reaches its open questions again on a later claim is not transitioning anywhere, and
    refusing that would turn every idempotent write into a platform defect.
    """
    if from_status is to_status:
        return True
    return from_status in _LEGAL_ORIGINS.get(to_status, frozenset())


def _require_legal(
    from_status: FeatureWorkflowStatus, to_status: FeatureWorkflowStatus, *, reason: str
) -> None:
    """Refuse an illegal transition or an unexplained one, as a classified platform defect."""
    if not reason.strip():
        msg = f"a transition to {to_status.value} must record why"
        raise IllegalFeatureTransition(
            msg,
            diagnostic=(
                f"The platform moved a feature to {to_status.value} without recording a "
                "reason. Every status transition is required to carry one."
            ),
        )
    if feature_transition_is_legal(from_status, to_status):
        return
    msg = f"illegal feature transition {from_status.value} -> {to_status.value}"
    raise IllegalFeatureTransition(
        msg,
        diagnostic=(
            f"The platform attempted to move a feature from {from_status.value} to "
            f"{to_status.value}, which no path is supposed to be able to do. This is a defect "
            "in the platform rather than a fault in the repository or the requirement."
        ),
    )


def transition_feature(
    state: FeatureWorkflowSnapshot,
    to_status: FeatureWorkflowStatus,
    *,
    reason: str,
    agent: str | None = None,
) -> FeatureWorkflowSnapshot:
    """Move one feature to a new status, in place, and record why.

    Returns the same object it was handed, so a caller that reads more naturally as an
    expression can use the result without a second source of truth existing for a moment.

    ``agent`` is written only when supplied. Most transitions hand the feature to a named
    agent and say so; a few -- the queue leaving a refused feature answerable, publication
    announcing itself before it starts -- keep whoever already owns it.
    """
    _require_legal(state.status, to_status, reason=reason)
    state.status = to_status
    state.transition_reason = reason
    if agent is not None:
        state.current_agent = agent
    return state


def transition_feature_state_json(
    state_json: dict[str, Any],
    to_status: FeatureWorkflowStatus,
    *,
    reason: str,
    agent: str | None = None,
) -> dict[str, Any]:
    """Apply the same transition to a snapshot still in its persisted JSON form.

    ``FeatureStore.retire`` writes the status column and the snapshot together inside one
    session, without rehydrating the record, because a retirement must leave every child row,
    pull request and journal entry exactly where it is. It gets the same table and the same
    recorded reason as every other transition rather than an exemption.
    """
    from_status = FeatureWorkflowStatus(str(state_json["status"]))
    _require_legal(from_status, to_status, reason=reason)
    updated = dict(state_json)
    updated["status"] = to_status.value
    updated["transition_reason"] = reason
    if agent is not None:
        updated["current_agent"] = agent
    return updated


__all__ = [
    "IllegalFeatureTransition",
    "feature_transition_is_legal",
    "transition_feature",
    "transition_feature_state_json",
]
