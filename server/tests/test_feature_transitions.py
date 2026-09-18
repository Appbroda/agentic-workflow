"""Feature status has one owner, one table, and a reason on every write.

Three things are proved here, in the order the task states them:

* the transition table is exhaustive over ``(from_status, to_status)`` and an illegal pair
  raises a classified platform defect rather than being coerced;
* no feature status assignment exists outside the transition service, checked against the
  source itself rather than against a list somebody maintains;
* every transition an end-to-end run performs records a non-empty reason.

The escape check is deliberately derived, in the same style as the terminal-path registry in
``tests/test_failure_diagnosis.py``. A hand-kept inventory of write sites is exactly what
decayed into forty-three of them.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from state.enums import FeatureWorkflowStatus
from state.external_operations import WorkflowCheckpointBoundary
from state.failure_diagnosis import FeatureFailureClassification
from state.feature_models import FeatureWorkflowSnapshot
from state.feature_transitions import (
    IllegalFeatureTransition,
    feature_transition_is_legal,
    transition_feature,
    transition_feature_state_json,
)
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import FeatureWorkflowOrchestrator

SERVER_ROOT = pathlib.Path(__file__).resolve().parents[1]
CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

# The one module allowed to assign a feature status. Anything else that does is the defect
# this guard exists to catch.
_TRANSITION_SERVICE = "state/feature_transitions.py"

# The two writes that are not transitions, named here rather than made invisible. Both would
# be trivial to hide from this guard by moving the enum into a local variable, and neither is,
# because a reader of this list should be able to see everything in `server/` that touches a
# feature status.
#
# `_preserve_cancellation` merges a stale in-flight runner result with a concurrent cancel
# request. The status it writes is the *persisted* one being carried forward, not a
# destination it chose, so expressing it as a transition would mean re-deciding a decision
# already made. The rule it encodes is in the roadmap's list of subsystems that are correct
# and must not be redesigned, so the merge is left exactly as it is.
#
# `retire` reads its own result back onto the queryable status column, which is duplicated
# between that column and the snapshot. It is the same projection `_apply_state` performs with
# `model.status = state.status` -- which this guard does not see, because it names no status
# at all -- and it names no destination: the destination was decided one statement earlier by
# `transition_feature_state_json`.
_NOT_TRANSITIONS = {
    ("storage/feature_store.py", "_preserve_cancellation"),
    ("storage/feature_store.py", "retire"),
}


def _fresh(feature_id: str = "feature-transitions") -> FeatureWorkflowSnapshot:
    """Return the state a feature has the moment it is accepted, before anything runs."""
    return _initial_feature_state(feature_id, StartFeatureRequest.model_validate(feature_payload()))


def _at(status: FeatureWorkflowStatus) -> FeatureWorkflowSnapshot:
    """Return a snapshot parked at one status, without going through the service to get there.

    Direct assignment on purpose: this is a fixture reaching a starting position, not the
    platform moving a feature. The guard below is scoped to non-test source for exactly this
    reason -- a test that could not construct an arbitrary origin could not check the table.
    """
    state = _fresh()
    state.status = status
    return state


# ---------------------------------------------------------------------------------------
# 1 -- the table, exhaustively
# ---------------------------------------------------------------------------------------


def test_every_status_pair_is_either_legal_or_a_classified_platform_defect() -> None:
    """Exhaustive over the seventeen statuses in both directions: 289 pairs, no gaps."""
    legal = 0
    refused = 0
    for origin in FeatureWorkflowStatus:
        for destination in FeatureWorkflowStatus:
            state = _at(origin)
            if feature_transition_is_legal(origin, destination):
                transition_feature(state, destination, reason="checking the table")
                assert state.status is destination
                legal += 1
                continue
            with pytest.raises(IllegalFeatureTransition) as raised:
                transition_feature(state, destination, reason="checking the table")
            assert (
                raised.value.failure_classification is FeatureFailureClassification.PLATFORM_DEFECT
            )
            assert origin.value in raised.value.diagnostics[0]
            assert destination.value in raised.value.diagnostics[0]
            # The refused transition left the feature exactly where it was.
            assert state.status is origin
            refused += 1
    assert legal + refused == len(FeatureWorkflowStatus) ** 2 == 289
    # A table that permitted everything would pass every other assertion in this module.
    assert refused > 0


def test_a_completed_feature_cannot_be_put_back_to_work() -> None:
    """The pair the audit named: nothing prevented ``completed -> running_child_workflows``."""
    completed = _at(FeatureWorkflowStatus.COMPLETED)

    with pytest.raises(IllegalFeatureTransition) as raised:
        transition_feature(
            completed,
            FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
            reason="a claim that should never have been acted on",
        )

    assert raised.value.failure_classification is FeatureFailureClassification.PLATFORM_DEFECT
    assert completed.status is FeatureWorkflowStatus.COMPLETED
    assert completed.transition_reason is None


@pytest.mark.parametrize(
    ("origin", "destination"),
    [
        (FeatureWorkflowStatus.COMPLETED, FeatureWorkflowStatus.CANCELLING),
        (FeatureWorkflowStatus.COMPLETED, FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN),
        (FeatureWorkflowStatus.CANCELLED, FeatureWorkflowStatus.PLANNING),
        (FeatureWorkflowStatus.PLANNING, FeatureWorkflowStatus.COMPLETED),
        (FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS, FeatureWorkflowStatus.CONTRACT_READY),
        (FeatureWorkflowStatus.CONTRACT_READY, FeatureWorkflowStatus.CREATING_PULL_REQUESTS),
        (FeatureWorkflowStatus.PLANNING, FeatureWorkflowStatus.CHANGES_REQUESTED),
        (FeatureWorkflowStatus.PLANNING, FeatureWorkflowStatus.PENDING),
        (FeatureWorkflowStatus.PLANNING, FeatureWorkflowStatus.FAILED),
    ],
)
def test_the_sequences_the_platform_cannot_actually_reach_are_refused(
    origin: FeatureWorkflowStatus, destination: FeatureWorkflowStatus
) -> None:
    """Each of these is a step order the code never produces, so the table must refuse it."""
    with pytest.raises(IllegalFeatureTransition):
        transition_feature(_at(origin), destination, reason="a sequence no path produces")


def test_restating_a_status_is_never_a_transition() -> None:
    """A feature that fails twice, or is cancelled twice, is not moving anywhere."""
    for status in FeatureWorkflowStatus:
        state = _at(status)
        transition_feature(state, status, reason="the same thing, again")
        assert state.status is status


def test_a_cancellation_may_be_finalised_after_the_workflow_already_wrote_cancelled() -> None:
    """The one backwards move out of a retired status, and the path that makes it.

    ``_mark_cancelled`` checkpoints ``cancelled`` from inside the run; ``_finish_cancellation``
    then re-reads that snapshot and writes ``cancelling`` back while external operations are
    still stopping. Recorded in the table rather than tolerated everywhere: the cancellation
    family may do this, and only to another member of the cancellation family.
    """
    cancelled = _at(FeatureWorkflowStatus.CANCELLED)
    transition_feature(
        cancelled, FeatureWorkflowStatus.CANCELLING, reason="external operations are still stopping"
    )
    assert cancelled.status is FeatureWorkflowStatus.CANCELLING

    with pytest.raises(IllegalFeatureTransition):
        transition_feature(
            _at(FeatureWorkflowStatus.CANCELLED),
            FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
            reason="ordinary work on a cancelled feature",
        )


# ---------------------------------------------------------------------------------------
# 2 -- every transition records why
# ---------------------------------------------------------------------------------------


def test_a_transition_with_no_reason_is_itself_a_platform_defect() -> None:
    """Requiring the reason is what stops it decaying back into an optional field."""
    for blank in ("", "   "):
        state = _fresh()
        with pytest.raises(IllegalFeatureTransition) as raised:
            transition_feature(state, FeatureWorkflowStatus.PLANNING, reason=blank)
        assert raised.value.failure_classification is FeatureFailureClassification.PLATFORM_DEFECT
        assert state.status is FeatureWorkflowStatus.PENDING


def test_a_transition_writes_the_status_the_reason_and_nothing_else() -> None:
    """The service owns three fields, and ``updated_at`` is deliberately not one of them."""
    state = _fresh()
    before = state.updated_at

    transition_feature(
        state, FeatureWorkflowStatus.PLANNING, reason="the contract is being written"
    )

    assert state.status is FeatureWorkflowStatus.PLANNING
    assert state.transition_reason == "the contract is being written"
    assert state.current_agent is None
    assert state.updated_at == before

    transition_feature(
        state,
        FeatureWorkflowStatus.CONTRACT_READY,
        reason="the plan exists",
        agent="feature_planner",
    )
    assert state.current_agent == "feature_planner"
    assert state.updated_at == before


def test_a_retirement_gets_the_same_table_as_every_other_transition() -> None:
    """``retire`` writes the persisted JSON directly; it is not exempt from the rules."""
    state = _at(FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS)
    persisted = state.model_dump(mode="json")

    retired = transition_feature_state_json(
        persisted,
        FeatureWorkflowStatus.CANCELLED,
        reason="an operator retired this feature",
        agent="operator",
    )

    assert retired["status"] == FeatureWorkflowStatus.CANCELLED.value
    assert retired["transition_reason"] == "an operator retired this feature"
    assert retired["current_agent"] == "operator"
    # The caller's dictionary is not mutated underneath it.
    assert persisted["status"] == FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS.value

    with pytest.raises(IllegalFeatureTransition):
        transition_feature_state_json(
            retired, FeatureWorkflowStatus.PLANNING, reason="picking a retired feature back up"
        )


@pytest.mark.asyncio
async def test_every_transition_a_whole_run_performs_records_a_reason() -> None:
    """Not the service in isolation: a real run, and every durable snapshot it wrote.

    A feature is accepted at ``pending`` with no transition behind it, so that is the one
    snapshot allowed to carry no reason. Everything after it moved, and every move says why.
    """
    captured: list[FeatureWorkflowSnapshot] = []

    async def _record(
        state: FeatureWorkflowSnapshot,
        boundary: WorkflowCheckpointBoundary,
        repository_id: str | None,
    ) -> None:
        del boundary, repository_id
        captured.append(state.model_copy(deep=True))

    orchestrator = FeatureWorkflowOrchestrator(checkpoint_writer=_record)
    final = await orchestrator.start(_fresh(), credentials=CREDENTIALS)

    assert final.status is FeatureWorkflowStatus.COMPLETED
    assert captured, "a completed run writes checkpoints"
    for snapshot in [*captured, final]:
        if snapshot.status is FeatureWorkflowStatus.PENDING:
            continue
        assert snapshot.transition_reason, (
            f"a feature in {snapshot.status.value} carries no reason for being there"
        )
        assert snapshot.transition_reason.strip()
    assert final.transition_reason == (
        "Every repository this feature needed has a pull request that was read back from "
        "the provider."
    )


@pytest.mark.asyncio
async def test_a_finished_feature_is_not_reopened_by_a_request_that_failed_after_it(
    tmp_path: pathlib.Path,
) -> None:
    """The one behaviour this task changed on purpose, and the defect the table found.

    ``_mark_failed_requires_human`` is the handler every unexpected exception reaches, and it
    was the only path that never asked what the feature it was about to tombstone had already
    reached. A repair approved against a completed feature, whose runner then raised, rewrote
    `completed` as `failed_requires_human` and threw the completion away -- pull requests,
    completion artifact and all -- because a later request failed.

    The caller still gets the same refusal: its request genuinely did not happen. What no
    longer happens is a finished feature being reopened by it.
    """
    from api.control_plane import FeatureOperationFailedError
    from storage.db import Database
    from storage.feature_store import SqlAlchemyFeatureControlPlane
    from tests.support import drain_feature_queue
    from tests.test_feature_state import BrokenAfterStartRunner

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'finished.db'}")
    await database.create_schema()
    try:
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "finished-feature"}
        )
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=BrokenAfterStartRunner())
        await store.start(
            request,
            idempotency_key="finished-feature",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        assert (
            await store.get_record("finished-feature")
        ).state.status is FeatureWorkflowStatus.COMPLETED

        with pytest.raises(FeatureOperationFailedError):
            await store.approve_repair(
                "finished-feature",
                repair_id="repair-1",
                actor_id="akhilesh",
                credentials=credentials,
            )

        record = await store.get_record("finished-feature")
        assert record.state.status is FeatureWorkflowStatus.COMPLETED
        # Silence would be the other defect. The timeline still says a request failed here.
        assert any(
            "already completed" in str(details.get("reason", ""))
            for _timestamp, _kind, _source, _event, details in await store.timeline(
                "finished-feature"
            )
        )
    finally:
        await database.drop_schema()
        await database.dispose()


# ---------------------------------------------------------------------------------------
# 3 -- nothing assigns a feature status outside the service
# ---------------------------------------------------------------------------------------


def _feature_status_writes() -> dict[tuple[str, str], int]:
    """Find every write of a feature status, by reading the source.

    Three shapes, because there are three ways to set a field on this platform's state:

    * ``x.status = FeatureWorkflowStatus.Y`` -- the forty-three sites this task replaced;
    * ``x["status"] = FeatureWorkflowStatus.Y.value`` -- how ``retire`` wrote the persisted
      snapshot, bypassing the model entirely;
    * ``x.model_copy(update={"status": FeatureWorkflowStatus.Y})`` -- how the state merge
      writes one, and the shape a future caller would reach for first if the assignment were
      the only thing guarded.

    ``model.status = state.status`` in ``_apply_state`` is not among them and is not meant to
    be: it projects a status the service already decided onto its queryable column, and it
    names no destination of its own.
    """
    sites: dict[tuple[str, str], int] = {}
    skip = {"tests", ".venv", "migrations", "__pycache__"}
    for path in sorted(SERVER_ROOT.rglob("*.py")):
        relative = path.relative_to(SERVER_ROOT)
        if skip & set(relative.parts) or str(relative) == _TRANSITION_SERVICE:
            continue
        tree = ast.parse(path.read_text())
        scopes: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _names_a_feature_status(node.value):
                written = any(_writes_a_status_field(target) for target in node.targets)
            elif isinstance(node, ast.Call):
                written = _copies_a_feature_status(node)
            else:
                continue
            if not written:
                continue
            key = (str(relative), _enclosing_function(scopes, node))
            sites[key] = sites.get(key, 0) + 1
    return sites


def _enclosing_function(
    scopes: list[ast.FunctionDef | ast.AsyncFunctionDef], node: ast.stmt | ast.expr
) -> str:
    """Name the innermost function a node sits in, or the module when it sits in none."""
    enclosing = max(
        (
            scope
            for scope in scopes
            if scope.lineno <= node.lineno <= (scope.end_lineno or scope.lineno)
        ),
        key=lambda scope: scope.lineno,
        default=None,
    )
    return enclosing.name if enclosing is not None else "<module>"


def _writes_a_status_field(target: ast.expr) -> bool:
    """Report whether an assignment target is a ``status`` attribute or dictionary key."""
    if isinstance(target, ast.Attribute):
        return target.attr == "status"
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "status"
    )


def _names_a_feature_status(node: ast.AST) -> bool:
    """Report whether an expression mentions the feature status enum anywhere inside it."""
    return any(
        isinstance(item, ast.Name) and item.id == "FeatureWorkflowStatus" for item in ast.walk(node)
    )


def _copies_a_feature_status(node: ast.Call) -> bool:
    """Report whether a ``model_copy`` call sets ``status`` to a feature status."""
    function = node.func
    if not (isinstance(function, ast.Attribute) and function.attr == "model_copy"):
        return False
    for keyword in node.keywords:
        if keyword.arg != "update" or not isinstance(keyword.value, ast.Dict):
            continue
        for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "status"
                and _names_a_feature_status(value)
            ):
                return True
    return False


def test_no_feature_status_is_written_outside_the_transition_service() -> None:
    """The structural half of "one component performs a feature status transition".

    Derived from the source rather than listed, so it cannot decay the way the forty-three
    sites did: a new write anywhere in ``server/`` fails this, and the only way to make it
    pass is to route it through the service or to add it above with a reason attached.
    """
    escapes = sorted(set(_feature_status_writes()) - _NOT_TRANSITIONS)

    assert not escapes, (
        "these write a feature status without going through `state.feature_transitions`, so "
        f"nothing validates the transition or records why it happened: {escapes}"
    )


def test_the_two_writes_that_are_not_transitions_still_exist_and_are_still_those_two() -> None:
    """An exemption for a site that has moved on would read as coverage and provide none."""
    assert set(_feature_status_writes()) == _NOT_TRANSITIONS
