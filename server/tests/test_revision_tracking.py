"""What the repository revision is for, once every attempt actually records one.

Capturing the revision is only half the change. The value has four consumers, and each of
them was silently degraded for the 165 of 281 recorded workstreams that carried none:

``repair_is_stale``          returned False unconditionally, so staleness detection was off
``_reviewed_revision``       degraded to the empty string
``_remediation_signature``   lost a discriminator
``ModelRoutingInputs``       received None

Turning a guard back on changes behaviour, which is the point and also the risk. Each test
below asserts the *direction* of that change rather than only that it happened, because a
staleness check that starts refusing work it should allow is worse than one that never ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from api.control_plane import RequestScopedCredentials, WorkflowConflictError
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import IntegrationReviewArtifact
from configs.model_roles import ModelRole
from services.repository_repair import current_repairs, repair_is_stale
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.test_feature_api import feature_payload
from tests.test_repository_repair import stopped_feature
from tools.model_routing import ModelExecutionMode, ModelRouter, ModelRoutingInputs
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowOrchestrator,
    MockChildWorkstreamExecutor,
    _propose_repository_repairs,
    _remediation_signature,
    _reviewed_child_result,
)

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


class _RevisionScriptExecutor:
    """Reject once with a revision, then settle without one.

    The shape that produced the defect: a workstream whose earlier attempt measured a
    revision and whose settling attempt reported none. Everything else is delegated to the
    deterministic mock executor, so this changes exactly one fact per attempt.

    The settling attempt approves rather than failing, deliberately. An approval returns
    from the retry loop early, which is the only way to reach the write this test is about:
    the exhausted-retry path passes through `_with_child_diagnostics`, which applies the
    same fallback first and would mask whether the write itself has one.
    """

    def __init__(self, revisions: list[str | None]) -> None:
        """Queue the revision each successive attempt reports."""
        self._revisions = list(revisions)
        self._inner = MockChildWorkstreamExecutor()
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Run the deterministic executor, then overwrite what it reports having reviewed."""
        execution = await self._inner.run(**kwargs)
        index = min(self.attempts, len(self._revisions) - 1)
        self.attempts += 1
        if index == len(self._revisions) - 1:
            return ChildExecution(
                result=execution.result.model_copy(
                    update={"current_revision": self._revisions[index]}
                ),
                code_completion=execution.code_completion,
                review=execution.review,
            )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "current_revision": self._revisions[index],
                    "status": "failed",
                    "pull_request_readiness": False,
                    # Different source and a different finding each attempt, so the
                    # convergence guards allow the next one rather than stopping the loop.
                    "production_files_changed": [f"src/module_{index}.py"],
                    "production_diff_fingerprint": f"fingerprint-{index}",
                    "blocking_issues": [f"The reviewer asked for change {index}."],
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


@pytest.mark.asyncio
async def test_the_final_write_cannot_null_a_revision_the_loop_already_knew() -> None:
    """The settling attempt must not erase the revision an earlier one established.

    The in-loop write always carried the child's previous revision forward; the write that
    persists the settled workstream did not. So the attempt that ends the loop -- reporting
    no revision, as most did before this task -- nulled out a value the loop already had.
    """
    state = _initial_feature_state(
        "feature-revision", StartFeatureRequest.model_validate(feature_payload())
    )
    executor = _RevisionScriptExecutor(["revision-one", None])
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=CREDENTIALS)

    backend = result.child_workflows["backend"]
    # More than one attempt ran, so the settling write was actually reached with a
    # revision-less result behind a revision-carrying one.
    assert backend.retry_count >= 1
    assert backend.current_revision == "revision-one"


def test_repair_staleness_is_off_without_a_revision_and_on_with_one() -> None:
    """The behaviour change this task turns on, stated as the two cases side by side.

    A proposal recorded before the platform tracked revisions has no revision to compare, and
    reading that as stale would refuse every historical repair. A proposal that does have one
    and finds the checkout has moved is stale, which is what was never detectable.
    """
    state = stopped_feature()
    _propose_repository_repairs(state)
    proposal = next(iter(current_repairs(state.artifacts).values()))

    # Pre-tracking rows: no recorded revision on either side, and not stale.
    assert repair_is_stale(proposal, current_revision=None) is False
    assert (
        repair_is_stale(
            proposal.model_copy(update={"proposed_at_revision": None}),
            current_revision="revision-two",
        )
        is False
    )
    # The checkout has not moved.
    assert repair_is_stale(proposal, current_revision="revision-one") is False
    # And it has.
    assert repair_is_stale(proposal, current_revision="revision-two") is True


@pytest.mark.asyncio
async def test_a_superseded_repair_refusal_is_what_gets_persisted(tmp_path: Path) -> None:
    """The supersession must survive the refusal that carries it, all the way to the row.

    This path has never executed in production -- `select count(*) from
    feature_repository_repairs` returns 0 -- and turning staleness detection on is what makes
    it reachable. `RepairSupersededError` carries the state change with the refusal precisely
    so a caller can persist it; a caller that discards the exception's state would refuse the
    approval and then offer the same stale repair again for ever.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'revision.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="revision-1",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        state = (await store.get_record("feature-login")).state
        state.child_workflows["backend"] = stopped_feature().child_workflows["backend"]
        _propose_repository_repairs(state)
        proposal = next(iter(current_repairs(state.artifacts).values()))
        await store._replace_state(  # noqa: SLF001 - exercising the projection directly
            "feature-login", state, "repository_repair_proposed"
        )
        assert proposal.proposed_at_revision == "revision-one"

        # The checkout moves, which is now a fact the platform records on every attempt
        # rather than only on the ones that reached a unanimous review.
        moved = (await store.get_record("feature-login")).state
        moved.child_workflows["backend"] = moved.child_workflows["backend"].model_copy(
            update={"current_revision": "revision-two"}
        )
        await store._replace_state(  # noqa: SLF001
            "feature-login", moved, "child_revision_advanced"
        )

        # The store translates the refusal for its callers, having first persisted the
        # state the refusal carried.
        with pytest.raises(WorkflowConflictError, match="superseded"):
            await store.approve_repair(
                "feature-login",
                repair_id=proposal.repair_id,
                actor_id="alex",
                credentials=CREDENTIALS,
            )

        # Read back from the database, not from the exception: the point of the contract is
        # that the refusal's state is the state that lands.
        listed = await store.repairs("feature-login")
        assert [item.status for item in listed] == ["superseded"]
    finally:
        await database.dispose()


def test_the_routing_fingerprint_includes_the_revision_and_still_grants_nothing() -> None:
    """Routing gains a discriminator it was missing, and gains no authority with it.

    `ModelRoutingInputs.repository_revision` was None on every revision-less workstream, so
    two attempts against genuinely different checkouts produced the same routing question
    and the second reused the first's answer. It now differs. What must not change is that
    the router still only ever selects a role: `decide_child_retry` is the sole authority on
    whether an attempt happens at all, and a routing decision that could revive a stopped
    workstream would put that authority in two places.
    """
    base = {
        "feature_id": "feature-revision",
        "repository_id": "backend",
        "execution_mode": ModelExecutionMode.INITIAL_IMPLEMENTATION,
        "attempt": 1,
    }
    without = ModelRoutingInputs.model_validate(base)
    first = ModelRoutingInputs.model_validate({**base, "repository_revision": "revision-one"})
    second = ModelRoutingInputs.model_validate({**base, "repository_revision": "revision-two"})

    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() != without.fingerprint()
    # The same checkout is the same question, so a resumed attempt reuses its decision
    # rather than paying for a second one.
    assert first.fingerprint() == first.model_copy().fingerprint()

    outcome = ModelRouter().route(inputs=second)

    # A role and a reason. No attempt, no budget, no escalation -- the outcome carries no
    # field capable of saying "run again".
    assert outcome.decision.role is ModelRole.CODING
    assert not hasattr(outcome.decision, "additional_attempts")
    assert not hasattr(outcome, "may_attempt")


@pytest.mark.asyncio
async def test_a_remediation_signature_separates_two_different_reviewed_revisions() -> None:
    """The discriminator a revision-less workstream never contributed.

    The direction matters more than the difference. A signature that changes means "new
    input", which *permits* an attempt; it can never make a previously eligible remediation
    ineligible. So the risk of restoring this discriminator is not that work gets refused --
    it is that two genuinely different checkouts stop colliding, which is the intent.
    """
    state = _initial_feature_state(
        "feature-signature", StartFeatureRequest.model_validate(feature_payload())
    )
    result = await FeatureWorkflowOrchestrator().start(state, credentials=CREDENTIALS)
    review = next(item for item in result.artifacts if isinstance(item, IntegrationReviewArtifact))
    reviewed = _reviewed_child_result(result, review, "backend")
    assert reviewed is not None

    def _signature_for(revision: str) -> str:
        moved = result.model_copy(deep=True)
        moved.artifacts = [
            item.model_copy(update={"current_revision": revision})
            if getattr(item, "artifact_id", None) == reviewed.artifact_id
            else item
            for item in moved.artifacts
        ]
        return _remediation_signature(moved, review, "backend")

    # Two identical revisions are the same question.
    assert _signature_for("revision-one") == _signature_for("revision-one")
    # Two different ones are not, which is what a null revision made impossible.
    assert _signature_for("revision-one") != _signature_for("revision-two")
