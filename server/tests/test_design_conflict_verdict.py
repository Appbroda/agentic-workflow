"""A design conflict is a question somebody can answer, and the answer binds the next attempt.

49- Part B stops a workstream the moment a demand it already satisfied is demanded again, and
that stop is right: another attempt would re-argue the decision rather than answer it. What it
lacked was an exit. The stop's whole content was a sentence in `blocking_issues`, so the only
things a person could do with a design question were cancel the feature or buy attempts and
hope the argument came out the other way.

These tests are about the exit, and about the three things that make it worth anything:

* the verdict reaches the resumed attempt as an instruction, in the words the person wrote --
  runs 193 and 194 showed that a clarification answer encoding the implementation strategy is
  what precedes a delivery, and a verdict summarised into a policy is not one;
* the recurrence it answers can no longer stop the loop, because a stop that fired again before
  the engineer was called would make answering the question change nothing;
* and the stop's question never reaches an engineer as work, whichever exit is taken. The
  verdict path drops it, because the decision has answered it. An ordinary retry grant answers
  nothing, so there the question is restated as a statement of record: the next attempt is still
  told its workstream stopped over a re-litigated decision, in a sentence that does not ask it
  to arbitrate two reviews.

Everything is asserted as an effect: what the lineage holds, what the executor was handed, what
the queue claimed. Nothing here asserts that a method was called.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from api.control_plane import RequestScopedCredentials, WorkflowConflictError
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import ChildWorkflowResultArtifact, DesignConflictArtifact
from main import create_app
from services.design_conflict import (
    open_design_conflicts,
    settled_questions,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import FeatureWorkflowSnapshot
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.support import drain_feature_queue, settle
from tests.test_feature_api import feature_payload
from tests.test_resolved_issue_ledger import (
    COMPENSATION_AS_INTEGRATION_FIX,
    COMPENSATION_RAISED,
    COMPENSATION_REWORDED,
    UNRELATED_DEFECT,
    DemandingIntegrationReviewer,
    ScriptedFindingsExecutor,
)
from tools.resolved_issue_ledger import (
    IssueAuthority,
    RecurringIssue,
    ResolvedIssue,
    SettledQuestion,
    recurring_resolved_issues,
    satisfied_invariant_lines,
    settled_question_lines,
)
from tools.review_fix_classification import fingerprint_for_text
from workflows.feature_workflow import FeatureWorkflowOrchestrator

# What a person would actually type: not "yes", but which position holds and how to build it.
# The prose is the deliverable here -- it is spliced into the next coding call verbatim.
DECISION_TEXT = (
    "The compensating cleanup stands. Delete the rows the bulk create inserted in the same "
    "transaction the failure is detected in, and leave the row-level validation where it is."
)
OVERRULE_TEXT = (
    "The compensating cleanup does not stand for this feature. Partial success is the "
    "documented behaviour of this endpoint; report the failed rows instead of unwinding."
)

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

API_KEY = "design-conflict-key"
API_AUTH = {"Authorization": f"Bearer {API_KEY}"}

# The client is tested against saved server responses rather than hand-written objects, because
# a fixture somebody typed agrees with the client and not with the server. This is where the
# API test below writes the one the design-verdict panel is rendered from.
_CLIENT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "client"
    / "tests"
    / "fixtures"
    / "clarification.awaiting-design-verdict.json"
)


def _payload_for(repositories: Sequence[str] | None) -> dict[str, Any]:
    """Return the shared feature request, narrowed to the repositories a test needs."""
    payload = feature_payload()
    if repositories is not None:
        payload["repositories"] = [
            item for item in payload["repositories"] if item["repository_id"] in repositories
        ]
    return payload


async def _stopped_on_a_design_conflict(
    feature_id: str,
    *,
    script: list[list[str]] | None = None,
    integration_reviewer: object | None = None,
    repositories: Sequence[str] | None = None,
) -> tuple[FeatureWorkflowOrchestrator, FeatureWorkflowSnapshot, ScriptedFindingsExecutor]:
    """Run a feature to the 49- Part B stop, and hand back the orchestrator that got there.

    The lineage is B1's: a demand raised, absent from the next attempt, and demanded again.
    Returned with its orchestrator because what these tests do next is answer the question,
    and the answer runs through the same orchestrator the stop came out of.
    """
    executor = ScriptedFindingsExecutor(
        script
        or [
            [COMPENSATION_RAISED],
            [UNRELATED_DEFECT],
            [COMPENSATION_REWORDED],
        ]
    )
    request = StartFeatureRequest.model_validate(_payload_for(repositories))
    state = _initial_feature_state(feature_id, request)
    # Generous, so nothing below can be mistaken for a stop caused by exhaustion.
    state.max_validation_retries = 8
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        **(
            {"integration_reviewer": cast(Any, integration_reviewer)}
            if integration_reviewer is not None
            else {}
        ),
    )
    state = await orchestrator.start(state, credentials=CREDENTIALS)
    return orchestrator, state, executor


def _open_conflict(state: FeatureWorkflowSnapshot) -> DesignConflictArtifact:
    """Return the one open design question this feature is holding."""
    conflicts = open_design_conflicts(state.artifacts)
    assert len(conflicts) == 1, [item.conflict_id for item in conflicts]
    return conflicts[0]


def _backend_triage(state: FeatureWorkflowSnapshot) -> dict[str, Any]:
    """Return the terminal triage the backend's final attempt recorded."""
    newest: dict[str, Any] = {}
    for artifact in state.artifacts:
        if (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.repository_id == "backend"
        ):
            newest = dict(artifact.metadata)
    return newest


@pytest.mark.asyncio
async def test_the_stop_writes_the_question_down_with_both_positions() -> None:
    """The stop is unchanged; what is new is that its question is answerable.

    Both positions are recorded as they were actually worded, because that pair is the whole
    content of the decision: the same thing was required, delivered, and let go. A conflict
    reduced to the current demand reads as a defect and is unanswerable.
    """
    _, state, executor = await _stopped_on_a_design_conflict("feature-verdict-recorded")

    assert executor.attempts == 3
    assert state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    conflict = _open_conflict(state)
    assert conflict.repository_id == "backend"
    assert conflict.fingerprint == fingerprint_for_text(COMPENSATION_RAISED)
    # Position one: what is being demanded now. Position two: how it was worded when it was
    # satisfied. Same defect, two sentences.
    assert conflict.demand == COMPENSATION_REWORDED
    assert conflict.satisfied_demand == COMPENSATION_RAISED
    assert conflict.demanded_by == "repository_review"
    assert conflict.satisfied_under == "repository_review"
    assert conflict.cross_authority is False
    assert conflict.status == "open"
    assert conflict.verdict is None
    # The artifact and the stopped attempt say the same thing, because both come from
    # `design_conflict_narrative`.
    assert conflict.question == _backend_triage(state)["operator_question"]
    assert conflict.attempts_spent == 3


@pytest.mark.asyncio
async def test_a_cross_authority_conflict_records_which_review_holds_which_position() -> None:
    """The 184 seam, as a question rather than only as a sentence.

    A person sent to arbitrate two reviews needs to know which review is on which side, and
    that must not be inferred from prose: the two shapes send a reader to two different
    decisions.
    """
    _, state, _ = await _stopped_on_a_design_conflict(
        "feature-verdict-cross-authority",
        script=[[COMPENSATION_RAISED], [], [UNRELATED_DEFECT]],
        integration_reviewer=DemandingIntegrationReviewer(
            repository_id="backend", recommended_fix=COMPENSATION_AS_INTEGRATION_FIX
        ),
    )

    conflict = _open_conflict(state)
    assert conflict.cross_authority is True
    assert conflict.demanded_by == "integration_review"
    assert conflict.satisfied_under == "repository_review"
    assert conflict.demand == COMPENSATION_AS_INTEGRATION_FIX
    assert conflict.satisfied_demand == COMPENSATION_RAISED


@pytest.mark.asyncio
async def test_a_verdict_reaches_the_next_attempt_as_an_instruction_it_may_not_argue() -> None:
    """The load-bearing half. An answered question is worth nothing until the engineer sees it.

    Three things are asserted together because each is useless without the others: another
    attempt ran, it was handed the decision in the decider's own words, and it was *not*
    handed the platform's question to the operator as though that were work to do.
    """
    orchestrator, state, executor = await _stopped_on_a_design_conflict(
        "feature-verdict-carried",
        # A fourth entry, so what the resumed attempt does is scripted rather than inherited
        # from the third: an unbounded script repeats its last verdict for ever, and the loop
        # would then be measured against a review that never changes its mind.
        script=[[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED], []],
    )
    conflict = _open_conflict(state)
    question = conflict.question

    decided = await orchestrator.answer_design_conflict(
        state,
        conflict_id=conflict.conflict_id,
        verdict="requirement_holds",
        decision=DECISION_TEXT,
        decided_by="akhilesh (via platform key)",
        additional_attempts=0,
        credentials=CREDENTIALS,
    )

    # A fourth attempt ran, which the stop by itself would never have allowed.
    assert executor.attempts == 4
    settled = settled_questions(decided.artifacts, repository_id="backend")
    assert [item.fingerprint for item in settled] == [conflict.fingerprint]
    assert settled[0].verdict == "requirement_holds"
    assert settled[0].decision == DECISION_TEXT
    assert settled[0].decided_by == "akhilesh (via platform key)"
    # What the resumed attempt was actually handed.
    feedback = executor.feedback[3]
    verdicts = [line for line in feedback if line.startswith("SETTLED BY A PERSON")]
    assert len(verdicts) == 1
    assert DECISION_TEXT in verdicts[0]
    assert COMPENSATION_REWORDED in verdicts[0]
    assert "This requirement stands." in verdicts[0]
    # The platform's question to an operator is not an instruction an engineer can act on, and
    # it lands in `blocking_issues` because that list is what reaches the console. Handing it
    # back as outstanding work would ask the engineer to fix the question.
    assert not any(question in line for line in feedback)


@pytest.mark.asyncio
async def test_the_same_recurrence_cannot_stop_the_loop_a_second_time() -> None:
    """Without this, answering the question changes nothing.

    The resumed attempt re-raises the very demand the verdict was about -- which is exactly
    what a review that has not changed its mind does. The stop must not fire again: it would
    end the workstream before the engineer was called, and put the same question in front of
    the same person.
    """
    orchestrator, state, executor = await _stopped_on_a_design_conflict(
        "feature-verdict-settled",
        script=[
            [COMPENSATION_RAISED],
            [UNRELATED_DEFECT],
            [COMPENSATION_REWORDED],
            # The resumed attempt meets the demand again, unchanged, which is what a review
            # that has not changed its mind does. This is the entry the test turns on.
            [COMPENSATION_REWORDED],
            # And the attempt after it delivers, which the loop only reaches if the settled
            # question did not end the workstream a second time.
            [],
        ],
    )
    conflict = _open_conflict(state)

    decided = await orchestrator.answer_design_conflict(
        state,
        conflict_id=conflict.conflict_id,
        verdict="requirement_holds",
        decision=DECISION_TEXT,
        decided_by="akhilesh",
        additional_attempts=0,
        credentials=CREDENTIALS,
    )

    # Five, not four: the fourth attempt met the settled demand and the loop carried on.
    assert executor.attempts == 5
    assert decided.child_workflows["backend"].status is ChildWorkflowStatus.APPROVED
    # No second question about one decision, and no second stop attributing it to a conflict.
    assert open_design_conflicts(decided.artifacts) == []
    assert _backend_triage(decided).get("terminal_cause") != "design_conflict"


@pytest.mark.asyncio
async def test_an_overruled_demand_is_not_also_handed_back_as_work_to_do() -> None:
    """A prompt cannot say both "fix this" and "a person decided this must not be built".

    `removal_holds` overrules the review, so the demand it made stops being outstanding work
    for the attempt the decision authorises. The line the engineer gets says what was decided
    and why, and the demand is not repeated beside it as a finding.
    """
    orchestrator, state, executor = await _stopped_on_a_design_conflict(
        "feature-verdict-overruled",
        script=[[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED], []],
    )
    conflict = _open_conflict(state)

    await orchestrator.answer_design_conflict(
        state,
        conflict_id=conflict.conflict_id,
        verdict="removal_holds",
        decision=OVERRULE_TEXT,
        decided_by="akhilesh",
        additional_attempts=0,
        credentials=CREDENTIALS,
    )

    feedback = executor.feedback[3]
    verdicts = [line for line in feedback if line.startswith("SETTLED BY A PERSON")]
    assert len(verdicts) == 1
    assert "does not stand and must not be implemented" in verdicts[0]
    assert OVERRULE_TEXT in verdicts[0]
    # The overruled demand appears once, inside the decision that overruled it, and nowhere
    # else -- not as an inherited finding the attempt is being asked to fix.
    assert [line for line in feedback if COMPENSATION_REWORDED in line] == verdicts


@pytest.mark.asyncio
async def test_a_granted_retry_is_told_the_stop_happened_without_being_asked_the_question() -> None:
    """The other exit from the stop, which used to hand the model the platform's own question.

    A retry grant decides nothing about the conflict -- it buys an attempt. So the stop's
    question is still unanswered when that attempt runs, and it was inherited verbatim: the
    coding model was handed "which authority's position holds for this feature?" in the list it
    reads as outstanding work, and the only way to act on a question is to answer it.

    What it gets instead is the same facts as a statement: why the last attempt ended, which
    review requires what, and the requirement itself. Nothing about the stop is hidden; the
    request to arbitrate is what stops travelling.

    One repository, because that is the shape the inheritance actually happens in. Where a
    sibling survived review, the contract validator's own finding about this repository is
    outstanding integration work, and a grant carries that instead -- the stopped attempt's
    blocking issues are read only when the integration review is asking this repository for
    nothing, which is every feature whose repositories all failed and every single-repository
    feature.
    """
    orchestrator, state, executor = await _stopped_on_a_design_conflict(
        "feature-grant-narrative",
        script=[[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED], []],
        repositories=("backend",),
    )
    question = _open_conflict(state).question

    granted = await orchestrator.grant_and_run_one_retry(
        state,
        repository_id="backend",
        additional_attempts=1,
        requested_by="akhilesh (via platform key)",
        reason="Buying one more attempt on the backend.",
        credentials=CREDENTIALS,
    )

    assert executor.attempts == 4
    assert granted.child_workflows["backend"].retry_grants[-1]["attempts"] == 1
    feedback = executor.feedback[3]
    # The question is gone -- as a whole sentence, as its interrogative tail, and as anything
    # at all this attempt could read as being asked of it.
    assert not any(question in line for line in feedback)
    assert not any("is the review wrong to require it" in line for line in feedback)
    assert not any(line.rstrip().endswith("?") for line in feedback)
    # And the fact is not gone with it. One line, stating what stopped the last attempt and
    # naming the requirement the argument is about.
    stated = [line for line in feedback if line.startswith("STOPPED OVER A DESIGN CONFLICT")]
    assert len(stated) == 1
    assert "not work assigned to this one" in stated[0]
    assert "already made and then removed once" in stated[0]
    assert COMPENSATION_REWORDED in stated[0]


@pytest.mark.asyncio
async def test_a_granted_retry_inherits_every_other_blocking_issue_byte_for_byte() -> None:
    """The findings are what the granted attempt exists to answer, so they travel untouched.

    This is the whole risk in restating anything: a filter that reads prose and decides what an
    engineer needs will eventually drop a review finding, and a dropped finding is an attempt
    spent on the wrong work. Identification is byte-equality with the recorded question, so
    everything else arrives exactly as the last attempt reported it, in the same order, with the
    statement of record standing where the question stood.
    """
    orchestrator, state, executor = await _stopped_on_a_design_conflict(
        "feature-grant-passthrough",
        script=[
            [COMPENSATION_RAISED],
            [UNRELATED_DEFECT],
            # The recurrence that stops the workstream, reported beside a defect that has
            # nothing to do with it -- the finding the granted attempt is being bought for.
            [COMPENSATION_REWORDED, UNRELATED_DEFECT],
            [],
        ],
        repositories=("backend",),
    )
    question = _open_conflict(state).question
    stopped = list(state.child_workflows["backend"].blocking_issues)
    assert question in stopped

    await orchestrator.grant_and_run_one_retry(
        state,
        repository_id="backend",
        additional_attempts=1,
        requested_by="akhilesh",
        reason="Buying one more attempt on the backend.",
        credentials=CREDENTIALS,
    )

    # What the attempt inherited leads the feedback it was handed, one line per stopped issue.
    inherited = executor.feedback[3][: len(stopped)]
    stated = [line for line in inherited if line.startswith("STOPPED OVER A DESIGN CONFLICT")]
    assert len(stated) == 1
    # Byte-identical, and in the order they were reported.
    assert [line for line in inherited if line not in stated] == [
        item for item in stopped if item != question
    ]
    # Restated in place, so nothing else moved.
    assert inherited.index(stated[0]) == stopped.index(question)


@pytest.mark.asyncio
async def test_answering_a_question_does_not_refund_the_attempts_the_stop_declined_to_spend() -> (
    None
):
    """49- Part B's accounting has to survive. Answering grants a decision, not a budget.

    The stop returned this workstream's remaining attempts unspent and said so, so the
    ordinary verdict buys nothing: no grant record, no raised ceiling. A grant is still
    possible, and when it happens it is recorded exactly as an operator's retry grant is.
    """
    orchestrator, state, _ = await _stopped_on_a_design_conflict(
        "feature-verdict-accounting",
        script=[[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED], []],
    )
    conflict = _open_conflict(state)
    before = state.child_workflows["backend"]

    decided = await orchestrator.answer_design_conflict(
        state,
        conflict_id=conflict.conflict_id,
        verdict="requirement_holds",
        decision=DECISION_TEXT,
        decided_by="akhilesh",
        additional_attempts=0,
        credentials=CREDENTIALS,
    )

    after = decided.child_workflows["backend"]
    assert after.granted_extra_attempts == before.granted_extra_attempts
    # No grant, no grant record: a row saying nobody bought anything reads in an audit as an
    # override that happened.
    assert after.retry_grants == before.retry_grants
    # And the refusal that quoted the stop is gone, because the stop has been answered.
    assert after.retry_refusal_reason is None


@pytest.mark.asyncio
async def test_a_workstream_with_nothing_left_to_spend_must_be_granted_an_attempt() -> None:
    """The one case a verdict has to buy something, said out loud rather than topped up.

    Silently granting an attempt here is what would make an answered question look like a
    refund. The refusal names the condition, and the same call with a grant is accepted and
    recorded.
    """
    orchestrator, state, _ = await _stopped_on_a_design_conflict("feature-verdict-exhausted")
    conflict = _open_conflict(state)
    child = state.child_workflows["backend"]
    # Spend the classification budget this workstream is judged against.
    state.child_workflows["backend"] = child.model_copy(
        update={"validation_retry_count": state.max_validation_retries}
    )

    with pytest.raises(Exception, match="no attempts left"):
        await orchestrator.answer_design_conflict(
            state,
            conflict_id=conflict.conflict_id,
            verdict="requirement_holds",
            decision=DECISION_TEXT,
            decided_by="akhilesh",
            additional_attempts=0,
            credentials=CREDENTIALS,
        )

    decided = await orchestrator.answer_design_conflict(
        state,
        conflict_id=conflict.conflict_id,
        verdict="requirement_holds",
        decision=DECISION_TEXT,
        decided_by="akhilesh",
        additional_attempts=1,
        credentials=CREDENTIALS,
    )

    granted = decided.child_workflows["backend"]
    assert granted.granted_extra_attempts == 1
    assert len(granted.retry_grants) == 1
    assert granted.retry_grants[0]["granted_by"] == "akhilesh"
    assert conflict.conflict_id in granted.retry_grants[0]["reason"]


@pytest.mark.asyncio
async def test_one_question_is_decided_once() -> None:
    """A second answer would be a different decision recorded over one already acted on."""
    orchestrator, state, _ = await _stopped_on_a_design_conflict("feature-verdict-once")
    conflict = _open_conflict(state)

    decided = await orchestrator.answer_design_conflict(
        state,
        conflict_id=conflict.conflict_id,
        verdict="requirement_holds",
        decision=DECISION_TEXT,
        decided_by="akhilesh",
        additional_attempts=0,
        credentials=CREDENTIALS,
    )

    with pytest.raises(Exception, match="already been decided"):
        await orchestrator.answer_design_conflict(
            decided,
            conflict_id=conflict.conflict_id,
            verdict="removal_holds",
            decision=OVERRULE_TEXT,
            decided_by="somebody else",
            additional_attempts=0,
            credentials=CREDENTIALS,
        )


@pytest.mark.asyncio
async def test_a_verdict_without_its_reasoning_is_refused() -> None:
    """The decision is the invariant. An unexplained verdict is not an instruction."""
    orchestrator, state, _ = await _stopped_on_a_design_conflict("feature-verdict-unexplained")
    conflict = _open_conflict(state)

    with pytest.raises(Exception, match="reasoning"):
        await orchestrator.answer_design_conflict(
            state,
            conflict_id=conflict.conflict_id,
            verdict="requirement_holds",
            decision="   ",
            decided_by="akhilesh",
            additional_attempts=0,
            credentials=CREDENTIALS,
        )

    with pytest.raises(Exception, match="not a design verdict"):
        await orchestrator.answer_design_conflict(
            state,
            conflict_id=conflict.conflict_id,
            verdict="the review is being unreasonable",
            decision=DECISION_TEXT,
            decided_by="akhilesh",
            additional_attempts=0,
            credentials=CREDENTIALS,
        )


@pytest.mark.asyncio
async def test_the_answer_is_accepted_and_queued_rather_than_executed_in_the_request(
    tmp_path: Path,
) -> None:
    """Everything a verdict frees is a real attempt, so none of it may run on a request thread.

    The same rule resume and a retry grant follow. What is asserted is the effect: the request
    returns with the feature untouched and an entry on the queue carrying every argument the
    worker needs -- including the verdict, so the decision cannot reach the attempt late.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'queued-verdict.db'}")
    await database.create_schema()
    try:
        executor = ScriptedFindingsExecutor(
            [[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED]]
        )
        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(child_executor=cast(Any, executor)),
        )
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="queued-verdict-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        state = (await store.get_record("feature-login")).state
        conflict = _open_conflict(state)

        # One attempt granted, because this feature runs on the deployment's default budget
        # and the reversal was recognised on the attempt that spent the last of it. The refusal
        # for the ungranted case is asserted at the orchestrator, where the accounting lives.
        await store.answer_design_conflict(
            "feature-login",
            conflict.conflict_id,
            verdict="requirement_holds",
            decision=DECISION_TEXT,
            decided_by="akhilesh (via platform key)",
            additional_attempts=1,
            credentials=CREDENTIALS,
        )

        # Nothing ran: the question is still open and no further attempt was made.
        attempts_when_accepted = executor.attempts
        assert open_design_conflicts((await store.get_record("feature-login")).state.artifacts)
        claimed = await store.queue.claim(owner="worker", lease_seconds=60)
        assert claimed is not None
        assert claimed.intent == "answer_design_verdict"
        assert claimed.payload == {
            "conflict_id": conflict.conflict_id,
            "repository_id": "backend",
            "verdict": "requirement_holds",
            "decision": DECISION_TEXT,
            "decided_by": "akhilesh (via platform key)",
            "additional_attempts": 1,
        }
        # The identity the worker resolves stored credentials against stays the submission's,
        # not the audit sentence: crossing the two killed AB-Feature-111's granted retry.
        assert claimed.requested_by != "akhilesh (via platform key)"
        assert attempts_when_accepted == 3
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_an_unknown_or_decided_question_is_refused_before_anything_is_queued(
    tmp_path: Path,
) -> None:
    """A refusal is an answer, and it must cost nothing -- no queue entry, no workspace."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'refused-verdict.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="refused-verdict-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        with pytest.raises(Exception, match="design conflict not found"):
            await store.answer_design_conflict(
                "feature-login",
                "conflict-backend-nothing",
                verdict="requirement_holds",
                decision=DECISION_TEXT,
                decided_by="akhilesh",
                additional_attempts=0,
                credentials=CREDENTIALS,
            )

        assert await store.queue.claim(owner="worker", lease_seconds=60) is None
    finally:
        await database.dispose()


def _app_with_a_scripted_backend() -> Any:
    """Return an application whose backend will re-litigate one demand.

    The runner seam the clarification-state tests already use. What is being tested here is
    the read model and the endpoint, so the scripted executor exists only to reach the stop.
    """
    app = create_app(platform_api_key=API_KEY)
    app.state.feature_control_plane._mock_runner = FeatureWorkflowOrchestrator(  # noqa: SLF001
        child_executor=cast(
            Any,
            ScriptedFindingsExecutor(
                [[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED], []]
            ),
        )
    )
    return app


@pytest.mark.asyncio
async def test_the_clarification_surface_serves_the_design_question_as_its_own_state() -> None:
    """One surface, one answer to "is anything waiting on me".

    This lands on the clarification read rather than an endpoint of its own because that is the
    read a client already polls to find out what is waiting on the user; a second one would make
    the question two questions with two answers that could disagree.

    `awaiting_answers` stays false, deliberately. That flag is the technical-PRD contract: it
    gates the answer validator and it makes the feature advertise ANSWER_CLARIFICATION, whose
    answers reopen planning. A design verdict is neither of those things.
    """
    app = _app_with_a_scripted_backend()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=API_AUTH, json=feature_payload())
        await settle(app)
        clarification = await client.get("/features/feature-login/clarification", headers=API_AUTH)
        feature = await client.get("/features/feature-login", headers=API_AUTH)
        workstreams = await client.get("/features/feature-login/workstreams", headers=API_AUTH)

    body = clarification.json()
    assert body["clarification_state"] == "awaiting_design_verdict"
    assert body["awaiting_answers"] is False
    assert body["questions"] == []
    conflicts = body["design_conflicts"]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["repository_id"] == "backend"
    # Both positions, each naming the review that holds it, so the client renders the decision
    # rather than working out which side is which from the prose.
    assert conflict["demanded"]["statement"] == COMPENSATION_REWORDED
    assert conflict["demanded"]["authority"] == "repository_review"
    assert conflict["satisfied"]["statement"] == COMPENSATION_RAISED
    assert conflict["cross_authority"] is False
    assert conflict["question"]
    assert conflict["evidence"]
    # Published, never inferred by the browser from the counters beside it.
    assert conflict["answerable"] is True
    assert "attempts_remaining" in conflict
    # And the control is advertised at both levels, so neither the feature card nor the
    # repository row is the only place a reader could have found it.
    assert "ANSWER_DESIGN_VERDICT" in feature.json()["available_actions"]
    backend = next(
        item for item in workstreams.json()["workstreams"] if item["repository_id"] == "backend"
    )
    assert "ANSWER_DESIGN_VERDICT" in backend["available_actions"]
    # The payload the client's own tests are written against, saved from this response rather
    # than hand-written: a fixture somebody typed is a fixture that agrees with the client and
    # not with the server, which has already cost this client a dozen dropped fields.
    _CLIENT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    _CLIENT_FIXTURE.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_the_endpoint_records_the_decision_and_answers_before_running_anything() -> None:
    """The whole point of the exit, end to end, through the surface a person actually uses.

    Accepted and queued: the response comes back with the decision recorded as pending work
    rather than with an attempt already run inside the request. Then the worker acts on it, and
    the question is closed with the verdict, the words and the author against it.
    """
    app = _app_with_a_scripted_backend()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=API_AUTH, json=feature_payload())
        await settle(app)
        conflict_id = (
            await client.get("/features/feature-login/clarification", headers=API_AUTH)
        ).json()["design_conflicts"][0]["conflict_id"]
        unexplained = await client.post(
            f"/features/feature-login/design-conflicts/{conflict_id}/answer",
            headers=API_AUTH,
            json={"verdict": "requirement_holds", "decision": ""},
        )
        invented = await client.post(
            f"/features/feature-login/design-conflicts/{conflict_id}/answer",
            headers=API_AUTH,
            json={"verdict": "the review is wrong", "decision": DECISION_TEXT},
        )
        unknown = await client.post(
            "/features/feature-login/design-conflicts/conflict-backend-nothing/answer",
            headers=API_AUTH,
            json={"verdict": "requirement_holds", "decision": DECISION_TEXT},
        )
        accepted = await client.post(
            f"/features/feature-login/design-conflicts/{conflict_id}/answer",
            headers=API_AUTH,
            json={
                "verdict": "requirement_holds",
                "decision": DECISION_TEXT,
                "additional_attempts": 1,
            },
        )
        await settle(app)
        artifacts = await client.get(
            "/features/feature-login/artifacts",
            headers=API_AUTH,
            params={"artifact_type": "design_conflict"},
        )

    # A verdict with no reasoning is refused by the request model: it is the invariant the next
    # attempt is bound by, and an unexplained one is not an instruction.
    assert unexplained.status_code == 422
    assert invented.status_code == 422
    assert unknown.status_code == 404
    assert accepted.status_code == 200, accepted.text
    # The durable action the client follows after losing the response.
    assert accepted.headers["X-Feature-Action-ID"]
    decided = artifacts.json()["artifacts"][-1]["payload"]
    assert decided["status"] == "answered"
    assert decided["verdict"] == "requirement_holds"
    assert decided["decision"] == DECISION_TEXT
    # Who decided it is the authenticated identity, not a name in the request body.
    assert decided["decided_by"]
    assert "platform" in decided["decided_by"]


def _resolved(issue: str) -> ResolvedIssue:
    """One ledger entry, for the unit rules that need no lineage."""
    return ResolvedIssue(
        fingerprint=fingerprint_for_text(issue),
        issue=issue,
        authority=IssueAuthority.REPOSITORY_REVIEW,
        resolved_at=1,
    )


def _settled(issue: str, verdict: str) -> SettledQuestion:
    """One decided question, as the retry machinery reads it."""
    return SettledQuestion(
        fingerprint=fingerprint_for_text(issue),
        verdict=verdict,
        decision=DECISION_TEXT if verdict == "requirement_holds" else OVERRULE_TEXT,
        demand=issue,
        authority=IssueAuthority.REPOSITORY_REVIEW,
        decided_by="akhilesh",
    )


def test_a_settled_question_is_not_reported_as_a_recurrence() -> None:
    """The exclusion the answerable stop rests on, asserted on its own.

    Same inputs, one difference: a person has decided this. The unsettled call still stops the
    loop, which is 49- Part B's behaviour and stays exactly as it was.
    """
    ledger = [_resolved(COMPENSATION_RAISED)]
    demanded = {IssueAuthority.REPOSITORY_REVIEW: [COMPENSATION_REWORDED]}

    assert len(recurring_resolved_issues(ledger, demanded=demanded)) == 1
    assert (
        recurring_resolved_issues(
            ledger, demanded=demanded, settled={fingerprint_for_text(COMPENSATION_RAISED)}
        )
        == []
    )


def test_a_decided_question_is_stated_once_and_never_as_a_derived_invariant() -> None:
    """One record, one line. Two would say the same thing twice with less authority behind it.

    And under `removal_holds` they would contradict each other outright: the derived line says
    an earlier attempt delivered this and must not undo it, which is the opposite of what the
    person decided.
    """
    ledger = [_resolved(COMPENSATION_RAISED)]
    fingerprints = {fingerprint_for_text(COMPENSATION_RAISED)}

    derived = satisfied_invariant_lines(ledger, settled=fingerprints)
    assert derived == []
    # Without the exclusion the derived line is exactly what would have doubled up.
    assert len(satisfied_invariant_lines(ledger)) == 1


def test_a_verdict_is_stated_even_while_the_demand_it_settles_is_outstanding() -> None:
    """The one exemption, and the reason the whole exit works.

    An ordinary satisfied requirement that is also being demanded is a contradiction kept out
    of a prompt. A verdict *about* a demand that is being made is the one sentence that
    resolves it -- so it is rendered from the decision rather than filtered against the demand.
    """
    lines = settled_question_lines([_settled(COMPENSATION_RAISED, "requirement_holds")])

    assert len(lines) == 1
    assert lines[0].startswith("SETTLED BY A PERSON")
    assert COMPENSATION_RAISED in lines[0]
    assert DECISION_TEXT in lines[0]
    assert "not open to review by this attempt" in lines[0]

    overruled = settled_question_lines([_settled(COMPENSATION_RAISED, "removal_holds")])
    assert "does not stand and must not be implemented" in overruled[0]


def test_a_conflict_records_which_review_is_on_which_side() -> None:
    """`cross_authority` is a fact about the two positions, not a reading of the prose."""
    resolved = _resolved(COMPENSATION_RAISED)
    same = RecurringIssue(
        issue=COMPENSATION_REWORDED,
        authority=IssueAuthority.REPOSITORY_REVIEW,
        resolved=resolved,
    )
    across = RecurringIssue(
        issue=COMPENSATION_AS_INTEGRATION_FIX,
        authority=IssueAuthority.INTEGRATION_REVIEW,
        resolved=resolved,
    )

    assert same.cross_authority is False
    assert across.cross_authority is True


@pytest.mark.asyncio
async def test_a_decided_question_survives_a_state_round_trip(tmp_path: Path) -> None:
    """The verdict is durable, or the attempt it authorises runs without it.

    A design conflict is stored inside parent state like every other artifact, so this is the
    trap a new artifact type walks into: a union member the loader cannot rebuild comes back as
    something else, and `settled_questions` then finds nothing at exactly the point the
    argument resumes.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'verdict-roundtrip.db'}")
    await database.create_schema()
    try:
        executor = ScriptedFindingsExecutor(
            [[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED]]
        )
        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(child_executor=cast(Any, executor)),
        )
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="verdict-roundtrip-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        state = (await store.get_record("feature-login")).state
        conflict = _open_conflict(state)
        await store.answer_design_conflict(
            "feature-login",
            conflict.conflict_id,
            verdict="requirement_holds",
            decision=DECISION_TEXT,
            decided_by="akhilesh",
            additional_attempts=1,
            credentials=CREDENTIALS,
        )
        await drain_feature_queue(store)

        reloaded = (await store.get_record("feature-login")).state
        settled = settled_questions(reloaded.artifacts, repository_id="backend")
        assert [item.decision for item in settled] == [DECISION_TEXT]
        assert open_design_conflicts(reloaded.artifacts) == []
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_refused_verdict_reaches_the_client_as_a_refusal(tmp_path: Path) -> None:
    """An unrecognised verdict is a wrong request, not a broken feature.

    Letting it through to the artifact would mark the feature `failed_requires_human` for
    somebody's typo, which is the shape the retry precondition was moved forward to prevent.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'bad-verdict.db'}")
    await database.create_schema()
    try:
        executor = ScriptedFindingsExecutor(
            [[COMPENSATION_RAISED], [UNRELATED_DEFECT], [COMPENSATION_REWORDED]]
        )
        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(child_executor=cast(Any, executor)),
        )
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="bad-verdict-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        conflict = _open_conflict((await store.get_record("feature-login")).state)

        with pytest.raises(WorkflowConflictError, match="not a design verdict"):
            await store.answer_design_conflict(
                "feature-login",
                conflict.conflict_id,
                verdict="whatever",
                decision=DECISION_TEXT,
                decided_by="akhilesh",
                additional_attempts=0,
                credentials=CREDENTIALS,
            )

        record = await store.get_record("feature-login")
        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert await store.queue.claim(owner="worker", lease_seconds=60) is None
    finally:
        await database.dispose()
