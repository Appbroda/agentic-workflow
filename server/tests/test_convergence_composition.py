"""86- Part C. Five lineages, end to end, with the narrowed marker and the new evidence.

Parts A and B are individually conservative and jointly re-arm five convergence rules against
evidence that did not previously exist. That combination is exactly where a false stop would
come from, and no test of either part alone would catch it -- so this module replays realistic
lineages through the whole retry loop and asserts, for each, that the workstream ends the way
it should.

The failure mode being guarded against has a name and a cost. **Feature -052b lost both
workstreams two attempts in because the reachability gate said the same true thing twice while
the attempts around it were changing.** That is why `deterministic_gate` exists at all, and
lineages 3 and 4 below are the two shapes it protects: an inspection repeating a true sentence
is the tool being consistent, and nothing in 86- may read it as a workstream stuck.

What separates the two halves is the *check that spoke*, not the gate it came from. A lint
rule, a formatting rule, a type error and a wiring finding all repeat for as long as their
condition holds. A failing test repeating is a change that has not moved.

Lineage 5 asserts the per-file consecutive counts **directly**, walking the recorded evidence
rather than asking the rule for its own answer: 49-C's own history includes a test that passed
because a counter reset for the wrong reason, and re-stating the rule's output would not have
caught it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from tests.test_unchanged_failure import (
    _CSV_REPORTS,
    _SERVICE_REPORTS,
    BULK_APP_SERVICE,
    CSV_COMMAND,
    CSV_TEST,
    CSV_UTIL,
    SERVICE_COMMAND,
    SERVICE_TEST,
    GateRejectionExecutor,
    _backend_attempts,
    _consecutive_runs,
    _gate_rejection,
    _run,
    _scoped_run,
    _workspace,
)
from tools.reachability import REACHABILITY_DIAGNOSTIC_PREFIX
from tools.source_formatting import SourceValidationError
from tools.unchanged_failure import failing_evidence_files
from workflows.feature_workflow import (
    ChildExecution,
    MockChildWorkstreamExecutor,
    source_rejection_disposition,
)

# The two sentences lineages 3 and 4 repeat, byte-identically, across their attempts. Both are
# deterministic inspections speaking about a condition that is still true, and neither says
# anything about whether the change is converging.
_REACHABILITY_SENTENCE = (
    f"{REACHABILITY_DIAGNOSTIC_PREFIX}: src/routes/status_formatter.js is not referred to by "
    "any production file in this repository."
)
_LINT_SENTENCE = (
    "Pre-commit source validation failed (tool=eslint): "
    "src/routes/status.js:14:7  error  'formatStatus' is not defined  no-undef"
)


class InspectionRejectionExecutor:
    """Fail the backend at the commit gate on a deterministic inspection's own sentence.

    Separate from `GateRejectionExecutor`, which raises and lets the workflow's handler build
    the record, for one reason: these lineages repeat their diagnostics *byte-identically*, so
    `meaningful_progress` needs the other half of its question answered -- did the source
    change -- and a raised rejection carries no code completion and therefore no changed
    files. Live, a gate-rejected attempt has written real files; here the fingerprints are
    scripted so that the only thing left to stop these lineages is the rule under test. This
    is the same shape `test_a_lint_rejection_holding_earlier_files_is_not_a_reproduction`
    uses, and for the same reason.

    The markers are **not** scripted. They come from `source_rejection_disposition`, given the
    same `SourceValidationError` the gate raises, because which checks keep the exemption is
    the whole of Part A and a fixture that stamped the flag itself would test nothing.
    """

    def __init__(self, diagnostics: Sequence[str], *, failures: int) -> None:
        """Repeat one rejection for `failures` attempts, then let the delegate approve."""
        self._delegate = MockChildWorkstreamExecutor()
        self._diagnostics = tuple(diagnostics)
        self._failures = failures
        self.attempts = 0
        self.feedback: list[list[str]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return the scripted rejection for this backend attempt, approving every sibling."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        self.feedback.append(list(kwargs["feedback"]))
        self.attempts += 1
        if self.attempts > self._failures:
            return execution
        classification, metadata = source_rejection_disposition(
            SourceValidationError(self._diagnostics)
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": list(self._diagnostics),
                    # Whatever the gate keeps for its own checks, which for an inspection is
                    # nothing: neither the linter nor the wiring check produces a structured
                    # result this platform retains.
                    "current_validation_results": [],
                    "pull_request_readiness": False,
                    "failure_classification": classification,
                    "metadata": {**execution.result.metadata, **metadata},
                    # Real, distinct source every attempt, so the loop-prevention guards are
                    # satisfied and the convergence rules are the only thing under test.
                    "production_files_changed": ["src/routes/status.js"],
                    "production_diff_fingerprint": f"production-{self.attempts}",
                    "test_diff_fingerprint": f"tests-{self.attempts}",
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


def _strategy_lines(feedback: Sequence[Sequence[str]]) -> list[str]:
    """Every UNCHANGED FAILURE line any attempt in this lineage was given."""
    return [line for lines in feedback for line in lines if line.startswith("UNCHANGED FAILURE")]


# --------------------------------------------------------------------------------------
# Lineage 1 -- the 218 lineage stops
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l1_the_218_lineage_is_stopped_as_non_converging(tmp_path: Path) -> None:
    """L1. Four commit-gate rejections on one suite, and the loop stops at the third.

    AB-Feature-218's backend failed `bulk-create-apps.service.test.js` on attempts 2, 3, 4
    and 5 with one substance -- row-validation and duplicate scenarios returning
    `BULK_APP_TRANSACTION_FAILED (500)` where the contract requires 400/409. Two of those
    were commit-gate rejections and two reached review. No rule fired at any point, and the
    workstream was ended at attempt 6 by an unrelated provider outage with
    `validation_retry_count` at 5 of 8: no budget was exhausted and no guard stopped it.

    Scripted as four gate rejections because that is the shape with no guard at all, and it
    is the one this item exists for. Each attempt is worded differently -- 218's own failing
    assertion moved from line 275 to line 291 between attempts -- so the token signature
    changes every round and neither the repeat rule nor the non-converging refusal can be
    what stops this. The per-file count is.

    Against Part B alone this lineage runs its full length and is approved on the fifth
    attempt: the evidence exists, and every rule that would read it is switched off.
    """
    workspace = _workspace(tmp_path)
    runs = [_scoped_run(workspace, report) for report in _SERVICE_REPORTS]
    executor = GateRejectionExecutor([*[_gate_rejection(run) for run in runs], [], []])

    result = await _run("composition-218", executor)

    # Stopped at the third attempt -- the third consecutive failure on one suite -- rather
    # than after a fourth coding call. The fifth entry above would have been approved, and
    # it is scripted precisely so its absence is provable.
    assert executor.attempts == 3
    lineage = _backend_attempts(result)
    assert len(lineage) == 3
    # The premise, asserted rather than assumed: the count really did reach three on the
    # suite, and it did so across attempts the commit gate stopped.
    assert _consecutive_runs(lineage)[SERVICE_TEST] == 3
    assert all(item.metadata.get("source_validation_rejected") for item in lineage)
    # The ×2 strategy line came first, so the attempt that was stopped had already been told
    # in so many words which file was not moving and what the previous repair touched.
    strategy = _strategy_lines(executor.feedback)
    assert len(strategy) == 1
    assert SERVICE_TEST in strategy[0]

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    assert "stopped producing new information" in (child.retry_refusal_reason or "")
    # The attempts it did not spend were returned rather than burned: the budget is eight.
    assert child.validation_retry_count < 8
    # An honest operator question: it names the command a person can run and the file it
    # keeps naming, and it does not claim the repository is broken.
    triage = dict(lineage[-1].metadata)
    question = triage["operator_question"]
    assert SERVICE_TEST in question
    assert SERVICE_COMMAND in question
    assert "3 consecutive attempts" in question
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


# --------------------------------------------------------------------------------------
# Lineage 2 -- a converging lineage is not stopped
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_a_converging_lineage_is_not_stopped(tmp_path: Path) -> None:
    """L2. Suite X, then suite Y, then green. Each file's run is its own.

    The narrowing in Part A arms these rules on exactly the attempts a failing test stopped,
    which is where a false stop would land. A lineage that is working through its defects one
    at a time must reach its approval: the first suite recovered, and the file that failed
    after it starts at one rather than inheriting the count of the file before it.
    """
    workspace = _workspace(tmp_path)
    executor = GateRejectionExecutor(
        [
            _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[0])),
            _gate_rejection(
                _scoped_run(workspace, _CSV_REPORTS[0], command=CSV_COMMAND),
            ),
            [],
        ]
    )

    result = await _run("composition-converging", executor)

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    lineage = _backend_attempts(result)
    # Counted separately, and the recovered file reset only itself: the second attempt's
    # failing file is at one, and the first attempt's does not appear in its run at all.
    counts = _consecutive_runs(lineage[:2])
    assert counts == {CSV_TEST: 1, CSV_UTIL: 1}
    assert SERVICE_TEST not in counts
    # Nothing was told its previous repair changed nothing, because none of them did.
    assert _strategy_lines(executor.feedback) == []


# --------------------------------------------------------------------------------------
# Lineage 3 -- the -052b lineage is not stopped
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l3_the_052b_lineage_is_not_stopped(tmp_path: Path) -> None:
    """L3. A wiring finding repeating one true sentence still gets its retries.

    Feature -052b lost both workstreams two attempts in because the reachability gate said
    the same true thing twice while the attempts around it were changing. That is the entire
    reason `deterministic_gate` exists, and Part A may not narrow it here: a wiring finding is
    a deterministic inspection whose sentence is fixed for as long as the module is
    unreferenced, and no model produced it.

    Byte-identical diagnostics on both attempts, on purpose. That is what arms the repeat
    rules, and the exemption is the only thing standing between this lineage and a stop two
    attempts in.
    """
    del tmp_path
    executor = InspectionRejectionExecutor([_REACHABILITY_SENTENCE], failures=2)

    result = await _run("composition-052b", executor)

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    lineage = _backend_attempts(result)
    # The premise: the exemption really is what carried it. Both rejections wore the marker,
    # and both said the same thing.
    assert [item.metadata.get("deterministic_gate") for item in lineage[:2]] == [True, True]
    assert lineage[0].blocking_issues == lineage[1].blocking_issues
    assert _strategy_lines(executor.feedback) == []


# --------------------------------------------------------------------------------------
# Lineage 4 -- the lint lineage is not stopped by the new evidence
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l4_the_lint_lineage_is_not_stopped_by_the_new_evidence(tmp_path: Path) -> None:
    """L4. A linter repeating itself is the tool being consistent.

    A linter emits the same sentence for as long as a rule is broken, so its repetition says
    nothing about whether the change is converging -- and rep-a's frontend was stopped three
    attempts into eight for exactly that. Part A leaves it exempt, and Part B gives it no new
    evidence to be stopped by: the lint gate keeps no structured result, so its attempts
    record none.

    The second half is the one worth asserting. It is what makes "Part B's evidence cannot
    arm a rule Part A left disabled" a fact about the record rather than a claim about the
    ordering of two conditions.
    """
    del tmp_path
    executor = InspectionRejectionExecutor([_LINT_SENTENCE], failures=2)

    result = await _run("composition-lint", executor)

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    lineage = _backend_attempts(result)
    assert [item.metadata.get("deterministic_gate") for item in lineage[:2]] == [True, True]
    # No new evidence reached the record, so there is nothing here for the coarse counter to
    # read even if the exemption were ever lifted from this path.
    assert all(item.current_validation_results == [] for item in lineage[:2])
    assert all(failing_evidence_files(item) is None for item in lineage[:2])
    assert _strategy_lines(executor.feedback) == []


# --------------------------------------------------------------------------------------
# Lineage 5 -- the alternation trap
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l5_two_files_at_two_each_are_not_one_file_at_four(tmp_path: Path) -> None:
    """L5. X, X, Y, Y: four failing attempts, and no file's run reaches three.

    The trap this exists for. Four consecutive failing attempts is the number a lineage-length
    counter would read, and it would stop this workstream; the rule counts *per file*, so each
    of the two files is at two -- owed the strategy-change line and nothing more -- and the
    fifth attempt is reached and approved.

    The counts are asserted **directly**, walking the recorded evidence rather than asking
    `unchanged_failure_run` for its own answer, because the failure being guarded against is a
    counter that reset for the wrong reason and 49-C's history has one. The control is in the
    same assertion: the same helper reads three on lineage 1's evidence, so a helper that
    could never see a run would not pass here.
    """
    workspace = _workspace(tmp_path)
    executor = GateRejectionExecutor(
        [
            _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[0])),
            _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[1])),
            _gate_rejection(_scoped_run(workspace, _CSV_REPORTS[0], command=CSV_COMMAND)),
            _gate_rejection(_scoped_run(workspace, _CSV_REPORTS[1], command=CSV_COMMAND)),
            [],
        ]
    )

    result = await _run("composition-alternation", executor)

    # Five attempts: four failures and the approval after them. Reaching the fifth is the
    # whole assertion -- a stop at the fourth is what a lineage-length count would have done.
    assert executor.attempts == 5
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    lineage = _backend_attempts(result)

    # The per-file counts, directly. Two files at two each, at the last failing attempt.
    assert _consecutive_runs(lineage[:4]) == {CSV_TEST: 2, CSV_UTIL: 2}
    # And the pair before them, so "two each" is measured on both halves rather than assumed
    # of the half the sentence happens to name.
    assert _consecutive_runs(lineage[:2]) == {SERVICE_TEST: 2, BULK_APP_SERVICE: 2}
    # No file ever reached three at any point in the lineage, which is the threshold, so the
    # loop was never owed a stop. Checked at every prefix rather than only at the end: a run
    # that peaked in the middle and recovered would be invisible to the last measurement.
    peaks = [
        max(_consecutive_runs(lineage[:length]).values(), default=0)
        for length in range(1, len(lineage) + 1)
    ]
    assert max(peaks) == 2, peaks


@pytest.mark.asyncio
async def test_l5c_each_pair_is_told_its_previous_repair_changed_nothing(tmp_path: Path) -> None:
    """L5c. The same lineage, from the rule's own side: two ×2 lines and no third.

    Split from L5 on purpose. L5's claim is an *invariant* -- this workstream is not stopped,
    and each file's run is two -- and it holds whether or not the rule is armed on these
    attempts. This one is the claim that Part A's narrowing is what arms it: before the
    narrowing every attempt here is exempt, `unchanged_failure_run` is never consulted, and
    the strategy-change line the second failure on each file is owed is never issued.

    Two lines, not four and not one. Four would mean the count is running per lineage; one
    would mean a recovered file took its sibling's count with it.
    """
    workspace = _workspace(tmp_path)
    executor = GateRejectionExecutor(
        [
            _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[0])),
            _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[1])),
            _gate_rejection(_scoped_run(workspace, _CSV_REPORTS[0], command=CSV_COMMAND)),
            _gate_rejection(_scoped_run(workspace, _CSV_REPORTS[1], command=CSV_COMMAND)),
            [],
        ]
    )

    result = await _run("composition-alternation-lines", executor)

    assert executor.attempts == 5
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    strategy = _strategy_lines(executor.feedback)
    assert len(strategy) == 2, strategy
    assert SERVICE_TEST in strategy[0]
    assert CSV_TEST in strategy[1]


@pytest.mark.asyncio
async def test_l5b_an_alternation_never_accumulates_at_all(tmp_path: Path) -> None:
    """L5b. X, Y, X, Y: every file's run is one, because none of them is consecutive.

    The other alternation, and the reason the count is per file *and* per consecutive run.
    Coming back to a state this workstream already submitted is real and has its own owner --
    `_revisits_an_earlier_attempt`, which compares diff bytes against every prior attempt.
    This counter answers the narrower question "did the last repair change anything at all",
    and an alternation answers it with yes every round.
    """
    workspace = _workspace(tmp_path)
    executor = GateRejectionExecutor(
        [
            _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[0])),
            _gate_rejection(_scoped_run(workspace, _CSV_REPORTS[0], command=CSV_COMMAND)),
            _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[1])),
            _gate_rejection(_scoped_run(workspace, _CSV_REPORTS[1], command=CSV_COMMAND)),
            [],
        ]
    )

    result = await _run("composition-alternation-abab", executor)

    assert executor.attempts == 5
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    lineage = _backend_attempts(result)
    assert _consecutive_runs(lineage[:4]) == {CSV_TEST: 1, CSV_UTIL: 1}
    assert _consecutive_runs(lineage[:3]) == {SERVICE_TEST: 1, BULK_APP_SERVICE: 1}
    assert _strategy_lines(executor.feedback) == []
