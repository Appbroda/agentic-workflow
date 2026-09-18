"""Coverage for the per-file failure count and the backstop it feeds.

AB-Feature-183's backend failed `bulk-create-apps.test.js` on attempts 1, 2, 3 and 4 and
nothing noticed, because every guard the loop had reads the *wording* of a failure. Sixteen
tests failed, then ten, with different excerpts and different result codes, so
`diagnostic_signature`'s token set moved every round while the failing file did not move at
all. The run was cancelled mid-attempt-4 against a ceiling of eight.

AB-Feature-194 then walked through the backstop 183 bought. Its key was the pair
`(failing command, files named)`, and `scoped_test_command` rewrites its file list from each
attempt's changed files -- so the command string moved most rounds while the failing test file
still did not, consecutive equality never held, and neither the x2 line nor the x3 stop ever
armed across six attempts. The count is now per file, and the last test group below is 194's
recorded sequence replayed against it.

The tests below are built on real evidence rather than on hand-written summaries: the
reference section is produced by the same writer the validation path uses, over real files in
a real temporary workspace, so a change to that format fails these tests instead of silently
making the evidence unreadable. And they assert effects -- what the next engineer was
actually told, and what a person reading the stopped workstream is sent to run.

**No fixture here asserts a disposition the platform owns** (87- Part D). A scripted
commit-gate rejection is scripted as the *rejection* -- its diagnostics and its terminal
outcome -- and `source_rejection_disposition` decides the classification and the markers.
`SequencedOutcomeExecutor` used to stamp `deterministic_gate: True` and an empty validation
list onto a guard-shaped stop, which left the two 194-group tests unable to observe either
half of 86-: not Part A, because the disposition was bypassed, and not Part B, because the
evidence was emptied. They passed against a state the platform had stopped producing, which
is the worst way for a test to pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import ChildWorkflowResultArtifact
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import FeatureWorkflowSnapshot
from tests.test_feature_workflow import feature_payload
from tools.retry_strategy import diagnostic_signature
from tools.scoped_tests import (
    ASSERTION_GUARD_OUTCOME,
    assertion_guard_diagnostic,
    scoped_test_diagnostic,
)
from tools.source_formatting import SourceValidationError
from tools.unchanged_failure import (
    _assertion_guard_evidence,
    _failing_required_evidence,
    coarse_failure_evidence,
    failing_evidence_files,
    unchanged_failure_run,
)

# The real writer, deliberately. A hand-written reference section would keep passing after the
# format it imitates changed, and the count that reads it would go quietly blind.
from tools.validation_tools import (
    ValidationResult,
    _summary_with_references,
    validation_summary,
    workspace_references_in_summary,
)
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowOrchestrator,
    MockChildWorkstreamExecutor,
    source_rejection_disposition,
)

# 183's own pair, verbatim: one required command, and the two workspace files every one of its
# failing reports named -- the engineer-authored fixture and the product file whose derived
# columns it omitted.
FAILING_COMMAND = (
    "npm run test -- server/tests/bulk-create-apps.test.js server/tests/csv.util.test.js"
)
BULK_TEST = "server/tests/bulk-create-apps.test.js"
BULK_SERVICE = "server/services/app/bulkAppImport.service.js"
CSV_TEST = "server/tests/csv.util.test.js"
CSV_UTIL = "server/util/csv.util.js"

# The same failure, reported three ways. Different counts, different excerpts, different
# result codes and different line numbers -- which is exactly what made 183 invisible -- over
# an unchanging pair of files.
_BULK_REPORTS = (
    (
        f"FAIL {BULK_TEST}\n"
        "  * bulk create > rejects rows missing derived columns\n"
        "    expect(received).toBe(expected)\n"
        f"      at Object.<anonymous> ({BULK_TEST}:41:18)\n"
        f"      at createApps ({BULK_SERVICE}:88:12)\n"
        "Tests:       16 failed, 4 passed, 20 total\n"
    ),
    (
        f"FAIL {BULK_TEST}\n"
        "  * bulk create > writes every derived column\n"
        "    ValidationError: notNull Violation: App.publisher_name cannot be null\n"
        f"      at insertApp ({BULK_SERVICE}:104:9)\n"
        f"      at Object.<anonymous> ({BULK_TEST}:77:5)\n"
        "Tests:       10 failed, 10 passed, 20 total\n"
    ),
    (
        f"FAIL {BULK_TEST}\n"
        "  * bulk create > derives the slug for every row\n"
        "    SEQUELIZE_NOT_NULL_VIOLATION: App.slug cannot be null\n"
        f"      at deriveSlug ({BULK_SERVICE}:131:7)\n"
        f"      at Object.<anonymous> ({BULK_TEST}:12:3)\n"
        "Tests:       3 failed, 17 passed, 20 total\n"
    ),
)

# A genuinely different failure, in a different command, naming a different pair of files --
# and worded three times, so a lineage can alternate between the two, or let one of them
# recover and come back, without any defect the ledger reads ever being re-made. That shape
# is Part B's, and it stops the loop first, which would make these tests pass for the wrong
# reason.
_CSV_REPORTS = (
    (
        f"FAIL {CSV_TEST}\n"
        "  * csv > tolerates an empty header row\n"
        "    RangeError: Invalid array length\n"
        f"      at parseHeader ({CSV_UTIL}:41:5)\n"
        f"      at Object.<anonymous> ({CSV_TEST}:9:1)\n"
        "Tests:       2 failed, 18 passed, 20 total\n"
    ),
    (
        f"FAIL {CSV_TEST}\n"
        "  * csv > keeps quoted delimiters inside a cell\n"
        "    AssertionError: expected 3 columns, received 4\n"
        f"      at splitRow ({CSV_UTIL}:63:11)\n"
        f"      at Object.<anonymous> ({CSV_TEST}:24:1)\n"
        "Tests:       1 failed, 19 passed, 20 total\n"
    ),
    (
        f"FAIL {CSV_TEST}\n"
        "  * csv > accepts a trailing newline\n"
        "    TypeError: Cannot read properties of undefined\n"
        f"      at trimTail ({CSV_UTIL}:88:3)\n"
        f"      at Object.<anonymous> ({CSV_TEST}:52:1)\n"
        "Tests:       4 failed, 16 passed, 20 total\n"
    ),
)
CSV_COMMAND = "npm run test -- server/tests/csv.util.test.js"


# AB-Feature-218's own pair, verbatim. The suite failed on attempts 2, 3, 4 and 5 with one
# substance -- row-validation and duplicate scenarios returning `BULK_APP_TRANSACTION_FAILED
# (500)` where the contract requires 400/409 -- and attempts 2 and 3 were commit-gate
# rejections, which is why nothing counted them.
SERVICE_TEST = "server/tests/bulk-create-apps.service.test.js"
BULK_APP_SERVICE = "server/services/app/bulkApp.service.js"
SERVICE_COMMAND = f"npm run test -- {SERVICE_TEST}"

# Four wordings of the one defect. The moved assertion line is 218's own: between attempts 2
# and 3 the failing assertion went from line 275 to line 291, and that alone was enough to
# make `diagnostics_changed` true and `meaningful_progress` call it progress. These wordings
# keep that property on purpose -- the token signature moves every attempt -- so nothing but
# the per-file count can be what stops the lineages below.
_SERVICE_REPORTS = tuple(
    f"FAIL {SERVICE_TEST}\n"
    f"  * bulk create apps > {scenario}\n"
    f"    expected {expected}, received BULK_APP_TRANSACTION_FAILED (500)\n"
    f"      at Object.<anonymous> ({SERVICE_TEST}:{line}:{column})\n"
    f"      at createBulkApps ({BULK_APP_SERVICE}:{line // 3}:{column})\n"
    f"Tests:       {failed} failed, {20 - failed} passed, 20 total\n"
    for scenario, expected, line, column, failed in (
        ("rejects a row missing app_title", "BULK_APP_VALIDATION_FAILED (400)", 275, 18, 6),
        ("rejects a row missing app_title", "BULK_APP_VALIDATION_FAILED (400)", 291, 11, 4),
        ("reports a duplicate app_id", "BULK_APP_DUPLICATE (409)", 318, 7, 3),
        ("reports a duplicate app_id", "BULK_APP_DUPLICATE (409)", 344, 22, 2),
    )
)


def _workspace(tmp_path: Path) -> Path:
    """Create the files 183's reports name, so the references resolve as they did live."""
    for relative in (BULK_TEST, BULK_SERVICE, CSV_TEST, CSV_UTIL, SERVICE_TEST, BULK_APP_SERVICE):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("module.exports = {};\n")
    return tmp_path


def _scoped_run(
    workspace: Path, report: str, *, command: str = SERVICE_COMMAND
) -> ValidationResult:
    """One narrowed test run that completed and rejected the source, as the gate ran it.

    Built with the real writer over real files, like every other piece of evidence in this
    module: a hand-written reference section would keep passing after the format it imitates
    changed, and the count that reads it would go quietly blind.
    """
    return ValidationResult(
        command=tuple(command.split()),
        return_code=1,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=12.0,
        validation_type="test",
        stderr_summary=_summary_with_references(report, workspace),
        result_code="TEST_VALIDATION_FAILED_EXIT_1",
        required=True,
    )


@dataclass(frozen=True)
class _GateRejection:
    """One commit-gate rejection, as the diagnostics and results the gate raises with."""

    diagnostics: tuple[str, ...]
    validation_results: tuple[dict[str, Any], ...]


def _gate_rejection(*runs: ValidationResult) -> _GateRejection:
    """Compose the rejection the gate raises from the runs that rejected the source."""
    return _GateRejection(
        diagnostics=tuple(scoped_test_diagnostic(run) for run in runs),
        validation_results=tuple(validation_summary(run) for run in runs),
    )


def _as_rejection(gate: _GateRejection) -> SourceValidationError:
    """Return one composed rejection as the exception the Engineer actually raises.

    One producer, two consumers: `GateRejectionExecutor` raises this and lets the workflow's
    handler build the record, and `SequencedOutcomeExecutor` hands it to
    `source_rejection_disposition` and builds the record itself. Neither of them may decide
    what markers it carries, which is what `_gate_rejection` above exists to keep true.
    """
    error = SourceValidationError(gate.diagnostics)
    error.validation_results = gate.validation_results
    return error


def _review_evidence(run: ValidationResult) -> dict[str, Any]:
    """The same run as the reviewer's authoritative pass would have recorded it."""
    return validation_summary(run)


def _consecutive_runs(lineage: Sequence[ChildWorkflowResultArtifact]) -> dict[str, int]:
    """Return, per file, the length of its consecutive run ending at the last attempt.

    Computed here rather than read off `unchanged_failure_run`, deliberately, and it lives in
    this module because this module is the one that owns the count. The rule's own output is
    what these lineages are about, and asserting it against itself would only re-state it --
    which is precisely how this measure once acquired a test that passed for the wrong reason.
    This walks the recorded evidence and counts, and it stops at the first attempt that named
    nothing, because unknown is not agreement.
    """
    named = [failing_evidence_files(attempt) for attempt in lineage]
    if not named or named[-1] is None:
        return {}
    counts: dict[str, int] = {}
    for file in named[-1]:
        run = 1
        for previous in reversed(named[:-1]):
            if previous is None or file not in previous:
                break
            run += 1
        counts[file] = run
    return counts


class GateRejectionExecutor:
    """Fail the backend at the commit gate, at review, or approve it -- per a script.

    A `_GateRejection` entry is an attempt the commit gate stopped: it raises the same
    `SourceValidationError` the Engineer raises, carrying the scoped-test diagnostics its
    narrowed run produced and -- 86- Part B -- the structured results of that run. Nothing
    here stamps `deterministic_gate` itself: the disposition the platform owns decides that,
    which is the whole point of scripting the rejection rather than the result.

    A list entry is one attempt's review-stage validation evidence, recorded as
    `ScriptedValidationExecutor` records it. An empty list is an approval.
    """

    def __init__(self, script: Sequence[Any]) -> None:
        """Record the per-attempt outcomes and count the attempts actually spent."""
        self._delegate = MockChildWorkstreamExecutor()
        self._script = list(script)
        self.attempts = 0
        self.feedback: list[list[str]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return -- or raise -- the scripted outcome, approving every sibling repository."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        self.feedback.append(list(kwargs["feedback"]))
        self.attempts += 1
        entry = self._script[self.attempts - 1] if self.attempts <= len(self._script) else None
        if not entry:
            return execution
        if isinstance(entry, _GateRejection):
            raise _as_rejection(entry)
        validations = list(entry)
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [
                        f"{item['command']} exited with code 1.\n{item['stderr_summary']}"
                        for item in validations
                    ],
                    "current_validation_results": validations,
                    "pull_request_readiness": False,
                    "production_files_changed": [
                        f"server/services/app/attempt{self.attempts}.service.js"
                    ],
                    "failure_classification": "validation_source_failure",
                    "production_diff_fingerprint": f"production-{self.attempts}",
                    "test_diff_fingerprint": f"tests-{self.attempts}",
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


def _validation(summary: str, *, command: str = FAILING_COMMAND) -> dict[str, Any]:
    """Record one failing required command exactly as `_validation_summary` persists it."""
    return {
        "name": "npm",
        "command": command,
        "passed": False,
        "validation_type": "test",
        "status": "failed",
        "stdout_summary": "",
        "stderr_summary": summary,
        "required": True,
    }


def _bulk_validations(workspace: Path) -> list[dict[str, Any]]:
    """Return 183's three attempts of validation evidence, written by the real writer."""
    return [_validation(_summary_with_references(report, workspace)) for report in _BULK_REPORTS]


def _csv_validations(workspace: Path) -> list[dict[str, Any]]:
    """Return the other command's evidence, the same way."""
    return [
        _validation(_summary_with_references(report, workspace), command=CSV_COMMAND)
        for report in _CSV_REPORTS
    ]


class ScriptedValidationExecutor:
    """Fail the backend with scripted validation evidence and blocking issues, per attempt.

    Every attempt reports genuinely different production source, so the existing convergence
    guards permit the next one and nothing but the scripted evidence can stop the loop. An
    empty entry means the attempt was approved.
    """

    def __init__(self, script: Sequence[Sequence[dict[str, Any]]]) -> None:
        """Record the per-attempt validation evidence and count the attempts actually spent."""
        self._delegate = MockChildWorkstreamExecutor()
        self._script = [list(entry) for entry in script]
        self.attempts = 0
        self.feedback: list[list[str]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return the scripted verdict for this backend attempt, approving every sibling."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        self.feedback.append(list(kwargs["feedback"]))
        self.attempts += 1
        validations = (
            self._script[self.attempts - 1]
            if self.attempts <= len(self._script)
            else self._script[-1]
        )
        if not validations:
            return execution
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    # What a reviewer does with a failing required command: quote it. The
                    # wording moves with the report, which is the whole difficulty.
                    "blocking_issues": [
                        f"{item['command']} exited with code 1.\n{item['stderr_summary']}"
                        for item in validations
                    ],
                    "current_validation_results": validations,
                    "pull_request_readiness": False,
                    "production_files_changed": [
                        f"server/services/app/attempt{self.attempts}.service.js"
                    ],
                    "failure_classification": "validation_source_failure",
                    "production_diff_fingerprint": f"production-{self.attempts}",
                    "test_diff_fingerprint": f"tests-{self.attempts}",
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


async def _run(feature_id: str, executor: object) -> FeatureWorkflowSnapshot:
    """Run one feature to its end with a scripted backend and generous ceilings."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state(feature_id, request)
    # Generous on purpose, both of them. Every assertion below is that a workstream stopped
    # well short of these, or was never stopped at all, so an exhaustion stop could not be
    # mistaken for the backstop under test -- nor its absence for the backstop's silence.
    state.max_validation_retries = 8
    state.max_child_review_cycles = 8
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    return await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )


def _backend_attempts(result: FeatureWorkflowSnapshot) -> list[ChildWorkflowResultArtifact]:
    """Return the backend's attempt results from the lineage, in the order they landed."""
    return [
        artifact
        for artifact in result.artifacts
        if isinstance(artifact, ChildWorkflowResultArtifact) and artifact.repository_id == "backend"
    ]


def test_the_reference_section_survives_the_round_trip(tmp_path: Path) -> None:
    """The premise every test below rests on: what the writer appends, the reader recovers.

    Line numbers are dropped deliberately. A repair that changes the file moves every line in
    the report below it, so an identity that carried them would move whenever the source did
    -- which is the one question it exists to answer.
    """
    workspace = _workspace(tmp_path)

    summary = _summary_with_references(_BULK_REPORTS[0], workspace)

    assert workspace_references_in_summary(summary) == [BULK_TEST, BULK_SERVICE]
    # A report naming no workspace file gains no section, and reading one back finds nothing.
    assert workspace_references_in_summary("npm run test exited with code 1") == []


def test_the_coarse_key_holds_where_the_token_key_moves(tmp_path: Path) -> None:
    """C1, first half. One pair, three reports: the fine key sees three failures, this sees one.

    This is 183 in miniature and the reason for a second key at all. Asserting the coarse
    fingerprints match would prove nothing on its own -- the claim is that they match *while*
    the machinery already in place sees change.
    """
    workspace = _workspace(tmp_path)
    validations = _bulk_validations(workspace)

    signatures = {
        diagnostic_signature([f"exited with code 1.\n{item['stderr_summary']}"])
        for item in validations
    }
    assert len(signatures) == 3

    files = {tuple(workspace_references_in_summary(item["stderr_summary"])) for item in validations}
    assert files == {(BULK_TEST, BULK_SERVICE)}


@pytest.mark.asyncio
async def test_three_wordings_of_one_failure_do_not_buy_a_fourth_attempt(tmp_path: Path) -> None:
    """C1. The 183 replay, end to end: the strategy-change line, then the stop.

    Three attempts, three genuinely different production diffs, three differently worded
    reports of the same required command failing on the same two files. Left alone, this
    shape burned four attempts of eight and would have burned the other four.
    """
    workspace = _workspace(tmp_path)
    executor = ScriptedValidationExecutor([[item] for item in _bulk_validations(workspace)])

    result = await _run("feature-unchanged-failure", executor)

    # Stopped on the third identical failure, not after another coding call.
    assert executor.attempts == 3
    attempts = _backend_attempts(result)
    assert len(attempts) == 3
    named = [failing_evidence_files(attempt) for attempt in attempts]
    assert named[0] == frozenset({BULK_TEST, BULK_SERVICE})
    assert named[0] == named[1] == named[2]

    # The second remediation prompt -- the one that bought the third attempt -- was told that
    # the previous repair was real and changed nothing, and which files it touched.
    strategy = [line for line in executor.feedback[2] if line.startswith("UNCHANGED FAILURE")]
    assert len(strategy) == 1
    assert "server/services/app/attempt2.service.js" in strategy[0]
    assert "do not repeat the previous strategy" in strategy[0]
    assert BULK_SERVICE in strategy[0]
    # And it led, because it is the only line about the approach rather than about a defect.
    assert executor.feedback[2][0] == strategy[0]
    # The first attempt's own remediation got no such line: one failure proves nothing.
    assert not any(line.startswith("UNCHANGED FAILURE") for line in executor.feedback[1])

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    assert child.retry_refusal_reason is not None
    assert "stopped producing new information" in child.retry_refusal_reason
    # What a person is sent to do names the command they can run and the files it keeps
    # naming, rather than the three generic possibilities the existing triage chooses between.
    triage = dict(attempts[-1].metadata)
    question = triage["operator_question"]
    assert FAILING_COMMAND in question
    assert "3 consecutive attempts" in question
    assert BULK_SERVICE in question
    assert "did not anticipate" not in question
    assert any(FAILING_COMMAND in item for item in triage["terminal_evidence"])
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


@pytest.mark.asyncio
async def test_two_failures_alternating_are_not_one_failure_repeating(tmp_path: Path) -> None:
    """C2. A -> B -> A -> B injects nothing and stops nothing.

    Coming back to a state this workstream already submitted is real and already has an owner
    -- `_revisits_an_earlier_attempt`, which compares diff bytes across every prior attempt.
    This counter answers a narrower question, "did the last repair change anything at all",
    and an alternation answers it with yes every time.
    """
    workspace = _workspace(tmp_path)
    bulk = _bulk_validations(workspace)
    csv = _csv_validations(workspace)
    # Four reports, four defect identities, two coarse fingerprints alternating between them.
    # Four identities on purpose: reusing a report would make this Part B's lineage instead --
    # a demand raised, satisfied and re-made -- and that stops the loop before this counter is
    # ever consulted, which would make the test pass for the wrong reason.
    executor = ScriptedValidationExecutor([[bulk[0]], [csv[0]], [bulk[1]], [csv[1]], []])

    result = await _run("feature-alternating-failures", executor)

    assert executor.attempts == 5
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    assert not any(
        line.startswith("UNCHANGED FAILURE") for lines in executor.feedback for line in lines
    )


@pytest.mark.asyncio
async def test_a_new_failure_after_two_identical_ones_resets_the_count(tmp_path: Path) -> None:
    """C3. First occurrences are free, and a changed fingerprint makes one.

    The run reaches two before the failure changes, which is where the strategy-change line is
    owed and the stop is not. What must not happen is the third attempt's brand-new failure
    inheriting a count it did not earn.
    """
    workspace = _workspace(tmp_path)
    bulk = _bulk_validations(workspace)
    csv = _csv_validations(workspace)
    executor = ScriptedValidationExecutor([[bulk[0]], [bulk[1]], [csv[0]], []])

    result = await _run("feature-failure-changed", executor)

    # Four attempts: the fourth was approved, so nothing stopped the third.
    assert executor.attempts == 4
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    # The count reached two, so the third attempt was owed the line -- and the fourth was not,
    # because the failure it followed was the first of its kind.
    assert any(line.startswith("UNCHANGED FAILURE") for line in executor.feedback[2])
    assert not any(line.startswith("UNCHANGED FAILURE") for line in executor.feedback[3])


# --------------------------------------------------------------------------------------
# The 194 seam: a guard-stopped attempt inherits the failure's identity
# --------------------------------------------------------------------------------------


def _refused_assertion_repair(
    workspace: Path, *, command: str = FAILING_COMMAND, report: str = _BULK_REPORTS[2]
) -> SourceValidationError:
    """One assertion-guard rejection, built exactly as the Engineer raises it.

    `agents/engineer/agent.py` composes this as
    ``SourceValidationError((assertion_guard_diagnostic(weakened), *error.diagnostics))`` with
    ``terminal_outcome`` set: the guard's own sentence first, then the scoped-test diagnostics
    the refused repair was asked to clear. Built with the real composers over the real
    reference writer, so a change to either shape fails this test instead of silently making
    the inheritance unreadable.

    Returned as the exception rather than as a list of sentences, because what the disposition
    decides about an attempt is a function of the whole rejection -- its diagnostics *and* its
    terminal outcome -- and a fixture that hands over only the sentences has to guess the rest.
    Guessing it is what pinned these two tests to a state the platform stopped producing.

    ``validation_results`` carries the run the refused repair was asked to clear, as the
    Engineer's own ``refused.validation_results = error.validation_results`` now does. The
    guard used to drop them when it rebuilt the rejection, so a guard-stopped attempt recorded
    `[]` and its failure identity had to be re-derived from the diagnostics.
    `_assertion_guard_evidence` still does that, and is still needed -- for the artifacts
    already in the durable record from before this shipped, and for any future rejection that
    genuinely has no structured result -- but on this path the record now states what ran.

    The command is a parameter because 194's was not a constant: the guard attempt's scoped
    command named a different file list than either neighbour's, which is the variation the
    per-file count exists to be blind to.
    """
    failing = ValidationResult(
        command=tuple(command.split()),
        return_code=1,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=1.0,
        validation_type="test",
        stderr_summary=_summary_with_references(report, workspace),
        result_code="TEST_VALIDATION_FAILED_EXIT_1",
    )
    rejection = SourceValidationError(
        (assertion_guard_diagnostic([BULK_TEST]), scoped_test_diagnostic(failing))
    )
    rejection.terminal_outcome = ASSERTION_GUARD_OUTCOME
    # Promoted required exactly as `RepositoryScopedTestRunner` promotes it, because the
    # coarse counter reads only required failing commands and this run is what ended the
    # attempt.
    rejection.validation_results = (validation_summary(replace(failing, required=True)),)
    return rejection


class SequencedOutcomeExecutor:
    """Fail the backend per a script of review-stage evidence or commit-gate rejections.

    A list entry is one attempt's validation evidence, recorded as
    `ScriptedValidationExecutor` records it. A `SourceValidationError` entry is an attempt the
    commit gate stopped -- the assertion guard, here -- and it is scripted as the *rejection*,
    not as the record: the classification, the disposition markers and the recorded validation
    results are all read off it by the platform's own
    `source_rejection_disposition`. An empty list means the attempt was approved.

    **Scripting the rejection rather than the result is the whole point of this class** (87-
    Part D). It used to hard-code `{"deterministic_gate": True, "source_validation_rejected":
    True}` and `current_validation_results: []` onto a guard-shaped stop, which meant the two
    tests below could observe neither half of 86-: not Part A, because the disposition was
    bypassed, and not Part B, because the evidence was emptied. They passed while pinned to a
    state the platform no longer produces -- and a test that looks load-bearing and is not is
    worse than a missing one, because the next reader takes it for coverage.

    What stays scripted is what a live gate-rejected attempt genuinely wrote: the changed
    files and the per-attempt fingerprints. Those are not the disposition's, and a raised
    rejection carries no code completion, so leaving them to the platform here would starve
    `meaningful_progress` of the one question it asks besides "did the diagnostics change".
    """

    def __init__(self, script: Sequence[Any]) -> None:
        """Record the per-attempt outcomes and count the attempts actually spent."""
        self._delegate = MockChildWorkstreamExecutor()
        self._script = list(script)
        self.attempts = 0
        self.feedback: list[list[str]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return the scripted outcome for this backend attempt, approving every sibling."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        self.feedback.append(list(kwargs["feedback"]))
        self.attempts += 1
        entry = self._script[self.attempts - 1] if self.attempts <= len(self._script) else None
        if not entry:
            return execution
        changed = [f"server/services/app/attempt{self.attempts}.service.js"]
        if isinstance(entry, SourceValidationError):
            # Asked, never asserted. Which markers a commit-gate rejection carries is the
            # platform's to decide from the rejection's own diagnostics, and 86- Part A
            # narrowed that decision: a rejection carrying a scoped-test diagnostic is not a
            # deterministic gate, because a test repeating itself is the change not moving.
            classification, disposition = source_rejection_disposition(entry)
            return ChildExecution(
                result=execution.result.model_copy(
                    update={
                        "status": "failed",
                        "blocking_issues": list(entry.diagnostics),
                        # Whatever the rejection carries, which for the assertion guard is
                        # nothing -- it builds a fresh error and drops the results of the run
                        # it was asked to clear. `_assertion_guard_evidence` is what recovers
                        # the failure identity from the diagnostics on this path.
                        "current_validation_results": list(entry.validation_results),
                        "pull_request_readiness": False,
                        "production_files_changed": changed,
                        "failure_classification": classification,
                        "metadata": {**execution.result.metadata, **disposition},
                        "production_diff_fingerprint": f"production-{self.attempts}",
                        "test_diff_fingerprint": f"tests-{self.attempts}",
                    }
                ),
                code_completion=execution.code_completion,
                review=execution.review,
            )
        validations = list(entry)
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [
                        f"{item['command']} exited with code 1.\n{item['stderr_summary']}"
                        for item in validations
                    ],
                    "current_validation_results": validations,
                    "pull_request_readiness": False,
                    "production_files_changed": changed,
                    "failure_classification": "validation_source_failure",
                    "production_diff_fingerprint": f"production-{self.attempts}",
                    "test_diff_fingerprint": f"tests-{self.attempts}",
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


@pytest.mark.asyncio
async def test_a_guard_stopped_attempt_does_not_reset_the_consecutive_counter(
    tmp_path: Path,
) -> None:
    """194's sequence, replayed: same -> same -> guard, and the guard attempt is the third.

    Live, the guard attempt carried no validation evidence, so its fingerprint was None and
    the walk back stopped at it: the counter restarted, the ×3 stop moved out by an attempt,
    and the run bought extra budget for the same defect the guard had just refused to paper
    over. Two things now prevent that, and the second is why this test's numbers moved.

    * **The evidence.** A guard-stopped attempt now records the run its refused repair was
      asked to clear, so it is not a hole in the lineage. Both routes to that identity are
      asserted below and they agree: the recorded results, and the derivation
      `_assertion_guard_evidence` performs over the diagnostics. The derivation is still what
      covers an artifact written before the guard carried its results.
    * **The disposition.** 86- Part A withheld `deterministic_gate` from any rejection
      carrying a scoped-test diagnostic, and the guard's rejection is
      `(assertion_guard_diagnostic(...), *scoped_test_diagnostics)`. So the guard attempt is
      **no longer exempt**, and the ×3 stop lands on it rather than on the attempt after it.

    That is one attempt earlier than this test used to assert, and the reason the number was
    wrong for a while is worth stating: this fixture hard-coded `deterministic_gate: True` and
    an empty validation list onto the guard attempt, so it could observe neither change. It
    now scripts the rejection and lets the platform decide, which is 87- Part D.

    Stopping here is the honest reading of the lineage. The counts are asserted below: the
    file is named by three consecutive attempts, and the third of them had already been handed
    the ×2 strategy line naming that file -- and answered it by trying to weaken the suite's
    assertion, which the guard refused. Nothing about that is a workstream converging.
    """
    workspace = _workspace(tmp_path)
    bulk = _bulk_validations(workspace)
    executor = SequencedOutcomeExecutor(
        [
            [bulk[0]],
            [bulk[1]],
            _refused_assertion_repair(workspace),
            [bulk[2]],
            # A fifth attempt would be approved. It must never be asked for: reaching it is
            # exactly the counter reset this test exists to rule out.
            [],
        ]
    )

    result = await _run("feature-guard-fingerprint", executor)

    # Stopped on the guard attempt -- the third consecutive failure on one file -- not after
    # a fourth coding call. The fourth and fifth entries above are scripted precisely so
    # their absence is provable, and the fifth would have been approved.
    assert executor.attempts == 3
    attempts = _backend_attempts(result)
    assert len(attempts) == 3
    named = [failing_evidence_files(attempt) for attempt in attempts]
    # The guard attempt inherited the files its refused repair was asked to clear: one file
    # set, three attempts, no unknown in the middle of the run.
    assert named[0] == frozenset({BULK_TEST, BULK_SERVICE})
    assert len(set(named)) == 1
    # The counts, per file and read off the recorded evidence rather than off the verdict.
    # Both files reach three at the guard attempt, which is `_STOP_ATTEMPTS`.
    assert _consecutive_runs(attempts[:1]) == {BULK_TEST: 1, BULK_SERVICE: 1}
    assert _consecutive_runs(attempts[:2]) == {BULK_TEST: 2, BULK_SERVICE: 2}
    assert _consecutive_runs(attempts) == {BULK_TEST: 3, BULK_SERVICE: 3}
    # What the guard attempt carries, so a regression in either 86- part fails here rather
    # than quietly restoring the counter reset: the disposition withholds the exemption, the
    # pre-commit fact survives, and the evidence comes from the diagnostics because the
    # rejection carries no structured results of its own.
    guard = attempts[2]
    assert not guard.metadata.get("deterministic_gate")
    assert guard.metadata["source_validation_rejected"] is True
    # What its gate ran, on the record rather than only in its prose -- and required, which
    # is the only kind of entry the coarse counter reads.
    assert len(guard.current_validation_results) == 1
    assert guard.current_validation_results[0]["required"] is True
    assert guard.current_validation_results[0]["passed"] is False
    assert failing_evidence_files(guard) == frozenset({BULK_TEST, BULK_SERVICE})
    # And the two routes to that identity agree. The recorded results are what the counter
    # reads now; `_assertion_guard_evidence` re-derives the same set from the diagnostics, and
    # it is what still covers a guard attempt recorded before the results were carried.
    assert _assertion_guard_evidence(guard) == _failing_required_evidence(
        guard.current_validation_results
    )
    # The ×2 strategy line still fired where it was owed -- and the attempt it was given to
    # is the guard's, which answered it by editing the suite.
    strategy = [line for line in executor.feedback[2] if line.startswith("UNCHANGED FAILURE")]
    assert len(strategy) == 1
    assert "2 consecutive attempts" in strategy[0]
    assert not any(line.startswith("UNCHANGED FAILURE") for line in executor.feedback[1])
    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    assert "stopped producing new information" in (child.retry_refusal_reason or "")
    assert "3 consecutive attempts" in dict(attempts[-1].metadata)["operator_question"]
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


# --------------------------------------------------------------------------------------
# 194 itself: the same test file, inside a command that is rewritten every round
# --------------------------------------------------------------------------------------

# `scoped_test_command` builds its file list from the attempt's own changed files, so the
# command string is rewritten whenever the previous repair touched a different file. These
# are the three shapes 194 produced -- one failing test file, three commands. Under the pair
# key `(command, files)` no two consecutive attempts ever compared equal, so a test that
# failed on five of six attempts armed neither threshold.
_SCOPED_COMMAND_VARIANTS = (
    f"npm run test -- {BULK_TEST}",
    f"npm run test -- {BULK_TEST} {CSV_TEST}",
    f"npm run test -- {BULK_TEST} {BULK_SERVICE}",
)


@pytest.mark.asyncio
async def test_a_varying_command_around_one_failing_file_arms_both_thresholds(
    tmp_path: Path,
) -> None:
    """194's recorded sequence, replayed: three command strings and a guard interruption.

    Live this ran six attempts -- the test file failed on 0, 1, 3, 4 and 5, with attempt 2
    stopped by the assertion guard -- and the ×2 line and the ×3 stop never fired at all,
    because the key was the pair `(command, files)` and the command was rewritten most
    rounds. The validation ceiling of eight was the only bound on the loop.

    Counting per file, the same evidence arms both. The line is owed at the file's second
    consecutive appearance, attempt 2. The stop is owed at its third, attempt 3 -- **and
    attempt 3 is the guard's, which is where it now lands.** This test used to say the stop
    moved out by one because a deterministic gate is exempt; 86- Part A withheld that
    exemption from any rejection carrying a scoped-test diagnostic, and the guard's rejection
    carries the ones its refused repair was asked to clear. Three of the six attempts are
    saved, and the fourth through sixth -- two of which would have been approved, and which
    this test scripts precisely so their absence is provable -- are never asked for.

    The premise this test exists for is unchanged and is still asserted: three distinct
    command strings around one file that never moves. It is read off `coarse_failure_evidence`
    rather than off `current_validation_results`, because that is the projection the counter
    itself consults -- and it is the only one that can see the guard attempt's command, which
    lives in its diagnostic's headline rather than in a recorded result.
    """
    workspace = _workspace(tmp_path)
    scripted = [
        [
            _validation(
                _summary_with_references(report, workspace),
                command=_SCOPED_COMMAND_VARIANTS[index],
            )
        ]
        for index, report in enumerate(_BULK_REPORTS)
    ]
    executor = SequencedOutcomeExecutor(
        [
            scripted[0],
            scripted[1],
            _refused_assertion_repair(
                workspace,
                command=_SCOPED_COMMAND_VARIANTS[2],
                report=_BULK_REPORTS[1],
            ),
            scripted[2],
            [],
            [],
        ]
    )

    result = await _run("feature-varying-command", executor)

    assert executor.attempts == 3
    attempts = _backend_attempts(result)
    # The premise, asserted rather than assumed: the commands really did vary while the file
    # set really did not. Without both halves this test would prove nothing about why the
    # pair key went blind. Read through the counter's own projection, which is the only one
    # that sees the guard attempt's command.
    evidence = [coarse_failure_evidence(attempt) for attempt in attempts]
    recorded = {command for entry in evidence if entry for command, _files in entry}
    assert recorded == set(_SCOPED_COMMAND_VARIANTS), recorded
    named = [failing_evidence_files(attempt) for attempt in attempts]
    assert all(entry is not None and BULK_TEST in entry for entry in named)
    # And the counts, per file and off the evidence: one file set, three consecutive attempts,
    # under three different commands.
    assert _consecutive_runs(attempts[:2]) == {BULK_TEST: 2, BULK_SERVICE: 2}
    assert _consecutive_runs(attempts) == {BULK_TEST: 3, BULK_SERVICE: 3}

    # The line at the second consecutive appearance, naming the file and the previous
    # repair's own edits -- and nothing at the first, which buys nothing.
    assert not any(line.startswith("UNCHANGED FAILURE") for line in executor.feedback[1])
    strategy = [line for line in executor.feedback[2] if line.startswith("UNCHANGED FAILURE")]
    assert len(strategy) == 1
    assert BULK_TEST in strategy[0]
    assert "server/services/app/attempt2.service.js" in strategy[0]

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    assert "stopped producing new information" in (child.retry_refusal_reason or "")
    # What a person is sent to run is the last attempt's command, and the sentence says so
    # rather than claiming every attempt ran that one. The last attempt is the guard's, and
    # its command is the third variant -- so this assertion is unchanged while the attempt it
    # describes has moved.
    triage = dict(attempts[-1].metadata)
    question = triage["operator_question"]
    assert BULK_TEST in question
    assert _SCOPED_COMMAND_VARIANTS[2] in question
    assert "3 consecutive attempts" in question
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


@pytest.mark.asyncio
async def test_this_files_fixtures_ask_the_disposition_and_never_assert_it(
    tmp_path: Path,
) -> None:
    """87- Part D's own guard: the executor above decides nothing about a rejection.

    The defect this closes was not a wrong number. `SequencedOutcomeExecutor` hard-coded both
    `deterministic_gate: True` and an empty validation list onto a guard-shaped stop, so the
    two tests driving it could observe neither half of 86-: not Part A, because the disposition
    was bypassed, and not Part B, because the evidence was emptied. They passed while pinned to
    a state the platform had stopped producing, and a test that looks load-bearing and is not
    is worse than a missing one.

    Proved behaviourally rather than by reading the fixture: one executor, three rejections,
    three different answers. A fixture that stamped anything could not produce all three.

    * a scoped-test rejection: **exempt withheld**, and its structured results recorded --
      one assertion per 86- part, so a regression in either fails here;
    * a lint-only rejection: **exempt granted**, unchanged, which is the -052b class;
    * the assertion guard's own rejection: exempt withheld, and the results of the run it was
      asked to clear recorded, because the guard now carries them out of the rebuild.
    """
    workspace = _workspace(tmp_path)
    scoped = _as_rejection(_gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[0])))
    lint = SourceValidationError(
        (
            "Pre-commit source validation failed (tool=eslint): "
            "src/routes/status.js:14:7  error  'formatStatus' is not defined  no-undef",
        )
    )
    guard = _refused_assertion_repair(workspace)

    executor = SequencedOutcomeExecutor([scoped, lint, guard, []])
    result = await _run("feature-disposition-is-asked", executor)

    attempts = _backend_attempts(result)
    assert len(attempts) >= 3
    by_kind = dict(zip(("scoped", "lint", "guard"), attempts[:3], strict=False))

    # 86- Part A, both directions, out of one code path.
    assert not by_kind["scoped"].metadata.get("deterministic_gate")
    assert by_kind["lint"].metadata["deterministic_gate"] is True
    assert not by_kind["guard"].metadata.get("deterministic_gate")
    # Where the attempt was stopped is unchanged in all three: that fact is what the
    # pre-commit retry allowance is about, and 83- established it must not be re-conflated.
    assert all(item.metadata["source_validation_rejected"] is True for item in attempts[:3])

    # 86- Part B: the scoped run's structured results are on the record and the counter can
    # read them; the lint gate keeps none, so its attempt is unknown, exactly as before.
    assert by_kind["scoped"].current_validation_results
    assert failing_evidence_files(by_kind["scoped"]) == frozenset({SERVICE_TEST, BULK_APP_SERVICE})
    assert by_kind["lint"].current_validation_results == []
    assert failing_evidence_files(by_kind["lint"]) is None
    # The guard: the run it refused a repair for, and the identity that follows from it.
    assert by_kind["guard"].current_validation_results
    assert failing_evidence_files(by_kind["guard"]) == frozenset({BULK_TEST, BULK_SERVICE})


@pytest.mark.asyncio
async def test_a_file_that_recovers_resets_only_its_own_count(tmp_path: Path) -> None:
    """Two files failing side by side, and one of them is repaired.

    Each file carries its own run, so the repair that cleared one of them says nothing about
    the other. The count that survives is the one whose file kept appearing; the count that
    restarts is only the repaired file's, and the stop names the surviving file alone -- a
    sentence naming both would send a reader to a file the evidence had just cleared.
    """
    workspace = _workspace(tmp_path)
    bulk = _bulk_validations(workspace)
    csv = _csv_validations(workspace)
    executor = ScriptedValidationExecutor(
        [
            # Both commands failing.
            [bulk[0], csv[0]],
            # The bulk test passes this round; only the csv one still fails.
            [csv[1]],
            # Bulk fails again -- a run of one -- while csv reaches three.
            [bulk[1], csv[2]],
            # Never reached.
            [],
        ]
    )

    result = await _run("feature-partial-recovery", executor)

    assert executor.attempts == 3
    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    assert "stopped producing new information" in (child.retry_refusal_reason or "")
    triage = dict(_backend_attempts(result)[-1].metadata)
    question = triage["operator_question"]
    assert CSV_TEST in question
    assert CSV_UTIL in question
    assert "3 consecutive attempts" in question
    # The file the second attempt repaired is not in the stop's evidence, even though the
    # third attempt failed on it again: its own run restarted and is one attempt long.
    assert BULK_TEST not in question
    assert not any(BULK_TEST in item for item in triage["terminal_evidence"])


# --------------------------------------------------------------------------------------
# 86- Part B: a gate-rejected attempt records what its gate ran
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_gate_rejected_attempt_names_the_suite_that_stopped_it(tmp_path: Path) -> None:
    """B1. The commit gate's own scoped run reaches the field the counter reads.

    `coarse_failure_evidence` reads `current_validation_results`, and `_child_result` fills
    that from the review -- so an attempt the commit gate stopped, which has no review, wrote
    it empty. Empty is read as *unknown* rather than as agreement, which is right and is
    exactly why the walk-back broke there: every commit-gate rejection was a hole in the
    lineage. Verified live on AB-Feature-218: backend attempts 0-3 and frontend 0-1 all carry
    `current_validation_results: []`.

    The assertion is the effect the counter consumes -- `failing_evidence_files` is not None
    and names the suite -- not the presence of a field.
    """
    workspace = _workspace(tmp_path)
    executor = GateRejectionExecutor(
        [_gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[0])), []]
    )

    result = await _run("feature-218-gate-evidence", executor)

    attempts = _backend_attempts(result)
    assert len(attempts) == 2
    named = failing_evidence_files(attempts[0])
    assert named is not None
    assert named == frozenset({SERVICE_TEST, BULK_APP_SERVICE})
    # And the attempt is still classified and marked exactly as a commit-gate rejection is:
    # Part B adds evidence and moves no verdict.
    assert attempts[0].failure_classification == "validation_source_failure"
    assert attempts[0].metadata["source_validation_rejected"] is True


@pytest.mark.asyncio
async def test_the_218_lineage_is_a_run_of_four_and_not_of_two(tmp_path: Path) -> None:
    """B2. Consecutive commit-gate rejections on one suite are a run the counter can see.

    218's recorded facts: `bulk-create-apps.service.test.js` failed on attempts 2, 3, 4 and 5
    with one substance; attempts 2 and 3 were commit-gate rejections and 4 and 5 reached
    review. At attempt 5 the true consecutive run was **four** and the counter read **two**,
    because the walk back from attempt 5 reached attempt 3, found
    `current_validation_results: []`, and stopped there -- below `_STOP_ATTEMPTS`. Nothing
    fired, with `validation_retry_count` at 5 of 8.

    Measured over the platform's own record: the executor raises the rejection the Engineer
    raises, and the workflow's handler builds the artifact, so nothing here can hand the
    counter evidence a live attempt would not have.

    Two claims, and they are different. **The reachable one** is what the loop now does: with
    the evidence in place the third consecutive failure already reads three, so the loop stops
    there and a fourth unbroken attempt is never asked for. **The arithmetic one** is the run
    of four itself -- 218's own number -- which is asserted by extending the recorded lineage
    with a fourth attempt of the same shape, since the loop will no longer produce one. The
    three attempts underneath it are the platform's, so a record that carried no evidence
    would read one here rather than four.
    """
    workspace = _workspace(tmp_path)
    runs = [_scoped_run(workspace, report) for report in _SERVICE_REPORTS]
    executor = GateRejectionExecutor([*[_gate_rejection(run) for run in runs], []])

    result = await _run("feature-218-lineage", executor)

    lineage = _backend_attempts(result)
    # The reachable claim: three attempts, then the stop. The fourth and fifth entries above
    # are scripted precisely so their absence is provable.
    assert executor.attempts == 3
    assert len(lineage) == 3
    named = [failing_evidence_files(attempt) for attempt in lineage]
    assert all(entry is not None and SERVICE_TEST in entry for entry in named), named
    assert all(item.metadata.get("source_validation_rejected") for item in lineage)
    reachable = unchanged_failure_run(
        list(lineage[:2]),
        child_workflow_id=lineage[2].child_workflow_id,
        current=lineage[2],
    )
    assert reachable is not None
    assert reachable.attempts == 3
    assert reachable.stops_the_loop is True
    assert SERVICE_TEST in reachable.files

    # The arithmetic claim: 218's four. The fourth attempt is the third one again, carrying
    # the fourth wording's evidence -- the shape the platform just produced, one attempt
    # further on than the loop is now willing to go.
    fourth = lineage[2].model_copy(
        update={"current_validation_results": [_review_evidence(runs[3])]}
    )
    run = unchanged_failure_run(
        list(lineage), child_workflow_id=fourth.child_workflow_id, current=fourth
    )
    assert run is not None
    assert run.attempts == 4
    assert run.stops_the_loop is True

    # And the counter-factual, in the same test, so neither number is read out of a fixture
    # that was always going to produce it: strip the evidence the gate now records -- which is
    # exactly the state of the record before this part -- and every attempt is unknown, so the
    # walk-back breaks at the first step and there is no run at all.
    blinded = [item.model_copy(update={"current_validation_results": []}) for item in lineage]
    assert all(failing_evidence_files(item) is None for item in blinded)
    assert (
        unchanged_failure_run(
            blinded[:2], child_workflow_id=blinded[2].child_workflow_id, current=blinded[2]
        )
        is None
    )
    # The four-long lineage too: the fourth attempt names the suite, and the attempt behind
    # it names nothing, so its run is one and the arithmetic claim above collapses.
    assert (
        unchanged_failure_run(blinded, child_workflow_id=fourth.child_workflow_id, current=fourth)
        is None
    )


@pytest.mark.asyncio
async def test_218s_own_two_and_two_shape_reaches_the_stop_threshold(tmp_path: Path) -> None:
    """B2b. 218's shape exactly: gate, gate, review -- and the third attempt reads three.

    The two-and-two lineage cannot be measured at four, because with the evidence in place
    the third attempt already reads three and the review attempt it lands on is not exempt,
    so the loop stops there. That is the fix working one attempt earlier than 218's own
    arithmetic suggested, and it is asserted here rather than reasoned about.

    Before this part the same three attempts read **one**: the walk back from the review
    attempt reached the gate rejection behind it and found nothing.
    """
    workspace = _workspace(tmp_path)
    runs = [_scoped_run(workspace, report) for report in _SERVICE_REPORTS]
    executor = GateRejectionExecutor(
        [_gate_rejection(runs[0]), _gate_rejection(runs[1]), [_review_evidence(runs[2])], []]
    )

    result = await _run("feature-218-two-and-two", executor)

    lineage = _backend_attempts(result)
    assert len(lineage) == 3
    assert [bool(item.metadata.get("source_validation_rejected")) for item in lineage] == [
        True,
        True,
        False,
    ]
    run = unchanged_failure_run(
        list(lineage[:2]),
        child_workflow_id=lineage[2].child_workflow_id,
        current=lineage[2],
    )
    assert run is not None
    assert run.attempts == 3
    assert run.stops_the_loop is True

    blinded = [
        item.model_copy(update={"current_validation_results": []})
        if item.metadata.get("source_validation_rejected")
        else item
        for item in lineage
    ]
    assert (
        unchanged_failure_run(
            blinded[:2], child_workflow_id=blinded[2].child_workflow_id, current=blinded[2]
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_gate_command_that_could_not_complete_contributes_no_evidence(
    tmp_path: Path,
) -> None:
    """B3. A timeout never produced a verdict, and the walk-back must still break there.

    Admitting one would let a capacity failure -- a suite that could not finish inside its
    limit -- accumulate on this counter as though the source had not moved, which is the
    AB-Feature-108 defect: rewriting the file under test cannot make a suite fit in memory.
    `rejects_the_source` draws the line, and the gate's structured record is filtered by that
    same predicate rather than by a second copy of the judgement.
    """
    workspace = _workspace(tmp_path)
    timed_out = ValidationResult(
        command=tuple(SERVICE_COMMAND.split()),
        return_code=None,
        stdout="",
        stderr="",
        timed_out=True,
        duration_seconds=1800.0,
        validation_type="test",
        stderr_summary=_summary_with_references(_SERVICE_REPORTS[0], workspace),
        required=True,
    )
    # The gate composes neither a diagnostic nor a result from this run, so what reaches the
    # record is a rejection carrying the *other* command's verdict and nothing of this one's.
    rejection = _gate_rejection(_scoped_run(workspace, _SERVICE_REPORTS[0]))
    assert validation_summary(timed_out) not in rejection.validation_results
    assert timed_out.failure_classification == "validation_timed_out"

    executor = GateRejectionExecutor(
        [
            _GateRejection(
                diagnostics=("The change's own tests failed (command=x; code=None).",),
                validation_results=(),
            ),
            [],
        ]
    )
    result = await _run("feature-218-timeout", executor)

    attempts = _backend_attempts(result)
    # No evidence at all: unknown, never unchanged. The walk-back breaks here as it always did.
    assert failing_evidence_files(attempts[0]) is None


@pytest.mark.asyncio
async def test_a_gate_output_naming_no_workspace_file_contributes_no_evidence(
    tmp_path: Path,
) -> None:
    """B4. A run whose output names nothing is unknown, and unknown is not agreement.

    The counted unit is a workspace file the failing evidence keeps naming. A summary with no
    reference section names none, so the entry is recorded -- the command really did run and
    really did reject the source -- and contributes nothing to the count.
    """
    workspace = _workspace(tmp_path)
    anonymous = ValidationResult(
        command=tuple(SERVICE_COMMAND.split()),
        return_code=1,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=9.0,
        validation_type="test",
        stderr_summary="npm ERR! Test failed. See above for more details.",
        result_code="TEST_VALIDATION_FAILED_EXIT_1",
        required=True,
    )
    del workspace
    executor = GateRejectionExecutor([_gate_rejection(anonymous), []])

    result = await _run("feature-218-anonymous", executor)

    attempts = _backend_attempts(result)
    # The command is on the record -- a reader can see what ran -- and the count sees nothing.
    assert attempts[0].current_validation_results
    assert (
        workspace_references_in_summary(
            str(attempts[0].current_validation_results[0]["stderr_summary"])
        )
        == []
    )
    assert failing_evidence_files(attempts[0]) is None
