"""Every terminal feature state has to explain itself, and this is what checks that it does.

The measurement this file exists to answer, taken from the live database over the fifty
features that failed since 2026-08-08:

    33  carried a real domain classification
    10  said `unclassified_failure` with an empty diagnostics array
     7  carried a bare Python exception type as their root classification --
        WorkspaceCapacityError x3, RuntimeConfigurationError, LLMAdapterError,
        DBAPIError, ValidationError
     5  had no diagnostics at all

`DBAPIError` and `ValidationError` are defects in this platform. Recording them as a feature's
root classification presented them as findings about the target repository, and a person was
asked to decide about a repository whose only fault was being open at the time.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.exc import DBAPIError

from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import (
    _initial_feature_state,
)
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    ContractChangeRequestArtifact,
    IntegrationContractArtifact,
    PullRequestArtifact,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import (
    CompensationStatus,
    ExternalOperationStatus,
    ExternalOperationType,
)
from state.failure_diagnosis import (
    DiagnosedFailure,
    FailureStage,
    FeatureFailureClassification,
    classification_of,
    fallback_diagnostic,
    is_retryable_classification,
    normalize_classification,
)
from state.feature_models import (
    ChildWorkflowReference,
    FeatureWorkflowSnapshot,
    ensure_feature_failure_summary,
)
from storage.db import Database
from storage.external_operation_store import (
    RECOVERY_DEFECT_ERROR_CODE,
    ExternalOperationJournal,
)
from storage.feature_store import (
    SqlAlchemyFeatureControlPlane,
    refusal_states_work_may_proceed,
)
from storage.models import FeatureExecutionQueueModel
from tests.support import drain_feature_queue
from tests.test_feature_workflow import (
    OneRepositoryFailsExecutor,
    PublishOnlyPublisher,
    RecordingReconnaissance,
    RejectingIntegrationReviewer,
    feature_payload,
)
from tools.retry_strategy import FailureClassification
from workflows import feature_workflow as feature_workflow_module
from workflows.feature_workflow import (
    FeatureWorkflowOrchestrator,
    PartialPullRequestError,
    exhausted_fault_diagnostics,
    transient_fault_classification,
)

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)
SERVER_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------------------


def test_the_feature_vocabulary_names_every_child_classification() -> None:
    """The mirrored half of `FeatureFailureClassification` cannot drift out of step.

    `FailureClassification` lives in `tools`, which imports `state`, so `state` cannot import
    it back without a cycle. The values are therefore repeated -- and repeated values drift.
    This is what stops them: a child classification added there and not here would arrive at
    a feature's terminal record as `platform_defect`, blaming the platform for a repository's
    failure.
    """
    feature_values = {item.value for item in FeatureFailureClassification}
    missing = sorted({item.value for item in FailureClassification} - feature_values)
    assert not missing, (
        "these child classifications have no feature-level equivalent, so a feature that "
        f"stopped on one would be recorded as a platform defect: {missing}"
    )


def test_an_unrecognised_classification_is_recorded_as_the_platform_failing() -> None:
    """A name nobody put in the vocabulary is evidence about this platform, not a repository."""
    assert normalize_classification("DBAPIError") is FeatureFailureClassification.PLATFORM_DEFECT
    assert (
        normalize_classification("ValidationError") is FeatureFailureClassification.PLATFORM_DEFECT
    )
    assert normalize_classification(None) is FeatureFailureClassification.PLATFORM_DEFECT
    assert normalize_classification("   ") is FeatureFailureClassification.PLATFORM_DEFECT


def test_a_provider_subtype_keeps_the_retryable_verdict_it_used_to_drive() -> None:
    """Retryability survives the move off substring-matched exception names.

    Before this, `root_classification` held the provider SDK's own exception name and the
    retryable flag was decided by matching substrings against it. A timeout that stopped
    being retryable would strand a feature that a plain resume would have cleared.
    """
    for name in ("APITimeoutError", "APIConnectionError", "RateLimitError"):
        assert normalize_classification(name) is FeatureFailureClassification.PROVIDER_UNAVAILABLE
    assert is_retryable_classification(FeatureFailureClassification.PROVIDER_UNAVAILABLE.value)
    assert not is_retryable_classification(FeatureFailureClassification.PLATFORM_DEFECT.value)
    # Rows written before this change still hold the SDK name, and reading one must not
    # silently change its verdict.
    assert is_retryable_classification("APITimeoutError")


def test_a_git_outage_keeps_the_retryable_verdict_the_provider_classification_carried() -> None:
    """Splitting the classification must move the prose and not the verdict.

    The five surfaces that key on retryability -- `ensure_feature_failure_summary`'s flag and
    next action, the status a failed execution is left in, the two resume seams, and whether
    `RESUME_WORKFLOW` is advertised -- all read this one predicate. A Git outage clears on its
    own exactly as a provider one does, and there is no model spend to protect either, so the
    answer here is deliberately the same for both.
    """
    assert is_retryable_classification(FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE.value)
    # And it says which service, which is the whole reason it is a separate value.
    git = fallback_diagnostic(FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE)
    assert "Git remote" in git
    # The provider is named only to rule it out. AB-Feature-190's operator went and looked at
    # one, so the sentence sends them back rather than leaving the omission to be noticed.
    assert "not the model provider" in git


def test_no_exception_type_name_resolves_to_the_git_classification() -> None:
    """It is only ever recorded where the effect is established not to have landed.

    A `GitAdapterError` that escapes publication may well have pushed. Reading the type name
    as "the Git remote did not answer" would make that retryable, and retrying an unconfirmed
    external effect is the one guess the operation journal exists to prevent.
    """
    assert (
        normalize_classification("GitAdapterError") is FeatureFailureClassification.PLATFORM_DEFECT
    )
    from adapters.git_adapter import GitAdapterError

    assert classification_of(GitAdapterError("git push failed")) is (
        FeatureFailureClassification.PLATFORM_DEFECT
    )


def test_a_diagnosed_failure_cannot_be_raised_without_something_to_say() -> None:
    """The contract is enforced in the constructor, not documented and hoped for."""

    class _Nothing(DiagnosedFailure):
        pass

    error = _Nothing(classification=FeatureFailureClassification.PLATFORM_DEFECT, diagnostics=())
    assert error.diagnostics
    assert "defect in the platform" in error.diagnostics[0]
    assert error.failure_classification is FeatureFailureClassification.PLATFORM_DEFECT


def test_an_exception_declaring_its_classification_is_taken_at_its_word() -> None:
    """The declared value wins over the type name, which is what the old code read."""

    class _Declared(RuntimeError):
        failure_classification = FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE.value

    assert classification_of(_Declared()) is FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE
    assert classification_of(TimeoutError()) is FeatureFailureClassification.PROVIDER_UNAVAILABLE
    assert classification_of(DBAPIError("s", {}, Exception())) is (
        FeatureFailureClassification.PLATFORM_DEFECT
    )


# ---------------------------------------------------------------------------------------
# 9.1 -- every terminal path, derived from the source rather than remembered
# ---------------------------------------------------------------------------------------


_TERMINAL_STATUS_NAMES = {"FAILED", "FAILED_REQUIRES_HUMAN"}
# The transition service from `30-`. A feature status is now written by calling one of these
# and by nothing else, which is enforced separately in `tests/test_feature_transitions.py`.
_TRANSITION_CALLS = {"transition_feature", "transition_feature_state_json"}


def _names_a_terminal_status(node: ast.AST) -> bool:
    """Report whether an expression is ``FeatureWorkflowStatus.FAILED[_REQUIRES_HUMAN]``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr in _TERMINAL_STATUS_NAMES
        and isinstance(node.value, ast.Name)
        and node.value.id == "FeatureWorkflowStatus"
    )


def _terminal_status_sites() -> dict[tuple[str, str], int]:
    """Find every path that puts a feature into a terminal status, by reading the source.

    Derived rather than listed. A hand-maintained inventory of terminal paths is exactly the
    thing that drifts, and a drifted inventory reads as coverage -- which is how a third of
    real failures came to carry no usable diagnosis while a test suite reported green.

    Two shapes are counted, and both must stay. Task `30-` moved every status write behind
    `state.feature_transitions`, so a terminal path is now a *call* naming a terminal status
    rather than an assignment of one; the assignment form is still recognised so that a
    regression which reintroduces one is counted here rather than disappearing from the
    inventory. Reading only one shape is how a guard stops being total without failing.

    Returns how many terminal paths each function contains, keyed by module and function.
    Line numbers are deliberately not part of the key: they move on every edit, and a
    registry that has to be renumbered is a registry nobody updates.
    """
    sites: dict[tuple[str, str], int] = {}
    skip = {"tests", ".venv", "migrations", "__pycache__"}
    for path in sorted(SERVER_ROOT.rglob("*.py")):
        relative = path.relative_to(SERVER_ROOT)
        if skip & set(relative.parts):
            continue
        tree = ast.parse(path.read_text())
        scopes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                terminal = _names_a_terminal_status(node.value) and any(
                    isinstance(target, ast.Attribute) and target.attr == "status"
                    for target in node.targets
                )
            elif isinstance(node, ast.Call):
                terminal = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in _TRANSITION_CALLS
                    and any(
                        _names_a_terminal_status(argument)
                        for argument in [
                            *node.args,
                            *[keyword.value for keyword in node.keywords],
                        ]
                    )
                )
            else:
                continue
            if not terminal:
                continue
            enclosing = max(
                (
                    scope
                    for scope in scopes
                    if scope.lineno <= node.lineno <= (scope.end_lineno or scope.lineno)
                ),
                key=lambda scope: scope.lineno,
                default=None,
            )
            key = (str(relative), enclosing.name if enclosing else "<module>")
            sites[key] = sites.get(key, 0) + 1
    return sites


# Every function that may put a feature into a terminal status, how many such assignments it
# contains, and the test in this module that drives it there and reads what it recorded.
#
# The counts are checked against the source above. A new terminal path in a listed function
# changes its count; one in a new function adds a key. Either fails, and the only way to make
# it pass is to add an entry -- which cannot be done without naming a test that proves the
# new path says something. That is the whole point: it must be hard to add a terminal state
# silently.
_TERMINAL_PATH_REGISTRY: dict[tuple[str, str], tuple[int, str]] = {
    ("storage/feature_store.py", "fail_queued"): (
        1,
        "test_a_durable_queue_refusal_explains_itself",
    ),
    ("storage/feature_store.py", "flag_unconfirmed_external_effect"): (
        1,
        "test_an_unconfirmed_external_effect_explains_itself",
    ),
    ("storage/feature_store.py", "_reconcile_abandoned_run"): (
        1,
        "test_an_abandoned_run_explains_itself",
    ),
    ("storage/feature_store.py", "_mark_failed_requires_human"): (
        1,
        "test_an_unexpected_runtime_error_is_recorded_as_the_platform_failing",
    ),
    ("storage/feature_store.py", "_stop_overrunning_run"): (
        1,
        "test_a_feature_stopped_at_its_runtime_ceiling_explains_itself",
    ),
    # Renamed, not new. `resume` used to settle the request and then run the whole feature;
    # `begin_resume` is the settling half, split out so a queue claim can do it and then run
    # one step. The terminal path -- a resume whose clarification rounds are spent -- is part
    # of the settling and came with it unchanged.
    ("workflows/feature_workflow.py", "begin_resume"): (
        1,
        "test_exhausted_clarification_rounds_explain_themselves",
    ),
    ("workflows/feature_workflow.py", "approve_contract_change"): (
        2,
        "test_contract_revision_terminal_paths_explain_themselves",
    ),
    # New in task 29-. A claim is one step now, so the loop that carries a feature from its
    # PRD to its pull requests lives between claims and is durable -- and a durable loop needs
    # something able to end it. This is that stop, and like every other one it has to say what
    # it was and why.
    ("storage/feature_store.py", "stop_unprogressing_run"): (
        1,
        "test_a_step_that_never_finishes_the_feature_is_stopped_by_the_step_ceiling",
    ),
    ("workflows/feature_workflow.py", "reject_contract_change"): (
        1,
        "test_a_rejected_contract_change_explains_itself",
    ),
    # Renamed, not new. `_execute_then_review` ran a whole feature -- one execution pass and
    # then the integration loop until it published; the integration half of it is now the
    # step a single claim runs, and the terminal path it owns came with it unchanged.
    ("workflows/feature_workflow.py", "_step_integration_review"): (
        1,
        "test_a_feature_whose_repositories_all_failed_explains_itself",
    ),
    ("workflows/feature_workflow.py", "_execute_workstreams"): (
        1,
        "test_stage_names_the_failing_child_not_the_publication_that_followed",
    ),
    ("workflows/feature_workflow.py", "_publish_pull_requests"): (
        4,
        "test_every_publication_terminal_path_explains_itself",
    ),
    # New in task 81-. Automatic publication narrowed to the feature that fully landed, so
    # the path that used to publish half a feature and record the other half as unfinished is
    # now a stop of its own: it publishes nothing and asks a person. It ends a feature, so it
    # owes the same account of itself every other stop here does.
    ("workflows/feature_workflow.py", "_hold_publication_for_a_person"): (
        1,
        "test_a_feature_holding_publication_explains_itself",
    ),
    # And the answer to that question is also terminal: the feature it publishes for is still
    # unfinished afterwards, so the record it leaves has to say what was opened and what was
    # not, rather than inheriting the hold's summary and telling the next reader to press a
    # button that now has nothing to do.
    ("workflows/feature_workflow.py", "_publish_awaiting_repositories"): (
        1,
        "test_a_publication_a_person_asked_for_explains_itself",
    ),
}


def test_the_terminal_path_registry_matches_what_the_source_actually_contains() -> None:
    """A terminal path that nothing diagnoses cannot be added without this failing.

    This is the guard the task asks for. It compares a registry against the source itself,
    so the failure modes it catches are the two that matter: a terminal status recorded in a
    function nobody registered, and an extra one added to a function that was already
    registered for fewer.
    """
    derived = _terminal_status_sites()
    registered = {key: count for key, (count, _test) in _TERMINAL_PATH_REGISTRY.items()}
    unregistered = sorted(set(derived) - set(registered))
    assert not unregistered, (
        "these functions record a terminal feature status and no registry entry covers "
        f"them, so nothing proves they say why: {unregistered}"
    )
    stale = sorted(set(registered) - set(derived))
    assert not stale, f"these registry entries no longer name a terminal path: {stale}"
    changed = sorted(key for key in derived if derived[key] != registered[key])
    assert not changed, (
        "these functions gained or lost a terminal path; register each one against a test "
        f"that drives it and reads the record it wrote: {changed}"
    )
    # Seventeen sites across five modules at the time of writing. The seventeenth is the
    # feature runtime ceiling; the repository runtime ceiling is deliberately not among them,
    # because it stops a workstream rather than assigning a feature status -- the feature it
    # leaves unfinished ends through `_execute_workstreams`, which is already registered.
    # Eighteen sites across five modules. The eighteenth is the step ceiling, which task 29-
    # added because a claim advancing a feature by one step makes the loop between claims
    # durable -- and a durable loop needs something able to end it.
    #
    # Seventeen across four modules since task 31-. `InMemoryFeatureControlPlane` reimplemented
    # the feature lifecycle for isolated applications and owned one of these; it is gone, and
    # its refusal path is the one the durable plane has always had.
    # Eighteen since task 81-. `_publish_pull_requests` lost one -- the path that published
    # half a feature and recorded the rest as unfinished -- and two took its place: publication
    # holding for a person, and the publication that person then asks for. Both end a feature
    # and both say why.
    assert sum(derived.values()) == 18


def test_every_registered_terminal_path_names_a_test_that_exists() -> None:
    """A registry entry pointing at nothing would read as coverage and provide none."""
    module = ast.parse(pathlib.Path(__file__).read_text())
    defined = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    missing = sorted({name for _count, name in _TERMINAL_PATH_REGISTRY.values()} - defined)
    assert not missing, f"registry entries name tests that do not exist: {missing}"


# ---------------------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------------------


def _assert_diagnosed(
    state: FeatureWorkflowSnapshot,
    *,
    stage: FailureStage | str | None = None,
    classification: FeatureFailureClassification | None = None,
) -> None:
    """Assert the three things every terminal record owes whoever reads it next."""
    summary = state.failure_summary
    assert summary is not None, "a terminal feature with no diagnosis is unreadable"
    assert summary.diagnostics, "a terminal feature must say something a person can act on"
    assert all(item.strip() for item in summary.diagnostics)
    assert summary.root_classification in {item.value for item in FeatureFailureClassification}, (
        f"{summary.root_classification!r} is not a classification this platform owns"
    )
    if stage is not None:
        assert summary.stage == str(stage)
    if classification is not None:
        assert summary.root_classification == classification.value


class _UnusedRunnerMethods:
    """The runner surface a control plane requires but these scenarios never reach.

    `advance_one_step` is what a queue claim calls, so it delegates to `start`: every double
    below reaches its outcome in one step, and defining the behaviour twice would let the two
    drift. `begin_resume` settles a resume request and reaches nothing, so it passes the state
    through.
    """

    async def advance_one_step(self, *args: object, **kwargs: object) -> Any:
        """One step of these doubles is the whole of them."""
        return await self.start(*args, **kwargs)  # type: ignore[attr-defined]

    async def begin_resume(
        self, state: FeatureWorkflowSnapshot, **_kwargs: object
    ) -> FeatureWorkflowSnapshot:
        """Accept the resume unchanged; these doubles decide in the step, not the request."""
        return state

    async def grant_and_run_one_retry(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def answer_design_conflict(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def resume(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def retry_workstream(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def publish_feature(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def approve_contract_change(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def reject_contract_change(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def approve_repository_repair(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError

    async def reject_repository_repair(self, *_args: object, **_kwargs: object) -> Any:
        raise NotImplementedError


class _RaisingPublisher:
    """Fail publication the way an unwrapped provider or driver error does."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def publish(self, **kwargs: Any) -> list[PullRequestArtifact]:
        """Raise before any pull request is created."""
        del kwargs
        raise self._error


class _PartiallyPublishingPublisher:
    """Open a pull request for one repository and fail on the other, as the real one does."""

    async def publish(self, **kwargs: Any) -> list[PullRequestArtifact]:
        """Raise the partial error the coordinated publisher raises."""
        repositories = list(kwargs["repositories"])
        raise PartialPullRequestError((), [item.repository_id for item in repositories])


class _PlatformDefectChildExecutor:
    """Raise the platform's own database driver error inside one repository's workstream."""

    def __init__(self, repository_id: str = "backend") -> None:
        self._repository_id = repository_id

    async def run(self, **kwargs: Any) -> Any:
        """Fail the named repository with an error that says nothing about its code."""
        if kwargs["repository"].repository_id == self._repository_id:
            raise DBAPIError("SELECT 1", {}, Exception("connection pool exhausted"))
        from workflows.feature_workflow import MockChildWorkstreamExecutor

        return await MockChildWorkstreamExecutor().run(**kwargs)


async def _run_feature(
    feature_id: str,
    **orchestrator_kwargs: Any,
) -> FeatureWorkflowSnapshot:
    """Start one deterministic two-repository feature and return its persisted diagnosis.

    `ensure_feature_failure_summary` is applied on the way out because that is precisely
    what `_replace_state` does to every snapshot it writes. A terminal path that stamps its
    own diagnosis is unaffected -- the funnel leaves an existing one alone -- so this reads
    the same record a durable run would have stored, without a database for each scenario.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state(feature_id, request)
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=RecordingReconnaissance(), **orchestrator_kwargs
    )
    return ensure_feature_failure_summary(await orchestrator.start(state, credentials=CREDENTIALS))


async def _durable_store(tmp_path: Path, name: str) -> tuple[Database, Any]:
    """Build one SQLite-backed control plane, as the durable-state tests do."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    store = SqlAlchemyFeatureControlPlane(
        database=database,
        mock_runner=FeatureWorkflowOrchestrator(reconnaissance=RecordingReconnaissance()),
    )
    return database, store


async def _accepted_feature(store: Any, *, feature_id: str, status: FeatureWorkflowStatus) -> str:
    """Accept one feature and move it into a status the sweeps consider live.

    The queue entry is settled first. A feature still waiting to be claimed is deliberately
    not abandoned -- the dispatcher has not started it yet -- so leaving the entry queued
    would make every recovery scenario below a no-op that looks like a passing test.
    """
    payload = dict(feature_payload())
    payload["feature_id"] = feature_id
    request = StartFeatureRequest.model_validate(payload)
    await store.start(
        request,
        idempotency_key=f"{feature_id}-key",
        credentials=CREDENTIALS,
        owner_id="platform-admin",
    )
    await store.queue.finish(feature_id, succeeded=True)
    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.status = status
    state.updated_at = datetime.now(UTC) - timedelta(hours=4)
    await store._replace_state(feature_id, state, "feature_started")
    return feature_id


# ---------------------------------------------------------------------------------------
# 9.2 -- publication failures
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_publication_terminal_path_explains_itself() -> None:
    """Every way `_publish_pull_requests` can end a feature now records a diagnosis.

    The generic handler logged `error_type` to structlog, set the status and returned. The
    summary was assembled afterwards from a state carrying no error type and no diagnostics,
    so it read `unclassified_failure` with an empty array -- three of the fifty measured
    failures, each a feature whose publication broke and which could say nothing about it.
    Container logs rotate; the database is the only durable record.
    """
    # 1. The publisher raised something nobody anticipated.
    broken = await _run_feature(
        "publication-raises",
        pull_request_publisher=_RaisingPublisher(
            DBAPIError("INSERT", {}, Exception("provider driver failed"))
        ),
    )
    assert broken.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    _assert_diagnosed(
        broken,
        stage=FailureStage.PULL_REQUEST_PUBLICATION,
        classification=FeatureFailureClassification.PLATFORM_DEFECT,
    )
    summary = broken.failure_summary
    assert summary is not None
    assert any("defect in the platform" in item for item in summary.diagnostics)
    # The repositories publication was running for, so an operator knows where to look for a
    # branch or pull request this run may have left behind.
    assert any("backend" in item and "frontend" in item for item in summary.diagnostics)

    # 2. Publication completed only partially.
    partial = await _run_feature(
        "publication-partial", pull_request_publisher=_PartiallyPublishingPublisher()
    )
    _assert_diagnosed(
        partial,
        stage=FailureStage.PULL_REQUEST_PUBLICATION,
        classification=FeatureFailureClassification.PUBLICATION_FAILURE,
    )
    partial_summary = partial.failure_summary
    assert partial_summary is not None
    assert any("backend" in item for item in partial_summary.diagnostics)

    # 3. A pull request this run reported creating could not be read back.
    unverified = await _run_feature(
        "publication-unverified", pull_request_publisher=PublishOnlyPublisher()
    )
    _assert_diagnosed(
        unverified,
        stage=FailureStage.PULL_REQUEST_PUBLICATION,
        classification=FeatureFailureClassification.PULL_REQUEST_UNVERIFIED,
    )

    # 4. Everything published, and the integration review never approved it.
    unapproved = await _run_feature(
        "publication-unapproved",
        integration_reviewer=RejectingIntegrationReviewer(responsible_repository_id=None),
    )
    _assert_diagnosed(
        unapproved,
        stage=FailureStage.INTEGRATION_REVIEW,
        classification=FeatureFailureClassification.INTEGRATION_REVIEW_UNSATISFIED,
    )


@pytest.mark.asyncio
async def test_a_feature_holding_publication_explains_itself() -> None:
    """A required repository never reached a review, so nothing published. Say both halves.

    This path used to publish the approved sibling and record the other half as unfinished.
    Task 81- narrowed automatic publication to the feature that fully landed, so what it
    records now is a *hold* -- and a hold that does not name what is waiting is the silent
    partial feature this platform already shipped once. No publication stage here on purpose:
    publication did not fail, and the repository that never reached it supplies both the
    stage and the classification. See the stage-attribution test for why that matters.
    """
    unfinished = await _run_feature(
        "publication-unfinished", child_executor=OneRepositoryFailsExecutor()
    )

    assert unfinished.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    _assert_diagnosed(unfinished, stage=FailureStage.CHILD_WORKFLOWS)
    summary = unfinished.failure_summary
    assert summary is not None
    # What did not land, what is ready to open, and what to do about it -- all three, in the
    # durable record an operator reads first rather than only on a screen.
    assert any("backend" in item for item in summary.diagnostics)
    assert any("Ready to open on request: frontend" in item for item in summary.diagnostics)
    assert any("Publish this feature from the console" in item for item in summary.diagnostics)


@pytest.mark.asyncio
async def test_a_publication_a_person_asked_for_explains_itself() -> None:
    """The answer to the hold is terminal too, and records what it opened and what it did not.

    The feature is still unfinished afterwards -- a person publishing the half that works
    does not make the half that failed land -- so this must not inherit the hold's summary,
    which tells its reader to press a button that now has nothing left to do.
    """
    unfinished = await _run_feature(
        "publication-by-request", child_executor=OneRepositoryFailsExecutor()
    )

    published = await FeatureWorkflowOrchestrator().publish_feature(
        unfinished,
        requested_by="an operator (via platform key)",
        reason="the backend is not going to land",
        credentials=CREDENTIALS,
    )

    assert published.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    _assert_diagnosed(published)
    summary = published.failure_summary
    assert summary is not None
    assert any("an operator (via platform key)" in item for item in summary.diagnostics)
    assert any("example/frontend" in item for item in summary.diagnostics)
    # And the hold's own advice is gone, rather than left telling somebody to publish work
    # that is now open.
    assert not any("Publish this feature from the console" in item for item in summary.diagnostics)


# ---------------------------------------------------------------------------------------
# 9.3 -- stage attribution
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_names_the_failing_child_not_the_publication_that_followed() -> None:
    """A repository that failed its own work is not filed under the stage that ran next.

    The exact sequence the live data shows: one repository fails, an approved sibling is
    published afterwards, and the feature ends needing a human. `stage` was read off
    `current_agent`, which publication had by then set to `github` -- so seventeen of
    ninety-nine recorded failures were attributed to `github` while only five of them were
    actually publication failures.

    Since task 81- the sibling is published on request rather than automatically, so the
    sequence is driven here rather than waited for. It is the same sequence and the same
    trap: by the end, `github` is the agent and the stage must still be the repository's.
    """
    held = await _run_feature("stage-attribution", child_executor=OneRepositoryFailsExecutor())

    assert held.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    result = await FeatureWorkflowOrchestrator().publish_feature(
        held,
        requested_by="an operator (via platform key)",
        reason="the backend is not going to land",
        credentials=CREDENTIALS,
    )

    # The sibling really was published afterwards, so this is the sequence the defect needed.
    assert result.child_workflows["frontend"].pull_request_artifact_id is not None
    assert result.current_agent == "github"
    summary = result.failure_summary
    assert summary is not None
    assert summary.stage == FailureStage.CHILD_WORKFLOWS.value
    assert summary.stage != "github"
    assert summary.repository_id == "backend"
    # And the classification is the repository's own, not "the feature was unfinished".
    assert summary.root_classification in {item.value for item in FailureClassification}
    _assert_diagnosed(result)


def test_the_advice_a_git_outage_carries_does_not_send_anyone_to_the_provider() -> None:
    """`retryable` is one property; "resume it" was paired with one service's sentence.

    A planning-stage fault has no workstream triage and no retry refusal to quote, so this is
    the line the summary offers -- and it read "the provider call failed transiently" for a
    clone that never reached a provider.
    """
    state = _stopped_feature(current_agent="feature_planner")
    diagnosed = ensure_feature_failure_summary(
        state,
        stage=FailureStage.FEATURE_PLANNER,
        classification=FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE,
        diagnostics=["The Git operation did not complete (GitAdapterError)."],
    )
    summary = diagnosed.failure_summary
    assert summary is not None
    assert summary.retryable
    assert "Git remote is reachable" in summary.next_action
    assert "provider call" not in summary.next_action
    # The provider's own wording is untouched, which is what makes the split worth having.
    provider = ensure_feature_failure_summary(
        _stopped_feature(current_agent="feature_planner"),
        stage=FailureStage.FEATURE_PLANNER,
        classification=FeatureFailureClassification.PROVIDER_UNAVAILABLE,
        diagnostics=["The model provider did not answer."],
    ).failure_summary
    assert provider is not None
    assert "the provider call failed transiently" in provider.next_action


def test_stage_and_agent_no_longer_both_mean_current_agent() -> None:
    """`agent` may still name who was assigned; `stage` may not be derived from it."""
    state = _stopped_feature(current_agent="github")
    diagnosed = ensure_feature_failure_summary(
        state, classification=FeatureFailureClassification.PLATFORM_DEFECT
    )
    summary = diagnosed.failure_summary
    assert summary is not None
    assert summary.agent == "github"
    assert summary.stage == FailureStage.CHILD_WORKFLOWS.value


# ---------------------------------------------------------------------------------------
# 9.4 -- platform defects
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_platform_defect_at_a_child_boundary_says_the_platform_failed() -> None:
    """A database driver error is not a finding about somebody's repository.

    One recorded failure carried `root_classification = "DBAPIError"` and the single
    diagnostic "The child workstream raised DBAPIError and could not complete." The feature's
    owner was asked to decide what to do about their repository on the strength of this
    platform's connection pool.
    """
    result = await _run_feature(
        "platform-defect-child", child_executor=_PlatformDefectChildExecutor()
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    child = result.child_workflows["backend"]
    assert child.failure_classification == FeatureFailureClassification.PLATFORM_DEFECT.value
    summary = result.failure_summary
    assert summary is not None
    assert summary.root_classification == FeatureFailureClassification.PLATFORM_DEFECT.value
    # The type name is kept -- it is platform-owned and it is what identifies the failure --
    # but it is no longer the classification, and the sentence says whose problem this is.
    assert any("DBAPIError" in item for item in child.blocking_issues)
    assert any("defect in the platform" in item for item in summary.diagnostics)
    # And nobody is asked a question about the requirement.
    assert "requirement" not in summary.next_action.lower()


@pytest.mark.asyncio
async def test_a_provider_that_did_not_answer_is_not_called_a_platform_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two are distinguished, because the actions they call for are different.

    The base is patched to zero for the reason the Git-outage test below patches it: driving
    the fault allowance to exhaustion serves the whole 5/10/20/40 tail, so this test really
    slept about seventy-six seconds on every run of the suite.
    """
    from adapters.llm_adapter import LLMAdapterError

    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)

    class _ProviderDown:
        async def run(self, **kwargs: Any) -> Any:
            del kwargs
            msg = "the provider did not answer"
            raise LLMAdapterError(msg)

    result = await _run_feature("provider-down", child_executor=_ProviderDown())
    child = result.child_workflows["backend"]
    assert child.failure_classification == FeatureFailureClassification.PROVIDER_UNAVAILABLE.value
    assert any("provider did not answer" in item for item in child.blocking_issues)
    assert any("LLMAdapterError" in item for item in child.blocking_issues)


@pytest.mark.asyncio
async def test_a_git_outage_names_git_and_not_the_model_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AB-Feature-190's trail, without the misdirection it actually carried.

    Its attempt-0 clones died on a forty-second GitHub outage. Four fault retries were spent
    and every sentence in the record read "The model provider did not answer
    (GitAdapterError)" -- so an operator went looking at a model provider that had answered
    nothing because it had never been asked, and found the real cause only by correlating
    clone timestamps by hand.

    54- Part 2 fixed the prose and left the classification, on the reasoning that the field
    encodes retryability and a Git outage is as retryable as a provider one. Both halves of
    that were true and the conclusion was still wrong: the surfaces that key on the
    classification rather than on the prose -- the feature summary below, its next action,
    the logbook's one-liner -- went on naming the model provider. So the classification is
    its own value now, and the retryable verdict it used to carry comes with it.
    """
    from adapters.git_adapter import GitAdapterError

    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)

    class _GitDown:
        async def run(self, **kwargs: Any) -> Any:
            del kwargs
            msg = "git clone failed (GIT_CLONE_FAILED_EXIT_128)."
            raise GitAdapterError(msg, diagnostics=[msg])

    result = await _run_feature("git-down", child_executor=_GitDown())
    child = result.child_workflows["backend"]
    assert child.failure_classification == (
        FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE.value
    )
    assert any("Git operation did not complete" in item for item in child.blocking_issues)
    assert any("GitAdapterError" in item for item in child.blocking_issues)
    assert not any("model provider did not answer" in item for item in child.blocking_issues)
    # The feature-level record the operator opens first carries it too, and still says a
    # resume is the thing to do -- which is the semantics that had to survive the split.
    summary = result.failure_summary
    assert summary is not None
    assert summary.root_classification == (
        FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE.value
    )
    assert summary.retryable
    # And the question somebody is handed names the service they can go and check, in both
    # halves of it: the fault exhaustion's triage is what an escalation reads first.
    attempt = next(
        artifact
        for artifact in reversed(result.artifacts)
        if isinstance(artifact, ChildWorkflowResultArtifact) and artifact.repository_id == "backend"
    )
    question = dict(attempt.metadata)["operator_question"]
    assert question.startswith("The Git remote failed")
    assert "Is the Git remote healthy" in question
    assert "model provider" not in question


@pytest.mark.asyncio
async def test_an_unclassified_child_fault_no_longer_reaches_the_summary_unclassified() -> None:
    """`_failed_child_execution`'s fallback classifies rather than leaving the field empty.

    Ten of the fifty measured failures read `unclassified_failure`, every one of them a child
    that raised something with no classification attached. The value said nothing, and it was
    the first thing an operator saw.
    """

    class _RaisesSomethingOrdinary:
        async def run(self, **kwargs: Any) -> Any:
            del kwargs
            msg = "the workstream could not complete"
            raise RuntimeError(msg)

    result = await _run_feature("unclassified-child", child_executor=_RaisesSomethingOrdinary())

    summary = result.failure_summary
    assert summary is not None
    assert summary.root_classification != "unclassified_failure"
    assert summary.root_classification == FeatureFailureClassification.PLATFORM_DEFECT.value
    for child in result.child_workflows.values():
        assert child.failure_classification == (FeatureFailureClassification.PLATFORM_DEFECT.value)


# ---------------------------------------------------------------------------------------
# 9.5 -- the refusal invariant
# ---------------------------------------------------------------------------------------


def test_a_reason_saying_work_may_proceed_is_recognised_as_one() -> None:
    """The predicate the persistence funnel enforces, on the exact production wording."""
    assert refusal_states_work_may_proceed("Attempt 3 may proceed with a changed strategy.")
    assert not refusal_states_work_may_proceed(
        "The implementation_retry_count budget of 4 is exhausted."
    )
    assert not refusal_states_work_may_proceed(None)


@pytest.mark.asyncio
async def test_a_workstream_that_stopped_says_why_it_stopped(tmp_path: Path) -> None:
    """A stopped workstream carries a reason describing the stop, and only then."""
    result = await _run_feature("refusal-invariant", child_executor=OneRepositoryFailsExecutor())
    backend = result.child_workflows["backend"]
    frontend = result.child_workflows["frontend"]

    assert backend.status is ChildWorkflowStatus.FAILED
    assert backend.retry_refusal_reason is not None
    assert not refusal_states_work_may_proceed(backend.retry_refusal_reason)
    # A repository that finished has nothing to refuse.
    assert frontend.retry_refusal_reason is None
    del tmp_path


@pytest.mark.asyncio
async def test_persisting_a_may_proceed_refusal_is_rejected_at_the_funnel(
    tmp_path: Path,
) -> None:
    """No future writer can reintroduce the reason six production workstreams carry.

    Enforced at `_replace_state` rather than at each caller, because that is the one place
    every writer of child state passes through.
    """
    from api.control_plane import WorkflowConflictError

    database, store = await _durable_store(tmp_path, "refusal.db")
    try:
        feature_id = await _accepted_feature(
            store,
            feature_id="feature-refusal-invariant",
            status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
        )
        state = (await store.get_record(feature_id)).state.model_copy(deep=True)
        repository_id = state.repository_specs[0].repository_id
        state.child_workflows[repository_id] = _failed_child(
            repository_id,
            retry_refusal_reason="Attempt 3 may proceed with a changed strategy.",
        )
        state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN

        with pytest.raises(WorkflowConflictError) as raised:
            await store._replace_state(feature_id, state, "feature_failed")
        assert "may proceed" in str(raised.value)
        # And nothing was written: the refusal never reaches the database.
        settled = await store.get_record(feature_id)
        assert settled.state.status is FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    finally:
        await database.drop_schema()
        await database.dispose()


# ---------------------------------------------------------------------------------------
# 9.6 -- empty diagnostics are impossible
# ---------------------------------------------------------------------------------------


def _failed_child(repository_id: str, **overrides: Any) -> ChildWorkflowReference:
    """Build one stopped workstream reference with nothing a summary can lean on."""
    return ChildWorkflowReference.model_validate(
        {
            "child_workflow_id": f"{repository_id}-child",
            "repository_id": repository_id,
            "workstream_id": repository_id,
            "status": ChildWorkflowStatus.FAILED,
            "branch_name": f"feature/{repository_id}",
            "workspace_path": f"/workspaces/{repository_id}",
            "retry_count": 1,
            **overrides,
        }
    )


def _stopped_feature(
    *,
    current_agent: str | None = "github",
    with_child: bool = True,
    child_classification: str | None = "implementation_missing",
    blocking_issues: Sequence[str] = (),
) -> FeatureWorkflowSnapshot:
    """Build one terminal snapshot with as little diagnostic evidence as asked for."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("diagnostics-floor", request)
    state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    state.current_agent = current_agent
    if not with_child:
        return state
    repository_id = state.repository_specs[0].repository_id
    state.child_workflows = {
        repository_id: _failed_child(
            repository_id,
            failure_classification=child_classification,
            blocking_issues=list(blocking_issues),
        )
    }
    return state


def test_a_summary_with_no_evidence_at_all_still_says_something() -> None:
    """The empty diagnostics array five recorded failures carry cannot be produced.

    A caller that supplies nothing and has no failed child is a bug in that caller. It is
    loud here; in production it falls back to the sentence its classification is worth,
    because a record that says nothing is worse than one that says only its category.
    """
    state = _stopped_feature(with_child=False, current_agent=None)
    diagnosed = ensure_feature_failure_summary(state)
    summary = diagnosed.failure_summary
    assert summary is not None
    assert summary.diagnostics
    assert summary.root_classification == FeatureFailureClassification.PLATFORM_DEFECT.value
    assert "defect in the platform" in summary.diagnostics[0]


def test_a_previously_recorded_empty_diagnosis_is_repaired_rather_than_carried_forward() -> None:
    """An existing summary is left alone, except for the one thing that makes it useless."""
    state = _stopped_feature(with_child=False, current_agent=None)
    first = ensure_feature_failure_summary(
        state, classification=FeatureFailureClassification.EXECUTOR_STOPPED
    )
    assert first.failure_summary is not None
    emptied = first.model_copy(
        update={"failure_summary": first.failure_summary.model_copy(update={"diagnostics": []})}
    )

    repaired = ensure_feature_failure_summary(emptied)

    summary = repaired.failure_summary
    assert summary is not None
    assert summary.diagnostics
    # Everything else about the existing diagnosis is preserved.
    assert summary.root_classification == FeatureFailureClassification.EXECUTOR_STOPPED.value
    assert summary.recorded_at == first.failure_summary.recorded_at


def test_a_terminal_paths_own_explanation_is_no_longer_dropped_by_a_failed_child() -> None:
    """What the path recorded and what the repository reported both reach the summary.

    The two used to be alternatives -- `child_diagnostics or supplied` -- so an abandoned
    run's careful account of what its executor had reached was discarded whenever any
    repository had also failed, which is the common case.
    """
    state = _stopped_feature(blocking_issues=["The repository rejected its own lint run."])
    diagnosed = ensure_feature_failure_summary(
        state,
        stage=FailureStage.RUN_RECOVERY,
        diagnostics=["The executor holding this feature stopped without a terminal status."],
    )
    summary = diagnosed.failure_summary
    assert summary is not None
    assert summary.diagnostics[0].startswith("The executor holding this feature stopped")
    assert "The repository rejected its own lint run." in summary.diagnostics


# ---------------------------------------------------------------------------------------
# 9.7 -- regression
# ---------------------------------------------------------------------------------------


def test_a_classified_child_failure_keeps_the_summary_it_always_had() -> None:
    """The shape and the classification of an ordinary child failure are unchanged."""
    state = _stopped_feature(blocking_issues=["The requirement expects production source."])
    summary = ensure_feature_failure_summary(state).failure_summary
    assert summary is not None
    assert summary.root_classification == "implementation_missing"
    assert summary.retryable is False
    assert summary.repository_id == next(iter(state.child_workflows))
    assert summary.diagnostics[0] == "The requirement expects production source."
    assert summary.next_action


# ---------------------------------------------------------------------------------------
# Terminal paths, one by one
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausted_clarification_rounds_explain_themselves() -> None:
    """A feature that ran out of clarification rounds says so rather than stopping silently."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("clarification-exhausted", request)
    state.status = FeatureWorkflowStatus.WAITING_FOR_HUMAN
    state.clarification_rounds = state.max_clarification_rounds
    orchestrator = FeatureWorkflowOrchestrator(reconnaissance=RecordingReconnaissance())

    result = await orchestrator.resume(state, answers=[], credentials=CREDENTIALS)

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    _assert_diagnosed(
        result,
        stage=FailureStage.HUMAN_CLARIFICATION,
        classification=FeatureFailureClassification.CLARIFICATION_UNRESOLVED,
    )


@pytest.mark.asyncio
async def test_contract_revision_terminal_paths_explain_themselves() -> None:
    """Both ways an approved contract revision can end a feature record a diagnosis."""
    from tests.test_feature_workflow import ContractChangeInOneSibling

    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("contract-revision-diagnosed", request)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=ContractChangeInOneSibling(), reconnaissance=RecordingReconnaissance()
    )
    paused = await orchestrator.start(state, credentials=CREDENTIALS)
    change_request = next(
        item
        for item in paused.artifacts
        if isinstance(item, ContractChangeRequestArtifact) and item.status == "pending"
    )
    current = next(
        item for item in reversed(paused.artifacts) if isinstance(item, IntegrationContractArtifact)
    )
    revised = current.model_copy(
        update={"artifact_id": "009_integration_contract.v2.json", "contract_version": "2.0.0"}
    )

    # 1. The revision limit is reached.
    at_limit = paused.model_copy(deep=True)
    at_limit.contract_revision_cycles = at_limit.max_contract_revision_cycles
    limited = await orchestrator.approve_contract_change(
        at_limit,
        request_id=change_request.change_request_id,
        updated_contract=revised,
        resolution="Adopt v2.",
        credentials=CREDENTIALS,
    )
    assert limited.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    _assert_diagnosed(
        limited,
        stage=FailureStage.CONTRACT_REVISION,
        classification=FeatureFailureClassification.CONTRACT_REVISION_LIMIT_REACHED,
    )

    # 2. A child has already committed, so a delta rerun cannot prove the branch.
    committed = paused.model_copy(deep=True)
    for key, child in committed.child_workflows.items():
        committed.child_workflows[key] = child.model_copy(
            update={"current_revision": "0123456789abcdef0123456789abcdef01234567"}
        )
    fresh = await orchestrator.approve_contract_change(
        committed,
        request_id=change_request.change_request_id,
        updated_contract=revised,
        resolution="Adopt v2.",
        credentials=CREDENTIALS,
    )
    assert fresh.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    _assert_diagnosed(
        fresh,
        stage=FailureStage.CONTRACT_REVISION,
        classification=(FeatureFailureClassification.CONTRACT_REVISION_REQUIRES_FRESH_BRANCHES),
    )


@pytest.mark.asyncio
async def test_a_rejected_contract_change_explains_itself() -> None:
    """Declining a contract change stops the feature, and says that is what stopped it."""
    from tests.test_feature_workflow import ContractChangeInOneSibling

    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("contract-rejected", request)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=ContractChangeInOneSibling(), reconnaissance=RecordingReconnaissance()
    )
    paused = await orchestrator.start(state, credentials=CREDENTIALS)
    change_request = next(
        item
        for item in paused.artifacts
        if isinstance(item, ContractChangeRequestArtifact) and item.status == "pending"
    )

    result = await orchestrator.reject_contract_change(
        paused,
        request_id=change_request.change_request_id,
        resolution="The contract stands.",
        credentials=CREDENTIALS,
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    _assert_diagnosed(
        result,
        stage=FailureStage.CONTRACT_REVISION,
        classification=FeatureFailureClassification.CONTRACT_REVISION_REJECTED,
    )


@pytest.mark.asyncio
async def test_a_feature_whose_repositories_all_failed_explains_itself() -> None:
    """Nothing survived review, so there was nothing to review together and nothing to open."""

    class _EverythingFails:
        async def run(self, **kwargs: Any) -> Any:
            del kwargs
            msg = "every repository failed"
            raise RuntimeError(msg)

    result = await _run_feature("everything-failed", child_executor=_EverythingFails())

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    # The failing repository owns the stage, because that is where the work stopped.
    _assert_diagnosed(result, stage=FailureStage.CHILD_WORKFLOWS)


@pytest.mark.asyncio
async def test_a_durable_queue_refusal_explains_itself(tmp_path: Path) -> None:
    """A refusal the queue makes without running anything still names its own cause."""
    database, store = await _durable_store(tmp_path, "queue-refusal.db")
    try:
        payload = dict(feature_payload())
        payload["feature_id"] = "feature-queue-refusal"
        request = StartFeatureRequest.model_validate(payload)
        started = await store.start(
            request,
            idempotency_key="queue-refusal-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        feature_id = started.record.state.feature_id

        await store.fail_queued(
            feature_id,
            event="feature_credentials_missing",
            reason="This feature needs a provider key configured for its author.",
            error_type=FeatureFailureClassification.PROVIDER_CREDENTIALS_MISSING.value,
        )

        record = await store.get_record(feature_id)
        summary = record.state.failure_summary
        assert summary is not None
        assert summary.stage == FailureStage.FEATURE_QUEUE.value
        assert summary.root_classification == (
            FeatureFailureClassification.PROVIDER_CREDENTIALS_MISSING.value
        )
        assert summary.diagnostics
        # Left answerable, exactly as before: a missing key is something somebody can fix.
        assert record.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_an_unexpected_runtime_error_is_recorded_as_the_platform_failing(
    tmp_path: Path,
) -> None:
    """A runner raising something nobody anticipated is this platform's defect."""

    class _RunnerThatRaises(_UnusedRunnerMethods):
        async def start(self, *_args: object, **_kwargs: object) -> Any:
            raise DBAPIError("SELECT 1", {}, Exception("pool exhausted"))

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'runtime-defect.db'}")
    await database.create_schema()
    store = SqlAlchemyFeatureControlPlane(database=database, mock_runner=_RunnerThatRaises())
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        started = await store.start(
            request,
            idempotency_key="runtime-defect-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        record = await store.get_record(started.record.state.feature_id)

        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        _assert_diagnosed(record.state, classification=FeatureFailureClassification.PLATFORM_DEFECT)
    finally:
        await database.drop_schema()
        await database.dispose()


# ---------------------------------------------------------------------------------------
# 9.4b -- the queue running out of attempts on weather
# ---------------------------------------------------------------------------------------


class _RunnerThatFaults(_UnusedRunnerMethods):
    """A runner whose every step dies on one external service failing to answer."""

    def __init__(self, error: Exception) -> None:
        """Hold the fault to raise, and count how many steps reached it."""
        self.error = error
        self.steps = 0

    async def start(self, *_args: object, **_kwargs: object) -> Any:
        """Fail the way weather does: nothing about the feature, nothing about the code."""
        self.steps += 1
        raise self.error


async def _exhaust_the_queue_on(tmp_path: Path, name: str, *, error: Exception) -> tuple[Any, Any]:
    """Drive one durable feature until the queue has no attempt left, and read the record."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    runner = _RunnerThatFaults(error)
    store = SqlAlchemyFeatureControlPlane(database=database, mock_runner=runner)
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        started = await store.start(
            request,
            idempotency_key=f"{name}-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        return runner, await store.get_record(started.record.state.feature_id)
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_git_outage_that_exhausts_the_queue_is_not_this_platforms_defect(
    tmp_path: Path,
) -> None:
    """Found and left during PR #39, which fixed only the route the fault loop takes.

    The queue reserves attempts for exactly this: an external service that did not answer.
    When the last one is spent, `fail_exhausted_fault` records the stop -- and it asked
    `classification_of`, which reads the exception's type name. `GitAdapterError` carries no
    transient marker, so a GitHub outage that outlasted the whole budget was filed as
    `platform_defect`: `retryable` false, "inspect the persisted diagnostics and repository
    evidence" as the next action, and a logbook one-liner reading "the platform itself failed
    while running it". Three surfaces sending somebody to debug this codebase over weather.
    """
    from adapters.git_adapter import GitAdapterError

    msg = "git clone failed (GIT_CLONE_FAILED_EXIT_128)."
    runner, record = await _exhaust_the_queue_on(
        tmp_path, "git-exhausted.db", error=GitAdapterError(msg, diagnostics=[msg])
    )

    assert runner.steps == 3, "the entry's whole budget must reach the service before it stops"
    summary = record.state.failure_summary
    assert summary is not None
    assert summary.root_classification == (
        FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE.value
    )
    assert summary.retryable
    # The next action names the service to check, and no longer sends a reader to the
    # repository evidence for a failure that never reached the repository.
    assert "Git remote is reachable" in summary.next_action
    assert "repository evidence" not in summary.next_action
    # And the record says the platform did try, which is the honest part of "it gave up".
    assert any("spent all 3 of the attempts" in item for item in summary.diagnostics)
    assert any("did not answer" in item for item in summary.diagnostics)
    assert not any("defect in the platform" in item for item in summary.diagnostics)
    # Left resumable, because the work already done replays out of the operation journal --
    # which is what the advertised RESUME_WORKFLOW was promising and could not deliver.
    assert record.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN


@pytest.mark.asyncio
async def test_a_provider_outage_that_exhausts_the_queue_says_provider(tmp_path: Path) -> None:
    """The same route, the other service. Symmetric because one discriminator answers both."""
    from adapters.llm_adapter import LLMAdapterError

    _, record = await _exhaust_the_queue_on(
        tmp_path,
        "provider-exhausted.db",
        error=LLMAdapterError(
            "the model provider did not answer", failure_classification="APITimeoutError"
        ),
    )

    summary = record.state.failure_summary
    assert summary is not None
    assert summary.root_classification == FeatureFailureClassification.PROVIDER_UNAVAILABLE.value
    assert summary.retryable
    assert record.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    # Both halves of the pair now open by naming the service; the provider line used to say
    # only "Resume this feature", for a stop that had already spent every attempt it had.
    assert "Check that the model provider is answering" in summary.next_action
    assert any("model provider did not answer" in item for item in summary.diagnostics)
    assert not any("Git remote" in item for item in summary.diagnostics)


@pytest.mark.asyncio
async def test_a_full_worker_volume_that_exhausts_the_queue_still_says_capacity(
    tmp_path: Path,
) -> None:
    """The other caller of this route is unchanged, and must stay that way.

    A workspace-capacity failure is not weather: the classification it declares is the
    honest one, the platform's own retry did not help, and no service is answering or not.
    It reaches `fail_exhausted_fault` through a different `except` and falls through the
    fault admission test, so it keeps `classification_of`.
    """
    capacity = DiagnosedFailure(
        "the worker volume is full",
        classification=FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE,
        diagnostics=["Workspace capacity preflight failed for `/workspaces`."],
    )
    _, record = await _exhaust_the_queue_on(tmp_path, "capacity-exhausted.db", error=capacity)

    summary = record.state.failure_summary
    assert summary is not None
    assert summary.root_classification == (
        FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE.value
    )
    assert not summary.retryable
    assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    # No exhaustion sentence: this route did not establish that anything was transient.
    assert not any("reserves for that" in item for item in summary.diagnostics)


def test_a_fault_classification_comes_from_the_effect_and_never_from_a_type_name() -> None:
    """PR #39's property, now that a second caller depends on it.

    `transient_fault_classification` names a service only where `is_transient_provider_fault`
    has established the operation did not land. Everything else keeps whatever it declares --
    which is what stops a `GitAdapterError` that escaped publication, and may well have
    pushed, from presenting itself as retryable.
    """
    from adapters.git_adapter import GitAdapterError, GitAuthenticationError
    from adapters.llm_adapter import LLMAdapterError

    assert transient_fault_classification(GitAdapterError("clone failed")) is (
        FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE
    )
    assert transient_fault_classification(LLMAdapterError("no answer")) is (
        FeatureFailureClassification.PROVIDER_UNAVAILABLE
    )
    # A remote that refused the credential answered, and it answers the same way every time.
    assert transient_fault_classification(GitAuthenticationError("403")) is not (
        FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE
    )
    # A provider answer a retry cannot change is not admitted either, so it falls through to
    # `classification_of` and keeps whatever that makes of what it declared -- anything but
    # "the provider did not answer", which is the claim this must never produce.
    assert (
        transient_fault_classification(
            LLMAdapterError("refused", failure_classification="model_refusal")
        )
        is not FeatureFailureClassification.PROVIDER_UNAVAILABLE
    )
    # And no type name reaches the Git value on its own, which is the invariant
    # `normalize_classification` holds.
    assert normalize_classification("GitAdapterError") is (
        FeatureFailureClassification.PLATFORM_DEFECT
    )
    # The sentences follow the same admission test, so nothing that was not confirmed
    # transient gets told the platform spent an allowance on it.
    assert exhausted_fault_diagnostics(GitAuthenticationError("403"), attempts=3) == ()
    git = exhausted_fault_diagnostics(GitAdapterError("clone failed"), attempts=3)
    assert any("The Git remote did not answer" in item for item in git)
    assert any("the Git remote is answering" in item for item in git)


def _figma_weather(status: int, *, retry_after_seconds: int | None = None) -> Exception:
    """One Figma refusal of the shape the design step's fault allowance is spent on."""
    from adapters.figma_adapter import FigmaClientError, FigmaFailureMode

    return FigmaClientError(
        f"figma call refused (figma_transport_{status}) on nodes",
        mode=FigmaFailureMode.TRANSPORT,
        error_code=f"figma_transport_{status}",
        endpoint="nodes",
        provider_status=status,
        retry_after_seconds=retry_after_seconds,
    )


def test_a_design_source_outage_names_the_design_source() -> None:
    """The third external service is named as itself, in the sentence and the classification.

    Both AB-Feature-227 and AB-Feature-228 ended on a Figma 429 and told their operator to
    confirm the *model provider* was answering: `_fault_source_name` and `_fault_classification`
    branched on Git-remote versus model-provider and nothing else. Two surfaces, because they
    are read by different readers -- fixing only the sentence is how the identical bug shipped
    for GitHub in 54- Part 2 and then again here.
    """
    from storage.external_operation_store import ExternalOperationError

    fault = _figma_weather(429)
    assert transient_fault_classification(fault) is (
        FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE
    )
    sentences = exhausted_fault_diagnostics(fault, attempts=3)
    assert any("The design source did not answer" in item for item in sentences)
    assert any("the design source is answering" in item for item in sentences)
    assert not any("model provider" in item for item in sentences)

    # And through the wrapper it actually arrives in: the resolver's fault reaches the step
    # inside an `ExternalOperationError`, so a top-level type check would have seen nothing.
    try:
        try:
            raise _figma_weather(429)
        except Exception as inner:  # noqa: BLE001 - the chain is the point
            msg = "external operation failed before a confirmed result"
            raise ExternalOperationError(msg) from inner
    except ExternalOperationError as wrapped:
        assert transient_fault_classification(wrapped) is (
            FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE
        )

    # The classification is retryable and carries its own sentence and next action, which is
    # what every classification-keyed surface reads instead of the narrative above.
    assert is_retryable_classification(FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE.value)
    design = fallback_diagnostic(FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE)
    assert "design source" in design
    assert "model provider" not in design.replace("neither the model provider", "")


def test_a_design_source_says_how_long_it_wants_to_be_left_alone() -> None:
    """A `Retry-After` becomes the sentence that makes the resume instruction actionable.

    AB-Feature-228's 429 carried `Retry-After: 261044` -- about three days -- and the record
    told its operator to resume. The wait is the one provider-authored number worth keeping,
    and it is stated beside the instruction rather than instead of it: a resume is still what
    clears this, and this says when it can work.
    """
    days = exhausted_fault_diagnostics(_figma_weather(429, retry_after_seconds=261_044), attempts=3)
    assert any("about another 3.0 days" in item for item in days)
    assert any("spend this feature's attempts against the same refusal" in item for item in days)

    minutes = exhausted_fault_diagnostics(_figma_weather(429, retry_after_seconds=900), attempts=3)
    assert any("about another 15 minutes" in item for item in minutes)

    # Silent where the service said nothing, rather than inventing a deadline. A model-provider
    # fault never carries one at all, so its record reads exactly as it always did.
    from adapters.llm_adapter import LLMAdapterError

    silent = exhausted_fault_diagnostics(_figma_weather(429), attempts=3)
    assert len(silent) == 2
    assert len(exhausted_fault_diagnostics(LLMAdapterError("no answer"), attempts=3)) == 2


@pytest.mark.asyncio
async def test_an_abandoned_run_explains_itself(tmp_path: Path) -> None:
    """A feature whose executor stopped says what it had reached and what was lost."""
    database, store = await _durable_store(tmp_path, "abandoned.db")
    try:
        feature_id = await _accepted_feature(
            store,
            feature_id="feature-abandoned-run",
            status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
        )

        reconciled = await store.reconcile_abandoned_runs(stale_after_seconds=60)

        assert feature_id in reconciled
        record = await store.get_record(feature_id)
        _assert_diagnosed(
            record.state,
            stage=FailureStage.RUN_RECOVERY,
            classification=FeatureFailureClassification.EXECUTOR_STOPPED,
        )
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_feature_stopped_at_its_runtime_ceiling_explains_itself(tmp_path: Path) -> None:
    """A run stopped by the clock is neither a platform defect nor a repository finding.

    The distinction this record has to carry. `EXECUTOR_STOPPED` next door means the process
    died; this one means the process was alive, writing, and possibly doing fine -- and was
    stopped anyway because it had run longer than this deployment pays for. Recording either
    as the other sends somebody to debug the wrong thing.
    """
    database, store = await _durable_store(tmp_path, "runtime-ceiling.db")
    try:
        feature_id = await _accepted_feature(
            store,
            feature_id="feature-runtime-ceiling",
            status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
        )
        # The queue entry is what carries the runtime clock, so it has to say when a worker
        # claimed this feature rather than when it was submitted.
        async with database.session() as session:
            await session.execute(
                update(FeatureExecutionQueueModel)
                .where(FeatureExecutionQueueModel.feature_id == feature_id)
                .values(started_at=datetime.now(UTC) - timedelta(hours=10))
            )
            await session.commit()

        assert await store.stop_overrunning_runs(runtime_limit_seconds=21_600.0) == [feature_id]

        record = await store.get_record(feature_id)
        _assert_diagnosed(
            record.state,
            stage=FailureStage.FEATURE_RUNTIME,
            classification=FeatureFailureClassification.FEATURE_RUNTIME_LIMIT_REACHED,
        )
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_an_unconfirmed_external_effect_explains_itself(tmp_path: Path) -> None:
    """An interrupted push nobody could confirm asks about the repository, and says why."""
    database, store = await _durable_store(tmp_path, "unconfirmed.db")
    try:
        feature_id = await _accepted_feature(
            store,
            feature_id="feature-unconfirmed-effect",
            status=FeatureWorkflowStatus.CREATING_PULL_REQUESTS,
        )

        flagged = await store.flag_unconfirmed_external_effect(
            feature_id,
            operation_type=ExternalOperationType.PUSH_BRANCH.value,
            repository_id="backend",
            reason="the push was interrupted before a confirmed result",
        )

        assert flagged
        record = await store.get_record(feature_id)
        _assert_diagnosed(
            record.state,
            stage=FailureStage.OPERATION_RECOVERY,
            classification=FeatureFailureClassification.UNCONFIRMED_EXTERNAL_EFFECT,
        )
        summary = record.state.failure_summary
        assert summary is not None
        assert any("Check the repository" in item for item in summary.diagnostics)
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_defect_in_the_recovery_sweep_is_not_reported_as_the_repositorys_problem(
    tmp_path: Path,
) -> None:
    """The sweep throwing is ours, and escalating it must say so.

    `RecoveryService._recover_incomplete_operations` catches any exception its own recovery
    code raises and stamps the operation `UNKNOWN_EXTERNAL_STATE` +
    `MANUAL_REVIEW_REQUIRED`, which now escalates the owning feature. Nothing was learned
    about the provider in that case -- a defect here threw -- so telling the feature's owner
    to go and check their repository puts this platform's bug in front of them as theirs.
    """
    database, store = await _durable_store(tmp_path, "recovery-defect.db")
    try:
        feature_id = await _accepted_feature(
            store,
            feature_id="feature-recovery-defect",
            status=FeatureWorkflowStatus.CREATING_PULL_REQUESTS,
        )

        flagged = await store.flag_unconfirmed_external_effect(
            feature_id,
            operation_type=ExternalOperationType.PUSH_BRANCH.value,
            repository_id="backend",
            reason="recovery could not reconcile the operation",
            error_code=RECOVERY_DEFECT_ERROR_CODE,
        )

        assert flagged
        record = await store.get_record(feature_id)
        _assert_diagnosed(
            record.state,
            stage=FailureStage.OPERATION_RECOVERY,
            classification=FeatureFailureClassification.PLATFORM_DEFECT,
        )
        summary = record.state.failure_summary
        assert summary is not None
        assert any("The platform failed here" in item for item in summary.diagnostics)
        assert any("recovery pass raised" in item for item in summary.diagnostics)
        # The operation is still worth checking; what changed is whose fault it says it is.
        assert any("has to be checked" in item for item in summary.diagnostics)
        assert not any("Check the repository for a branch" in item for item in summary.diagnostics)
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_the_recovery_sweep_marks_its_own_failures_with_the_code_that_says_so(
    tmp_path: Path,
) -> None:
    """The two kinds of unresolved operation are distinguishable in the journal."""
    from services.recovery_service import RecoveryService

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'sweep.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database=database, stale_after_seconds=0.001)
    try:
        operation = await journal.create_operation(
            workflow_id="feature-sweep",
            feature_id="feature-sweep",
            child_workflow_id=None,
            repository_id="backend",
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key="sweep-push-1",
            input_fingerprint="sweep-push-1",
            max_attempts=1,
            safe_metadata={},
        )
        await journal.claim_operation(operation.operation_id)
        await asyncio.sleep(0.01)

        class _RecoveryThatRaises(RecoveryService):
            async def _recover(self, operation: Any) -> str:
                msg = "a defect in the recovery code"
                raise RuntimeError(msg)

        service = _RecoveryThatRaises(journal=journal, workspace_root=tmp_path)
        summary = await service.recover_incomplete_operations()

        assert summary.failures == 1
        # And the row it wrote is the one the escalation reads, carrying the code that
        # distinguishes a defect here from an effect nobody could confirm.
        stored = next(
            item
            for item in await journal.list_operations_requiring_manual_review()
            if item.operation_id == operation.operation_id
        )
        assert stored.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
        assert stored.compensation_status is CompensationStatus.MANUAL_REVIEW_REQUIRED
        assert stored.error_code == RECOVERY_DEFECT_ERROR_CODE
    finally:
        await database.drop_schema()
        await database.dispose()


# ---------------------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_queue_entry_records_what_the_feature_became(tmp_path: Path) -> None:
    """A run that ended in failure does not close its queue entry as a success.

    `execute_queued` returns normally after recording a feature as failed, so the dispatcher
    had nothing but "no exception escaped" to close on. The live queue reads 73 `succeeded`
    entries against 95 features waiting on a human.
    """

    class _RunnerThatFails(_UnusedRunnerMethods):
        async def start(self, state: FeatureWorkflowSnapshot, **_kwargs: object) -> Any:
            failed = state.model_copy(deep=True)
            failed.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
            failed.current_agent = "child_workflows"
            return failed

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'queue-outcome.db'}")
    await database.create_schema()
    store = SqlAlchemyFeatureControlPlane(database=database, mock_runner=_RunnerThatFails())
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        started = await store.start(
            request,
            idempotency_key="queue-outcome-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        feature_id = started.record.state.feature_id
        await drain_feature_queue(store)

        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert await store.feature_run_succeeded(feature_id) is False
    finally:
        await database.drop_schema()
        await database.dispose()


def _workstream_for(repository_id: str) -> Any:
    """Build the minimum workstream a failed-child result needs to be constructed."""
    from artifacts.schemas import RepositoryWorkstreamPlan

    return RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": repository_id,
            "repository_id": repository_id,
            "role": "service",
            "requirement_ids": [],
            "scoped_requirements": [],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Implement the assigned work."],
            "task_ids": [f"{repository_id}-task"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["Review is approved."],
            "test_requirements": ["Run configured validation."],
            "documentation_requirements": [],
            "expected_files_or_areas": [],
            "required": True,
            "implementation_expectations": [],
        }
    )


__all__: list[str] = []


# Referenced so the linter keeps the import that documents what a child reference is.
_CHILD_REFERENCE_TYPE = ChildWorkflowReference


@pytest.mark.asyncio
async def test_a_step_that_never_finishes_the_feature_is_stopped_by_the_step_ceiling(
    tmp_path: Path,
) -> None:
    """The loop between claims is durable now, so something has to be able to end it.

    A step that goes on naming itself would re-queue for ever and spend a real provider call
    every time. This is not a hypothetical shape: it is what a defect in `next_step` or in a
    step's own durable writes produces, and while this change was being built it produced
    exactly two hundred claims against one feature before this stopped it.

    The ceiling is deliberately far above anything a healthy feature reaches -- the assertion
    below is that a feature is stopped *and told why*, not that the number is right.
    """

    class _NeverFinishes(FeatureWorkflowOrchestrator):
        """A step that does nothing durable, so the feature is owed it again next claim."""

        async def advance_one_step(
            self, state: FeatureWorkflowSnapshot, *, credentials: Any
        ) -> FeatureWorkflowSnapshot:
            del credentials
            stalled = state.model_copy(deep=True)
            stalled.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
            stalled.updated_at = datetime.now(UTC)
            return stalled

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'no-progress.db'}")
    await database.create_schema()
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=_NeverFinishes())
    try:
        request = StartFeatureRequest.model_validate(
            {**feature_payload(), "feature_id": "feature-stuck"}
        )
        await store.start(
            request,
            idempotency_key="feature-stuck",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        async with database.session() as session:
            await session.execute(
                update(FeatureExecutionQueueModel)
                .where(FeatureExecutionQueueModel.feature_id == "feature-stuck")
                .values(max_steps=4)
            )
            await session.commit()

        claims = await drain_feature_queue(store)

        async with database.session() as session:
            entry = await session.get(FeatureExecutionQueueModel, "feature-stuck")
        assert entry is not None
        assert claims == entry.max_steps + 1, "the ceiling did not bound the re-queue loop"
        assert entry.status == "failed"
        stopped = (await store.get_record("feature-stuck")).state
        summary = stopped.failure_summary
        assert summary is not None
        assert summary.diagnostics, "a feature stopped by the ceiling said nothing about it"
        assert any("steps this deployment allows" in item for item in summary.diagnostics)
        assert summary.root_classification == (
            FeatureFailureClassification.FEATURE_RUNTIME_LIMIT_REACHED.value
        )
        assert stopped.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert "feature_step_budget_exhausted" in [
            event for *_, event, _details in await store.timeline("feature-stuck")
        ]
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_refused_git_credential_names_it_and_spends_no_fault_retries() -> None:
    """Runs 190 and 191: a PAT that expired at midnight, treated as weather for four rounds.

    The stored GitHub credential, created 2026-08-26, expired at 2026-09-02 00:00 UTC. Clones
    succeeded at 23:58 and failed at 00:04, and from then on every authenticated clone failed
    as an anonymous `GitAdapterError`: four fault retries per repository, reported as "The
    model provider did not answer", two features dead before anybody could correlate the
    timestamps by hand.

    The refusal is raised here by the real `_require_success`, over Git's own stderr, so the
    marker table and the composed sentence are both exercised rather than imitated.
    """
    from adapters.git_adapter import CredentialProvenance
    from adapters.interruptible_git import _require_success

    class _CredentialRefused:
        def __init__(self) -> None:
            self.attempts = 0

        async def run(self, **kwargs: Any) -> Any:
            del kwargs
            self.attempts += 1
            _require_success(
                "remote: Support for password authentication was removed.\n"
                "fatal: Authentication failed for 'https://github.com/example/backend.git/'\n",
                128,
                "git clone",
                credential=CredentialProvenance(
                    provider="github", stored_at=datetime(2026, 8, 26, 9, 14, tzinfo=UTC)
                ),
            )

    executor = _CredentialRefused()
    result = await _run_feature("credential-refused", child_executor=executor)

    # One attempt per repository and not one more. The fault allowance is four; every one of
    # them would have re-asked a question the remote had already answered.
    assert executor.attempts == 2, "a refused credential must not spend the fault allowance"
    child = result.child_workflows["backend"]
    assert (
        child.failure_classification
        == FeatureFailureClassification.PROVIDER_CREDENTIALS_MISSING.value
    )
    diagnosis = "\n".join(child.blocking_issues)
    assert "the stored github credential (stored 2026-08-26)" in diagnosis
    assert "refused by the remote" in diagnosis
    assert "may have expired or had its access revoked" in diagnosis
    # And nobody is sent to the model provider, which was never asked anything.
    assert "model provider" not in diagnosis
    # The remote's own words never reach the record: the marker selected a sentence this
    # platform composed, and the URL Git quoted is not in it.
    assert "github.com/example/backend" not in diagnosis
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


def test_only_an_authentication_shape_is_read_as_a_refusal() -> None:
    """The fault allowance still covers everything that is not a decision.

    Both halves matter and the asymmetry is deliberate. Missing a refusal costs a few wasted
    retries and a misdirected sentence, which is what runs 190 and 191 cost. Reading a 5xx or
    a dropped connection as a refusal would stop a workstream that was going to recover on
    its own, and hand somebody a credential to replace that is perfectly good -- so anything
    not on the marker list keeps the behaviour it has always had.
    """
    from adapters.git_adapter import (
        CredentialProvenance,
        GitAdapterError,
        GitAuthenticationError,
    )
    from adapters.interruptible_git import _require_success
    from workflows.feature_workflow import is_transient_provider_fault

    credential = CredentialProvenance(provider="github", stored_at=None)
    transient = (
        "fatal: unable to access 'https://github.com/example/backend.git/': "
        "The requested URL returned error: 503\n",
        "fatal: unable to access 'https://github.com/example/backend.git/': "
        "Could not resolve host: github.com\n",
        "error: RPC failed; curl 56 Recv failure: Connection reset by peer\n",
        "fatal: the remote end hung up unexpectedly\n",
    )
    for stderr in transient:
        with pytest.raises(GitAdapterError) as raised:
            _require_success(stderr, 128, "git clone", credential=credential)
        assert not isinstance(raised.value, GitAuthenticationError), stderr
        assert is_transient_provider_fault(raised.value), stderr

    refusals = (
        "fatal: Authentication failed for 'https://github.com/example/backend.git/'\n",
        "fatal: unable to access '...': The requested URL returned error: 403\n",
        "remote: Permission to example/backend.git denied to someone.\n",
        "git@github.com: Permission denied (publickey).\n",
    )
    for stderr in refusals:
        with pytest.raises(GitAuthenticationError) as refused:
            _require_success(stderr, 128, "git clone", credential=credential)
        assert not is_transient_provider_fault(refused.value), stderr
        # Without a credential to name, the same output is the generic failure it was
        # before: a local `git add` reading somebody's hook output must not be turned into
        # a claim about a credential nothing in that command used.
        with pytest.raises(GitAdapterError) as ungated:
            _require_success(stderr, 128, "git add")
        assert not isinstance(ungated.value, GitAuthenticationError), stderr


def test_a_refusal_wrapped_in_an_operation_error_is_still_a_refusal() -> None:
    """A clone reaches the child loop wrapped, which is how the first two guards were blind.

    `_deterministic_provider_error` exists because the coding call's adapter error arrives
    inside `ExternalOperationError` and the top-level type carries nothing. The same is true
    of a clone, so this decision is chain-walked too.
    """
    from adapters.git_adapter import GitAuthenticationError
    from storage.external_operation_store import ExternalOperationError
    from workflows.feature_workflow import is_transient_provider_fault

    refusal = GitAuthenticationError("git clone was refused", diagnostics=["refused"])
    try:
        try:
            raise refusal
        except GitAuthenticationError as inner:
            msg = "external operation failed before a confirmed result"
            raise ExternalOperationError(msg) from inner
    except ExternalOperationError as wrapped:
        assert not is_transient_provider_fault(wrapped)
