"""A recurring end-to-end run that asserts the platform's invariants on real work.

Audit risk P1-8, third tier (C5). Every other tier answers a question somebody thought to
ask. This one asks the six questions production has already answered wrongly, on a run that
went all the way through, and reports the answers in a form a regression is visible in
without reading a log.

What it composes, and why each part is the real one:

* the real `FeatureWorkflowOrchestrator`, the real `SqlAlchemyFeatureControlPlane`, the real
  queue and two real dispatchers -- so claims, steps, re-queues and terminal transitions are
  the deployment's;
* the real `LiveChildWorkstreamExecutor` against real git checkouts, with `npm`, `node`,
  `ruff` and `pytest` executed as actual subprocesses;
* a **local bare repository** as each checkout's origin, so branch and push have somewhere
  real to land and nothing can reach a hosting provider;
* a deterministic model double for the Engineer and the Reviewer, because the canary is
  about the orchestration and a model would make it neither cheap nor repeatable;
* its own disposable PostgreSQL database, created from the packaged migrations. It never
  touches the deployment's, and `tests/postgres_support.py` refuses by name to manage one.

What it therefore cannot tell you: whether the *agents* behave. There is no model call and
no remote side effect here. A green canary says the platform did not lose, duplicate or
misreport work -- not that the work was any good.

Run it:

    uv run python -m tests.canary --json canary.json

Exit code 0 when every invariant holds, 1 when one does not, and 2 when the run could not be
performed at all (no PostgreSQL, no toolchain) -- which is deliberately distinct, because a
canary that could not run is not a canary that passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from adapters.github_adapter import MockGitHubService
from api.control_plane import RequestScopedCredentials
from api.feature_schemas import StartFeatureRequest
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_queue import FeatureQueueDispatcher
from services.feature_runtime import LiveChildWorkstreamExecutor
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import (
    ChildWorkflowModel,
    ExternalOperationModel,
    FeatureExecutionQueueModel,
    FeaturePullRequestModel,
    FeatureWorkflowModel,
)
from tests.fixtures.real_repositories import (
    missing_executables,
    node_npm_repository,
    python_uv_repository,
)
from tests.postgres_support import PostgresTier, database_url_for, server_is_reachable
from tests.real_repository_support import RecordedRealProcessRunner
from tests.test_live_child_executor import ScriptedLLMClient
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowOrchestrator,
    GitHubPullRequestPublisher,
)

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

# The runtime ceiling a live deployment gives one feature. Nothing in a canary approaches it;
# what the invariant asserts is that no feature is *sitting* past it without a verdict, which
# is the shape a feature that ran for 571 minutes and stopped saying anything had.
RUNTIME_CEILING_SECONDS = 21_600.0

# The statuses that mean this feature is finished deciding.
_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "failed",
        "failed_requires_human",
        "cancelled",
        "cancelled_with_external_side_effects",
        "retired",
    }
)
_UNFINISHED_CHILD_STATUSES = frozenset({"pending", "running"})
_ATTEMPTED_CHILD_STATUSES = frozenset(
    {"approved", "completed", "failed", "review_rejected", "waiting_for_contract_change"}
)


@dataclass(frozen=True, slots=True)
class Invariant:
    """One question, its answer, and the production violation that made it a question."""

    name: str
    passed: bool
    observed: int
    detail: str
    production_baseline: str

    def as_json(self) -> dict[str, Any]:
        """Return the machine-readable form a CI job reads instead of a log."""
        return asdict(self)


@dataclass(slots=True)
class CanaryReport:
    """Everything one canary run produced, in the order a reader needs it."""

    scenarios: list[dict[str, Any]] = field(default_factory=list)
    invariants: list[Invariant] = field(default_factory=list)
    error: str | None = None
    # Which position the review-scope switch was in for this run. Reported because two
    # canary summaries are only comparable when each says which behaviour it measured.
    bounded_review_scope: bool = False

    @property
    def passed(self) -> bool:
        """A canary passes only when it ran and every invariant held."""
        return self.error is None and all(item.passed for item in self.invariants)

    def as_json(self) -> dict[str, Any]:
        """Return the whole run as one JSON-serialisable object."""
        return {
            "passed": self.passed,
            "error": self.error,
            "bounded_review_scope": self.bounded_review_scope,
            "scenarios": self.scenarios,
            "invariants": [item.as_json() for item in self.invariants],
        }


# --------------------------------------------------------------------------------------
# The work the canary performs
# --------------------------------------------------------------------------------------


class _CanaryChildExecutor:
    """Route each repository to the real live executor against its own real checkout."""

    def __init__(
        self,
        workspaces: dict[str, Path],
        payloads: dict[str, dict[str, Any]],
        reviews: dict[str, str],
    ) -> None:
        """Bind one checkout, one Engineer response and one Reviewer shape per repository."""
        self._workspaces = workspaces
        self._payloads = payloads
        self._reviews = reviews
        self.runner = RecordedRealProcessRunner()
        self._journal: ExternalOperationJournal | None = None
        self._settings: Any = None

    def bind(self, journal: ExternalOperationJournal, settings: Any) -> None:
        """Give the executor the durable journal and workspace policy it runs under."""
        self._journal = journal
        self._settings = settings

    async def run(self, **arguments: Any) -> ChildExecution:
        """Execute one repository through the production child path."""
        assert self._journal is not None and self._settings is not None
        repository_id = str(arguments["repository"].repository_id)
        key = f"{arguments['feature'].feature_id}:{repository_id}"
        child = arguments["child"].model_copy(
            update={
                "workspace_path": str(self._workspaces[key]),
                "branch_name": f"ai/{key.replace(':', '/')}/canary",
                # Provisioning would need a network clone; the checkout is already here, and
                # everything after provisioning is the genuine live path.
                "retry_count": max(int(arguments["child"].retry_count), 1),
            }
        )
        executor = LiveChildWorkstreamExecutor(
            settings=self._settings,
            git_environment={},
            engineer_client=ScriptedLLMClient([self._payloads[key]]),
            reviewer_client=ScriptedLLMClient(
                [_review_payload(arguments["workstream"], self._reviews.get(key, "approve"))]
            ),
            journal=self._journal,
            cancellation_token=MockCancellationToken(),
            process_runner=self.runner,
        )
        return await executor.run(**{**arguments, "child": child})


async def run_canary(root: Path, *, bounded_review_scope: bool = False) -> CanaryReport:
    """Run every scenario, then answer the seven invariants from the durable record.

    ``bounded_review_scope`` selects the position of the review-scope switch for the whole
    run. It exists so the same fixtures can be driven twice and compared, which is the only
    way a behaviour change measured statistically can be judged before it is deployed.
    """
    report = CanaryReport(bounded_review_scope=bounded_review_scope)
    missing = missing_executables("git", "node", "npm")
    if missing:
        report.error = f"the canary needs {', '.join(missing)} on PATH"
        return report
    if not await server_is_reachable():
        report.error = "the canary needs a PostgreSQL server to create its own database on"
        return report

    tier = await PostgresTier.provision()
    name = await tier.create_database()
    database = Database(database_url_for(name))
    try:
        report.scenarios = await _run_scenarios(
            database, root, bounded_review_scope=bounded_review_scope
        )
        report.invariants = await _evaluate(database)
    finally:
        await database.dispose()
        await tier.drop_database(name)
        await tier.dispose()
    return report


async def _run_scenarios(
    database: Database, root: Path, *, bounded_review_scope: bool = False
) -> list[dict[str, Any]]:
    """Drive three features through the real queue: one publishes, one stops, one is judged.

    The third exists for this comparison specifically. Its Engineer produces complete,
    validated work and its Reviewer declines over a finding that names nothing the
    workstream was asked for -- which is the shape 34 of 281 recorded workstreams ended in,
    and the only shape whose outcome the review-scope switch changes.
    """
    workspaces: dict[str, Path] = {}
    payloads: dict[str, dict[str, Any]] = {}
    reviews: dict[str, str] = {}
    scenarios = [
        ("canary-published", {"backend": "node", "service": "python"}, "complete", "approve"),
        ("canary-stopped", {"backend": "node"}, "tests_only", "approve"),
        ("canary-unscoped-review", {"backend": "node"}, "complete", "unscoped_rejection"),
    ]
    for feature_id, repositories, shape, review in scenarios:
        for repository_id, technology in repositories.items():
            key = f"{feature_id}:{repository_id}"
            workspaces[key] = _checkout(root, feature_id, repository_id, technology)
            payloads[key] = _engineer_payload(technology, shape)
            reviews[key] = review

    executor = _CanaryChildExecutor(workspaces, payloads, reviews)
    settings = load_settings(
        workspace_root=root / "workspaces", bounded_review_scope=bounded_review_scope
    )
    journal = ExternalOperationJournal(database)
    executor.bind(journal, settings)
    github = MockGitHubService()
    store = SqlAlchemyFeatureControlPlane(
        database,
        mock_runner=FeatureWorkflowOrchestrator(
            child_executor=executor,
            pull_request_publisher=GitHubPullRequestPublisher(github_service=github),
            # The pre-coding calls journal themselves per feature, exactly as the live
            # composition wires them; the invariant below reads what they leave behind.
            operation_executor_factory=lambda state: ExternalOperationExecutor(
                journal=journal,
                cancellation_token=MockCancellationToken(),
                scope=ExternalOperationScope(
                    workflow_id=state.workflow_id, feature_id=state.feature_id
                ),
            ),
        ),
    )

    for feature_id, repositories, _shape, _review in scenarios:
        await store.start(
            StartFeatureRequest.model_validate(_feature_request(feature_id, repositories)),
            idempotency_key=feature_id,
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )

    async def credentials_for(_owner: str | None) -> RequestScopedCredentials:
        return CREDENTIALS

    # Two workers, because a canary that ran one would never exercise the claim the whole
    # stepping design turns on.
    workers = [
        FeatureQueueDispatcher(queue=store.queue, executor=store, credentials_for=credentials_for)
        for _ in range(2)
    ]
    await asyncio.gather(*(worker.drain() for worker in workers))

    summaries: list[dict[str, Any]] = []
    for feature_id, repositories, shape, review in scenarios:
        record = await store.get_record(feature_id)
        summaries.append(
            {
                "feature_id": feature_id,
                "shape": shape,
                "review": review,
                "repositories": sorted(repositories),
                "status": record.state.status.value,
                "children": {
                    key: child.status.value for key, child in record.state.child_workflows.items()
                },
                # What each workstream cost. A change that bounds review scope is only worth
                # having if the attempts it saves are attempts nobody needed to spend.
                "attempts": {
                    key: child.retry_count + 1
                    for key, child in record.state.child_workflows.items()
                },
                # What the switch is measured on: how many of this feature's findings the
                # platform blocked, advised on, and would have blocked before.
                "review_findings": _review_finding_summary(record),
            }
        )
    return summaries


def _review_finding_summary(record: Any) -> dict[str, int]:
    """Total each repository's recorded finding counts across this feature's attempts."""
    totals = {"total": 0, "blocking": 0, "advisory": 0, "untraceable": 0}
    for artifact in record.state.artifacts:
        counts = getattr(artifact, "review_finding_counts", None)
        if counts is None:
            continue
        totals["total"] += counts.findings_total
        totals["blocking"] += counts.findings_blocking
        totals["advisory"] += counts.findings_advisory
        totals["untraceable"] += counts.findings_untraceable
    return totals


# --------------------------------------------------------------------------------------
# The seven invariants
# --------------------------------------------------------------------------------------


async def _evaluate(database: Database) -> list[Invariant]:
    """Answer every invariant from the rows an operator would query, not from memory."""
    async with database.session() as session:
        features = list(await session.scalars(select(FeatureWorkflowModel)))
        children = list(await session.scalars(select(ChildWorkflowModel)))
        pull_requests = list(await session.scalars(select(FeaturePullRequestModel)))
        operations = list(await session.scalars(select(ExternalOperationModel)))
        entries = list(await session.scalars(select(FeatureExecutionQueueModel)))

    terminal = {item.feature_id for item in features if item.status.value in _TERMINAL_STATUSES}
    return [
        _publication_invariant(children, pull_requests),
        _no_unfinished_child_invariant(children, terminal),
        _classified_summary_invariant(features, terminal),
        _revision_invariant(children),
        _no_duplicate_effect_invariant(operations),
        _runtime_ceiling_invariant(features, entries, terminal),
        _planning_calls_journaled_invariant(features, operations),
        _routing_names_the_tier_invariant(features, children),
    ]


def _routing_names_the_tier_invariant(
    features: Sequence[Any], children: Sequence[Any]
) -> Invariant:
    """Every attempt's persisted routing record names its feature's pinned tier.

    Read from the rows rather than from memory: the tier is a column on the feature and a
    key inside each child's `model_routing`, and cost attribution needs the two to agree on
    every attempt that routed.
    """
    tier_by_feature = {item.feature_id: item.performance_tier for item in features}
    routed = [item for item in children if item.model_routing is not None]
    mismatched = [
        f"{item.feature_id}/{item.repository_id}"
        for item in routed
        if item.model_routing.get("performance_tier") != tier_by_feature.get(item.feature_id)
    ]
    # A run where nothing routed proves nothing, and "zero mismatches" would read as though it
    # had. The count checked is reported so a passing invariant cannot be vacuous unnoticed.
    return Invariant(
        name="every_routing_record_names_the_tier",
        passed=not mismatched and bool(routed),
        observed=len(mismatched),
        detail=(
            f"{len(routed)} routing records each name their feature's pinned tier"
            if not mismatched and routed
            else (
                "no attempt persisted a routing record, so the tier was never recorded"
                if not routed
                else f"{len(mismatched)} routing records disagree with the tier: {mismatched}"
            )
        ),
        production_baseline="live attempts' cost basis was not attributable before tiers",
    )


# The pre-coding model calls every canary feature makes, and must therefore journal. The
# deterministic composition asks no clarification questions and reads no checkouts, so
# grounding and reconnaissance rows are covered by their own suites rather than here.
_JOURNALED_PLANNING_CALLS = (
    ExternalOperationType.RUN_PRODUCT_MANAGER,
    ExternalOperationType.RUN_FEATURE_PLANNER,
)


def _planning_calls_journaled_invariant(
    features: Sequence[Any], operations: Sequence[Any]
) -> Invariant:
    """Every pre-coding model call left a completed row with a start, heartbeat, and end.

    AB-Feature-173's planning stage ran 41 minutes with no record at all while an operator
    asked twice whether it was stuck. The rows this checks are the record that stretch was
    missing -- and they must be complete on a finished feature, because a row that is still
    RUNNING after its feature ended is the observability lying.
    """
    missing: list[str] = []
    for feature in features:
        for kind in _JOURNALED_PLANNING_CALLS:
            rows = [
                item
                for item in operations
                if item.feature_id == feature.feature_id
                and ExternalOperationType(item.operation_type) is kind
            ]
            complete = [
                item
                for item in rows
                if ExternalOperationStatus(item.status) is ExternalOperationStatus.SUCCEEDED
                and item.started_at is not None
                and item.heartbeat_at is not None
                and item.completed_at is not None
            ]
            if not complete:
                missing.append(f"{feature.feature_id}:{kind.value}")
    return Invariant(
        name="every_planning_call_is_journaled",
        passed=not missing,
        observed=len(missing),
        detail=(
            "every pre-coding model call left a completed journal row"
            if not missing
            else f"{len(missing)} planning calls left no completed row: {missing}"
        ),
        production_baseline="AB-Feature-173 planned for 41 minutes with no journaled call",
    )


def _publication_invariant(children: Sequence[Any], pull_requests: Sequence[Any]) -> Invariant:
    """Reviewed work always gets a pull request. Nine live children did not."""
    published = {(item.feature_id, item.repository_id) for item in pull_requests}
    unpublished = [
        f"{item.feature_id}/{item.repository_id}"
        for item in children
        if item.status.value in {"approved", "completed"}
        and item.pull_request_artifact_id is None
        and (item.feature_id, item.repository_id) not in published
    ]
    return Invariant(
        name="every_approved_child_has_a_pull_request",
        passed=not unpublished,
        observed=len(unpublished),
        detail=(
            "every approved child has a pull request"
            if not unpublished
            else f"{len(unpublished)} approved children had no pull request: {unpublished}"
        ),
        production_baseline="9 approved children had no PR",
    )


def _no_unfinished_child_invariant(children: Sequence[Any], terminal: set[str]) -> Invariant:
    """A finished feature leaves nothing running. Sixteen live child rows were left running."""
    stranded = [
        f"{item.feature_id}/{item.repository_id}={item.status.value}"
        for item in children
        if item.feature_id in terminal and item.status.value in _UNFINISHED_CHILD_STATUSES
    ]
    return Invariant(
        name="no_child_left_running_under_a_terminal_feature",
        passed=not stranded,
        observed=len(stranded),
        detail=(
            "no child is running or pending under a terminal feature"
            if not stranded
            else f"{len(stranded)} child rows were left running: {stranded}"
        ),
        production_baseline="16 child rows were left running",
    )


def _classified_summary_invariant(features: Sequence[Any], terminal: set[str]) -> Invariant:
    """Every terminal feature says what stopped it, and gives at least one diagnostic."""
    unexplained: list[str] = []
    for item in features:
        if item.feature_id not in terminal or item.status.value == "completed":
            continue
        summary = (item.state_json or {}).get("failure_summary")
        classified = isinstance(summary, dict) and bool(summary.get("root_classification"))
        diagnosed = isinstance(summary, dict) and bool(summary.get("diagnostics"))
        if not (classified and diagnosed):
            unexplained.append(f"{item.feature_id}={item.status.value}")
    return Invariant(
        name="every_terminal_feature_is_classified_and_diagnosed",
        passed=not unexplained,
        observed=len(unexplained),
        detail=(
            "every terminal feature carries a classification and at least one diagnostic"
            if not unexplained
            else f"{len(unexplained)} terminal features said nothing: {unexplained}"
        ),
        production_baseline="of 50 failures, 10 were unclassified and 5 had no diagnostics",
    )


def _revision_invariant(children: Sequence[Any]) -> Invariant:
    """A workstream that ran an attempt names the revision it ran against."""
    missing = [
        f"{item.feature_id}/{item.repository_id}"
        for item in children
        if item.status.value in _ATTEMPTED_CHILD_STATUSES and not item.current_revision
    ]
    return Invariant(
        name="every_attempted_workstream_has_a_current_revision",
        passed=not missing,
        observed=len(missing),
        detail=(
            "every workstream that ran an attempt records its revision"
            if not missing
            else f"{len(missing)} workstreams had no revision: {missing}"
        ),
        production_baseline="165 of 281 workstreams had no revision",
    )


def _no_duplicate_effect_invariant(operations: Sequence[Any]) -> Invariant:
    """One clone, one commit, one push, one pull request -- per intent, not per attempt.

    Asked of the journal rather than of a counter, because the journal is what the platform
    itself uses to answer it: two rows with one idempotency key is a duplicated effect, and
    a succeeded operation replayed more attempts than it was given is the same thing seen
    from the other side.
    """
    keys: dict[str, int] = {}
    for item in operations:
        keys[item.idempotency_key] = keys.get(item.idempotency_key, 0) + 1
    duplicated = sorted(key for key, count in keys.items() if count > 1)
    overrun = sorted(
        f"{item.operation_type}:{item.attempt}"
        for item in operations
        if item.status.value == "succeeded" and item.attempt > 1
    )
    return Invariant(
        name="no_duplicate_clone_commit_push_or_pull_request",
        passed=not duplicated and not overrun,
        observed=len(duplicated) + len(overrun),
        detail=(
            f"{len(operations)} journalled effects, each with one idempotency key"
            if not duplicated and not overrun
            else f"duplicated: {duplicated}; retried after success: {overrun}"
        ),
        production_baseline="7 of 7 pull-request cross-links were re-attempted after success",
    )


def _runtime_ceiling_invariant(
    features: Sequence[Any], entries: Sequence[Any], terminal: set[str]
) -> Invariant:
    """No feature is left past its runtime ceiling without a verdict.

    The ceiling is clocked from the queue entry's `started_at`, which `advance_to_next_step`
    deliberately preserves across every step. A feature that keeps making small amounts of
    progress must still stop eventually, and this is the check that says whether it did.
    """
    del features
    overrunning: list[str] = []
    for entry in entries:
        if entry.feature_id in terminal or entry.started_at is None:
            continue
        elapsed = (datetime.now(UTC) - entry.started_at).total_seconds()
        if elapsed > RUNTIME_CEILING_SECONDS:
            overrunning.append(f"{entry.feature_id}={int(elapsed)}s")
    return Invariant(
        name="no_feature_exceeds_its_runtime_ceiling_without_a_terminal_state",
        passed=not overrunning,
        observed=len(overrunning),
        detail=(
            f"{len(entries)} queue entries, none past the {int(RUNTIME_CEILING_SECONDS)}s ceiling"
            if not overrunning
            else f"{len(overrunning)} features ran past the ceiling with no verdict: {overrunning}"
        ),
        production_baseline="live features averaged 86 minutes and reached 571",
    )


# --------------------------------------------------------------------------------------
# Fixtures, payloads and the request the canary submits
# --------------------------------------------------------------------------------------


def _checkout(root: Path, feature_id: str, repository_id: str, technology: str) -> Path:
    """Build one real checkout with a real bare repository as its origin."""
    workspace = root / "workspaces" / feature_id / repository_id
    builders = {"node": node_npm_repository, "python": python_uv_repository}
    builders[technology](workspace)
    remote = root / "remotes" / f"{feature_id}-{repository_id}.git"
    remote.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ("git", "init", "--bare", str(remote)), check=True, capture_output=True, timeout=30
    )
    _git(workspace, "remote", "add", "origin", str(remote))
    _git(workspace, "push", "origin", "main")
    _git(workspace, "checkout", "-b", f"ai/{feature_id}/{repository_id}/canary")
    return workspace


def _engineer_payload(technology: str, shape: str) -> dict[str, Any]:
    """Return the deterministic change this repository's Engineer produces."""
    if shape == "tests_only":
        return {
            "summary": "Cover the status route.",
            "files": [
                {
                    "path": "test/status.test.js",
                    "content": (
                        "const { test } = require('node:test');\n"
                        "test('the status route is covered', () => {});\n"
                    ),
                }
            ],
        }
    if technology == "node":
        return {
            "summary": "Report recovery state from the status route.",
            "files": [
                {
                    "path": "src/routes/status.js",
                    "content": (
                        "function statusRoute(request, response) {\n"
                        "  response.json({ status: 'ok', canary: true });\n"
                        "}\n\n"
                        "module.exports = { statusRoute };\n"
                    ),
                },
                {
                    "path": "test/status.test.js",
                    "content": (
                        "const { test } = require('node:test');\n"
                        "const assert = require('node:assert');\n"
                        "const { statusRoute } = require('../src/routes/status');\n\n"
                        "test('the status route answers', () => {\n"
                        "  let body = null;\n"
                        "  statusRoute({}, { json: (value) => { body = value; } });\n"
                        "  assert.deepStrictEqual(body, { status: 'ok', canary: true });\n"
                        "});\n"
                    ),
                },
            ],
        }
    return {
        "summary": "Add the status feed.",
        "files": [
            {
                "path": "app/status_feed.py",
                "content": (
                    '"""Server status feed."""\n\n\n'
                    "def status_feed() -> dict[str, str]:\n"
                    '    """Return the current status feed."""\n'
                    '    return {"status": "ok"}\n'
                ),
            },
            {
                # Edited rather than added: a module nothing refers to is unreachable, which
                # is a rejection the canary is not trying to produce here.
                "path": "app/status.py",
                "content": (
                    '"""Service status."""\n\n'
                    "from app.status_feed import status_feed\n\n\n"
                    "def server_status() -> dict[str, str]:\n"
                    '    """Return the current service status."""\n'
                    "    return status_feed()\n"
                ),
            },
            {
                "path": "tests/test_status_feed.py",
                "content": (
                    '"""Status feed tests."""\n\n'
                    "from app.status_feed import status_feed\n\n\n"
                    "def test_status_feed_reports_the_service_status() -> None:\n"
                    '    """The feed reports what the service reports."""\n'
                    '    assert status_feed() == {"status": "ok"}\n'
                ),
            },
        ],
    }


def _review_payload(workstream: Any, shape: str = "approve") -> dict[str, Any]:
    """Return the review this scenario's reviewer gives, covering its assigned requirements.

    ``unscoped_rejection`` is the production shape of `review_scope_failure`: a correct
    senior-engineer critique of code the attempt touched, deriving from no scoped
    requirement, no contract section, no failed command and no implementation expectation.
    It is what makes the canary's on/off comparison say anything -- an always-approving
    reviewer answers the same in both positions.
    """
    approved = shape == "approve"
    return {
        "verdict": "approved" if approved else "changes_requested",
        "summary": f"{workstream.repository_id} implements its scoped requirements.",
        "requirement_checks": [
            {
                "requirement_id": reference.requirement_id,
                "passed": True,
                "evidence": "The scoped source area contains the implementation.",
            }
            for reference in workstream.scoped_requirements
        ],
        "findings": []
        if approved
        else [
            {
                "finding_id": "review-scope-1",
                "severity": "high",
                "title": "The upsert does not reread the winner on a duplicate key",
                "description": (
                    "ensureRetryState performs findOneAndUpdate with upsert but does not "
                    "catch a duplicate-key result and reread the winner."
                ),
                "recommendation": (
                    "Catch the duplicate-key result and reread the winning document."
                ),
                "file_path": None,
                "line_number": None,
                "finding_category": "code_quality",
            }
        ],
        "architecture_assessment": "The change follows the existing repository layout.",
        "security_assessment": "No credential handling changed.",
        "test_coverage_assessment": "The repository's configured tests cover the change.",
    }


def _feature_request(feature_id: str, repositories: dict[str, str]) -> dict[str, Any]:
    """Return one PRD assigning every repository exactly one scoped requirement."""
    return {
        "feature_id": feature_id,
        # A non-default tier, deliberately: the invariant that every attempt's routing record
        # names its feature's tier only proves threading if the value could not have come
        # from a default.
        "performance_tier": "medium",
        "prd": {
            "title": "Server status",
            "problem_statement": "Operators cannot see server status across services.",
            "goals": ["Expose server status."],
            "user_stories": [
                {
                    "story_id": "status",
                    "persona": "Operator",
                    "need": "see server status",
                    "benefit": "faster triage",
                    "acceptance_criteria": ["Status is visible."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": f"{repository_id}-status",
                    "description": f"Implement the {repository_id} side of server status.",
                    "priority": "must",
                    "acceptance_criteria": [f"The {repository_id} status work is complete."],
                    "dependencies": [],
                }
                for repository_id in repositories
            ],
            "constraints": [],
            "out_of_scope": [],
            "stakeholders": ["Platform"],
        },
        "repositories": [
            {
                "repository_id": repository_id,
                "name": repository_id,
                "role": "backend" if repository_id == "backend" else "service",
                "repository_url": f"https://github.com/example/{repository_id}.git",
                "default_branch": "main",
            }
            for repository_id in repositories
        ],
    }


def _git(root: Path, *arguments: str) -> None:
    """Run one git command against a checkout with no shell."""
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canary and print its machine-readable summary."""
    parser = argparse.ArgumentParser(description="Run the orchestration canary.")
    parser.add_argument("--json", type=Path, default=None, help="write the summary to this file")
    parser.add_argument("--quiet", action="store_true", help="print nothing to stdout")
    parser.add_argument(
        "--bounded-review-scope",
        action="store_true",
        help="require a blocking review finding to name what it derives from",
    )
    arguments = parser.parse_args(argv)

    root = Path(tempfile.mkdtemp(prefix="canary-"))
    try:
        report = asyncio.run(run_canary(root, bounded_review_scope=arguments.bounded_review_scope))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    payload = report.as_json()
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not arguments.quiet:
        print(json.dumps(payload, indent=2))
    if report.error is not None:
        return 2
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - the command-line entry point
    sys.exit(main())
