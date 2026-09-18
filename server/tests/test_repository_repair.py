"""Repository repair as a decision somebody makes, rather than a message they read.

The platform stops a repository whose own checked-in setup will not let it run its checks,
because nothing written there could be validated. What it must not do is quietly fix somebody
else's repository. These tests are about the shape of that boundary: which failures produce a
proposal, what the platform refuses to do with one, and what happens when the repository moves
on after the diagnosis was written.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import ContractChangeRequestArtifact, RepositoryRepairProposalArtifact
from services.cancellation import MockCancellationToken
from services.repository_repair import (
    build_repair_payload,
    current_repairs,
    open_repair_for,
    repair_commands,
    repair_is_stale,
    repairable_issues,
    safe_package_names,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus, RepositoryRepairStatus
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.test_feature_api import feature_payload
from tools.repository_preflight import PreflightIssue
from tools.retry_strategy import FailureClassification
from workflows.feature_workflow import (
    FeatureWorkflowError,
    FeatureWorkflowOrchestrator,
    RepairSupersededError,
    _propose_repository_repairs,
)

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


def issue(
    *,
    category: str = "invalid_lint_configuration",
    severity: str = "critical",
    repairable: bool = True,
) -> dict[str, Any]:
    """Build one preflight finding in the shape the child persists it."""
    return PreflightIssue(
        issue_id=f"{category}-1",
        category=category,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        description="eslint-config-house is required by .eslintrc.json and is not declared",
        evidence="npx eslint . exited 2: Cannot find module 'eslint-config-house'",
        recommended_action="Declare eslint-config-house as a development dependency",
        automatically_repairable=repairable,
    ).model_dump(mode="json")


def stopped_feature(
    *,
    classification: str = FailureClassification.VALIDATION_CONFIGURATION_FAILURE.value,
    issues: list[dict[str, Any]] | None = None,
    revision: str | None = "revision-one",
) -> Any:
    """Return a feature whose backend stopped on its own setup, ready to diagnose."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-repair", request)
    from state.feature_models import ChildWorkflowReference

    state.child_workflows["backend"] = ChildWorkflowReference(
        child_workflow_id="feature-repair:backend",
        repository_id="backend",
        workstream_id="backend",
        status=ChildWorkflowStatus.FAILED,
        branch_name="feature/repair",
        workspace_path="/workspaces/feature-repair/backend",
        retry_count=1,
        blocking_setup_issues=[issue()] if issues is None else issues,
        failure_classification=classification,
        preflight_status="blocked",
        current_revision=revision,
    )
    return state


def test_a_repository_setup_failure_produces_one_proposal() -> None:
    """The finding a person can act on becomes a proposal they can approve."""
    state = stopped_feature()

    _propose_repository_repairs(state)

    repairs = current_repairs(state.artifacts)
    assert len(repairs) == 1
    proposal = next(iter(repairs.values()))
    assert proposal.repository_id == "backend"
    assert proposal.status == "proposed"
    # The stage that found it, not the state it left the repository in. A card promising
    # "found by" displayed "blocked" until these stopped being conflated.
    assert proposal.originating_stage == "repository_preflight"
    assert proposal.proposed_at_revision == "revision-one"
    assert "eslint-config-house" in proposal.detected_problem
    # A setup repair restores the ability to check the code. It must never be described as
    # something that changes what the code does.
    assert proposal.changes_source_logic is False


def test_an_ordinary_coding_failure_produces_no_proposal() -> None:
    """Asking somebody to approve a repository change for a failing test would be wrong."""
    state = stopped_feature(
        classification=FailureClassification.VALIDATION_SOURCE_FAILURE.value,
    )

    _propose_repository_repairs(state)

    assert current_repairs(state.artifacts) == {}


def test_a_finding_no_command_can_fix_produces_no_proposal() -> None:
    """A source-structure problem is not something a deterministic repair addresses."""
    state = stopped_feature(issues=[issue(category="source_structure")])

    _propose_repository_repairs(state)

    assert current_repairs(state.artifacts) == {}


def test_a_repository_that_already_has_an_open_proposal_gets_no_second_one() -> None:
    """Diagnosis runs after every attempt, and must not accumulate duplicates."""
    state = stopped_feature()

    _propose_repository_repairs(state)
    _propose_repository_repairs(state)
    _propose_repository_repairs(state)

    assert len(current_repairs(state.artifacts)) == 1


def test_repairable_issues_ignores_classifications_that_are_not_about_the_repository() -> None:
    """The filter is the domain boundary, so it is asserted directly as well."""
    findings = [PreflightIssue.model_validate(issue())]

    assert repairable_issues(
        findings, classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE
    )
    assert (
        repairable_issues(findings, classification=FailureClassification.IMPLEMENTATION_MISSING)
        == []
    )


def test_a_proposal_is_stale_once_the_repository_moves() -> None:
    """The commands were chosen against a checkout that no longer exists."""
    state = stopped_feature()
    _propose_repository_repairs(state)
    proposal = next(iter(current_repairs(state.artifacts).values()))

    assert repair_is_stale(proposal, current_revision="revision-one") is False
    assert repair_is_stale(proposal, current_revision="revision-two") is True
    # A proposal from before revisions were recorded is not refused; that would strand
    # features that are already stopped.
    assert repair_is_stale(proposal, current_revision=None) is False


@pytest.mark.asyncio
async def test_rejecting_a_repair_records_who_and_why() -> None:
    """A stop that a person decided to leave in place must say that it was decided."""
    state = stopped_feature()
    _propose_repository_repairs(state)
    proposal = next(iter(current_repairs(state.artifacts).values()))
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.reject_repository_repair(
        state,
        repair_id=proposal.repair_id,
        actor_id="alex",
        reason="We are removing that lint rule instead.",
    )

    decided = current_repairs(result.artifacts)[proposal.repair_id]
    assert decided.status == "rejected"
    assert decided.rejected_by == "alex"
    assert decided.rejection_reason == "We are removing that lint rule instead."
    # The original proposal is still there. It is the evidence the decision was made against.
    assert any(
        isinstance(item, RepositoryRepairProposalArtifact) and item.status == "proposed"
        for item in result.artifacts
    )


@pytest.mark.asyncio
async def test_rejecting_a_repair_twice_is_the_same_rejection() -> None:
    """A repeated decision is one state transition, not an error and not a second one."""
    state = stopped_feature()
    _propose_repository_repairs(state)
    proposal = next(iter(current_repairs(state.artifacts).values()))
    orchestrator = FeatureWorkflowOrchestrator()

    once = await orchestrator.reject_repository_repair(
        state, repair_id=proposal.repair_id, actor_id="alex", reason="Not this way."
    )
    twice = await orchestrator.reject_repository_repair(
        once, repair_id=proposal.repair_id, actor_id="alex", reason="Not this way."
    )

    rejections = [
        item
        for item in twice.artifacts
        if isinstance(item, RepositoryRepairProposalArtifact) and item.status == "rejected"
    ]
    assert len(rejections) == 1


@pytest.mark.asyncio
async def test_a_rejected_repair_cannot_then_be_approved() -> None:
    """A decision that has been made is not quietly reopened by a different button."""
    state = stopped_feature()
    _propose_repository_repairs(state)
    proposal = next(iter(current_repairs(state.artifacts).values()))
    orchestrator = FeatureWorkflowOrchestrator()
    rejected = await orchestrator.reject_repository_repair(
        state, repair_id=proposal.repair_id, actor_id="alex", reason="Not this way."
    )

    with pytest.raises(FeatureWorkflowError, match="rejected"):
        await orchestrator.approve_repository_repair(
            rejected,
            repair_id=proposal.repair_id,
            actor_id="alex",
            credentials=CREDENTIALS,
        )


@pytest.mark.asyncio
async def test_a_repair_written_against_an_older_checkout_is_superseded() -> None:
    """Applying it could install something somebody has already removed."""
    state = stopped_feature()
    _propose_repository_repairs(state)
    proposal = next(iter(current_repairs(state.artifacts).values()))
    # The repository moved on: somebody pushed, or an attempt committed.
    state.child_workflows["backend"] = state.child_workflows["backend"].model_copy(
        update={"current_revision": "revision-two"}
    )
    orchestrator = FeatureWorkflowOrchestrator()

    with pytest.raises(RepairSupersededError, match="superseded") as refusal:
        await orchestrator.approve_repository_repair(
            state,
            repair_id=proposal.repair_id,
            actor_id="alex",
            credentials=CREDENTIALS,
        )

    # The refusal carries the state change, so the caller can persist it. Without that the
    # supersession is written to a discarded copy and the same stale repair is offered again.
    superseded = current_repairs(refusal.value.state.artifacts)[proposal.repair_id]
    assert superseded.status == "superseded"


@pytest.mark.asyncio
async def test_approving_an_unknown_repair_says_so() -> None:
    """An identifier that belongs to no repair is a lookup failure, not a workflow failure."""
    from services.repository_repair import RepairNotFoundError

    state = stopped_feature()
    orchestrator = FeatureWorkflowOrchestrator()

    with pytest.raises(RepairNotFoundError):
        await orchestrator.approve_repository_repair(
            state, repair_id="repair-nothing", actor_id="alex", credentials=CREDENTIALS
        )


def test_open_repair_is_found_by_repository_id_and_not_by_role() -> None:
    """A feature may contain any number of repositories, and two may be stopped at once."""
    state = stopped_feature()
    from state.feature_models import ChildWorkflowReference

    state.child_workflows["docs-site"] = ChildWorkflowReference(
        child_workflow_id="feature-repair:docs-site",
        repository_id="docs-site",
        workstream_id="docs-site",
        status=ChildWorkflowStatus.FAILED,
        branch_name="feature/repair",
        workspace_path="/workspaces/feature-repair/docs-site",
        retry_count=1,
        blocking_setup_issues=[issue(category="missing_dependency")],
        failure_classification=FailureClassification.DEPENDENCY_INSTALLATION_FAILURE.value,
        preflight_status="blocked",
        current_revision="revision-nine",
    )

    _propose_repository_repairs(state)

    backend = open_repair_for(state.artifacts, "backend")
    docs = open_repair_for(state.artifacts, "docs-site")
    assert backend is not None
    assert docs is not None
    assert backend.repair_id != docs.repair_id
    assert open_repair_for(state.artifacts, "a-repository-with-no-repair") is None


def test_a_proposal_needs_at_least_one_finding() -> None:
    """A repair with nothing behind it is not something anybody could evaluate."""
    with pytest.raises(ValueError, match="at least one"):
        build_repair_payload(
            feature_id="feature-repair",
            repository_id="backend",
            child_workflow_id=None,
            originating_stage="repository_preflight",
            classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
            issues=[],
            current_revision=None,
        )


@pytest.mark.asyncio
async def test_repairs_survive_being_written_to_the_database(tmp_path: Path) -> None:
    """The decision index has to hold a repair through its whole lifecycle.

    Written against a real database because the projection is keyed on the repair identifier
    while artifact history appends a revision per decision -- the exact shape that silently
    broke contract-change approval, where two artifacts with one identifier became two rows
    with one primary key.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'repair.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        request = StartFeatureRequest.model_validate(feature_payload())
        await store.start(
            request, idempotency_key="repair-1", credentials=CREDENTIALS, owner_id="platform-admin"
        )
        record = await store.get_record("feature-login")

        state = record.state
        from state.feature_models import ChildWorkflowReference

        state.child_workflows["backend"] = ChildWorkflowReference(
            child_workflow_id="feature-login:backend",
            repository_id="backend",
            workstream_id="backend",
            status=ChildWorkflowStatus.FAILED,
            branch_name="feature/login",
            workspace_path="/workspaces/feature-login/backend",
            retry_count=1,
            blocking_setup_issues=[issue()],
            failure_classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE.value,
            preflight_status="blocked",
            current_revision="revision-one",
        )
        _propose_repository_repairs(state)
        proposal = next(iter(current_repairs(state.artifacts).values()))
        await store._replace_state(  # noqa: SLF001 - exercising the projection directly
            "feature-login", state, "repository_repair_proposed"
        )

        listed = await store.repairs("feature-login")
        assert [item.repair_id for item in listed] == [proposal.repair_id]
        assert listed[0].status == "proposed"

        rejected = await store.reject_repair(
            "feature-login",
            repair_id=proposal.repair_id,
            actor_id="alex",
            reason="We will remove the rule instead.",
        )
        assert rejected.state.feature_id == "feature-login"

        # One row per repair, carrying its newest state -- not one row per artifact.
        after = await store.repairs("feature-login")
        assert len(after) == 1
        assert after[0].status == "rejected"
        assert after[0].rejection_reason == "We will remove the rule instead."
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_contract_change_decision_can_be_persisted(tmp_path: Path) -> None:
    """The same append-a-revision shape, for the request type that already used it.

    This path had no durable coverage and did not work: both the pending artifact and its
    approved revision were projected into a table keyed on the change request identifier, so
    the first approval to reach the database failed on an integrity error.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'contract.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        request = StartFeatureRequest.model_validate(feature_payload())
        await store.start(
            request,
            idempotency_key="contract-1",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        record = await store.get_record("feature-login")
        state = record.state

        from agents.shared.contracts import create_artifact

        pending = create_artifact(
            ContractChangeRequestArtifact,
            workflow_id="feature-login",
            artifact_id="013_contract_change_request.backend.0.json",
            producer="child_workflow",
            payload={
                "change_request_id": "feature-login:backend:0",
                "feature_id": "feature-login",
                "current_contract_version": "1.0.0",
                "requested_by_repository_id": "backend",
                "requested_changes": ["Add a field."],
                "reason": "The backend needs it.",
                "affected_workstreams": ["backend"],
                "compatibility_impact": "Requires approval.",
                "migration_requirements": [],
                "status": "pending",
                "resolution": None,
                "new_contract_artifact_id": None,
            },
            metadata={},
        )
        state.artifacts.append(pending)
        approved = pending.model_copy(
            update={
                "artifact_id": "013_contract_change_request.backend.0.revision-2.json",
                "status": "approved",
                "resolution": "Approved.",
            }
        )
        state.artifacts.append(approved)

        await store._replace_state(  # noqa: SLF001 - exercising the projection directly
            "feature-login", state, "contract_change_decided"
        )

        reread = await store.get_record("feature-login")
        decisions = [
            item
            for item in reread.state.artifacts
            if isinstance(item, ContractChangeRequestArtifact)
        ]
        assert [item.status for item in decisions] == ["pending", "approved"]
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_approving_a_repair_requires_acknowledging_the_repository_change() -> None:
    """The endpoint is reachable without the console, so the server asks for itself."""
    from main import create_app

    app = create_app(platform_api_key="repair-key")
    headers = {"Authorization": "Bearer repair-key", "Idempotency-Key": "repair-http"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        without = await client.post(
            "/features/feature-login/repairs/repair-1/approve",
            headers={"Authorization": "Bearer repair-key"},
            json={},
        )
        unauthenticated = await client.post(
            "/features/feature-login/repairs/repair-1/approve",
            json={"acknowledge_repository_change": True},
        )
        listed = await client.get(
            "/features/feature-login/repairs", headers={"Authorization": "Bearer repair-key"}
        )

    assert without.status_code == 422
    assert unauthenticated.status_code == 401
    assert listed.status_code == 200
    assert listed.json()["repairs"] == []


@pytest.mark.asyncio
async def test_rejecting_a_repair_requires_a_reason() -> None:
    """A stop left in place without a recorded reason explains nothing to the next person."""
    from main import create_app

    app = create_app(platform_api_key="repair-key")
    headers = {"Authorization": "Bearer repair-key", "Idempotency-Key": "repair-reason"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        empty = await client.post(
            "/features/feature-login/repairs/repair-1/reject",
            headers={"Authorization": "Bearer repair-key"},
            json={"reason": ""},
        )

    assert empty.status_code == 422


def test_the_repair_status_enum_covers_every_artifact_status() -> None:
    """The projection casts one to the other, so a new state must exist in both."""
    artifact_statuses = {
        "proposed",
        "approved",
        "executing",
        "succeeded",
        "failed",
        "rejected",
        "superseded",
    }

    assert {item.value for item in RepositoryRepairStatus} == artifact_statuses


def test_a_feature_with_no_stopped_repository_proposes_nothing() -> None:
    """Diagnosis is called after every attempt, including the ones that went fine."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-fine", request)
    state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS

    _propose_repository_repairs(state)

    assert current_repairs(state.artifacts) == {}


def test_a_proposal_names_the_packages_the_finding_actually_blames() -> None:
    """Read from the finding's own fields rather than recovered from its prose.

    The preflight classifier extracts exactly which packages would not resolve. Those were
    being dropped when the finding was persisted, so a proposal had only a sentence to work
    from -- and scraping a sentence for package names produced nothing for the commonest
    repair there is, which a browser pass found by seeing an empty card.
    """
    finding = PreflightIssue(
        issue_id="LINT_CONFIGURATION_REQUIRES_HUMAN",
        category="missing_dependency",
        severity="high",
        description="Lint exited before source checks because its configuration extends a "
        "shared config the repository does not declare as a dependency.",
        evidence="Command npx eslint . exited before linting source files.",
        recommended_action="Declare the shared configuration as a repository dependency.",
        automatically_repairable=False,
        unresolved_references=["eslint-config-house", "@scope/eslint-plugin"],
        undeclared_references=["eslint-config-house"],
    )

    payload = build_repair_payload(
        feature_id="feature-repair",
        repository_id="backend",
        child_workflow_id=None,
        originating_stage="repository_preflight",
        classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
        issues=[finding],
        current_revision="abc",
        package_manager="npm",
    )

    assert payload["affected_dependencies"] == ["@scope/eslint-plugin", "eslint-config-house"]
    # The manifest a dependency change is made in, for the package manager this repository
    # actually uses -- not a path scraped out of an error message.
    assert payload["affected_files"] == ["package.json"]


def test_a_proposal_names_no_file_when_the_platform_cannot_say_which() -> None:
    """A repair that names a file it is guessing at sends somebody to edit the wrong thing."""
    finding = PreflightIssue.model_validate(issue())
    payload = build_repair_payload(
        feature_id="feature-repair",
        repository_id="backend",
        child_workflow_id=None,
        originating_stage="repository_preflight",
        classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
        issues=[finding],
        current_revision="abc",
        package_manager=None,
    )

    assert payload["affected_files"] == []
    assert payload["affected_dependencies"] == []


def test_the_lint_classifier_hands_its_references_to_the_finding(tmp_path: Path) -> None:
    """The seam that was dropping them, asserted where it is: preflight to persisted issue."""
    from services.feature_runtime import _block_unrepairable_lint_configuration
    from tools.repository_preflight import RepositoryPreflightResult
    from tools.validation_tools import ValidationResult, ValidationStatus

    preflight = RepositoryPreflightResult(
        repository_id="backend",
        revision="abc",
        dependency_install_status="installed",
        validation_readiness="ready",
    )
    failed = ValidationResult(
        command=("npx", "eslint", "."),
        return_code=2,
        # The wording the classifier keys on for a configuration that will not load, which
        # is the path that produces the references this test is about.
        stdout=(
            'Failed to load config "house" to extend from.\n'
            "Cannot find module 'eslint-config-house'"
        ),
        stderr="",
        timed_out=False,
        duration_seconds=1.0,
        output_truncated=False,
        cancelled=False,
        validation_type="lint",
        status=ValidationStatus.FAILED,
    )

    # A real directory: the classifier reads the repository's manifest to decide whether a
    # reference is declared, and an empty checkout declares nothing -- which is the case
    # that produces an undeclared reference.
    blocked = _block_unrepairable_lint_configuration(preflight, [failed], tmp_path)

    assert blocked.validation_readiness == "blocked"
    carried = blocked.blocking_issues[0]
    assert carried.unresolved_references, "the packages the classifier found must survive"


def test_a_repair_that_cannot_be_carried_out_is_not_proposed() -> None:
    """An Approve button that changes nothing is worse than saying a person has to act.

    A dependency the repository never declared is fixed only by declaring it. Where the
    package manager has no command that records a declaration -- pip installs into an
    environment and writes no manifest -- there is nothing here to approve.
    """
    state = stopped_feature(
        issues=[
            {
                **issue(category="missing_dependency"),
                "undeclared_references": ["eslint-config-house"],
            }
        ]
    )
    state.child_workflows["backend"] = state.child_workflows["backend"].model_copy(
        update={"selected_package_manager": "pip"}
    )

    _propose_repository_repairs(state)

    assert current_repairs(state.artifacts) == {}


def test_a_repair_that_can_be_carried_out_carries_its_commands() -> None:
    """Approving has to do something, and the card has to be able to say what."""
    state = stopped_feature(
        issues=[
            {
                **issue(category="missing_dependency"),
                "undeclared_references": ["eslint-config-house"],
            }
        ]
    )
    state.child_workflows["backend"] = state.child_workflows["backend"].model_copy(
        update={"selected_package_manager": "npm"}
    )

    _propose_repository_repairs(state)

    proposal = next(iter(current_repairs(state.artifacts).values()))
    assert [item.command for item in proposal.commands] == [
        ["npm", "install", "--save-dev", "--no-audit", "eslint-config-house"]
    ]
    assert proposal.affected_dependencies == ["eslint-config-house"]
    assert proposal.risk == "medium", "a repair that runs a command is not a low-risk one"


def test_a_failure_that_a_fresh_attempt_fixes_needs_no_command() -> None:
    """A declared dependency that is merely absent is repaired by installing deterministically.

    The granted attempt clones afresh and installs from the lockfile, so the repair is the
    attempt. Requiring a command here would refuse a repair that works.
    """
    state = stopped_feature(
        classification=FailureClassification.DEPENDENCY_INSTALLATION_FAILURE.value,
        issues=[issue(category="dependency_configuration")],
    )

    _propose_repository_repairs(state)

    proposal = next(iter(current_repairs(state.artifacts).values()))
    assert proposal.commands == []
    assert proposal.status == "proposed"


def test_a_reference_that_is_not_a_package_name_never_reaches_a_command() -> None:
    """These are parsed out of a package manager's own error output, which is untrusted.

    Commands run without a shell, so an argument cannot become a second command -- but a name
    is still handed to a tool that will try to fetch it, and anything unrecognisable must not
    get that far.
    """
    hostile = [
        "; rm -rf /",
        "../../etc/passwd",
        "$(whoami)",
        "package name with spaces",
        "--registry=http://evil.example",
        "eslint-config-house",
        "@scope/plugin",
    ]

    assert safe_package_names(hostile) == ["@scope/plugin", "eslint-config-house"]
    assert repair_commands("npm", ["; rm -rf /"]) == []


@pytest.mark.asyncio
async def test_an_approved_repair_runs_its_commands_once_in_the_checkout(
    tmp_path: Path,
) -> None:
    """A repair that does not run is not a repair.

    Exercised against the real executor and a real journal, because the two properties being
    claimed are that the command reaches the workspace and that approving twice does not run
    it twice -- and both live in the seam between them.
    """
    from unittest.mock import patch

    from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
    from services.feature_runtime import LiveChildWorkstreamExecutor
    from storage.external_operation_store import ExternalOperationJournal

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'repair-run.db'}")
    await database.create_schema()
    try:
        journal = ExternalOperationJournal(database)
        executed: list[tuple[str, ...]] = []

        class RecordingRunner:
            """Record what was run instead of running it."""

            async def run(
                self,
                command: Sequence[str],
                cwd: Path,
                timeout_seconds: float,
                cancellation_token: Any,
                environment: Any = None,
            ) -> Any:
                """Record the argv and report success."""
                del cwd, timeout_seconds, cancellation_token, environment
                executed.append(tuple(command))
                from services.process_runner import ProcessResult

                return ProcessResult(
                    command=tuple(command),
                    return_code=0,
                    stdout="added 1 package",
                    stderr="",
                    timed_out=False,
                    duration_seconds=0.1,
                    output_truncated=False,
                    cancelled=False,
                )

        state = stopped_feature(
            issues=[
                {
                    **issue(category="missing_dependency"),
                    "undeclared_references": ["eslint-config-house"],
                }
            ]
        )
        state.child_workflows["backend"] = state.child_workflows["backend"].model_copy(
            update={"selected_package_manager": "npm"}
        )
        _propose_repository_repairs(state)
        proposal = next(iter(current_repairs(state.artifacts).values()))
        # Authorized, which is the only state in which a checkout may be touched.
        from workflows.feature_workflow import revise_repository_repair

        revise_repository_repair(state, proposal, status="approved", approved_by="alex")

        executor = LiveChildWorkstreamExecutor.__new__(LiveChildWorkstreamExecutor)
        executor._process_runner = cast(Any, RecordingRunner())  # noqa: SLF001
        executor._cancellation_token = MockCancellationToken()  # noqa: SLF001
        executor._settings = type("_S", (), {"commit_gate_timeout_seconds": 60})()  # noqa: SLF001

        operations = ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id="feature-repair",
                feature_id="feature-repair",
                child_workflow_id="feature-repair:backend",
                repository_id="backend",
            ),
        )

        with patch("services.feature_runtime.repository_subprocess_environment", dict):
            applied = await executor._apply_approved_repair(  # noqa: SLF001
                state,
                repository_id="backend",
                workspace=tmp_path,
                operation_executor=operations,
            )
            # The same approval reaching the executor again -- a retried request, a resumed
            # attempt -- must not install a second time.
            again = await executor._apply_approved_repair(  # noqa: SLF001
                state,
                repository_id="backend",
                workspace=tmp_path,
                operation_executor=operations,
            )

        assert applied == ["npm install --save-dev --no-audit eslint-config-house"]
        assert executed == [
            ("npm", "install", "--save-dev", "--no-audit", "eslint-config-house")
        ], "the journal must replay the recorded result rather than run the command twice"
        assert again == ["npm install --save-dev --no-audit eslint-config-house"]
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_repair_nobody_approved_never_touches_the_checkout(tmp_path: Path) -> None:
    """A proposal is a diagnosis waiting for a decision, not an instruction."""
    from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
    from services.feature_runtime import LiveChildWorkstreamExecutor
    from storage.external_operation_store import ExternalOperationJournal

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'unapproved.db'}")
    await database.create_schema()
    try:
        state = stopped_feature(
            issues=[
                {
                    **issue(category="missing_dependency"),
                    "undeclared_references": ["eslint-config-house"],
                }
            ]
        )
        state.child_workflows["backend"] = state.child_workflows["backend"].model_copy(
            update={"selected_package_manager": "npm"}
        )
        _propose_repository_repairs(state)

        executor = LiveChildWorkstreamExecutor.__new__(LiveChildWorkstreamExecutor)

        class _Unreachable:
            """Fail the test if anything is executed at all."""

            async def run(self, *args: Any, **kwargs: Any) -> Any:
                """Never reached: an unapproved repair runs nothing."""
                raise AssertionError("an unapproved repair must not run anything")

        executor._process_runner = cast(Any, _Unreachable())  # noqa: SLF001
        executor._cancellation_token = MockCancellationToken()  # noqa: SLF001
        executor._settings = type("_S", (), {"commit_gate_timeout_seconds": 60})()  # noqa: SLF001

        applied = await executor._apply_approved_repair(  # noqa: SLF001
            state,
            repository_id="backend",
            workspace=tmp_path,
            operation_executor=ExternalOperationExecutor(
                journal=ExternalOperationJournal(database),
                cancellation_token=MockCancellationToken(),
                scope=ExternalOperationScope(workflow_id="feature-repair"),
            ),
        )

        assert applied == []
    finally:
        await database.dispose()
