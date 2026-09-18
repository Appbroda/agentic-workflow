"""Regression tests for DB-owned workspace retention and volume-capacity guardrails."""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agents.shared.contracts import create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import ChildWorkflowResultArtifact
from configs.settings import load_settings
from services.feature_queue import (
    FeatureQueueDispatcher,
    FeatureWorkspaceCapacityUnavailable,
    InMemoryFeatureExecutionQueue,
)
from services.feature_runtime import (
    ProductionFeatureRunner,
    WorkspaceCapacityError,
    WorkspaceMaintenanceError,
    _maintain_feature_workspaces,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus, WorkstreamRole
from state.failure_diagnosis import FeatureFailureClassification, classification_of
from state.feature_models import ChildWorkflowReference, FeatureWorkflowSnapshot
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import FeatureWorkflowModel, RepositorySpecModel
from tests.test_feature_workflow import feature_payload
from workflows.feature_workflow import (
    _feature_branch_segment,
    _has_unpublished_approved_child,
    feature_has_publishable_work,
    feature_is_at_rest,
)


@pytest.mark.asyncio
async def test_retention_removes_only_db_confirmed_terminal_auto_workspaces(
    tmp_path: Path,
) -> None:
    """Status, execution mode, path ownership, and digest identity all gate deletion."""
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    outside = tmp_path / "outside-symlink-target"
    outside.mkdir()
    (outside / "evidence.txt").write_text("retain\n", encoding="utf-8")
    now = datetime.now(UTC)
    records = [
        ("completed-new", FeatureWorkflowStatus.COMPLETED, "live", None, now),
        (
            "completed-old",
            FeatureWorkflowStatus.COMPLETED,
            "live",
            None,
            now - timedelta(hours=2),
        ),
        (
            "cancelled-old",
            FeatureWorkflowStatus.CANCELLED,
            "live",
            None,
            now - timedelta(hours=3),
        ),
        (
            "cancelled-side-effects",
            FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
            "live",
            None,
            now - timedelta(hours=3, minutes=30),
        ),
        ("running", FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS, "live", None, now),
        ("failed-human", FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN, "live", None, now),
        (
            "user-workspace",
            FeatureWorkflowStatus.COMPLETED,
            "live",
            str(tmp_path / "operator-checkout"),
            now - timedelta(hours=4),
        ),
        (
            "mock-completed",
            FeatureWorkflowStatus.COMPLETED,
            "mock",
            None,
            now - timedelta(hours=5),
        ),
        (
            "symlink-terminal",
            FeatureWorkflowStatus.COMPLETED,
            "live",
            None,
            now - timedelta(hours=6),
        ),
    ]
    try:
        for feature_id, status, mode, local_path, updated_at in records:
            await _insert_feature(
                database,
                feature_id=feature_id,
                status=status,
                execution_mode=mode,
                local_workspace_path=local_path,
                updated_at=updated_at,
            )
            path = workspace_root / _feature_branch_segment(feature_id)
            if feature_id == "symlink-terminal":
                workspace_root.mkdir(parents=True, exist_ok=True)
                path.symlink_to(outside, target_is_directory=True)
            else:
                path.mkdir(parents=True)
                (path / "evidence.txt").write_text(feature_id, encoding="utf-8")
        unknown = workspace_root / "unknown-directory"
        unknown.mkdir()

        result = await _maintain_feature_workspaces(
            workspace_root,
            database=database,
            keep=1,
            minimum_free_bytes=0,
        )

        assert set(result.removed_feature_ids) == {"completed-old", "cancelled-old"}
        assert result.skipped_symlink_feature_ids == ("symlink-terminal",)
        assert (workspace_root / _feature_branch_segment("completed-new")).is_dir()
        assert not (workspace_root / _feature_branch_segment("completed-old")).exists()
        assert not (workspace_root / _feature_branch_segment("cancelled-old")).exists()
        for retained in (
            "running",
            "failed-human",
            "cancelled-side-effects",
            "user-workspace",
            "mock-completed",
        ):
            assert (workspace_root / _feature_branch_segment(retained)).is_dir()
        assert unknown.is_dir()
        symlink = workspace_root / _feature_branch_segment("symlink-terminal")
        assert symlink.is_symlink()
        assert (outside / "evidence.txt").read_text(encoding="utf-8") == "retain\n"
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_concurrent_retention_runs_delete_each_workspace_at_most_once(tmp_path: Path) -> None:
    """Concurrent starts share in-process and filesystem locks instead of racing rmtree."""
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    now = datetime.now(UTC)
    try:
        for feature_id, updated_at in (
            ("retained", now),
            ("removed-once", now - timedelta(hours=1)),
        ):
            await _insert_feature(
                database,
                feature_id=feature_id,
                status=FeatureWorkflowStatus.COMPLETED,
                execution_mode="live",
                local_workspace_path=None,
                updated_at=updated_at,
            )
            (workspace_root / _feature_branch_segment(feature_id)).mkdir(parents=True)

        first, second = await asyncio.gather(
            _maintain_feature_workspaces(
                workspace_root, database=database, keep=1, minimum_free_bytes=0
            ),
            _maintain_feature_workspaces(
                workspace_root, database=database, keep=1, minimum_free_bytes=0
            ),
        )

        removals = (*first.removed_feature_ids, *second.removed_feature_ids)
        assert removals == ("removed-once",)
        assert (workspace_root / _feature_branch_segment("retained")).is_dir()
        assert not (workspace_root / _feature_branch_segment("removed-once")).exists()
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_capacity_preflight_runs_after_cleanup_and_returns_actionable_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A low-volume worker fails before clone/install and reports measured versus required bytes."""
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    now = datetime.now(UTC)
    try:
        for feature_id, updated_at in (
            ("capacity-retained", now),
            ("capacity-removed", now - timedelta(hours=1)),
        ):
            await _insert_feature(
                database,
                feature_id=feature_id,
                status=FeatureWorkflowStatus.COMPLETED,
                execution_mode="live",
                local_workspace_path=None,
                updated_at=updated_at,
            )
            (workspace_root / _feature_branch_segment(feature_id)).mkdir(parents=True)
        monkeypatch.setattr(
            "services.feature_runtime.shutil.disk_usage",
            lambda _path: SimpleNamespace(free=99),
        )

        with pytest.raises(WorkspaceCapacityError, match="99 bytes free") as captured:
            await _maintain_feature_workspaces(
                workspace_root,
                database=database,
                keep=1,
                minimum_free_bytes=100,
            )

        assert not (workspace_root / _feature_branch_segment("capacity-removed")).exists()
        assert captured.value.free_bytes == 99
        assert captured.value.required_bytes == 100
        assert "expand the workspace volume" in captured.value.diagnostics[0]
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_retention_propagates_deletion_errors_instead_of_ignoring_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unreadable retained tree stops the run with safe diagnostics and remains intact."""
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    now = datetime.now(UTC)
    try:
        for feature_id, updated_at in (
            ("permission-retained", now),
            ("permission-error", now - timedelta(hours=1)),
        ):
            await _insert_feature(
                database,
                feature_id=feature_id,
                status=FeatureWorkflowStatus.COMPLETED,
                execution_mode="live",
                local_workspace_path=None,
                updated_at=updated_at,
            )
            (workspace_root / _feature_branch_segment(feature_id)).mkdir(parents=True)

        def fail_delete(_path: Path) -> None:
            raise PermissionError(13, "permission denied")

        monkeypatch.setattr("services.feature_runtime.shutil.rmtree", fail_delete)

        with pytest.raises(WorkspaceMaintenanceError, match="errno 13") as captured:
            await _maintain_feature_workspaces(
                workspace_root,
                database=database,
                keep=1,
                minimum_free_bytes=0,
            )

        assert "Check worker-volume permissions" in captured.value.diagnostics[0]
        assert (workspace_root / _feature_branch_segment("permission-error")).is_dir()
    finally:
        await database.dispose()


async def _database(tmp_path: Path) -> Database:
    """Create one isolated control-plane schema for retention eligibility queries."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'retention.db'}")
    await database.create_schema()
    return database


async def _insert_feature(
    database: Database,
    *,
    feature_id: str,
    status: FeatureWorkflowStatus,
    execution_mode: str,
    local_workspace_path: str | None,
    updated_at: datetime,
) -> None:
    """Persist the minimum parent and repository ownership evidence cleanup requires."""
    async with database.session() as session:
        session.add(
            FeatureWorkflowModel(
                owner_id="platform-admin",
                feature_id=feature_id,
                status=status,
                title=feature_id,
                execution_mode=execution_mode,
                merge_strategy=None,
                deployment_strategy=None,
                state_json={},
                created_at=updated_at,
                updated_at=updated_at,
            )
        )
        session.add(
            RepositorySpecModel(
                feature_id=feature_id,
                repository_id=f"{feature_id}-repository",
                name=feature_id,
                role=WorkstreamRole.SERVICE.value,
                repository_url=f"https://github.com/example/{feature_id}.git",
                default_branch="main",
                local_workspace_path=local_workspace_path,
                required=True,
                implementation_order=0,
                metadata_json={},
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_a_failed_workspace_is_reclaimed_once_it_is_old_enough(tmp_path: Path) -> None:
    """A checkout kept for a human to open must not be kept forever.

    Retaining every `failed_requires_human` workspace reclaimed nothing during a pilot whose
    runs mostly ended that way: the volume filled and the capacity preflight refused to start
    -071 at all. The database keeps the artifacts, events and operation journal regardless, so
    the only thing lost is a checkout nobody opened inside the retention window.
    """
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    now = datetime.now(UTC)
    records = [
        ("failed-fresh", now - timedelta(hours=1)),
        ("failed-stale", now - timedelta(hours=100)),
    ]
    for feature_id, updated_at in records:
        await _insert_feature(
            database,
            feature_id=feature_id,
            status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            execution_mode="live",
            local_workspace_path=None,
            updated_at=updated_at,
        )
        path = workspace_root / _feature_branch_segment(feature_id)
        path.mkdir(parents=True)
        (path / "evidence.txt").write_text(feature_id, encoding="utf-8")

    result = await _maintain_feature_workspaces(
        workspace_root,
        database=database,
        keep=0,
        minimum_free_bytes=0,
        failed_retention_hours=72,
    )

    assert result.removed_feature_ids == ("failed-stale",)
    assert (workspace_root / _feature_branch_segment("failed-fresh")).is_dir()
    assert not (workspace_root / _feature_branch_segment("failed-stale")).exists()


@pytest.mark.asyncio
async def test_failed_workspaces_are_kept_when_no_retention_window_is_configured(
    tmp_path: Path,
) -> None:
    """Zero hours keeps the previous behaviour, so an operator can opt out of reclaiming."""
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    await _insert_feature(
        database,
        feature_id="failed-ancient",
        status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
        execution_mode="live",
        local_workspace_path=None,
        updated_at=datetime.now(UTC) - timedelta(days=400),
    )
    path = workspace_root / _feature_branch_segment("failed-ancient")
    path.mkdir(parents=True)
    (path / "evidence.txt").write_text("retain", encoding="utf-8")

    result = await _maintain_feature_workspaces(
        workspace_root,
        database=database,
        keep=0,
        minimum_free_bytes=0,
        failed_retention_hours=0,
    )

    assert result.removed_feature_ids == ()
    assert path.is_dir()


# --------------------------------------------------------------------------------------
# Run 193: a day of retained failures, and a delivered feature nearly lost to them
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retained_failures_beyond_the_count_are_reclaimed_oldest_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The age window alone cannot bound a day's worth of them, which is what run 193 met.

    Every retained workspace on that volume was from the same day, so none was old enough to
    reclaim: the volume reached 99% with 1.84 GB free, and the capacity preflight then failed
    a feature whose repositories had both been approved. A count bound is what makes a day of
    failures bounded rather than only a week of them.

    Asserted on the directories, and on one loud line per deletion: a retained failure
    workspace is the only copy of what that attempt wrote, so an operator who goes looking
    for one that is gone has to be able to find the line that says this pass removed it.
    """
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    now = datetime.now(UTC)
    # Six failures, every one of them well inside a 72-hour window, so the age rule reclaims
    # none of them and only the count can.
    failures = [(f"failed-{index}", now - timedelta(minutes=index * 5)) for index in range(6)]
    try:
        for feature_id, updated_at in failures:
            await _insert_feature(
                database,
                feature_id=feature_id,
                status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                execution_mode="live",
                local_workspace_path=None,
                updated_at=updated_at,
            )
            path = workspace_root / _feature_branch_segment(feature_id)
            path.mkdir(parents=True)
            (path / "evidence.txt").write_text(feature_id, encoding="utf-8")

        with caplog.at_level("WARNING", logger="services.feature_runtime"):
            result = await _maintain_feature_workspaces(
                workspace_root,
                database=database,
                keep=0,
                minimum_free_bytes=0,
                failed_retention_hours=72,
                failed_retention_features=4,
            )

        # Oldest-first beyond the bound, and the newest four kept whatever else happens.
        assert set(result.removed_feature_ids) == {"failed-4", "failed-5"}
        for retained in ("failed-0", "failed-1", "failed-2", "failed-3"):
            assert (workspace_root / _feature_branch_segment(retained)).is_dir()
        for reclaimed in ("failed-4", "failed-5"):
            assert not (workspace_root / _feature_branch_segment(reclaimed)).exists()
        # One line per deletion, naming the feature and which bound it was over.
        reclaimed_lines = [
            record
            for record in caplog.records
            if record.getMessage() == "retained_feature_workspace_reclaimed"
        ]
        assert len(reclaimed_lines) == 2
        named = {
            (record.__dict__["feature_id"], record.__dict__["reason"]) for record in reclaimed_lines
        }
        assert named == {
            ("failed-4", "failed_beyond_retention_count"),
            ("failed-5", "failed_beyond_retention_count"),
        }
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_the_two_failure_bounds_are_independent(tmp_path: Path) -> None:
    """Either bound is enough, and neither weakens the other.

    A count of ten does not buy an ancient checkout another week, and a 72-hour window does
    not license a hundred same-day ones. Both halves are asserted in one lineage because the
    defect either would have on its own is "the other rule silently stopped applying".
    """
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    now = datetime.now(UTC)
    records = [
        ("failed-recent", now - timedelta(minutes=1)),
        ("failed-ancient", now - timedelta(hours=200)),
    ]
    try:
        for feature_id, updated_at in records:
            await _insert_feature(
                database,
                feature_id=feature_id,
                status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                execution_mode="live",
                local_workspace_path=None,
                updated_at=updated_at,
            )
            (workspace_root / _feature_branch_segment(feature_id)).mkdir(parents=True)

        result = await _maintain_feature_workspaces(
            workspace_root,
            database=database,
            keep=0,
            minimum_free_bytes=0,
            failed_retention_hours=72,
            # Generous enough that the count reclaims nothing here, so what remains is the
            # age rule -- which must still fire.
            failed_retention_features=10,
        )

        assert result.removed_feature_ids == ("failed-ancient",)
        assert (workspace_root / _feature_branch_segment("failed-recent")).is_dir()
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_finished_workspace_is_not_counted_against_the_failure_bound(
    tmp_path: Path,
) -> None:
    """The two policies are per kind, so a busy week of deliveries cannot evict evidence.

    A completed feature's checkout is a convenience and a failed one's is the only copy of
    what the attempt wrote. Counting them together would let either kind push the other out
    for reasons that have nothing to do with why it was kept.
    """
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    now = datetime.now(UTC)
    records = [
        ("done-new", FeatureWorkflowStatus.COMPLETED, now),
        ("done-old", FeatureWorkflowStatus.COMPLETED, now - timedelta(minutes=2)),
        ("failed-only", FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN, now - timedelta(minutes=3)),
    ]
    try:
        for feature_id, status, updated_at in records:
            await _insert_feature(
                database,
                feature_id=feature_id,
                status=status,
                execution_mode="live",
                local_workspace_path=None,
                updated_at=updated_at,
            )
            (workspace_root / _feature_branch_segment(feature_id)).mkdir(parents=True)

        result = await _maintain_feature_workspaces(
            workspace_root,
            database=database,
            keep=1,
            minimum_free_bytes=0,
            failed_retention_hours=72,
            failed_retention_features=1,
        )

        # `keep=1` reclaims the second finished workspace and nothing else: the failure is
        # the newest one of its own kind, and the finished ones do not occupy its bound.
        assert result.removed_feature_ids == ("done-old",)
        assert (workspace_root / _feature_branch_segment("failed-only")).is_dir()
        assert (workspace_root / _feature_branch_segment("done-new")).is_dir()
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# The publication invariant loses its disk-space exception
# --------------------------------------------------------------------------------------


def _approved_unpublished_feature(feature_id: str) -> FeatureWorkflowSnapshot:
    """One feature holding a reviewed, pushed result nobody has opened a pull request for.

    Exactly run 193's state at the moment the preflight refused it: both repositories through
    their own reviews, nothing published, and the whole delivery still ahead of it.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state(feature_id, request)
    for repository_id in ("backend", "frontend"):
        state.child_workflows[repository_id] = ChildWorkflowReference(
            child_workflow_id=f"{feature_id}:{repository_id}",
            repository_id=repository_id,
            workstream_id=repository_id,
            branch_name=f"ai/{feature_id}/{repository_id}/work",
            workspace_path=f"/workspaces/{feature_id}/{repository_id}",
            status=ChildWorkflowStatus.APPROVED,
            retry_count=0,
        )
        state.artifacts.append(
            create_artifact(
                ChildWorkflowResultArtifact,
                workflow_id=feature_id,
                artifact_id=f"{repository_id}-result",
                producer="engineer",
                payload={
                    "feature_id": feature_id,
                    "parent_workflow_id": feature_id,
                    "child_workflow_id": f"{feature_id}:{repository_id}",
                    "repository_id": repository_id,
                    "workstream_id": repository_id,
                    "branch_name": f"ai/{feature_id}/{repository_id}/work",
                    "workspace_path": f"/workspaces/{feature_id}/{repository_id}",
                    "code_completion_artifact_id": f"{repository_id}-completion",
                    "review_artifact_id": f"{repository_id}-review",
                    "changed_files": [],
                    "validation_results": [],
                    "status": "approved",
                    "blocking_issues": [],
                    "pull_request_readiness": True,
                    "contract_sections_consumed": [],
                    "contract_sections_implemented": [],
                },
                metadata={},
            )
        )
    return state


@pytest.mark.asyncio
async def test_a_full_volume_no_longer_refuses_a_claim_that_owes_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Approved work always reaches a pull request; disk space was the missing exception.

    Run 193: both repositories APPROVED, the 2 GB preflight failed at 1.84 GB free, and the
    feature flipped `failed_requires_human` with no salvage pull requests -- because the
    refusal happened before the orchestrator that would have published them was entered at
    all. Publication needs no workspace: those branches were already on the remote.
    """
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir(parents=True)
    monkeypatch.setattr(
        "services.feature_runtime.shutil.disk_usage", lambda _path: SimpleNamespace(free=99)
    )
    runner = ProductionFeatureRunner(
        load_settings(workspace_root=workspace_root, workspace_minimum_free_bytes=100),
        database=database,
    )
    try:
        owed = _approved_unpublished_feature("feature-193")
        # The precondition, asserted rather than assumed: this is the state the publication
        # invariant is about, and it is the same predicate the invariant itself reads.
        assert _has_unpublished_approved_child(owed)
        with caplog.at_level("WARNING", logger="services.feature_runtime"):
            await runner._maintain_workspaces_for(owed)
        assert any(
            record.getMessage() == "workspace_capacity_yielded_to_publication"
            for record in caplog.records
        ), "the yield has to be legible; a silent bypass of a capacity floor is worse"
        # And the feature is not at rest even at `failed_requires_human`, so the claim that
        # gets past the floor is one the platform will take unasked. What it then does with
        # it -- publish the approved work -- is the invariant's own coverage, asserted end to
        # end against real pull-request records in `test_feature_workflow`'s salvage tests.
        owed.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert not feature_is_at_rest(owed)

        # And the floor is untouched for a feature with nothing published to salvage, which
        # is what it exists for: refusing a clone the volume cannot hold.
        nothing_owed = _initial_feature_state(
            "feature-fresh", StartFeatureRequest.model_validate(feature_payload())
        )
        with pytest.raises(WorkspaceCapacityError, match="99 bytes free"):
            await runner._maintain_workspaces_for(nothing_owed)
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_full_volume_does_not_refuse_a_publication_of_rejected_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Task 81- added a second kind of publishable work, and the yield has to cover it too.

    A workstream whose every required check passed and whose review rejected it is publishable
    on a person's instruction -- and it was never committed, so the only copy is its checkout.
    Keyed on approved-and-unpublished work, the capacity yield did not see it: a feature whose
    only publishable work was clean-but-rejected met the free-space refusal while the console
    was offering to publish it. Run 193's lesson, reached by the other door.
    """
    database = await _database(tmp_path)
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir(parents=True)
    monkeypatch.setattr(
        "services.feature_runtime.shutil.disk_usage", lambda _path: SimpleNamespace(free=99)
    )
    runner = ProductionFeatureRunner(
        load_settings(workspace_root=workspace_root, workspace_minimum_free_bytes=100),
        database=database,
    )
    try:
        rejected = _rejected_but_clean_feature("feature-81-rejected", root=tmp_path)
        # The precondition, and the whole point: the narrower predicate says there is nothing
        # to salvage here, and the one the yield now reads says there is.
        assert not _has_unpublished_approved_child(rejected)
        assert feature_has_publishable_work(rejected)

        with caplog.at_level("WARNING", logger="services.feature_runtime"):
            await runner._maintain_workspaces_for(rejected)

        assert any(
            record.getMessage() == "workspace_capacity_yielded_to_publication"
            for record in caplog.records
        ), "the yield has to be legible; a silent bypass of a capacity floor is worse"

        # And the floor still refuses once the checkout is gone, because then there is
        # genuinely nothing to publish -- the commit this class needs has nowhere to come from.
        shutil.rmtree(tmp_path / "feature-81-rejected" / "backend")
        assert not feature_has_publishable_work(rejected)
        with pytest.raises(WorkspaceCapacityError, match="99 bytes free"):
            await runner._maintain_workspaces_for(rejected)
    finally:
        await database.dispose()


def _rejected_but_clean_feature(feature_id: str, *, root: Path) -> FeatureWorkflowSnapshot:
    """One feature whose only publishable work is clean, rejected, and still on disk."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state(feature_id, request)
    state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    workspace = root / feature_id / "backend"
    (workspace / ".git").mkdir(parents=True)
    state.child_workflows["backend"] = ChildWorkflowReference(
        child_workflow_id=f"{feature_id}:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name=f"ai/{feature_id}/backend/work",
        workspace_path=str(workspace),
        status=ChildWorkflowStatus.REVIEW_REJECTED,
        retry_count=0,
    )
    state.artifacts.append(
        create_artifact(
            ChildWorkflowResultArtifact,
            workflow_id=feature_id,
            artifact_id="backend-result",
            producer="engineer",
            payload={
                "feature_id": feature_id,
                "parent_workflow_id": feature_id,
                "child_workflow_id": f"{feature_id}:backend",
                "repository_id": "backend",
                "workstream_id": "backend",
                "branch_name": f"ai/{feature_id}/backend/work",
                "workspace_path": str(workspace),
                "code_completion_artifact_id": "backend-completion",
                "review_artifact_id": "backend-review",
                "changed_files": [
                    {
                        "path": "server/app.py",
                        "change_type": "modified",
                        "description": "the work the reviewer rejected",
                    }
                ],
                "validation_results": [],
                "current_validation_results": [
                    {"name": "repository lint", "required": True, "passed": True},
                    {"name": "repository tests", "required": True, "passed": True},
                ],
                "status": "failed",
                "blocking_issues": ["the reviewer wanted this written another way"],
                "pull_request_readiness": False,
                "contract_sections_consumed": [],
                "contract_sections_implemented": [],
            },
            metadata={},
        )
    )
    return state


# --------------------------------------------------------------------------------------
# A claim that is still retrying a capacity condition has not decided anything
# --------------------------------------------------------------------------------------


class _VolumeFullRunner:
    """A runner whose every step fails the capacity preflight, and nothing else."""

    def __init__(self, workspace_root: Path) -> None:
        """Record how many claims reached it, and the volume it reports as full."""
        self._workspace_root = workspace_root
        self.claims = 0

    async def advance_one_step(self, state: Any, *, credentials: Any) -> FeatureWorkflowSnapshot:
        """Fail exactly as `_maintain_feature_workspaces` does when the floor is not met."""
        del state, credentials
        self.claims += 1
        raise WorkspaceCapacityError(
            workspace_root=self._workspace_root, free_bytes=1_975_684_301, required_bytes=2**31
        )

    async def start(self, state: Any, *, credentials: Any) -> FeatureWorkflowSnapshot:
        """Start takes the same preflight, so it fails the same way."""
        return await self.advance_one_step(state, credentials=credentials)


@pytest.mark.asyncio
async def test_a_claim_retrying_a_capacity_condition_does_not_read_failed_requires_human(
    tmp_path: Path,
) -> None:
    """Run 193's status was the lie, not its diagnostic text.

    The preflight failed, the executor recorded the feature `failed_requires_human` and
    returned normally -- so the console said a person was needed while the dispatcher went on
    claiming the feature for as long as the volume stayed full. A manual cleanup then let the
    very next claim complete the feature end to end, which is exactly what the status had
    been denying. The condition is now raised, so the queue spends the attempts it has
    reserved and the feature keeps its status until they are gone.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'capacity.db'}")
    await database.create_schema()
    runner = _VolumeFullRunner(tmp_path / "workspaces")
    store = SqlAlchemyFeatureControlPlane(database=database, live_runner=cast(Any, runner))
    credentials = RequestScopedCredentials(openai_api_key="key", github_token="token")
    try:
        payload = dict(feature_payload())
        payload["execution_mode"] = "live"
        accepted = await store.start(
            StartFeatureRequest.model_validate(payload),
            idempotency_key="capacity-key",
            credentials=credentials,
            owner_id="platform-admin",
        )
        feature_id = accepted.record.state.feature_id

        with pytest.raises(FeatureWorkspaceCapacityUnavailable):
            await store.execute_queued(feature_id, credentials=credentials)

        assert runner.claims == 1
        record = await store.get_record(feature_id)
        # Not decided. The status is whatever the feature had reached, and no failure
        # summary claims a person now owns it.
        assert record.state.status is not FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert record.state.failure_summary is None
        # The timeline still says what happened, which is the half that was already right:
        # the diagnostic was never the problem.
        events = [event.event for event in record.lifecycle_events]
        assert "feature_workspace_capacity_unavailable" in events
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_capacity_condition_that_outlasts_its_attempts_is_recorded_as_the_stop(
    tmp_path: Path,
) -> None:
    """A bound has to exist, and it is the entry's own attempt budget.

    The queue spends every attempt reserved for the condition and only then records the
    feature as stopped -- with the capacity classification and its measured numbers, not with
    a bare exception type and not as a finding about the repository.
    """
    queue = InMemoryFeatureExecutionQueue()
    executor = _CapacityUnavailableExecutor(tmp_path / "workspaces")
    await queue.enqueue(
        feature_id="feature-capacity",
        execution_mode="live",
        agent_platform="openai",
        requested_by="platform-admin",
    )

    async def credentials_for(owner_id: str) -> RequestScopedCredentials:
        del owner_id
        return RequestScopedCredentials(openai_api_key="key", github_token="token")

    dispatcher = FeatureQueueDispatcher(
        queue=queue,
        executor=cast(Any, executor),
        credentials_for=credentials_for,
        lease_seconds=30,
    )

    assert [await dispatcher.run_once() for _ in range(3)] == [True, True, True]

    assert executor.attempts == 3, "every reserved attempt must be spent on the condition"
    # Told about exactly once, and only once there is nothing left to try.
    assert executor.recorded == ["fail_exhausted_fault"]
    assert executor.classifications == [
        FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE.value
    ]
    assert await queue.pending_count() == 0
    assert await dispatcher.run_once() is False


class _CapacityUnavailableExecutor:
    """A queued executor whose every claim meets a volume that is still too full."""

    def __init__(self, workspace_root: Path) -> None:
        """Count the claims and record the terminal verdicts reached about the feature."""
        self._workspace_root = workspace_root
        self.attempts = 0
        self.recorded: list[str] = []
        self.classifications: list[str] = []

    async def execute_queued(self, feature_id: str, **kwargs: Any) -> None:
        """Raise the condition the storage layer raises when the preflight refuses a claim."""
        del feature_id, kwargs
        self.attempts += 1
        raise FeatureWorkspaceCapacityUnavailable(
            WorkspaceCapacityError(
                workspace_root=self._workspace_root,
                free_bytes=1_975_684_301,
                required_bytes=2**31,
            )
        )

    async def fail_queued(self, feature_id: str, **kwargs: Any) -> None:
        """Refusals the queue decides without running anything do not belong to this test."""
        del feature_id, kwargs
        self.recorded.append("fail_queued")

    async def fail_exhausted_fault(self, feature_id: str, **kwargs: Any) -> None:
        """Record the terminal verdict, and the classification it was reached with."""
        del feature_id
        self.recorded.append("fail_exhausted_fault")
        self.classifications.append(classification_of(cast(Exception, kwargs["error"])).value)

    async def feature_owes_another_step(self, feature_id: str, **kwargs: Any) -> bool:
        """Never reached: every claim against this executor raises before it can be asked."""
        del feature_id, kwargs
        return False

    async def stop_unprogressing_run(self, feature_id: str, *, steps: int) -> None:
        """Never reached: this executor never completes a step."""
        del feature_id, steps

    async def feature_run_succeeded(self, feature_id: str) -> bool:
        """Report the feature as unfinished; this double never completes one."""
        del feature_id
        return False
