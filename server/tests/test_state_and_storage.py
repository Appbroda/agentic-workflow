"""Tests for structured LangGraph state and asynchronous database persistence."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from state.enums import ApprovalState, ConversationEventType, LogLevel, WorkflowStatus
from state.models import (
    ConversationEntry,
    WorkspaceDescriptor,
    create_initial_agent_state,
)
from storage.db import Database
from storage.models import ArtifactModel, ExecutionLogModel, WorkflowModel


def workspace_descriptor() -> WorkspaceDescriptor:
    """Create a valid workspace descriptor for a workflow test."""
    return WorkspaceDescriptor(
        workspace_id="workspace-1",
        root_path="/srv/workspaces/workspace-1",
        source_repo_url="https://github.com/example/platform.git",
        default_branch="main",
        working_branch="workflow/workflow-1",
    )


def test_workflow_enums_and_initial_state() -> None:
    """A newly created state has the required status, approval, and empty collections."""
    descriptor = workspace_descriptor()

    state = create_initial_agent_state(workflow_id="workflow-1", workspace_descriptor=descriptor)

    assert WorkflowStatus.PENDING.value == "pending"
    assert WorkflowStatus.FAILED_REQUIRES_HUMAN.value == "failed_requires_human"
    assert state["current_step"] is WorkflowStatus.PENDING
    assert state["approval_state"] is ApprovalState.NOT_REQUIRED
    assert state["current_agent"] is None
    assert state["conversation_history"] == []
    assert state["artifacts"] == []
    assert state["execution_logs"] == []
    assert state["checkpoints"] == []


def test_structured_conversation_entries_reject_invalid_references() -> None:
    """Conversation history keeps typed event references instead of opaque dictionaries."""
    entry = ConversationEntry(
        entry_id="event-1",
        sender="product_manager",
        recipient="human",
        event_type=ConversationEventType.CLARIFICATION_REQUESTED,
        artifact_ids=["001-prd"],
        question_ids=["question-1"],
        timestamp=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        metadata={"round": 1},
    )

    assert entry.model_dump(mode="json")["event_type"] == "clarification_requested"

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        ConversationEntry.model_validate(
            {
                **entry.model_dump(mode="python"),
                "artifact_ids": ["001-prd", "001-prd"],
            }
        )

    with pytest.raises(ValidationError, match="working_branch must differ"):
        WorkspaceDescriptor.model_validate(
            {
                **workspace_descriptor().model_dump(mode="python"),
                "working_branch": "main",
            }
        )

    with pytest.raises(ValidationError, match="must not contain credentials"):
        WorkspaceDescriptor.model_validate(
            {
                **workspace_descriptor().model_dump(mode="python"),
                "source_repo_url": "https://request-token@github.com/example/platform.git",
            }
        )


async def test_database_crud_for_workflows_artifacts_and_execution_logs(tmp_path: Path) -> None:
    """Persist, update, query, and delete all Phase 3 database models asynchronously."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'workflow-state.db'}")
    await database.create_schema()

    try:
        async with database.session() as session:
            workflow = WorkflowModel(
                workflow_id="workflow-1",
                workflow_schema_version="1.0",
                created_by_build_revision="test-build",
                last_executor_build_revision="test-build",
                status=WorkflowStatus.RUNNING,
                current_agent="engineer",
                confidence=0.75,
                retry_count=0,
                approval_state=ApprovalState.PENDING,
                conversation_history=[
                    {
                        "entry_id": "event-1",
                        "event_type": "artifact_produced",
                    }
                ],
                workspace_descriptor=workspace_descriptor().model_dump(mode="json"),
                checkpoints=[],
            )
            artifact = ArtifactModel(
                artifact_id="artifact-1",
                workflow_id="workflow-1",
                artifact_type="prd",
                schema_version="1.0",
                producer="product_manager",
                validation_status="valid",
                payload={"title": "Build the platform"},
                metadata_json={"sequence": 1},
            )
            execution_log = ExecutionLogModel(
                log_id="log-1",
                workflow_id="workflow-1",
                level=LogLevel.INFO,
                source="engineer",
                event="implementation_started",
                message="The engineering phase started.",
                context={"task_count": 2},
            )
            session.add_all([workflow, artifact, execution_log])
            await session.commit()

        async with database.session() as session:
            stored_workflow = await session.get(WorkflowModel, "workflow-1")
            artifact_result = await session.execute(
                select(ArtifactModel).where(ArtifactModel.artifact_id == "artifact-1")
            )
            stored_artifact = artifact_result.scalar_one()
            log_result = await session.execute(
                select(ExecutionLogModel).where(ExecutionLogModel.workflow_id == "workflow-1")
            )
            stored_log = log_result.scalar_one()

            assert stored_workflow is not None
            assert stored_workflow.status is WorkflowStatus.RUNNING
            assert stored_workflow.approval_state is ApprovalState.PENDING
            assert stored_workflow.workspace_descriptor["workspace_id"] == "workspace-1"
            assert stored_artifact is not None
            assert stored_artifact.payload == {"title": "Build the platform"}
            assert stored_artifact.metadata_json == {"sequence": 1}
            assert stored_log.level is LogLevel.INFO
            assert stored_log.context == {"task_count": 2}

            stored_workflow.status = WorkflowStatus.APPROVED
            stored_workflow.retry_count = 1
            await session.commit()

        async with database.session() as session:
            stored_workflow = await session.get(WorkflowModel, "workflow-1")

            assert stored_workflow is not None
            assert stored_workflow.status is WorkflowStatus.APPROVED
            assert stored_workflow.retry_count == 1

            await session.delete(stored_workflow)
            await session.commit()

        async with database.session() as session:
            deleted_workflow = await session.get(WorkflowModel, "workflow-1")
            deleted_artifact = await session.execute(
                select(ArtifactModel).where(ArtifactModel.artifact_id == "artifact-1")
            )
            deleted_logs = await session.execute(
                select(ExecutionLogModel).where(ExecutionLogModel.workflow_id == "workflow-1")
            )

            assert deleted_workflow is None
            assert deleted_artifact.scalars().all() == []
            assert deleted_logs.scalars().all() == []
    finally:
        await database.drop_schema()
        await database.dispose()
