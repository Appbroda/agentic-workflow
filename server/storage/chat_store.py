"""Durable and in-memory transcripts for the feature assistant."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import count
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

from services.feature_chat import ChatMessage
from storage.db import Database
from storage.models import FeatureChatMessageModel


class DatabaseChatMessageStore:
    """Persist the conversation beside the feature it belongs to."""

    def __init__(self, database: Database) -> None:
        """Bind the shared database used by the rest of the feature record."""
        self._database = database

    async def append(self, feature_id: str, message: ChatMessage) -> ChatMessage:
        """Persist one message and return it with the identifier a confirmation will use."""
        row = FeatureChatMessageModel(
            feature_id=feature_id,
            role=message.role,
            content=message.content,
            proposed_action=message.proposed_action,
            action_status=message.action_status,
            action_result=message.action_result,
            action_id=message.action_id,
            created_at=datetime.now(tz=UTC),
        )
        async with self._database.session() as session:
            session.add(row)
            # Committed here, not merely flushed. `Database.session` rolls back on error and
            # otherwise closes without committing, so a flush alone assigned the identifier,
            # returned a complete-looking message and then discarded it: live, the assistant
            # answered and the transcript stayed empty.
            await session.commit()
            await session.refresh(row)
            return _as_message(row)

    async def history(self, feature_id: str, *, limit: int) -> list[ChatMessage]:
        """Return the most recent messages in conversational order.

        Read newest-first and reversed: a long conversation should cost the last `limit`
        messages, not the whole transcript.
        """
        statement = (
            select(FeatureChatMessageModel)
            .where(FeatureChatMessageModel.feature_id == feature_id)
            .order_by(FeatureChatMessageModel.id.desc())
            .limit(limit)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [_as_message(row) for row in reversed(rows)]

    async def get(self, feature_id: str, message_id: int) -> ChatMessage | None:
        """Return one message, scoped to its feature so an id cannot reach another's."""
        statement = select(FeatureChatMessageModel).where(
            FeatureChatMessageModel.id == message_id,
            FeatureChatMessageModel.feature_id == feature_id,
        )
        async with self._database.session() as session:
            row = (await session.execute(statement)).scalar_one_or_none()
        return None if row is None else _as_message(row)

    async def claim_action(self, feature_id: str, message_id: int) -> ChatMessage | None:
        """Atomically claim one pending proposal before its side effect begins."""
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(FeatureChatMessageModel)
                    .where(
                        FeatureChatMessageModel.id == message_id,
                        FeatureChatMessageModel.feature_id == feature_id,
                        FeatureChatMessageModel.action_status == "pending",
                    )
                    .values(action_status="executing")
                ),
            )
            await session.commit()
            if result.rowcount != 1:
                return None
            row = (
                await session.execute(
                    select(FeatureChatMessageModel).where(
                        FeatureChatMessageModel.id == message_id,
                        FeatureChatMessageModel.feature_id == feature_id,
                    )
                )
            ).scalar_one()
            return _as_message(row)

    async def record_action_outcome(
        self, feature_id: str, message_id: int, *, status: str, result: str
    ) -> ChatMessage:
        """Record what happened to a proposal, conditional on it still being pending.

        Only the request that atomically claimed the proposal may finish it. Retained for
        rejection, which is decided here and nowhere else; a confirmed action's outcome is
        mirrored from its durable record by ``project_action_status`` instead.
        """
        async with self._database.session() as session:
            await session.execute(
                update(FeatureChatMessageModel)
                .where(
                    FeatureChatMessageModel.id == message_id,
                    FeatureChatMessageModel.feature_id == feature_id,
                    FeatureChatMessageModel.action_status == "executing",
                )
                .values(action_status=status, action_result=result)
            )
            await session.commit()
            row = (
                await session.execute(
                    select(FeatureChatMessageModel).where(
                        FeatureChatMessageModel.id == message_id,
                        FeatureChatMessageModel.feature_id == feature_id,
                    )
                )
            ).scalar_one()
            return _as_message(row)

    async def attach_action(
        self, feature_id: str, message_id: int, *, action_id: str, status: str
    ) -> ChatMessage:
        """Bind a confirmed proposal to the durable action it became."""
        return await self._write(
            feature_id,
            message_id,
            values={"action_id": action_id, "action_status": status},
        )

    async def project_action_status(
        self, feature_id: str, message_id: int, *, status: str, result: str | None
    ) -> ChatMessage:
        """Mirror a durable action's state onto its message, unconditionally.

        Deliberately not guarded on the message's current status. The guarded write this
        replaces could only be performed by the request that had claimed the proposal, so a
        message stranded mid-execution by a crash could never be corrected afterwards -- not
        by a later reader, and not by recovery. The action record is the authority here; the
        message is a view of it.
        """
        values: dict[str, Any] = {"action_status": status}
        if result is not None:
            values["action_result"] = result
        return await self._write(feature_id, message_id, values=values)

    async def _write(
        self, feature_id: str, message_id: int, *, values: dict[str, Any]
    ) -> ChatMessage:
        """Apply an update scoped to one feature's message and return the stored row."""
        async with self._database.session() as session:
            await session.execute(
                update(FeatureChatMessageModel)
                .where(
                    FeatureChatMessageModel.id == message_id,
                    FeatureChatMessageModel.feature_id == feature_id,
                )
                .values(**values)
            )
            await session.commit()
            row = (
                await session.execute(
                    select(FeatureChatMessageModel).where(
                        FeatureChatMessageModel.id == message_id,
                        FeatureChatMessageModel.feature_id == feature_id,
                    )
                )
            ).scalar_one()
            return _as_message(row)


class InMemoryChatMessageStore:
    """Transcript for mock-mode features and tests, with the same contract."""

    def __init__(self) -> None:
        """Start empty with a monotonic identifier sequence."""
        self._messages: dict[str, list[ChatMessage]] = {}
        self._ids = count(1)

    async def append(self, feature_id: str, message: ChatMessage) -> ChatMessage:
        """Assign an identifier and keep the message in order."""
        message.id = next(self._ids)
        message.created_at = datetime.now(tz=UTC)
        self._messages.setdefault(feature_id, []).append(message)
        return message

    async def history(self, feature_id: str, *, limit: int) -> list[ChatMessage]:
        """Return the last `limit` messages in order."""
        return list(self._messages.get(feature_id, []))[-limit:]

    async def get(self, feature_id: str, message_id: int) -> ChatMessage | None:
        """Return one message from this feature's transcript."""
        return next(
            (item for item in self._messages.get(feature_id, []) if item.id == message_id), None
        )

    async def claim_action(self, feature_id: str, message_id: int) -> ChatMessage | None:
        """Claim synchronously so concurrent tasks cannot both observe a pending proposal."""
        message = await self.get(feature_id, message_id)
        if message is None or message.action_status != "pending":
            return None
        message.action_status = "executing"
        return message

    async def record_action_outcome(
        self, feature_id: str, message_id: int, *, status: str, result: str
    ) -> ChatMessage:
        """Record the decision on a proposal."""
        message = await self.get(feature_id, message_id)
        if message is None:
            msg = f"unknown chat message: {message_id}"
            raise KeyError(msg)
        message.action_status = status
        message.action_result = result
        return message

    async def attach_action(
        self, feature_id: str, message_id: int, *, action_id: str, status: str
    ) -> ChatMessage:
        """Bind a confirmed proposal to the durable action it became."""
        message = await self._require(feature_id, message_id)
        message.action_id = action_id
        message.action_status = status
        return message

    async def project_action_status(
        self, feature_id: str, message_id: int, *, status: str, result: str | None
    ) -> ChatMessage:
        """Mirror a durable action's state onto its message, unconditionally."""
        message = await self._require(feature_id, message_id)
        message.action_status = status
        if result is not None:
            message.action_result = result
        return message

    async def _require(self, feature_id: str, message_id: int) -> ChatMessage:
        """Return one message or say which identifier was not found."""
        message = await self.get(feature_id, message_id)
        if message is None:
            msg = f"unknown chat message: {message_id}"
            raise KeyError(msg)
        return message


def _as_message(row: FeatureChatMessageModel) -> ChatMessage:
    """Project a durable row onto the shape the service and API share."""
    return ChatMessage(
        id=row.id,
        role=row.role,
        content=row.content,
        proposed_action=dict(row.proposed_action) if row.proposed_action else None,
        action_status=row.action_status,
        action_result=row.action_result,
        action_id=row.action_id,
        created_at=row.created_at,
    )


__all__ = ["DatabaseChatMessageStore", "InMemoryChatMessageStore"]
