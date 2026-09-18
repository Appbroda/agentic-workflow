"""The images somebody attached to a submission: metadata always, bytes only when asked for.

A submitter can describe a screen; without this they cannot show one. The bytes live in their
own table beside the other per-owner stores rather than inside an artifact payload, because
`artifact_json(prd)` is pasted into a prompt, sent as `input_text`, and returned whole by the
artifacts API -- a base64 blob in there would be three separate mistakes at once.

Two rules this module exists to keep:

- **Content is read by an explicit call.** `get_metadata` never loads the blob. Nothing should
  read five megabytes to answer "how big is it", and every caller but one wants the metadata.
- **Sniffed, never declared.** `sniff_image_media_type` is the only thing that decides what a
  set of bytes is. The filename and the browser's `Content-Type` are both the uploader's, and
  the stored `media_type` is what the bytes are later served back as.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from storage.db import Database
from storage.models import PRDAttachmentModel

# ------------------------------------------------------------------ the limits, in one place
#
# Quoted verbatim in the API's refusals and mirrored by the submission form, so the three
# never disagree about what is allowed. Small on purpose: this is the first user-supplied
# blob in this database and the first thing in it that grows without anybody deploying.

# One image. A screenshot of a full-page mock-up at retina scale is comfortably under this;
# a photograph of a whiteboard from a modern phone is not, and being told so at upload is
# better than a truncated read.
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
# Images per feature. Eight screens is a feature's worth of screens.
MAX_ATTACHMENTS_PER_FEATURE = 8
# And the total, which is the bound that actually protects the table: eight files each just
# under the per-file cap would be 40 MiB.
MAX_ATTACHMENT_BYTES_PER_FEATURE = 20 * 1024 * 1024

# What the platform accepts. Three raster formats a browser can produce and display and that
# both model providers document as image inputs. SVG is deliberately absent -- it is markup
# with script and remote-fetch reach, and it is served back to a browser; see
# `sniff_image_media_type`, which refuses it by its bytes rather than by its name.
ACCEPTED_MEDIA_TYPES: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")

# How long an attachment nobody submitted is kept before the sweep deletes it. A person who
# uploads three images and closes the tab must not leave them in the database forever; a day
# is long enough that somebody who wandered off mid-submission and came back has not lost
# their work.
UNBOUND_ATTACHMENT_TTL_SECONDS = 24 * 60 * 60

# How many forgotten uploads one sweep tick deletes at most. The doomed ids travel in an
# `IN` clause, and an unbounded list eventually exceeds a driver's bind-parameter cap; the
# sweep runs every thirty seconds, so any backlog beyond this drains within minutes.
SWEEP_BATCH_LIMIT = 500


class AttachmentError(ValueError):
    """An attachment was refused by the store, with a message meant for whoever uploaded it."""


@dataclass(frozen=True, slots=True)
class Attachment:
    """One attachment's record, with no bytes in it.

    `content_present` rather than the content: this is what every reader but the product
    manager wants, and a dataclass that could carry five megabytes is a dataclass that
    eventually does.
    """

    attachment_id: str
    owner_id: str
    feature_id: str | None
    filename: str
    media_type: str
    byte_size: int
    sha256: str
    content_present: bool
    created_at: datetime
    bound_at: datetime | None
    purged_at: datetime | None


@dataclass(frozen=True, slots=True)
class AttachmentStorageReport:
    """What the sweep found, for the one structured log line that accounts for the table."""

    bound_bytes: int
    unbound_bytes: int
    rows_deleted: int


class AttachmentStore(Protocol):
    """Read and write submitted attachment bytes, metadata apart from content throughout."""

    async def create(
        self, *, owner_id: str, filename: str, media_type: str, content: bytes
    ) -> Attachment:
        """Store one uploaded image and return its record. The identifier is the server's."""

    async def get_metadata(self, attachment_id: str) -> Attachment | None:
        """Return one attachment's record without reading its bytes, or nothing."""

    async def get_content(self, attachment_id: str) -> bytes | None:
        """Return one attachment's bytes, or nothing when the row is purged or absent."""

    async def bind(self, attachment_ids: Sequence[str], *, feature_id: str) -> None:
        """Attach a submission's images to a feature, in a transaction of this store's own.

        An attachment binds once: an id already bound to a *different* feature is an error,
        and one already bound to this feature is a replay and is done.
        """

    async def bind_within(
        self, session: AsyncSession, attachment_ids: Sequence[str], *, feature_id: str
    ) -> None:
        """Bind using a transaction the caller already holds.

        This is the shape acceptance needs. The feature row, its PRD artifact and the binding
        of the evidence that artifact quotes have to commit together or not at all: binding
        beforehand would leave attachments claimed by a feature a later conflict meant never
        existed, and binding afterwards would publish an artifact whose references resolve to
        unbound rows a sweep may delete.

        The durable implementation joins the session. The in-memory one has no transaction to
        join and binds immediately -- which is correct for it, and is why this is on the
        protocol rather than being a free function only one implementation can satisfy.
        """

    async def delete_unbound(self, attachment_id: str, *, owner_id: str) -> bool:
        """Forget one of this identity's unbound attachments, and say whether there was one."""

    async def purge_content_for_feature(self, feature_id: str) -> int:
        """Null the content of one feature's attachments, keeping every row, and count them."""

    async def sweep_unbound_before(self, cutoff: datetime) -> AttachmentStorageReport:
        """Delete unbound attachments older than the cutoff and report the table's size."""

    async def list_for_feature(self, feature_id: str) -> list[Attachment]:
        """Return one feature's bound attachments, oldest first."""

    async def list_for_owner(
        self, owner_id: str, *, unbound_only: bool = False
    ) -> list[Attachment]:
        """Return this identity's attachments, oldest first."""


def sniff_image_media_type(content: bytes) -> str | None:
    """Return what these bytes actually are, or nothing for anything else.

    Magic bytes only, and only for the three accepted formats plus SVG -- which is recognised
    precisely so it can be refused with a message that sends somebody to export a PNG rather
    than to try harder at the same thing.

    Nothing here decodes. A decoder is a new attack surface running over user-supplied bytes,
    and the mitigation chosen instead is exactly this: never decode, cap hard, sniff, and
    serve back sandboxed with `nosniff`.
    """
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    # RIFF....WEBP -- the four size bytes sit between the two tags.
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    if _looks_like_svg(content):
        return "image/svg+xml"
    return None


def _looks_like_svg(content: bytes) -> bool:
    """Recognise SVG well enough to refuse it, including behind a comment or a declaration.

    Deliberately generous: the point is that an SVG cannot reach the serving endpoint by
    being called `.png`, so a leading XML declaration, a byte-order mark or a comment must
    not be a way past it. It only has to be right about the files it rejects, and no accepted
    format's magic bytes are text.
    """
    head = content[:1024].lstrip(b"\xef\xbb\xbf").lstrip()
    if not head.startswith(b"<"):
        return False
    return b"<svg" in head.lower()


def new_attachment_id() -> str:
    """Return a server-allocated identifier, in the shape every other id here has."""
    return f"attachment-{uuid4()}"


def sha256_of(content: bytes) -> str:
    """Return the identity of these bytes, which is what the artifact quotes."""
    return hashlib.sha256(content).hexdigest()


def _validated(*, filename: str, media_type: str, content: bytes) -> tuple[str, str]:
    """Hold the two fields the store owns, and refuse what the table cannot represent.

    The size and type policy lives at the API boundary, where the refusals have to name the
    caps. What is checked here is what a store must check whoever calls it: that there are
    bytes, that they are not over the absolute cap, and that the type is one this platform
    serves back.
    """
    label = filename.strip()
    if not label:
        msg = "an attachment needs a filename"
        raise AttachmentError(msg)
    if len(label) > 512:
        label = label[:512]
    if not content:
        msg = "an attachment cannot be empty"
        raise AttachmentError(msg)
    if len(content) > MAX_ATTACHMENT_BYTES:
        msg = f"an attachment must be at most {MAX_ATTACHMENT_BYTES} bytes"
        raise AttachmentError(msg)
    if media_type not in ACCEPTED_MEDIA_TYPES:
        msg = f"unsupported attachment type: {media_type}"
        raise AttachmentError(msg)
    return label, media_type


class DatabaseAttachmentStore:
    """Attachments in PostgreSQL, with content read only when it is asked for by name."""

    def __init__(self, database: Database) -> None:
        """Bind the database these attachments live in."""
        self._database = database

    async def create(
        self, *, owner_id: str, filename: str, media_type: str, content: bytes
    ) -> Attachment:
        """Store one uploaded image. The caller has already sniffed and capped it."""
        label, sniffed = _validated(filename=filename, media_type=media_type, content=content)
        model = PRDAttachmentModel(
            attachment_id=new_attachment_id(),
            owner_id=owner_id,
            feature_id=None,
            filename=label,
            media_type=sniffed,
            byte_size=len(content),
            sha256=sha256_of(content),
            content=content,
            created_at=datetime.now(UTC),
        )
        async with self._database.session() as session:
            session.add(model)
            await session.commit()
            return _attachment(model)

    async def get_metadata(self, attachment_id: str) -> Attachment | None:
        """Return one record without its bytes.

        The column list is explicit rather than `session.get`: loading the ORM row would
        bring the blob into memory to answer a question about its size.
        """
        statement = select(
            PRDAttachmentModel.attachment_id,
            PRDAttachmentModel.owner_id,
            PRDAttachmentModel.feature_id,
            PRDAttachmentModel.filename,
            PRDAttachmentModel.media_type,
            PRDAttachmentModel.byte_size,
            PRDAttachmentModel.sha256,
            PRDAttachmentModel.content.isnot(None).label("content_present"),
            PRDAttachmentModel.created_at,
            PRDAttachmentModel.bound_at,
            PRDAttachmentModel.purged_at,
        ).where(PRDAttachmentModel.attachment_id == attachment_id)
        async with self._database.session() as session:
            row = (await session.execute(statement)).one_or_none()
        return None if row is None else _attachment_from_row(row)

    async def get_content(self, attachment_id: str) -> bytes | None:
        """Return the bytes, or nothing when the row is absent or its content is purged."""
        statement = select(PRDAttachmentModel.content).where(
            PRDAttachmentModel.attachment_id == attachment_id
        )
        async with self._database.session() as session:
            return await session.scalar(statement)

    async def bind(self, attachment_ids: Sequence[str], *, feature_id: str) -> None:
        """Bind a submission's attachments in a transaction of this store's own.

        Acceptance does not come through here -- it holds its own transaction and calls
        `bind_within`. This is the standalone form, for a caller with no transaction to
        share.
        """
        async with self._database.session() as session:
            await bind_in_session(session, attachment_ids, feature_id=feature_id)
            await session.commit()

    async def bind_within(
        self, session: AsyncSession, attachment_ids: Sequence[str], *, feature_id: str
    ) -> None:
        """Bind inside the caller's transaction, so acceptance is one commit."""
        await bind_in_session(session, attachment_ids, feature_id=feature_id)

    async def delete_unbound(self, attachment_id: str, *, owner_id: str) -> bool:
        """Forget one unbound attachment of this identity's. A bound one is not deletable."""
        async with self._database.session() as session:
            model = await session.get(PRDAttachmentModel, attachment_id)
            if model is None or model.owner_id != owner_id or model.feature_id is not None:
                return False
            await session.delete(model)
            await session.commit()
        return True

    async def purge_content_for_feature(self, feature_id: str) -> int:
        """Null the content of one feature's attachments and keep every row."""
        statement = select(PRDAttachmentModel).where(
            PRDAttachmentModel.feature_id == feature_id,
            PRDAttachmentModel.content.isnot(None),
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
            now = datetime.now(UTC)
            for model in rows:
                model.content = None
                model.purged_at = now
            await session.commit()
            return len(rows)

    async def sweep_unbound_before(self, cutoff: datetime) -> AttachmentStorageReport:
        """Delete forgotten uploads and report what the table holds afterwards."""
        async with self._database.session() as session:
            # Bounded, because the id list becomes an `IN` clause: a backlog large enough to
            # exceed a driver's parameter cap must drain across ticks (one every 30s), not
            # fail every tick forever.
            doomed = (
                (
                    await session.execute(
                        select(PRDAttachmentModel.attachment_id)
                        .where(
                            PRDAttachmentModel.feature_id.is_(None),
                            PRDAttachmentModel.created_at < cutoff,
                        )
                        .limit(SWEEP_BATCH_LIMIT)
                    )
                )
                .scalars()
                .all()
            )
            deleted = 0
            if doomed:
                # The delete repeats the whole predicate, not just the ids: an attachment
                # bound between the select and this statement is no longer forgotten, and a
                # delete matching on id alone would destroy an image a feature just accepted.
                result = await session.execute(
                    delete(PRDAttachmentModel).where(
                        PRDAttachmentModel.attachment_id.in_(doomed),
                        PRDAttachmentModel.feature_id.is_(None),
                        PRDAttachmentModel.created_at < cutoff,
                    )
                )
                deleted = int(cast("CursorResult[Any]", result).rowcount or 0)
            await session.commit()
            bound = await session.scalar(
                select(func.coalesce(func.sum(PRDAttachmentModel.byte_size), 0)).where(
                    PRDAttachmentModel.feature_id.isnot(None),
                    PRDAttachmentModel.content.isnot(None),
                )
            )
            unbound = await session.scalar(
                select(func.coalesce(func.sum(PRDAttachmentModel.byte_size), 0)).where(
                    PRDAttachmentModel.feature_id.is_(None)
                )
            )
        return AttachmentStorageReport(
            bound_bytes=int(bound or 0),
            unbound_bytes=int(unbound or 0),
            rows_deleted=deleted,
        )

    async def list_for_feature(self, feature_id: str) -> list[Attachment]:
        """Return one feature's attachments, oldest first, without their bytes."""
        return await self._listed(PRDAttachmentModel.feature_id == feature_id)

    async def list_for_owner(
        self, owner_id: str, *, unbound_only: bool = False
    ) -> list[Attachment]:
        """Return this identity's attachments, oldest first, without their bytes."""
        condition = PRDAttachmentModel.owner_id == owner_id
        if unbound_only:
            return await self._listed(condition, PRDAttachmentModel.feature_id.is_(None))
        return await self._listed(condition)

    async def _listed(self, *conditions: object) -> list[Attachment]:
        """Run one metadata-only query, which is the only shape of read this store does."""
        statement = (
            select(
                PRDAttachmentModel.attachment_id,
                PRDAttachmentModel.owner_id,
                PRDAttachmentModel.feature_id,
                PRDAttachmentModel.filename,
                PRDAttachmentModel.media_type,
                PRDAttachmentModel.byte_size,
                PRDAttachmentModel.sha256,
                PRDAttachmentModel.content.isnot(None).label("content_present"),
                PRDAttachmentModel.created_at,
                PRDAttachmentModel.bound_at,
                PRDAttachmentModel.purged_at,
            )
            .where(*conditions)  # type: ignore[arg-type]
            .order_by(PRDAttachmentModel.created_at, PRDAttachmentModel.attachment_id)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return [_attachment_from_row(row) for row in rows]


async def bind_in_session(
    session: AsyncSession, attachment_ids: Sequence[str], *, feature_id: str
) -> None:
    """Bind attachments to a feature using a transaction the caller already holds.

    This is the shape acceptance needs: the feature row, its PRD artifact and the binding of
    the evidence that artifact quotes all commit together, in `_create_or_replay`'s single
    transaction. Binding beforehand would leave attachments claimed by a feature that a later
    conflict meant never existed.

    An attachment binds once. One already bound to *this* feature is the idempotent-replay
    arm doing its job and is treated as done; one bound to another feature, or missing, is an
    error naming the id, because a second submission naming somebody's evidence is not a
    thing this platform lets happen quietly.
    """
    now = datetime.now(UTC)
    for attachment_id in attachment_ids:
        model = await session.get(PRDAttachmentModel, attachment_id)
        if model is None:
            msg = f"attachment not found: {attachment_id}"
            raise AttachmentError(msg)
        if model.feature_id == feature_id:
            continue
        if model.feature_id is not None:
            msg = f"attachment is already attached to another feature: {attachment_id}"
            raise AttachmentError(msg)
        model.feature_id = feature_id
        model.bound_at = now
    await session.flush()


class InMemoryAttachmentStore:
    """The same contract without a database, for isolated applications.

    Content is held apart from the record for the reason the database keeps it in its own
    column: a metadata read that could reach the bytes is a metadata read that eventually
    does, and the test that proves it does not needs somewhere to look.
    """

    def __init__(self) -> None:
        """Start with nothing uploaded."""
        self._rows: dict[str, Attachment] = {}
        self._content: dict[str, bytes] = {}
        # Counts how many times bytes were actually read, so a test can assert that a
        # metadata read did not.
        self.content_reads = 0

    async def create(
        self, *, owner_id: str, filename: str, media_type: str, content: bytes
    ) -> Attachment:
        """Store one uploaded image, generating its identifier."""
        label, sniffed = _validated(filename=filename, media_type=media_type, content=content)
        record = Attachment(
            attachment_id=new_attachment_id(),
            owner_id=owner_id,
            feature_id=None,
            filename=label,
            media_type=sniffed,
            byte_size=len(content),
            sha256=sha256_of(content),
            content_present=True,
            created_at=datetime.now(UTC),
            bound_at=None,
            purged_at=None,
        )
        self._rows[record.attachment_id] = record
        self._content[record.attachment_id] = bytes(content)
        return record

    async def get_metadata(self, attachment_id: str) -> Attachment | None:
        """Return one record. Never touches `_content`."""
        return self._rows.get(attachment_id)

    async def get_content(self, attachment_id: str) -> bytes | None:
        """Return the bytes, counting the read."""
        if attachment_id not in self._rows:
            return None
        self.content_reads += 1
        return self._content.get(attachment_id)

    async def bind(self, attachment_ids: Sequence[str], *, feature_id: str) -> None:
        """Bind a submission's attachments, refusing one already bound elsewhere."""
        now = datetime.now(UTC)
        for attachment_id in attachment_ids:
            existing = self._rows.get(attachment_id)
            if existing is None:
                msg = f"attachment not found: {attachment_id}"
                raise AttachmentError(msg)
            if existing.feature_id == feature_id:
                continue
            if existing.feature_id is not None:
                msg = f"attachment is already attached to another feature: {attachment_id}"
                raise AttachmentError(msg)
        for attachment_id in attachment_ids:
            existing = self._rows[attachment_id]
            if existing.feature_id == feature_id:
                continue
            self._rows[attachment_id] = replace(existing, feature_id=feature_id, bound_at=now)

    async def bind_within(
        self, session: AsyncSession, attachment_ids: Sequence[str], *, feature_id: str
    ) -> None:
        """Bind immediately. There is no transaction here to join, and saying so is honest."""
        del session
        await self.bind(attachment_ids, feature_id=feature_id)

    async def delete_unbound(self, attachment_id: str, *, owner_id: str) -> bool:
        """Forget one unbound attachment of this identity's."""
        existing = self._rows.get(attachment_id)
        if existing is None or existing.owner_id != owner_id or existing.feature_id is not None:
            return False
        del self._rows[attachment_id]
        self._content.pop(attachment_id, None)
        return True

    async def purge_content_for_feature(self, feature_id: str) -> int:
        """Drop the bytes of one feature's attachments and keep the records."""
        now = datetime.now(UTC)
        purged = 0
        for attachment_id, record in list(self._rows.items()):
            if record.feature_id != feature_id or not record.content_present:
                continue
            self._rows[attachment_id] = replace(record, content_present=False, purged_at=now)
            self._content.pop(attachment_id, None)
            purged += 1
        return purged

    async def sweep_unbound_before(self, cutoff: datetime) -> AttachmentStorageReport:
        """Delete forgotten uploads and report what is held afterwards."""
        deleted = 0
        for attachment_id, record in list(self._rows.items()):
            if record.feature_id is None and record.created_at < cutoff:
                del self._rows[attachment_id]
                self._content.pop(attachment_id, None)
                deleted += 1
        bound = sum(
            item.byte_size
            for item in self._rows.values()
            if item.feature_id is not None and item.content_present
        )
        unbound = sum(item.byte_size for item in self._rows.values() if item.feature_id is None)
        return AttachmentStorageReport(
            bound_bytes=bound, unbound_bytes=unbound, rows_deleted=deleted
        )

    async def list_for_feature(self, feature_id: str) -> list[Attachment]:
        """Return one feature's attachments, oldest first."""
        return sorted(
            (item for item in self._rows.values() if item.feature_id == feature_id),
            key=lambda item: (item.created_at, item.attachment_id),
        )

    async def list_for_owner(
        self, owner_id: str, *, unbound_only: bool = False
    ) -> list[Attachment]:
        """Return this identity's attachments, oldest first."""
        return sorted(
            (
                item
                for item in self._rows.values()
                if item.owner_id == owner_id and (not unbound_only or item.feature_id is None)
            ),
            key=lambda item: (item.created_at, item.attachment_id),
        )


def _attachment(model: PRDAttachmentModel) -> Attachment:
    """Project one ORM row onto the record, without carrying its bytes any further."""
    return Attachment(
        attachment_id=model.attachment_id,
        owner_id=model.owner_id,
        feature_id=model.feature_id,
        filename=model.filename,
        media_type=model.media_type,
        byte_size=model.byte_size,
        sha256=model.sha256,
        content_present=model.content is not None,
        created_at=model.created_at,
        bound_at=model.bound_at,
        purged_at=model.purged_at,
    )


def _attachment_from_row(row: object) -> Attachment:
    """Project one metadata-only result row onto the record."""
    return Attachment(
        attachment_id=row.attachment_id,  # type: ignore[attr-defined]
        owner_id=row.owner_id,  # type: ignore[attr-defined]
        feature_id=row.feature_id,  # type: ignore[attr-defined]
        filename=row.filename,  # type: ignore[attr-defined]
        media_type=row.media_type,  # type: ignore[attr-defined]
        byte_size=row.byte_size,  # type: ignore[attr-defined]
        sha256=row.sha256,  # type: ignore[attr-defined]
        content_present=bool(row.content_present),  # type: ignore[attr-defined]
        created_at=row.created_at,  # type: ignore[attr-defined]
        bound_at=row.bound_at,  # type: ignore[attr-defined]
        purged_at=row.purged_at,  # type: ignore[attr-defined]
    )


__all__ = [
    "ACCEPTED_MEDIA_TYPES",
    "MAX_ATTACHMENTS_PER_FEATURE",
    "MAX_ATTACHMENT_BYTES",
    "MAX_ATTACHMENT_BYTES_PER_FEATURE",
    "UNBOUND_ATTACHMENT_TTL_SECONDS",
    "Attachment",
    "AttachmentError",
    "AttachmentStorageReport",
    "AttachmentStore",
    "DatabaseAttachmentStore",
    "InMemoryAttachmentStore",
    "bind_in_session",
    "new_attachment_id",
    "sha256_of",
    "sniff_image_media_type",
]
