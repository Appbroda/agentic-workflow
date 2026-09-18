"""The store that holds a submitted image's bytes (item 89-, Part A).

Two properties carry the weight here, and both are asserted against the durable
implementation rather than only the in-memory one:

- a metadata read never loads the blob, and
- a row whose content has been purged still answers metadata, because the artifact that
  quoted its hash is still readable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Connection, event, text

from storage.attachment_store import (
    ACCEPTED_MEDIA_TYPES,
    Attachment,
    AttachmentError,
    DatabaseAttachmentStore,
    InMemoryAttachmentStore,
    sniff_image_media_type,
)
from storage.db import EphemeralDatabase
from storage.models import PRDAttachmentModel
from tests import attachment_fixtures as fixtures

OWNER = "actor-1"


def durable() -> DatabaseAttachmentStore:
    """One attachment store on a private database of this test's own."""
    return DatabaseAttachmentStore(EphemeralDatabase())


@pytest.mark.parametrize(("path", "media_type"), fixtures.ACCEPTED)
@pytest.mark.asyncio
async def test_each_accepted_type_round_trips_byte_for_byte(path: Path, media_type: str) -> None:
    """The bytes that come back are the bytes that went in, and the hash says which."""
    store = durable()
    content = fixtures.content(path)

    record = await store.create(
        owner_id=OWNER, filename=path.name, media_type=media_type, content=content
    )
    fetched = await store.get_content(record.attachment_id)
    metadata = await store.get_metadata(record.attachment_id)

    assert fetched == content
    assert metadata is not None
    assert metadata.media_type == media_type
    assert metadata.byte_size == len(content)
    assert metadata.sha256 == fixtures.digest(path)
    assert metadata.feature_id is None
    assert metadata.content_present is True


@pytest.mark.parametrize(("path", "media_type"), fixtures.ACCEPTED)
@pytest.mark.asyncio
async def test_the_in_memory_store_round_trips_the_same_way(path: Path, media_type: str) -> None:
    """The isolated application's store answers exactly what the durable one does."""
    store = InMemoryAttachmentStore()
    content = fixtures.content(path)

    record = await store.create(
        owner_id=OWNER, filename=path.name, media_type=media_type, content=content
    )

    assert await store.get_content(record.attachment_id) == content
    assert record.sha256 == fixtures.digest(path)
    assert record.media_type == media_type


@pytest.mark.asyncio
async def test_a_metadata_read_does_not_fetch_the_content() -> None:
    """Nothing should load five megabytes to answer "how big is it"."""
    store = InMemoryAttachmentStore()
    record = await store.create(
        owner_id=OWNER,
        filename="pixel.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )

    await store.get_metadata(record.attachment_id)
    await store.list_for_owner(OWNER)

    assert store.content_reads == 0
    await store.get_content(record.attachment_id)
    assert store.content_reads == 1


@pytest.mark.asyncio
async def test_the_durable_metadata_read_never_selects_the_blob() -> None:
    """The same rule, proved on the SQL the deployment's store actually issues.

    Asserted on the executed statement rather than on a counter, because the counter belongs
    to the in-memory store and this is the implementation a deployment runs. The read is
    allowed to ask *whether* content is present -- that is a boolean the database computes --
    and is not allowed to select the column itself.
    """
    store = durable()
    record = await store.create(
        owner_id=OWNER,
        filename="pixel.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )
    statements: list[str] = []

    @event.listens_for(store._database.engine.sync_engine, "before_cursor_execute")
    def capture(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    metadata = await store.get_metadata(record.attachment_id)

    assert metadata is not None
    reads = [item for item in statements if item.lstrip().upper().startswith("SELECT")]
    assert reads, "the metadata read issued no select"
    for read in reads:
        columns = read.split("FROM", 1)[0]
        assert "IS NOT NULL" in columns, "the read should ask whether content is present"
        assert "prd_attachments.content \n" not in columns
        assert "prd_attachments.content," not in columns


@pytest.mark.asyncio
async def test_a_purged_row_still_answers_metadata() -> None:
    """The record of what was submitted outlives the thing submitted."""
    store = durable()
    record = await store.create(
        owner_id=OWNER,
        filename="pixel.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )
    await store.bind([record.attachment_id], feature_id="feature-1")

    purged = await store.purge_content_for_feature("feature-1")
    metadata = await store.get_metadata(record.attachment_id)
    content = await store.get_content(record.attachment_id)

    assert purged == 1
    assert content is None
    assert metadata is not None
    assert metadata.content_present is False
    assert metadata.purged_at is not None
    # Still the whole record: filename, size and hash are what the artifact quotes.
    assert metadata.filename == "pixel.png"
    assert metadata.byte_size == len(fixtures.content(fixtures.PNG))
    assert metadata.sha256 == fixtures.digest(fixtures.PNG)


@pytest.mark.asyncio
async def test_binding_is_once_and_a_replay_of_the_same_feature_is_done() -> None:
    """An attachment binds once; the same feature binding again is the replay arm."""
    store = durable()
    record = await store.create(
        owner_id=OWNER,
        filename="pixel.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )

    await store.bind([record.attachment_id], feature_id="feature-1")
    # A replay of the same accepted request finds it already bound to this feature.
    await store.bind([record.attachment_id], feature_id="feature-1")
    with pytest.raises(AttachmentError, match="already attached"):
        await store.bind([record.attachment_id], feature_id="feature-2")

    bound = await store.get_metadata(record.attachment_id)
    assert bound is not None
    assert bound.feature_id == "feature-1"
    assert bound.bound_at is not None


@pytest.mark.asyncio
async def test_a_bound_attachment_is_not_deletable_and_an_unbound_one_is() -> None:
    """A bound attachment is part of a submitted record; retiring the feature removes it."""
    store = durable()
    png = fixtures.content(fixtures.PNG)
    kept = await store.create(owner_id=OWNER, filename="a.png", media_type="image/png", content=png)
    loose = await store.create(
        owner_id=OWNER, filename="b.png", media_type="image/png", content=png
    )
    await store.bind([kept.attachment_id], feature_id="feature-1")

    assert await store.delete_unbound(kept.attachment_id, owner_id=OWNER) is False
    assert await store.delete_unbound(loose.attachment_id, owner_id="somebody-else") is False
    assert await store.delete_unbound(loose.attachment_id, owner_id=OWNER) is True
    assert await store.get_metadata(loose.attachment_id) is None


@pytest.mark.asyncio
async def test_the_sweep_deletes_the_forgotten_and_keeps_the_recent_and_the_bound() -> None:
    """Three rows, one cutoff, and only the abandoned upload goes."""
    store = durable()
    png = fixtures.content(fixtures.PNG)
    old = await store.create(
        owner_id=OWNER, filename="old.png", media_type="image/png", content=png
    )
    recent = await store.create(
        owner_id=OWNER, filename="new.png", media_type="image/png", content=png
    )
    bound = await store.create(
        owner_id=OWNER, filename="bound.png", media_type="image/png", content=png
    )
    await store.bind([bound.attachment_id], feature_id="feature-1")
    await _age(store, old.attachment_id, hours=48)

    report = await store.sweep_unbound_before(datetime.now(UTC) - timedelta(hours=24))

    assert report.rows_deleted == 1
    assert await store.get_metadata(old.attachment_id) is None
    assert await store.get_metadata(recent.attachment_id) is not None
    assert await store.get_metadata(bound.attachment_id) is not None
    # The accounting the sweep logs: bound and unbound bytes, counted apart.
    assert report.bound_bytes == len(png)
    assert report.unbound_bytes == len(png)


@pytest.mark.asyncio
async def test_the_sweeps_delete_repeats_the_unbound_predicate_and_bounds_its_batch() -> None:
    """The delete must not trust the ids it selected a moment earlier.

    Between the select and the delete, an accepting transaction can bind one of the doomed
    ids to a feature. A delete matching on id alone would destroy an image a feature just
    accepted, so the executed statement carries the whole predicate again -- proved on the
    SQL the deployment's store actually issues, exactly as the metadata-read rule above is.
    And the id list becomes an `IN` clause, so the select that builds it is bounded.
    """
    store = durable()
    png = fixtures.content(fixtures.PNG)
    old = await store.create(
        owner_id=OWNER, filename="old.png", media_type="image/png", content=png
    )
    await _age(store, old.attachment_id, hours=48)
    statements: list[str] = []

    @event.listens_for(store._database.engine.sync_engine, "before_cursor_execute")
    def capture(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    report = await store.sweep_unbound_before(datetime.now(UTC) - timedelta(hours=24))

    assert report.rows_deleted == 1
    deletes = [item for item in statements if item.lstrip().upper().startswith("DELETE")]
    assert deletes, "the sweep issued no delete"
    for statement in deletes:
        assert "feature_id IS NULL" in statement, statement
        assert "created_at" in statement, statement
    selects = [item for item in statements if item.lstrip().upper().startswith("SELECT")]
    doomed_reads = [
        item for item in selects if "feature_id IS NULL" in item and "attachment_id" in item
    ]
    assert doomed_reads, "the sweep issued no candidate select"
    assert any("LIMIT" in item.upper() for item in doomed_reads), doomed_reads


@pytest.mark.asyncio
async def test_a_row_bound_between_the_select_and_the_delete_survives_the_sweep() -> None:
    """The race itself, driven through the statement seam the store already exposes.

    The bind lands after the sweep has chosen its doomed ids and before its delete executes.
    The repeated predicate is what keeps the just-accepted image alive.
    """
    store = durable()
    png = fixtures.content(fixtures.PNG)
    old = await store.create(
        owner_id=OWNER, filename="old.png", media_type="image/png", content=png
    )
    await _age(store, old.attachment_id, hours=48)
    bound_between: list[bool] = []

    @event.listens_for(store._database.engine.sync_engine, "before_cursor_execute")
    def bind_before_the_delete(
        connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if not statement.lstrip().upper().startswith("DELETE") or bound_between:
            return
        bound_between.append(True)
        # On the sweep's own connection, so SQLite's single writer cannot deadlock the test:
        # what is being proven is the statement's predicate, not the isolation level.
        assert isinstance(connection, Connection)
        connection.execute(
            text(
                "UPDATE prd_attachments SET feature_id = 'feature-raced' WHERE attachment_id = :id"
            ),
            {"id": old.attachment_id},
        )

    report = await store.sweep_unbound_before(datetime.now(UTC) - timedelta(hours=24))

    assert bound_between == [True], "the delete never executed"
    assert report.rows_deleted == 0
    survivor = await store.get_metadata(old.attachment_id)
    assert survivor is not None
    assert survivor.feature_id == "feature-raced"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (fixtures.PNG, "image/png"),
        (fixtures.JPEG, "image/jpeg"),
        (fixtures.WEBP, "image/webp"),
        (fixtures.SVG, "image/svg+xml"),
        (fixtures.ZIP_NAMED_PNG, None),
    ],
)
def test_the_sniffer_reads_the_bytes_and_not_the_name(path: Path, expected: str | None) -> None:
    """Every fixture is committed, so these expectations cannot drift with an encoder."""
    assert sniff_image_media_type(fixtures.content(path)) == expected


def test_svg_is_recognised_behind_a_declaration_or_a_comment() -> None:
    """A leading XML declaration, byte-order mark or comment is not a way past the refusal."""
    body = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    assert sniff_image_media_type(b"\xef\xbb\xbf" + body) == "image/svg+xml"
    assert sniff_image_media_type(b"<!-- a mock -->\n" + body) == "image/svg+xml"
    assert sniff_image_media_type(b"<?xml version='1.0'?>\n" + body) == "image/svg+xml"


def test_the_accepted_set_is_the_three_raster_types() -> None:
    """SVG is not an image for this purpose, and the constant is where that is stated."""
    assert ACCEPTED_MEDIA_TYPES == ("image/png", "image/jpeg", "image/webp")
    assert "image/svg+xml" not in ACCEPTED_MEDIA_TYPES


@pytest.mark.asyncio
async def test_the_store_refuses_what_the_table_cannot_represent() -> None:
    """Whoever calls it: empty bytes, a nameless file, and a type nothing serves back."""
    store = InMemoryAttachmentStore()
    with pytest.raises(AttachmentError, match="cannot be empty"):
        await store.create(owner_id=OWNER, filename="a.png", media_type="image/png", content=b"")
    with pytest.raises(AttachmentError, match="needs a filename"):
        await store.create(
            owner_id=OWNER,
            filename="  ",
            media_type="image/png",
            content=fixtures.content(fixtures.PNG),
        )
    with pytest.raises(AttachmentError, match="unsupported attachment type"):
        await store.create(
            owner_id=OWNER,
            filename="a.svg",
            media_type="image/svg+xml",
            content=fixtures.content(fixtures.SVG),
        )


def test_the_record_carries_no_bytes() -> None:
    """The one field that must never appear: a metadata record with content in it."""
    assert "content" not in Attachment.__dataclass_fields__
    assert "content_present" in Attachment.__dataclass_fields__


async def _age(store: DatabaseAttachmentStore, attachment_id: str, *, hours: int) -> None:
    """Move one row's `created_at` back, which is the only way to test a time-based sweep."""
    async with store._database.session() as session:
        model = await session.get(PRDAttachmentModel, attachment_id)
        assert model is not None
        model.created_at = datetime.now(UTC) - timedelta(hours=hours)
        await session.commit()
