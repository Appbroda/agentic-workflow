"""What the PRD artifact records about an attached image (89-, Part D).

The load-bearing test in this file is the length bound. `artifact_json(prd)` is pasted into
the product-manager prompt, sent as that call's `input_text`, and returned whole by the
artifacts API -- so a `data` field somebody helpfully adds later would be sent to the model as
text, counted against nothing, and echoed to every reader of the artifact list. Three images
must cost a few hundred bytes of rendered JSON, not a few megabytes, and that is asserted
rather than assumed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agents.shared.contracts import artifact_json
from artifacts.schemas import PRDArtifact, PRDAttachment
from tests import attachment_fixtures as fixtures


def attachment(marker: str, **overrides: object) -> PRDAttachment:
    """One recorded attachment, with the committed fixture's real hash and size."""
    values: dict[str, object] = {
        "attachment_id": f"attachment-{marker}",
        "marker": marker,
        "caption": "The error a user sees after a failed login",
        "filename": f"{marker}.png",
        "media_type": "image/png",
        "byte_size": len(fixtures.content(fixtures.PNG)),
        "sha256": fixtures.digest(fixtures.PNG),
    }
    values.update(overrides)
    return PRDAttachment.model_validate(values)


def prd(*attachments: PRDAttachment) -> PRDArtifact:
    """One submitted PRD artifact carrying the given attachment records."""
    return PRDArtifact(
        schema_version="1.0.0",
        workflow_id="feature-1",
        artifact_id="001_prd.json",
        producer="api",
        timestamp=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
        metadata={"source": "api"},
        validation_status="valid",
        title="Login audit trail",
        problem_statement="Support staff need to trace failed login attempts.",
        attachments=list(attachments),
    )


def test_the_artifact_round_trips_through_persistence_with_its_attachments() -> None:
    """What the store writes and reads back is what was submitted, field for field.

    Through `model_validate_json` over the dumped payload, because that is exactly what the
    feature store does with the JSON column it persisted -- a strict schema will not coerce a
    timestamp string, so a test that used `model_validate` on a JSON dump would be exercising
    a path no reader takes.
    """
    original = prd(attachment("login-error"), attachment("empty-state"))

    payload = original.model_dump(mode="json")
    restored = PRDArtifact.model_validate_json(json.dumps(payload))

    assert restored == original
    assert [item.marker for item in restored.attachments] == ["login-error", "empty-state"]
    assert restored.attachments[0].sha256 == fixtures.digest(fixtures.PNG)


def test_the_rendered_artifact_grows_by_hundreds_of_bytes_and_not_megabytes() -> None:
    """The guard against somebody adding a `data` field to this schema later.

    A base64 blob for three 5 MiB images would be twenty megabytes of prompt text. The bound
    below is generous for metadata and impossible for content.
    """
    empty = artifact_json(prd())
    with_three = artifact_json(
        prd(attachment("login-error"), attachment("empty-state"), attachment("checkout"))
    )

    growth = len(with_three) - len(empty)
    assert growth > 0, "three recorded attachments must be visible in the rendered artifact"
    # Roughly 250 bytes each of identifiers, names, a caption and a 64-character hash.
    assert growth < 2_000, f"three attachments rendered {growth} bytes; something carries content"
    # And no field that could hold bytes exists at all.
    rendered = with_three
    for forbidden in ('"data"', '"content"', '"bytes"', "base64"):
        assert forbidden not in rendered, f"{forbidden} must never appear in a PRD artifact"


def test_two_attachments_with_one_marker_are_refused() -> None:
    """A marker is how prose addresses an image; two answering to it point at neither."""
    with pytest.raises(ValidationError, match="attachment marker"):
        prd(attachment("same"), attachment("same", attachment_id="attachment-other"))


def test_a_type_the_platform_does_not_serve_cannot_be_recorded() -> None:
    """The artifact's `media_type` is closed, so a purged row still names something servable."""
    with pytest.raises(ValidationError):
        attachment("svg", media_type="image/svg+xml")
    with pytest.raises(ValidationError):
        attachment("gif", media_type="image/gif")


def test_a_zero_byte_or_unnamed_attachment_cannot_be_recorded() -> None:
    """Every field but the caption is required to say something."""
    with pytest.raises(ValidationError):
        attachment("empty", byte_size=0)
    with pytest.raises(ValidationError):
        attachment("nameless", filename="")
    with pytest.raises(ValidationError):
        attachment("hashless", sha256="")


def test_an_empty_caption_is_the_default_and_is_accepted() -> None:
    """A screenshot with no caption is an ordinary screenshot, not an incomplete record."""
    recorded = attachment("plain", caption="")
    assert recorded.caption == ""


def test_a_prd_with_no_attachments_reads_as_an_empty_list_and_renders_no_key() -> None:
    """The unchanged case: the field is there, it is empty, and it is not serialized.

    An empty list in Python and an absent key on the wire, which is the whole point of
    `without_empty_prd_evidence`: `{{ prd }}` is this artifact serialized, and
    `"attachments": []` in front of a model is an instruction to look at pictures on a
    feature that has none. See `test_design_source.py` for the byte-identity assertion that
    covers both 89- items' evidence fields together.
    """
    document = prd()
    assert document.attachments == []
    assert '"attachments"' not in artifact_json(document)
    # And a PRD that did attach something says so, because then it is part of the request.
    assert '"attachments"' in artifact_json(prd(attachment("login-error")))
