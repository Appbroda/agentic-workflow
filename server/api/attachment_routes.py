"""Uploading an image, fetching it back, and the one refusal that has its own sentence.

Three routes and no more. A submission's evidence arrives here as bytes, is sniffed rather
than believed, and is read back only by somebody the platform already lets read the PRD it
belongs to. No route here returns a URL a browser could follow without a token, and nothing
re-encodes: the bytes are served exactly as stored, so nothing has to be trusted to re-encode
safely.

The alternative -- a base64 string field on `PRDSubmission` -- was rejected for three
independent reasons: it inflates the payload by a third, it passes a multi-megabyte string
through `APIModel`'s `str_strip_whitespace`, and it puts the blob inside every request log
line that ever prints a body. Multipart costs one dependency (`python-multipart`) and none of
that.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from api.auth import PlatformAuthenticator, current_actor, requires
from api.identity import Actor, Permission
from api.schemas import APIModel
from storage.attachment_store import (
    ACCEPTED_MEDIA_TYPES,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_BYTES_PER_FEATURE,
    MAX_ATTACHMENTS_PER_FEATURE,
    Attachment,
    AttachmentError,
    AttachmentStore,
    sniff_image_media_type,
)

# How the endpoint reads an upload: in bounded pieces, stopping the moment the total exceeds
# the cap. A single `await file.read()` would have to buffer whatever was actually sent before
# it could measure it, which makes the cap a suggestion.
_READ_CHUNK_BYTES = 64 * 1024

# What one multipart request carrying one capped image may weigh, headers and boundary
# included. This is the *pre-parse* half of the size rule and it is enforced in the
# application's body-size middleware, not here: FastAPI resolves a `File()` parameter by
# parsing the whole body before the handler is entered, so by the time this module sees an
# `UploadFile` the bytes have already been spooled. Content-Length is the only thing anybody
# can refuse before that happens.
#
# The margin is generous on purpose -- a boundary, a disposition header and a filename are
# small and variable -- because it is not the cap. The cap is `MAX_ATTACHMENT_BYTES`, applied
# to the bytes that actually arrived.
ATTACHMENT_REQUEST_BYTES = MAX_ATTACHMENT_BYTES + 64 * 1024

# The path this allowance applies to, matched by the middleware. Named here so the number and
# the route it belongs to are declared in one place.
ATTACHMENT_UPLOAD_PATH = "/attachments"

# What the browser is told to do with these bytes, and what it is told not to do. `nosniff`
# because the whole point of storing a sniffed type is that the response's own declaration is
# the truth; the sandboxed, source-less CSP because this is user-supplied content served from
# the platform's own origin.
_SERVING_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
    "Cache-Control": "private, max-age=3600",
}


class AttachmentResponse(APIModel):
    """What an upload answers with: identifiers and facts, never a URL and never the bytes."""

    attachment_id: str
    filename: str
    media_type: str
    byte_size: int
    sha256: str


class AttachmentLimitsResponse(APIModel):
    """The caps, published so the submission form states the same numbers the server enforces."""

    max_attachment_bytes: int
    max_attachments_per_feature: int
    max_attachment_bytes_per_feature: int
    accepted_media_types: list[str]


def size_limit_message() -> str:
    """The per-file refusal, quoting the constant rather than restating the number."""
    return (
        f"an image must be at most {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MiB; "
        f"at most {MAX_ATTACHMENTS_PER_FEATURE} images and "
        f"{MAX_ATTACHMENT_BYTES_PER_FEATURE // (1024 * 1024)} MiB in total per feature"
    )


def unsupported_type_message(sniffed: str | None) -> str:
    """The type refusal, with SVG's own sentence.

    "Unsupported type" sends somebody away to try harder at the same thing, which for an SVG
    means exporting another SVG. The message names the way out instead.
    """
    if sniffed == "image/svg+xml":
        return (
            "SVG is markup, not an image; export a PNG. It carries script and remote-fetch "
            "reach and this platform serves attachments back to a browser, so it is refused "
            "by its contents rather than by its name."
        )
    accepted = ", ".join(ACCEPTED_MEDIA_TYPES)
    if sniffed is None:
        return f"this file is not an image the platform accepts; accepted types are {accepted}"
    return f"unsupported image type {sniffed}; accepted types are {accepted}"


def attachment_store(request: Request) -> AttachmentStore:
    """Return the deployment's attachment store, or refuse honestly."""
    store = getattr(request.app.state, "attachments", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not persist attachments.",
        )
    return store  # type: ignore[no-any-return]


def create_attachment_router(*, authenticator: PlatformAuthenticator) -> APIRouter:
    """Create `/attachments/*` behind the same authentication as everything else."""
    router = APIRouter(
        prefix="/attachments",
        tags=["attachments"],
        dependencies=[Depends(authenticator)],
    )

    @router.get("/limits", response_model=AttachmentLimitsResponse)
    async def attachment_limits() -> AttachmentLimitsResponse:
        """Publish the caps, so the interface's helper text cannot drift from the server's."""
        return AttachmentLimitsResponse(
            max_attachment_bytes=MAX_ATTACHMENT_BYTES,
            max_attachments_per_feature=MAX_ATTACHMENTS_PER_FEATURE,
            max_attachment_bytes_per_feature=MAX_ATTACHMENT_BYTES_PER_FEATURE,
            accepted_media_types=list(ACCEPTED_MEDIA_TYPES),
        )

    @router.post(
        "",
        response_model=AttachmentResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(requires(Permission.FEATURE_CREATE))],
    )
    async def upload_attachment(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
        file: Annotated[UploadFile, File()],
    ) -> AttachmentResponse:
        """Accept one image. Guarded by the permission that creates features, because that is
        what an attachment is for: an upload is the first half of a submission.

        The validation order is the point of this handler:

        1. the declared length, against the cap, before the bytes are read -- which for a
           multipart body means Content-Length, refused by the body-size middleware using
           `ATTACHMENT_REQUEST_BYTES`, because a `File()` parameter is parsed before this
           function is entered;
        2. the declared length again as the parser measured it, which is `file.size`;
        3. the bytes that were actually read, counted as they are read, because a declared
           length is a claim;
        4. the magic bytes, which decide what this is;
        5. the declared *type* against those bytes -- a mismatch is a lie, not a preference.

        Step 5 only fires on a declaration that names an image. A client that sends
        `application/octet-stream`, or nothing, has made no claim about which image this is
        and is not caught lying; a client that says `image/jpeg` over PNG bytes has.
        """
        store = attachment_store(request)
        _refuse_declared_size(file)
        content = await _read_within_cap(file)
        sniffed = sniff_image_media_type(content)
        if sniffed not in ACCEPTED_MEDIA_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=unsupported_type_message(sniffed),
            )
        declared = (file.content_type or "").split(";")[0].strip().lower()
        if declared.startswith("image/") and declared != sniffed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"this file was sent as {declared} and its contents are {sniffed}; "
                    "upload the file you meant to"
                ),
            )
        try:
            record = await store.create(
                owner_id=actor.actor_id,
                filename=file.filename or "image",
                media_type=sniffed,
                content=content,
            )
        except AttachmentError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        return _response(record)

    @router.get("/{attachment_id}")
    async def get_attachment(
        attachment_id: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> Response:
        """Serve the bytes to somebody entitled to them, and 404 to everybody else.

        `404` rather than `403`, deliberately: a `403` on an id somebody guessed confirms the
        id exists, which turns this route into an existence oracle for other people's
        evidence. The two answers are the same answer here.

        `410` for a purged attachment, which is a different fact and worth saying: it existed,
        the feature it belonged to was retired, and the artifact still records its hash. "Gone"
        and "never was" are not the same answer to somebody following a reference.
        """
        store = attachment_store(request)
        metadata = await store.get_metadata(attachment_id)
        if metadata is None or not _may_read(actor, metadata):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"attachment not found: {attachment_id}",
            )
        if not metadata.content_present:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=(
                    "this attachment's content was purged when its feature was retired; "
                    "the submitted PRD still records its name, size and hash"
                ),
            )
        content = await store.get_content(attachment_id)
        if content is None:
            # Purged between the two reads. The same answer as above rather than a 500: a
            # narrow race is not a platform fault, and the caller's next read agrees.
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="this attachment's content was purged while it was being read",
            )
        # Served as stored, with the sniffed type. Nothing re-encodes, so nothing has to be
        # trusted to re-encode safely.
        return Response(
            content=content,
            media_type=metadata.media_type,
            headers=dict(_SERVING_HEADERS),
        )

    @router.delete("/{attachment_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_attachment(
        attachment_id: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> None:
        """Forget one unbound upload of the caller's own.

        A bound attachment is part of a submitted record and is not deletable here: it is
        removed by retiring the feature, which purges the content and keeps the row so the
        artifact's reference still resolves. The same 404 covers "not yours" and "bound",
        for the reason the fetch does.
        """
        store = attachment_store(request)
        if not await store.delete_unbound(attachment_id, owner_id=actor.actor_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"attachment not found: {attachment_id}",
            )

    return router


def _may_read(actor: Actor, metadata: Attachment) -> bool:
    """Decide who may read one attachment's bytes.

    While unbound it is the uploader's alone -- nobody else has any way to know it exists,
    and it is not part of any record yet. Once bound it belongs to a feature, and the
    permission that already shows somebody that feature's PRD is the one that shows them the
    images the PRD points at: a narrower rule would hide a submission's evidence from the
    people reviewing the submission.
    """
    if metadata.feature_id is None:
        return metadata.owner_id == actor.actor_id
    return actor.may(Permission.FEATURE_READ)


def _refuse_declared_size(file: UploadFile) -> None:
    """Refuse an oversized upload on the length the part declares, before reading it.

    Second of the three size checks. The first is Content-Length in the middleware, which is
    the only one that can happen before any bytes are accepted; this one is the parser's
    count for this part; the third counts the bytes as they are read.
    """
    declared = file.size
    if declared is not None and declared > MAX_ATTACHMENT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=size_limit_message()
        )


async def _read_within_cap(file: UploadFile) -> bytes:
    """Read the upload, refusing the moment it exceeds the cap.

    The second half of the size rule. A declared length is a claim, so the bytes that
    actually arrive are counted as they arrive and the read stops at the first chunk that
    takes the total over -- rather than buffering an arbitrary amount and measuring it
    afterwards, which is not a cap.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=size_limit_message()
            )
        chunks.append(chunk)
    content = b"".join(chunks)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="this upload carried no bytes",
        )
    return content


def _response(record: Attachment) -> AttachmentResponse:
    """Project one stored attachment onto the upload response."""
    return AttachmentResponse(
        attachment_id=record.attachment_id,
        filename=record.filename,
        media_type=record.media_type,
        byte_size=record.byte_size,
        sha256=record.sha256,
    )


__all__ = [
    "AttachmentLimitsResponse",
    "AttachmentResponse",
    "attachment_store",
    "create_attachment_router",
    "size_limit_message",
    "unsupported_type_message",
]
