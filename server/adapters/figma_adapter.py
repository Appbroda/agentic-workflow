"""The two Figma REST calls this platform makes, classified the platform's own way.

Two endpoints and no SDK, in `slack_adapter.py`'s style: `httpx` is already pinned and in the
lock, the client is created per call, the import is deferred to the first call so a run that
resolves nothing never initializes network-capable code, and one timeout constant bounds
everything.

The closed set is deliberate and it is exactly two:

* ``GET /v1/files/:key/nodes?ids=...`` -- the node subtree, which is the whole resolution. A
  citation naming no frame is a whole-file citation, and it is answered through this same
  endpoint by asking for the document root ``0:0`` at depth 2: a file's canvases and their
  top-level frames. That keeps the set at two rather than adding ``GET /v1/files/:key``.
* ``GET /v1/images/:key?ids=...&format=png&version=...`` -- the preview render, for the
  console and for nothing else. Never an input to an agent: the coding and judging calls are
  text in, text out, and the design reaches them as text or not at all.

Nothing else. In particular **not** the variables API: measured against the live API on
2026-09-07, ``GET /v1/files/:key/variables/local`` answers
``403 Invalid scope(s) ... requires the file_variables:read scope`` for a read-only personal
access token, and parts of it are Enterprise-only. So this platform reads the style *names*
already present on the nodes, and the snapshot records that that is what it did.

Every refusal is reclassified before it leaves this module, the way `slack_adapter` and
`github_adapter` do it: the platform's own error code, the endpoint, the HTTP status, and a
cause stated as a possibility. No stack trace, no provider message quoted, and never the
token -- which does not leave the server at all.

Its one reader is `services.design_resolution`, which is where the wire shape stops: that
module turns a returned subtree into typed `DesignNode` records, and nothing past it deals in
Figma's own JSON.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

# Figma's API root. Shared with `services.credential_verification`, which asks `GET /v1/me`
# about a stored token: one spelling of where Figma is.
FIGMA_API_BASE = "https://api.figma.com"

# One bound on every call. Larger than the Slack adapter's ten seconds because a file read is
# a genuinely large response -- the design files this item was measured against answer in
# hundreds of kilobytes -- and a timeout that fires on an ordinary read is a fault that is not
# one.
_REQUEST_TIMEOUT_SECONDS = 60.0

# The document root every Figma file has. Asking for it at depth 2 returns the file's canvases
# and their top-level children, which is what a whole-file citation resolves to.
_DOCUMENT_ROOT_NODE_ID = "0:0"
_TOP_LEVEL_DEPTH = 2

# The node types a whole-file citation is worth resolving to. A canvas's other children --
# loose vectors, stray rectangles, a comment pin -- are not frames somebody designed a screen
# in, and offering them as "the design" would pad the snapshot with noise.
_TOP_LEVEL_NODE_TYPES = frozenset({"FRAME", "COMPONENT", "COMPONENT_SET", "SECTION"})

# The HTTP statuses that are weather rather than an answer, on `_GITHUB_WEATHER_HTTP_STATUSES`'
# reasoning exactly. A 429 is the provider asking for a later retry -- and it is not
# hypothetical here: the live API rate-limited this item's own measurement runs. A 408 is it
# timing out. Every other 4xx is Figma answering "no", identically, every time.
_WEATHER_HTTP_STATUSES = frozenset({408, 429})

# A `Retry-After` longer than a week is not a wait anybody is going to sit through, and a
# number that large is likelier a misparse than an instruction. Discarded rather than
# recorded: the sentence a person reads must not carry a deadline this module invented.
_MAX_RETRY_AFTER_SECONDS = 7 * 24 * 60 * 60

# Where Figma's images endpoint actually mints render URLs: its own domain, and the S3 bucket
# the presigned URLs live on (`figma-alpha-api.s3.<region>.amazonaws.com`, observed live).
# `fetch_rendered_bytes` GETs whatever URL the response body named, so anything outside this
# set -- an internal host, a metadata service, a plain-HTTP URL -- is refused before a single
# byte is requested.
_RENDER_URL_HOST_SUFFIXES = (".figma.com", ".amazonaws.com")

# What a PNG's first eight bytes are, per the PNG specification. The preview endpoint serves
# the fetched body onward as `image/png`, so a body that is not one is refused here.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class FigmaFailureMode(StrEnum):
    """What one refused Figma call was, in the platform's own vocabulary."""

    # Figma down, a 5xx, a timeout, a dropped connection, or a 429: ask again later.
    TRANSPORT = "transport"
    # The token was refused, or does not hold a scope this call needs. A person replaces it.
    CREDENTIAL_REFUSED = "credential_refused"
    # The file does not exist, or this token cannot read it. The person's URL to fix.
    FILE_UNREACHABLE = "file_unreachable"
    # Figma answered and said no about the request itself -- a malformed node id, a bad
    # parameter. Deterministic: it says the same thing every time.
    REQUEST_REFUSED = "request_refused"


@dataclass(frozen=True, slots=True)
class FigmaNodeSubtree:
    """One cited node as Figma returned it, with the maps its names are resolved through.

    `document` is Figma's own node tree and is the one place in this platform that shape
    exists. `services.design_resolution` is its only reader; everything past that reads
    `DesignNode` records.
    """

    node_id: str
    document: Mapping[str, Any]
    # Style id to `{name, styleType}`. This is where a hex code becomes `color/surface/raised`,
    # which is the difference between telling an engineer what to hardcode and telling it what
    # the repository already has.
    styles: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, Any] = field(default_factory=dict)
    component_sets: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FigmaNodesResult:
    """What one nodes call answered: the file it read, and which ids it could not find.

    `absent_node_ids` is a fact Figma reports by answering `200` with `nodes[id] = null`, which
    is verified live: a node id that is well-formed but not in the file is not an error, it is
    an answer. Carried separately so the resolution can record it rather than mistake it for a
    transport failure.
    """

    file_key: str
    file_name: str
    # The file version Figma reported, which is what a preview render is later pinned to.
    file_version: str
    subtrees: tuple[FigmaNodeSubtree, ...]
    absent_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FigmaTopLevelFrame:
    """One top-level frame of a file, as a whole-file citation resolves to it."""

    node_id: str
    name: str
    node_type: str
    canvas_name: str


@dataclass(frozen=True, slots=True)
class FigmaTopLevelResult:
    """A file's top-level frames, and how many were past the bound the caller asked for."""

    file_key: str
    file_name: str
    file_version: str
    frames: tuple[FigmaTopLevelFrame, ...]
    # Reported rather than dropped: an unreported cap reads as "the whole file was considered".
    frames_beyond_bound: int


class FigmaClientError(RuntimeError):
    """One classified Figma refusal, carrying only platform-owned diagnostics.

    ``error_code`` is the platform's classification (`figma_transport_503`,
    `figma_file_unreachable_404`), never a provider message. ``provider_status`` is the number
    the retryability predicate reads, and it reads the number and never the message.

    ``retry_after_seconds`` is the one provider-authored number worth keeping: a `Retry-After`
    is an instruction, not prose, and it is the difference between weather that clears in a
    minute and a lockout measured in days. AB-Feature-228 spent all nine of its allowed calls
    in fifty-one seconds against a 429 whose `Retry-After` was 261,044 -- about seventy-two
    hours -- and the record it left said only that a provider had not answered.
    """

    def __init__(
        self,
        message: str,
        *,
        mode: FigmaFailureMode,
        error_code: str,
        endpoint: str,
        provider_status: int | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        """Carry the classification and nothing the provider authored."""
        super().__init__(message)
        self.mode = mode
        self.error_code = error_code
        self.endpoint = endpoint
        self.provider_status = provider_status
        self.retry_after_seconds = retry_after_seconds

    @property
    def operation_outcome(self) -> str:
        """The token `ExternalOperationExecutor` puts in the journal row's `error_code`.

        The seam already exists for callers that can tell their own failures apart, and this
        is one: without it every design fetch that ever failed shared `fetch_design_reference_
        failed`, so "why did the design fail" could not be answered from the journal at all --
        the nine rows AB-Feature-228 left were distinguishable from a revoked token only by
        going and asking Figma directly. Platform-owned by construction; `error_code` is this
        module's own vocabulary.
        """
        return self.error_code

    @property
    def retryable(self) -> bool:
        """Whether asking again could answer differently.

        A 4xx that is not weather is Figma deciding, and it decides the same way every time --
        the rule `_github_rest_fault` applies to the same shape of question. The workflow's own
        `is_transient_provider_fault` is the single reader of this.
        """
        if self.mode is FigmaFailureMode.TRANSPORT:
            return True
        status = self.provider_status
        return status is not None and status in _WEATHER_HTTP_STATUSES

    @property
    def is_the_persons_to_fix(self) -> bool:
        """Whether the remedy is a person changing the citation or replacing the token.

        The other half of the failure policy: the provider answered, so no amount of retrying
        helps, and the feature stops before planning with something actionable to say.
        """
        return self.mode in (
            FigmaFailureMode.CREDENTIAL_REFUSED,
            FigmaFailureMode.FILE_UNREACHABLE,
            FigmaFailureMode.REQUEST_REFUSED,
        )


class FigmaDesignClient:
    """The calls the resolver and the preview endpoint make, one signature each.

    A `Protocol` in spirit and a plain base class in practice, for `SlackClient`'s reason: the
    fake in the suite and the real adapter are checked against the same signatures by mypy
    without a structural cast at every injection site.
    """

    async def fetch_nodes(self, file_key: str, node_ids: Sequence[str]) -> FigmaNodesResult:
        """Read the cited nodes' subtrees, and report which ids the file does not define."""
        raise NotImplementedError

    async def fetch_top_level_frames(self, file_key: str, *, limit: int) -> FigmaTopLevelResult:
        """List a file's top-level frames, bounded, reporting how many were past the bound."""
        raise NotImplementedError

    async def render_preview(self, file_key: str, node_id: str, *, version: str) -> str:
        """Return a short-lived render URL for one node, pinned to one file version."""
        raise NotImplementedError

    async def fetch_rendered_bytes(self, url: str) -> bytes:
        """Retrieve the PNG a render URL points at."""
        raise NotImplementedError


class HttpxFigmaDesignClient(FigmaDesignClient):
    """The real adapter. Construction takes a token and opens nothing."""

    def __init__(self, token: str) -> None:
        """Hold the token for the calls below. It never leaves this process."""
        self._token = token

    async def fetch_nodes(self, file_key: str, node_ids: Sequence[str]) -> FigmaNodesResult:
        """Read one or more cited nodes in a single call, as the endpoint is designed for."""
        requested = list(dict.fromkeys(node_ids))
        body = await self._get(
            f"/v1/files/{file_key}/nodes",
            {"ids": ",".join(requested)},
            endpoint="files.nodes",
        )
        entries = body.get("nodes")
        nodes = entries if isinstance(entries, dict) else {}
        subtrees: list[FigmaNodeSubtree] = []
        absent: list[str] = []
        for node_id in requested:
            entry = nodes.get(node_id)
            if not isinstance(entry, dict) or not isinstance(entry.get("document"), dict):
                # `200` with `nodes[id] = null` is Figma answering "the file does not define
                # that", which is a fact about the citation and not a failure of the call.
                absent.append(node_id)
                continue
            subtrees.append(
                FigmaNodeSubtree(
                    node_id=node_id,
                    document=entry["document"],
                    styles=_mapping(entry.get("styles")),
                    components=_mapping(entry.get("components")),
                    component_sets=_mapping(entry.get("componentSets")),
                )
            )
        return FigmaNodesResult(
            file_key=file_key,
            file_name=str(body.get("name") or ""),
            file_version=str(body.get("version") or ""),
            subtrees=tuple(subtrees),
            absent_node_ids=tuple(absent),
        )

    async def fetch_top_level_frames(self, file_key: str, *, limit: int) -> FigmaTopLevelResult:
        """Answer a whole-file citation through the nodes endpoint, not a third one.

        The document root at depth 2 is a file's canvases and their immediate children, which
        is exactly "the file's top-level frames" and costs no new endpoint.
        """
        body = await self._get(
            f"/v1/files/{file_key}/nodes",
            {"ids": _DOCUMENT_ROOT_NODE_ID, "depth": str(_TOP_LEVEL_DEPTH)},
            endpoint="files.nodes.root",
        )
        entries = body.get("nodes")
        root = (entries or {}).get(_DOCUMENT_ROOT_NODE_ID) if isinstance(entries, dict) else None
        document = root.get("document") if isinstance(root, dict) else None
        found: list[FigmaTopLevelFrame] = []
        for canvas in (document or {}).get("children") or []:
            if not isinstance(canvas, dict):
                continue
            canvas_name = str(canvas.get("name") or "")
            for child in canvas.get("children") or []:
                if not isinstance(child, dict):
                    continue
                if str(child.get("type") or "") not in _TOP_LEVEL_NODE_TYPES:
                    continue
                found.append(
                    FigmaTopLevelFrame(
                        node_id=str(child.get("id") or ""),
                        name=str(child.get("name") or ""),
                        node_type=str(child.get("type") or ""),
                        canvas_name=canvas_name,
                    )
                )
        return FigmaTopLevelResult(
            file_key=file_key,
            file_name=str(body.get("name") or ""),
            file_version=str(body.get("version") or ""),
            frames=tuple(found[:limit]),
            frames_beyond_bound=max(0, len(found) - limit),
        )

    async def render_preview(self, file_key: str, node_id: str, *, version: str) -> str:
        """Render one node as a PNG at the file version the snapshot recorded.

        ``version`` is not optional and not defaulted. Figma's images endpoint renders the
        *current* file state unless told otherwise, so an unpinned re-render would show a
        design that has changed under a snapshot that must not change -- which would make the
        console quietly disagree with the text every judge was given.
        """
        body = await self._get(
            f"/v1/images/{file_key}",
            {"ids": node_id, "format": "png", "version": version},
            endpoint="images",
        )
        images = body.get("images")
        url = (images or {}).get(node_id) if isinstance(images, dict) else None
        if not isinstance(url, str) or not url:
            raise _refusal(
                "images",
                mode=FigmaFailureMode.REQUEST_REFUSED,
                error_code="figma_render_returned_no_url",
            )
        return url

    async def fetch_rendered_bytes(self, url: str) -> bytes:
        """Retrieve the PNG the images endpoint just minted, and nothing else.

        Not a third API endpoint: it is the collection of the asset the second one produced,
        from the short-lived storage URL that call returned. It exists because the console
        fetches a preview through this platform with its own bearer token like every other
        read -- an `<img src>` cannot carry one -- so the bytes come back through the server
        rather than the browser being handed a URL to Figma's storage.

        The URL is used once and returned to nobody. Nothing stores it: Safety rule 4 forbids
        re-minting the frozen snapshot to refresh a URL, and a stored one would go stale inside
        a record that must not change.

        The URL is validated before anything is fetched. It came out of a provider response
        body, and this method GETs whatever it is handed -- which is a server-side request
        forgery seam the moment that body says `http://169.254.169.254/`. The allowed hosts
        are the two Figma actually mints render URLs on: its own domain and its S3 buckets
        (`figma-alpha-api.s3.<region>.amazonaws.com`). And the answer must actually be a PNG,
        because the endpoint serves it onward as `image/png` to a browser.
        """
        _require_figma_render_url(url)
        httpx = importlib.import_module("httpx")
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.get(url)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _refusal(
                "images.render",
                mode=FigmaFailureMode.TRANSPORT,
                error_code=f"figma_transport_{type(error).__name__.lower()}",
            ) from error
        status = int(response.status_code)
        if status != 200:  # noqa: PLR2004 - the one success status for a stored asset
            # A render URL expires. An expired one is weather rather than an answer about the
            # design: the caller asks for a fresh render and gets one.
            raise _refusal(
                "images.render",
                mode=FigmaFailureMode.TRANSPORT,
                error_code=f"figma_render_unavailable_{status}",
                provider_status=status,
            )
        rendered = bytes(response.content)
        if not rendered.startswith(_PNG_MAGIC):
            # A 200 that is not a PNG is an error page or a bucket answer, and passing it on
            # as `image/png` would hand a browser bytes nobody classified.
            raise _refusal(
                "images.render",
                mode=FigmaFailureMode.REQUEST_REFUSED,
                error_code="figma_render_not_a_png",
                provider_status=status,
            )
        return rendered

    async def _get(self, path: str, params: dict[str, str], *, endpoint: str) -> dict[str, Any]:
        """Make one GET and reclassify every way it can refuse."""
        httpx = importlib.import_module("httpx")
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    f"{FIGMA_API_BASE}{path}",
                    params=params,
                    headers={"X-Figma-Token": self._token},
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Timeouts, DNS failures, resets. The type name is platform-safe; the message may
            # carry a URL or a proxy detail, so only the type is kept.
            raise _refusal(
                endpoint,
                mode=FigmaFailureMode.TRANSPORT,
                error_code=f"figma_transport_{type(error).__name__.lower()}",
            ) from error
        return _classify(endpoint, response)


def _classify(endpoint: str, response: Any) -> dict[str, Any]:
    """Turn one HTTP response into a body or a classified refusal, never a raw error."""
    status = int(response.status_code)
    if status in _WEATHER_HTTP_STATUSES or status >= 500:  # noqa: PLR2004 - the HTTP 5xx range
        # The only branch that reads `Retry-After`, because it is the only one a retry is ever
        # made against. A 401 or a 404 that carried the header would be describing a wait that
        # changes nothing.
        raise _refusal(
            endpoint,
            mode=FigmaFailureMode.TRANSPORT,
            error_code=f"figma_transport_{status}",
            provider_status=status,
            retry_after_seconds=_retry_after_seconds(response),
        )
    if status in (401, 403):
        # Verified live: a missing or invalid token answers 401, and a scope the token does not
        # hold answers 403 with Figma's own list of the scopes it does hold. Both mean a person
        # replaces or re-scopes the credential, and neither changes on a retry.
        raise _refusal(
            endpoint,
            mode=FigmaFailureMode.CREDENTIAL_REFUSED,
            error_code=f"figma_credential_refused_{status}",
            provider_status=status,
        )
    if status == 404:  # noqa: PLR2004 - the HTTP status, not a magic bound
        # Verified live: a file this token cannot read is indistinguishable from one that does
        # not exist, and Figma answers 404 for both. Either way the URL is the person's to fix.
        raise _refusal(
            endpoint,
            mode=FigmaFailureMode.FILE_UNREACHABLE,
            error_code="figma_file_unreachable_404",
            provider_status=status,
        )
    if status >= 400:  # noqa: PLR2004 - the HTTP 4xx range, not a magic bound
        raise _refusal(
            endpoint,
            mode=FigmaFailureMode.REQUEST_REFUSED,
            error_code=f"figma_request_refused_{status}",
            provider_status=status,
        )
    try:
        body = response.json()
    except ValueError as error:
        raise _refusal(
            endpoint,
            mode=FigmaFailureMode.TRANSPORT,
            error_code="figma_transport_unreadable_response",
            provider_status=status,
        ) from error
    if not isinstance(body, dict):
        raise _refusal(
            endpoint,
            mode=FigmaFailureMode.TRANSPORT,
            error_code="figma_transport_unreadable_response",
            provider_status=status,
        )
    return body


def _mapping(value: Any) -> Mapping[str, Any]:
    """Read one of the file-level maps, tolerating its absence."""
    return value if isinstance(value, dict) else {}


def _require_figma_render_url(url: str) -> None:
    """Refuse any render URL that is not HTTPS on a host Figma actually mints them on.

    Raised as the adapter's own refusal type -- deterministic, never retryable -- and raised
    *before* any fetch: the point is that no request is made at all, not that a made request
    is discarded.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        parsed = None
    host = (parsed.hostname or "").lower() if parsed is not None else ""
    acceptable = (
        parsed is not None
        and parsed.scheme == "https"
        and (host == "figma.com" or host.endswith(_RENDER_URL_HOST_SUFFIXES))
    )
    if not acceptable:
        # The error code carries no part of the URL: a refused one is exactly the kind of
        # thing that must not be quoted into durable state.
        raise _refusal(
            "images.render",
            mode=FigmaFailureMode.REQUEST_REFUSED,
            error_code="figma_render_url_refused",
        )


def _retry_after_seconds(response: Any) -> int | None:
    """Read `Retry-After` as a number of seconds, or report that it said nothing usable.

    Only the delta-seconds form is honoured. The HTTP-date form is equally legal and would
    need this module to hold a clock and a timezone to mean anything, and a wrong deadline is
    worse here than no deadline: it is read by the sentence that tells a person when to come
    back. Figma sends delta-seconds -- `retry-after: 261044`, observed live on 2026-09-12.

    Defensive about the response shape because the suite's fake clients are plain objects,
    and about the value because it is the provider's: a negative, a date, or a gigantic number
    is discarded rather than propagated into a record that must stay true.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    if not isinstance(raw, str) or not raw.strip().isdigit():
        return None
    seconds = int(raw.strip())
    return seconds if 0 < seconds <= _MAX_RETRY_AFTER_SECONDS else None


def _refusal(
    endpoint: str,
    *,
    mode: FigmaFailureMode,
    error_code: str,
    provider_status: int | None = None,
    retry_after_seconds: int | None = None,
) -> FigmaClientError:
    """Build one classified refusal whose message carries nothing provider-authored."""
    return FigmaClientError(
        f"figma call refused ({error_code}) on {endpoint}",
        mode=mode,
        error_code=error_code,
        endpoint=endpoint,
        provider_status=provider_status,
        retry_after_seconds=retry_after_seconds,
    )


__all__ = [
    "FIGMA_API_BASE",
    "FigmaClientError",
    "FigmaDesignClient",
    "FigmaFailureMode",
    "FigmaNodeSubtree",
    "FigmaNodesResult",
    "FigmaTopLevelFrame",
    "FigmaTopLevelResult",
    "HttpxFigmaDesignClient",
]
