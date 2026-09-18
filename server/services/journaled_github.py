"""Cancellation-aware, idempotent GitHub mutations backed by the external-operation journal."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from adapters.github_adapter import GitHubService, PullRequestDetails, select_pull_request
from services.cancellation import CancellationRequested, await_cancellable
from services.external_operations import (
    EffectAbsent,
    EffectAmbiguous,
    EffectUnproven,
    ExternalOperationExecutor,
    ReconciliationOutcome,
    UnknownExternalOperation,
)
from state.external_operations import ExternalOperation, ExternalOperationType
from storage.external_operation_store import OperationResult
from workflows.feature_workflow import is_transient_provider_fault

# A comment is not a set: unlike labels and reviewers, replaying an unconfirmed one can
# double-post it. The budget matches the labels/reviewers precedent, and it is safe only
# because `add_comment`'s action and reconcile both prove presence or absence first.
_COMMENT_ATTEMPTS = 3
# The driver's wait between admitted faults, doubling per fault. Deliberately its own small
# constant rather than the workflow loops': this backoff runs inside a publication step a
# person is watching, and two waits bound it to fifteen seconds.
_COMMENT_FAULT_BACKOFF_SECONDS = 5.0


class JournaledGitHubService:
    """Wrap synchronous PyGithub/mock services without allowing duplicate PR-side mutations."""

    def __init__(
        self,
        service: GitHubService,
        *,
        operation_executor: ExternalOperationExecutor,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._service = service
        self._operations = operation_executor
        if timeout_seconds <= 0:
            msg = "timeout_seconds must be positive"
            raise ValueError(msg)
        self._timeout_seconds = timeout_seconds

    async def create_pull_request(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        draft: bool = False,
        expected_head_sha: str | None = None,
    ) -> PullRequestDetails:
        """Create or reuse a PR after durable intent and cancellation gates are clear."""
        body_fingerprint = hashlib.sha256(body.encode("utf-8")).hexdigest()

        async def action() -> tuple[PullRequestDetails, OperationResult]:
            existing = await self._find_existing(repository, source_branch, target_branch, title)
            if existing is not None:
                return existing, _pr_result(existing, reused=True)
            try:
                create_arguments: dict[str, Any] = {
                    "title": title,
                    "body": body,
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                    "draft": draft,
                }
                # Older provider doubles and persisted feature operations predate the legacy
                # completion guard. Omit the optional argument unless this caller is binding
                # the PR to a reviewed commit, preserving their exact operation semantics.
                if expected_head_sha is not None:
                    create_arguments["expected_head_sha"] = expected_head_sha
                details = await asyncio.wait_for(
                    await_cancellable(
                        asyncio.to_thread(
                            self._service.create_pull_request,
                            repository,
                            **create_arguments,
                        ),
                        self._operations.cancellation_token,
                    ),
                    timeout=self._timeout_seconds,
                )
            except (CancellationRequested, TimeoutError):
                existing = await self._find_existing(
                    repository, source_branch, target_branch, title
                )
                if existing is not None:
                    return existing, _pr_result(existing, recovered_after_cancellation=True)
                raise UnknownExternalOperation(
                    "pull-request creation may have completed after cancellation"
                ) from None
            return details, _pr_result(details)

        async def reconcile(_operation: ExternalOperation) -> ReconciliationOutcome:
            candidates = await self._candidate_pull_requests(
                repository, source_branch, target_branch
            )
            if candidates is None:
                return EffectUnproven(
                    method="pull_request_lookup",
                    detail="the provider could not be asked whether the pull request exists",
                )
            if not candidates:
                return EffectAbsent(
                    method="pull_request_lookup",
                    detail="no open pull request exists for this head and base branch",
                    observations={"head_branch": source_branch, "base_branch": target_branch},
                )
            existing = select_pull_request(
                candidates, title=title, expected_head_sha=expected_head_sha
            )
            if existing is None:
                return EffectAmbiguous(
                    method="pull_request_lookup",
                    detail=(
                        "more than one open pull request shares this head and base branch and "
                        "none can be identified as the intended one"
                    ),
                    observations={
                        "head_branch": source_branch,
                        "base_branch": target_branch,
                        "candidate_pull_request_numbers": sorted(
                            item.number for item in candidates
                        ),
                    },
                )
            return _pr_result(
                existing, recovered=True, recovery_method="existing_pull_request_adopted"
            )

        safe_input: dict[str, Any] = {
            "repository": repository,
            "head_branch": source_branch,
            "base_branch": target_branch,
            "title": title,
            "body_fingerprint": body_fingerprint,
            "draft": draft,
        }
        if expected_head_sha is not None:
            safe_input["expected_head_sha"] = expected_head_sha
        result = await self._operations.run(
            operation_type=ExternalOperationType.CREATE_PULL_REQUEST,
            logical_step=f"create_pr:{source_branch}:{target_branch}",
            safe_input=safe_input,
            action=action,
            reconcile=reconcile,
            # Without a replay budget the first failed create is journaled as terminal, and a
            # terminal operation refuses to run again -- so the publisher's own retry, and
            # every later resume, hit that refusal instead of the provider, and a reviewed
            # branch could never get its pull request. Replaying cannot duplicate one: the
            # action looks the pull request up first and returns the existing one.
            #
            # Deliberately larger than the publisher's in-run retry budget. Every one of its
            # attempts spends one of these, so an equal budget would be gone within seconds
            # of a provider outage and leave a resume nothing to retry with.
            max_attempts=6,
        )
        if result.reused:
            return _pr_from_payload(result.operation.result_payload or {})
        return cast(PullRequestDetails, result.value)

    async def add_labels(
        self, repository: str, pull_request_number: int, labels: Sequence[str]
    ) -> None:
        """Attach labels once; the journal makes a restart reuse the successful mutation."""
        expected = list(labels)

        async def reconcile(_operation: ExternalOperation) -> ReconciliationOutcome:
            observed = await self._read_capability(
                "get_pull_request_labels", repository, pull_request_number
            )
            if observed is None:
                return EffectUnproven(
                    method="pull_request_labels",
                    detail="the pull request's labels could not be read",
                )
            missing = [label for label in expected if label not in observed]
            if not missing:
                return OperationResult(
                    payload={"completed": True, "recovery_method": "labels_present"}
                )
            # Read from the provider, not inferred from the incomplete operation record: a
            # label the provider does not have is one the mutation genuinely did not apply,
            # and re-applying it is idempotent on GitHub's side anyway.
            return EffectAbsent(
                method="pull_request_labels",
                detail="the pull request is missing labels this operation intended to add",
                observations={"missing_labels": missing},
            )

        await self._mutate(
            operation_type=ExternalOperationType.ADD_LABELS,
            logical_step=f"labels:{pull_request_number}",
            safe_input={
                "repository": repository,
                "pull_request_number": pull_request_number,
                "labels": expected,
            },
            action=lambda: self._service.add_labels(repository, pull_request_number, labels),
            reconcile=reconcile,
            # A replay budget, so a reconciliation that proves the labels absent has an
            # attempt left to apply them with. Without one, every interrupted label mutation
            # was terminal at attempt one and the only remaining disposition for it was to
            # ask a person about a decoration on a pull request. Replaying cannot duplicate
            # anything: labels are a set on the provider's side.
            max_attempts=3,
        )

    async def request_reviewers(
        self, repository: str, pull_request_number: int, reviewers: Sequence[str]
    ) -> None:
        """Request reviews once; cancellation prevents a later mutation from being started."""
        expected = list(reviewers)

        async def reconcile(_operation: ExternalOperation) -> ReconciliationOutcome:
            observed = await self._read_capability(
                "get_requested_reviewers", repository, pull_request_number
            )
            if observed is None:
                return EffectUnproven(
                    method="pull_request_reviewers",
                    detail="the pull request's requested reviewers could not be read",
                )
            missing = [reviewer for reviewer in expected if reviewer not in observed]
            if not missing:
                return OperationResult(
                    payload={"completed": True, "recovery_method": "reviewers_requested"}
                )
            # A reviewer who has already reviewed no longer appears as requested, and the
            # provider's answer cannot distinguish that from never having been asked. Calling
            # this absent would re-request a review somebody has already given, so it stays
            # deferred and says why.
            return EffectUnproven(
                method="pull_request_reviewers",
                detail=(
                    "a reviewer that is not currently requested may have been requested and "
                    "already reviewed; the provider cannot distinguish the two"
                ),
                observations={"unconfirmed_reviewers": missing},
            )

        await self._mutate(
            operation_type=ExternalOperationType.ADD_REVIEWERS,
            logical_step=f"reviewers:{pull_request_number}",
            safe_input={
                "repository": repository,
                "pull_request_number": pull_request_number,
                "reviewers": expected,
            },
            action=lambda: self._service.request_reviewers(
                repository, pull_request_number, reviewers
            ),
            reconcile=reconcile,
            # Same reasoning as the labels above; a review request is also a set.
            max_attempts=3,
        )

    async def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        """Post a comment once: prove presence before believing, absence before retrying.

        The one mutation in this file that is not a set. Labels and reviewers can be replayed
        blindly because the provider merges them; a comment posts again every time, so the
        retry budget here is safe only because the action and the reconcile both list the pull
        request's comments and match on the body fingerprint first. Sixteen of the seventeen
        cross-link comments in the deployed record died at ``attempt=1, max_attempts=1`` --
        this closes that retry class.

        The loop at the bottom is the driver none of the siblings needed. The journal ladders
        ``FAILED_RETRYABLE`` from the budget, but a retryable row only runs again when
        something calls ``run()`` again -- and the publisher calls once and absorbs, so
        without the loop the budget would just sit there. A comment failure still gates
        nothing: exhaustion and refusals raise to the caller, which absorbs them as before.
        """
        body_fingerprint = hashlib.sha256(body.encode("utf-8")).hexdigest()

        async def observed_fingerprints() -> set[str] | None:
            """The fingerprint of every comment the provider shows, or ``None`` if unreadable."""
            bodies = await self._read_capability(
                "get_pull_request_comments", repository, pull_request_number
            )
            if bodies is None:
                return None
            return {hashlib.sha256(item.encode("utf-8")).hexdigest() for item in bodies}

        present = OperationResult(payload={"completed": True, "recovery_method": "comment_present"})
        reached_provider = False

        async def action() -> tuple[None, OperationResult]:
            # Look first, exactly as `create_pull_request`'s action does, so a retry after a
            # lost answer can never post the comment a second time.
            nonlocal reached_provider
            observed = await observed_fingerprints()
            if observed is not None and body_fingerprint in observed:
                return None, present
            if observed is None and reached_provider:
                # An earlier try reached the provider and the comments cannot be read back.
                # Posting now is the blind replay the journal exists to prevent; deferring is
                # honest, and the reconcile settles it on a later credentialed pass.
                raise UnknownExternalOperation(
                    "an earlier try may have posted this comment and the pull request's "
                    "comments cannot be read back"
                )
            reached_provider = True
            try:
                await asyncio.wait_for(
                    await_cancellable(
                        asyncio.to_thread(
                            self._service.add_comment, repository, pull_request_number, body
                        ),
                        self._operations.cancellation_token,
                    ),
                    timeout=self._timeout_seconds,
                )
            except (CancellationRequested, TimeoutError):
                raise UnknownExternalOperation(
                    "GitHub mutation may have completed after cancellation"
                ) from None
            return None, OperationResult(payload={"completed": True})

        async def reconcile(_operation: ExternalOperation) -> ReconciliationOutcome:
            observed = await observed_fingerprints()
            if observed is None:
                return EffectUnproven(
                    method="pull_request_comments",
                    detail="the pull request's comments could not be read",
                )
            if body_fingerprint in observed:
                return present
            # Read from the provider, not inferred from the incomplete operation record: a
            # comment the provider does not show is one that genuinely never posted, and only
            # that proof makes running the action again safe.
            return EffectAbsent(
                method="pull_request_comments",
                detail="no comment with this body exists on the pull request",
                observations={"observed_comment_count": len(observed)},
            )

        for attempt in range(1, _COMMENT_ATTEMPTS + 1):
            try:
                await self._operations.run(
                    operation_type=ExternalOperationType.UPDATE_PULL_REQUEST,
                    logical_step=f"comment:{pull_request_number}:{body_fingerprint[:16]}",
                    safe_input={
                        "repository": repository,
                        "pull_request_number": pull_request_number,
                        "body_fingerprint": body_fingerprint,
                    },
                    action=action,
                    reconcile=reconcile,
                    max_attempts=_COMMENT_ATTEMPTS,
                    # So the row's terminal ladder and this loop cannot disagree: a fault the
                    # predicate refuses is recorded terminal even with budget left, because no
                    # driver will ever spend it.
                    fault_admits_retry=is_transient_provider_fault,
                )
                return
            except CancellationRequested:
                raise
            except Exception as error:  # noqa: BLE001 - re-raised unless the one predicate admits it
                # Bounded by the same number as the budget, so the loop can never meet a
                # terminal row; stopped dead by a refused or deterministic answer -- a 403 is
                # a permission grant a person has to make, and sixteen recorded failures were
                # exactly that.
                if attempt >= _COMMENT_ATTEMPTS or not is_transient_provider_fault(error):
                    raise
                await await_cancellable(
                    asyncio.sleep(_COMMENT_FAULT_BACKOFF_SECONDS * 2 ** (attempt - 1)),
                    self._operations.cancellation_token,
                )

    async def close_pull_request(self, repository: str, pull_request_number: int) -> None:
        """Close a superseded pull request once, proving its state before acting on it.

        Naturally idempotent -- closing a closed pull request changes nothing -- so the replay
        budget is safe without a fingerprint: the action reads the state first and a pull
        request that is already closed *or merged* is left exactly as it is. Merged counts as
        settled, not as a failure: a person accepting the superseded work is their decision,
        and this operation must never report it as an unclosed pull request.
        """
        settled = OperationResult(
            payload={"completed": True, "recovery_method": "pull_request_not_open"}
        )

        async def action() -> tuple[None, OperationResult]:
            state = await self._read_pull_request_state(repository, pull_request_number)
            if state in {"closed", "merged"}:
                return None, settled
            try:
                await asyncio.wait_for(
                    await_cancellable(
                        asyncio.to_thread(
                            self._service.close_pull_request, repository, pull_request_number
                        ),
                        self._operations.cancellation_token,
                    ),
                    timeout=self._timeout_seconds,
                )
            except (CancellationRequested, TimeoutError):
                raise UnknownExternalOperation(
                    "GitHub mutation may have completed after cancellation"
                ) from None
            return None, OperationResult(payload={"completed": True})

        async def reconcile(_operation: ExternalOperation) -> ReconciliationOutcome:
            state = await self._read_pull_request_state(repository, pull_request_number)
            if state is None:
                return EffectUnproven(
                    method="pull_request_state",
                    detail="the pull request's state could not be read",
                )
            if state in {"closed", "merged"}:
                return settled
            # Read from the provider, not inferred: a pull request the provider still shows
            # open genuinely was not closed, and closing it again is idempotent anyway.
            return EffectAbsent(
                method="pull_request_state",
                detail="the pull request is still open",
                observations={"state": state},
            )

        await self._operations.run(
            operation_type=ExternalOperationType.CLOSE_PULL_REQUEST,
            logical_step=f"close:{pull_request_number}",
            safe_input={
                "repository": repository,
                "pull_request_number": pull_request_number,
            },
            action=action,
            reconcile=reconcile,
            max_attempts=3,
        )

    async def _read_pull_request_state(
        self, repository: str, pull_request_number: int
    ) -> str | None:
        """Read a pull request's lifecycle state, with every failure meaning 'cannot confirm'."""
        reader = getattr(self._service, "get_pull_request_state", None)
        if reader is None:
            return None
        try:
            value = await asyncio.wait_for(
                asyncio.to_thread(reader, repository, pull_request_number),
                timeout=self._timeout_seconds,
            )
        except CancellationRequested:
            raise
        except Exception:
            return None
        return str(value) if isinstance(value, str) and value else None

    async def _mutate(
        self,
        *,
        operation_type: ExternalOperationType,
        logical_step: str,
        safe_input: dict[str, Any],
        action: Any,
        reconcile: Callable[[ExternalOperation], Awaitable[ReconciliationOutcome]] | None = None,
        max_attempts: int = 1,
    ) -> None:
        """Run a bounded thread-backed mutation and report ambiguity honestly after cancellation."""

        async def invoke() -> tuple[None, OperationResult]:
            try:
                await asyncio.wait_for(
                    await_cancellable(
                        asyncio.to_thread(action), self._operations.cancellation_token
                    ),
                    timeout=self._timeout_seconds,
                )
            except (CancellationRequested, TimeoutError):
                raise UnknownExternalOperation(
                    "GitHub mutation may have completed after cancellation"
                ) from None
            return None, OperationResult(payload={"completed": True})

        await self._operations.run(
            operation_type=operation_type,
            logical_step=logical_step,
            safe_input=safe_input,
            action=invoke,
            reconcile=reconcile,
            max_attempts=max_attempts,
        )

    async def _read_capability(
        self, name: str, repository: str, pull_request_number: int
    ) -> list[str] | None:
        """Read an optional provider list without ever letting a lookup become a mutation.

        Follows ``_find_existing``: the capability is optional, its absence means "cannot
        confirm" rather than "not there", and every failure to reach the provider is the
        same answer -- nothing was established.
        """
        reader = getattr(self._service, name, None)
        if reader is None:
            return None
        try:
            value = await asyncio.wait_for(
                asyncio.to_thread(reader, repository, pull_request_number),
                timeout=self._timeout_seconds,
            )
        except CancellationRequested:
            raise
        except Exception:
            return None
        if not isinstance(value, (list, tuple)):
            return None
        return [str(item) for item in value]

    async def _candidate_pull_requests(
        self, repository: str, source_branch: str, target_branch: str
    ) -> list[PullRequestDetails] | None:
        """List every pull request on this head and base, or ``None`` if none can be read."""
        finder = getattr(self._service, "find_pull_requests", None)
        if finder is None:
            return None
        try:
            value = await asyncio.wait_for(
                asyncio.to_thread(
                    finder,
                    repository,
                    source_branch=source_branch,
                    target_branch=target_branch,
                ),
                timeout=self._timeout_seconds,
            )
        except CancellationRequested:
            raise
        except Exception:
            return None
        if not isinstance(value, (list, tuple)):
            return None
        return [item for item in value if isinstance(item, PullRequestDetails)]

    async def find_pull_request(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> PullRequestDetails | None:
        """Fetch a pull request back so completion can be verified rather than assumed.

        Exposed publicly because the completion guard has to confirm that every published
        pull request really exists. Only the private reconciliation helper offered this, so
        the guard raised an AttributeError and reported a finished feature as unverifiable.
        A lookup changes nothing, so it is not journaled.
        """
        return await self._find_existing(repository, source_branch, target_branch, title)

    async def _find_existing(
        self, repository: str, source_branch: str, target_branch: str, title: str
    ) -> PullRequestDetails | None:
        """Use an optional adapter capability to reconcile an interrupted PR create safely."""
        finder = getattr(self._service, "find_pull_request", None)
        if finder is None:
            return None
        value = await asyncio.wait_for(
            asyncio.to_thread(
                finder,
                repository,
                source_branch=source_branch,
                target_branch=target_branch,
                title=title,
            ),
            timeout=self._timeout_seconds,
        )
        return value if isinstance(value, PullRequestDetails) else None


def _pr_result(details: PullRequestDetails, **extra: Any) -> OperationResult:
    """Retain exactly the non-secret fields needed to rebuild a PR artifact after restart."""
    return OperationResult(
        external_reference=details.url,
        payload={
            "repository": details.repository,
            "number": details.number,
            "url": details.url,
            "title": details.title,
            "source_branch": details.source_branch,
            "target_branch": details.target_branch,
            "head_sha": details.head_sha,
            **extra,
        },
    )


def _pr_from_payload(payload: dict[str, Any]) -> PullRequestDetails:
    """Rehydrate a prior successful PR operation without making another GitHub API request."""
    try:
        return PullRequestDetails(
            repository=str(payload["repository"]),
            number=int(payload["number"]),
            url=str(payload["url"]),
            title=str(payload["title"]),
            source_branch=str(payload["source_branch"]),
            target_branch=str(payload["target_branch"]),
            head_sha=(str(payload["head_sha"]) if payload.get("head_sha") else None),
        )
    except (KeyError, TypeError, ValueError) as error:
        msg = "completed pull-request operation has invalid persisted result data"
        raise UnknownExternalOperation(msg) from error


__all__ = ["JournaledGitHubService"]
