"""The vocabulary a stopped feature explains itself in, and the contract that carries it.

Three things live here, and they live together because they are one decision:

``FeatureFailureClassification``
    Every value ``FeatureFailureSummary.root_classification`` is allowed to take. Before
    this, that field accepted any string, so a ``DBAPIError`` and a ``ValidationError``
    were recorded as though they said something about the target repository -- and a person
    was asked to decide about a repository whose only fault was being open at the time.

``FailureStage``
    Where a failure happened, as opposed to whichever agent the orchestrator happened to
    have assigned when the record was written.

``DiagnosedFailure``
    The exception contract. ``RepositoryWorkspaceError`` and its siblings already carried a
    classification and diagnostics informally, discovered by ``getattr`` at the boundary.
    Formalising it is what makes "this exception explains itself" checkable rather than
    hopeful.

Deliberately dependency-free beyond the standard library. Everything that records a terminal
feature state imports this -- ``state``, ``storage``, ``services``, ``workflows``, ``api`` --
and a single import edge into any of those would put a cycle in the middle of the platform's
error path.
"""

from __future__ import annotations

from enum import StrEnum

# Substrings of the exception or provider class names raised when a provider or the network
# fails to answer, rather than answering with something the platform must act on. Matching on
# the name keeps this independent of which provider SDK raised it: an OpenAI `APITimeoutError`,
# a stdlib `TimeoutError`, and a `ConnectionResetError` are the same operational event.
_TRANSIENT_MARKERS = (
    "timeout",
    "connection",
    "ratelimit",
    "serviceunavailable",
    "internalservererror",
)


class FeatureFailureClassification(StrEnum):
    """Every cause a terminal feature record is allowed to name.

    The first eight mirror ``tools.retry_strategy.FailureClassification``, which classifies
    one child attempt's blocker and decides which retry budget it spends. They are repeated
    rather than imported because ``tools`` sits above ``state`` in the import graph -- a test
    asserts the two enums agree, so the mirror cannot drift silently.

    The rest are feature-level: causes that stop a feature without any one repository having
    failed, and which therefore have no retry budget of their own.
    """

    # -- Mirrors of the child-attempt classifications ------------------------------------
    IMPLEMENTATION_MISSING = "implementation_missing"
    VALIDATION_SOURCE_FAILURE = "validation_source_failure"
    VALIDATION_CAPACITY_FAILURE = "validation_capacity_failure"
    VALIDATION_CONFIGURATION_FAILURE = "validation_configuration_failure"
    DEPENDENCY_INSTALLATION_FAILURE = "dependency_installation_failure"
    TEST_INFRASTRUCTURE_MISSING = "test_infrastructure_missing"
    REVIEW_SCOPE_FAILURE = "review_scope_failure"
    CONTRACT_MISMATCH = "contract_mismatch"

    # -- Ours -----------------------------------------------------------------------------
    # An unexpected exception type reached a boundary, or a path recorded a terminal state
    # without saying why. Either way the platform is what failed, and the record has to say
    # so rather than presenting a defect in this code as a statement about the repository.
    PLATFORM_DEFECT = "platform_defect"
    # The worker volume could not accept another clone or install. Not a defect -- a limit.
    PLATFORM_CAPACITY_FAILURE = "platform_capacity_failure"
    # The workspace could not be prepared or cleaned before coding could start.
    WORKSPACE_MAINTENANCE_FAILURE = "workspace_maintenance_failure"

    # -- Runtime ceilings --------------------------------------------------------------------
    # Both are backstops, and both are deliberately separate from every value above. A run
    # that exceeds its ceiling is neither a platform defect nor a finding about the target
    # repository: nothing went wrong that anyone can point at, the work simply did not
    # converge inside the time this deployment is willing to spend on it. Filing that as
    # `PLATFORM_DEFECT` would send somebody to debug this codebase, and filing it as any of
    # the validation classes would send them to debug a repository that may be fine.
    #
    # Distinct from `EXECUTOR_STOPPED`, which means the process died. A feature that reaches
    # one of these was still making progress when it was stopped.
    FEATURE_RUNTIME_LIMIT_REACHED = "feature_runtime_limit_reached"
    REPOSITORY_RUNTIME_LIMIT_REACHED = "repository_runtime_limit_reached"

    # -- The external services this platform depends on ------------------------------------
    # The model provider did not answer. Retryable on its own, and the reason the
    # retryability rule cannot be deleted along with the substring table.
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    # The Git remote did not answer, or the platform could not reach it. Retryable for the
    # same reason and by the same rule -- but a separate value, because these two are the
    # only classifications a reader can act on by going and looking at a *service*, and
    # there is no way to tell which service from `provider_unavailable`. AB-Feature-190's
    # clones died on a GitHub outage and every classification-keyed surface said "the model
    # provider did not answer"; 54- Part 2 fixed the narrative sentence and left the
    # classification, so the feature-level summary and the logbook one-liner -- which read
    # the classification and not the sentence -- went on naming the wrong service.
    GIT_REMOTE_UNAVAILABLE = "git_remote_unavailable"
    # The design source did not answer. The third external service, and here for the reason
    # the second one is: a reader can act on this by going and looking at Figma, and there was
    # no way to tell that from `provider_unavailable`. AB-Feature-227 and AB-Feature-228 both
    # died on a Figma 429 and both told their operator to confirm the *model* provider was
    # answering -- the identical bug this enum's own comment above records being fixed for
    # GitHub in 54- Part 2, shipped a third time because naming was not extended when a third
    # external service arrived.
    DESIGN_SOURCE_UNAVAILABLE = "design_source_unavailable"
    # This deployment has no credentials for the account that asked.
    PROVIDER_CREDENTIALS_MISSING = "provider_credentials_missing"

    # -- Publication and external effects --------------------------------------------------
    PUBLICATION_FAILURE = "publication_failure"
    PULL_REQUEST_UNVERIFIED = "pull_request_unverified"
    UNCONFIRMED_EXTERNAL_EFFECT = "unconfirmed_external_effect"

    # -- Feature lifecycle ------------------------------------------------------------------
    CLARIFICATION_UNRESOLVED = "clarification_unresolved"
    CONTRACT_REVISION_LIMIT_REACHED = "contract_revision_limit_reached"
    CONTRACT_REVISION_REJECTED = "contract_revision_rejected"
    CONTRACT_REVISION_REQUIRES_FRESH_BRANCHES = "contract_revision_requires_fresh_branches"
    INTEGRATION_REVIEW_UNSATISFIED = "integration_review_unsatisfied"
    WORKSTREAM_UNFINISHED = "workstream_unfinished"
    EXECUTOR_STOPPED = "executor_stopped"
    FEATURE_QUEUE_REFUSED = "feature_queue_refused"


class FailureStage(StrEnum):
    """Where a failure happened.

    Distinct from the responsible agent on purpose. ``FeatureFailureSummary`` used to derive
    both from ``current_agent``, which names whatever the orchestrator last assigned -- so a
    workstream that failed its own validation was filed under ``github`` because publication
    ran for an approved sibling afterwards. Seventeen of ninety-nine recorded failures were
    attributed that way; five of them were actually publication failures.
    """

    PRODUCT_MANAGER = "product_manager"
    RECONNAISSANCE = "reconnaissance"
    FEATURE_PLANNER = "feature_planner"
    CHILD_WORKFLOWS = "child_workflows"
    INTEGRATION_REVIEW = "integration_review"
    PULL_REQUEST_PUBLICATION = "pull_request_publication"
    CONTRACT_REVISION = "contract_revision"
    HUMAN_CLARIFICATION = "human_clarification"
    FEATURE_QUEUE = "feature_queue"
    OPERATION_RECOVERY = "operation_recovery"
    RUN_RECOVERY = "run_recovery"
    FEATURE_RUNTIME = "feature_runtime"


# What to say when a terminal path supplies no diagnostic of its own. Never an empty list:
# five recorded failures carried one, which tells a reader nothing at all and is
# indistinguishable from a feature that has not failed yet.
_CLASSIFICATION_FALLBACKS: dict[FeatureFailureClassification, str] = {
    FeatureFailureClassification.PLATFORM_DEFECT: (
        "The platform failed while running this feature and recorded no cause. This is a "
        "defect in the platform, not in the target repository or the requirement."
    ),
    FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE: (
        "The worker could not provide the workspace this feature needs. This is a platform "
        "capacity limit, not a fault in the target repository."
    ),
    FeatureFailureClassification.WORKSPACE_MAINTENANCE_FAILURE: (
        "The platform could not prepare or clean a workspace for this feature."
    ),
    FeatureFailureClassification.FEATURE_RUNTIME_LIMIT_REACHED: (
        "This feature ran for longer than this deployment allows a single feature to run, "
        "so it was stopped. Nothing here says the repository or the requirement is wrong."
    ),
    FeatureFailureClassification.REPOSITORY_RUNTIME_LIMIT_REACHED: (
        "One repository's workstream ran for longer than this deployment allows a single "
        "repository to run, so no further attempt was scheduled for it."
    ),
    FeatureFailureClassification.PROVIDER_UNAVAILABLE: (
        "The model provider did not answer. Nothing here says anything about the target "
        "repository or the requirement."
    ),
    FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE: (
        "The Git remote did not answer, or the platform could not reach it with the access "
        "it has. This is not the model provider, and nothing here says anything about the "
        "target repository or the requirement."
    ),
    FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE: (
        "The design source did not answer, so the designs this feature cites could not be "
        "read. This is neither the model provider nor the Git remote, and nothing here says "
        "anything about the target repository or the requirement."
    ),
    FeatureFailureClassification.PROVIDER_CREDENTIALS_MISSING: (
        "This feature needs provider credentials that are not configured for the account "
        "that requested it."
    ),
    FeatureFailureClassification.PUBLICATION_FAILURE: (
        "Pull-request publication did not complete for every repository this feature "
        "reviewed and approved."
    ),
    FeatureFailureClassification.PULL_REQUEST_UNVERIFIED: (
        "A pull request this run reported creating could not be read back from the provider."
    ),
    FeatureFailureClassification.UNCONFIRMED_EXTERNAL_EFFECT: (
        "An operation against the provider was interrupted and its effect could not be "
        "confirmed. Check the repository before this work is attempted again."
    ),
    FeatureFailureClassification.CLARIFICATION_UNRESOLVED: (
        "This feature reached its limit of clarification rounds without its open questions "
        "being resolved."
    ),
    FeatureFailureClassification.CONTRACT_REVISION_LIMIT_REACHED: (
        "This feature reached its limit of shared-contract revisions."
    ),
    FeatureFailureClassification.CONTRACT_REVISION_REJECTED: (
        "A person declined the shared-contract change this feature asked for, so the "
        "workstreams that depend on it cannot continue."
    ),
    FeatureFailureClassification.CONTRACT_REVISION_REQUIRES_FRESH_BRANCHES: (
        "A shared-contract revision after a child commit requires a new feature run on "
        "fresh branches."
    ),
    FeatureFailureClassification.INTEGRATION_REVIEW_UNSATISFIED: (
        "The integration review never approved this feature, so no completion may claim "
        "that it did."
    ),
    FeatureFailureClassification.WORKSTREAM_UNFINISHED: (
        "A required repository produced no review-approved, publishable result."
    ),
    FeatureFailureClassification.EXECUTOR_STOPPED: (
        "The executor holding this feature stopped without writing a terminal status."
    ),
    FeatureFailureClassification.FEATURE_QUEUE_REFUSED: (
        "The platform declined to start this feature and recorded no further detail."
    ),
    FeatureFailureClassification.IMPLEMENTATION_MISSING: (
        "A repository workstream stopped without a production implementation to review."
    ),
    FeatureFailureClassification.VALIDATION_SOURCE_FAILURE: (
        "A repository workstream stopped on its own validation commands rejecting the "
        "source that was written."
    ),
    FeatureFailureClassification.VALIDATION_CAPACITY_FAILURE: (
        "A required validation command never finished here, so it returned no verdict on "
        "the code that was written."
    ),
    FeatureFailureClassification.VALIDATION_CONFIGURATION_FAILURE: (
        "A repository could not run its own checks on an untouched checkout."
    ),
    FeatureFailureClassification.DEPENDENCY_INSTALLATION_FAILURE: (
        "Deterministic dependency installation failed, so no validation could run."
    ),
    FeatureFailureClassification.TEST_INFRASTRUCTURE_MISSING: (
        "A repository has no configured test infrastructure for this work to be checked against."
    ),
    FeatureFailureClassification.REVIEW_SCOPE_FAILURE: (
        "A repository workstream stopped with review findings it did not address."
    ),
    FeatureFailureClassification.CONTRACT_MISMATCH: (
        "A repository's implementation did not match the approved integration contract."
    ),
}

# Which classifications mean "attempt this again without asking anyone". Retryability used
# to be inferred by matching substrings against whatever string happened to be recorded, which
# worked only because provider SDK exception names were being persisted verbatim. The
# substring rule is retained below for exactly that reason: durable rows written before this
# change still carry those names, and reading one must not silently change its verdict.
#
# Two values, one property. Both name an external service that failed to answer, and the
# retry rule is about the answer and not the service: a fault is retryable because the effect
# was confirmed *not* to have landed, never because of the exception's type. A Git outage is
# recorded here by the one path that has already established that -- the fault loop, whose
# admission test is `is_transient_provider_fault` -- so splitting the classification moves
# the prose across the five surfaces that key on it and moves the verdict nowhere.
#
# The one thing a Git fault does not share is cost: a clone retry spends no model tokens, so
# the platform can afford to wait considerably longer before asking a person. That difference
# lives in the backoff schedules, not here.
_RETRYABLE_CLASSIFICATIONS = frozenset(
    {
        FeatureFailureClassification.PROVIDER_UNAVAILABLE,
        FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE,
        FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE,
    }
)


def _looks_transient(value: str) -> bool:
    """Return whether a classification names the provider failing to answer."""
    normalized = value.replace("_", "").replace(" ", "").lower()
    return any(marker in normalized for marker in _TRANSIENT_MARKERS)


def is_retryable_classification(classification: str) -> bool:
    """Return whether a failure was an external service not answering, not the work being wrong.

    A transient failure is worth resuming: the operation journal replays completed effects,
    so a resume costs one retry rather than a whole feature. Recording every failure as
    terminal -- as this once did -- forced a human decision on a timeout that a plain resume
    would have cleared, and stranded features that had already pushed commits.

    Which service it was makes no difference here, and deliberately so. This is the single
    predicate behind the summary's `retryable` flag, the status a failed execution is left in,
    the two resume seams, and whether `RESUME_WORKFLOW` is advertised at all -- so it answers
    one question about the failure and not about its origin.
    """
    if classification in _RETRYABLE_CLASSIFICATIONS:
        return True
    return _looks_transient(classification)


def normalize_classification(value: str | None) -> FeatureFailureClassification:
    """Turn any recorded failure name into one this vocabulary can state.

    A value already in the enum is kept. A provider SDK's exception name -- which is what
    ``LLMAdapterError`` carries, and what used to be persisted verbatim -- becomes
    ``PROVIDER_UNAVAILABLE`` when it names the provider failing to answer. Anything else is a
    type name that reached a boundary nobody classified, which is this platform's defect and
    is recorded as one rather than as a claim about the repository.

    ``GIT_REMOTE_UNAVAILABLE`` and ``DESIGN_SOURCE_UNAVAILABLE`` are deliberately absent from
    that mapping: no exception type name resolves to either. A Git failure is only that
    classification where a caller has established the effect did not land, which is the fault
    loop and nowhere else -- and a ``FigmaClientError`` is admitted by that same loop through
    the adapter's own ``retryable``, never by anything reading its type name here. A
    ``GitAdapterError`` that escapes publication is the opposite case -- a push may well have
    reached the remote -- and presenting it as retryable is precisely the guess the operation
    journal exists to prevent.
    """
    if value is None or not value.strip():
        return FeatureFailureClassification.PLATFORM_DEFECT
    candidate = value.strip()
    try:
        return FeatureFailureClassification(candidate)
    except ValueError:
        pass
    if _looks_transient(candidate):
        return FeatureFailureClassification.PROVIDER_UNAVAILABLE
    return FeatureFailureClassification.PLATFORM_DEFECT


def fallback_diagnostic(classification: FeatureFailureClassification) -> str:
    """Return the sentence a classification is worth on its own, so no record says nothing."""
    return _CLASSIFICATION_FALLBACKS.get(
        classification, _CLASSIFICATION_FALLBACKS[FeatureFailureClassification.PLATFORM_DEFECT]
    )


def workspace_capacity_diagnostic(
    *, workspace_root: str, free_bytes: int, required_bytes: int
) -> str:
    """Say the worker volume is too full, in one sentence, wherever that is discovered.

    Two places ask this question about the same volume: the capacity preflight that runs
    while a feature is executing, and the check that refuses a submission before accepting
    it. They must not answer it in two different wordings -- an operator who reads the
    refusal and then reads the mid-run failure is looking at one condition, not two.

    It lives here, in the module with no dependencies beyond the standard library, because
    the two callers sit on opposite sides of the import graph: ``services`` imports
    ``storage``, so the storage layer cannot reach the exception class that used to own
    this text.
    """
    return (
        f"Workspace capacity preflight failed for `{workspace_root}`: {free_bytes} bytes "
        f"free, but {required_bytes} bytes are required. Automatic cleanup only removes "
        "older DB-confirmed completed or cancelled live-feature workspaces; expand the "
        "workspace volume or archive an operator-reviewed retained workspace before retrying."
    )


class DiagnosedFailure(Exception):
    """An exception that states its own classification and carries safe diagnostics.

    The contract every exception crossing a workstream or feature boundary is expected to
    satisfy. Two fields, both mandatory:

    ``failure_classification``
        A ``FeatureFailureClassification``, never a free string. A free string is how
        ``platform_capacity_failure`` and ``APITimeoutError`` ended up in the same field as
        ``validation_source_failure``, with nothing able to tell which kind of thing it was.

    ``diagnostics``
        At least one sentence a person can act on, composed from platform constants and
        platform-generated identifiers only -- never from model output, a checkout, or the
        process environment. ``safe_error_diagnostics`` reads this attribute, so a raiser
        that passes text it did not compose itself is publishing it into durable state.

    Both are enforced in ``__init__`` rather than documented, because the informal version of
    this contract was satisfied by two of the three exceptions that claimed to follow it.
    """

    failure_classification: FeatureFailureClassification
    diagnostics: tuple[str, ...]

    def __init__(
        self,
        *args: object,
        classification: FeatureFailureClassification,
        diagnostics: tuple[str, ...] | list[str],
    ) -> None:
        """Record a classification from the enum and at least one safe sentence."""
        safe = tuple(str(item) for item in diagnostics if str(item).strip())
        if not safe:
            # A raiser with nothing to say still says something, because the alternative is
            # the empty diagnostics array this class exists to make impossible.
            safe = (fallback_diagnostic(classification),)
        self.failure_classification = classification
        self.diagnostics = safe
        super().__init__(*args)


def classification_of(error: BaseException) -> FeatureFailureClassification:
    """Classify any exception that reached a terminal boundary.

    Reads the contract first, then the informal attribute the adapters already set, then the
    type name. An exception that declares nothing is not evidence about the repository: it is
    evidence that a code path here did not anticipate it, so it lands on ``PLATFORM_DEFECT``.
    """
    declared = getattr(error, "failure_classification", None)
    if isinstance(declared, FeatureFailureClassification):
        return declared
    if isinstance(declared, str) and declared.strip():
        return normalize_classification(declared)
    return normalize_classification(type(error).__name__)


__all__ = [
    "DiagnosedFailure",
    "FailureStage",
    "FeatureFailureClassification",
    "classification_of",
    "fallback_diagnostic",
    "is_retryable_classification",
    "normalize_classification",
    "workspace_capacity_diagnostic",
]
