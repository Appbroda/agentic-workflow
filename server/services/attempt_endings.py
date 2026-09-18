"""Where every finished attempt ended, and what the platform recorded about it.

The agent-work drawer renders one attempt at a time from the operation journal, and the
journal cannot say where an attempt *ended*: AB-Feature-201's backend attempt 0 ran 57
minutes, journaled a clone, a branch, an install, three baseline commands and a coding call,
every one of them ``succeeded`` -- and was then stopped by the implementation self-review
before the reviewer was ever called. A wall of green ticks is what the drawer drew, and the
person writing this module believed it had been rejected at review until the records said
otherwise.

So the ending is a served fact rather than a client guess, assembled here from records the
platform already writes and nothing new:

* ``011_child_workflow_result.<repo>.attempt-N.json`` -- that attempt's own status, failure
  classification, blocking issues, validation results and ``attempt_inputs``;
* ``006_code_completion.<repo>.attempt-N.json`` -- the self-review record and the in-attempt
  repair pass count, which live in the completion's metadata and on no served response;
* ``007_review.<repo>.attempt-N.json`` -- the verdict and its findings, reached through the
  result's own ``review_artifact_id`` rather than by guessing the filename.

Two rules run through the whole of it.

**Status first, classification second.** 201's backend attempts 2, 6, 7 and 8 are ``approved``
and *still carry* a ``failure_classification``. Reading the classification first reports four
delivered attempts as failures, which is how the read that produced this module started.

**One ending per attempt.** An attempt can satisfy two values at once -- rejected at review
and later superseded; a required command that failed inside a git fault. ``ended_by`` names
the record that ended *this attempt's own cycle*; ``superseded`` is reserved for an attempt
that completed its cycle and was reopened by a later one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

from artifacts.schemas import (
    Artifact,
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    ReviewArtifact,
)
from state.enums import ChildWorkflowStatus
from state.feature_models import ChildWorkflowReference, FeatureWorkflowSnapshot


class AttemptEndingKind(StrEnum):
    """The record kinds that can end one repository attempt.

    Deliberately not a new vocabulary: each value names a record the platform already writes,
    so a reader who knows the run knows the word. Item 17's lesson is that two vocabularies
    for one fact is how a label comes to mean the wrong thing on half the records.
    """

    # The implementation self-review stopped the attempt before the reviewer was called.
    SELF_REVIEW = "self_review"
    # The repository reviewer answered, and its answer was not an approval.
    REVIEW_REJECTED = "review_rejected"
    # A required validation command failed.
    VALIDATION_FAILED = "validation_failed"
    # The attempt declined to proceed and said why.
    REFUSAL = "refusal"
    # The platform classified the attempt's own blocker, and it is not one of the above.
    FAULT = "fault"
    # The attempt completed its cycle and a later attempt reopened it.
    SUPERSEDED = "superseded"
    APPROVED = "approved"
    PUBLISHED = "published"


# The three self-review outcomes that stop an attempt. `clean`, `corrected` and `unavailable`
# do not: the first two continue into validation and review, and the third records that the
# reviewer was composed and could not run, which is a fact about the pass and not an ending.
_SELF_REVIEW_STOPS = frozenset({"corrections_failed", "correction_rejected", "substantive_problem"})

# Which ending a failed attempt takes, decided by the classification the record itself
# carries and by nothing else -- the 20- split, one source, never re-derived. These two
# classes say the change failed its own gate, so the ending is `validation_failed`. A red
# command under any other class is the same red command and a different ending: a repository
# whose toolchain cannot run the gate at all did not fail validation, and calling it
# `validation_failed` would send a reader to look at a diff that is fine.
#
# 201's backend attempt 4 is why the command itself cannot be the discriminator: its
# `current_validation_results` is empty, because the gate that rejected it was the Engineer's
# own in-attempt source validation, which nothing journals. The classification still names
# what happened.
_VALIDATION_FAILURE_CLASSES = frozenset(
    {"validation_source_failure", "validation_capacity_failure"}
)

# Which stage an ending landed in, for the classes that do not name one themselves. The
# server owns this because it owns the stage vocabulary the journal rows are stamped with;
# a client that worked it out from a classification string would be inferring an ending,
# which is the one thing part A forbids.
_CLASSIFICATION_STAGE = {
    "implementation_missing": "coding",
    "review_scope_failure": "coding",
    "contract_mismatch": "coding",
    "validation_source_failure": "validation",
    "validation_capacity_failure": "validation",
    "validation_configuration_failure": "setup",
    "dependency_installation_failure": "setup",
    "test_infrastructure_missing": "setup",
}

# A child that is neither queued nor executing has moved past its current attempt, so that
# attempt has an ending. The final attempt of a stopped child is therefore described, which
# matters: it is the marker case this whole item opens with.
_CHILD_EXECUTING = frozenset({ChildWorkflowStatus.PENDING, ChildWorkflowStatus.RUNNING})

# Statuses a result can carry that name no ending in the vocabulary above. Both are real
# states and neither is an ending: a contract-change wait is a human gate the attempt is
# held at, and a cancellation stopped the feature rather than judging the attempt. They are
# served as "no ending recorded", which the drawer already has honest words for.
_NOT_AN_ENDING = frozenset({"waiting_for_contract_change", "cancelled"})


@dataclass(frozen=True, slots=True)
class AttemptEnding:
    """Where one attempt ended, and the facts its two surfaces render from.

    ``stage`` is the stage name the ending landed in, in the same vocabulary the journal rows
    carry, so the drawer marks a stage without deciding which one. ``self_review_outcome``
    and ``source_repair_passes`` ride here rather than on a second response because the
    graph's sub-stage strip needs exactly them and they exist today only in completion
    metadata, on no served response at all.
    """

    attempt: int
    ended_by: AttemptEndingKind
    stage: str
    detail: str | None
    workspace: str | None
    self_review_outcome: str | None
    self_review_corrected_files: int | None
    source_repair_passes: int | None
    # How many times this attempt's unjournaled in-attempt passes had to issue their model
    # request again because the stream never spoke. The one measurement of the first-event
    # budget that no journal row can carry -- those passes write none.
    stream_reissues: int | None


class AttemptEndingCache:
    """Assembled endings, kept because an ending is immutable once its attempt is over.

    The operations endpoint is timer-polled per repository, so the assembly below must not
    run on every poll. It does not have to: an attempt the child has moved past cannot end
    differently later, so its ending is assembled once and answered from memory afterwards.

    Process-local and rebuildable on purpose -- no table, no migration, no invalidation
    rules. A restart or a second worker reassembles an immutable fact rather than losing one.
    """

    def __init__(self) -> None:
        """Start empty; the first poll of each repository pays for its own history once."""
        # Absence and a cached ``None`` are different answers, so membership is tested rather
        # than truthiness: an attempt whose status names no ending must not be reassembled on
        # every poll just because it produced nothing.
        self._endings: dict[tuple[str, str, int], AttemptEnding | None] = {}
        # How many times the assembly below actually ran. Read by the caching test, which is
        # what turns the paragraph above from a comment into a contract.
        self.assemblies = 0

    def endings_for(
        self, state: FeatureWorkflowSnapshot, repository_id: str
    ) -> list[AttemptEnding]:
        """Return one ending per finished attempt of this repository, oldest attempt first.

        The in-flight attempt is deliberately absent. It has no ending by definition, so
        assembling "wherever nothing is cached yet" would scan artifacts on every poll for
        the one attempt anybody is watching -- exactly the cost the cache exists to avoid.
        """
        child = state.child_workflows.get(repository_id)
        if child is None:
            return []
        reader = _ResultReader(state.artifacts, repository_id)
        endings: list[AttemptEnding] = []
        for attempt, result in reader.by_attempt():
            if not _has_moved_past(child, attempt):
                continue
            key = (state.feature_id, repository_id, attempt)
            if key not in self._endings:
                self.assemblies += 1
                self._endings[key] = _ending_for(reader, attempt=attempt, result=result)
            ending = self._endings[key]
            if ending is not None:
                endings.append(ending)
        return endings


def _has_moved_past(child: ChildWorkflowReference, attempt: int) -> bool:
    """Whether the child is done with this attempt: a later one exists, or it stopped."""
    return attempt < child.retry_count or child.status not in _CHILD_EXECUTING


class _ResultReader:
    """One repository's per-attempt records, indexed the three ways this module reads them."""

    def __init__(self, artifacts: Sequence[Artifact], repository_id: str) -> None:
        """Index once, so an attempt's completion and review are lookups rather than scans."""
        self._by_id: dict[str, Artifact] = {item.artifact_id: item for item in artifacts}
        self._results: dict[int, ChildWorkflowResultArtifact] = {}
        for item in artifacts:
            if not isinstance(item, ChildWorkflowResultArtifact):
                continue
            if item.repository_id == repository_id:
                self._results[_result_attempt(item)] = item

    def by_attempt(self) -> list[tuple[int, ChildWorkflowResultArtifact]]:
        """The results this repository recorded, oldest attempt first."""
        return sorted(self._results.items())

    def completion(self, result: ChildWorkflowResultArtifact) -> CodeCompletionArtifact | None:
        """The completion this attempt produced, where it produced one."""
        found = self._by_id.get(result.code_completion_artifact_id or "")
        return found if isinstance(found, CodeCompletionArtifact) else None

    def review(self, result: ChildWorkflowResultArtifact) -> ReviewArtifact | None:
        """The review this attempt received, reached through the result's own reference.

        Never by filename. A ``ReviewArtifact`` records no repository of its own, so the
        result's reference is the only thing that ties one to one attempt.
        """
        found = self._by_id.get(result.review_artifact_id or "")
        return found if isinstance(found, ReviewArtifact) else None

    def has_later_attempt(self, attempt: int) -> bool:
        """Whether this repository recorded a result for any attempt after this one."""
        return any(number > attempt for number in self._results)


def _result_attempt(result: ChildWorkflowResultArtifact) -> int:
    """The zero-based attempt one result describes, as its own metadata stamped it."""
    recorded = result.metadata.get("child_retry_count")
    return recorded if isinstance(recorded, int) and not isinstance(recorded, bool) else 0


class _Verdict(NamedTuple):
    """Which record ended the attempt, the stage it landed in, and the clause quoting it."""

    ended_by: AttemptEndingKind
    stage: str
    detail: str


def _ending_for(
    reader: _ResultReader, *, attempt: int, result: ChildWorkflowResultArtifact
) -> AttemptEnding | None:
    """Assemble one attempt's ending, or nothing where its status names none."""
    if result.status in _NOT_AN_ENDING:
        return None
    completion = reader.completion(result)
    self_review = _mapping(_metadata(completion).get("self_review"))
    corrections = self_review.get("corrections_applied")
    verdict = (
        _delivered_verdict(reader, attempt, completion)
        if result.status == "approved"
        else _failure_verdict(reader, result, self_review)
    )
    return AttemptEnding(
        attempt=attempt,
        ended_by=verdict.ended_by,
        stage=verdict.stage,
        detail=verdict.detail,
        workspace=_text(_mapping(result.metadata.get("attempt_inputs")).get("workspace")),
        # `absent` is a seventh state the records show and the code does not enumerate: 201's
        # backend attempt 4 carries no `self_review` key at all, and a cell that renders
        # nothing there reads as clean.
        self_review_outcome=_text(self_review.get("outcome")),
        # Files, not correction rounds: `corrections_applied` is a list of paths.
        self_review_corrected_files=len(corrections) if isinstance(corrections, list) else None,
        # Usually absent, and `0` on both of 201's nine backend completions that carry it. A
        # "0 repairs" badge on every healthy attempt is noise, so the reading stays faithful
        # and the surface decides what to draw.
        source_repair_passes=_integer(_metadata(completion).get("source_repair_passes")),
        # Present and `0` where passes ran on a streaming transport and every stream spoke
        # first time. Absent where nothing measured it, which a surface must not draw as
        # "none": the difference is exactly what says whether the budget is sized right.
        stream_reissues=_integer(_metadata(completion).get("stream_reissues")),
    )


def _delivered_verdict(
    reader: _ResultReader, attempt: int, completion: CodeCompletionArtifact | None
) -> _Verdict:
    """How an approved attempt ended -- never the classification it happens to still hold.

    201's backend attempts 2, 6, 7 and 8 are ``approved`` *and* carry
    ``review_scope_failure`` or ``validation_source_failure``. An assembler that read the
    classification first would render four delivered attempts as failures.
    """
    if reader.has_later_attempt(attempt):
        return _Verdict(
            AttemptEndingKind.SUPERSEDED,
            "publication",
            "passed its own review and was reopened by a later attempt",
        )
    published = _metadata(completion).get("published_after_approval") is True
    return _Verdict(
        AttemptEndingKind.PUBLISHED if published else AttemptEndingKind.APPROVED,
        "publication",
        "approved by review and published to the branch" if published else "approved by review",
    )


def _failure_verdict(
    reader: _ResultReader,
    result: ChildWorkflowResultArtifact,
    self_review: Mapping[str, Any],
) -> _Verdict:
    """Which record ended this attempt's own cycle, tested in one stated order."""
    classification = _text(result.failure_classification)
    refusal = _text(result.metadata.get("retry_refusal_reason"))
    review = reader.review(result)

    # A refusal is the attempt declining to proceed and saying why, so it outranks whatever
    # the work would otherwise have been judged on.
    if refusal is not None:
        return _Verdict(AttemptEndingKind.REFUSAL, "coding", _first_sentence(refusal))
    # The reviewer answered and its answer was not an approval. `run_reviewer` succeeding
    # means the call answered; the ✗ belongs on the stage from this record and never on the
    # row, whose own status is correct.
    if review is not None and review.verdict != "approved":
        return _Verdict(AttemptEndingKind.REVIEW_REJECTED, "review", _review_detail(review))
    # The gate before the reviewer. 201's attempt 0: `corrections_failed` against one
    # finding, no 007 artifact for the attempt at all, and the drawer's silence about it is
    # the defect this whole item exists to remove.
    outcome = _text(self_review.get("outcome"))
    if review is None and outcome in _SELF_REVIEW_STOPS:
        return _Verdict(
            AttemptEndingKind.SELF_REVIEW, "review", _self_review_detail(outcome, self_review)
        )
    if classification in _VALIDATION_FAILURE_CLASSES:
        # The command where the journaled set holds one; otherwise the classification,
        # because the gate that rejected the attempt may have been an in-attempt one that
        # nothing journals.
        detail = _failed_command(result) or _classified(classification)
        return _Verdict(AttemptEndingKind.VALIDATION_FAILED, "validation", detail)
    return _Verdict(
        AttemptEndingKind.FAULT,
        _CLASSIFICATION_STAGE.get(classification or "", "coding"),
        _classified(classification),
    )


def _classified(classification: str | None) -> str:
    """The platform's own class for why the attempt stopped, in the platform's own word."""
    if classification is None:
        return "stopped with no classification recorded"
    return f"classified as {classification.replace('_', ' ')}"


def _review_detail(review: ReviewArtifact) -> str:
    """The verdict, the finding count, and the worst blocking finding's own identifier."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    blocking = [item for item in review.findings if item.severity in {"critical", "high"}]
    considered = blocking or list(review.findings)
    verdict = review.verdict.replace("_", " ")
    clause = f"review returned {verdict} with {_count(len(review.findings), 'finding')}"
    if not considered:
        return clause
    worst = min(considered, key=lambda item: order.get(item.severity, 5))
    return f"{clause}, blocking on {worst.finding_id}"


def _self_review_detail(outcome: str | None, self_review: Mapping[str, Any]) -> str:
    """What the self-review gate said, quoting its own record rather than restating it.

    The findings it records carry a requirement and a classification rather than a finding
    id, so those are what the clause names. The blocking issue the attempt also recorded is
    deliberately not quoted here: it is a paragraph, and this clause has to fit on a
    dropdown line beside a duration.
    """
    recorded = self_review.get("findings")
    findings: list[Any] = recorded if isinstance(recorded, list) else []
    named = (outcome or "outcome not recorded").replace("_", " ")
    clause = f"self-review {named} against {_count(len(findings), 'finding')}"
    first = _mapping(findings[0]) if findings else {}
    handles = [_text(first.get("requirement_id")), _text(first.get("classification"))]
    quoted = ", ".join(item for item in handles if item)
    return f"{clause} ({quoted})" if quoted else clause


def _failed_command(result: ChildWorkflowResultArtifact) -> str | None:
    """The first failed command in the revision-bound set this attempt was judged on.

    ``current_validation_results`` rather than ``validation_results``: the first is the set
    the retry policy actually read, and it is the one bound to the revision the attempt
    ended at.
    """
    for item in result.current_validation_results:
        if not isinstance(item, Mapping) or item.get("passed") is True:
            continue
        raw = item.get("command")
        command = " ".join(str(part) for part in raw) if isinstance(raw, list) else _text(raw)
        gate = _text(item.get("validation_type"))
        exit_code = _integer(item.get("exit_code"))
        parts = [
            f"the required {gate} command failed" if gate else "a required command failed",
            f": {command}" if command else "",
            f" (exit {exit_code})" if exit_code is not None else "",
        ]
        return "".join(parts)
    return None


def _count(value: int, noun: str) -> str:
    """A count and its noun, pluralised, so no sentence reads "1 findings"."""
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


def _first_sentence(value: str | None) -> str:
    """One sentence of a recorded reason, so a paragraph does not become a dropdown line."""
    text = (value or "").strip()
    if not text:
        return ""
    head, separator, _ = text.partition(". ")
    return f"{head}." if separator else text


def _metadata(artifact: Artifact | None) -> Mapping[str, Any]:
    """One artifact's metadata, or an empty reading where the artifact is absent."""
    return artifact.metadata if artifact is not None else {}


def _mapping(value: object) -> Mapping[str, Any]:
    """A recorded mapping, or an empty one -- never a partial read of the wrong shape."""
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    """A recorded string with content, or nothing. Blank is not a value."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _integer(value: object) -> int | None:
    """A recorded integer, rejecting the bool that Python counts as one."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
