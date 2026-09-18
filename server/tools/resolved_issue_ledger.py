"""Remember which review demands were satisfied, so a later attempt cannot quietly undo one.

Everything the retry machinery already has compares an attempt against its immediate
predecessor, and that is exactly the comparison a re-litigated design decision survives:

* ``unresolved_since`` counts how many attempts running a finding has lasted, and deletes the
  counter the moment the finding disappears. Raised -> satisfied -> demanded again therefore
  restarts at one, indistinguishable from a demand nobody has ever seen.
* ``diagnostic_signature`` and its repeat rule look only at the attempt before this one.
* ``_revisits_an_earlier_attempt`` matches diff bytes, so it catches an oscillation that
  reproduces a change exactly and misses the same demand re-broken with new source.

AB-Feature-184 spent two attempts on the second shape and then hit the third: its frontend was
rejected twice for one finding, and at the integration seam the reviewer demanded compensating
cleanup for a design this repository's own review had just approved without it. Two authorities,
one question, and every attempt spent re-arguing it.

This module answers only "has this demand already been made and met", from artifacts that are
already persisted for every retry. It adds no state, calls no model, and decides nothing: it
reports what was satisfied and what has come back, and its caller decides what that means.

Identity is ``fingerprint_for_text`` throughout -- for both authorities, deliberately. A
repository blocking issue and an integration ``recommended_fix`` are both review prose, and
fingerprinting them two different ways would make the cross-authority case undetectable, which
is the one this exists for.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import NamedTuple

from pydantic import Field

from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    IntegrationReviewArtifact,
    ReviewArtifact,
    ReviewFinding,
)
from state.models import StateModel
from tools.retry_strategy import diagnostic_signature
from tools.review_fix_classification import (
    defect_identity,
    fingerprint_for_text,
    normalized_finding_text,
)


class IssueAuthority(StrEnum):
    """Which review demanded a change. There are two of them, and they can disagree."""

    REPOSITORY_REVIEW = "repository_review"
    INTEGRATION_REVIEW = "integration_review"


# How this authority is named to an engineer and to an operator. Kept beside the enum so the
# two audiences never drift apart, and phrased as the review rather than the agent: what a
# person has to arbitrate is which *review* is right, not which model wrote it.
_AUTHORITY_NAMES: dict[IssueAuthority, str] = {
    IssueAuthority.REPOSITORY_REVIEW: "this repository's review",
    IssueAuthority.INTEGRATION_REVIEW: "the integration review",
}


def authority_name(authority: IssueAuthority) -> str:
    """Name one review the way every prompt and every operator sentence already names it.

    Published rather than copied. A second table elsewhere is how "this repository's review"
    becomes "the repository reviewer" in one surface and not another, and the two audiences
    this table exists to keep together then read two different names for one authority.
    """
    return _AUTHORITY_NAMES[authority]


# The whole satisfied-invariant section's budget, and the longest single completion summary
# quoted back as the grounds for a removal. Both bound a prompt that is spliced into every
# remediation task, so neither may grow with the number of attempts a workstream has spent.
_INVARIANT_SECTION_MAX_CHARACTERS = 2_000
_GROUNDS_MAX_CHARACTERS = 400


class ResolvedIssue(StateModel):
    """A demand one review made, which a later cycle of this workstream stopped reporting."""

    fingerprint: str = Field(min_length=1)
    # The demand's own words the last time it was made. Whole, never sliced: a half-quoted
    # requirement handed back to an engineer is an instruction with its object missing.
    issue: str = Field(min_length=1)
    authority: IssueAuthority
    # Position in the feature's artifact lineage where this stopped being demanded. The two
    # authorities count their cycles separately, so their own indices are not comparable;
    # this is the one ordering both share.
    resolved_at: int = Field(ge=0)
    # How many times this demand has gone from made to absent. Two or more is the sentence
    # "the implementation removed it twice" -- a fact about the lineage, not an inference.
    removals: int = Field(ge=1, default=1)
    # What the attempt that stopped reporting it said it had done. Empty where the lineage
    # holds no completion summary for that cycle, which is every integration review: those
    # record a verdict on someone else's work and never an account of the work itself.
    grounds: str = ""


class RecurringIssue(StateModel):
    """A demand being made now that this workstream already satisfied once."""

    # The current wording, which is routinely not the wording that was satisfied. The pair is
    # what makes a recurrence legible to a person: same defect, two sentences.
    issue: str = Field(min_length=1)
    authority: IssueAuthority
    resolved: ResolvedIssue

    @property
    def cross_authority(self) -> bool:
        """Report whether a different review is now demanding what the first one let go."""
        return self.authority is not self.resolved.authority


class SettledQuestion(NamedTuple):
    """A design question a person has already decided, as the retry machinery reads it.

    Deliberately not a ``ResolvedIssue``. A resolved issue is something this module *derived*
    from two review verdicts; this is something a human *decided*, and the difference decides
    behaviour: a resolved issue coming back stops the loop, and a settled question coming back
    may not, because stopping again would ask the same person the same question.
    """

    fingerprint: str
    verdict: str
    # The decision in the decider's own words. Runs 193 and 194 showed that a clarification
    # answer which encodes the implementation strategy is what precedes a delivery, so this is
    # handed to the engineer verbatim rather than summarised into a policy.
    decision: str
    demand: str
    authority: IssueAuthority
    decided_by: str
    # Which conflict artifact holds the decision, and when it was made. Carried so the 73-
    # demotion can stamp its audit trace -- which finding, which verdict, whose decision,
    # when -- without a second lookup that could disagree with this one. Defaulted, because
    # neither field changes what a settled question *does*; they only let a record say so.
    conflict_id: str = ""
    decided_at: datetime | None = None


class _Cycle(NamedTuple):
    """One review verdict in the feature's lineage, as this module needs to read it."""

    authority: IssueAuthority
    position: int
    issues: tuple[str, ...]
    grounds: str


def _integration_demands(review: IntegrationReviewArtifact, repository_id: str) -> tuple[str, ...]:
    """Return what one integration review asked of one repository.

    Only a ``changes_requested`` review demands anything -- the same rule
    ``_outstanding_integration_fixes`` applies, and for the same reason. An approved review is
    therefore an empty demand set, which is exactly right here: it is what makes "the
    integration review stopped asking for this" a fact the ledger can record.
    """
    if review.review_status != "changes_requested":
        return ()
    return tuple(
        item.recommended_fix
        for item in review.cross_repository_findings
        if item.responsible_repository_id == repository_id and item.recommended_fix
    )


def _review_cycles(
    artifacts: Sequence[object], *, child_workflow_id: str, repository_id: str
) -> list[_Cycle]:
    """Read both authorities' verdicts out of the artifact lineage, in the order they landed.

    Artifacts are appended as they are produced, so their position in this list is the feature's
    own chronology -- the only clock the two authorities share.

    Only attempts in which a review actually spoke count as repository-review cycles: a child
    result carrying the ``deterministic_gate`` marker, or naming no review artifact, is an
    attempt a gate stopped before any review ran. A gate that failed an attempt has not
    withdrawn a demand -- it has prevented one from being made, and absence of evidence is not
    the evidence the ledger needs. AB-Feature-216 was ended on exactly that misreading: a
    demand raised at one review, absent from a review-less reachability-gate failure, was
    recorded as satisfied, and its fingerprint's return one review later was read as a
    reversal. ``recurring_demand_theme`` below has always had this fence; this is the same one.
    """
    summaries = {
        artifact.artifact_id: artifact.summary
        for artifact in artifacts
        if isinstance(artifact, CodeCompletionArtifact)
    }
    cycles: list[_Cycle] = []
    for position, artifact in enumerate(artifacts):
        if (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.child_workflow_id == child_workflow_id
            and not artifact.metadata.get("deterministic_gate")
            and artifact.review_artifact_id
        ):
            # A stopped attempt carries the platform's own question to an operator in this
            # list, because that is what reaches the console. It is not something a review
            # demanded and not something an engineer can satisfy, so it must never come back
            # as either an invariant to preserve or a demand that recurred.
            question = artifact.metadata.get("operator_question")
            cycles.append(
                _Cycle(
                    authority=IssueAuthority.REPOSITORY_REVIEW,
                    position=position,
                    issues=tuple(issue for issue in artifact.blocking_issues if issue != question),
                    grounds=summaries.get(artifact.code_completion_artifact_id or "", ""),
                )
            )
        elif isinstance(artifact, IntegrationReviewArtifact):
            cycles.append(
                _Cycle(
                    authority=IssueAuthority.INTEGRATION_REVIEW,
                    position=position,
                    issues=_integration_demands(artifact, repository_id),
                    grounds="",
                )
            )
    return cycles


def resolved_issue_ledger(
    artifacts: Sequence[object], *, child_workflow_id: str, repository_id: str
) -> list[ResolvedIssue]:
    """Return the demands this workstream has already satisfied, most recently satisfied first.

    A demand is satisfied when the authority that made it made it again without it -- not when
    an engineer says it fixed something. The evidence is the next verdict from the same review,
    which is why the two authorities are walked separately: an integration review saying nothing
    about a repository finding is silence, not agreement.

    Derived entirely from persisted artifacts, so it survives a crash, a resume and the
    integration-fix re-entry that starts a fresh child loop with an empty ``unresolved_since``.
    """
    cycles = _review_cycles(
        artifacts, child_workflow_id=child_workflow_id, repository_id=repository_id
    )
    resolved: dict[str, ResolvedIssue] = {}
    for authority in IssueAuthority:
        ordered = [cycle for cycle in cycles if cycle.authority is authority]
        outstanding: dict[str, str] = {}
        for previous, current in zip(ordered, ordered[1:], strict=False):
            for issue in previous.issues:
                outstanding[fingerprint_for_text(issue)] = issue
            present = {fingerprint_for_text(issue) for issue in current.issues}
            for fingerprint, issue in list(outstanding.items()):
                if fingerprint in present:
                    continue
                del outstanding[fingerprint]
                earlier = resolved.get(fingerprint)
                # Later resolutions replace earlier ones, and the removal count accumulates
                # across both. A demand removed, re-made and removed again is one entry that
                # knows it happened twice -- which is the difference between "changed its mind"
                # and "keeps reversing this decision".
                if earlier is not None and earlier.resolved_at > current.position:
                    continue
                resolved[fingerprint] = ResolvedIssue(
                    fingerprint=fingerprint,
                    issue=issue,
                    authority=authority,
                    resolved_at=current.position,
                    removals=(earlier.removals + 1) if earlier is not None else 1,
                    grounds=current.grounds[:_GROUNDS_MAX_CHARACTERS],
                )
    return sorted(resolved.values(), key=lambda entry: entry.resolved_at, reverse=True)


# The prose agreement a lone result-code identity needs before it may report a recurrence,
# as Jaccard overlap of the two sentences' normalized token sets. Calibration: AB-Feature-216's
# two findings -- different demands about one endpoint -- share the code token and measure
# 0.172 (44 and 58 tokens, 15 shared); a verbatim or near-verbatim repeat measures at or near
# 1.0. No second signature token can break the tie, because fingerprint equality already means
# the two texts' whole signatures are identical.
_CODE_ONLY_PROSE_OVERLAP_MINIMUM = 0.5


def _corroborated(resolved_issue: str, demanded_issue: str) -> bool:
    """Say whether a fingerprint match is one defect twice, or two defects naming one code.

    In a contract-driven feature every finding about an endpoint names that endpoint's error
    codes, so an identity that is exactly one ``code:`` token is the normal collision case,
    not a rare one -- AB-Feature-216's missing-test-coverage demand and its failure-path
    demand were one fingerprint that way. Such a match must also agree in prose. Any other
    identity shape -- a location, an exception, a lint rule, several tokens, or normalized
    prose itself -- already says which defect it is, and behaves exactly as before.
    """
    tokens = diagnostic_signature([demanded_issue])
    if len(tokens) != 1 or not tokens[0].startswith("code:"):
        return True
    resolved_tokens = set(normalized_finding_text(resolved_issue).split())
    demanded_tokens = set(normalized_finding_text(demanded_issue).split())
    union = resolved_tokens | demanded_tokens
    if not union:
        return True
    overlap = len(resolved_tokens & demanded_tokens) / len(union)
    return overlap >= _CODE_ONLY_PROSE_OVERLAP_MINIMUM


def recurring_resolved_issues(
    ledger: Sequence[ResolvedIssue],
    *,
    demanded: Mapping[IssueAuthority, Sequence[str]],
    settled: Collection[str] = (),
) -> list[RecurringIssue]:
    """Return the demands being made right now that this workstream already satisfied.

    Cross-authority recurrences are ordered first. Both stop the loop, but only one of them
    names a disagreement between two reviews, and that is the sentence a person needs to read
    before anything else.

    A match whose shared identity is a single result-code token is reported only when the two
    sentences also agree in prose (``_corroborated``); the identity functions themselves are
    untouched, so the 48- replay guard, ``unchanged_failure``, the convergence rules and the
    settled-question binding keep their meaning.

    ``settled`` is the fingerprints of questions a person has already decided. Those are
    excluded, and that exclusion is what makes the stop answerable rather than merely terminal:
    without it the attempt a verdict authorises would meet the same recurrence and stop again
    before the engineer was called, so answering the question would change nothing. It is not a
    weakening -- a settled question is one this platform has already put to the owner the
    evidence names, and stopping on it twice asks the same person the same thing.
    """
    settled_fingerprints = set(settled)
    by_fingerprint = {
        entry.fingerprint: entry
        for entry in ledger
        if entry.fingerprint not in settled_fingerprints
    }
    recurrences: list[RecurringIssue] = []
    seen: set[str] = set()
    for authority, issues in demanded.items():
        for issue in issues:
            fingerprint = fingerprint_for_text(issue)
            entry = by_fingerprint.get(fingerprint)
            if entry is None or fingerprint in seen:
                continue
            if not _corroborated(entry.issue, issue):
                continue
            seen.add(fingerprint)
            recurrences.append(RecurringIssue(issue=issue, authority=authority, resolved=entry))
    return sorted(recurrences, key=lambda item: not item.cross_authority)


def settled_question_lines(settled: Sequence[SettledQuestion]) -> list[str]:
    """Render each decided design question as the instruction the next attempt is bound by.

    This is the load-bearing half of making the stop answerable. The verdict is spliced in
    ahead of the derived invariants and is *not* filtered against what a review is demanding
    right now -- which is the opposite of the rule the satisfied-invariant section follows, and
    deliberately so. An ordinary satisfied requirement that is also outstanding is a
    contradiction to keep out of a prompt; a verdict about an outstanding demand is the one
    sentence that resolves it, and leaving it out would hand the engineer the argument again.

    Whole decisions only, quoted as the person wrote them. A verdict paraphrased into a policy
    is the platform re-litigating the answer it was given.
    """
    lines: list[str] = []
    for entry in settled:
        stance = (
            "This requirement stands."
            if entry.verdict == "requirement_holds"
            else "This requirement does not stand and must not be implemented."
        )
        lines.append(
            "SETTLED BY A PERSON -- a human owner of this feature was asked whether this "
            f"design requirement holds, and decided: {stance} That decision is final for this "
            "feature and is not open to review by this attempt. Do not argue it, do not "
            "reverse it, and if a review raises it again, say so in the completion summary "
            f"and leave the decision in place. The requirement in question: {entry.demand} "
            f"The decision, in the words of the person who made it: {entry.decision}"
        )
    return lines


def satisfied_invariant_lines(
    ledger: Sequence[ResolvedIssue],
    *,
    demanded: Sequence[str] = (),
    settled: Collection[str] = (),
) -> list[str]:
    """Render the ledger as feedback lines an engineer cannot mistake for outstanding work.

    Every line stands alone, carrying its own status and its own authority. That is not style:
    these are spliced into individual task descriptions, so a heading placed above the group
    would be a heading some tasks never receive -- and a satisfied requirement read as an open
    one is an instruction to do the work twice.

    Whole issues only. An entry that does not fit the remaining budget is dropped rather than
    sliced, and the count of dropped entries is stated, because a silently halved list reads as
    a complete one.

    ``demanded`` is whatever a review is asking for right now, and anything matching it is left
    out entirely. A demand that is both outstanding and satisfied is a recurrence, which is the
    caller's to stop; handing an engineer "fix this" and "do not undo this" about one defect
    would be the contradiction this section exists to prevent.

    ``settled`` is left out for a different reason: a question a person has decided is already
    stated, in their words, by ``settled_question_lines``. Repeating it here as a derived
    invariant would say the same thing twice with less authority behind it, and where the
    verdict overruled the demand the two lines would flatly contradict each other.
    """
    outstanding = {fingerprint_for_text(issue) for issue in demanded} | set(settled)
    lines: list[str] = []
    characters = 0
    omitted = 0
    for entry in ledger:
        if entry.fingerprint in outstanding:
            continue
        line = (
            f"ALREADY SATISFIED -- {_AUTHORITY_NAMES[entry.authority]} required this and an "
            "earlier attempt of this workstream delivered it. Do not undo it. If you believe "
            "it is wrong, say so in the completion summary instead of reverting it: "
            f"{entry.issue}"
        )
        if characters + len(line) > _INVARIANT_SECTION_MAX_CHARACTERS and lines:
            omitted += 1
            continue
        lines.append(line)
        characters += len(line)
    if omitted:
        lines.append(
            f"{omitted} further requirement(s) this workstream has already satisfied were left "
            "out of the list above to keep it bounded; the most recently satisfied are shown. "
            "The same rule applies to them: do not undo work an earlier attempt delivered."
        )
    return lines


# How many review cycles must make a same-theme blocking demand before the loop treats it as
# a design question a person owns (77-, item 35). Three, matching the repeated-signature and
# capability-finding allowances and for the same reason: twice can be a model that read its
# feedback badly once; a third round against changed source is a pattern. Calibration from
# the charter: over-grouping costs one early human question; under-grouping is AB-Feature-208
# burning six same-theme rounds to the resubmission stop without a human ever being asked.
SAME_THEME_BLOCKING_REJECTIONS = 3


class DemandWording(StateModel):
    """One round's wording of a recurring demand, quoted whole with the attempt it blocked."""

    attempt: int | None = None
    issue: str = Field(min_length=1)


class RecurringDemandTheme(StateModel):
    """A demand blocking its third review cycle under a wording no exact key could match.

    ``wordings`` holds every round's phrasing in lineage order, the current one last -- the
    drift is exactly what the person deciding it needs to see. The theme key that grouped
    them is deliberately not carried: it is derived fresh from the lineage every cycle and
    never stored (the 49- rule), and nothing downstream may bind a verdict through it (73-).
    """

    issue: str = Field(min_length=1)
    authority: IssueAuthority
    wordings: list[DemandWording] = Field(min_length=1)


def demand_theme(finding: ReviewFinding) -> str:
    """Reduce one review finding to the theme it keeps demanding, however it is worded.

    Anchored on the structured fields a reviewer keeps stable while it rewords its prose:
    the requirement it derives from and the file it names. AB-Feature-208's four wordings of
    one demand share both anchors and not one prose token, so any prose component in an
    anchored key would un-group exactly the repetition this exists to see. Prose
    participates only where no anchor exists -- `defect_identity` over the finding's own
    words, which groups nothing looser than a rewording that normalizes identically.

    Files come from the structured ``file_path`` alone, never extracted from prose: the
    lint-rule pattern's documented slash-token trap means a path fragment in a sentence is
    not reliably a path.
    """
    requirement = finding.requirement_id or ""
    file_path = finding.file_path or ""
    if requirement or file_path:
        return f"anchored:{requirement}|{file_path}"
    return f"prose:{defect_identity(finding.title, finding.description)}"


def _themed_cycle_findings(
    result: ChildWorkflowResultArtifact, review: ReviewArtifact
) -> dict[str, str]:
    """Return theme -> wording for the review judgements that actually blocked one attempt.

    Three fences, each against a sentence no theme may group (item 30's trap: an
    infrastructure sentence read as a review demand raises a conflict about a measurement,
    which 73- correctly forbids overruling):

    * only findings whose description is byte-equal to one of the result's blocking issues
      -- the demands that failed the attempt, not advisory remarks;
    * never the platform's own operator question;
    * never a finding the review metadata names in ``deterministic_finding_ids`` -- the
      exact present-minus-model-supplied census the 73- demotion already trusts to refuse
      overruling a measurement.
    """
    deterministic = set(review.metadata.get("deterministic_finding_ids") or ())
    question = result.metadata.get("operator_question")
    blocked = {issue for issue in result.blocking_issues if issue != question}
    themes: dict[str, str] = {}
    for finding in review.findings:
        if finding.finding_id in deterministic or finding.description not in blocked:
            continue
        themes.setdefault(demand_theme(finding), finding.description)
    return themes


def recurring_demand_theme(
    artifacts: Sequence[object],
    *,
    child_workflow_id: str,
    current_review: ReviewArtifact,
    current_result: ChildWorkflowResultArtifact,
    current_attempt: int | None = None,
    settled: Collection[str] = (),
) -> RecurringDemandTheme | None:
    """Return the demand now blocking its third same-theme review cycle, if there is one.

    Derived entirely from persisted review and result artifacts plus the attempt in hand, so
    a crash, a resume, or a rewording changes nothing: the wordings are in the lineage and
    the theme is recomputed from them every cycle -- never stored.

    Cycles are counted distinct, not consecutive: an A -> B -> A alternation is exactly the
    shape a consecutive counter misses (the 49-C lesson). Eligible cycles are review
    verdicts only -- a child result that names a ``ReviewArtifact`` and does not carry the
    ``deterministic_gate`` marker -- so a preflight or install failure can never contribute
    a theme however often it repeats.

    ``settled`` is the exact fingerprints of questions a person has decided. A theme any of
    whose wordings fingerprints into that set is not asked again -- exclusion by the exact
    key of a real sentence, never by the theme itself, so the verdict's binding stays
    exactly as narrow as 73- made it.
    """
    reviews = {
        artifact.artifact_id: artifact
        for artifact in artifacts
        if isinstance(artifact, ReviewArtifact)
    }
    cycles: list[tuple[int | None, dict[str, str]]] = []
    for artifact in artifacts:
        if (
            not isinstance(artifact, ChildWorkflowResultArtifact)
            or artifact.child_workflow_id != child_workflow_id
            or artifact.metadata.get("deterministic_gate")
        ):
            continue
        review = reviews.get(artifact.review_artifact_id or "")
        if review is None:
            continue
        attempt = artifact.metadata.get("child_retry_count")
        cycles.append(
            (
                attempt if isinstance(attempt, int) else None,
                _themed_cycle_findings(artifact, review),
            )
        )
    current_themes = _themed_cycle_findings(current_result, current_review)
    cycles.append((current_attempt, current_themes))
    settled_fingerprints = set(settled)
    best: RecurringDemandTheme | None = None
    best_rounds = 0
    for theme, current_wording in current_themes.items():
        wordings = [
            DemandWording(attempt=attempt, issue=themed[theme])
            for attempt, themed in cycles
            if theme in themed
        ]
        if len(wordings) < SAME_THEME_BLOCKING_REJECTIONS:
            continue
        if any(fingerprint_for_text(wording.issue) in settled_fingerprints for wording in wordings):
            continue
        if len(wordings) > best_rounds:
            best_rounds = len(wordings)
            best = RecurringDemandTheme(
                issue=current_wording,
                authority=IssueAuthority.REPOSITORY_REVIEW,
                wordings=wordings,
            )
    return best


def recurring_demand_narrative(theme: RecurringDemandTheme) -> tuple[str, list[str]]:
    """State the recurring demand as the question a person now owns, quoting every wording.

    Deliberately not the reversal narrative: nothing here was "already made and then
    removed" -- the demand was never satisfied at all -- and it is not a defect or a
    capability sentence either. What a person arbitrates is one requirement asked for N
    ways, so every wording travels whole, in lineage order, with the attempt it blocked.
    """
    rounds = len(theme.wordings)
    question = (
        f"{_AUTHORITY_NAMES[theme.authority].capitalize()} has blocked {rounds} attempts of "
        "this workstream on the same underlying demand, reworded each round, and no attempt "
        "has satisfied it. Repetition in new words is a requirement being re-asserted, not "
        "a defect being repaired, and further attempts would keep failing the same test. "
        "Should the requirement stand as demanded, or is the review wrong to keep "
        "requiring it?"
    )
    evidence = [
        (
            f"As demanded on attempt {wording.attempt}: {wording.issue}"
            if wording.attempt is not None
            else f"As demanded: {wording.issue}"
        )
        for wording in theme.wordings
    ]
    evidence.append(
        "Each round's wording differed, so no exact-text guard could see the repetition."
    )
    return question, evidence


# Wording by which a completion summary says it took something out. The grounds quoted in a
# reversal narrative are that summary verbatim, and quoting one that never claims a removal
# tells the operator the attempt removed a thing its own words say it added. Conservative and
# stem-shaped on purpose: a false suppression costs one explanatory clause, a false quote
# sends a person to arbitrate a story the evidence contradicts.
_REMOVAL_WORDING_STEMS = ("remov", "delet", "revert", "undo", "undid", "dropp")


def _grounds_support_a_removal(grounds: str) -> bool:
    """Say whether the quoted summary contains any removal-shaped wording at all."""
    lowered = grounds.lower()
    return any(stem in lowered for stem in _REMOVAL_WORDING_STEMS)


def design_conflict_narrative(recurrence: RecurringIssue) -> tuple[str, list[str]]:
    """State the conflict a person now owns, and the lineage facts that establish it.

    Two shapes, because they send that person to two different decisions. One review reversing
    itself is a contract question inside this repository; two reviews disagreeing is a question
    about which of them holds. Neither is a defect and neither is a capability limit, so neither
    sentence may borrow that vocabulary -- an operator sent to repair a repository over a design
    disagreement finds nothing broken.

    The "on the grounds" clause is quoted only where the grounds contain removal-shaped wording
    at all. This is a belt, not the fix: the ledger's review fence is what prevents a
    review-less cycle from minting grounds in the first place, and a summary can still describe
    removing something other than the demand -- AB-Feature-216's did -- which no keyword can see.
    """
    resolved = recurrence.resolved
    evidence = [
        f"{_AUTHORITY_NAMES[resolved.authority]} raised this and later stopped raising it.",
        f"Demanded again now by {_AUTHORITY_NAMES[recurrence.authority]}: {recurrence.issue}",
        f"As it was worded when it was satisfied: {resolved.issue}",
    ]
    if resolved.grounds:
        evidence.append(f"The attempt that removed it reported: {resolved.grounds}")
    if recurrence.cross_authority:
        question = (
            f"{_AUTHORITY_NAMES[resolved.authority].capitalize()} approved what "
            f"{_AUTHORITY_NAMES[recurrence.authority]} rejects; the two authorities disagree "
            "and retrying cannot settle it. This workstream already built what is being asked "
            "for and the other review let it go, so another attempt would re-argue the "
            "question rather than answer it. Which authority's position holds for this "
            "feature?"
        )
        return question, evidence
    removals = "twice" if resolved.removals >= 2 else "once"
    grounds = (
        f", most recently on the grounds: {resolved.grounds}"
        if resolved.grounds and _grounds_support_a_removal(resolved.grounds)
        else ""
    )
    question = (
        f"{_AUTHORITY_NAMES[resolved.authority].capitalize()} requires a change this "
        f"workstream already made and then removed {removals}{grounds}. That is a design "
        "decision being re-litigated, not a defect being repaired, and another attempt would "
        "reverse it again. Should the change stand, or is the review wrong to require it?"
    )
    return question, evidence


__all__ = [
    "SAME_THEME_BLOCKING_REJECTIONS",
    "DemandWording",
    "IssueAuthority",
    "RecurringDemandTheme",
    "RecurringIssue",
    "ResolvedIssue",
    "SettledQuestion",
    "authority_name",
    "demand_theme",
    "design_conflict_narrative",
    "recurring_demand_narrative",
    "recurring_demand_theme",
    "recurring_resolved_issues",
    "resolved_issue_ledger",
    "satisfied_invariant_lines",
    "settled_question_lines",
]
