"""A re-litigated design decision as a domain concept: what it is, and what may be decided.

49- Part B established the detection. When one review demands a change, a later cycle of that
review stops demanding it, and it is demanded again -- by the same review or by the other one --
the workstream stops, because another attempt would re-argue the question rather than answer it.
That stop is correct and stays: nothing here weakens it.

What it lacked is an exit. The stop's whole content was a sentence in `blocking_issues` and a
terminal cause, so the only things a person could do with a design question were cancel the
feature or buy attempts and hope the argument came out the other way. This module makes it an
answerable question instead: the platform writes down the two positions as they were actually
worded, somebody says which one holds and why, and the verdict becomes an invariant the next
attempt is bound by rather than a preference it may reverse.

Two rules shape everything here:

*The verdict is the human's, and only the human's.* No model call decides a conflict, ranks the
two positions, or suggests an answer -- 49-'s "arbitrating who is right is explicitly out of
scope" applies to this surface too. The platform states the question and records the answer.

*An answered question is settled, not re-opened.* The verdict is looked up by the same
fingerprint the ledger compares demands on, from the same artifact lineage, so the recurrence
stop cannot fire twice on one question and the next engineer is told the decision as a fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from artifacts.schemas import Artifact, DesignConflictArtifact
from tools.resolved_issue_ledger import (
    IssueAuthority,
    RecurringDemandTheme,
    RecurringIssue,
    SettledQuestion,
    authority_name,
)
from tools.review_fix_classification import fingerprint_for_text

# What a verdict may say. `requirement_holds` keeps the demand: the next attempt implements it
# and may not remove it again. `removal_holds` overrules the review: the next attempt must not
# implement it, and must say so rather than quietly complying.
VERDICTS = ("requirement_holds", "removal_holds")


class DesignConflictNotFoundError(LookupError):
    """Raised when a conflict identifier does not belong to this feature."""


def current_design_conflicts(
    artifacts: Sequence[Artifact],
) -> dict[str, DesignConflictArtifact]:
    """Return the newest artifact for each conflict, which is that conflict's current state.

    Artifact history is append-only: a decision is recorded by appending a revision carrying
    the same ``conflict_id``, never by editing what somebody read before answering it.
    """
    latest: dict[str, DesignConflictArtifact] = {}
    for artifact in artifacts:
        if isinstance(artifact, DesignConflictArtifact):
            latest[artifact.conflict_id] = artifact
    return latest


def find_design_conflict(artifacts: Sequence[Artifact], conflict_id: str) -> DesignConflictArtifact:
    """Return one conflict's current state, or say which identifier was not found."""
    conflict = current_design_conflicts(artifacts).get(conflict_id)
    if conflict is None:
        msg = f"design conflict not found: {conflict_id}"
        raise DesignConflictNotFoundError(msg)
    return conflict


def open_design_conflicts(
    artifacts: Sequence[Artifact], *, repository_id: str | None = None
) -> list[DesignConflictArtifact]:
    """Return the questions nobody has answered yet, oldest first.

    Identified by repository id and never by role: a feature may contain any number of
    repositories, and two of them can be stopped on a design question at the same time.
    """
    return [
        item
        for item in current_design_conflicts(artifacts).values()
        if item.status == "open" and (repository_id is None or item.repository_id == repository_id)
    ]


def answered_design_conflicts(
    artifacts: Sequence[Artifact], *, repository_id: str
) -> list[DesignConflictArtifact]:
    """Return the questions a person has decided about one repository's work.

    Scoped by repository and deliberately not by child workflow id. An integration-fix re-entry
    starts a fresh child loop under a new id, and a verdict that stopped applying the moment
    the loop restarted would be a decision the platform forgot at exactly the point the
    argument was about to resume -- which is the seam AB-Feature-184 died at.
    """
    return [
        item
        for item in current_design_conflicts(artifacts).values()
        if item.status == "answered" and item.repository_id == repository_id
    ]


def settled_questions(
    artifacts: Sequence[Artifact], *, repository_id: str
) -> list[SettledQuestion]:
    """Project the answered conflicts onto what the ledger, the retry prompt and review read.

    This is the lookup the whole feature turns on. `recurring_resolved_issues` takes the
    fingerprints as the demands it may no longer stop on, `satisfied_invariant_lines` takes
    the same entries as the invariants the next engineer is handed, and the review outcome
    takes the `removal_holds` entries as the demands a review may no longer block on -- so one
    record decides "this question is settled", "here is what was decided", and "this demand
    cannot fail an attempt", and the three cannot drift.
    """
    return [
        SettledQuestion(
            fingerprint=item.fingerprint,
            verdict=item.verdict or "",
            decision=item.decision,
            demand=item.demand,
            authority=IssueAuthority(item.demanded_by),
            decided_by=item.decided_by or "",
            conflict_id=item.conflict_id,
            decided_at=item.decided_at,
        )
        for item in answered_design_conflicts(artifacts, repository_id=repository_id)
        if item.verdict is not None
    ]


def stop_statement(conflict: DesignConflictArtifact) -> str:
    """Restate one stop as a fact about the last attempt rather than a question to an operator.

    The stop's own sentence is written for the person who has to arbitrate: it lays out both
    positions and ends by asking which of them holds. An engineer handed that sentence as
    outstanding work reads the question as its instruction and answers it -- by picking a side,
    which is the one thing no attempt is allowed to do.

    So the fact survives and the question does not. Why the last attempt ended, which review
    is on which side, and the requirement it is about all stay, because an attempt that does
    not know its workstream stopped over a re-litigated decision is free to re-litigate it
    again. What goes is the interrogative and everything addressed to whoever answers it.
    """
    if conflict.kind == "recurring_demand":
        positions = (
            f"{authority_name(IssueAuthority(conflict.demanded_by)).capitalize()} kept "
            "blocking this workstream on one demand, reworded each round and never satisfied"
        )
    elif conflict.cross_authority:
        positions = (
            f"{authority_name(IssueAuthority(conflict.satisfied_under)).capitalize()} approved "
            f"what {authority_name(IssueAuthority(conflict.demanded_by))} rejects, so the two "
            "authorities disagree about it"
            if conflict.satisfied_under is not None
            else "the two authorities disagree about it"
        )
    else:
        removals = "twice" if conflict.removals >= 2 else "once"
        positions = (
            f"{authority_name(IssueAuthority(conflict.demanded_by)).capitalize()} requires a "
            f"change this workstream already made and then removed {removals}"
        )
    return (
        "STOPPED OVER A DESIGN CONFLICT -- a statement of record about why the previous attempt "
        f"of this workstream ended, and not work assigned to this one. {positions}. Another "
        "attempt would have reversed that decision again rather than answered it, so the "
        "workstream was stopped and the decision was put to a person. The requirement it is "
        f"about: {conflict.demand} Which position holds is not this attempt's to decide or to "
        "argue; where a person has decided it, their decision reaches you separately, in their "
        "own words."
    )


def restated_for_the_next_attempt(
    issues: Sequence[str], artifacts: Sequence[Artifact], *, repository_id: str
) -> list[str]:
    """Return one repository's inherited diagnostics with any stop question restated as a fact.

    The stop writes its question into ``blocking_issues`` because that list is what reaches the
    console, and the same list is what the next attempt is handed as outstanding work. The
    verdict path already removes it there, where a decision has answered it. An ordinary retry
    grant answers nothing, so it inherited the question verbatim and handed the coding model the
    platform's own request to an operator to arbitrate two reviews.

    Substantive findings are what the granted attempt exists to answer, so identification is by
    byte-equality against the question this platform recorded for this repository and nothing
    else -- no prose heuristic, no interrogative sniffing. A line that is not that exact
    sentence is passed through unchanged, question mark or no question mark, because a review
    finding phrased as a question is still a review finding.

    Restated in place: a statement of record where the question stood, so the order the rest of
    the diagnostics arrive in does not move. Idempotent, because the statement is not itself
    any conflict's question -- the next inheritance finds nothing left to restate.
    """
    restatement = {
        item.question: stop_statement(item)
        for item in current_design_conflicts(artifacts).values()
        if item.repository_id == repository_id
    }
    return [restatement.get(issue, issue) for issue in issues]


def conflict_payload(
    recurrence: RecurringIssue,
    *,
    feature_id: str,
    repository_id: str,
    child_workflow_id: str,
    question: str,
    evidence: Sequence[str],
    attempts_spent: int,
) -> dict[str, object]:
    """Write down one recurrence as the question a person is being asked.

    Both positions travel as they were actually worded. A conflict reduced to the current
    demand is unanswerable: what makes it a decision rather than a defect is that the same
    thing was required, delivered, and then let go, and only the pair shows that.
    """
    resolved = recurrence.resolved
    return {
        "conflict_id": f"conflict-{repository_id}-{recurrence.resolved.fingerprint}",
        "feature_id": feature_id,
        "repository_id": repository_id,
        "child_workflow_id": child_workflow_id,
        "fingerprint": resolved.fingerprint,
        "question": question,
        "evidence": list(evidence),
        "demanded_by": recurrence.authority.value,
        "demand": recurrence.issue,
        "satisfied_under": resolved.authority.value,
        "satisfied_demand": resolved.issue,
        "grounds": resolved.grounds,
        "removals": resolved.removals,
        "attempts_spent": attempts_spent,
        "cross_authority": recurrence.cross_authority,
        "status": "open",
        "verdict": None,
        "decision": "",
        "decided_by": None,
        "decided_at": None,
    }


def recurring_demand_payload(
    theme: RecurringDemandTheme,
    *,
    feature_id: str,
    repository_id: str,
    child_workflow_id: str,
    question: str,
    evidence: Sequence[str],
    attempts_spent: int,
) -> dict[str, object]:
    """Write down one recurring demand as the question a person is being asked (77-, item 35).

    No satisfied position exists -- the demand was never met -- so the second position the
    reversal shape records is absent by construction, and the wordings travel in the
    evidence. Identity is the theme's **earliest** wording's exact fingerprint: a real
    sentence's key, stable across later rewordings (so a re-stop after another reworded
    round finds the same conflict instead of filing a second form), and the key the verdict
    then binds through -- never the theme itself (73-, nothing looser).
    """
    first = theme.wordings[0].issue
    fingerprint = fingerprint_for_text(first)
    return {
        "conflict_id": f"conflict-{repository_id}-{fingerprint}",
        "feature_id": feature_id,
        "repository_id": repository_id,
        "child_workflow_id": child_workflow_id,
        "kind": "recurring_demand",
        "fingerprint": fingerprint,
        "question": question,
        "evidence": list(evidence),
        "demanded_by": theme.authority.value,
        "demand": theme.issue,
        "satisfied_under": None,
        "satisfied_demand": None,
        "grounds": "",
        "removals": 0,
        "attempts_spent": attempts_spent,
        "cross_authority": False,
        "status": "open",
        "verdict": None,
        "decision": "",
        "decided_by": None,
        "decided_at": None,
    }


def answered_payload(
    *, verdict: str, decision: str, decided_by: str, decided_at: datetime
) -> dict[str, object]:
    """Return the fields that turn an open question into a decided one."""
    return {
        "status": "answered",
        "verdict": verdict,
        "decision": decision,
        "decided_by": decided_by,
        "decided_at": decided_at,
    }


__all__ = [
    "VERDICTS",
    "DesignConflictNotFoundError",
    "answered_design_conflicts",
    "answered_payload",
    "conflict_payload",
    "current_design_conflicts",
    "find_design_conflict",
    "open_design_conflicts",
    "recurring_demand_payload",
    "restated_for_the_next_attempt",
    "settled_questions",
    "stop_statement",
]
