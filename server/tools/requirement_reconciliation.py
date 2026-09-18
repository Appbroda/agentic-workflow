"""What a clarification answer does to the requirement text it settles.

An answer used to land in the Technical PRD's *metadata* and nowhere else. The requirement
descriptions, dependencies and acceptance criteria -- the text the planner scopes and the
reviewer enforces -- kept saying whatever they said before the human answered, so a feature
whose answer overruled a requirement ran with two specifications that disagreed.
AB-Feature-200 is the record: its final PRD still demanded transactional rollback and
"repository tests cover rollback behavior" while its own execution plan, derived from the
same answer, said Mongo documents are compensated *without* transactions. The engineer built
a hybrid of both and the self-review correctly proved the hybrid incoherent, four attempts
running, until the no-meaningful-change guard stopped the workstream.

This module is the deterministic half of the fix: the shape one reconciliation model call
must answer in, every structural constraint that answer has to satisfy, and the mechanics of
applying it. It holds no prompt and makes no call -- the product manager owns the PRD's
authorship, so the call is that agent's -- and it is deliberately free of anything
repository-specific, exactly as `acceptance_criteria` is.

Three constraints are enforced by *construction* rather than by inspecting model output,
because a check that can be got wrong on a live run is worse than a shape that cannot express
the mistake:

* **Nothing is added or removed.** The response echoes the complete requirement id list, in
  order, and any disagreement with the PRD's own list is refused. That is the only place a
  drop, a rename or an invention can appear at all.
* **Priorities do not change.** Priority is not part of the response. It is context the model
  reads and never something it returns.
* **A requirement no answer addresses is byte-identical.** Untouched requirements are copied
  from the existing artifact, not re-read from the response, so drift in an echo cannot reach
  the PRD even if a model produces it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from artifacts.schemas import ClarificationQuestion, Requirement

# The exact key set one reconciliation response is allowed to carry. `parse_model_json`
# compares it as a set, so an extra key is a rejection rather than something silently ignored.
RECONCILIATION_RESPONSE_KEYS = (
    "requirement_ids",
    "changed_requirements",
    "unsettled_contradictions",
)

# Every question this module writes carries it. `resume` matches answers against question ids
# exactly and the console renders whatever list it is given, so the prefix is load-bearing for
# neither -- it is how an operator, and the platform's own already-asked filter, tell a
# reconciliation question apart from the product manager's, from the checkouts' (`recon-`) and
# from the unverifiable-criteria gate's (`criteria-`).
QUESTION_ID_PREFIX = "reconcile-"

# How much of an identifier a rejection message may echo back out of the response.
_MAX_ECHOED_CHARACTERS = 80

_SLUG_CHARACTERS = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_CHARACTERS = 56


class ReconciliationRejected(ValueError):
    """A reconciliation response that may not become a PRD revision, and why.

    A plain `ValueError` subclass on purpose: this module sits under `tools` and is imported
    by the workflow as well as by the agent, and the agent is what owns the repair loop -- it
    catches this alongside `ValidationError`, states the violation back to the provider once,
    and fails the stage loudly if the repair violates it too. Nothing here decides that a
    feature stops; it decides only that a particular response cannot be believed.

    It carries `diagnostics` so that `safe_error_diagnostics` keeps the sentence rather than
    reducing a rejected reconciliation to its exception type. Every message below is composed
    here from constants and from requirement or question identifiers the feature already holds
    in durable state -- the same standard the planner's own rejections are written to -- and
    anything echoed back out of the response is truncated on the way in.
    """

    def __init__(self, *args: object) -> None:
        """Expose the violated rule so a failure past the repair says what was wrong."""
        self.diagnostics = tuple(str(item) for item in args if str(item))
        super().__init__(*args)


@dataclass(frozen=True, slots=True)
class AnsweredClarification:
    """One question a person answered, with their answer exactly as they wrote it.

    The question travels with the answer because the answer alone is frequently a fragment --
    "use compensating deletes" settles nothing without the question it answers -- and because
    `apply_clarification_answers` empties `unresolved_questions` when it records the answers,
    so the newest PRD revision no longer contains the text being answered.
    """

    question_id: str
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class RequirementRewrite:
    """One requirement restated to say what an answer settled.

    Exactly the three fields every downstream judge reads: `feature_planner` scopes tasks and
    acceptance criterion ids from `acceptance_criteria`, and the reviewer validates its
    requirement checks against the scope built from the same text. Priority and identity are
    absent because reconciliation restates decisions and never redesigns scope.
    """

    requirement_id: str
    description: str
    dependencies: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    # Which answers drove this rewrite. Recorded on the revision so a reader can trace a
    # changed acceptance criterion back to the sentence a person actually wrote.
    driving_answer_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnsettledContradiction:
    """Two statements that cannot both hold, named by the ids that carry them.

    Deliberately ids and one sentence, never quoted text. The platform quotes both sides
    itself, from the accepted answers and from the PRD, so "quoting both sides verbatim" is a
    property of this code rather than an instruction a model is trusted to have followed. A
    human's answer cannot be paraphrased into the question that asks them about it.
    """

    answer_ids: tuple[str, ...]
    # Empty when the conflict is answer-against-answer rather than answer-against-requirement.
    requirement_id: str
    conflict: str


@dataclass(frozen=True, slots=True)
class RequirementReconciliation:
    """What one reconciliation call concluded: what to restate, and what it could not settle."""

    rewrites: tuple[RequirementRewrite, ...] = ()
    contradictions: tuple[UnsettledContradiction, ...] = ()

    @property
    def settles_everything(self) -> bool:
        """Whether every answer was integrated, leaving nothing for a person to decide."""
        return not self.contradictions

    @property
    def changes_nothing(self) -> bool:
        """Whether this call found nothing to do, so no new PRD revision may be written."""
        return not self.rewrites and not self.contradictions


def parse_reconciliation(
    payload: Mapping[str, Any],
    *,
    requirements: Sequence[Requirement],
    answers: Sequence[AnsweredClarification],
) -> RequirementReconciliation:
    """Validate one reconciliation response against the PRD and the answers it was given.

    Every refusal here is structural: an identifier that does not exist, a requirement list
    that is not the PRD's own, a rewrite with nothing in it. None of it judges whether the
    rewrite is *right* -- that is what the human's answer, quoted verbatim in the metadata,
    remains the authority on.
    """
    known_ids = [requirement.requirement_id for requirement in requirements]
    _require_unchanged_requirement_list(payload.get("requirement_ids"), known_ids)
    answer_ids = {item.question_id for item in answers}
    return RequirementReconciliation(
        rewrites=_rewrites(payload.get("changed_requirements"), set(known_ids), answer_ids),
        contradictions=_contradictions(
            payload.get("unsettled_contradictions"), set(known_ids), answer_ids
        ),
    )


def reconciled_requirements(
    requirements: Sequence[Requirement], rewrites: Sequence[RequirementRewrite]
) -> list[Requirement]:
    """Return the requirement list with each rewrite applied and everything else untouched.

    The untouched ones are the *same objects*, not re-serialized copies, which is what makes
    "a requirement no answer addresses is byte-identical" a fact rather than an assertion.
    """
    by_id = {rewrite.requirement_id: rewrite for rewrite in rewrites}
    applied: list[Requirement] = []
    for requirement in requirements:
        rewrite = by_id.get(requirement.requirement_id)
        if rewrite is None:
            applied.append(requirement)
            continue
        applied.append(
            requirement.model_copy(
                update={
                    "description": rewrite.description,
                    "dependencies": list(rewrite.dependencies),
                    "acceptance_criteria": list(rewrite.acceptance_criteria),
                }
            )
        )
    return applied


def reconciliation_change_list(
    rewrites: Sequence[RequirementRewrite],
) -> dict[str, list[str]]:
    """Return `requirement_id -> the answer ids that drove its rewrite`, for the revision."""
    return {rewrite.requirement_id: list(rewrite.driving_answer_ids) for rewrite in rewrites}


def contradiction_questions(
    contradictions: Sequence[UnsettledContradiction],
    *,
    answers: Sequence[AnsweredClarification],
    requirements: Sequence[Requirement],
) -> list[ClarificationQuestion]:
    """Ask which of two statements holds, quoting both of them verbatim.

    The platform composes every word of this except the one sentence naming what conflicts:
    both quoted sides are read out of the accepted answers and the PRD by id, so a question
    cannot misquote the person it is asking, and an answer cannot be paraphrased into the
    question about it. No suggested answer is attached -- a suggestion here would be the
    platform guessing at precisely the decision it has just established it cannot make.
    """
    answer_text = {item.question_id: item for item in answers}
    requirement_text = {item.requirement_id: item for item in requirements}
    questions: list[ClarificationQuestion] = []
    used: set[str] = set()
    for contradiction in contradictions:
        sides = [
            f'the answer to "{answer_text[answer_id].question}" says: '
            f'"{answer_text[answer_id].answer}"'
            for answer_id in contradiction.answer_ids
            if answer_id in answer_text
        ]
        requirement = requirement_text.get(contradiction.requirement_id)
        if requirement is not None:
            sides.append(
                f'requirement {requirement.requirement_id} says: "{requirement.description}" '
                f"and is accepted on: {_quoted_list(requirement.acceptance_criteria)}"
            )
        if len(sides) < 2:
            # Both sides or no question. One side is not a contradiction anybody can decide,
            # and composing the other half is the guess this whole path exists to refuse.
            continue
        questions.append(
            ClarificationQuestion(
                question_id=_question_id(contradiction, used),
                question=(
                    "These cannot both hold, so nothing is planned until you say which does. "
                    f"First, {sides[0]}. Second, {sides[1]}. Which one holds, and what should "
                    "the other one become?"
                ),
                rationale=(
                    f"{contradiction.conflict} Your answers are what this feature's "
                    "requirements are being rewritten to say, and these two cannot both be "
                    "written down. The platform will not choose between them for you."
                ),
                required=True,
            )
        )
    return questions


def _question_id(contradiction: UnsettledContradiction, used: set[str]) -> str:
    """Return a stable, readable id for one contradiction, unique within its round.

    Stable rather than pretty, and that is the whole requirement. The already-asked filter
    drops a question this feature has put to somebody before, so the same unsettled
    contradiction reported on a later round has to produce the same id -- otherwise a person is
    asked the identical question every round until the rounds cap ends the feature.
    """
    key = "-".join([contradiction.requirement_id, *sorted(contradiction.answer_ids)])
    slug = _SLUG_CHARACTERS.sub("-", key.lower()).strip("-") or "answers"
    if len(slug) > _MAX_SLUG_CHARACTERS:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:_MAX_SLUG_CHARACTERS]}-{digest}"
    question_id = f"{QUESTION_ID_PREFIX}{slug}"
    if question_id in used:
        question_id = f"{question_id}-{len(used) + 1}"
    used.add(question_id)
    return question_id


def _quoted_list(values: Sequence[str]) -> str:
    """Quote a criteria list into one readable sentence fragment."""
    return "; ".join(f'"{value}"' for value in values)


def _require_unchanged_requirement_list(value: Any, known_ids: Sequence[str]) -> None:
    """Refuse a response whose requirement list is not the PRD's own, in the PRD's own order.

    The one check that catches an added, dropped or renamed requirement. Reconciliation
    restates decisions inside requirements that already exist; a response proposing a
    different set of requirements is proposing to redesign the scope a person approved, and
    the answer to that is a refusal and not a merge.
    """
    if not isinstance(value, list) or [str(item) for item in value] != list(known_ids):
        msg = (
            "reconciliation must echo requirement_ids exactly as given, in the same order, "
            f"adding, removing and renaming nothing. Expected: {list(known_ids)}"
        )
        raise ReconciliationRejected(msg)


def _rewrites(
    value: Any, known_ids: set[str], answer_ids: set[str]
) -> tuple[RequirementRewrite, ...]:
    """Validate every proposed rewrite, or refuse the response."""
    if not isinstance(value, list):
        msg = "reconciliation changed_requirements must be a list"
        raise ReconciliationRejected(msg)
    rewrites: list[RequirementRewrite] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            msg = "each changed requirement must be an object"
            raise ReconciliationRejected(msg)
        requirement_id = _text(item.get("requirement_id"), "requirement_id")
        if requirement_id not in known_ids:
            msg = (
                f"reconciliation rewrote an unknown requirement: {_short(requirement_id)}. "
                f"The only requirements it may restate are: {sorted(known_ids)}"
            )
            raise ReconciliationRejected(msg)
        if requirement_id in seen:
            msg = f"reconciliation rewrote {requirement_id} more than once"
            raise ReconciliationRejected(msg)
        seen.add(requirement_id)
        criteria = _text_list(item.get("acceptance_criteria"), "acceptance_criteria")
        if not criteria:
            msg = (
                f"the rewrite of {requirement_id} has no acceptance criteria; a requirement "
                "may be restated but never emptied"
            )
            raise ReconciliationRejected(msg)
        driving = _text_list(item.get("driving_answer_ids"), "driving_answer_ids")
        unknown = [item_id for item_id in driving if item_id not in answer_ids]
        if not driving or unknown:
            msg = (
                f"the rewrite of {requirement_id} must name the answers that drove it. "
                f"The accepted answers are: {sorted(answer_ids)}"
            )
            raise ReconciliationRejected(msg)
        rewrites.append(
            RequirementRewrite(
                requirement_id=requirement_id,
                description=_text(item.get("description"), "description"),
                dependencies=tuple(_text_list(item.get("dependencies"), "dependencies")),
                acceptance_criteria=tuple(criteria),
                driving_answer_ids=tuple(driving),
            )
        )
    return tuple(rewrites)


def _contradictions(
    value: Any, known_ids: set[str], answer_ids: set[str]
) -> tuple[UnsettledContradiction, ...]:
    """Validate every reported contradiction, or refuse the response."""
    if not isinstance(value, list):
        msg = "reconciliation unsettled_contradictions must be a list"
        raise ReconciliationRejected(msg)
    contradictions: list[UnsettledContradiction] = []
    for item in value:
        if not isinstance(item, dict):
            msg = "each unsettled contradiction must be an object"
            raise ReconciliationRejected(msg)
        reported = _text_list(item.get("answer_ids"), "answer_ids")
        unknown = [item_id for item_id in reported if item_id not in answer_ids]
        if not reported or unknown:
            msg = (
                "each unsettled contradiction must name the accepted answers it is between. "
                f"The accepted answers are: {sorted(answer_ids)}"
            )
            raise ReconciliationRejected(msg)
        requirement_id = str(item.get("requirement_id") or "").strip()
        if requirement_id and requirement_id not in known_ids:
            msg = (
                "an unsettled contradiction names an unknown requirement: "
                f"{_short(requirement_id)}. "
                f"The only requirements it may name are: {sorted(known_ids)}"
            )
            raise ReconciliationRejected(msg)
        if len(reported) < 2 and not requirement_id:
            msg = (
                "an unsettled contradiction is between two things: name two answers, or one "
                "answer and the requirement it invalidates"
            )
            raise ReconciliationRejected(msg)
        contradictions.append(
            UnsettledContradiction(
                answer_ids=tuple(reported),
                requirement_id=requirement_id,
                conflict=_text(item.get("conflict"), "conflict"),
            )
        )
    return tuple(contradictions)


def _short(value: str) -> str:
    """Bound one identifier echoed out of a response before it reaches a durable diagnostic.

    An identifier the response invented is model output, and a rejection message is recorded
    when the repair fails too. Naming it is what makes the rejection actionable; naming all of
    it, whatever length it happens to be, is publishing model text into the failure record.
    """
    return value if len(value) <= _MAX_ECHOED_CHARACTERS else f"{value[:_MAX_ECHOED_CHARACTERS]}…"


def _text(value: Any, field: str) -> str:
    """Return one required non-empty string, or refuse the response."""
    if not isinstance(value, str) or not value.strip():
        msg = f"reconciliation field '{field}' must be a non-empty string"
        raise ReconciliationRejected(msg)
    return value.strip()


def _text_list(value: Any, field: str) -> list[str]:
    """Return one list of non-empty strings, or refuse the response."""
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        msg = f"reconciliation field '{field}' must be a list of non-empty strings"
        raise ReconciliationRejected(msg)
    return [item.strip() for item in value]
