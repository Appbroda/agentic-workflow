"""Ask the Engineer's own model, once, whether its finished work meets the assignment.

Until now the Engineer's exit condition was mechanical: lint, format, typecheck, the
change's own tests and the wiring inspection all green. Every one of those is a check a
deterministic tool can run, and none of them can say whether the *assignment* was met -- an
implementation that handles create and silently omits the update half of a create-and-update
requirement passes all of them and costs a full review cycle to learn about.

This module is the boundary for the one question that has no deterministic answer: read the
requirements, the plan and the files the attempt actually changed, and say what is missing.
It runs on the CODING role at the feature's pinned platform and tier -- never the REVIEW
role, whose independence belongs to the reviewer that judges this work afterwards, and never
a silently escalated tier.

The bounds are the whole safety case, and they are enforced by the caller
(`EngineerAgent`), not requested of the model:

* exactly one review pass per attempt -- there is no loop here to bound;
* localized findings are corrected once in the same attempt and re-validated;
* a substantive finding leaves the attempt with the named outcome below;
* the result gates ``completion_status`` and nothing further -- the commit gate and the
  independent Reviewer run afterwards, unchanged, whatever this pass said.

Every failure of this boundary -- the model erring, the response refusing to parse -- is
``None``, and the caller proceeds exactly as it did before self-review existed. This exists
to catch a known gap earlier, never to invent a new way for an attempt to fail.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Protocol

from adapters.llm_adapter import LLMClient
from agents.shared.contracts import EXECUTION_METADATA_KEY, execution_metadata
from services.cancellation import CancellationRequested

_LOGGER = logging.getLogger(__name__)

# The named terminal outcome for an attempt whose own review found a problem no bounded
# correction can answer -- rearchitecting, a contract change, a broad rewrite. It leaves the
# attempt as a sentence rather than as a finding fragment, because every escape from an
# attempt has to be explainable from durable state alone.
SELF_REVIEW_SUBSTANTIVE_OUTCOME = "SELF_REVIEW_SUBSTANTIVE_PROBLEM"

# How every diagnostic composed from a self-review finding begins. Deliberately NOT one of
# the repair loop's mechanical prefixes: a self-review finding is a model's judgement about
# the assignment, not a check of this workspace speaking, and the bounded mechanical repair
# loop must hand it back untouched. These diagnostics travel on the attempt's failure record
# so the next attempt's context assembly can resolve what they name.
SELF_REVIEW_DIAGNOSTIC_PREFIX = "The implementation self-review found"

_FINDING_CLASSIFICATIONS = frozenset({"localized", "substantive"})
_COVERAGE_STATUSES = frozenset({"implemented", "partial", "missing", "not_applicable"})
# Bounds on what one review may report. A response naming forty findings is not a review of
# this change, it is a model filling space -- and every finding below feeds either a
# correction prompt or a retry's context assembly, both of which have budgets of their own.
_MAX_FINDINGS = 12
_MAX_FINDING_CHARACTERS = 1_200
_MAX_FINDING_PATHS = 8

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# What the review pass is asked, appended to the Engineer's own instructions so the reviewer
# reads the same assignment the implementation did. The deterministic evidence is put first
# in the input so the pass does not spend attention re-deriving what is already established.
SELF_REVIEW_DIRECTIVE = (
    "\n\nYou have just finished the implementation described above, and every deterministic "
    "check this repository configures has accepted it: its own lint and format commands, its "
    "typecheck, the change's own tests, and the wiring inspection. Their evidence is in the "
    "input under `validation_evidence`; do not re-derive what they already established.\n\n"
    "The input quotes the candidate change -- every file this workstream's attempts have "
    "changed, this attempt's own writes first. Three states, all declared: a file under "
    "`changed_files` is quoted; a file under `withheld_paths` was changed by this work but "
    "could not be quoted, and `withheld_reasons` says why; a quoted file ending in a "
    "`[truncated for size...]` marker has an unseen tail. A file you were not shown is a "
    "file you cannot judge: never report a withheld file, or content beyond a truncation "
    "marker, as missing, absent, or unimplemented -- absence of quotation is not absence of "
    "implementation, the platform records what you could not see as a limitation, and a "
    "finding that names only unseen files is discarded. When visibility limits a coverage "
    "judgement, say so in that entry's `evidence` text.\n\n"
    "Review your finished work exactly once, against the assignment, and answer only the "
    "questions no deterministic check can:\n"
    "- Requirement completeness: for each scoped requirement, is it implemented, partial, "
    "missing, or not applicable to this repository -- judged against the changed files "
    "quoted under `changed_files`, never from memory of what you meant to write.\n"
    "- Behavioural completeness: error paths, empty and null handling, and every branch the "
    "requirement names -- the update half of a create-and-update requirement, for example.\n"
    "- Scope discipline: unrelated files, debug output, commented-out blocks, or TODO "
    "placeholders left behind by this change.\n"
    "- Contract compliance: names, shapes, enums and error forms, where the plan or the "
    "shared contract states them.\n\n"
    "Classify every finding:\n"
    '- "localized": correctable with a small, bounded edit to files this change already '
    "touches.\n"
    '- "substantive": requires rearchitecting, a contract change, or a broad rewrite.\n\n'
    "Respond with exactly one JSON object and nothing else -- no Markdown, no code fence, no "
    "prose around it:\n"
    '{"summary": "<one short paragraph>",\n'
    ' "requirement_coverage": [{"requirement_id": "<id>", "status": '
    '"implemented|partial|missing|not_applicable", "evidence": "<what in the changed files '
    'shows this>"}],\n'
    ' "findings": [{"description": "<the gap and where it is>", "classification": '
    '"localized|substantive", "paths": ["<workspace-relative path>"], "requirement_id": '
    '"<id or null>"}]}\n\n'
    "An empty findings list means the work is ready as it stands. Report only defects in "
    "this change against its own assignment: style preferences, refactors, and improvements "
    "the assignment never asked for are not findings."
)

_REASKED = (
    "\n\nYour previous response could not be read as the review it was asked for. The "
    "reason was:\n{reason}\n\nReturn the same review again as exactly one valid JSON object "
    "with the keys `summary`, `requirement_coverage` and `findings`, and nothing around it."
)


@dataclass(frozen=True, slots=True)
class SelfReviewFinding:
    """One gap the review found in the finished work, and how bounded its correction is."""

    description: str
    classification: str
    paths: tuple[str, ...] = ()
    requirement_id: str | None = None

    @property
    def substantive(self) -> bool:
        """Whether this finding leaves the attempt rather than being corrected inside it."""
        return self.classification == "substantive"

    def as_record(self) -> dict[str, Any]:
        """Render this finding for the completion metadata, which is the measurement."""
        return {
            "description": self.description,
            "classification": self.classification,
            "paths": list(self.paths),
            "requirement_id": self.requirement_id,
        }


@dataclass(frozen=True, slots=True)
class SelfReviewAssessment:
    """What one review pass concluded, and what executed it.

    ``requirement_coverage`` is recorded verbatim rather than acted on: the gate reads only
    ``findings``, and the directive requires every missing or partial requirement to appear
    there. Coverage exists for the measurement 47- ships with -- "Reviewer blocking findings
    a self-review should reasonably have caught" is computed by joining these records
    against the review artifacts that follow them.
    """

    findings: tuple[SelfReviewFinding, ...]
    requirement_coverage: tuple[dict[str, Any], ...]
    summary: str
    response_id: str
    model: str
    execution: dict[str, Any]

    @property
    def substantive_findings(self) -> tuple[SelfReviewFinding, ...]:
        """The findings that cannot be corrected inside this attempt."""
        return tuple(finding for finding in self.findings if finding.substantive)

    @property
    def localized_findings(self) -> tuple[SelfReviewFinding, ...]:
        """The findings a single bounded correction may answer."""
        return tuple(finding for finding in self.findings if not finding.substantive)


class SelfReviewer(Protocol):
    """Ask one bounded review of an attempt's finished work."""

    async def review(self, *, instructions: str, input_text: str) -> SelfReviewAssessment | None:
        """Return the assessment, or None when the review could not be had at all."""


class NullSelfReviewer:
    """Review nothing, which is what a composition with no configured boundary must do."""

    async def review(self, *, instructions: str, input_text: str) -> SelfReviewAssessment | None:
        """Report that no review ran, so the attempt proceeds exactly as before Part E."""
        del instructions, input_text
        return None


class CodingRoleSelfReviewer:
    """One review pass on an injected model boundary -- the CODING role, pinned tier.

    The caller resolves the client; this class only asks the question and reads the answer.
    Text in, text out: no tool loop, no adapter change, exactly the constraint 47- puts on
    every model call this platform composes. A response that cannot be read is re-asked once
    -- the same bounded repair the coding executor and the reviewer already give a malformed
    response -- and a second failure is ``None``, never an exception the attempt inherits.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Inject the resolved model boundary."""
        self._llm_client = llm_client

    async def review(self, *, instructions: str, input_text: str) -> SelfReviewAssessment | None:
        """Run the single review pass and parse its structured verdict."""
        directive = f"{instructions}{SELF_REVIEW_DIRECTIVE}"
        try:
            response = await self._llm_client.respond(instructions=directive, input_text=input_text)
            try:
                parsed = _parse_assessment(response.output_text)
            except ValueError as reason:
                response = await self._llm_client.respond(
                    instructions=f"{directive}{_REASKED.format(reason=reason)}",
                    input_text=input_text,
                )
                parsed = _parse_assessment(response.output_text)
        except CancellationRequested:
            raise
        except Exception as review_error:  # noqa: BLE001 - the review is best-effort
            # In the message, not in `extra`: the deployment's plain formatter drops
            # `extra`, and an unexplained absent review must never need journal forensics.
            _LOGGER.warning(
                "the implementation self-review did not run: %s. The attempt proceeds and "
                "the independent reviewer decides [outcome=self_review_unavailable "
                "agent=engineer]",
                type(review_error).__name__,
            )
            return None
        findings, coverage, summary = parsed
        return SelfReviewAssessment(
            findings=findings,
            requirement_coverage=coverage,
            summary=summary,
            response_id=response.response_id,
            model=response.model,
            execution=execution_metadata(
                agent_type="Engineer",
                provider=response.provider,
                model=response.model,
                reasoning_effort=response.reasoning_effort,
                model_role=response.model_role,
                model_variable=response.model_variable,
                routing_reason=response.routing_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )[EXECUTION_METADATA_KEY],
        )


def _parse_assessment(
    output_text: str,
) -> tuple[tuple[SelfReviewFinding, ...], tuple[dict[str, Any], ...], str]:
    """Read one review response strictly, so a malformed verdict is re-asked, not guessed at.

    Findings are validated hard -- an unknown classification is a response this gate cannot
    act on, and acting on a guess would either fail an attempt on nothing or wave through a
    gap the model did name. Coverage entries are informational and validated only for shape;
    a malformed one is dropped rather than failing the review that carried it.
    """
    body = _JSON_FENCE.sub("", output_text).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        msg = "the response contains no JSON object"
        raise ValueError(msg)
    try:
        payload = json.loads(body[start : end + 1])
    except json.JSONDecodeError as error:
        msg = f"the response is not valid JSON: {error.msg}"
        raise ValueError(msg) from None
    if not isinstance(payload, dict):
        msg = "the response is not a JSON object"
        raise ValueError(msg)
    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list):
        msg = "the response has no `findings` list"
        raise ValueError(msg)
    findings: list[SelfReviewFinding] = []
    for item in raw_findings[:_MAX_FINDINGS]:
        if not isinstance(item, dict):
            msg = "a finding is not a JSON object"
            raise ValueError(msg)
        description = item.get("description")
        classification = item.get("classification")
        if not isinstance(description, str) or not description.strip():
            msg = "a finding has no description"
            raise ValueError(msg)
        if classification not in _FINDING_CLASSIFICATIONS:
            msg = 'a finding\'s classification is not "localized" or "substantive"'
            raise ValueError(msg)
        raw_paths = item.get("paths")
        paths = tuple(
            str(path)
            for path in (raw_paths if isinstance(raw_paths, list) else [])[:_MAX_FINDING_PATHS]
            if isinstance(path, str) and path.strip()
        )
        requirement = item.get("requirement_id")
        findings.append(
            SelfReviewFinding(
                description=description.strip()[:_MAX_FINDING_CHARACTERS],
                classification=classification,
                paths=paths,
                requirement_id=(
                    requirement.strip()
                    if isinstance(requirement, str) and requirement.strip()
                    else None
                ),
            )
        )
    coverage = tuple(
        {
            "requirement_id": str(item.get("requirement_id", "")),
            "status": item["status"],
            "evidence": str(item.get("evidence", "")),
        }
        for item in payload.get("requirement_coverage", [])
        if isinstance(item, dict) and item.get("status") in _COVERAGE_STATUSES
    )
    summary = payload.get("summary")
    return (
        tuple(findings),
        coverage,
        summary.strip() if isinstance(summary, str) else "",
    )


def evidence_blind(finding: SelfReviewFinding, unseen: Collection[str]) -> bool:
    """Whether this finding judges only evidence the review was declared not to have.

    True only for a substantive finding that names at least one path and whose every named
    path was declared not fully shown -- withheld from the quoted evidence, or quoted with a
    truncated tail. Such a finding is an accurate description of blindness, not of the work
    (AB-Feature-207 lost a green workstream to one), so the caller records it as a
    limitation and composes no diagnostic from it: a limitation is metadata, never a
    diagnostic (76-).

    Deliberately narrow. A named path outside ``unseen`` -- including one no attempt ever
    wrote -- keeps the finding alive, because "this file was never created" is exactly the
    true substantive finding the gate exists for: unwritten is not unseen. A pathless
    finding also stands; the directive is the only defense there, and the record shows it.
    Localized findings are never demoted: their correction pass reads the workspace itself
    and is re-validated, so a blind one costs a bounded pass, not an attempt.
    """
    return (
        finding.substantive
        and bool(finding.paths)
        and all(path in unseen for path in finding.paths)
    )


def self_review_diagnostic(finding: SelfReviewFinding) -> str:
    """Compose one attempt-record diagnostic from a finding, naming what it names.

    The paths ride in the sentence because they are what the next attempt's context
    assembly resolves: a finding whose subject is neither imported, named by the plan, nor
    touched by a prior attempt is otherwise selected by lexical ranking alone and lost.
    """
    located = f" [files: {', '.join(finding.paths)}]" if finding.paths else ""
    scoped = f" [requirement: {finding.requirement_id}]" if finding.requirement_id else ""
    return (
        f"{SELF_REVIEW_DIAGNOSTIC_PREFIX} ({finding.classification}): "
        f"{finding.description}{located}{scoped}"
    )


__all__ = [
    "SELF_REVIEW_DIAGNOSTIC_PREFIX",
    "SELF_REVIEW_DIRECTIVE",
    "SELF_REVIEW_SUBSTANTIVE_OUTCOME",
    "CodingRoleSelfReviewer",
    "NullSelfReviewer",
    "SelfReviewAssessment",
    "SelfReviewFinding",
    "SelfReviewer",
    "evidence_blind",
    "self_review_diagnostic",
]
