"""Classify an engineer-remediable review finding by the reasoning its fix needs.

This answers one question -- how hard is this correction to reason about -- and nothing else.
It does not decide whether a retry is allowed, which repository is affected, or what the fix
should be. Those belong to the retry policy, the workstream and the Engineer respectively.

Two rules shape everything here:

*Severity is not complexity.* A high-severity missing null check is a mechanical edit; a
medium-severity race condition is not. Severity ranks how much a defect matters, and the
model that should fix it depends on how much of the system has to be held in mind to fix it
correctly. Severity therefore only ever routes a finding *upward*, never downward.

*Uncertainty stays primary.* A finding this cannot place is substantive by default. The cost
of that mistake is some tokens; the cost of the opposite mistake is a scoped model making a
confident, wrong change to a transaction boundary.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from enum import StrEnum

from pydantic import Field, field_validator

from state.models import StateModel
from tools.retry_strategy import diagnostic_signature


class ReviewFixComplexity(StrEnum):
    """How much reasoning one review finding's correction requires."""

    STANDARD = "STANDARD"
    COMPLEX = "COMPLEX"
    CRITICAL = "CRITICAL"


# Ranked, so a set of findings routes to whatever its hardest member needs.
_COMPLEXITY_ORDER: dict[ReviewFixComplexity, int] = {
    ReviewFixComplexity.STANDARD: 0,
    ReviewFixComplexity.COMPLEX: 1,
    ReviewFixComplexity.CRITICAL: 2,
}


class ReviewFixClassification(StateModel):
    """One finding's classification, with why it was reached."""

    classification: ReviewFixComplexity
    reason: str = Field(min_length=1)
    # Whether the structured review metadata alone decided this. False means the decision came
    # from the conservative default, which is worth knowing when reading a routing record: it
    # is the difference between "the reviewer said concurrency" and "nothing said anything".
    deterministic: bool = True
    finding_id: str = ""
    fingerprint: str = ""


class ReviewFixClassifierVerdict(StateModel):
    """What an optional classification model may return. It never proposes a fix."""

    classification: ReviewFixComplexity
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)

    @field_validator("classification", mode="before")
    @classmethod
    def classification_must_be_allowed(
        cls, value: ReviewFixComplexity | str
    ) -> ReviewFixComplexity:
        """Accept only the three allowed answers, from an enum or a model's JSON string."""
        return value if isinstance(value, ReviewFixComplexity) else ReviewFixComplexity(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_may_be_written_as_an_integer(cls, value: float | int) -> float:
        """Accept `1` for full confidence, which a model writes and strict mode would reject."""
        return float(value) if isinstance(value, int) and not isinstance(value, bool) else value


# Structured review categories that decide a classification on their own. The reviewer already
# assigns these, so preferring them over prose is both cheaper and steadier: a category does
# not get reworded between attempts.
_CRITICAL_CATEGORIES = frozenset({"security"})
_COMPLEX_CATEGORIES = frozenset({"contract"})

# Vocabulary. Each entry is a phrase that appears in a review's own words when it is talking
# about the corresponding kind of defect. This is deliberately about *review prose*, not about
# any repository's layout, language or framework -- nothing here encodes a target project.
#
# Matched as whole words so `authentication` cannot be found inside an unrelated identifier and
# `test` cannot match `latest`.
_CRITICAL_TERMS: tuple[str, ...] = (
    "authentication",
    "authenticate",
    "authorization",
    "authorize",
    "authorisation",
    "access control",
    "privilege escalation",
    "permission check",
    "secret",
    "credential",
    "api key",
    "token leak",
    "vulnerability",
    "injection",
    "xss",
    "csrf",
    "data loss",
    "data corruption",
    "irreversible",
    "destructive",
    "drop table",
    "dangerous migration",
    "tenant isolation",
    "sensitive data",
)
_COMPLEX_TERMS: tuple[str, ...] = (
    "race condition",
    "concurrency",
    "concurrent",
    "atomic",
    "lock",
    "deadlock",
    "transaction",
    "transactional",
    "idempotency",
    "idempotent",
    "state machine",
    "state transition",
    "retry semantics",
    "retry policy",
    "orchestration",
    "workflow state",
    "architecture",
    "architectural",
    "architecturally",
    "refactor",
    "cross-service",
    "cross-repository",
    "distributed",
    "api contract",
    "contract mismatch",
    "consistency",
    "persistence",
    "crash recovery",
    "recovery",
    "cancellation",
    "migration",
    "ambiguous",
    "unclear",
    "business logic",
    "invariant",
)
_STANDARD_TERMS: tuple[str, ...] = (
    "module not found",
    "cannot find module",
    "cannot resolve import",
    "cannot resolve module",
    "wrong relative import",
    "missing named export",
    "incorrect named import",
    "incorrect module path",
    "wrong imported type",
    "simple circular import introduced by recent change",
    "lint",
    "eslint",
    "ruff",
    "flake8",
    "pylint",
    "prettier",
    "formatting",
    "unused import",
    "unused variable",
    "defined but never used",
    "no-undef",
    "no-unused-vars",
    "import order",
    "import/order",
    "import/no-duplicates",
    "imported multiple times",
    "trailing whitespace",
    "simple rule violation",
    "incorrect path",
    "missing small dependency declaration",
    "single broken assertion",
    "test fixture mismatch",
    "incorrect mock import",
    "deterministic test correction",
    "wrong extension",
    "small package script issue",
    "simple module resolution problem",
)


def _term_pattern(terms: Iterable[str]) -> re.Pattern[str]:
    """Compile a whole-word alternation over a fixed vocabulary."""
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(term) for term in ordered) + r")(?![\w-])"
    )


_CRITICAL_PATTERN = _term_pattern(_CRITICAL_TERMS)
_COMPLEX_PATTERN = _term_pattern(_COMPLEX_TERMS)
_STANDARD_PATTERN = _term_pattern(_STANDARD_TERMS)
_REPOSITORY_CONFIGURATION_PATTERN = _term_pattern(
    (
        "eslint configuration",
        "eslint-config",
        "lint configuration",
        "parser configuration",
        "tooling configuration",
        "repository setup",
        "dependency not installed",
        "dependency is not installed",
        "unsupported runtime",
    )
)
_LOCAL_SCOPE_PATTERN = _term_pattern(
    (
        "single",
        "one",
        "local",
        "localized",
        "narrow",
        "small",
        "recent change",
        "this file",
        "this module",
    )
)
_TYPE_PROBLEM_PATTERN = re.compile(
    r"(?<![\w-])(?:interface mismatch|property.{0,80}?\bmissing|missing property|"
    r"generic mismatch|typescript compile error|type error)(?![\w-])"
)
_PATH_RESOLUTION_PATTERN = re.compile(
    r"(?<![\w-])cannot resolve\s+(?:\.\.?/|/|@)[a-z0-9_./:@-]+(?![\w-])"
)

# Collapsed before fingerprinting and before matching. A reviewer rewords the same defect
# between cycles, and a fingerprint that changed with the wording would report every repeat as
# a brand-new finding, which would also defeat the retry system's convergence checks.
_WHITESPACE = re.compile(r"\s+")
_NON_SEMANTIC = re.compile(r"[^a-z0-9\s./:_-]+")
# Numbers that are part of a diagnostic's identity are kept by the location tokens below; a
# bare number inside prose ("3 of 8 attempts") is not.
_DESCRIPTION_MAX_CHARACTERS = 400


def normalized_finding_text(*parts: str) -> str:
    """Reduce review prose to the words that identify the defect it is about."""
    joined = " ".join(part for part in parts if part)
    lowered = _NON_SEMANTIC.sub(" ", joined.lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def defect_identity(*parts: str) -> str:
    """Reduce a finding's own words to the defect they name, or to normalized prose.

    ``diagnostic_signature`` is the platform's existing answer to "which defect is this
    diagnostic about, as opposed to how it was worded this time": a path with a line number, an
    exception name, a result code, a linter rule. Reusing it here rather than writing a second
    normalizer is what makes a reworded repeat match the existing convergence checks.

    Where nothing identifying can be extracted, the normalized prose stands in. That is weaker,
    and deliberately so: two findings whose wording differs entirely are more likely to be two
    findings than one reworded, and treating them as one would report a repeat nobody has tried
    to fix yet.
    """
    tokens = diagnostic_signature([part for part in parts if part])
    if tokens:
        return "|".join(tokens)
    return normalized_finding_text(*parts)[:_DESCRIPTION_MAX_CHARACTERS]


def finding_fingerprint(finding: object) -> str:
    """Fingerprint the defect a finding names, independently of how it was worded.

    Built from the structured fields that identify a defect -- what it is about, where, and
    under which requirement -- plus the defect its own text names. Severity is included because
    a reviewer that raises the same observation to critical has said something new about it.

    Accepts anything with the reviewer's finding attributes, so a persisted dict and a
    ``ReviewFinding`` both work without this module importing the artifact schema.
    """
    fields = _finding_fields(finding)
    digest = hashlib.sha256()
    for value in (
        fields["requirement_id"],
        fields["finding_category"],
        fields["severity"],
        fields["repository_id"],
        fields["file_path"],
        fields["contract_reference"],
        defect_identity(fields["title"], fields["description"]),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def fingerprint_for_text(text: str) -> str:
    """Fingerprint a bare blocking-issue string, for a failure that produced no findings.

    A deterministic gate and an integration reviewer both hand back sentences rather than
    structured findings. They still need a stable identity so convergence can tell a repeat
    from something new.
    """
    digest = hashlib.sha256(defect_identity(text).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def classify_review_finding(finding: object) -> ReviewFixClassification:
    """Classify one structured review finding deterministically.

    The order is the point. A security or data-risk finding is critical whatever else it looks
    like, so it never selects scoped fix. Then the architecture, concurrency and state
    vocabulary. Only then the mechanical vocabulary -- and only when nothing above it matched,
    because "add a test for the race condition" is not a test-shaped task.
    """
    fields = _finding_fields(finding)
    fingerprint = finding_fingerprint(finding)
    text = normalized_finding_text(
        fields["title"], fields["description"], fields["recommendation"], fields["recommended_fix"]
    )
    finding_id = fields["finding_id"]
    if fields["finding_category"] in _CRITICAL_CATEGORIES:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.CRITICAL,
            reason=f"The reviewer categorised this finding as {fields['finding_category']}.",
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    if (match := _CRITICAL_PATTERN.search(text)) is not None:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.CRITICAL,
            reason=f"The finding concerns {match.group(0)}, which carries security or data risk.",
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    if fields["finding_category"] in _COMPLEX_CATEGORIES:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.COMPLEX,
            reason=(
                f"The reviewer categorised this finding as {fields['finding_category']}, which "
                "spans more than one repository's behaviour."
            ),
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    if (match := _REPOSITORY_CONFIGURATION_PATTERN.search(text)) is not None:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.COMPLEX,
            reason=(
                f"The finding concerns {match.group(0)}, which is not established as a "
                "feature-introduced localized source defect."
            ),
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    if (match := _COMPLEX_PATTERN.search(text)) is not None:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.COMPLEX,
            reason=f"The finding concerns {match.group(0)}, which needs system-level reasoning.",
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    if (match := _STANDARD_PATTERN.search(text)) is not None:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.STANDARD,
            reason=f"The finding is a localized {match.group(0)} correction.",
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    if (match := _PATH_RESOLUTION_PATTERN.search(text)) is not None:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.STANDARD,
            reason=f"The finding is a localized path-resolution correction: {match.group(0)}.",
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    # Type diagnostics can describe anything from one renamed property to a contract redesign.
    # They are scoped only when the finding also says the affected area is bounded; an
    # unqualified "type error" is deliberately not enough.
    if (match := _TYPE_PROBLEM_PATTERN.search(text)) is not None and _LOCAL_SCOPE_PATTERN.search(
        text
    ) is not None:
        return ReviewFixClassification(
            classification=ReviewFixComplexity.STANDARD,
            reason=f"The finding is a localized {match.group(0)} correction.",
            finding_id=finding_id,
            fingerprint=fingerprint,
        )
    return ReviewFixClassification(
        classification=ReviewFixComplexity.COMPLEX,
        reason=(
            "Nothing in the finding's category or wording places it, and an unplaced finding "
            "is treated as needing reasoning rather than as a mechanical edit."
        ),
        deterministic=False,
        finding_id=finding_id,
        fingerprint=fingerprint,
    )


def classify_blocking_issue(text: str, *, finding_id: str = "") -> ReviewFixClassification:
    """Classify a bare blocking-issue sentence with the same vocabulary and the same defaults."""
    return classify_review_finding(
        {
            "finding_id": finding_id,
            "description": text,
            "severity": "",
            "finding_category": "",
        }
    )


def apply_classifier_verdict(
    deterministic: ReviewFixClassification,
    verdict: ReviewFixClassifierVerdict,
    *,
    minimum_confidence: float,
) -> ReviewFixClassification:
    """Fold an optional classification model's answer into the deterministic one.

    The model is consulted only where the deterministic rules did not place a finding, and its
    answer is accepted only where it is confident and does not lower the classification.
    A deterministic CRITICAL is never revisited: the whole reason that rule runs first is that
    a security or data-risk finding must not be talked down by anything.
    """
    if deterministic.classification is ReviewFixComplexity.CRITICAL:
        return deterministic
    if verdict.confidence < minimum_confidence:
        return deterministic.model_copy(
            update={
                "classification": highest_complexity(
                    (deterministic.classification, ReviewFixComplexity.COMPLEX)
                ),
                "reason": (
                    f"The classification model answered {verdict.classification.value} at "
                    f"{verdict.confidence:.2f} confidence, below the configured minimum of "
                    f"{minimum_confidence:.2f}, so this routes upward instead."
                ),
                "deterministic": False,
            }
        )
    return deterministic.model_copy(
        update={
            "classification": highest_complexity(
                (deterministic.classification, verdict.classification)
            ),
            "reason": verdict.reason,
            "deterministic": False,
        }
    )


def highest_complexity(
    classifications: Iterable[ReviewFixComplexity],
) -> ReviewFixComplexity:
    """Return the hardest classification in a set, which is the one the fix must satisfy."""
    return max(
        classifications,
        key=lambda item: _COMPLEXITY_ORDER[item],
        default=ReviewFixComplexity.COMPLEX,
    )


def classify_findings(
    findings: Sequence[object],
    *,
    blocking_issues: Sequence[str] = (),
    previous_attempt_intact: bool = True,
) -> tuple[ReviewFixClassification, ...]:
    """Classify every structured finding, falling back to the blocking issues it produced.

    Structured findings are preferred, exactly as the PRD asks: they carry a category, a
    requirement and a path, and none of that survives being flattened into a sentence. The
    sentences are used when there are no findings at all -- a deterministic gate, or an
    integration reviewer's recommended fix -- because a workstream still has to be routed.

    ``previous_attempt_intact`` says whether the retry will run against the previous
    attempt's files. When it will not -- the workspace-reset path, where the retry is handed
    a capture that may withhold files -- a mechanical finding may not classify as a scoped
    fix, however mechanical it looks: the finding is one import, but the job is the
    reconstruction. AB-Feature-171 routed a twenty-file regeneration to the scoped-fix model
    exactly this way. The demotion is expressed as `deterministic=False`, which is the
    signal `determine_repair_scope` already reads as "route this upward".
    """
    classified = tuple(classify_review_finding(finding) for finding in findings)
    if not classified:
        classified = tuple(
            classify_blocking_issue(issue, finding_id=f"BLOCKING-{index + 1}")
            for index, issue in enumerate(blocking_issues)
            if issue.strip()
        )
    if previous_attempt_intact:
        return classified
    return tuple(_demoted_for_reset_fragment(item) for item in classified)


def _demoted_for_reset_fragment(item: ReviewFixClassification) -> ReviewFixClassification:
    """Keep a mechanical verdict from selecting the scoped-fix role over a fragment."""
    if item.classification is not ReviewFixComplexity.STANDARD or not item.deterministic:
        return item
    return item.model_copy(
        update={
            "deterministic": False,
            "reason": (
                f"{item.reason} The retry does not have the previous attempt intact, so the "
                "remediation is a reconstruction rather than a scoped correction."
            ),
        }
    )


def _finding_fields(finding: object) -> dict[str, str]:
    """Read the reviewer's finding fields from an artifact or a persisted mapping."""

    def read(name: str) -> str:
        value = (
            finding.get(name)
            if isinstance(finding, dict)
            else getattr(finding, name, None)  # ReviewFinding and any structurally equal object
        )
        return str(value) if value is not None else ""

    return {
        name: read(name)
        for name in (
            "finding_id",
            "requirement_id",
            "finding_category",
            "severity",
            "repository_id",
            "file_path",
            "contract_reference",
            "title",
            "description",
            "recommendation",
            "recommended_fix",
        )
    }


__all__ = [
    "ReviewFixClassification",
    "ReviewFixClassifierVerdict",
    "ReviewFixComplexity",
    "apply_classifier_verdict",
    "classify_blocking_issue",
    "classify_findings",
    "classify_review_finding",
    "defect_identity",
    "finding_fingerprint",
    "fingerprint_for_text",
    "highest_complexity",
    "normalized_finding_text",
]
