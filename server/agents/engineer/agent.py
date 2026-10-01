"""Workspace-bounded software-engineering node."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, overload

from adapters.git_adapter import GitService
from adapters.llm_adapter import CodingExecutor, ImageInput, is_transport_fault
from agents.product_manager.agent import AttachmentContentSource
from agents.shared.contract_sections import (
    contract_section_context_from_state,
    contract_section_context_json,
)
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    EXECUTION_METADATA_KEY,
    AgentArtifactError,
    artifact_id_matches_lineage,
    artifact_json,
    artifact_update,
    attempt_artifact_id,
    create_artifact,
    execution_metadata,
    require_artifact,
)
from agents.shared.design_snapshot import (
    design_snapshot_context_from_state,
    design_snapshot_context_json,
)
from artifacts.schemas import (
    CodeCompletionArtifact,
    FileChange,
    RepositoryReconnaissanceArtifact,
    RequirementImplementationEvidence,
    ReviewArtifact,
    TaskPlanArtifact,
)
from prompts.prompt_loader import PromptLoader
from services.cancellation import CancellationRequested, CancellationToken
from services.external_operations import ExternalOperationExecutor
from services.process_runner import redact_source_credentials
from state.failure_diagnosis import DiagnosedFailure, FeatureFailureClassification
from state.models import AgentState
from tools.assigned_file_conformance import (
    AssignedFileChecker,
    NullAssignedFileChecker,
    assigned_file_diagnostic,
    is_assigned_file_diagnostic,
)
from tools.channel_packages import channel_seam_scan
from tools.definition_sites import DefinitionIndex, DefinitionSite, definition_symbols
from tools.dependency_sync import (
    DependencySynchronizer,
    NullDependencySynchronizer,
    installed_lockfiles,
)
from tools.design_value_conformance import (
    DesignValueChecker,
    design_value_diagnostic,
)
from tools.file_tools import (
    FileTool,
    PathLike,
    WorkspaceFileTools,
    carries_key_material,
    is_model_safe_context_path,
    resolve_workspace_path,
)
from tools.implementation_completeness import classify_file_change
from tools.lint_capabilities import LintCapabilityProbe, NullLintCapabilityProbe
from tools.model_routing import ModelRoutingDecision
from tools.reachability import (
    REACHABILITY_DIAGNOSTIC_PREFIX,
    NullReachabilityChecker,
    ReachabilityChecker,
    is_reachability_diagnostic,
    module_specifiers,
    reachability_diagnostic,
    resolve_specifier,
)
from tools.repo_tools import RepositoryTool, WorkspaceRepositoryTools
from tools.retry_strategy import diagnostic_file_references, diagnostic_signature
from tools.scoped_tests import (
    ASSERTION_GUARD_OUTCOME,
    SCOPED_TEST_DIAGNOSTIC_PREFIX,
    NullScopedTestRunner,
    ScopedTestRunner,
    assertion_guard_diagnostic,
    changed_test_paths,
    is_scoped_test_diagnostic,
    weakened_test_paths,
)
from tools.self_review import (
    SELF_REVIEW_SUBSTANTIVE_OUTCOME,
    SelfReviewer,
    evidence_blind,
    self_review_diagnostic,
)
from tools.source_formatting import (
    NullSourceFormatter,
    SourceFormatter,
    SourceValidationError,
)
from tools.source_windows import SOURCE_REGION_RADIUS_LINES, source_region
from tools.typecheck import (
    TYPECHECK_DIAGNOSTIC_PREFIX,
    NullTypecheckRunner,
    TypecheckOutcome,
    TypecheckRunner,
    is_typecheck_diagnostic,
)

type FileToolsFactory = Callable[[PathLike], FileTool]
type RepositoryToolsFactory = Callable[[PathLike], RepositoryTool]
type GitServiceFactory = Callable[[str], GitService | Any]

# The character budget below is the real bound on prompt size. This count only stops a
# repository of tiny files from becoming a long list of them, so it has to leave room for
# the checkout's own root metadata before the ranked candidates begin: at 24 a backend
# spent sixteen slots on root files and cut off the middleware the task named at 29.
# The character constants are the DEFAULT budget: what an attempt runs under when the
# deployment has not declared the routed model's context window. They equal exactly what
# `repository_snapshot_budget` derives for a 200k-token window -- the current-generation
# size the 9baf758 interim measured against AB-Feature-212, where an ultra-tier attempt 0
# wrote a change set whose attempt-1 required union (prior files + imports + seam context +
# plan targets) exceeded the previous 80_000-character wall in BOTH repositories, and the
# 61- refusal correctly killed each workstream before the model was called. The budget must
# hold the required union of the largest first draft a configured model writes, or every
# attempt after that draft is impossible by arithmetic -- which is why the budget is now a
# function of the declared window (MODEL_CONTEXT_WINDOW_TOKENS) rather than a constant
# about an era. The file count stays a constant on purpose: its rationale above is
# model-independent.
_REPOSITORY_CONTEXT_MAX_FILES = 60
_REPOSITORY_CONTEXT_MAX_CHARACTERS = 160_000
_REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS = 16_000
_REPOSITORY_CONTEXT_MAX_INVENTORY_FILES = 1_000
# The snapshot's share of the model's context window. The other 80% is instructions,
# diagnostics, evidence, and the model's own output -- thinking and the answer share the
# window with everything the prompt carries.
_SNAPSHOT_WINDOW_TOKEN_SHARE = 0.20
# The same coarse characters-per-token ratio the interim sizing used (160k characters of
# snapshot ~= 40k tokens). Coarse is enough: the share above, not this ratio, is the dial.
_SNAPSHOT_CHARACTERS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class RepositorySnapshotBudget:
    """The character budget one attempt's repository snapshot ran under, and its origin.

    ``source`` is ``"declared:<model>:<window>"`` when the deployment declared the routed
    model's context window, and ``"default"`` otherwise -- recorded on the completion so a
    forensic pass can tell "this attempt ran under the declared budget" from "the
    declaration never crossed the compose boundary" without re-deriving anything.
    """

    max_characters: int
    per_file_max_characters: int
    source: str


def repository_snapshot_budget(
    model: str | None, declared_windows: Mapping[str, int]
) -> RepositorySnapshotBudget:
    """Derive the snapshot budget from the routed model's declared context window -- once.

    The key is the resolved model identifier, never the tier: a tier resolves to a model and
    the model resolves to a window, so the CUSTOM tier and every future tier inherit correct
    budgets with no new variables. The identity property is the safety proof, pinned in a
    test: a declared 200_000-token window derives exactly the default constants, so a
    declared current-generation model behaves byte-identically to an undeclared one -- and
    an undeclared model falls back to exactly those defaults (the temperature-pattern
    posture: absence of a declaration is never a punishment).
    """
    window = declared_windows.get(model) if model else None
    if window is None or window <= 0:
        return _DEFAULT_SNAPSHOT_BUDGET
    max_characters = int(window * _SNAPSHOT_WINDOW_TOKEN_SHARE * _SNAPSHOT_CHARACTERS_PER_TOKEN)
    return RepositorySnapshotBudget(
        max_characters=max_characters,
        # A tenth of the whole, which is what the default constants already encode: one
        # oversized file must never be able to spend the union's entire budget on itself.
        per_file_max_characters=max_characters // 10,
        source=f"declared:{model}:{window}",
    )


_DEFAULT_SNAPSHOT_BUDGET = RepositorySnapshotBudget(
    max_characters=_REPOSITORY_CONTEXT_MAX_CHARACTERS,
    per_file_max_characters=_REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS,
    source="default",
)
# Registry files are force-included ahead of the ranked snapshot, so each one is a file the
# task's own modules do not get. Reconnaissance names a handful at most -- the pilot repos
# yield nought to two from their conventions, then a structural list ordered by how many of
# the repository's own modules each file pulls in. The cap bounds the budget an unusually
# large list could take; the engineer is told about exactly the paths that survive it, so
# the prompt never claims a file is in the snapshot when it is not.
_MAX_REGISTRY_CONTEXT_PATHS = 6
# The plan names a handful of files at most, and each one force-included is a file the
# ranked snapshot does not get. Bounded for the same reason the registry list is.
_MAX_ASSIGNED_CONTEXT_PATHS = 6
# How many of the previous attempts' own files a retry may force-include. Double the other
# force-include caps, because a remediation's target set is routinely this wide -- 176's
# backend change was ten files -- but still bounded, so a sprawling lineage cannot evict
# the configs and registries ranked after it.
_MAX_PRIOR_ATTEMPT_CONTEXT_PATHS = 12
_REPOSITORY_CONTEXT_METADATA_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".lock", ".md", ".toml", ".yaml", ".yml"}
)
# Words that name a repository's own structure rather than this task's subject. They match
# most paths in a checkout, so counting them would rank every file equally and lose the
# signal. This is a vocabulary about repository layout, not about any one project.
_REPOSITORY_CONTEXT_GENERIC_TERMS = frozenset(
    {
        "app",
        "file",
        "files",
        "index",
        "lib",
        "main",
        "new",
        "package",
        "path",
        "project",
        "repo",
        "repository",
        "source",
        "src",
        "the",
        "use",
        "using",
    }
)
_REPOSITORY_CONTEXT_TERM_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
# Shortest word for which a shared prefix is evidence rather than coincidence.
_REPOSITORY_CONTEXT_MIN_PREFIX_MATCH = 4

_LOGGER = logging.getLogger(__name__)

# The hard ceiling on repair passes inside one attempt: a backstop against a pathological
# linter whose diagnostics never stabilise, not a policy. The loop's real bound is changed
# diagnostics. Four, because the longest mechanical chain observed live was three rules deep
# (AB-Feature-171: `render-result-naming` -> `no-node-access` -> `no-unnecessary-act`), so
# the ceiling covers the observed worst case plus one -- and four failed passes cost minutes
# where one destroyed-and-regenerated attempt costs 15-40.
_MAX_SOURCE_REPAIR_PASSES = 4
# The ceiling on wiring repair passes, which is lower than the one above it for a reason that
# is about the defect and not about caution. A lint chain surfaces the next rule as it clears
# the last, which is why four; adding a reference to a host file is one edit, and a pass that
# did not manage it is not going to manage it differently on a third telling. Bounded by the
# same changed-signature rule, so two is only the backstop.
_MAX_WIRING_REPAIR_PASSES = 2
# Every diagnostic this platform composes for the repair loop starts with one of these.
# Anything else -- a review finding, a platform sentence -- is not a check speaking about
# this workspace, and the loop must hand it back to the retry policy untouched.
#
# Everything after the first two is imported from the module that composes it rather than
# copied here as a bare sentence: a predicate coupled to a literal is a predicate that drifts
# away from its producer silently, and the two above are already an argument against adding
# more of them. That is why widening this to a new check is done by *tagging the diagnostic at
# its source* -- a third opaque string here would make the shape worse, not wider.
#
# What each entry admits is a different kind of claim. The typecheck entry admits the
# checkout's own compiler saying the code does not fit together, which is unambiguous: there
# is no assertion to weaken and no judgement to smuggle. The test entry admits a *failing
# test*, which is ambiguous in a way neither a lint rule nor a type error is -- see the
# assertion guard in `_verified_with_repairs`, which is the condition on which admitting it at
# all is safe. The reachability entry admits a deterministic wiring finding, which no model
# produced and which no rewriting of a test can make go away.
_MECHANICAL_DIAGNOSTIC_PREFIXES = (
    "Pre-commit source validation failed",
    "This repository's commit gate was already failing",
    TYPECHECK_DIAGNOSTIC_PREFIX,
    SCOPED_TEST_DIAGNOSTIC_PREFIX,
    REACHABILITY_DIAGNOSTIC_PREFIX,
)
# Bounds on what the repair prompt quotes, so the source regions can never displace the
# diagnostics they exist to explain: per file, and across every file one pass quotes.
_REPAIR_REGION_MAX_CHARACTERS = 4_800
_REPAIR_REGIONS_MAX_TOTAL_CHARACTERS = 12_000
# How many definition sites one repair pass may be shown. Bounded twice over -- by this and
# by whatever the diagnostics left of the character budget above, which they claim first --
# because a definition region exists to explain a diagnostic and must never displace one.
# Three, because the case this serves is a handful of unresolved names at most: a pass facing
# more than that is not failing for want of one import.
_MAX_DEFINITION_REGIONS = 3
# The largest file the repair pass will read for a region. Well above anything a linter
# meaningfully rejects line-by-line; a bound because this read bypasses the file tools.
_REPAIR_REGION_MAX_SOURCE_BYTES = 2_000_000
# Bounds on what the self-review pass is shown of the attempt's own files. Per file and in
# total, sized like the repository-context budget rather than the repair-region one: this
# pass judges requirement coverage of whole files, not one diagnostic's neighbourhood, and a
# review of a truncated file must know it was truncated -- the marker below says so in-line.
_SELF_REVIEW_FILE_MAX_CHARACTERS = 12_000
_SELF_REVIEW_FILES_MAX_TOTAL_CHARACTERS = 60_000
_SELF_REVIEW_TRUNCATION_MARKER = "\n[truncated for size: review the visible portion only]"


def _mechanical_diagnostics(diagnostics: Sequence[str]) -> bool:
    """Say whether every diagnostic is the commit gate's own lint/format output."""
    return bool(diagnostics) and all(
        diagnostic.startswith(_MECHANICAL_DIAGNOSTIC_PREFIXES) for diagnostic in diagnostics
    )


@dataclass(frozen=True, slots=True)
class _SourceRepair:
    """One in-attempt correction of the repository's own pre-commit verification."""

    paths: tuple[str, ...]
    response_id: str
    diagnostics: tuple[str, ...]
    # What executed this pass, `execution_metadata`-shaped. The audit of 175-180 could not
    # tell "the scoped-fix model never runs" from the artifacts; this is what says which
    # model repaired, and whether the scoped-fix role resolved or the primary stood in.
    execution: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _GateOutcome:
    """Everything the in-attempt verification ran, and every repair it took to get there."""

    formatting_commands: tuple[str, ...]
    # The narrowed test commands that actually executed, empty when the checkout configured
    # none, when the attempt changed no test file, or when narrowing would have changed the
    # question. Recorded either way: "the change's own tests were never run here" and "they
    # ran and passed" are different facts about an attempt.
    scoped_test_commands: tuple[str, ...]
    # The checkout's own typecheck commands that executed, empty when it configures none.
    # Recorded for the same reason as the line above: an attempt whose types were never
    # checked here and one whose types were checked and agreed are different attempts.
    typecheck_commands: tuple[str, ...]
    repairs: list[_SourceRepair]
    # What the in-attempt reachability check still had to say when this attempt gave up
    # trying to clear it. Empty is the ordinary case and means either that the check found
    # nothing or that a repair wired everything it named. Non-empty does *not* fail the
    # attempt: this layer is fast feedback, and the runtime's own gate is the authority that
    # decides -- see `_wired_with_repairs`.
    reachability_issues: tuple[str, ...] = ()
    # The candidates the inspection's bound left unexamined, so a truncated run can never be
    # read as a clean one. 46- calls a silent cap out by name and this is exactly that kind.
    reachability_candidates_dropped: tuple[str, ...] = ()
    # What the in-attempt placement check still had to say, on the same terms as the two
    # fields above it: empty is the ordinary case, and non-empty does not fail the attempt.
    # Recorded because this is the one check here that can be wrong about a legitimate
    # layout, and the decision to drop it has to be answerable from run records.
    assigned_file_issues: tuple[str, ...] = ()
    # What the deterministic design-value check still had to say. Empty is the ordinary case
    # and means either that the change used the design's colours and type or that the feature
    # cited no design at all. Non-empty does not fail the attempt, for the reason the check's
    # own module gives: a repository with a design system should write its token rather than
    # the literal, so a named value may be correctly absent.
    design_value_issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _InAttemptFindings:
    """What the deterministic inspections after the gate said about this workspace.

    All of them cost no model call, all are fast feedback rather than authority, and none
    fails the attempt. They are held together because the repair loop below takes one
    diagnostic set: a pass is told everything wrong with what it wrote, once.

    They are also *ordered*, which is the reason this type exists rather than three locals.
    A wiring finding and a placement finding give contradictory instructions -- one says
    leave the added module exactly where it is and edit a host, the other says the code
    belongs in a file the change never opened -- so placement is reported only when nothing
    is unreachable. Unreachable code is the defect the authoritative gate will stop, and it
    is answered first.

    The design finding sits outside that ordering because it contradicts neither: "you did not
    use the design's own colours" is true wherever the code ended up, and it is the only thing
    this platform can say deterministically about whether a change matches the design it was
    given -- every command a repository ships passes identically on an invented palette.
    """

    wiring_issues: tuple[str, ...] = ()
    placement_issues: tuple[str, ...] = ()
    design_issues: tuple[str, ...] = ()
    dropped_candidates: tuple[str, ...] = ()

    @property
    def issues(self) -> tuple[str, ...]:
        """The findings a repair pass is started for.

        Design findings are deliberately **not** here, and that is the difference between this
        check and the two beside it. A wiring finding and a placement finding each name
        something definitely wrong: code nothing reaches, code in a file the plan did not
        assign. A design-value finding names something that may be *correct* -- a repository
        with a design system should write `bg-surface-raised` rather than the literal
        `#0a0a0a`, and that reads here as the colour being absent.

        Spending repair passes on it was not merely wasteful, it was the defect that ended
        AB-Feature-231 and AB-Feature-232: both had clean wiring and placement, so this check
        alone started a remediation loop, and five consecutive provider faults inside it spent
        the workstream's whole fault allowance. AB-Feature-229, which ran before this check
        existed, completed.

        The findings still reach the prompt through `diagnostics` and the artifact through
        `_GateOutcome`, so nothing is hidden -- they are told, not repaired.
        """
        return (*self.wiring_issues, *self.placement_issues)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Every finding, tagged with the check that produced it, for the repair prompt."""
        return (
            *(reachability_diagnostic(issue) for issue in self.wiring_issues),
            *(assigned_file_diagnostic(issue) for issue in self.placement_issues),
            *(design_value_diagnostic(issue) for issue in self.design_issues),
        )


# What the repair is told rejected its work. Which one is used is read off the diagnostics
# rather than passed in, so the sentence cannot claim the commit gate spoke when the test
# runner did -- a repair told the wrong thing about its own failure looks for the defect in
# the wrong place.
_GATE_REJECTION = (
    "The changes you just wrote were rejected by this repository's own pre-commit "
    "verification, which runs before anything is staged. Its fixers have already applied "
    "everything they can rewrite automatically, so what remains needs a source change. "
    "Here is exactly what it reported:"
)
_TYPECHECK_REJECTION = (
    "The changes you just wrote passed this repository's pre-commit verification and were "
    "then rejected by its own typechecker. Nothing has been staged and nothing has been "
    "committed. A type error is this checkout's compiler saying the code does not fit "
    "together, so the fix is to make the types agree -- not to widen a type to `any`, cast "
    "the problem away, or add a suppression comment. Here is exactly what it reported:"
)
_TEST_REJECTION = (
    "The changes you just wrote passed this repository's pre-commit verification and were "
    "then rejected by its own test runner, narrowed to the test files this change touched. "
    "Nothing has been staged and nothing has been committed. Here is exactly what it "
    "reported:"
)
_WIRING_REJECTION = (
    "The changes you just wrote are correct as source -- this repository's own lint, format "
    "and test commands all accept them -- and part of what they add still cannot be reached "
    "when the application runs. Nothing references it, so it will never execute. This was "
    "established by reading the checkout itself, not by judgement, and it is the single "
    "largest reason implementations here get sent back. Here is exactly what was found:"
)
_PLACEMENT_REJECTION = (
    "The changes you just wrote are correct as source -- this repository's own lint, format "
    "and test commands all accept them -- and they put new code in a new module beside the "
    "file this workstream's plan named, leaving that file untouched. Nothing has been staged "
    "and nothing has been committed. This was established by comparing the plan's own "
    "assignment against `git status`, not by judgement, and it is a rejection review has "
    "already spent whole attempts on. Here is exactly what was found:"
)
# The reachability clause, and the reason this repair is unlike the two above it. Both of
# those are told to change the file the diagnostic names; this one is told to change a
# *different* file, and to leave the file it names alone. Getting that backwards is how a
# module gets rewritten until it is unreachable in a new way.
_WIRING_INTEGRITY_CLAUSE = (
    "\n\nOne of these findings is about wiring. Do not rewrite, rename, move or delete the "
    "code that cannot be reached -- it is finished, and the defect is that nothing calls it. "
    "Return only the host file named above, as an `edits` entry so the rest of that file is "
    "untouched, adding the reference in the form that file already uses for the modules it "
    "reaches. If a finding names a host you can see is genuinely the wrong one, reference the "
    "code from whichever existing file this change is actually for and say why in your "
    "summary; leaving it unreachable is the one outcome that cannot be accepted."
)
# The assertion guard's first half, and the reason Part C can ship at all. A model asked to
# make diagnostics go away can satisfy a failing test by weakening its assertion, and no
# later gate catches that -- the suite is green and smaller. AB-Feature-182 is the
# precedent: three remediation attempts edited a test fixture when the defect was in the
# product file. The second half is deterministic and lives in the loop; this half exists so
# the loop rarely has to fire it.
_TEST_INTEGRITY_CLAUSE = (
    "\n\nOne of these failures is a test that this repository's own runner ran. Fix the code "
    "under test, never the test itself. Do not edit, weaken, delete, skip or narrow any "
    "assertion, expectation or matcher, and do not change the data a test asserts against, "
    "in order to make it pass. If the test is correct and the implementation is what is "
    "wrong, change the implementation. If you conclude the test itself is wrong -- that it "
    "asserts something the requirement never asked for -- say so in your summary and change "
    "nothing, because that is a judgement for review and not a repair. Repairing genuinely "
    "broken test scaffolding is allowed and is often the fix: an import that does not "
    "resolve, a mock or fake that throws, a setup step that never ran, a helper called with "
    "the wrong shape. Repair that, and leave every assertion exactly as it stands."
)


# What the one bounded self-review correction is told. Unlike the repair rejections above,
# nothing deterministic rejected this source -- every configured check is green -- so the
# sentence says what spoke (the attempt's own review of its finished work) and holds the
# correction to exactly the findings, on the same reasoning `_source_repair_instructions`
# gives: a correction told to "improve things" trades the reported gap for new ones. The
# assertion sentence is here because the re-validation that follows runs the change's own
# tests, and a correction that satisfied a finding by quieting a claim would pass it.
_SELF_REVIEW_CORRECTION = (
    "\n\nYou reviewed the implementation you just finished, and the review found the "
    "specific gaps listed below. Every check this repository configures is green as the "
    "files stand, so correct exactly these findings and nothing else: do not rewrite "
    "working code, do not rename or restructure anything a finding does not name, and "
    "prefer `edits` entries so the rest of each file stays untouched. Do not weaken, "
    "delete, skip or narrow any test assertion, expectation or matcher in the course of a "
    "correction. Here is what the review found:\n"
)


def _self_review_correction_instructions(instructions: str, diagnostics: Sequence[str]) -> str:
    """Anchor the one correction pass to the review's own findings, verbatim."""
    return f"{instructions}{_SELF_REVIEW_CORRECTION}" + "\n".join(diagnostics)


def _source_repair_instructions(
    instructions: str,
    diagnostics: Sequence[str],
    source_regions: Sequence[tuple[str, int | None, str]] = (),
    definition_regions: Sequence[tuple[str, int | None, str]] = (),
) -> str:
    """Anchor the correction to the diagnostics: permit what they require, forbid the rest.

    Deliberately narrow, for the reason the product manager's equivalent is: an attempt told
    to "try again" trades the reported defect for new ones. But narrowness must not forbid
    the fix itself. The previous wording ended "do not restructure it, do not rename
    anything" -- and AB-Feature-171's repair then ran twice against diagnostics that demanded
    exactly a rename (`render-result-naming-convention`) and exactly a restructure
    (`no-node-access`), and could clear nothing. A rename the diagnostic explicitly asks for
    is the fix; a rename it never mentioned is drift.

    ``source_regions`` are the workspace's current bytes around each location a diagnostic
    names, and they are the difference between a repair and a guess: 176's three repair
    passes each regenerated an 11.5 KB file from memory because the prompt carried the
    eslint line and nothing of the source it pointed at, and each rewrite reproduced the
    same one-token defect. A repair that can read `const nestedDocument;` beside the eslint
    line is a one-edit fix.

    ``definition_regions`` answer the question those windows structurally cannot. The
    commonest mechanical diagnostic there is says `'permissionUtils' is not defined`, and the
    thirty lines around the line it names contain the *use* -- never the definition. To fix
    it the pass has to know which module exports that name and under what specifier, and it
    has no way to look: there is no tool call here and the snapshot was frozen before the
    coding call. So the checkout is searched for the name deterministically beforehand, and
    what it defines is quoted under a heading of its own. Separately, because these bytes are
    not where the defect is: telling a pass that the defect is inside a file it was shown only
    for reference is how a working module gets rewritten.

    The rejection is attributed to whatever actually spoke -- the typechecker, the test
    runner, the wiring inspection, the plan's own file assignment -- rather than to the commit
    gate, and where a check has a clause of its own it is appended. Neither is optional: a
    repair that thinks the linter rejected a test failure looks for a style defect that is not
    there, and one that thinks the linter rejected a wiring finding rewrites the very file it
    was told to leave alone.

    The typecheck rejection carries no integrity clause of the kind the test one does, and
    that asymmetry is the point of admitting it: a failing test can be satisfied by weakening
    what it claims, and a type error cannot. What its sentence does say is that widening to
    `any`, casting, or suppressing is not making the types agree -- the same shape of
    instruction, aimed at the only way this check could be silenced rather than answered.
    """
    failing_tests = any(is_scoped_test_diagnostic(diagnostic) for diagnostic in diagnostics)
    unreachable = any(is_reachability_diagnostic(diagnostic) for diagnostic in diagnostics)
    reported = "\n".join(diagnostics)
    quoted = ""
    if source_regions:
        quoted = (
            "\n\nThe current workspace content around each location the diagnostics name, "
            "exactly as the files stand now. The defect is inside what is quoted; correct "
            "it with the smallest edit the diagnostic requires, and copy any `find` text "
            "verbatim from these bytes:\n" + _quoted_blocks(source_regions)
        )
    defined = ""
    if definition_regions:
        defined = (
            "\n\nWhere this checkout defines the names those diagnostics mention, read from "
            "the files themselves. Use these to write the exact specifier and the exact shape "
            "rather than guessing at either -- a guessed import path fails the next check or "
            "fails at review. The defect is not in these files: do not change one unless a "
            "diagnostic above names it:\n" + _quoted_blocks(definition_regions)
        )
    # Attribution is by what the diagnostics *are*, not by what most of them are: a set that
    # is entirely one check's gets that check's sentence, and a mixed set gets the gate's,
    # which is the only one of the four that claims nothing about the others.
    only_tests = failing_tests and all(
        is_scoped_test_diagnostic(diagnostic) for diagnostic in diagnostics
    )
    only_unreachable = unreachable and all(
        is_reachability_diagnostic(diagnostic) for diagnostic in diagnostics
    )
    only_types = bool(diagnostics) and all(
        is_typecheck_diagnostic(diagnostic) for diagnostic in diagnostics
    )
    only_misplaced = bool(diagnostics) and all(
        is_assigned_file_diagnostic(diagnostic) for diagnostic in diagnostics
    )
    if only_tests:
        rejection = _TEST_REJECTION
    elif only_unreachable:
        rejection = _WIRING_REJECTION
    elif only_types:
        rejection = _TYPECHECK_REJECTION
    elif only_misplaced:
        rejection = _PLACEMENT_REJECTION
    else:
        rejection = _GATE_REJECTION
    return (
        f"{instructions}\n\n"
        f"{rejection}\n"
        f"{reported}"
        f"{quoted}"
        f"{defined}"
        f"{_TEST_INTEGRITY_CLAUSE if failing_tests else ''}"
        f"{_WIRING_INTEGRITY_CLAUSE if unreachable else ''}\n\n"
        "Return only the files that must change to clear these diagnostics, and make exactly "
        "the change each diagnostic requires: when a diagnostic demands a rename, rename "
        "precisely what it names; when it demands restructuring, restructure exactly that "
        "code. Change nothing the diagnostics do not require: do not rewrite working code, "
        "do not rename or restructure anything the diagnostics did not name, and do not "
        "remove behaviour the diagnostics did not mention. If a diagnostic names a rule "
        "that your code genuinely needs to break, restructure that code so the rule is "
        "satisfied rather than disabling the rule."
    )


def _diagnostic_file_locations(
    diagnostics: Sequence[str], workspace_root: Path
) -> dict[str, int | None]:
    """Resolve the files these diagnostics name into workspace-relative paths, with lines.

    Gate and validation diagnostics spell a file two ways this code could not previously
    use: an absolute workspace path, and a relative path with a `line:col` suffix. 176's
    gate said `/workspaces/feature-.../server/services/app/appBulkImport.service.js:179:23`
    three times, and the membership test `Path(item) in existing_files` matched none of
    them, so the one file the attempt had to edit was never promoted into required context.

    Absolute paths are stripped to the workspace root through ``resolve_workspace_path``,
    which also refuses anything outside it -- a diagnostic naming another feature's
    workspace resolves to nothing. Only paths naming a file actually present here survive.
    First-seen order; the line kept is the first one a diagnostic attached to that path.
    """
    locations: dict[str, int | None] = {}
    for token, line in diagnostic_file_references(diagnostics):
        try:
            resolved = resolve_workspace_path(workspace_root, token)
        except (ValueError, OSError):
            continue
        if not resolved.is_file():
            continue
        relative = resolved.relative_to(workspace_root).as_posix()
        if relative not in locations:
            locations[relative] = line
    return locations


def _quoted_blocks(regions: Sequence[tuple[str, int | None, str]]) -> str:
    """Render quoted source regions as labelled blocks, each naming where it came from."""
    return "\n\n".join(
        f"===== {f'{path}, around line {line}' if line is not None else path} =====\n{region}"
        for path, line, region in regions
    )


def _quoted_regions(
    workspace_root: Path, locations: Sequence[tuple[str, int | None]], *, budget: int
) -> tuple[list[tuple[str, int | None, str]], int]:
    """Read each of these workspace locations as a bounded region, and say what it cost.

    Best-effort by design, like the repair it feeds: a file that cannot be read, decodes as
    something other than UTF-8, or carries key material contributes nothing, and the repair
    proceeds on whatever else it has exactly as it always has. Being quoted for reference is
    not permission to leak, so the key-material test applies here identically to how it
    applies to a file the diagnostics named.
    """
    regions: list[tuple[str, int | None, str]] = []
    total = 0
    for path, line in locations:
        try:
            target = resolve_workspace_path(workspace_root, path)
            if target.stat().st_size > _REPAIR_REGION_MAX_SOURCE_BYTES:
                continue
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if carries_key_material(content):
            continue
        region, _start, _end = source_region(
            content, line, max_characters=_REPAIR_REGION_MAX_CHARACTERS
        )
        region = redact_source_credentials(region)
        if not region.strip():
            continue
        if total + len(region) > budget:
            break
        total += len(region)
        regions.append((path, line, region))
    return regions, total


def _diagnostic_source_regions(
    workspace_root: Path, diagnostics: Sequence[str]
) -> list[tuple[str, int | None, str]]:
    """Read the workspace region each diagnostic points at, redacted and bounded."""
    regions, _total = _quoted_regions(
        workspace_root,
        list(_diagnostic_file_locations(diagnostics, workspace_root).items()),
        budget=_REPAIR_REGIONS_MAX_TOTAL_CHARACTERS,
    )
    return regions


def _repair_regions(
    workspace_root: Path,
    diagnostics: Sequence[str],
    definitions: DefinitionIndex | None,
) -> tuple[list[tuple[str, int | None, str]], list[tuple[str, int | None, str]]]:
    """Return what the repair may read: the diagnostics' own locations, then the definitions.

    The order is the budget rule and it is not negotiable. Definition regions exist to explain
    a diagnostic, so the diagnostics' own windows claim the character bound first and the
    definitions get whatever is left -- which may be nothing, in which case the pass is
    exactly as well informed as it was before this lookup existed.

    A definition site already quoted as a diagnostic location is dropped rather than repeated:
    the same bytes twice under two headings would spend the budget saying one thing.

    The lookup runs over the workspace as it stands, so a name the *attempt itself* defined in
    a module it just wrote resolves like any other. Absent an index -- a composition that
    passes none, which is every existing caller -- this degrades to exactly today's behaviour.
    """
    locations = _diagnostic_file_locations(diagnostics, workspace_root)
    regions, used = _quoted_regions(
        workspace_root, list(locations.items()), budget=_REPAIR_REGIONS_MAX_TOTAL_CHARACTERS
    )
    if definitions is None:
        return regions, []
    sites = definitions.sites(definition_symbols(diagnostics), limit=_MAX_DEFINITION_REGIONS)
    defined, _cost = _quoted_regions(
        workspace_root,
        [(site.path, site.line) for site in sites if site.path not in locations],
        budget=_REPAIR_REGIONS_MAX_TOTAL_CHARACTERS - used,
    )
    return regions, defined


def _guarded_test_paths(paths: Sequence[str], diagnostics: Sequence[str]) -> list[str]:
    """Name every test file a repair pass could weaken while clearing these diagnostics.

    The attempt's own test files, plus any the diagnostics name. The first set is what the
    narrowed run was given, so it is the complete answer in practice; the second is there
    because a runner is free to blame a suite the attempt did not write, and a guard that
    only watched the expected files would be a guard with a documented way around it.
    """
    named = [path for path, _line in diagnostic_file_references(diagnostics)]
    return changed_test_paths([*paths, *named])


def _read_sources(workspace_root: Path, paths: Sequence[str]) -> dict[str, str]:
    """Read these workspace files as text, skipping any that cannot be read as text.

    Bounded and best-effort, like every other read in the repair path. A file this cannot
    read contributes no `before` bytes and is therefore never compared -- which is the safe
    direction: the guard refuses a repair on evidence, never on the absence of it.
    """
    sources: dict[str, str] = {}
    for path in paths:
        try:
            target = resolve_workspace_path(workspace_root, path)
            if not target.is_file() or target.stat().st_size > _REPAIR_REGION_MAX_SOURCE_BYTES:
                continue
            sources[path] = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
    return sources


class EngineerAgent:
    """Apply task-plan changes through a configured coding executor and safe workspace tools."""

    def __init__(
        self,
        *,
        prompt_loader: PromptLoader,
        coding_executor: CodingExecutor,
        # The boundary the in-attempt source repair runs on: the SCOPED_FIX role's model,
        # resolved by the caller from the feature's pinned platform and tier. Optional
        # because deployments and doubles that configure no scoped-fix boundary must keep
        # the repair loop: absent, the primary coding executor repairs, exactly as before.
        scoped_fix_executor: CodingExecutor | None = None,
        file_tools_factory: FileToolsFactory = WorkspaceFileTools,
        repository_tools_factory: RepositoryToolsFactory = WorkspaceRepositoryTools,
        git_service_factory: GitServiceFactory | None = None,
        cancellation_token: CancellationToken | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
        source_formatter: SourceFormatter | None = None,
        # The repository's own test runner, narrowed to the test files this attempt changed
        # and asked *inside* the attempt. Optional for the same reason the formatter is:
        # absent, nothing runs and the attempt behaves exactly as it did before the change's
        # own tests were ever consulted here.
        scoped_test_runner: ScopedTestRunner | None = None,
        # The repository's own typecheck, asked *inside* the attempt. Optional for the same
        # reason the formatter is: absent, nothing runs and the attempt behaves exactly as it
        # did when the first thing to typecheck a change was the reviewer, a full outer
        # attempt later.
        typecheck_runner: TypecheckRunner | None = None,
        # The deterministic wiring inspection, asked *inside* the attempt so a module nothing
        # reaches is repaired here rather than costing a whole outer attempt -- 24% of live
        # attempt-0s across runs 66 to 106. Optional like the two boundaries above it, and
        # non-authoritative by construction: the runtime asks the same question again
        # afterwards and that answer is the one that decides.
        reachability_checker: ReachabilityChecker | None = None,
        # The deterministic placement check: the plan named an existing file by exact path,
        # this change never opened it, and it added a module beside it instead. All three, or
        # nothing. Optional like every boundary above it, non-authoritative in the same way,
        # and confined to this one seam on purpose -- it is the one check here that can be
        # wrong about a legitimate layout, and it has to be droppable without ceremony.
        assigned_file_checker: AssignedFileChecker | None = None,
        # The deterministic design-value inspection, and the design it compares against.
        # Both optional: a composition that supplies neither runs exactly as it did before.
        design_value_checker: DesignValueChecker | None = None,
        # Bytes-by-id, so the coding call can be shown the frames it is building.
        attachments: AttachmentContentSource | None = None,
        design_contents: Sequence[Mapping[str, Any]] = (),
        # One bounded review of the finished work, on the CODING role at the feature's pinned
        # tier, gating `completion_status` and nothing further. Optional like every model
        # boundary above it: absent, no review runs and `completed` means exactly what it
        # meant before Part E -- which is also what every existing composition gets.
        self_reviewer: SelfReviewer | None = None,
        dependency_synchronizer: DependencySynchronizer | None = None,
        lint_capability_probe: LintCapabilityProbe | None = None,
        previous_attempt_diff: str | None = None,
        # The files the bounded capture withheld from `previous_attempt_diff`, when any
        # were. Named in the prompt so the byte-identical instruction binds only to the
        # files actually shown, and the model is never asked to reproduce what it cannot see.
        previous_attempt_withheld_files: Sequence[str] = (),
        # Whether the previous attempt's files are still in this workspace. When they are,
        # the retry is asked to correct them in place with `edits` rather than to regenerate
        # the change set from a diff shown in the prompt.
        previous_attempt_preserved: bool = False,
        # Which configured model role is executing this attempt, and why. Decided by the
        # orchestration layer that owns the retry policy; this agent records it and tells the
        # model what kind of job it has been given, and decides nothing about it.
        model_routing: ModelRoutingDecision | None = None,
        # The snapshot character budget this attempt runs under, derived ONCE by the caller
        # from the routed model's declared context window (`repository_snapshot_budget`).
        # Defaults to the module constants so every existing composition -- engineer_node
        # and the test constructions among them -- is byte-identical to before.
        snapshot_budget: RepositorySnapshotBudget = _DEFAULT_SNAPSHOT_BUDGET,
    ) -> None:
        """Inject all code-writing and workspace-inspection boundaries."""
        self._prompt_loader = prompt_loader
        self._coding_executor = coding_executor
        self._scoped_fix_executor = scoped_fix_executor
        self._file_tools_factory = file_tools_factory
        self._repository_tools_factory = repository_tools_factory
        # Kept in the constructor for source compatibility with existing compositions.  Git
        # mutations deliberately live in ``ApprovedChangePublisher`` now: allowing this coding
        # boundary to commit or push would publish work before the reviewer has approved it.
        del git_service_factory
        self._cancellation_token = cancellation_token
        self._operation_executor = operation_executor
        self._source_formatter = source_formatter or NullSourceFormatter()
        self._scoped_test_runner = scoped_test_runner or NullScopedTestRunner()
        self._typecheck_runner = typecheck_runner or NullTypecheckRunner()
        self._reachability_checker = reachability_checker or NullReachabilityChecker()
        self._assigned_file_checker = assigned_file_checker or NullAssignedFileChecker()
        # Nullable rather than a null object, like the scoped-fix executor above: "no review
        # was composed" and "a review was composed and could not run" are different facts,
        # and only the second earns a record on the attempt.
        self._self_reviewer = self_reviewer
        self._dependency_synchronizer = dependency_synchronizer or NullDependencySynchronizer()
        self._lint_capability_probe = lint_capability_probe or NullLintCapabilityProbe()
        self._design_value_checker = design_value_checker
        self._attachments = attachments
        self._previous_attempt_diff = previous_attempt_diff
        # The design this workstream was assigned, at build fidelity, for the deterministic
        # value check below. Empty for every feature that cited nothing, which is most of
        # them -- and an empty sequence makes the check return nothing, so those attempts are
        # byte-identical to what they were before it existed.
        self._design_contents = tuple(design_contents)
        self._previous_attempt_withheld_files = tuple(previous_attempt_withheld_files)
        self._previous_attempt_preserved = previous_attempt_preserved
        self._model_routing = model_routing
        self._snapshot_budget = snapshot_budget

    async def _verified(
        self,
        workspace_root: Path,
        paths: Sequence[str],
        prior_diagnostics: Sequence[str] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Run the commit gate, then the types, then the change's own tests, over these paths.

        Order is the whole design: a deeper check run over source a cheaper one has already
        rejected spends its minutes reporting defects that were named first, so the gate
        speaks, then the typechecker, and only then the test runner. A type error also makes
        a test run worth very little -- a suite exercising code the compiler rejects reports
        the same defect at ten times the price -- so each check is asked only of source the
        one before it accepted.

        Every verdict leaves here as the same ``SourceValidationError`` the gate has always
        raised, which is what lets one bounded repair loop clear all three without knowing the
        difference between them.

        A check failing *to run* is not a verdict. Anything either boundary raises is
        swallowed and the attempt proceeds exactly as it did before that check was consulted
        here: these exist to make a known failure cheaper, never to invent a new way to fail.
        """
        formatting = await self._source_formatter.format_paths(workspace_root, paths)
        try:
            typecheck = await self._typecheck_runner.run_typecheck(workspace_root)
        except CancellationRequested:
            raise
        except Exception as typecheck_error:  # noqa: BLE001 - the typecheck is best-effort
            _LOGGER.warning(
                "the repository's own typecheck could not be run in-attempt: %s. The attempt "
                "proceeds and the reviewer's own run decides "
                "[outcome=typecheck_unavailable agent=engineer]",
                type(typecheck_error).__name__,
            )
            typecheck = TypecheckOutcome()
        if typecheck.diagnostics:
            raise SourceValidationError(typecheck.diagnostics)
        try:
            outcome = await self._scoped_test_runner.run_changed_tests(
                workspace_root, paths, prior_diagnostics
            )
        except CancellationRequested:
            raise
        except Exception as runner_error:  # noqa: BLE001 - the narrowed run is best-effort
            _LOGGER.warning(
                "the change's own tests could not be run in-attempt: %s. The attempt "
                "proceeds and the reviewer's own run decides "
                "[outcome=scoped_tests_unavailable agent=engineer]",
                type(runner_error).__name__,
            )
            return formatting, typecheck.commands, ()
        if outcome.diagnostics:
            # The structured results ride out with the rejection, not just the sentences the
            # repair loop reads. A scoped-test rejection is the one commit-gate verdict that
            # says something about *convergence* rather than about a rule still being broken,
            # and the counter that measures convergence reads recorded validation results --
            # which a gate-rejected attempt had none of.
            rejection = SourceValidationError(outcome.diagnostics)
            rejection.validation_results = outcome.validation_results
            raise rejection
        return formatting, typecheck.commands, outcome.commands

    def _refused_assertion_repair(
        self,
        workspace_root: Path,
        *,
        before: dict[str, str],
        repaired_paths: Sequence[str],
    ) -> list[str]:
        """Put back a repair that made a test pass by changing what it asserts.

        The deterministic half of the guard that lets a failing test into this loop at all.
        A pass that touched nothing but test files and changed a test's assertion shape has
        not repaired anything -- it has made the suite agree with the implementation, which
        is the one outcome no later gate can catch, because the suite is green afterwards.

        Restoring the bytes is not optional. The workspace survives a mechanical failure by
        design, so a weakened assertion left on disk would be inherited by the next attempt
        and reviewed as though the attempt had written it.

        A pass that also touched production files is *recorded* and not refused: it changed
        the code under test, which is what it was asked to do, and refusing it would reject
        the legitimate case where an implementation fix requires the fixture around it to
        move with it. That boundary is deliberate and it is the guard's known limit.
        """
        weakened = weakened_test_paths(before, _read_sources(workspace_root, list(before)))
        if not weakened:
            return []
        if not all(classify_file_change(path) == "test" for path in repaired_paths):
            _LOGGER.warning(
                "an in-attempt repair changed what %s asserts while also editing production "
                "code; it is recorded rather than refused "
                "[outcome=repair_changed_assertions agent=engineer]",
                ", ".join(weakened),
            )
            return []
        for path in weakened:
            try:
                resolve_workspace_path(workspace_root, path).write_text(
                    before[path], encoding="utf-8"
                )
            except (OSError, ValueError):
                # Best-effort, like every other write in this loop. The attempt is failing
                # either way; what matters is that it is not *silently* failing.
                _LOGGER.warning(
                    "the refused repair's edit to %s could not be undone [agent=engineer]",
                    path,
                )
        return weakened

    async def _verified_with_repairs(
        self,
        workspace_root: Path,
        paths: list[str],
        *,
        instructions: str,
        input_text: str,
        definitions: DefinitionIndex | None = None,
        prior_diagnostics: Sequence[str] = (),
    ) -> _GateOutcome:
        """Verify, repairing until the diagnostics stop changing.

        The repository's own fixers run first and resolve everything mechanical. What
        reaches here is a rule no fixer can rewrite -- an unused import the linter will not
        remove for you, a test-style rule -- and clearing one such rule routinely surfaces
        the next: AB-Feature-171's frontend went `render-result-naming` -> `no-node-access`
        -> `no-unnecessary-act`, and with exactly one repair per attempt each remaining rule
        cost a full 15-40-minute re-implementation cycle.

        So the repair loops inside the one attempt the retry policy already granted, bounded
        by *changed diagnostics rather than by count*: another pass runs only while the
        diagnostic signature changed, the source actually changed, and the diagnostics are
        still a check of this workspace speaking. A repeating signature, a pass that changes
        nothing, or a diagnostic that is neither the gate nor the runner stops it on the
        spot -- those belong to the retry policy and the reviewer. This loop grants nothing,
        raises nothing, and changes no counter; `_MAX_SOURCE_REPAIR_PASSES` is only a
        backstop against a pathological check whose output never stabilises.

        The change's own tests are admitted here too, narrowed by `scoped_test_command` to
        the test files this attempt touched. That reverses what this docstring used to say --
        that a test failure "belongs to the retry policy and the reviewer, never to this
        loop" -- and it may only be reversed while carrying the guard that decision was
        protecting. A failing test is ambiguous where a lint rule is not: it may be broken
        scaffolding, or it may be a correct test rejecting the implementation. The repair
        instruction forbids editing assertions, expectations and test data, and
        `_refused_assertion_repair` puts back any pass that touched nothing but test files
        and changed what one of them asserts, stopping the loop with a named outcome. Without
        that half, this loop would make AB-Feature-182's failure -- three attempts editing a
        fixture when the defect was in the product file -- cheaper to reach rather than
        dearer.

        Reachability is *not* asked here. It runs afterwards, in `_wired_with_repairs`, for
        the reason that method's docstring gives: a wiring finding is the only diagnostic in
        this agent that must never fail the attempt, and mixing it into this loop's escape
        paths would make it one that does.

        ``definitions`` is what makes a missing import repairable here rather than at review.
        The prompt otherwise quotes only the lines a diagnostic names, which for an
        unresolved identifier hold the use and never the definition -- so the pass guessed a
        specifier, and a guess fails the next pass or fails the reviewer. The lookup is
        deterministic and runs before the call: no tool loop, no second boundary, and nothing
        the model asks for.

        The pass runs on the SCOPED_FIX role's model when one was injected, because that is
        exactly the job the role exists for -- a localized mechanical correction -- and until
        it was wired here every per-tier scoped-fix selection was dead configuration on this
        path: 175-180 never executed it once. The prompt quotes the workspace bytes around
        each named location for the same audit's reason: three blind full-file rewrites
        reproduced one typo that a single sighted edit removes. Which model repaired, and
        whether the role resolved or the primary stood in, is recorded per pass and surfaced
        as `source_repair_execution` in the completion metadata.

        Two properties keep this safe on the most delicate path in the platform:

        * **No pass is journaled.** A second ``RUN_CODING_EXECUTOR`` record under one child
          attempt would give crash recovery two completed coding effects to choose between,
          and that reconciliation is the single most re-broken invariant here. Passing no
          operation executor leaves exactly one journaled effect per attempt, as now. The
          writes are workspace-local and rewriting a file with the same bytes is idempotent,
          so an interrupted repair replays as the whole attempt replays today.
        * **Every failure degrades to current behaviour.** If the model errors, if a pass
          changes nothing, or if verification never stabilises, this raises
          ``SourceValidationError`` just as it always has.

        What it no longer does is raise *silently*. The passes that ran are stamped onto the
        rejection on the way out, at one place rather than at each of the loop's seven escapes,
        so a gate-rejected attempt records the same repair history a successful one does.
        Zero passes is then a fact the record states rather than the absence of a record: it
        was the difference between "the scoped-fix loop never engaged" and "it engaged four
        times and lost", and until now the artifacts could not tell them apart.
        """
        repairs: list[_SourceRepair] = []
        try:
            return await self._repair_until_verified(
                workspace_root,
                paths,
                repairs,
                instructions=instructions,
                input_text=input_text,
                definitions=definitions,
                prior_diagnostics=prior_diagnostics,
            )
        except SourceValidationError as rejected:
            rejected.source_repairs = tuple(repairs)
            raise

    async def _repair_until_verified(
        self,
        workspace_root: Path,
        paths: list[str],
        repairs: list[_SourceRepair],
        *,
        instructions: str,
        input_text: str,
        definitions: DefinitionIndex | None = None,
        prior_diagnostics: Sequence[str] = (),
    ) -> _GateOutcome:
        """Run the repair loop `_verified_with_repairs` documents, accumulating into ``repairs``.

        Split out for one reason: ``repairs`` has to outlive the raise. It is the caller's list,
        appended to here, so whichever of this loop's escapes fires, the caller still holds
        every pass that ran and can attach them to the rejection.
        """
        try:
            formatting, typechecks, scoped_tests = await self._verified(
                workspace_root, paths, prior_diagnostics
            )
            return _GateOutcome(formatting, scoped_tests, typechecks, [])
        except SourceValidationError as first:
            first.modified_paths = tuple(paths)
            error = first
        merged = list(paths)
        last_signature = diagnostic_signature(error.diagnostics)
        repair_executor = self._scoped_fix_executor or self._coding_executor
        for pass_number in range(1, _MAX_SOURCE_REPAIR_PASSES + 1):
            if not _mechanical_diagnostics(error.diagnostics):
                # Not a check of this workspace speaking. A review finding or a platform
                # sentence belongs to the retry policy and the reviewer, never to this loop.
                _LOGGER.warning(
                    "engineer source repair refused pass %d: the diagnostics are not this "
                    "workspace's own checks [outcome=repair_not_mechanical agent=engineer]",
                    pass_number,
                )
                raise error
            if self._cancellation_token is not None:
                await self._cancellation_token.raise_if_cancelled()
            # The bytes of every test file this pass could weaken, captured before the model
            # is allowed to touch them. Only when a failing test is what it is being asked
            # to clear: a lint repair has no assertion to protect and pays nothing here.
            guarded = (
                _read_sources(workspace_root, _guarded_test_paths(merged, error.diagnostics))
                if any(is_scoped_test_diagnostic(item) for item in error.diagnostics)
                else {}
            )
            quoted, defined = _repair_regions(
                workspace_root,
                error.diagnostics,
                definitions.including(merged) if definitions is not None else None,
            )
            try:
                repaired = await repair_executor.execute(
                    workspace_root=workspace_root,
                    instructions=_source_repair_instructions(
                        instructions, error.diagnostics, quoted, defined
                    ),
                    input_text=input_text,
                    cancellation_token=self._cancellation_token,
                    # Deliberately unjournaled; see the docstring.
                    operation_executor=None,
                )
            except CancellationRequested:
                raise
            except Exception as model_error:  # noqa: BLE001 - the repair is best-effort
                # In the message, not in `extra`: the deployment's plain formatter drops
                # `extra`, and §1.6 needed journal-timestamp forensics to establish the
                # repair had run at all. That must never be necessary again.
                _LOGGER.warning(
                    "engineer source repair did not run: pass %d raised %s against %d "
                    "diagnostic(s) [outcome=repair_model_error agent=engineer]",
                    pass_number,
                    type(model_error).__name__,
                    len(error.diagnostics),
                )
                if is_transport_fault(model_error):
                    # The transport failed, so this attempt has no verdict about its source
                    # to report -- the check that rejected it was never answered. Raised as
                    # itself so the child-workstream loop sees the provider fault it is, and
                    # spends its fault allowance rather than an attempt.
                    #
                    # AB-Feature-215 is why. A repair call was accepted and sent nothing for
                    # thirty minutes; the gate's `SourceValidationError` was re-raised in its
                    # place, so a dead socket was recorded as `validation_source_failure`,
                    # charged a full attempt, and sent the next one to fix source that had
                    # never been the problem. Its sibling repository hit the identical fault
                    # on a journaled call, was classified correctly, and cost nothing but
                    # time. The only difference between them was which call had a journal row.
                    raise
                # Everything else is the provider answering, or this adapter's own decision,
                # and both are settled: the attempt fails on the diagnostics it already had,
                # exactly as before.
                raise error from None
            try:
                repaired_paths = [
                    resolve_workspace_path(workspace_root, modified_file)
                    .relative_to(workspace_root)
                    .as_posix()
                    for modified_file in repaired.modified_files
                ]
            except ValueError:
                # A repair that names a path outside the workspace is a malformed repair,
                # not a platform defect: the correction is best-effort, so the attempt
                # fails on the diagnostics the gate already produced, exactly as when the
                # repair model errors. WorkspacePathError is a ValueError subclass.
                _LOGGER.warning(
                    "engineer source repair returned an unusable path: pass %d is "
                    "discarded and the attempt fails on the gate's own diagnostics "
                    "[outcome=repair_model_error agent=engineer]",
                    pass_number,
                )
                raise error from None
            if not repaired_paths:
                # Nothing was rewritten, so re-verifying would ask the same question of the
                # same bytes and get the same answer.
                _LOGGER.warning(
                    "engineer source repair changed nothing: pass %d returned no files "
                    "against %d diagnostic(s), so the attempt fails on what the gate "
                    "already said [outcome=repair_changed_nothing agent=engineer]",
                    pass_number,
                    len(error.diagnostics),
                )
                raise error from None
            weakened = (
                self._refused_assertion_repair(
                    workspace_root, before=guarded, repaired_paths=repaired_paths
                )
                if guarded
                else []
            )
            if weakened:
                # The one stop this loop makes that is not about diagnostics: the pass was
                # undone, so re-verifying would only reproduce the failure it was given, and
                # a second pass would be asked the same question that produced this answer.
                _LOGGER.warning(
                    "engineer source repair refused pass %d: it changed only test files and "
                    "changed what %s asserts, so it was discarded and those files restored "
                    "[outcome=%s agent=engineer]",
                    pass_number,
                    ", ".join(weakened),
                    ASSERTION_GUARD_OUTCOME.lower(),
                )
                refused = SourceValidationError(
                    (assertion_guard_diagnostic(weakened), *error.diagnostics)
                )
                refused.modified_paths = tuple(merged)
                refused.terminal_outcome = ASSERTION_GUARD_OUTCOME
                # The run this refused repair was asked to clear, carried on rather than
                # dropped. This is a *fresh* rejection built around the guard's own sentence,
                # and rebuilding it discarded the structured results the rejection it wraps is
                # already holding -- so a guard-stopped attempt recorded `[]` and the durable
                # record could not say what its gate had run, one frame below where the answer
                # was in hand. `_assertion_guard_evidence` recovers the same file set by
                # re-reading the diagnostics, which is why nothing miscounted; what was missing
                # is the record, and a derivation is not a substitute for the measurement.
                refused.validation_results = error.validation_results
                raise refused from None
            merged = list(dict.fromkeys([*merged, *repaired_paths]))
            repairs.append(
                _SourceRepair(
                    paths=tuple(repaired_paths),
                    response_id=repaired.response_id,
                    diagnostics=error.diagnostics,
                    execution={
                        **execution_metadata(
                            agent_type="Engineer",
                            provider=repaired.provider,
                            model=repaired.model,
                            reasoning_effort=repaired.reasoning_effort,
                            model_role=repaired.model_role,
                            model_variable=repaired.model_variable,
                            routing_reason=repaired.routing_reason,
                            input_tokens=repaired.input_tokens,
                            output_tokens=repaired.output_tokens,
                        )[EXECUTION_METADATA_KEY],
                        # Explicit rather than inferred from `model_role`: a double reports
                        # no role at all, and "the scoped-fix boundary was never wired" is
                        # exactly the fact the audit could not read from the artifacts.
                        "scoped_fix_role_resolved": self._scoped_fix_executor is not None,
                        # This pass is unjournaled by construction, so the attempt's own
                        # record is the only place its re-issues can be read. Omitted rather
                        # than zeroed when the transport took no measurement.
                        **(
                            {"stream_reissues": repaired.stream_reissues}
                            if repaired.stream_reissues is not None
                            else {}
                        ),
                    },
                )
            )
            # Verified again over everything the attempt now touches, so a repair that fixed
            # the reported rule while breaking another is still caught here -- and so a
            # repair that cleared a lint rule by breaking the change's own tests is too.
            try:
                applied, typechecks, scoped_tests = await self._verified(
                    workspace_root, merged, prior_diagnostics
                )
            except SourceValidationError as still_failing:
                still_failing.modified_paths = tuple(merged)
                signature = diagnostic_signature(still_failing.diagnostics)
                if not signature or signature == last_signature:
                    # A repeat -- or a signature that captured nothing, which must be read
                    # as unknown rather than as changed -- is a defect this loop cannot act
                    # on. Stop on the spot and let the retry policy judge it.
                    _LOGGER.warning(
                        "engineer source repair stopped at pass %d: rewrote %s and the "
                        "diagnostic signature did not change "
                        "[outcome=repair_signature_repeated agent=engineer]",
                        pass_number,
                        ", ".join(repaired_paths),
                    )
                    raise
                _LOGGER.info(
                    "engineer source repair pass %d rewrote %s; the diagnostics changed, "
                    "so another pass runs [outcome=repair_diagnostics_changed "
                    "agent=engineer]",
                    pass_number,
                    ", ".join(repaired_paths),
                )
                last_signature = signature
                error = still_failing
                continue
            _LOGGER.info(
                "engineer source repair cleared the gate in %d pass(es), rewriting %s "
                "[outcome=repair_cleared agent=engineer]",
                pass_number,
                ", ".join(sorted({path for repair in repairs for path in repair.paths})),
            )
            return _GateOutcome(applied, scoped_tests, typechecks, repairs)
        _LOGGER.warning(
            "engineer source repair hit its pass ceiling of %d with the diagnostics still "
            "changing [outcome=repair_pass_ceiling agent=engineer]",
            _MAX_SOURCE_REPAIR_PASSES,
        )
        raise error

    async def _wired_with_repairs(
        self,
        workspace_root: Path,
        gate: _GateOutcome,
        paths: list[str],
        *,
        instructions: str,
        input_text: str,
        assigned_paths: Sequence[str],
        definitions: DefinitionIndex | None = None,
        prior_diagnostics: Sequence[str] = (),
    ) -> _GateOutcome:
        """Ask whether this change added code nothing reaches or wrote it in the wrong file.

        The highest-value check in this agent and the one that costs no model call to *ask*.
        A module nobody references was, until now, found only by the runtime after the
        Engineer had already returned `completed` -- so it could not be repaired where it was
        made and always cost a full outer attempt. It was the single largest cause of wasted
        attempts on this platform: 15 of 62 attempt-0s across live runs 66 to 106.

        Runs last, after the commit gate and the change's own tests are green, because a
        wiring instruction given to a model whose source the linter is still rejecting spends
        a pass on the wrong problem.

        **This check never fails the attempt, and that is a deliberate decision rather than
        caution.** Two reasons, and either alone would be sufficient:

        * The runtime's gate is the authority, and it carries the plumbing that makes an
          unrepaired wiring failure *useful*: `wiring_repairs` reaches the next attempt's
          retry strategy and force-includes each target file in its snapshot. Raising here
          would fail the attempt as a source-validation rejection instead, discarding the one
          hint that closed the third of the three gaps behind the 24% figure -- the next
          attempt would be told to edit a file it had never been shown, which is the exact
          failure that measurement was about.
        * Fast feedback inside the attempt must not become a second authority. An agent that
          both writes the code and rules on whether it is reachable has no gate at all.

        So the worst case here is precisely today's behaviour: the attempt proceeds, the
        runtime asks the same question over the same checkout, and it decides. The best case
        is a wiring miss that costs one repair pass instead of a re-implementation.

        Bounded like the loop above it: another pass runs only while the findings actually
        changed and the pass actually wrote something, with `_MAX_WIRING_REPAIR_PASSES` as a
        backstop. Nothing is journaled, nothing is granted, and no counter moves. When a pass
        does write, the commit gate and the tests run again over the merged paths through
        `_verified_with_repairs`, because a reference added to a host file can break that
        file's own lint -- and unlike a wiring finding, a lint failure genuinely must stop the
        attempt.

        The placement check rides in the same loop, on the same terms and behind the same
        precedence `_InAttemptFindings` documents: it is the same shape of defect -- source
        every command accepts, in a place review will send back -- and giving it a loop of its
        own would mean two repair budgets for one attempt.
        """
        findings = await self._in_attempt_findings(workspace_root, assigned_paths)
        if findings is None:
            return gate
        dropped = findings.dropped_candidates
        if not findings.issues:
            return _GateOutcome(
                gate.formatting_commands,
                gate.scoped_test_commands,
                gate.typecheck_commands,
                gate.repairs,
                reachability_candidates_dropped=dropped,
            )
        repair_executor = self._scoped_fix_executor or self._coding_executor
        # Everything this attempt has touched so far, the gate's own repair passes included:
        # the re-verification at the end has to be asked about all of it, or a file a lint
        # repair rewrote a moment ago would be left out of the run that accepts this attempt.
        merged = list(
            dict.fromkeys([*paths, *(item for repair in gate.repairs for item in repair.paths)])
        )
        repairs = list(gate.repairs)
        wrote_anything = False
        # The `wiring_repair_*` outcome tokens below are this loop's historical names and are
        # left alone deliberately: they are what an operator's log queries and the 47- Part D
        # measurement already key on, and renaming them to cover the second check would break
        # every one of those readers to describe the same event. The sentences say which check
        # spoke; the tokens name the loop.
        last_signature = diagnostic_signature(findings.issues)
        for pass_number in range(1, _MAX_WIRING_REPAIR_PASSES + 1):
            if self._cancellation_token is not None:
                await self._cancellation_token.raise_if_cancelled()
            diagnostics = findings.diagnostics
            # The same lookup the gate's own repair gets, for the same reason: a wiring
            # finding names the module to reference and the host to reference it from, and a
            # pass that cannot read either writes the reference against a guessed shape.
            quoted, defined = _repair_regions(
                workspace_root,
                diagnostics,
                definitions.including(merged) if definitions is not None else None,
            )
            try:
                repaired = await repair_executor.execute(
                    workspace_root=workspace_root,
                    instructions=_source_repair_instructions(
                        instructions, diagnostics, quoted, defined
                    ),
                    input_text=input_text,
                    # Deliberately unjournaled, for the reason `_verified_with_repairs`
                    # documents: one journaled coding effect per attempt.
                    operation_executor=None,
                    cancellation_token=self._cancellation_token,
                )
                repaired_paths = [
                    resolve_workspace_path(workspace_root, modified_file)
                    .relative_to(workspace_root)
                    .as_posix()
                    for modified_file in repaired.modified_files
                ]
            except CancellationRequested:
                raise
            except Exception as model_error:  # noqa: BLE001 - the repair is best-effort
                _LOGGER.warning(
                    "the in-attempt wiring repair did not run: pass %d raised %s against %d "
                    "finding(s). The attempt proceeds and the runtime's own gate decides "
                    "[outcome=wiring_repair_model_error agent=engineer]",
                    pass_number,
                    type(model_error).__name__,
                    len(diagnostics),
                )
                break
            if not repaired_paths:
                _LOGGER.warning(
                    "the in-attempt wiring repair changed nothing: pass %d returned no files "
                    "against %d finding(s), so re-inspecting would ask the same question of "
                    "the same bytes [outcome=wiring_repair_changed_nothing agent=engineer]",
                    pass_number,
                    len(diagnostics),
                )
                break
            wrote_anything = True
            merged = list(dict.fromkeys([*merged, *repaired_paths]))
            repairs.append(
                _SourceRepair(
                    paths=tuple(repaired_paths),
                    response_id=repaired.response_id,
                    diagnostics=diagnostics,
                    execution={
                        **execution_metadata(
                            agent_type="Engineer",
                            provider=repaired.provider,
                            model=repaired.model,
                            reasoning_effort=repaired.reasoning_effort,
                            model_role=repaired.model_role,
                            model_variable=repaired.model_variable,
                            routing_reason=repaired.routing_reason,
                            input_tokens=repaired.input_tokens,
                            output_tokens=repaired.output_tokens,
                        )[EXECUTION_METADATA_KEY],
                        "scoped_fix_role_resolved": self._scoped_fix_executor is not None,
                        # This pass is unjournaled by construction, so the attempt's own
                        # record is the only place its re-issues can be read. Omitted rather
                        # than zeroed when the transport took no measurement.
                        **(
                            {"stream_reissues": repaired.stream_reissues}
                            if repaired.stream_reissues is not None
                            else {}
                        ),
                    },
                )
            )
            rechecked = await self._in_attempt_findings(workspace_root, assigned_paths)
            if rechecked is None:
                findings = _InAttemptFindings(dropped_candidates=dropped)
                break
            findings = rechecked
            dropped = findings.dropped_candidates
            if not findings.issues:
                _LOGGER.info(
                    "the in-attempt repair cleared every finding in %d pass(es), rewriting "
                    "%s [outcome=wiring_repair_cleared agent=engineer]",
                    pass_number,
                    ", ".join(repaired_paths),
                )
                break
            signature = diagnostic_signature(findings.issues)
            if not signature or signature == last_signature:
                # A repeat -- or a signature that captured nothing, which must be read as
                # unknown rather than as changed -- is a defect another identical telling
                # will not fix. -052b repeated the same change until the identical-diagnostic
                # rule ended the workstream.
                _LOGGER.warning(
                    "the in-attempt wiring repair stopped at pass %d: it rewrote %s and the "
                    "findings did not change [outcome=wiring_repair_repeated agent=engineer]",
                    pass_number,
                    ", ".join(repaired_paths),
                )
                break
            last_signature = signature
        if findings.wiring_issues:
            _LOGGER.warning(
                "the in-attempt reachability check still reports %d finding(s) after %d "
                "pass(es); the attempt proceeds and the runtime's authoritative gate decides "
                "[outcome=reachability_unrepaired agent=engineer]",
                len(findings.wiring_issues),
                len(repairs) - len(gate.repairs),
            )
        if findings.placement_issues:
            # A different sentence, because there is no authoritative gate behind this one:
            # what decides a placement question after this point is the reviewer, reading the
            # change. An unrepaired finding here is a prediction of a review cycle, not of a
            # rejection -- and it is the number to watch when deciding whether this check
            # earns its place at all.
            _LOGGER.warning(
                "the in-attempt placement check still reports %d finding(s) after %d pass(es); "
                "the attempt proceeds and review decides "
                "[outcome=assigned_file_unrepaired agent=engineer]",
                len(findings.placement_issues),
                len(repairs) - len(gate.repairs),
            )
        if not wrote_anything:
            return _GateOutcome(
                gate.formatting_commands,
                gate.scoped_test_commands,
                gate.typecheck_commands,
                repairs,
                reachability_issues=findings.wiring_issues,
                reachability_candidates_dropped=dropped,
                assigned_file_issues=findings.placement_issues,
                design_value_issues=findings.design_issues,
            )
        # A reference added to a host file is a source change like any other, so the checks
        # that already accepted this attempt are asked again over everything it now touches.
        # This one *may* fail the attempt: a host file whose lint the new reference broke is
        # a mechanical failure of exactly the kind the loop above exists for.
        try:
            verified = await self._verified_with_repairs(
                workspace_root,
                merged,
                instructions=instructions,
                input_text=input_text,
                definitions=definitions,
                prior_diagnostics=prior_diagnostics,
            )
        except SourceValidationError as rejected:
            # This loop's own passes are part of the attempt's repair record too, and the
            # re-verification that rejected it knows nothing about them -- it stamps only the
            # passes it ran itself. `repairs` already carries the commit gate's earlier passes
            # as well, so the record is the whole attempt in the order the passes happened.
            rejected.source_repairs = (*repairs, *rejected.source_repairs)
            raise
        return _GateOutcome(
            verified.formatting_commands,
            verified.scoped_test_commands,
            verified.typecheck_commands,
            [*repairs, *verified.repairs],
            reachability_issues=findings.wiring_issues,
            reachability_candidates_dropped=dropped,
            assigned_file_issues=findings.placement_issues,
            design_value_issues=findings.design_issues,
        )

    async def _in_attempt_findings(
        self, workspace_root: Path, assigned_paths: Sequence[str]
    ) -> _InAttemptFindings | None:
        """Ask both deterministic inspections what is wrong with what this attempt wrote.

        ``None`` means the wiring inspection itself could not run, which is not a finding
        about the change: the caller returns the attempt exactly as it was, because this
        check exists to make a known failure cheaper and never to invent a new way to fail.

        The placement check is asked **only when nothing is unreachable**, for the reason
        `_InAttemptFindings` gives: the two findings' instructions contradict each other, and
        the unreachable one is the one an authoritative gate is waiting to stop. Its own
        failure is absorbed the same way -- a placement inspection that cannot run leaves the
        attempt with whatever the wiring inspection said, which is today's behaviour.
        """
        try:
            outcome = await self._reachability_checker.issues(
                workspace_root, assigned_paths=assigned_paths
            )
        except CancellationRequested:
            raise
        except Exception as check_error:  # noqa: BLE001 - the inspection is best-effort
            _LOGGER.warning(
                "the in-attempt reachability check could not run: %s. The attempt proceeds "
                "and the runtime's own gate decides "
                "[outcome=reachability_unavailable agent=engineer]",
                type(check_error).__name__,
            )
            return None
        design = await self._design_value_findings(workspace_root)
        if outcome.issues:
            return _InAttemptFindings(
                wiring_issues=outcome.issues,
                design_issues=design,
                dropped_candidates=outcome.dropped_candidates,
            )
        try:
            placement = await self._assigned_file_checker.issues(
                workspace_root, assigned_paths=assigned_paths
            )
        except CancellationRequested:
            raise
        except Exception as check_error:  # noqa: BLE001 - the inspection is best-effort
            _LOGGER.warning(
                "the in-attempt placement check could not run: %s. The attempt proceeds and "
                "review decides [outcome=assigned_file_unavailable agent=engineer]",
                type(check_error).__name__,
            )
            placement = ()
        return _InAttemptFindings(
            placement_issues=tuple(placement),
            design_issues=design,
            dropped_candidates=outcome.dropped_candidates,
        )

    async def _design_pictures(self, state: AgentState, task_plan: Any) -> tuple[ImageInput, ...]:
        """Fetch this workstream's assigned frames as pictures, or nothing.

        Every absence is ordinary and silent: no attachment source, a model that cannot see, a
        frame whose render was refused, a purged blob. Each leaves the coding call exactly as
        it was before pictures existed, and none of them is a defect in the change being made.
        """
        if self._attachments is None:
            return ()
        design = design_snapshot_context_from_state(state, task_plan)
        if design is None:
            return ()
        images: list[ImageInput] = []
        for node in design.get("design_nodes") or []:
            attachment_id = node.get("preview_attachment_id") if isinstance(node, dict) else None
            if not isinstance(attachment_id, str) or not attachment_id:
                continue
            content = await self._attachments.get_content(attachment_id)
            if content:
                images.append(
                    ImageInput(media_type="image/png", data=b64encode(content).decode("ascii"))
                )
        return tuple(images)

    async def _design_value_findings(self, workspace_root: Path) -> tuple[str, ...]:
        """Report the assigned design's colours and type families this change never mentions.

        The only deterministic answer this platform has to "does it match the design": build,
        lint, format and test all pass identically on an invented palette, because no command
        any repository ships checks a colour.

        Absorbed like every other inspection of this kind -- a check that cannot run reports
        nothing, because turning its own failure into a finding would replace a design
        question with a platform one.
        """
        if self._design_value_checker is None or not self._design_contents:
            return ()
        try:
            issues: tuple[str, ...] = await self._design_value_checker.issues(
                workspace_root, contents=self._design_contents
            )
            return issues
        except CancellationRequested:
            raise
        except Exception as check_error:  # noqa: BLE001 - the inspection is best-effort
            _LOGGER.warning(
                "the in-attempt design-value check could not run: %s. The attempt proceeds "
                "[outcome=design_value_unavailable agent=engineer]",
                type(check_error).__name__,
            )
            return ()

    def _self_review_files(
        self, workspace_root: Path, paths: Sequence[str]
    ) -> tuple[list[tuple[str, str, bool]], dict[str, str]]:
        """Read the candidate's files for the review, and name every one it may not see whole.

        The key-material rule applies here identically to everywhere else: one predicate for
        the path, one for the bytes, no variant and no bypass -- being the attempt's own work
        is not permission to leak (F1). A file withheld for any reason is *named with that
        reason*, and a quoted file that was cut carries its truncation flag, because "the
        review never saw the file (whole)" has to stay answerable from the artifact, exactly
        as `required_omitted_paths` answers it for the coding snapshot. The reasons are the
        seam-omission vocabulary (76-): sensitive_path | unreadable | key_material |
        evidence_budget -- do not invent a fifth spelling.
        """
        contents: list[tuple[str, str, bool]] = []
        withheld: dict[str, str] = {}
        total = 0
        for path in paths:
            if not _safe_context_path(Path(path)):
                withheld[path] = "sensitive_path"
                continue
            try:
                target = resolve_workspace_path(workspace_root, path)
                if not target.is_file():
                    continue
                if target.stat().st_size > _REPAIR_REGION_MAX_SOURCE_BYTES:
                    withheld[path] = "unreadable"
                    continue
                text = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                withheld[path] = "unreadable"
                continue
            if carries_key_material(text):
                withheld[path] = "key_material"
                continue
            text = redact_source_credentials(text)
            truncated = len(text) > _SELF_REVIEW_FILE_MAX_CHARACTERS
            if truncated:
                text = text[:_SELF_REVIEW_FILE_MAX_CHARACTERS] + _SELF_REVIEW_TRUNCATION_MARKER
            if total + len(text) > _SELF_REVIEW_FILES_MAX_TOTAL_CHARACTERS:
                withheld[path] = "evidence_budget"
                continue
            total += len(text)
            contents.append((path, text, truncated))
        return contents, withheld

    def _self_review_rejection(
        self,
        diagnostics: Sequence[str],
        gate: _GateOutcome,
        merged: Sequence[str],
        record: dict[str, Any],
    ) -> SourceValidationError:
        """Build the named escape a self-review takes out of the attempt.

        The same exception every other in-attempt rejection travels on, so the failed
        completion, the retry lineage and the workspace preservation all work unchanged --
        but carrying the outcome name, because a raw finding line must never be the only
        explanation of why an attempt ended (46- §36). The record rides along so the
        measurement survives the attempts that most need it: the ones the review stopped.
        """
        rejection = SourceValidationError(tuple(diagnostics))
        rejection.modified_paths = tuple(merged)
        rejection.terminal_outcome = SELF_REVIEW_SUBSTANTIVE_OUTCOME
        rejection.source_repairs = tuple(gate.repairs)
        rejection.self_review = dict(record)
        return rejection

    async def _self_reviewed(
        self,
        workspace_root: Path,
        gate: _GateOutcome,
        paths: Sequence[str],
        *,
        prior_changed_paths: Sequence[str] = (),
        instructions: str,
        input_text: str,
        definitions: DefinitionIndex | None,
        prior_diagnostics: Sequence[str] = (),
    ) -> tuple[_GateOutcome, dict[str, Any]]:
        """Review the finished work once, correct what is bounded, and gate `completed` on it.

        Runs last deliberately: every deterministic check is green by now, so the one model
        pass spends its attention on the questions none of them can answer -- requirement
        completeness, behavioural completeness, scope discipline, contract compliance. The
        bounds are the whole safety case, enforced here rather than requested of the model:

        * **Exactly one review pass.** There is no loop in this method; corrections are not
          re-reviewed. What confirms a localized finding resolved is deterministic -- the
          correction wrote files and the re-run validation accepted everything the attempt
          now touches -- never a second judgement, which would be the review-of-a-review
          46- §32 forbids.
        * **Localized findings are corrected once, in the same attempt,** by the primary
          coding executor -- the CODING role this attempt is already pinned to, unjournaled
          like every repair pass, so one journaled coding effect per attempt survives.
        * **A substantive finding leaves the attempt** with the named outcome, as does a
          correction that could not be had: `completed` now means the review's findings were
          resolved, and an attempt that cannot resolve them may not claim it.
        * **It gates `completion_status` and nothing further.** The commit gate and the
          independent Reviewer run afterwards regardless of what this pass said; a review
          that found nothing approves nothing.

        The one failure that does *not* leave the attempt is the boundary's own: a review
        that could not run or could not be parsed is recorded as unavailable and the attempt
        proceeds exactly as it did before Part E existed. The gap this closes was open for
        every attempt until now; an attempt that briefly reopens it is strictly no worse.

        Corrections are bounded edits to the files the change already touches, so the wiring
        inspection is not re-run here: the runtime's authoritative reachability gate still
        judges the whole attempt afterwards, and that two-layer split is the design.
        """
        if self._self_reviewer is None:
            return gate, {}
        merged = list(
            dict.fromkeys(
                [
                    *paths,
                    *(item for repair in gate.repairs for item in repair.paths),
                    # The prior lineage last, deliberately: under the total evidence budget,
                    # drops land on the oldest evidence -- and are declared either way (76-).
                    *prior_changed_paths,
                ]
            )
        )
        contents, withheld = self._self_review_files(workspace_root, merged)
        truncated_paths = [path for path, _content, truncated in contents if truncated]
        review_input = json.dumps(
            {
                # Evidence first, so the pass does not re-derive what is already known.
                "validation_evidence": {
                    "formatting_commands": list(gate.formatting_commands),
                    "typecheck_commands": list(gate.typecheck_commands),
                    "scoped_test_commands": list(gate.scoped_test_commands),
                    "source_repair_passes": len(gate.repairs),
                    "reachability_issues_unrepaired": list(gate.reachability_issues),
                    "assigned_file_issues_unrepaired": list(gate.assigned_file_issues),
                },
                "changed_paths": merged,
                "changed_files": [
                    {"path": path, "content": content} for path, content, _truncated in contents
                ],
                "withheld_paths": list(withheld),
                "withheld_reasons": dict(withheld),
            },
            indent=2,
            sort_keys=True,
        )
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()
        assessment = await self._self_reviewer.review(
            instructions=instructions, input_text=review_input
        )
        if assessment is None:
            # Composed and could not run: a fact about this attempt, recorded as such --
            # distinct from the no-record case where no reviewer was composed at all.
            return gate, {"self_review": {"ran": False, "outcome": "unavailable"}}
        record: dict[str, Any] = {
            "ran": True,
            "outcome": "clean",
            "response_id": assessment.response_id,
            "execution": dict(assessment.execution),
            "summary": assessment.summary,
            "requirement_coverage": [dict(item) for item in assessment.requirement_coverage],
            "findings": [finding.as_record() for finding in assessment.findings],
            "withheld_paths": list(withheld),
            "withheld_reasons": dict(withheld),
            "truncated_paths": truncated_paths,
            "corrections_applied": [],
        }
        # The code, not the model, is the authority on blindness (76-): a substantive
        # finding judged entirely from files the evidence declared not fully shown is an
        # accurate description of the budget, not of the work. It is recorded as a
        # limitation and composes no diagnostic -- it must never become a blocking issue,
        # a ledger demand, or a repeat-rule sentence. AB-Feature-207 lost a green
        # workstream to exactly that sentence.
        unseen = {*withheld, *truncated_paths}
        blind = tuple(finding for finding in assessment.findings if evidence_blind(finding, unseen))
        if blind:
            record["evidence_blind_findings"] = [finding.as_record() for finding in blind]
            _LOGGER.warning(
                "the implementation self-review reported %d substantive finding(s) judged "
                "only from files declared not fully shown; recorded as an evidence "
                "limitation, not a finding "
                "[outcome=self_review_evidence_blind agent=engineer]",
                len(blind),
            )
        # The symmetric half of the same rule, and 218 is why it is needed. 76- stopped a
        # BLIND FINDING from being treated as a finding; nothing stopped a BLIND PASS from
        # being treated as a pass. The self-review is the platform's cheapest chance to stop
        # a bad attempt before a review cycle is spent, and in 218 it ran twice on files it
        # had itself recorded as truncated and returned `clean` both times -- once on the
        # frontend attempt that was then approved carrying two production blockers, and once
        # on a backend attempt whose own summary said "some service internals inferred from
        # tests because the quoted bulk service file is truncated". A verdict may not be
        # stronger than its evidence, in either direction.
        inconclusive = sorted(
            path
            for path in unseen
            if path in set(merged) and classify_file_change(path) == "production"
        )
        active = tuple(finding for finding in assessment.findings if finding not in blind)
        if not active:
            if inconclusive:
                # Exactly the `clean` code path, and that is the whole safety case: the
                # attempt proceeds, no diagnostic is composed, no `SourceValidationError` is
                # raised, and nothing here reaches the ledger, the repeat rules or the
                # blocking issues. It withholds an assurance and never adds a demand. An
                # `inconclusive` self-review that could fail an attempt would be the 207
                # defect with the sign flipped.
                record["outcome"] = "inconclusive"
                record["inconclusive_paths"] = inconclusive
                _LOGGER.warning(
                    "the implementation self-review reported nothing while %d changed "
                    "production file(s) were withheld or truncated in its own evidence; "
                    "recorded as inconclusive rather than clean, and the attempt proceeds "
                    "[outcome=self_review_inconclusive agent=engineer]",
                    len(inconclusive),
                )
                return gate, {"self_review": record}
            _LOGGER.info(
                "the implementation self-review found nothing to correct "
                "[outcome=self_review_clean agent=engineer]"
            )
            return gate, {"self_review": record}
        diagnostics = tuple(self_review_diagnostic(finding) for finding in active)
        if any(finding.substantive for finding in active):
            # Not repaired in a loop, by design: rearchitecting under a repair budget is how
            # a bounded correction becomes a second retry engine. The findings travel as the
            # attempt's blocking diagnostics, so the next attempt's context assembly can
            # resolve what they name.
            record["outcome"] = "substantive_problem"
            _LOGGER.warning(
                "the implementation self-review found %d substantive problem(s); the "
                "attempt ends with a named outcome rather than a correction "
                "[outcome=self_review_substantive agent=engineer]",
                sum(1 for finding in active if finding.substantive),
            )
            raise self._self_review_rejection(diagnostics, gate, merged, record)
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()
        try:
            corrected = await self._coding_executor.execute(
                workspace_root=workspace_root,
                instructions=_self_review_correction_instructions(instructions, diagnostics),
                input_text=input_text,
                cancellation_token=self._cancellation_token,
                # Deliberately unjournaled: one journaled coding effect per attempt, for the
                # reason `_verified_with_repairs` documents.
                operation_executor=None,
            )
            corrected_paths = [
                resolve_workspace_path(workspace_root, modified_file)
                .relative_to(workspace_root)
                .as_posix()
                for modified_file in corrected.modified_files
            ]
        except CancellationRequested:
            raise
        except Exception as model_error:  # noqa: BLE001 - the correction is best-effort
            # The findings stand and nothing answered them, so `completed` is not available:
            # it now means the review's findings were resolved, and these were not. The
            # record says the correction failed rather than that the problem was judged
            # substantive; the terminal outcome is the same named escape either way.
            record["outcome"] = "corrections_failed"
            _LOGGER.warning(
                "the self-review correction did not run: %s against %d finding(s); the "
                "attempt ends with its findings unresolved "
                "[outcome=self_review_correction_failed agent=engineer]",
                type(model_error).__name__,
                len(active),
            )
            if is_transport_fault(model_error):
                # The same rule the source repair above follows, for the same reason: a
                # correction that never reached a provider has established nothing about
                # this work, and a review verdict is the one thing it must not be recorded
                # as. Raised as itself, to the fault allowance rather than to the attempt.
                raise
            raise self._self_review_rejection(diagnostics, gate, merged, record) from None
        if not corrected_paths:
            record["outcome"] = "corrections_failed"
            _LOGGER.warning(
                "the self-review correction changed nothing against %d finding(s); the "
                "attempt ends with its findings unresolved "
                "[outcome=self_review_correction_failed agent=engineer]",
                len(active),
            )
            raise self._self_review_rejection(diagnostics, gate, merged, record)
        record["corrections_applied"] = corrected_paths
        record["correction_response_id"] = corrected.response_id
        record["correction_execution"] = execution_metadata(
            agent_type="Engineer",
            provider=corrected.provider,
            model=corrected.model,
            reasoning_effort=corrected.reasoning_effort,
            model_role=corrected.model_role,
            model_variable=corrected.model_variable,
            routing_reason=corrected.routing_reason,
            input_tokens=corrected.input_tokens,
            output_tokens=corrected.output_tokens,
        )[EXECUTION_METADATA_KEY]
        merged = list(dict.fromkeys([*merged, *corrected_paths]))
        # Re-run the relevant validation over everything the attempt now touches. A
        # correction is a source change like any other, and one that broke the gate, the
        # types or the change's own tests must stop the attempt exactly as the original
        # write would have -- through the same bounded repair loop, with the same escapes.
        try:
            verified = await self._verified_with_repairs(
                workspace_root,
                merged,
                instructions=instructions,
                input_text=input_text,
                definitions=definitions,
                prior_diagnostics=prior_diagnostics,
            )
        except SourceValidationError as rejected:
            record["outcome"] = "correction_rejected"
            rejected.source_repairs = (*gate.repairs, *rejected.source_repairs)
            rejected.self_review = dict(record)
            raise
        record["outcome"] = "corrected"
        _LOGGER.info(
            "the self-review correction resolved %d finding(s), rewriting %s, and the "
            "re-run validation accepted it [outcome=self_review_corrected agent=engineer]",
            len(active),
            ", ".join(corrected_paths),
        )
        return (
            _GateOutcome(
                verified.formatting_commands,
                verified.scoped_test_commands,
                verified.typecheck_commands,
                [*gate.repairs, *verified.repairs],
                reachability_issues=gate.reachability_issues,
                reachability_candidates_dropped=gate.reachability_candidates_dropped,
                assigned_file_issues=gate.assigned_file_issues,
            ),
            {"self_review": record},
        )

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Apply workspace updates and publish ``006_code_completion.json``."""
        task_plan = require_artifact(
            state,
            TaskPlanArtifact,
            artifact_id=ARTIFACT_FILENAMES["task_plan"],
        )
        prior_review = _prior_review_for_retry(
            state, allow_without_review=bool(task_plan.metadata.get("retry_plan"))
        )
        workspace_root = resolve_workspace_path(state["workspace_descriptor"].root_path, ".")
        repository_tools = self._repository_tools_factory(workspace_root)
        existing_files = set(repository_tools.scan_directory().files)
        file_tools = self._file_tools_factory(workspace_root)
        # A retry is told which module the reviewer blocked on, so the review text selects
        # context as strongly as the plan does: the attempt that failed for guessing at a
        # module's exports is the attempt that most needs to be shown it.
        execution_context = _execution_context(task_plan)
        changed_paths = list(_prior_attempt_file_changes(state))
        # Two readings of one assignment, and they are not interchangeable. `assigned_paths`
        # is what the plan ORDERS into the prompt, and the refusal below reads it, so it
        # stays files-only. `scan_sources` is what the import and channel-seam scans READ,
        # and there an area the change has touched is the strongest thing the plan said:
        # 218's nine assigned directories left the scans with two JSON files and no imports.
        assigned_paths = _assigned_context_paths(execution_context.get("expected_files_or_areas"))
        scan_sources = _assigned_scan_sources(
            execution_context.get("expected_files_or_areas"),
            [*changed_paths, *_diff_source_paths(self._previous_attempt_diff or "")],
        )
        blocking_diagnostics = _blocking_diagnostics(prior_review, execution_context)
        # The files the diagnostics themselves name, normalized from however the gate spelled
        # them -- absolute workspace paths, `line:col` suffixes -- into paths this snapshot
        # can actually include. The line numbers ride along so a file too large to show whole
        # can still be excerpted around the exact location the gate rejected.
        diagnostic_locations = _diagnostic_file_locations(blocking_diagnostics, workspace_root)
        diagnostic_file_paths = tuple(
            path for path in diagnostic_locations if Path(path) in existing_files
        )
        # One deterministic definition-site lookup over this checkout, used at both of the
        # places that need it: here, to force the file a finding's subject is defined in into
        # this attempt's snapshot, and inside the repair loop, to quote it beside a
        # diagnostic that names a symbol and no file. It is the same function at two call
        # sites and not two mechanisms.
        definitions = _definition_index(existing_files, file_tools)
        # Resolved once and used twice: the sites' paths join the required set (the A.2
        # lookup), and their lines anchor the excerpt of a required file too large to show
        # whole. 88% of review findings carry no line number, so without this fallback the
        # anchor of last resort was the head of the file -- which put 199's ordered block
        # 367 lines below the bottom of its own evidence, on four straight attempts.
        definition_anchor_sites = definitions.sites(
            definition_symbols(blocking_diagnostics), limit=_MAX_DIAGNOSTIC_CONTEXT_PATHS
        )
        # A second, independent use of the same lookup: not a diagnostic's symbol, but one the
        # plan or the review that rejected the last attempt names in its own prose. The ranked
        # candidate pool below only ever sees term overlap against that same prose -- a file
        # relevant purely because it defines a name the plan talks about, with no shared path
        # words and no diagnostic pointing at it yet, can miss the first attempt's snapshot on
        # that basis alone. One capped scan, reused as a flat path set rather than a per-file
        # method, so ranking a hundred candidates costs one lookup and not a hundred.
        plan_relevance_sites = definitions.sites(
            definition_symbols(
                (
                    artifact_json(task_plan),
                    artifact_json(prior_review) if prior_review is not None else "",
                )
            ),
            limit=_MAX_DIAGNOSTIC_CONTEXT_PATHS,
        )
        relevance_defining_paths = frozenset(site.path for site in plan_relevance_sites)
        # Resolved here rather than inline, because this tier now answers with two things: the
        # modules it force-includes, and the ones its own cap refused. The refusals are handed
        # to the snapshot so they reach the record -- a module this change imports that no
        # attempt was shown is the single defect that cost AB-Feature-218 four attempts, and
        # it was invisible in every field the completion wrote.
        imported = _imported_repository_paths(
            changed_paths,
            existing_files,
            file_tools,
            self._previous_attempt_diff or "",
            scan_sources,
        )
        repository_context = _repository_context(
            existing_files,
            file_tools,
            _relevance_terms(
                artifact_json(task_plan),
                artifact_json(prior_review) if prior_review is not None else "",
            ),
            _required_context_paths(
                execution_context,
                imported.paths,
                # What blocked the last attempt names the module it got wrong. Resolved to
                # the file that defines it, because an attempt told only that a symbol is
                # undefined has no more information than the one before it had.
                _diagnostic_context_paths(
                    _diagnostic_symbol_paths(
                        blocking_diagnostics,
                        [*scan_sources, *changed_paths],
                        existing_files,
                        file_tools,
                        self._previous_attempt_diff or "",
                    ),
                    _definition_context_paths(definition_anchor_sites),
                ),
                prior_changed_paths=_prior_attempt_required_paths(
                    _prior_attempt_paths_most_recent_first(state), diagnostic_file_paths
                ),
                diagnostic_file_paths=diagnostic_file_paths,
                # The modules that co-import a channel package this change's own code
                # imports -- the seam the change stands on, which the imported-modules
                # argument above structurally cannot surface for a bare package import.
                channel_seam_paths=_channel_seam_paths(
                    changed_paths,
                    existing_files,
                    file_tools,
                    self._previous_attempt_diff or "",
                    scan_sources,
                ),
            ),
            required_line_numbers=_required_anchor_lines(
                diagnostic_locations, definition_anchor_sites
            ),
            candidates_discarded=imported.discarded,
            relevance_defining_paths=relevance_defining_paths,
            max_characters=self._snapshot_budget.max_characters,
            per_file_max_characters=self._snapshot_budget.per_file_max_characters,
        )
        # Before the model call, and before any budget is spent on rendering one: an attempt
        # whose own target file is absent from the snapshot must not run at all.
        _refuse_if_required_context_dropped(
            repository_context,
            assigned_paths=assigned_paths,
            diagnostic_file_paths=diagnostic_file_paths,
            wiring_target_paths=_wiring_repair_target_paths(execution_context),
            budget_source=self._snapshot_budget.source,
        )
        # Resolved from a file already in the checkout, because the paths this attempt will
        # write do not exist yet and the configuration is a property of the repository.
        lint_capabilities = await self._lint_capability_probe.describe(
            workspace_root, sorted(path.as_posix() for path in existing_files)
        )

        # The lineage the workspace still holds, when the previous attempt was preserved.
        # Named from the recorded completions rather than from a directory listing, so the
        # prompt claims presence only for files an attempt actually wrote.
        previous_attempt_files_present = (
            sorted(_workspace_present_changes(_prior_attempt_file_changes(state), workspace_root))
            if self._previous_attempt_preserved
            else []
        )
        instructions = self._prompt_loader.render(
            "engineer/v1.jinja2",
            workflow_id=state["workflow_id"],
            workspace_descriptor=state["workspace_descriptor"].model_dump(mode="json"),
            task_plan=artifact_json(task_plan),
            # The contract's own text for the sections the task plan assigns, from the same
            # selection function the independent Reviewer is given on the same two inputs.
            # Rendered into the instructions rather than added to `input_text` because the
            # self-review pass is handed exactly this string and nothing else: putting the
            # sections anywhere but here would give two of the three judges the contract and
            # leave the third inferring it from section names, which is the defect being fixed.
            contract_sections=contract_section_context_json(
                contract_section_context_from_state(state, task_plan)
            ),
            # The design this workstream's own plan assigns it, from the same selection
            # function the independent Reviewer is given on the same two inputs. Rendered into
            # the instructions rather than added to `input_text` for the contract block's
            # reason: the self-review pass is handed exactly this string and nothing else, so
            # putting the design anywhere but here would give two of the three judges the
            # design and leave the third inferring it from frame ids.
            design_snapshot=design_snapshot_context_json(
                design_snapshot_context_from_state(state, task_plan)
            ),
            design_pictures=len(await self._design_pictures(state, task_plan)),
            repository_context=json.dumps(repository_context, indent=2, sort_keys=True),
            execution_context=json.dumps(execution_context, indent=2, sort_keys=True),
            lint_capabilities=json.dumps(lint_capabilities, indent=2, sort_keys=True),
            previous_attempt_diff=self._previous_attempt_diff or "",
            previous_attempt_withheld_files=list(self._previous_attempt_withheld_files),
            previous_attempt_files_present=previous_attempt_files_present,
            primary_language=_primary_language(state),
        )
        input_text = json.dumps(
            {
                "task_plan": task_plan.model_dump(mode="json"),
                "repository_context": repository_context,
                "prior_review": (
                    prior_review.model_dump(mode="json") if prior_review is not None else None
                ),
                "execution_context": execution_context,
            },
            indent=2,
            sort_keys=True,
        )
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()
        result = await self._coding_executor.execute(
            workspace_root=workspace_root,
            instructions=instructions,
            input_text=input_text,
            cancellation_token=self._cancellation_token,
            operation_executor=self._operation_executor,
            # The frames this workstream was assigned, as pictures. The role that writes the
            # CSS could read the design's coordinates and could not see it, and produced a
            # screen of correctly-positioned empty boxes on AB-Feature-231.
            images=await self._design_pictures(state, task_plan),
        )
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()
        if not result.summary.strip():
            msg = "coding executor returned an empty completion summary"
            raise AgentArtifactError(msg)

        attempt_changed_paths = [
            resolve_workspace_path(workspace_root, modified_file)
            .relative_to(workspace_root)
            .as_posix()
            for modified_file in result.modified_files
        ]
        attempt_changed_paths = list(dict.fromkeys(attempt_changed_paths))
        # Install first: a package this change declares but never installs cannot be resolved
        # by the linter that runs next, and rewriting the source will not conjure it.
        install_commands = await self._dependency_synchronizer.sync(
            workspace_root, attempt_changed_paths
        )

        def build_completion(
            *,
            completion_status: str,
            changed_paths: Sequence[str],
            formatting_commands: Sequence[str],
            scoped_test_commands: Sequence[str],
            typecheck_commands: Sequence[str],
            source_repairs: Sequence[_SourceRepair],
            extra_metadata: dict[str, Any],
        ) -> CodeCompletionArtifact:
            """Record one attempt's complete change set, whether the gate accepted it or not."""
            # A retry runs in the same dirty workspace. Its executor commonly reports only
            # the file it repaired, but every earlier uncommitted file is still part of the
            # candidate change. Preserve the complete lineage so review and publication
            # cannot silently drop an untouched file from a prior attempt. Lineage entries
            # whose file is not in this workspace -- a reset discarded them -- are dropped,
            # because a commit stages workspace bytes and cannot stage a file that is not
            # there; the attempt's own paths are still read unconditionally, so an executor
            # that claims a write it never made still fails loudly.
            file_changes_by_path = _workspace_present_changes(
                _prior_attempt_file_changes(state), workspace_root
            )
            for relative_path in changed_paths:
                # Presence, not readability. This check exists so an executor that claims a
                # write it never made fails loudly; reading the bytes back through the text
                # size limit made it a second gate nobody designed. AB-Feature-174's console
                # died here as `platform_defect`: the dependency sync regenerated a 1.3 MB
                # package-lock.json -- a platform-appended, generated file -- and the read
                # rejected it for exceeding `max_workspace_file_bytes`. A binary asset would
                # have died the same way; the commit stages bytes, not decoded text.
                written = resolve_workspace_path(workspace_root, relative_path)
                if not written.is_file():
                    msg = (
                        "the coding executor claimed a write that is not in the workspace: "
                        f"{relative_path}"
                    )
                    raise AgentArtifactError(msg)
                file_changes_by_path[relative_path] = FileChange(
                    path=relative_path,
                    change_type="modified" if Path(relative_path) in existing_files else "added",
                    description="Updated by the configured coding executor.",
                )
            file_changes = list(file_changes_by_path.values())
            source_artifact_ids = [task_plan.artifact_id]
            if prior_review is not None:
                source_artifact_ids.append(prior_review.artifact_id)
            return create_artifact(
                CodeCompletionArtifact,
                workflow_id=state["workflow_id"],
                artifact_id=attempt_artifact_id(
                    ARTIFACT_FILENAMES["code_completion"], state["retry_count"]
                ),
                producer="engineer",
                payload={
                    "completion_status": completion_status,
                    "summary": result.summary,
                    "file_changes": [change.model_dump(mode="json") for change in file_changes],
                    "validation_results": [],
                    "test_coverage_percent": None,
                    "remaining_work": [],
                    # The reviewer must assess the actual workspace diff before any remote
                    # side effect.  The post-approval publisher creates the commit and
                    # replaces this value in a new, traceable artifact revision.
                    "commit_sha": None,
                    **_completion_evidence(file_changes, file_tools, execution_context),
                },
                metadata=_completion_metadata(
                    source_artifact_ids=source_artifact_ids,
                    result=result,
                    repository_context=repository_context,
                    formatting_commands=formatting_commands,
                    scoped_test_commands=scoped_test_commands,
                    typecheck_commands=typecheck_commands,
                    install_commands=install_commands,
                    source_repairs=source_repairs,
                    model_routing=self._model_routing,
                    attempt_changed_paths=changed_paths,
                    extra_metadata=extra_metadata,
                    snapshot_budget_source=self._snapshot_budget.source,
                ),
            )

        # Normalize generated code with the repository's own tooling before it is staged,
        # then ask its own runner about the tests this change touched, then ask the checkout
        # whether what was added can be reached at all. A repository that enforces formatting
        # in a pre-commit hook otherwise rejects the commit outright, discarding an
        # implementation that was functionally complete -- and a suite the change itself
        # broke, or a module nothing references, otherwise costs a full attempt to learn
        # about when the answer was in the files the model had just written.
        #
        # In that order, cheapest first, and each one only asked of source the previous one
        # accepted: a wiring instruction given to a model whose source the linter is still
        # rejecting spends a pass on the wrong problem.
        try:
            gate = await self._verified_with_repairs(
                workspace_root,
                attempt_changed_paths,
                instructions=instructions,
                input_text=input_text,
                definitions=definitions,
                # The suites this attempt was ASKED to fix, not only the ones it edited. A
                # production-only remediation otherwise reaches the reviewer with the suite
                # it was granted to repair still red -- 218's attempts 4 and 5, one review
                # cycle each.
                prior_diagnostics=blocking_diagnostics,
            )
            gate = await self._wired_with_repairs(
                workspace_root,
                gate,
                attempt_changed_paths,
                instructions=instructions,
                input_text=input_text,
                definitions=definitions,
                # The plan's own list, not the file-only subset the snapshot force-includes:
                # both layers of this check must be asked the identical question, and the
                # inspection does its own filtering of the areas that are not files.
                assigned_paths=_plan_assigned_paths(execution_context),
                prior_diagnostics=blocking_diagnostics,
            )
            # Last, after every deterministic check is green (46- §27): one bounded review
            # of the finished work, gating `completion_status` and nothing further.
            gate, self_review_record = await self._self_reviewed(
                workspace_root,
                gate,
                attempt_changed_paths,
                # The completion view, not this attempt's keyhole: the same cumulative
                # union `build_completion` reports as `file_changes`, so the set the
                # review judges is the candidate the artifact claims (76-). AB-Feature-207
                # was refused twice for files a prior attempt wrote and this list omitted.
                prior_changed_paths=list(
                    _workspace_present_changes(_prior_attempt_file_changes(state), workspace_root)
                ),
                instructions=instructions,
                input_text=input_text,
                definitions=definitions,
                prior_diagnostics=blocking_diagnostics,
            )
        except SourceValidationError as error:
            # The gate rejected the attempt. What the Engineer wrote is still the only
            # record of what this attempt was, so it is attached to the rejection rather
            # than discarded with it: `validation_source_failure` is the dominant failure
            # class, and until now it recorded nothing about what the model was shown or
            # produced. The completion is `failed`, so nothing downstream can publish it.
            failed_paths = list(dict.fromkeys([*attempt_changed_paths, *error.modified_paths]))
            failed_paths.extend(
                path
                for path in installed_lockfiles(workspace_root, failed_paths)
                if path not in failed_paths
            )
            try:
                error.code_completion = build_completion(
                    completion_status="failed",
                    changed_paths=failed_paths,
                    formatting_commands=(),
                    scoped_test_commands=(),
                    typecheck_commands=(),
                    # The passes that actually ran, not the empty tuple this used to hard-code.
                    # A rejected attempt is the one whose repair history matters most: it is
                    # the attempt whose loop was asked to fix something and could not.
                    source_repairs=error.source_repairs,
                    extra_metadata={
                        # False when the attempt's own review is what stopped it: nothing
                        # about the *source* was rejected -- every configured check accepted
                        # it -- and the retry loop's pre-commit allowance is for attempts the
                        # commit gate stopped, which this was not.
                        "source_validation_rejected": (
                            error.terminal_outcome != SELF_REVIEW_SUBSTANTIVE_OUTCOME
                        ),
                        # Stated on every rejection, including when nothing ran, because zero
                        # passes and no record are different facts and only one of them is
                        # about the loop. `source_repair_passes` is restated by the shared
                        # metadata builder with the identical value when passes did run; here
                        # it is what makes the never-engaged case readable at all.
                        "source_repair_engaged": bool(error.source_repairs),
                        "source_repair_passes": len(error.source_repairs),
                        # Whether the SCOPED_FIX role resolved, asked of the composition rather
                        # than inferred from a pass that may not exist. "Was the scoped-fix
                        # boundary wired at all?" is exactly the question three runs' artifacts
                        # could not answer, and a gate-rejected attempt is where it is asked.
                        "source_repair_scoped_fix_role_resolved": self._scoped_fix_executor
                        is not None,
                        # Present only when something other than a check's own verdict
                        # decided this: today, an in-attempt repair the assertion guard
                        # refused. Absent means the checks simply rejected the source.
                        **(
                            {"terminal_outcome": error.terminal_outcome}
                            if error.terminal_outcome
                            else {}
                        ),
                        # The review that stopped -- or preceded -- this rejection, when one
                        # ran. Attached to the failed completion because the attempts a
                        # review ends are exactly the ones the 47- measurement is about.
                        **(
                            {"self_review": error.self_review}
                            if error.self_review is not None
                            else {}
                        ),
                    },
                )
            except Exception as build_error:  # noqa: BLE001 - never displace the gate's verdict
                _LOGGER.warning(
                    "gate-rejected completion could not be recorded (%s); the attempt still "
                    "fails on the gate's own diagnostics [agent=engineer]",
                    type(build_error).__name__,
                )
            raise
        for source_repair in gate.repairs:
            attempt_changed_paths.extend(
                path for path in source_repair.paths if path not in attempt_changed_paths
            )
        # Added after formatting, never before: a lockfile is generated output that its
        # package manager owns, and running a code formatter over it would rewrite thousands
        # of lines the change never touched.
        attempt_changed_paths.extend(
            path
            for path in installed_lockfiles(workspace_root, attempt_changed_paths)
            if path not in attempt_changed_paths
        )
        code_completion = build_completion(
            completion_status="completed",
            changed_paths=attempt_changed_paths,
            formatting_commands=gate.formatting_commands,
            scoped_test_commands=gate.scoped_test_commands,
            typecheck_commands=gate.typecheck_commands,
            source_repairs=gate.repairs,
            extra_metadata={**_in_attempt_check_metadata(gate), **self_review_record},
        )
        return artifact_update("engineer", [code_completion])


def _in_attempt_check_metadata(gate: _GateOutcome) -> dict[str, Any]:
    """Record what the in-attempt checks said, but only when they said something.

    Absent means the ordinary case: the inspections ran, found everything reachable and in
    the file the plan named, and examined every candidate. The three keys below are the ways
    that can fail to be true, and each has to be answerable from the artifact rather than
    from a log line:

    * `reachability_issues_unrepaired` -- this attempt could not wire what it added, and is
      proceeding to the runtime's authoritative gate expecting to be stopped there. Without
      this, a reader comparing the Engineer's `completed` against the gate's rejection would
      have no way to tell an in-attempt check that failed from one that never ran.
    * `reachability_candidates_dropped` -- the inspection's bound left these unexamined, so
      the absence of a finding is not evidence about them. A silent cap of exactly this kind
      is what made a change adding nine new modules read as "wiring passed".
    * `assigned_file_issues_unrepaired` -- the placement check named a file the plan assigned
      and the repair did not use it. No gate stops this one, so it is a *prediction* that
      review will send the change back, and it is the number that decides whether the check
      keeps its place: a run of these on changes review then approved is the evidence for
      dropping it.
    * `design_value_issues_unrepaired` -- the change does not use some of the design's own
      colours or type families. Recorded for a reason the others do not share: it is the only
      measurement this platform has of whether an implementation matches the design it was
      given, and without it here the finding reaches a repair prompt and then vanishes. On
      AB-Feature-231 the change used 5 of the design's 14 colours and referenced Inter
      nowhere, and no artifact, judge or reader could say so.
    """
    return {
        **(
            {"reachability_issues_unrepaired": list(gate.reachability_issues)}
            if gate.reachability_issues
            else {}
        ),
        **(
            {"reachability_candidates_dropped": list(gate.reachability_candidates_dropped)}
            if gate.reachability_candidates_dropped
            else {}
        ),
        **(
            {"assigned_file_issues_unrepaired": list(gate.assigned_file_issues)}
            if gate.assigned_file_issues
            else {}
        ),
        **(
            {"design_value_issues_unrepaired": list(gate.design_value_issues)}
            if gate.design_value_issues
            else {}
        ),
        **_in_attempt_stream_reissues(gate),
    }


def _in_attempt_stream_reissues(gate: _GateOutcome) -> dict[str, Any]:
    """Total the re-issues this attempt's unjournaled repair passes spent, when measured.

    The one number about the in-attempt layer that no journal row can carry. Its passes are
    unjournaled by construction, so a repair whose stream went silent and then answered on a
    second issue leaves nothing behind at all -- and that is precisely the reading that says
    whether the first-event budget is sized right for the model this attempt ran on.

    Absent when no pass reported a measurement: a deployment on a non-streaming transport, a
    test double, or an attempt with no repair passes. Present and `0` where passes ran and
    every stream spoke first time, because that is the reassuring answer and it has to be
    distinguishable from nobody having asked.
    """
    measured = [
        repair.execution["stream_reissues"]
        for repair in gate.repairs
        if isinstance(repair.execution.get("stream_reissues"), int)
    ]
    return {"stream_reissues": sum(measured)} if measured else {}


def _workspace_present_changes(
    changes: dict[str, FileChange], workspace_root: Path
) -> dict[str, FileChange]:
    """Keep only the lineage entries whose file this workspace actually holds.

    Presence, not readability: a prior attempt may legitimately have written a binary
    asset, and the commit that stages these paths stages bytes, not decoded text.
    """
    present: dict[str, FileChange] = {}
    dropped: list[str] = []
    for path, change in changes.items():
        if change.change_type != "deleted":
            try:
                if not resolve_workspace_path(workspace_root, path).is_file():
                    dropped.append(path)
                    continue
            except (ValueError, OSError):
                dropped.append(path)
                continue
        present[path] = change
    if dropped:
        _LOGGER.info(
            "prior-attempt lineage dropped %d file(s) absent from this workspace (%s) "
            "[agent=engineer]",
            len(dropped),
            ", ".join(dropped),
        )
    return present


def _completion_metadata(
    *,
    source_artifact_ids: list[str],
    result: Any,
    repository_context: dict[str, Any],
    formatting_commands: Sequence[str],
    scoped_test_commands: Sequence[str],
    typecheck_commands: Sequence[str],
    install_commands: Sequence[str],
    source_repairs: Sequence[_SourceRepair],
    model_routing: ModelRoutingDecision | None,
    attempt_changed_paths: Sequence[str],
    extra_metadata: dict[str, Any],
    snapshot_budget_source: str = "default",
) -> dict[str, Any]:
    """Build the completion metadata shared by the accepted and gate-rejected records."""
    return {
        "source_artifact_ids": source_artifact_ids,
        "coding_model": result.model,
        "coding_response_id": result.response_id,
        "prompt_template": "engineer/v1.jinja2",
        # What this attempt itself wrote, as distinct from the lineage union in
        # `file_changes`. A remediation review's blocking authority is bounded to defects
        # in what changed since the previously reviewed revision, and this is that delta.
        "attempt_changed_paths": list(attempt_changed_paths),
        **extra_metadata,
        # What this execution was: an initial implementation, or a classified
        # remediation. Persisted here because "which model wrote this, and why that
        # one" is a question asked of a completed attempt long after the child state
        # has moved on to the next one.
        "execution_mode": (
            model_routing.execution_mode.value if model_routing is not None else None
        ),
        "model_routing": (
            model_routing.model_dump(mode="json") if model_routing is not None else None
        ),
        # Recorded so a reviewer can tell repository-applied formatting apart
        # from the model's own edits when reading the diff.
        "applied_formatting_commands": list(formatting_commands),
        # A reviewer reading the diff must be able to tell a lockfile that moved
        # because of an install apart from one the model edited by hand.
        "applied_dependency_commands": list(install_commands),
        # The narrowed test commands this attempt ran on itself, before it declared itself
        # done. Empty means none ran, which is a fact about the checkout or the change --
        # no test file was touched, no test command is configured, or the configured one
        # cannot be narrowed -- and never an assertion that the tests passed.
        "scoped_test_commands": list(scoped_test_commands),
        # The repository's own typecheck commands this attempt ran on itself, before it
        # declared itself done. Empty means the checkout configures none -- a fact about the
        # repository -- and never an assertion that the types were checked and agreed.
        "typecheck_commands": list(typecheck_commands),
        # Exactly which files this attempt could read, and how many the budget left
        # out. Recorded because "was the file it was told to edit actually in its
        # snapshot?" is the first question worth asking of a failed coding attempt,
        # and until now it was unanswerable: the snapshot was built, sent, and never
        # written down. A whole class of failure read as the model ignoring an
        # instruction when the instruction named a file it had never seen.
        "context_file_paths": [item["path"] for item in repository_context["files"]],
        "context_omitted_file_count": repository_context["omitted_file_count"],
        # The budget this attempt's snapshot was packed under and where that number came
        # from -- the run-to-build-attribution instrument for budgets. A forensic pass must
        # be able to tell "this attempt ran under the declared 400k budget" from "the
        # declaration never crossed the compose boundary" without re-deriving anything.
        "context_budget_characters": repository_context.get(
            "budget_characters", _REPOSITORY_CONTEXT_MAX_CHARACTERS
        ),
        "context_budget_per_file_characters": repository_context.get(
            "budget_per_file_characters", _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS
        ),
        "context_budget_source": snapshot_budget_source,
        # Every *required* path the snapshot could not show whole, split by what actually
        # happened to it -- the 61- analysis showed the old union field was read as "not
        # shown" and meant either. `excerpted` means the file IS in the prompt as a bounded
        # window; `dropped` means its bytes never reached the prompt at all, with the reason
        # per path beside it. Recorded even when empty, so "was the file it was told to edit
        # in its snapshot" is a lookup and never again a re-derivation of budget arithmetic.
        "context_required_excerpted": list(repository_context.get("required_excerpted_paths", [])),
        "context_required_dropped": list(repository_context.get("required_dropped_paths", [])),
        "context_required_dropped_reasons": dict(
            repository_context.get("required_dropped_reasons", {})
        ),
        # Files this change's own code imports that a bounded selector refused before they
        # could be ordered -- the discard that used to be silent. AB-Feature-218's four
        # duplicate attempts turned on one such path, and no field of the completion said
        # so: it was in neither `context_file_paths` nor `context_required_dropped`, and
        # "was it ever offered?" could only be answered by re-running the selector.
        "context_candidates_discarded": list(
            repository_context.get("candidate_discarded_paths", [])
        ),
        "context_candidates_discarded_reasons": dict(
            repository_context.get("candidate_discarded_reasons", {})
        ),
        # The union of the two lists above, kept for one release so existing readers -- the
        # 52-/58- reports among them -- do not silently change meaning.
        "context_required_omitted": list(repository_context.get("required_omitted_paths", [])),
        # Whether this attempt needed a second pass to clear the repository's own
        # pre-commit verification, and what it was told. Recorded because an attempt
        # that took two calls is a different thing from one that took one, and a
        # reader comparing attempts should not have to infer it.
        **(
            {
                # The last pass's response is the one whose bytes the gate accepted; the
                # paths are the union across passes in first-seen order; the diagnostics are
                # what the first pass was asked to clear. `source_repair_passes` is what
                # says how many passes the loop took.
                "source_repair_response_id": source_repairs[-1].response_id,
                "source_repair_paths": list(
                    dict.fromkeys(path for repair in source_repairs for path in repair.paths)
                ),
                "source_repair_diagnostics": list(source_repairs[0].diagnostics),
                "source_repair_passes": len(source_repairs),
                # Which model actually repaired, and whether the SCOPED_FIX role resolved
                # or the primary coding executor stood in. The last pass's execution, like
                # the response id beside it: its bytes are the ones the gate accepted.
                "source_repair_execution": dict(source_repairs[-1].execution),
            }
            if source_repairs
            else {}
        ),
        **execution_metadata(
            agent_type="Engineer",
            provider=result.provider,
            # The model the provider echoed back for this call, not the one the role
            # nominally resolves to. They differ when a provider serves a dated
            # snapshot, and the record should name what answered.
            model=result.model,
            reasoning_effort=result.reasoning_effort,
            model_role=(
                model_routing.role.value if model_routing is not None else result.model_role
            ),
            model_variable=result.model_variable,
            routing_reason=(
                model_routing.routing_reason if model_routing is not None else result.routing_reason
            ),
            input_tokens=getattr(result, "input_tokens", None),
            output_tokens=getattr(result, "output_tokens", None),
        ),
    }


async def engineer_node(
    state: AgentState,
    *,
    prompt_loader: PromptLoader,
    coding_executor: CodingExecutor,
    file_tools_factory: FileToolsFactory = WorkspaceFileTools,
    repository_tools_factory: RepositoryToolsFactory = WorkspaceRepositoryTools,
    git_service_factory: GitServiceFactory | None = None,
    cancellation_token: CancellationToken | None = None,
    operation_executor: ExternalOperationExecutor | None = None,
) -> dict[str, Any]:
    """Run an engineering node with explicitly injected executor and workspace tools."""
    return await EngineerAgent(
        prompt_loader=prompt_loader,
        coding_executor=coding_executor,
        file_tools_factory=file_tools_factory,
        repository_tools_factory=repository_tools_factory,
        git_service_factory=git_service_factory,
        cancellation_token=cancellation_token,
        operation_executor=operation_executor,
    ).run(state)


def _prior_review_for_retry(
    state: AgentState, *, allow_without_review: bool = False
) -> ReviewArtifact | None:
    """Return the prior review artifact and require it whenever an engineer retry is requested."""
    reviews = [
        artifact
        for artifact in state["artifacts"]
        if isinstance(artifact, ReviewArtifact)
        and artifact_id_matches_lineage(artifact.artifact_id, ARTIFACT_FILENAMES["review"])
    ]
    if reviews:
        review = reviews[-1]
        if review.workflow_id != state["workflow_id"]:
            msg = "prior review artifact belongs to a different workflow"
            raise AgentArtifactError(msg)
        return review
    if state["retry_count"] > 0 and not allow_without_review:
        msg = "engineer retry requires prior 007_review.json"
        raise AgentArtifactError(msg)
    return None


def _primary_language(state: AgentState) -> str | None:
    """Return this workstream's repository's detected primary language, if reconnaissance saw it.

    Reconnaissance runs once per repository at the feature level; the child workflow that runs
    this agent is re-homed onto exactly one repository, named by its own workspace id. Absence
    -- reconnaissance never ran, ran blind, or could not confidently detect one -- returns
    `None`, so the caller renders the base prompt exactly as it does today.
    """
    repository_id = state["workspace_descriptor"].workspace_id
    for artifact in state["artifacts"]:
        if (
            isinstance(artifact, RepositoryReconnaissanceArtifact)
            and artifact.repository_id == repository_id
        ):
            profile = artifact.metadata.get("technology_profile")
            if isinstance(profile, dict):
                language = profile.get("primary_language")
                if isinstance(language, str) and language:
                    return language
    return None


def _prior_attempt_file_changes(state: AgentState) -> dict[str, FileChange]:
    """Return the cumulative uncommitted file set in stable first-seen order."""
    changes: dict[str, FileChange] = {}
    for artifact in state["artifacts"]:
        if not isinstance(artifact, CodeCompletionArtifact):
            continue
        if not artifact_id_matches_lineage(
            artifact.artifact_id, ARTIFACT_FILENAMES["code_completion"]
        ):
            continue
        if artifact.metadata.get("published_after_approval") is True:
            continue
        for change in artifact.file_changes:
            changes[change.path] = change
    return changes


# How many modules the previous attempt imported may be force-included. Bounded like the
# registry list, and for the same reason: everything forced in is budget the ranked snapshot
# does not get.
_MAX_IMPORTED_CONTEXT_PATHS = 6
# How many cap-refused import candidates the record may name. The ledger exists so a discard
# is answerable rather than invisible; it is not an inventory of the checkout's import graph,
# and a change touching a hundred files must not be able to write one.
_MAX_RECORDED_IMPORT_DISCARDS = 24
# Why a candidate the imported-modules tier resolved never became a required path. One reason
# today, named rather than inlined so the vocabulary is greppable alongside
# `required_dropped_reasons`, whose shape this deliberately matches.
_DISCARD_IMPORTED_MODULE_CAP = "imported_module_cap"


@dataclass(frozen=True, slots=True)
class _BoundedSelection:
    """What a bounded selector kept, beside what it refused and why.

    The refusals are the point. A selector that returns only its winners makes "the file was
    never offered" and "the file was offered and refused" indistinguishable downstream, and
    AB-Feature-218 spent four attempts inside that ambiguity: the module it was told to reuse
    was resolved, refused by a cap, and named in no field of the record.
    """

    paths: tuple[str, ...] = ()
    discarded: Mapping[str, str] = field(default_factory=dict)


# Extensions tried when a specifier omits one, in the order a resolver would try them.
# How many modules named by a blocking diagnostic may be force-included, across every finding
# a review produced and both of the ways one is resolved -- through an import that already
# binds the name, and through the definition site the checkout holds.
#
# Raised from four. Four was chosen when there was one resolver and it could only follow an
# existing import, so it was rarely reached; a review with findings across several modules
# resolved at most four of them and the rest fell to lexical ranking, which is the failure
# `_definition_context_paths` exists to close. Eight is still under the assignment's own claim
# on the budget and well under the sixty-file ceiling, and every path here is one a check or a
# reviewer named -- the strongest evidence this selector has about what the next attempt must
# read. One bound for the whole category rather than one per resolver, so adding the second
# way to resolve a symbol did not quietly double what this category may take.
_MAX_DIAGNOSTIC_CONTEXT_PATHS = 8
# The owning name in a dotted reference a diagnostic makes -- the `appValidation` of
# `appValidation.addAppSchema.body`. Deliberately not restricted to any language's syntax:
# whatever it matches is only a candidate until an import in this checkout binds that exact
# name, which is the test that decides.
_DIAGNOSTIC_SYMBOL = re.compile(r"\b(?P<root>[A-Za-z_$][\w$]*)\.[A-Za-z_$][\w$]*")
# A bare name, used to read which names an import statement binds.
_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")
# Severities that block an attempt. A diagnostic that did not stop the change is not
# evidence about what the next attempt has to be shown.
_BLOCKING_SEVERITIES = frozenset({"critical", "high"})


def _required_context_paths(
    execution_context: dict[str, Any],
    imported_paths: tuple[str, ...] = (),
    diagnostic_paths: tuple[str, ...] = (),
    *,
    prior_changed_paths: tuple[str, ...] = (),
    diagnostic_file_paths: tuple[str, ...] = (),
    channel_seam_paths: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the files this attempt must be shown whatever the task's wording ranks highest.

    Two citizenships, packed ordered-first: a file the attempt is ORDERED to work on -- a
    wiring repair's `target_path`, the plan's assignment, a file the blocking diagnostics
    name -- claims the budget before every ADVISORY file, because dropping an ordered file
    refuses the whole attempt while dropping an advisory one is recorded and survived.
    AB-Feature-212 is why the order is load-bearing: its plan-assigned files packed after
    twelve prior-attempt files, the advisory set spent the budget first, and the 61- refusal
    then fired on files the advisory set had evicted -- both workstreams died at r1 over
    packing order, not over capacity.

    ``prior_changed_paths`` -- the files the previous attempts of this workstream changed --
    lead the advisory set, because a remediation's target is usually its own prior work: no
    ranked snapshot may outrank the file the reviewer or the gate just rejected. 176's
    backend spent three full attempts and three repair passes rewriting an 11.5 KB service
    file from memory because its remediation context held 46 files -- 2022 log files among
    them -- and none of the ten this feature had written, the one carrying the parse error
    included. (The file a diagnostic actually names is not left to this tier: it packs in
    the ordered set as ``diagnostic_file_paths``.)

    Then the modules the previous attempt's own code imports. Nothing ranks a file higher
    than the change depending on it, and being shown one is the difference between using an
    export and guessing at it: AB-Feature-128's backend wrote
    `appValidation.addAppSchema.body.validate(row)` and spent five of its eleven attempts on
    `TypeError: addAppSchema.body is undefined`, never once seeing the file that defines it.
    The ranked snapshot is relevance-scored from the task's wording, which is why a module
    the task never names can lose the budget to files nobody needs -- the failure
    `_context_candidates` already documents as "told a module existed without being shown how
    it exports, and it guessed a shape instead".

    A wiring repair's ``example_path`` follows the imports: it is a file to imitate, not an
    order, so it must never evict one -- the split citizenship is the AB-Feature-212 fix
    applied to the one contributor that used to carry both claims under one name.

    The repository's registry files come last, so an attempt can wire a new module in without
    having to guess where this repository mounts things. **They were ahead of the imports
    until now, and that ordering was the opposite of what the incident above argues**: being
    last makes a category the first thing squeezed out when the required set overflows, and
    the category being squeezed out was the one whose absence cost five attempts. The two
    claims are not comparable in strength -- a registry is where this repository mounts
    modules *in general*, an import is a shape this change *actually depends on* -- and the
    wiring case that wants a registry now has three answers ahead of it that the import case
    does not: `wiring_repairs` force-includes the exact host file first, the in-attempt
    reachability check names it before the attempt ends, and the runtime's gate names it
    again afterwards. Nothing analogous rescues a module whose shape was guessed.

    ``diagnostic_paths`` -- the modules whose *symbols* the diagnostics mention -- open the
    advisory set's tail behind ``prior_changed_paths``: a symbol appearing in a message is a
    weaker claim than "this is the file the gate rejected", which is ordered.

    ``channel_seam_paths`` -- the modules that co-import a channel package this change's own
    code imports -- sit after the imports and ahead of the registries. The imported-modules
    tier cannot carry them: importing a bare *package* surfaces no repository file, so the
    module that configures that package is absent from the import graph by construction
    (AB-Feature-206's blocker, on attempt 0, where prior-attempt imports do not exist
    either). The tier is derived from the plan's assigned files as much as from prior
    attempts, so it must never depend on attempt history.
    """
    wiring_targets = _wiring_repair_target_paths(execution_context)
    wiring_examples = _wiring_repair_example_paths(execution_context)
    assigned = _assigned_context_paths(execution_context.get("expected_files_or_areas"))
    registry = _registry_context_paths(execution_context.get("registry_paths"))
    ordered: list[str] = list(wiring_targets)
    # Ordered files first -- their drop refuses the attempt -- then the advisory tiers in
    # strength order. First-seen dedup throughout, so a path with two citizenships keeps
    # the stronger (earlier) one.
    for path in (
        *assigned,
        *diagnostic_file_paths,
        *prior_changed_paths,
        *diagnostic_paths,
        *imported_paths,
        *wiring_examples,
        *channel_seam_paths,
        *registry,
    ):
        if path not in ordered:
            ordered.append(path)
    return tuple(ordered)


def _diagnostic_context_paths(*resolved: Sequence[str]) -> tuple[str, ...]:
    """Merge every way a diagnostic's symbols resolved, under one bound for the category.

    Import-resolved paths are offered first because they are the narrower claim -- this
    change's own code binds that exact name -- and the definition sites follow, which is the
    only answer available when nothing imports the subject yet. First-seen order, so a file
    both resolvers found is force-included once.
    """
    ordered: list[str] = []
    for paths in resolved:
        for path in paths:
            if path not in ordered:
                ordered.append(path)
    return tuple(ordered[:_MAX_DIAGNOSTIC_CONTEXT_PATHS])


def _definition_index(existing_files: set[Path], file_tools: FileTool) -> DefinitionIndex:
    """Bind the definition-site lookup to this checkout and its workspace-bound reader.

    Reading through `file_tools` rather than through the filesystem is what keeps the size
    bound, the traversal refusal and the encoding contract identical to every other read this
    agent makes of the checkout. Anything unreadable answers with nothing, which is the same
    best-effort contract the rest of the repair path has.
    """

    def read(path: str) -> str | None:
        try:
            return file_tools.read_file(path)
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            return None

    return DefinitionIndex(
        paths=tuple(sorted(item.as_posix() for item in existing_files)), read=read
    )


def _definition_context_paths(definition_sites: Sequence[DefinitionSite]) -> tuple[str, ...]:
    """Return the files defining the symbols a blocking finding named, for the next attempt.

    The A.2 half of one lookup, and the half `_diagnostic_symbol_paths` structurally cannot
    do: that one resolves *through an import that already exists*, and a finding whose subject
    nothing imports yet is frequently the point of the finding -- "this duplicates the
    existing retry helper", "the empty-selection case is unhandled". Neither names a path,
    neither binds a symbol, and lexical ranking scores the file at zero against the wording.

    The sites arrive resolved rather than being resolved here, because the caller needs the
    same lookup twice: these paths for the required set, and the definition *lines* as
    excerpt anchors (`_required_anchor_lines`). One resolution, two readings.

    A path resolved here is offered to the required set like any other, which means it is
    still screened by `carries_key_material` and still reported through
    ``required_dropped_paths`` when it is withheld. That combination is deliberate: a file
    holding key material is correctly withheld and the finding is simply not fixable by that
    attempt, and the artifact has to be able to say so rather than leaving "it never saw the
    file" to be re-derived.
    """
    return tuple(site.path for site in definition_sites)


def _required_anchor_lines(
    diagnostic_locations: dict[str, int | None],
    definition_sites: Sequence[DefinitionSite],
) -> dict[str, int]:
    """Return the line to center each required file's excerpt on, best evidence winning.

    A diagnostic that names a line is the strongest claim there is -- the gate pointed at a
    character -- so it wins outright. The definition site is the fallback for the common
    case: 88% of review findings name a symbol and no line, and with no anchor at all the
    excerpt degrades to the head of the file, which is how 199's Engineer was shown lines
    1-61 of a 643-line file whose ordered block started at line 428. The line this checkout
    defines the finding's subject on is where that subject lives; centering there is not a
    guess. First site per path wins, because ``definition_sites`` arrives ranked best-first.
    """
    anchors: dict[str, int] = {}
    for site in definition_sites:
        anchors.setdefault(site.path, site.line)
    for path, line in diagnostic_locations.items():
        if line is not None:
            anchors[path] = line
    return anchors


def _prior_attempt_paths_most_recent_first(state: AgentState) -> list[str]:
    """Return the paths this workstream's prior attempts wrote, most recently changed first.

    Read from each completion's own ``attempt_changed_paths`` -- what that attempt itself
    wrote -- because ``file_changes`` carries the cumulative lineage union and therefore
    says nothing about recency. Completions that predate the field fall back to the union,
    which degrades to first-seen order rather than losing the paths entirely.
    """
    ordered: list[str] = []
    for artifact in state["artifacts"]:
        if not isinstance(artifact, CodeCompletionArtifact):
            continue
        if not artifact_id_matches_lineage(
            artifact.artifact_id, ARTIFACT_FILENAMES["code_completion"]
        ):
            continue
        if artifact.metadata.get("published_after_approval") is True:
            continue
        attempt_paths = artifact.metadata.get("attempt_changed_paths")
        paths = (
            [item for item in attempt_paths if isinstance(item, str)]
            if isinstance(attempt_paths, list) and attempt_paths
            else [
                change.path for change in artifact.file_changes if change.change_type != "deleted"
            ]
        )
        for path in paths:
            if path in ordered:
                ordered.remove(path)
            ordered.append(path)
    ordered.reverse()
    return ordered


def _prior_attempt_required_paths(
    recent_first: Sequence[str], diagnostic_file_paths: Sequence[str]
) -> tuple[str, ...]:
    """Bound the prior-attempt list, keeping the files the current diagnostics name first.

    The cap exists so a sprawling lineage cannot evict the configs and registries ordered
    after it; inside the cap, a file the blocking diagnostics name outranks recency because
    it is the one file this remediation was granted to fix.
    """
    named = [path for path in diagnostic_file_paths if path in set(recent_first)]
    rest = [path for path in recent_first if path not in named]
    return tuple([*named, *rest][:_MAX_PRIOR_ATTEMPT_CONTEXT_PATHS])


def _assigned_context_paths(expected: Any) -> tuple[str, ...]:
    """Return the files the plan assigned, dropping the areas that are not files.

    ``expected_files_or_areas`` mixes both -- `src/pages/Profile.js` beside `src/apiUtils`
    and `src` -- and a directory force-included is either meaningless or the whole checkout.
    Anything without a suffix is treated as an area and left to the ranked snapshot, and the
    non-existent are dropped downstream by the caller's checkout membership test.
    """
    if not isinstance(expected, list):
        return ()
    files = [
        item.strip().lstrip("./")
        for item in expected
        if isinstance(item, str) and PurePosixPath(item.strip()).suffix
    ]
    return tuple(dict.fromkeys(item for item in files if item))[:_MAX_ASSIGNED_CONTEXT_PATHS]


def _assigned_scan_sources(expected: Any, changed_paths: Sequence[str]) -> tuple[str, ...]:
    """Return the plan's files, plus the change's own files under the areas it named.

    The scanning half of the plan's assignment, and deliberately not the same answer as
    `_assigned_context_paths`. That one decides what to FORCE INTO the prompt, where a
    directory really is either meaningless or the whole checkout, so it drops every entry
    without a suffix. This one decides which files the import and channel-seam scans READ to
    learn what the change depends on -- and there, discarding the areas throws away the only
    thing the plan said about most workstreams.

    AB-Feature-218 is the cost of conflating them. Its backend plan named nine directories
    and two JSON files, so the assigned set was exactly `package.json` and
    `package-lock.json`: two files with no module specifiers between them. The tier
    `_imported_repository_paths` documents as "in the live executor, the only one that says
    anything at all" was handed nothing, `_change_sources` fell through to the lineage's
    first-seen order, `app.service.js` spent the whole category cap on its own six imports,
    and the scan returned before reaching `app.route.js` -- whose third resolved import is
    the `server/validation/app.validation.js` that four attempts were told to reuse and none
    was shown. Reading the change's own route file first is what puts it in front of the
    engineer, and that is decided here.

    An area contributes only files THIS CHANGE ACTUALLY TOUCHES, never a directory walk:
    "every file under `server/utils`" is the whole checkout wearing a plan's clothes. The
    directory itself is never returned, nothing is force-included for being under one, and
    the result feeds the scans alone -- `_required_context_paths` still asks
    `_assigned_context_paths`, so nothing here can reach the ordered set or the refusal that
    reads it. The plan's own order is preserved across both kinds of entry, which is what
    makes a plan naming only files answer byte-identically to before.
    """
    if not isinstance(expected, list):
        return ()
    ordered: list[str] = []
    for raw in expected:
        if not isinstance(raw, str):
            continue
        entry = raw.strip().lstrip("./")
        if not entry:
            continue
        if PurePosixPath(entry).suffix:
            ordered.append(entry)
            continue
        area = entry.rstrip("/")
        ordered.extend(path for path in changed_paths if PurePosixPath(path).is_relative_to(area))
    return tuple(dict.fromkeys(ordered))[:_MAX_ASSIGNED_CONTEXT_PATHS]


def _plan_assigned_paths(execution_context: dict[str, Any]) -> tuple[str, ...]:
    """Return this workstream's ``expected_files_or_areas`` exactly as the plan wrote it.

    Distinct from `_assigned_context_paths`, which drops directories and takes the first few
    because it is choosing what to put in a prompt. The reachability inspection is asked the
    same question in the runtime, from the workstream record, and the two layers agreeing
    depends on being given the same list: a cap applied on one side only is a way for the
    Engineer's check and the authoritative gate to name different host files for one module.
    """
    expected = execution_context.get("expected_files_or_areas")
    if not isinstance(expected, list):
        return ()
    return tuple(item for item in expected if isinstance(item, str))


def _imported_repository_paths(
    changed_paths: Sequence[str],
    existing_files: set[Path],
    file_tools: FileTool,
    previous_attempt_diff: str = "",
    assigned_paths: Sequence[str] = (),
) -> _BoundedSelection:
    """Return the checkout files that this change's own code imports.

    Three sources, and the order between them is the point.

    **The files the plan assigned, read from the checkout.** These exist before the first
    attempt writes anything, which is what makes this the only source that says anything on
    attempt 1 -- and, in the live executor, the only one that says anything at all. It is
    also the only source that can see an import the change *uses* without having *added*:
    AB-Feature-170's route file already imported the module defining the schema, so three
    attempts wrote against a shape they had never been shown while the definition sat one
    resolved specifier away.

    **The previous attempt's diff.** Added lines only, so it speaks for the code that
    attempt wrote. A rejected attempt is reset back to the committed checkout before the
    next one starts, so what it wrote in a *new* file survives nowhere else.

    **The uncommitted change set.** Kept because a caller that has one is describing the
    same workspace the attempt is about to edit.

    Only modules resolving to a file already in this checkout. A package specifier resolves
    to nothing here and is dropped, which is correct: a dependency's source is not the
    repository's to show, and its shape is not what these attempts get wrong.

    **The cap is spent in `_change_sources` order, and what it discards is now recorded.**
    The bound is global across every source, so the first source read can spend all of it
    and every later source contributes nothing -- which is exactly what happened to
    AB-Feature-218. Its `_change_sources` order led with the plan's two assigned files
    (`package.json`, `package-lock.json`, no module specifiers between them, because
    `_assigned_context_paths` had dropped the plan's nine directories), then the lineage's
    first-seen order: `app.service.js` resolved six imports on its own, the cap was reached,
    and this function returned before reading `app.route.js` -- whose third resolved import
    is `server/validation/app.validation.js`, the file four attempts were told to reuse and
    never shown. Ordering is what fixes that, and it is fixed where the order is decided:
    `_assigned_context_paths` now lets the plan's areas contribute the change's own files,
    so the file the change is about is read before the lineage's incidental neighbours.

    This function's remaining duty is that the discard not be silent. A candidate the cap
    refuses is returned in ``discarded`` with a reason, because a required path may be
    dropped and may not vanish: before this, a path lost here reached no diagnostic field at
    all -- not `required_dropped`, not `required_omitted` -- and "was it ever offered?" was
    unanswerable from the record. That is the hole 61-/62- were meant to close, one tier
    upstream of where they close it.
    """
    resolved: list[str] = []
    discarded: dict[str, str] = {}
    sources = _change_sources(changed_paths, file_tools, previous_attempt_diff, assigned_paths)
    for path, content in sources:
        directory = PurePosixPath(path).parent
        for specifier in module_specifiers(content):
            candidate = resolve_specifier(specifier, directory, existing_files)
            if (
                candidate is None
                or candidate in resolved
                or candidate in discarded
                or candidate in changed_paths
            ):
                continue
            if len(resolved) >= _MAX_IMPORTED_CONTEXT_PATHS:
                # Past the cap: keep resolving so the record can name what was refused,
                # bounded in turn so a sprawling change cannot write an unbounded ledger.
                if len(discarded) < _MAX_RECORDED_IMPORT_DISCARDS:
                    discarded[candidate] = _DISCARD_IMPORTED_MODULE_CAP
                continue
            resolved.append(candidate)
    return _BoundedSelection(tuple(resolved), discarded)


def _change_sources(
    changed_paths: Sequence[str],
    file_tools: FileTool,
    previous_attempt_diff: str = "",
    assigned_paths: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Return the change's own sources as ``(path, content)``, plan-assigned files first.

    One definition of "this change's own code", shared by the imported-modules tier and the
    channel-seam tier so the two cannot disagree about what the change is. The stable sort
    puts the plan's files first: the budget each consumer applies is global and filled in
    arrival order, so one broad source could otherwise spend all of it before the file the
    change is actually about was read.
    """
    sources: list[tuple[str, str]] = []
    for assigned in assigned_paths:
        try:
            sources.append((assigned, file_tools.read_file(assigned)))
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            continue
    sources.extend(_diff_added_source(previous_attempt_diff))
    for changed in changed_paths:
        try:
            sources.append((changed, file_tools.read_file(changed)))
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            continue
    return _assigned_first(sources, assigned_paths)


def _channel_seam_paths(
    changed_paths: Sequence[str],
    existing_files: set[Path],
    file_tools: FileTool,
    previous_attempt_diff: str = "",
    assigned_paths: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return the modules that co-import a channel package this change's own code imports.

    The hole this closes is the one `_imported_repository_paths` structurally cannot: a bare
    *package* import (an HTTP client, a database driver) resolves to no checkout file, so the
    module that CONFIGURES that package -- base URL, credentials, interceptors -- is absent
    from the change's import graph by construction. AB-Feature-206 imported a channel package
    bare on attempt 0, every command gate passed, and the change shipped dead on arrival.

    Derived from the same sources as the imported-modules tier -- the plan's assigned files
    exist before the first attempt writes anything, so this tier says something on attempt 0
    and never depends on attempt history. The common case pays nothing: a change with no
    channel import never touches the checkout walk.
    """
    sources = _change_sources(changed_paths, file_tools, previous_attempt_diff, assigned_paths)

    def read(path: str) -> str | None:
        try:
            return file_tools.read_file(path)
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            return None

    scan = channel_seam_scan(
        sources,
        (item.as_posix() for item in existing_files),
        read,
        exclude=set(changed_paths),
    )
    return tuple(item.path for item in scan.co_importers)


@overload
def _assigned_first(items: Sequence[str], assigned_paths: Sequence[str]) -> list[str]: ...


@overload
def _assigned_first(
    items: Sequence[tuple[str, str]], assigned_paths: Sequence[str]
) -> list[tuple[str, str]]: ...


def _assigned_first(
    items: Sequence[str] | Sequence[tuple[str, str]], assigned_paths: Sequence[str]
) -> list[str] | list[tuple[str, str]]:
    """Order sources so the files the plan named come first, keeping the rest as they were.

    A stable sort, so sources the plan did not name stay in the order the attempt produced
    them. Nothing is dropped: this decides who reaches a bounded budget first, not who is
    eligible for it.
    """
    rank = {path: index for index, path in enumerate(assigned_paths)}
    return sorted(  # type: ignore[return-value]
        items,
        key=lambda item: rank.get(item if isinstance(item, str) else item[0], len(rank)),
    )


def _diagnostic_symbol_paths(
    diagnostics: Sequence[str],
    changed_paths: Sequence[str],
    existing_files: set[Path],
    file_tools: FileTool,
    previous_attempt_diff: str = "",
) -> tuple[str, ...]:
    """Return the files defining the modules a blocking diagnostic named a symbol of.

    `appValidation.addAppSchema.body is undefined` says two things. One is prose the next
    attempt is already told. The other is structural and was being thrown away: whatever
    `appValidation` is, the attempt had the wrong idea of its shape, and this checkout can
    say exactly which file it came from -- the import that binds that name.

    Resolution goes through the import, never through the name: a symbol is only followed
    when a source of this change imports something under exactly that name, and only when
    that specifier resolves to a file already in the checkout. So a diagnostic quoting a
    dotted filename, a dependency's API, or an ordinary sentence resolves to nothing and
    costs nothing, and no heuristic here has to know what any name means.
    """
    symbols = {
        match.group("root") for text in diagnostics for match in _DIAGNOSTIC_SYMBOL.finditer(text)
    }
    if not symbols:
        return ()
    sources: list[tuple[str, str]] = list(_diff_added_source(previous_attempt_diff))
    for changed in changed_paths:
        try:
            sources.append((changed, file_tools.read_file(changed)))
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            continue
    resolved: list[str] = []
    for path, content in sources:
        directory = PurePosixPath(path).parent
        for line in content.splitlines():
            for specifier in module_specifiers(line):
                # The specifier names the module, not the binding. Read the names around it
                # with it removed, so a path that happens to contain the symbol's spelling
                # cannot stand in for an import that actually binds it.
                if not symbols & set(_IDENTIFIER.findall(line.replace(specifier, " "))):
                    continue
                candidate = resolve_specifier(specifier, directory, existing_files)
                if candidate is not None and candidate not in resolved:
                    resolved.append(candidate)
                if len(resolved) >= _MAX_DIAGNOSTIC_CONTEXT_PATHS:
                    return tuple(resolved)
    return tuple(resolved)


def _blocking_diagnostics(
    prior_review: ReviewArtifact | None, execution_context: dict[str, Any]
) -> tuple[str, ...]:
    """Return the text of what blocked the previous attempt, from both places it survives.

    The review is the richer source and is present whenever one ran. The retry plan is read
    too because it is what a retry granted without a review still carries, and because it is
    the field `build_retry_plan` fills from the findings -- so a path that reaches the
    engineer without the review artifact is not silently left with nothing to go on.

    A finding's structured `file_path` is spelled back into the text, because the location
    selectors downstream read only text: a finding whose prose never repeats its own citation
    would compete for context by wording alone, exactly the gap 47- Part A left open. The
    citation is a claim, not a fact -- `_diagnostic_file_locations` only promotes a path that
    resolves to a real file in this workspace, so a wrong citation costs nothing.

    A context-partition scope (80-) outranks both sources: a scoped attempt of a partition
    wave is narrowed to exactly its cluster's diagnostic texts, so the prior review's full
    findings -- the union whose required set could not fit -- must not be re-derived here.
    """
    partition = _context_partition_scope(execution_context)
    if partition is not None:
        scoped = [
            item for item in partition.get("blocking_diagnostics", []) if isinstance(item, str)
        ]
        if scoped:
            return tuple(scoped)
    texts: list[str] = []
    if prior_review is not None:
        for finding in prior_review.findings:
            if finding.severity not in _BLOCKING_SEVERITIES:
                continue
            texts.extend(
                value
                for value in (
                    finding.title,
                    finding.description,
                    finding.recommendation,
                    finding.evidence,
                    finding.recommended_fix,
                )
                if value
            )
            if finding.file_path:
                location = (
                    finding.file_path
                    if finding.line_number is None
                    else f"{finding.file_path}:{finding.line_number}"
                )
                texts.append(f"The review locates this finding at {location}")
    retry_plan = execution_context.get("retry_plan")
    if isinstance(retry_plan, dict):
        if isinstance(root_cause := retry_plan.get("root_cause"), str):
            texts.append(root_cause)
        texts.extend(
            item
            for item in retry_plan.get("files_or_areas_to_inspect", [])
            if isinstance(item, str)
        )
    return tuple(texts)


def _diff_added_source(diff: str) -> list[tuple[str, str]]:
    """Return each file in a unified diff with the lines that attempt added.

    Added lines only. A removed `require` is a dependency the attempt was getting rid of,
    and forcing its module into the next attempt's budget would spend the space on the one
    file the change is trying to stop using.
    """
    sources: list[tuple[str, str]] = []
    path: str | None = None
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ "):
            if path is not None and added:
                sources.append((path, "\n".join(added)))
            target = line[4:].strip()
            path = target[2:] if target.startswith(("a/", "b/")) else target
            path = None if path == "/dev/null" else path
            added = []
        elif path is not None and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    if path is not None and added:
        sources.append((path, "\n".join(added)))
    return sources


def _diff_source_paths(diff: str) -> tuple[str, ...]:
    """Return the files a captured attempt wrote, for matching against the plan's areas.

    The reset path is the case: what a rejected attempt wrote in a NEW file survives only in
    the capture, so on that path `changed_paths` alone cannot say which of the plan's areas
    this change actually touched.
    """
    return tuple(dict.fromkeys(path for path, _content in _diff_added_source(diff)))


def _registry_context_paths(registry_paths: Any) -> list[str]:
    """Return the bounded registry list, so the context and the snapshot cannot disagree."""
    if not isinstance(registry_paths, list):
        return []
    paths: list[str] = []
    for path in registry_paths:
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)
        if len(paths) == _MAX_REGISTRY_CONTEXT_PATHS:
            break
    return paths


def _wiring_repair_paths(
    execution_context: dict[str, Any], keys: tuple[str, ...] = ("target_path", "example_path")
) -> tuple[str, ...]:
    """Return the files scoped wiring repairs name, selected by which claim each key makes.

    `target_path` is the file to change -- an order, whose drop refuses the attempt;
    `example_path` is only there to be read, so the engineer can see how this repository
    already reaches a module of the same kind -- advisory, its drop recorded and survived.
    The two citizenships pack at opposite ends of the required set, which is why callers
    select by key instead of taking the union this function used to return.

    Every repair contributes, not just the first. A change that leaves two modules
    unreachable is told about both, and quoting a file back exactly is only possible for a
    file in the snapshot: with one target forced in, the second diagnostic asked for an edit
    to a file the engineer could not see, and repeated verbatim on the next attempt.
    """
    retry_plan = execution_context.get("retry_plan")
    if not isinstance(retry_plan, dict):
        return ()
    repairs = retry_plan.get("wiring_repairs")
    if not isinstance(repairs, list):
        repair = retry_plan.get("wiring_repair")
        repairs = [repair] if isinstance(repair, dict) else []
    paths: list[str] = []
    for repair in repairs:
        if not isinstance(repair, dict):
            continue
        paths.extend(
            value
            for key in keys
            if isinstance(value := repair.get(key), str) and value and value not in paths
        )
    return tuple(paths)


def _wiring_repair_target_paths(execution_context: dict[str, Any]) -> tuple[str, ...]:
    """Return only the files the wiring repairs order this attempt to edit."""
    return _wiring_repair_paths(execution_context, keys=("target_path",))


def _wiring_repair_example_paths(execution_context: dict[str, Any]) -> tuple[str, ...]:
    """Return only the files the wiring repairs offer as siblings to imitate."""
    return _wiring_repair_paths(execution_context, keys=("example_path",))


def _repository_context(
    existing_files: set[Path],
    file_tools: FileTool,
    relevance_terms: frozenset[str] = frozenset(),
    required_paths: tuple[str, ...] = (),
    required_line_numbers: dict[str, int] | None = None,
    *,
    # Paths a bounded selector upstream resolved and refused, so they never reached
    # ``required_paths`` to be dropped here. Reported, never packed.
    candidates_discarded: Mapping[str, str] | None = None,
    # Candidates this checkout's own definition-site lookup found to define a name the plan or
    # the rejecting review names in prose -- a second, independent relevance signal alongside
    # ``relevance_terms``'s path-word overlap. Additive only: a path in this set is never
    # ranked worse for it, only ever no worse than the term heuristic already had it.
    relevance_defining_paths: frozenset[str] = frozenset(),
    # The budgets as parameters with constant defaults (the `contract_sections` precedent):
    # the derivation from the declared model window happens exactly once, at the composition
    # site, and this function packs under whatever it was handed.
    max_characters: int = _REPOSITORY_CONTEXT_MAX_CHARACTERS,
    per_file_max_characters: int = _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS,
) -> dict[str, Any]:
    """Give the coding model a small, secret-free snapshot of the cloned repository.

    ``required_paths`` get first claim on the whole character budget, before any ranked
    candidate spends a byte of it. A scoped repair has to quote the target file back exactly,
    so that file cannot be left to compete with the rest of the checkout -- and "compete" is
    exactly what the previous greedy walk allowed: it skipped any file that did not fit the
    remaining budget while continuing to pack smaller ones, so 176's 11.5 KB service file
    lost its slot to 2022 log files that still squeezed in later.

    A required file larger than the per-file bound is excerpted around the line the
    diagnostics named (``required_line_numbers``) rather than dropped silently, at a line
    radius derived from the per-file character budget -- the budget, not the thirty-line
    repair radius, is what a required file is entitled to. Which of two different things
    happened to a required path is returned as two lists, because the 61- analysis showed
    the old union was read as one thing and meant either: ``required_excerpted_paths`` names
    files that ARE in the snapshot as a bounded window, and ``required_dropped_paths`` names
    files that are not in the snapshot at all, with the reason per path in
    ``required_dropped_reasons``. ``required_omitted_paths`` remains their union for one
    release so existing readers do not silently change meaning. Required files still pass
    through ``carries_key_material`` like everything else; being required is not permission
    to leak.

    ``candidates_discarded`` is the third thing that can happen to a file the change depends
    on, and until AB-Feature-218 it was the invisible one: a bounded selector upstream
    resolved the path and refused it, so it never became a required path and therefore could
    not be reported as dropped. These are surfaced under their own key rather than merged
    into ``required_dropped_paths``, deliberately -- `_refuse_if_required_context_dropped`
    reads that list and stops the attempt, and a path that was never ordered must never be
    able to stop one. A candidate that reached the snapshot by some other route is not
    reported at all: what is recorded is what the prompt did not get.
    """
    # The inventory is paths only, and it is deliberately screened by name alone. A filename
    # can disclose -- a service-account key is named for the project it belongs to -- but the
    # only way to catch one here is to read every one of a thousand candidates, where the
    # snapshot below reads sixty. That is a second pass over the checkout to withhold a
    # string, and against it stands what 35- established: telling an attempt a file exists is
    # most of what stops it inventing one. So a file caught by its bytes stays listed and its
    # contents still never leave. Weighed and chosen, not overlooked.
    all_inventory = sorted(path.as_posix() for path in existing_files if _safe_context_path(path))
    inventory = all_inventory[:_REPOSITORY_CONTEXT_MAX_INVENTORY_FILES]
    ranked = _context_candidates(existing_files, relevance_terms, relevance_defining_paths)
    required = [Path(item) for item in required_paths if Path(item) in existing_files]
    required_set = set(required)
    line_numbers = required_line_numbers or {}
    files: list[dict[str, Any]] = []
    required_excerpted: list[str] = []
    required_dropped: list[str] = []
    required_dropped_reasons: dict[str, str] = {}

    def drop(posix: str, reason: str) -> None:
        required_dropped.append(posix)
        required_dropped_reasons[posix] = reason

    characters_used = 0
    for path in required:
        posix = path.as_posix()
        if len(files) >= _REPOSITORY_CONTEXT_MAX_FILES:
            drop(posix, "file_count_cap")
            continue
        try:
            content = file_tools.read_file(path)
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            # Fail closed. A file that cannot be read or decoded cannot be shown to be free
            # of key material, so it is never placed in a model request.
            drop(posix, "unreadable")
            continue
        if carries_key_material(content):
            drop(posix, "key_material")
            continue
        entry: dict[str, Any] = {"path": posix}
        excerpted = False
        if len(content) > per_file_max_characters:
            # The whole file's length, recorded before the window replaces the content, so
            # the prompt can say "you are shown lines 1-431 of 643" rather than leaving the
            # model to treat a window as the file (199 built a parallel module around one).
            total_lines = len(content.splitlines())
            content, start_line, end_line = source_region(
                content,
                line_numbers.get(posix),
                max_characters=per_file_max_characters,
                radius_lines=_required_excerpt_radius(content, per_file_max_characters),
            )
            entry["excerpt_lines"] = {"start": start_line, "end": end_line}
            entry["total_lines"] = total_lines
            excerpted = True
        if characters_used + len(content) > max_characters:
            # Even first claim ends at the whole budget: the required set alone must fit
            # inside it, and one that does not is recorded rather than silently truncated.
            # A file that was excerpted and then still did not fit is a drop, not an
            # excerpt: what matters downstream is whether its bytes reached the prompt.
            drop(posix, "required_budget_exhausted")
            continue
        if excerpted:
            required_excerpted.append(posix)
        entry["content"] = content
        files.append(entry)
        characters_used += len(content)
    for path in ranked:
        if path in required_set:
            continue
        if len(files) >= _REPOSITORY_CONTEXT_MAX_FILES:
            break
        try:
            content = file_tools.read_file(path)
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            # Fail closed, as above; that this also skips files for unrelated reasons is
            # incidental. The safety case is the reason.
            continue
        # The path cleared `_safe_context_path`, which is only half the question: a Google
        # service-account key is indistinguishable from `package.json` by filename. These
        # are bytes already read for context, so asking costs no second pass.
        if carries_key_material(content):
            continue
        if len(content) > per_file_max_characters:
            continue
        if characters_used + len(content) > max_characters:
            continue
        files.append({"path": path.as_posix(), "content": content})
        characters_used += len(content)
    # An upstream refusal only counts if the file really is absent from the prompt. A
    # candidate the ranked walk picked up anyway was shown, and reporting it as discarded
    # would be a false negative in exactly the field a forensic pass consults first.
    shown = {item["path"] for item in files}
    candidate_discards = {
        path: reason for path, reason in (candidates_discarded or {}).items() if path not in shown
    }
    return {
        # The budget this snapshot was packed under, on the record itself: what the refusal
        # sentences and the completion metadata state about the budget is a lookup of these
        # keys and never a re-derivation (the 61- law).
        "budget_characters": max_characters,
        "budget_per_file_characters": per_file_max_characters,
        "file_inventory": inventory,
        "files": files,
        "omitted_file_count": max(0, len(all_inventory) - len(files)),
        "omitted_inventory_file_count": max(0, len(all_inventory) - len(inventory)),
        "required_excerpted_paths": required_excerpted,
        "required_dropped_paths": required_dropped,
        "required_dropped_reasons": required_dropped_reasons,
        # Deliberately outside `required_dropped_paths`: these were resolved and refused
        # before they could be ordered, and the refusal that stops an attempt reads that
        # list. Recorded so a discard is answerable; never able to fail an attempt.
        "candidate_discarded_paths": sorted(candidate_discards),
        "candidate_discarded_reasons": dict(sorted(candidate_discards.items())),
        # The union of the two lists above, kept for one release so readers of the old
        # field -- the 52-/58- reports among them -- do not silently change meaning.
        "required_omitted_paths": list(dict.fromkeys([*required_excerpted, *required_dropped])),
    }


def _required_excerpt_radius(content: str, max_characters: int) -> int:
    """The line radius at which ``max_characters``, not the radius, is what binds.

    A required file's excerpt is entitled to the whole per-file character budget, and the
    fixed thirty-line radius was throwing most of it away: 199's 17,905-character service
    file fit two-thirds of itself inside 12,000 characters and was shown 61 lines. Derived
    from this file's own average line length so the selected window slightly overshoots the
    budget and ``source_region``'s trimming makes the fine cut; floored at the default
    radius so a file of enormous lines never gets a *narrower* window than it does today.
    """
    average_line_length = max(1, len(content) // max(1, len(content.splitlines())))
    return max(SOURCE_REGION_RADIUS_LINES, max_characters // average_line_length // 2 + 1)


class RequiredContextRefusal(DiagnosedFailure):
    """The snapshot dropped a file this attempt is ordered to work on, so the attempt refuses.

    Raised before the model is called, which is the whole point: 58 recorded attempts in
    runs 182-199 ran against a snapshot missing a required file and burned their budget
    guessing at code they were never shown. The refusal is a decision, not a fault -- like
    ``GitSafetyError`` it escapes the child loop without spending any retry counter, and its
    diagnostics are self-built from platform-owned sentences and repository paths, which is
    what lets ``safe_error_diagnostics`` record them verbatim.

    Deliberately narrow: it fires only for a path the plan assigned or a blocking diagnostic
    named that was *dropped* -- absent from the prompt entirely. It is not a judgement of
    whether the context was "sufficient"; the 61- analysis (Spec-2) forbids growing this
    check into that authority.
    """

    def __init__(
        self,
        diagnostics: Sequence[str],
        *,
        # The ordered files that did not fit, and the budget facts the snapshot record
        # already carries -- structured so the workflow's context partition can cluster a
        # refused retry's demands without parsing sentences. Optional with empty defaults
        # so every existing raise (and the workflow tests' stand-ins) still constructs.
        dropped_paths: Sequence[str] = (),
        budget_max_characters: int | None = None,
        budget_source: str | None = None,
    ) -> None:
        super().__init__(
            "a required file was dropped from the repository snapshot; "
            "the attempt was refused before the model was called",
            # The 61- analysis files D2 as a PLATFORM-layer defect: the context selector
            # could not supply a file the order names. An existing value rather than a new
            # one -- item 26 established that a new classification springs the two-authority
            # retryability trap, so the refusal-specific facts live in the sentences.
            classification=FeatureFailureClassification.PLATFORM_DEFECT,
            diagnostics=tuple(diagnostics),
        )
        self.dropped_paths = tuple(dropped_paths)
        self.budget_max_characters = budget_max_characters
        self.budget_source = budget_source


# How each drop reason reads in a refusal sentence. Keys are the values recorded in
# ``required_dropped_reasons``; the key-material reason is deliberately absent because its
# refusal names redaction policy, never the context budget.
# Where the platform writes a design's exported artwork, mirrored from
# `services.design_resolution._ASSET_DIRECTORY`. Duplicated rather than imported because
# `agents` must not depend on `services`; a test asserts the two agree.
_PLACED_ASSET_DIRECTORY = ".design/assets"

# The drop reason that means "these bytes are not text", and the extensions for which that is
# a permanent fact rather than a budget decision.
_UNREADABLE_DROP_REASON = "unreadable"
_UNUSABLE_BINARY_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".ico", ".pdf", ".woff", ".woff2"}
)

_DROP_REASON_SENTENCES = {
    "file_count_cap": "the snapshot's file-count cap was already spent",
    "unreadable": "the file could not be read or decoded",
    "required_budget_exhausted": (
        "the required set exceeded the snapshot's total character budget"
    ),
}


def _is_unusable_binary_drop(path: str, reason: str) -> bool:
    """Whether a dropped file is one no attempt could have used the bytes of anyway.

    The refusal exists because "an attempt that cannot see its own target answers findings
    with guesses". That reasoning is about *source*: text the attempt has to read to change.
    It does not reach an image. A PNG cannot be quoted into a text snapshot -- it is dropped
    as `unreadable` every single time, by construction -- and the Engineer could not have
    edited its bytes if it had been. Refusing over one costs an attempt and tells nobody
    anything.

    **This was keyed on a directory and that was the mistake.** `fix/106` excluded
    `.design/assets`, the staging path artwork was written to; `fix/108` then moved artwork to
    wherever the repository actually keeps its images, and the exclusion silently stopped
    matching. AB-Feature-241 died on exactly the files AB-Feature-234 had died on, under their
    new home in `public/`. A rule about *what a file is* survives a change to *where it goes*;
    a rule about where it lives does not.

    Still narrow: only the `unreadable` reason, and only for image extensions. A dropped
    *source* file the plan assigned is a real refusal and stays one, and so does an image
    dropped because a budget ran out rather than because it could not be decoded.
    """
    if reason != _UNREADABLE_DROP_REASON:
        return False
    return Path(path).suffix.lower() in _UNUSABLE_BINARY_SUFFIXES


def _refuse_if_required_context_dropped(
    repository_context: dict[str, Any],
    *,
    assigned_paths: Sequence[str],
    diagnostic_file_paths: Sequence[str],
    wiring_target_paths: Sequence[str] = (),
    budget_source: str | None = None,
) -> None:
    """Refuse the attempt when a file the attempt is ordered to work on was dropped.

    Excerpted files do not refuse -- their bytes are in the prompt, bounded, and the prompt
    says which lines. Dropped files the order does not name do not refuse either: required
    paths routinely include benign force-includes (a lockfile, a registry, a wiring repair's
    example file) whose absence an attempt survives, and over-refusing on those would turn a
    working attempt into a stop. Only the intersection -- dropped, and named by the plan, a
    blocking diagnostic, or a wiring repair's `target_path` (an order: an attempt that
    cannot see the file it was told to rewire answers with guesses) -- is an attempt that
    would run blind against its own target, and that one must not run.

    Everything stated here is read off ``repository_context``'s own record -- the drop
    lists and the budget keys the packer wrote -- never re-derived (the 61- law).
    ``budget_source`` is carried onto the exception for the same record-keeping reason; it
    decides nothing.
    """
    named = {*assigned_paths, *diagnostic_file_paths, *wiring_target_paths}
    reasons = repository_context.get("required_dropped_reasons", {})
    dropped = [
        path
        for path in repository_context.get("required_dropped_paths", [])
        if path in named and not _is_unusable_binary_drop(path, str(reasons.get(path, "")))
    ]
    if not dropped:
        return
    sentences = [
        "This attempt was refused before the model was called: the repository snapshot "
        "dropped a file the attempt is ordered to work on, and an attempt that cannot see "
        "its own target answers findings with guesses. No retry was spent on this refusal."
    ]
    for path in dropped:
        reason = reasons.get(path, "")
        if reason == "key_material":
            sentences.append(
                f"Required file {path} is named by the plan or by a blocking diagnostic, "
                "but its bytes carry key material, so it is withheld by redaction policy "
                "-- not by any context budget -- and no attempt may be shown it. The work "
                "needs re-scoping away from that file, or the secret moved out of it."
            )
        else:
            explanation = _DROP_REASON_SENTENCES.get(reason, "the snapshot could not include it")
            sentences.append(
                f"Required file {path} is named by the plan or by a blocking diagnostic, "
                f"but was dropped from the snapshot: {explanation}."
            )
    raise RequiredContextRefusal(
        sentences,
        dropped_paths=dropped,
        budget_max_characters=repository_context.get("budget_characters"),
        budget_source=budget_source,
    )


def _context_candidates(
    existing_files: set[Path],
    relevance_terms: frozenset[str] = frozenset(),
    relevance_defining_paths: frozenset[str] = frozenset(),
) -> list[Path]:
    """Prefer checkout root files, then the candidates this task actually names.

    The content reader already rejects binary, oversized, and invalid UTF-8 files.  Do
    not decide what counts as source by maintaining a language-extension allowlist.

    Nested files were once ordered by path alone, which made the snapshot alphabetical
    rather than relevant: a checkout whose `middlewares/` sits after several full
    `controllers/` directories never fit the file budget, so the model was told a module
    existed without being shown how it exports, and it guessed a shape instead.
    """
    safe_files = {path for path in existing_files if _safe_context_path(path)}
    root_files = sorted(
        (path for path in safe_files if path.parent == Path(".")), key=_context_path_order
    )
    nested_files = sorted(
        (path for path in safe_files if path.parent != Path(".")),
        key=lambda path: _context_path_order(path, relevance_terms, relevance_defining_paths),
    )
    return [*root_files, *nested_files]


def _context_path_order(
    path: Path,
    relevance_terms: frozenset[str] = frozenset(),
    relevance_defining_paths: frozenset[str] = frozenset(),
) -> tuple[int, int, str]:
    """Prioritize task relevance, then repository metadata, without a language decision.

    Metadata is only the preference to fall back on when nothing better is known. Ranking
    it ahead of relevance starved the signal entirely: a checkout carrying a directory of
    schema documents spent the whole file budget on them, so the module the task names was
    never shown and the model kept guessing at an interface it was never allowed to read.
    """
    metadata_first = path.suffix.lower() in _REPOSITORY_CONTEXT_METADATA_SUFFIXES
    score = _relevance_score(path, relevance_terms, relevance_defining_paths)
    return (-score, 0 if metadata_first else 1, path.as_posix())


def _relevance_score(
    path: Path,
    relevance_terms: frozenset[str],
    relevance_defining_paths: frozenset[str] = frozenset(),
) -> int:
    """Count the distinct task terms a path matches, weighting its own file name highest.

    ``relevance_defining_paths`` is a second, independent signal: this checkout's own
    definition-site lookup found the path defines a name the plan or the rejecting review
    names in its own prose, not by path-word overlap. Additive only -- a path the term
    heuristic already ranked keeps that rank; this only lifts a candidate term overlap alone
    would have missed.
    """
    if not relevance_terms and path.as_posix() not in relevance_defining_paths:
        return 0
    stem_terms = set(_split_context_terms(path.stem))
    directory_terms = {term for parent in path.parts[:-1] for term in _split_context_terms(parent)}
    # A directory says which area of the repository a file belongs to; the file name says
    # whether this is the module the task talks about. Rank the stronger claim higher.
    symbol_score = 1 if path.as_posix() in relevance_defining_paths else 0
    return (
        2 * _matching_term_count(stem_terms, relevance_terms)
        + _matching_term_count(directory_terms, relevance_terms)
        + symbol_score
    )


def _matching_term_count(words: set[str], relevance_terms: frozenset[str]) -> int:
    """Count path words a task term names, allowing one to be a prefix of the other.

    Prose and paths rarely agree on word boundaries: a review asking about the `homepage`
    has to reach `Home.js`, which an equality test never does. Both sides must be long
    enough that a shared prefix means something.
    """
    return sum(
        1
        for word in words
        if word in relevance_terms
        or (
            len(word) >= _REPOSITORY_CONTEXT_MIN_PREFIX_MATCH
            and any(
                len(term) >= _REPOSITORY_CONTEXT_MIN_PREFIX_MATCH
                and (term.startswith(word) or word.startswith(term))
                for term in relevance_terms
            )
        )
    )


def _split_context_terms(text: str) -> set[str]:
    """Split a path segment into lowercase words across camel case and separators."""
    words: set[str] = set()
    for token in _REPOSITORY_CONTEXT_TERM_PATTERN.findall(text):
        words.add(token.lower())
        for part in re.findall(r"[A-Z]?[a-z0-9]+", token):
            if len(part) > 2:
                words.add(part.lower())
    return words - _REPOSITORY_CONTEXT_GENERIC_TERMS


def _relevance_terms(*texts: str) -> frozenset[str]:
    """Derive the vocabulary this task uses, so context can be selected by relevance.

    The terms come only from the plan and the review that rejected the last attempt, so
    nothing here encodes one repository's layout, language, or framework.
    """
    terms: set[str] = set()
    for text in texts:
        for token in _REPOSITORY_CONTEXT_TERM_PATTERN.findall(text):
            terms |= _split_context_terms(token)
    return frozenset(terms)


def _safe_context_path(path: Path) -> bool:
    """Screen a path before it is read. Clearing this is not on its own permission to send.

    Only the filename is decidable here, and a service-account key hides behind an ordinary
    one. `_repository_context` asks `carries_key_material` of the bytes as well.
    """
    return is_model_safe_context_path(path)


def _context_partition_scope(execution_context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the context-partition cluster scope this attempt runs under, if any (80-).

    Carried inside the persisted retry plan because that is the one channel that already
    flows from the workflow's durable child state into this agent's execution context. A
    scoped attempt of a partition wave sees only its cluster's diagnostics and files; the
    review scope on the task plan stays whole, because the review judges the workstream,
    not the delta (66-).
    """
    retry_plan = execution_context.get("retry_plan")
    if not isinstance(retry_plan, dict):
        return None
    partition = retry_plan.get("context_partition")
    return partition if isinstance(partition, dict) else None


def _execution_context(task_plan: TaskPlanArtifact) -> dict[str, Any]:
    """Expose only persisted retry/preflight context that changes an engineer strategy."""
    review_scope = task_plan.metadata.get("review_scope")
    scope = review_scope if isinstance(review_scope, dict) else {}
    context = {
        "model_routing": _routing_context(task_plan.metadata.get("model_routing")),
        "repository_revision": task_plan.metadata.get("repository_revision"),
        "preflight_result": task_plan.metadata.get("preflight_result"),
        "retry_plan": task_plan.metadata.get("retry_plan"),
        # Truncated here rather than at the point of use, so what the engineer is told about
        # is exactly what was force-included: advertising a registry that lost the budget is
        # how a prompt ends up naming a file the snapshot does not contain.
        "registry_paths": _registry_context_paths(task_plan.metadata.get("registry_paths")),
        # The plan's own answer to "which files does this workstream touch". Force-included
        # below rather than left to compete on wording: `src/pages/Profile.js` scores nothing
        # against a task about build information, and AB-Feature-166 spent three attempts
        # writing an API helper without ever editing the page it was assigned.
        "expected_files_or_areas": scope.get("expected_files_or_areas", []),
        "implementation_expectations": scope.get("implementation_expectations", []),
        "scoped_requirements": scope.get("scoped_requirements", []),
    }
    # A context-partition cluster narrows what this attempt is ordered to work on to its
    # own file set (80-). The review scope above is deliberately left whole.
    partition = _context_partition_scope(context)
    if partition is not None:
        files = [
            item for item in partition.get("expected_files_or_areas", []) if isinstance(item, str)
        ]
        if files:
            context["expected_files_or_areas"] = files
    return context


def _routing_context(model_routing: Any) -> dict[str, Any] | None:
    """Tell the model what kind of job this is, and nothing about how it was selected.

    The fingerprints and model name are the platform's bookkeeping:
    they would only invite the response to reason about its own selection, and a model asked to
    justify being the cheap choice is not being asked to fix the finding. What it needs is
    whether this is a first implementation or a correction, and how much of the system the
    correction has been judged to touch.
    """
    if not isinstance(model_routing, dict):
        return None
    context = {
        key: model_routing.get(key)
        for key in (
            "execution_mode",
            "failure_classification",
            "classification",
            "repair_scope",
        )
        if model_routing.get(key) is not None
    }
    return context or None


def _completion_evidence(
    changes: list[FileChange], file_tools: FileTool, execution_context: dict[str, Any]
) -> dict[str, Any]:
    """Build file/symbol evidence without allowing test edits to imply production completion."""
    paths_by_kind: dict[str, list[str]] = {
        "production": [],
        "test": [],
        "configuration": [],
    }
    symbols_by_path: dict[str, list[str]] = {}
    for change in changes:
        kind = classify_file_change(change.path)
        if kind in paths_by_kind:
            paths_by_kind[kind].append(change.path)
        if kind == "production":
            try:
                symbols_by_path[change.path] = _source_symbols(file_tools.read_file(change.path))
            except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
                symbols_by_path[change.path] = []
    expectations = execution_context.get("implementation_expectations", [])
    implemented: list[str] = []
    missing: list[str] = []
    satisfied: list[str] = []
    evidence: list[dict[str, Any]] = []
    for item in expectations if isinstance(expectations, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("requirement_id"), str):
            continue
        requirement_id = item["requirement_id"]
        categories = item.get("expected_change_categories", [])
        production_required = isinstance(categories, list) and any(
            category
            in {
                "production",
                "route",
                "controller",
                "service",
                "model",
                "migration",
                "frontend_component",
                "client",
            }
            for category in categories
        )
        tests_required = bool(item.get("tests_required", False))
        completed = (not production_required or bool(paths_by_kind["production"])) and (
            not tests_required or bool(paths_by_kind["test"])
        )
        if not completed:
            missing.append(requirement_id)
            continue
        artifact_paths = (
            paths_by_kind["production"] if production_required else paths_by_kind["test"]
        )
        if not artifact_paths:
            missing.append(requirement_id)
            continue
        implemented.append(requirement_id)
        satisfied.append(requirement_id)
        evidence.append(
            RequirementImplementationEvidence(
                requirement_id=requirement_id,
                files=artifact_paths,
                symbols=[
                    symbol for path in artifact_paths for symbol in symbols_by_path.get(path, [])
                ],
                description="Requirement is evidenced by the changed repository files.",
                validation_results=[],
            ).model_dump(mode="json")
        )
    return {
        "production_files_changed": paths_by_kind["production"],
        "test_files_changed": paths_by_kind["test"],
        "configuration_files_changed": paths_by_kind["configuration"],
        "requirements_implemented": list(dict.fromkeys(implemented)),
        "requirements_not_implemented": list(dict.fromkeys(missing)),
        "implementation_expectations_satisfied": list(dict.fromkeys(satisfied)),
        "requirement_implementation_evidence": evidence,
        "production_diff_fingerprint": _production_content_fingerprint(
            paths_by_kind["production"], file_tools
        ),
    }


def _production_content_fingerprint(paths: list[str], file_tools: FileTool) -> str | None:
    """Hash production path and content so same-file remediation counts as real progress."""
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        content = file_tools.read_file(path).encode("utf-8")
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _source_symbols(content: str) -> list[str]:
    """Extract a small, language-agnostic symbol hint for requirement-to-file evidence."""
    import re

    matches = re.findall(
        r"(?:class|def|function|const|let|export\s+(?:async\s+)?function)\s+([A-Za-z_][A-Za-z0-9_]*)",
        content,
    )
    return list(dict.fromkeys(matches))[:24]
