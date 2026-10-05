"""Validation-backed quality-review node."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from base64 import b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import ValidationError

from adapters.git_adapter import GitSafetyError
from adapters.interruptible_git import (
    HASH_CHUNK_BYTES,
    reviewed_content_fingerprint,
    streamed_file_digest,
)
from adapters.llm_adapter import ImageInput, LLMClient, LLMResponse
from agents.product_manager.agent import AttachmentContentSource
from agents.shared.contract_sections import (
    contract_reference_resolves,
    contract_section_context_from_state,
    contract_section_context_json,
    contract_section_names_from_state,
)
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    AgentArtifactError,
    artifact_id_matches_lineage,
    artifact_json,
    artifact_update,
    attempt_artifact_id,
    create_artifact,
    execution_metadata,
    parse_model_json,
    require_artifact,
)
from agents.shared.design_snapshot import (
    design_snapshot_context_from_state,
    design_snapshot_context_json,
)
from artifacts.schemas import (
    CodeCompletionArtifact,
    FileChange,
    ReviewArtifact,
    ReviewFinding,
    TaskPlanArtifact,
    TechnicalPRDArtifact,
)
from configs.model_roles import ModelRole
from prompts.prompt_loader import PromptLoader
from services.cancellation import CancellationRequested, CancellationToken, MockCancellationToken
from services.external_operations import ExternalOperationExecutor
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessRunner,
    redact_source_credentials,
    sanitized_subprocess_environment,
)
from state.external_operations import ExternalOperationType
from state.models import AgentState
from storage.external_operation_store import OperationResult
from tools.acceptance_criteria import (
    UnverifiableCriterion,
    requires_deployed_measurement,
    unverifiable_criteria,
)
from tools.channel_packages import SeamCoImporter, channel_seam_scan
from tools.dependency_sync import generated_lockfile_manifest
from tools.file_tools import (
    DEFAULT_MAX_FILE_BYTES,
    WorkspaceFileTools,
    carries_key_material,
    is_credential_shaped_path,
    resolve_workspace_path,
    resolve_workspace_root,
)
from tools.implementation_completeness import classify_file_change
from tools.lineage_base import (
    CONTRACT_PROJECTION_PATH,
    LineageBaseRevision,
    branch_change_evidence,
)
from tools.lockfile_hosts import scan_resolved_hosts
from tools.repo_tools import DEFAULT_IGNORED_DIRECTORIES, scan_directory
from tools.resolved_issue_ledger import SettledQuestion
from tools.review_fix_classification import fingerprint_for_text
from tools.scoped_tests import seam_mock_notes
from tools.source_windows import source_region
from tools.validation_tools import (
    CAPACITY_FAILURE_CLASSIFICATIONS,
    VALIDATION_RESOURCE_EXHAUSTED,
    ValidationCommand,
    ValidationResult,
    ValidationStatus,
    ValidationTool,
    validation_summary,
)

_REVIEW_EVIDENCE_MAX_FILES = 48
_REVIEW_EVIDENCE_MAX_CHARACTERS = 128_000
# Raised from 16,000 after AB-Feature-168, where the two principal implementation files came
# in at 16,346 and 16,828 characters -- over by 2% and 5% -- and the resulting truncation
# ended both repositories. Headroom is not the real fix, which is below: truncation is now a
# thing the next attempt can act on rather than a reason to stop. But a limit that a
# straightforward React modal exceeds by 346 characters is simply set too low.
_REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS = 24_000

# The constants above are the DEFAULT evidence budget: what a review runs under when the
# deployment has not declared the routed model's context window. Mirrors
# `agents.engineer.agent.repository_snapshot_budget` -- the same reasoning applies here as
# there: a table of hand-tuned constants must be corrected by hand every time a workload or a
# model changes (this file's own per-file constant was already raised once, from 16,000, after
# AB-Feature-168), while a budget derived from the routed model's declared context window
# corrects itself. The share and ratio below are chosen so that a declared 200,000-token
# window -- the current-generation size these constants were tuned against -- derives exactly
# the two constants above, which is the identity property a test pins: a deployment that
# declares nothing behaves byte-identically to one that declares the size already in use.
_EVIDENCE_WINDOW_TOKEN_SHARE = 0.16
_EVIDENCE_CHARACTERS_PER_TOKEN = 4
_EVIDENCE_PER_FILE_SHARE_OF_MAX = (
    _REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS / _REVIEW_EVIDENCE_MAX_CHARACTERS
)


@dataclass(frozen=True, slots=True)
class ReviewEvidenceBudget:
    """The character budget one review's workspace evidence ran under, and its origin.

    ``seam_reserved_characters`` is what the channel-seam scan (the unchanged modules that
    co-import a package the change imports) is guaranteed even when changed-file evidence has
    already claimed the rest of ``max_characters``. Without a reserved floor, a change whose
    own diff is merely large -- not any larger than usual -- can starve seam evidence to zero
    before a single co-importer is considered, which is indistinguishable from a change that
    never touched a shared client at all. One file's worth is the floor: enough to show the
    single most relevant co-importer (the ranking in `channel_seam_scan` already puts the
    actual configuration module first), which is a strictly better position than showing none.

    ``source`` is ``"declared:<model>:<window>"`` when the deployment declared the routed
    model's context window, and ``"default"`` otherwise -- recorded on the completion for the
    same forensic reason `RepositorySnapshotBudget.source` is.
    """

    max_characters: int
    per_file_max_characters: int
    seam_reserved_characters: int
    source: str


def review_evidence_budget(
    model: str | None, declared_windows: Mapping[str, int]
) -> ReviewEvidenceBudget:
    """Derive the review evidence budget from the routed model's declared context window.

    Same shape as `agents.engineer.agent.repository_snapshot_budget`, deliberately: the
    reviewer's evidence assembly is a second, independent place a hardcoded budget goes stale
    exactly the way the engineer's snapshot budget already stopped being allowed to. An
    undeclared model falls back to `_DEFAULT_REVIEW_EVIDENCE_BUDGET`, byte-identical to this
    file's behaviour before this function existed -- absence of a declaration is never a
    punishment.
    """
    window = declared_windows.get(model) if model else None
    if window is None or window <= 0:
        return _DEFAULT_REVIEW_EVIDENCE_BUDGET
    max_characters = int(window * _EVIDENCE_WINDOW_TOKEN_SHARE * _EVIDENCE_CHARACTERS_PER_TOKEN)
    per_file_max_characters = int(max_characters * _EVIDENCE_PER_FILE_SHARE_OF_MAX)
    return ReviewEvidenceBudget(
        max_characters=max_characters,
        per_file_max_characters=per_file_max_characters,
        seam_reserved_characters=per_file_max_characters,
        source=f"declared:{model}:{window}",
    )


_DEFAULT_REVIEW_EVIDENCE_BUDGET = ReviewEvidenceBudget(
    max_characters=_REVIEW_EVIDENCE_MAX_CHARACTERS,
    per_file_max_characters=_REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS,
    seam_reserved_characters=_REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS,
    source="default",
)

# Which limitations are somebody's decision to make, and which are only this platform's own
# bounds. Redaction and a never-publish path mean the review deliberately did not look at
# something, and no number of retries changes that -- a person has to decide.
#
# Truncation and the file-count ceiling are different in kind: they say the change was larger
# than the evidence policy carries, which is a fact about our budget, not a finding about the
# code. Treating them alike ended AB-Feature-168 at `retry_count` 0 with all eight
# implementation retries unspent, on an attempt whose real defect -- thirteen failing tests --
# was exactly what those retries exist to fix.
_HUMAN_DECISION_LIMITATIONS = frozenset({"redacted_content", "sensitive_content", "sensitive_path"})
# Read failures stay a human decision. Unlike a file that is merely long, nothing about the
# next attempt makes an unreadable file readable, and guessing at why is how a broken checkout
# turns into a retry loop.
_UNREADABLE_LIMITATIONS = frozenset({"unavailable_content"})
# `orphan_generated_lockfile` belongs here and emphatically not above. A lockfile the platform's
# own install regenerated is not a file a human has to look at -- it is the package manager's
# output, reproducible from the manifest -- so the only thing worth blocking on is the case
# where it moved *without* the manifest that would explain it. That is a finding a retry can
# resolve, either by making the declaration the churn implies or by reverting the churn, so it
# takes the retryable class exactly like a budget limitation does. Run 189 died because the
# unreadable class had no such distinction: `package-lock.json` at 1,290,320 bytes was refused
# with the severity reserved for a credential, terminally, with every retry unspent.
_EVIDENCE_BUDGET_LIMITATIONS = frozenset(
    {
        "file_limit_exceeded",
        "truncated_content",
        "orphan_generated_lockfile",
        "seam_context_omitted",
    }
)
_ORPHAN_LOCKFILE_LIMITATION = "orphan_generated_lockfile"
# 87- Part C's two dials. How many disjoint changed regions in one declared file are enough
# to ask the reviewer about them region by region, and how many paths the question names
# before it stops. The first is a cost policy and not a rule: a feature may legitimately edit
# one file in eight places, and this asks -- it does not find. Both are recorded on every
# review so the live matrix can retune them, exactly like the second-opinion size measures.
_WIDE_CHANGE_HUNK_COUNT = 6
_MAX_OUT_OF_SCOPE_PATHS = 12
# A module this change co-imports a channel package with -- the seam the change stands on --
# was too large for the evidence budget and was omitted whole rather than trimmed, per the
# 51-A rule: a drop is a drop, and it is declared. The budget class rather than the human
# class, because the omission is a fact about this platform's budget, not about the code;
# treating an unshowable-but-unchanged file as terminal is how run 189 died with every retry
# unspent.
_SEAM_CONTEXT_LIMITATION = "seam_context_omitted"
# The order changed regions are selected in, stated on the entry rather than left for a
# reader to infer from the line numbers. Earliest hunk first, each with the unchanged source
# around it, and context is what gets shed when the budget runs short -- never a hunk.
_CHANGED_REGION_ORDER = "changed-hunks-in-file-order-with-surrounding-context"
# The new-file side of a unified diff hunk header: `@@ -12,3 +14,5 @@`. Only the positions are
# read from the diff, never its text. A unified diff carries the bytes the change *removed*,
# which the current file no longer holds and which the whole-file key-material check therefore
# never examined; quoting from the worktree instead cannot leak them.
_DIFF_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
# The evidence gate's own finding. Named once because two places need it: the gate below,
# which adopts a finding already carrying this id instead of appending its own, and the
# platform-injected census, which must count that adoption as the gate firing.
_REVIEW_EVIDENCE_FINDING_ID = "REVIEW_EVIDENCE_INCOMPLETE"
# The second-opinion gate (75-). The pass runs only for an approved review whose change
# touches a channel package or is big; everything else pays nothing beyond the metadata
# record. Both size numbers are half the primary evidence ceilings above: a change at half
# the budget is one the primary still saw whole, but whose surface is large enough that the
# AB-Feature-206 lesson applies. They are a cost-policy dial, recorded on every review so the
# live matrix can retune them -- not a mechanism.
_SECOND_OPINION_MIN_FILES = 24
_SECOND_OPINION_MIN_CHARACTERS = 64_000
# The read protocol's bounds: one ask, one read. The per-file bound is the primary's own;
# the total stays under the primary evidence ceiling because this pass supplements a review
# that already carried full evidence.
_SECOND_OPINION_MAX_REQUESTED_FILES = 12
_SECOND_OPINION_MAX_CHARACTERS = 96_000
_SECOND_OPINION_INVENTORY_MAX_CHARACTERS = 40_000
_SECOND_OPINION_TEMPLATE = "reviewer/second_opinion_v1.jinja2"
# Merged findings are prefixed so the artifact's finding-id uniqueness holds and a reader can
# tell which authority raised what. The prefix is display identity only: the ledger and the
# 73- demotion fingerprint the description, never the id.
_SECOND_OPINION_FINDING_PREFIX = "SO-"


class ReviewerAgent:
    """Run bounded validation and publish a structured requirement and quality review."""

    def __init__(
        self,
        *,
        prompt_loader: PromptLoader,
        llm_client: LLMClient,
        validation_tool: ValidationTool,
        validation_timeout_seconds: float = 120.0,
        cancellation_token: CancellationToken | None = None,
        process_runner: ProcessRunner | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
        # Recorded on the artifact so a review states which configured role judged the work.
        # Always the review role in the live composition; absent for a deterministic double.
        model_role: ModelRole | None = None,
        # Whether a blocking finding must name what it derives from. Off is the behaviour
        # this agent has always had, byte for byte: nothing below runs.
        bounded_review_scope: bool = False,
        # The design questions a person has decided for this workstream's repository (73-).
        # Only the `removal_holds` entries reach the prompt: the reviewer is told which
        # demands a person overruled, in the decider's own words. Informational -- the
        # platform's verdict handling demotes a re-raised demand whatever the model does
        # with the telling, the same honesty `review_round` is rendered with below.
        settled_questions: Sequence[SettledQuestion] = (),
        # Bytes-by-id, so this agent can be shown the design it is judging against rather than
        # only a description of it. A reviewer reading JSON coordinates cannot tell whether an
        # implementation looks like the picture; one holding the picture can. Optional: a
        # composition without it judges from text exactly as before.
        attachments: AttachmentContentSource | None = None,
        # Derived from the routed model's declared context window where the caller has one to
        # derive it from (`review_evidence_budget`); `None` keeps this agent's original,
        # constant-budget behaviour exactly, byte for byte.
        evidence_budget: ReviewEvidenceBudget | None = None,
    ) -> None:
        """Inject versioned prompting, model assessment, and workspace validation tools."""
        self._prompt_loader = prompt_loader
        self._llm_client = llm_client
        self._validation_tool = validation_tool
        self._validation_timeout_seconds = validation_timeout_seconds
        self._cancellation_token = cancellation_token
        self._process_runner = process_runner or AsyncioProcessRunner(output_limit_bytes=128 * 1024)
        self._operation_executor = operation_executor
        self._model_role = model_role
        self._bounded_review_scope = bounded_review_scope
        self._overruled_demands = [
            {"demand": item.demand, "decision": item.decision, "decided_by": item.decided_by}
            for item in settled_questions
            if item.verdict == "removal_holds"
        ]
        self._attachments = attachments
        self._evidence_budget = evidence_budget or _DEFAULT_REVIEW_EVIDENCE_BUDGET

    async def _design_pictures(self, design: dict[str, Any] | None) -> tuple[ImageInput, ...]:
        """Fetch the frames this review judges against, as pictures.

        The reviewer has held the design as JSON since 96- and still could not answer the one
        question a person actually asks -- does it look like the design. Coordinates do not
        answer that; a picture does.

        Every absence is ordinary and silent here, deliberately: no attachment source, a model
        that cannot see, a frame whose render was refused, a purged blob. Each leaves the
        review exactly as it was before pictures existed, and none of them is a finding about
        the change under review.
        """
        if design is None or self._attachments is None or not self._llm_client.vision_capable:
            return ()
        images: list[ImageInput] = []
        for node in design.get("design_nodes") or []:
            attachment_id = node.get("preview_attachment_id") if isinstance(node, dict) else None
            if not isinstance(attachment_id, str) or not attachment_id:
                continue
            content = await self._attachments.get_content(attachment_id)
            if not content:
                continue
            images.append(
                ImageInput(media_type="image/png", data=b64encode(content).decode("ascii"))
            )
        return tuple(images)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Publish ``007_review.json`` and turn failed validation into structured findings."""
        technical_prd = require_artifact(
            state,
            TechnicalPRDArtifact,
            artifact_id=ARTIFACT_FILENAMES["technical_prd"],
        )
        code_completion = require_artifact(
            state,
            CodeCompletionArtifact,
            artifact_id=ARTIFACT_FILENAMES["code_completion"],
        )
        task_plan = require_artifact(
            state,
            TaskPlanArtifact,
            artifact_id=ARTIFACT_FILENAMES["task_plan"],
        )
        review_scope = _review_scope(task_plan)
        # The contract's own text for the sections this workstream implements or consumes. The
        # same selection the Engineer and its self-review are given, from the same function on
        # the same two inputs: a review that may block on contract compliance and name the
        # section it judges against is the last place that should be inferring the section's
        # contents from its name. `None` for a single-repository workflow, which has no
        # contract, and the prompt then renders nothing.
        contract_sections = contract_section_context_from_state(state, task_plan)
        # The same selection the Engineer and its self-review were given, from the same
        # function on the same two inputs. A reviewer holding the design can say "this uses a
        # raw hex where the design names a token"; a reviewer holding nothing has been saying
        # "looks reasonable" for the entire life of this platform.
        design_snapshot = design_snapshot_context_from_state(state, task_plan)
        contract_reference_names = contract_section_names_from_state(state)
        review_technical_prd = _review_technical_prd(technical_prd, review_scope)
        # Withholding an unmeetable criterion is right; doing it silently is not. Whatever was
        # set aside is named in the artifact so the author of the requirement can see that a
        # criterion they wrote was not part of the judgement.
        withheld_criteria = [
            *unverifiable_criteria(
                [
                    *technical_prd.functional_requirements,
                    *technical_prd.non_functional_requirements,
                ]
            ),
            *[
                UnverifiableCriterion(
                    requirement_id=str(review_scope.get("workstream_id") or "workstream scope"),
                    criterion=item,
                )
                for item in review_scope.get("acceptance_criteria_not_reviewable", [])
            ],
        ]
        prior_findings = _prior_findings(state, review_scope)
        prior_reviews = _prior_reviews(state)
        await self._raise_if_cancelled()
        validation_results = await _run_validations(
            self._validation_tool, timeout_seconds=self._validation_timeout_seconds
        )
        await self._raise_if_cancelled()
        workspace_evidence = await _workspace_change_evidence(
            state,
            process_runner=self._process_runner,
            cancellation_token=self._cancellation_token or MockCancellationToken(),
            # The workstream's declared scope, so the evidence can say which of the change's
            # paths it never named (87- Part C).
            expected_files_or_areas=review_scope.get("expected_files_or_areas") or (),
            evidence_budget=self._evidence_budget,
            # The most recent prior review's own record of which seam files its evidence
            # budget already omitted, so a second consecutive omission of the *same* file can
            # be told from the first: a budget limitation this platform cannot resolve by
            # retrying identically is not the same thing as a fresh one worth stating plainly.
            previously_omitted_seam_paths=_previously_omitted_seam_paths(prior_reviews),
        )
        # What the platform will actually execute here. The review has always been given the
        # results and never the plan, so it could see that a command produced no result and
        # not that no such command exists -- and a demand to run one is a demand the engineer
        # has no way to meet. Rendered as the bound on what any finding may ask for.
        planned_commands = _planned_validation_commands(self._validation_tool)
        # Fetched before the prompt is rendered, because the prompt has to say how many
        # pictures accompany it: an unexplained image is a worse input than none.
        design_pictures = await self._design_pictures(design_snapshot)
        instructions = self._prompt_loader.render(
            "reviewer/v1.jinja2",
            workflow_id=state["workflow_id"],
            technical_prd=json.dumps(review_technical_prd, indent=2, sort_keys=True),
            review_scope=json.dumps(review_scope, indent=2, sort_keys=True),
            contract_sections=contract_section_context_json(contract_sections),
            design_snapshot=design_snapshot_context_json(design_snapshot),
            design_pictures=len(design_pictures),
            code_completion=artifact_json(code_completion),
            # Which round this is decides the blocking authority: the first review sets the
            # whole bar; a remediation review may only hold the work to that bar and to
            # defects in what changed since. Told to the model here, enforced by the
            # platform's own verdict handling either way.
            review_round="remediation" if prior_reviews else "first",
            prior_blocking_findings=json.dumps(
                _prior_blocking_finding_summaries(prior_reviews), indent=2, sort_keys=True
            ),
            # The seam clauses render only when the diff actually touches a channel package:
            # the common change pays no prompt space for a seam it does not have.
            channel_packages=workspace_evidence["channel_packages"],
            # And the configuration clauses only when there is a configuration module to
            # name, or a channel this checkout offered none for. Both are evidence: the
            # first says which file to compare the change's requests against, the second
            # says the comparison could not be set up -- neither demands a verdict.
            seam_configuration_paths=workspace_evidence["seam_configuration_paths"],
            seam_configuration_unlocated=workspace_evidence["seam_configuration_unlocated"],
            # Seam files a repeat omission already proved unresolvable (AB-Feature-176): named
            # so the model does not raise its own finding over what this platform already
            # declared it cannot show and will not block on.
            seam_unresolvable=[item["path"] for item in workspace_evidence["seam_unresolvable"]],
            # 87- Part C's question, rendered only when there is something to ask about. A
            # change entirely inside what the plan declared produces the prompt it always did.
            undeclared_change_paths=workspace_evidence["undeclared_change_paths"],
            widely_changed_paths=workspace_evidence["widely_changed_paths"],
            # Rendered only when a person has overruled a demand for this repository, so
            # the common review pays nothing for a verdict nobody has given.
            overruled_demands=self._overruled_demands,
            planned_validation_commands=[
                {
                    "command": " ".join(item.command),
                    "validation_type": item.validation_type,
                    "required": item.required,
                }
                for item in planned_commands
            ],
        )
        review_input = {
            "technical_prd": review_technical_prd,
            "review_scope": review_scope,
            "task_plan": task_plan.model_dump(mode="json"),
            "code_completion": code_completion.model_dump(mode="json"),
            "validation": [validation_summary(result) for result in validation_results],
            "prior_repository_findings": prior_findings,
            "workspace_change_evidence": workspace_evidence["model_input"],
        }
        input_text = json.dumps(review_input, indent=2, sort_keys=True)

        def build(model_response: LLMResponse) -> ReviewArtifact:
            payload = parse_model_json(model_response.output_text)
            _validate_requirement_checks(payload, review_scope)
            # Whatever the model wrote, before the deterministic injectors add to it. The
            # difference is exactly what this platform put on the review itself, and a
            # platform-injected finding blocks whether or not it names a requirement: it is
            # the record of a command that failed or of source the review could not read.
            model_finding_ids = _finding_ids(payload)
            _apply_completion_findings(payload, code_completion, review_scope)
            _apply_validation_findings(
                payload,
                validation_results,
                test_validation_required=bool(review_scope.get("test_requirements")),
                # What the attempt wrote, and what the plan is able to run over it. An empty
                # suite only says something about the change when the planned runner could
                # have seen the change's tests -- see `_plan_could_collect`.
                changed_test_paths=_completion_test_paths(code_completion),
                planned_commands=planned_commands,
            )
            _apply_workspace_evidence_findings(
                payload, workspace_evidence, evidence_budget=self._evidence_budget
            )
            _apply_unreviewable_criteria_findings(payload, withheld_criteria)
            # Recorded unconditionally, not only under the bounded scope that first needed
            # it: the 73- demotion reads this to refuse to overrule a measurement, and a
            # verdict is not gated by the traceability experiment. Metadata only -- with the
            # bounded flag off it changes no verdict and blocks nothing.
            scope_metadata: dict[str, Any] = {
                "deterministic_finding_ids": _platform_finding_ids(
                    payload,
                    model_supplied=model_finding_ids,
                    workspace_evidence=workspace_evidence,
                )
            }
            # Constructed inside, so schema rejection is caught by the repair below rather
            # than escaping past it: the failing field is only discovered here.
            return create_artifact(
                ReviewArtifact,
                workflow_id=state["workflow_id"],
                artifact_id=attempt_artifact_id(ARTIFACT_FILENAMES["review"], state["retry_count"]),
                producer="reviewer",
                payload=payload,
                metadata={
                    "source_artifact_ids": [
                        technical_prd.artifact_id,
                        task_plan.artifact_id,
                        code_completion.artifact_id,
                    ],
                    "model": model_response.model,
                    # The role, not only the model: two roles may legitimately be configured
                    # with the same model, and what makes this review independent is that it
                    # was performed as a review and never wrote code.
                    "model_role": (
                        self._model_role.value if self._model_role is not None else None
                    ),
                    "response_id": model_response.response_id,
                    "prompt_template": "reviewer/v1.jinja2",
                    "validation": [validation_summary(item) for item in validation_results],
                    "review_scope": review_scope,
                    "review_evidence_version": "workspace-source-v1",
                    "reviewed_file_paths": workspace_evidence["reviewed_file_paths"],
                    "paths_the_attempt_did_not_declare": workspace_evidence[
                        "paths_the_attempt_did_not_declare"
                    ],
                    "reviewed_content_fingerprint": workspace_evidence[
                        "reviewed_content_fingerprint"
                    ],
                    "manual_review_required": workspace_evidence["manual_review_required"],
                    "retryable": not workspace_evidence["manual_review_required"],
                    # The seam facts, durable on the artifact: which channel packages the
                    # change imports, which co-importing modules entered evidence, which were
                    # omitted for the budget, and each changed test that mocks a channel
                    # package and asserts on that mock. The notes survive here so "was the
                    # seam verified" is answerable from the review record alone.
                    "channel_seam_packages": workspace_evidence["channel_packages"],
                    "channel_seam_paths": workspace_evidence["seam_context_paths"],
                    # Which seam file configures the channel, and which channel this checkout
                    # offered no visible configuration for. Durable so "did the review read
                    # the client the call goes through" is answerable from the record: 218's
                    # frontend review recorded `channel_seam_packages: []` and approved a
                    # change whose two blockers were both in `src/config/axios.js`.
                    "channel_seam_configuration_paths": workspace_evidence[
                        "seam_configuration_paths"
                    ],
                    "channel_seam_configuration_not_located": workspace_evidence[
                        "seam_configuration_unlocated"
                    ],
                    "seam_context_omitted": workspace_evidence["seam_omitted"],
                    # Plain paths, read back by `_previously_omitted_seam_paths` on the
                    # *next* attempt so a second consecutive omission of the same file can be
                    # told from a first one. Includes this round's summarized and unresolvable
                    # paths as well as its whole omissions: once a path has been flagged any of
                    # the three ways, it stays flagged on every later attempt rather than
                    # oscillating back to a whole, blocking omission the one attempt it happens
                    # to fit the ordinary check again.
                    "seam_evidence_budget_omitted_paths": [
                        item["path"] for item in workspace_evidence["seam_omitted"]
                    ]
                    + [item["path"] for item in workspace_evidence["seam_summarized"]]
                    + [item["path"] for item in workspace_evidence["seam_unresolvable"]],
                    # What the reviewer was asked about that the workstream never declared
                    # (87- Part C). Durable because the question is only worth asking if
                    # somebody can later check whether it was answered: 218's frontend
                    # silenced two pre-existing error paths and the review's
                    # `architecture_assessment` mentions none of them.
                    "undeclared_change_paths": workspace_evidence["undeclared_change_paths"],
                    "widely_changed_paths": workspace_evidence["widely_changed_paths"],
                    "wide_change_region_threshold": _WIDE_CHANGE_HUNK_COUNT,
                    "seam_mock_notes": workspace_evidence["seam_mock_notes"],
                    # Which changed production files the attempt's own cheap gate could not
                    # read. An `inconclusive` self-review is a withheld assurance, not a
                    # finding: it composes no diagnostic and demands nothing, and it is here
                    # so a repository review knows which of these files no cheaper gate has
                    # actually looked at. 218's frontend was approved with two production
                    # blockers by a self-review that recorded `AllApps.js` as truncated and
                    # answered `clean`.
                    "self_review_inconclusive_paths": _self_review_inconclusive_paths(
                        code_completion
                    ),
                    "acceptance_criteria_not_reviewed": [
                        {"requirement_id": item.requirement_id, "criterion": item.criterion}
                        for item in withheld_criteria
                    ],
                    "contract_sections_quoted": _quoted_section_summary(contract_sections),
                    # Recorded, not acted on. See `_unresolved_contract_references`.
                    "unresolved_contract_references": _unresolved_contract_references(
                        payload, contract_reference_names
                    ),
                    **scope_metadata,
                    **_validation_context_metadata(self._validation_tool),
                    **execution_metadata(
                        agent_type="Code reviewer",
                        provider=model_response.provider,
                        model=model_response.model,
                        reasoning_effort=model_response.reasoning_effort,
                        model_role=(
                            self._model_role.value
                            if self._model_role is not None
                            else model_response.model_role
                        ),
                        model_variable=model_response.model_variable,
                        routing_reason=model_response.routing_reason,
                    ),
                },
            )

        async def repair_contradiction(
            review: ReviewArtifact, *, contradiction: str
        ) -> ReviewArtifact:
            """Ask once more when the review's verdict and its findings disagree.

            The same one-shot repair the schema uses, for a defect the schema cannot see: a
            review that declines the work while every finding it raises is `low`, which is
            reserved for wording the plan should correct and explicitly not a change to
            make. Nothing is promoted to blocking either way -- if the review comes back
            saying the same thing, its findings stay advisory and the first review stands,
            because asking twice is the bound and a third opinion is a person's.
            """
            repaired = await self._llm_client.respond(
                instructions=(
                    f"{instructions}\n\n"
                    "Your previous review contradicted itself. Here is your previous "
                    f"response:\n{artifact_json(review)}\n\n"
                    f"Here is the contradiction:\n{contradiction}\n\n"
                    "Return the complete review again. Either raise the severity of the "
                    "findings that genuinely require a change and name the scoped "
                    "requirement, contract section, or implementation expectation each one "
                    "derives from, or approve the work and leave the remarks as they are. "
                    "Do not invent a finding you did not already have."
                ),
                input_text=input_text,
            )
            await self._raise_if_cancelled()
            try:
                return build(repaired)
            except (AgentArtifactError, ValidationError):
                return review

        async def produce_primary() -> ReviewArtifact:
            response = await self._llm_client.respond(
                instructions=instructions, input_text=input_text, images=design_pictures
            )
            await self._raise_if_cancelled()
            try:
                built = build(response)
            except (AgentArtifactError, ValidationError) as error:
                # The planner already repairs a rejected response instead of failing the run.
                # Without the same bounded attempt here a single malformed field -- one finding
                # given a category outside the allowed set -- discarded an implementation that
                # had already passed formatting, linting, tests, and the build.
                repair = await self._llm_client.respond(
                    instructions=(
                        f"{instructions}\n\n"
                        "Your previous response was rejected by deterministic validation. "
                        "Here is your previous response:\n"
                        f"{response.output_text}\n\n"
                        f"Here is exactly what was wrong with it:\n{error}\n\n"
                        "Return the complete review again with only the fields named above "
                        "changed. Every other field must be byte-identical to your previous "
                        "response, and every enumerated field must use one of its listed values."
                    ),
                    input_text=json.dumps(
                        {
                            "rejected_response": response.output_text,
                            "workspace_change_evidence": workspace_evidence["model_input"],
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                )
                await self._raise_if_cancelled()
                # A schema repair has already cost this review its one extra call. Whatever
                # it returned stands, contradiction or not.
                return build(repair)
            if not self._bounded_review_scope:
                return built
            contradiction = _rejection_contradiction(built)
            if contradiction is None:
                return built
            return await repair_contradiction(built, contradiction=contradiction)

        async def produce_review() -> ReviewArtifact:
            # The second opinion judges the review the loop would otherwise publish, so it
            # runs after every primary path -- schema repair and contradiction repair
            # included -- and inside the same journaled operation, so replay reconstructs
            # the merged artifact without a model call.
            primary = await produce_primary()
            return await self._second_opinion_pass(
                primary, state=state, workspace_evidence=workspace_evidence
            )

        if self._operation_executor is None:
            review = await produce_review()
        else:
            signature = _review_input_signature(review_input)

            async def journaled_review() -> tuple[ReviewArtifact, OperationResult]:
                produced = await produce_review()
                return produced, OperationResult(
                    external_reference=str(produced.metadata.get("response_id") or ""),
                    payload={"review_artifact": produced.model_dump(mode="json")},
                )

            journaled = await self._operation_executor.run(
                operation_type=ExternalOperationType.RUN_REVIEWER,
                logical_step="repository_review",
                safe_input={
                    "review_input_sha256": signature,
                    "reviewed_content_fingerprint": workspace_evidence[
                        "reviewed_content_fingerprint"
                    ],
                },
                idempotency_input={"review_input_sha256": signature},
                action=journaled_review,
                attempt_metadata={"provider": "repository_reviewer"},
                max_attempts=3,
            )
            if journaled.reused:
                payload = journaled.operation.result_payload or {}
                persisted = payload.get("review_artifact")
                if not isinstance(persisted, dict):
                    msg = "completed reviewer operation has no valid review artifact"
                    raise AgentArtifactError(msg)
                # Strict artifacts accept RFC 3339 timestamps from JSON, not from a Python
                # dict whose string has already lost the JSON decoding context.
                review = ReviewArtifact.model_validate_json(json.dumps(persisted))
            else:
                review = cast(ReviewArtifact, journaled.value)
        return artifact_update("reviewer", [review])

    async def _raise_if_cancelled(self) -> None:
        """Prevent a cancelled live child from accepting review or starting later GitHub work."""
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()

    async def _second_opinion_pass(
        self,
        review: ReviewArtifact,
        *,
        state: AgentState,
        workspace_evidence: dict[str, Any],
    ) -> ReviewArtifact:
        """Run the gated second-opinion review (75-) on a review the loop would publish.

        The coder, its self-review, and the primary reviewer share context selection, so
        they share blind spots (AB-Feature-206). This pass gives an approved channel-touching
        or large change to a reviewer that was never shown the requirement, the author's
        report, or the retry history, and that chooses its own evidence: it sees the change
        and the checkout's inventory, asks for the files it wants, and reads them. Merged
        findings enter the existing loop as this review's own -- same blocking division,
        same fingerprint ledger, same 73- demotion -- and every review records the
        ``second_opinion`` measurement whether or not the pass ran.
        """
        gate: dict[str, Any] = {
            "channel_packages": list(workspace_evidence["channel_packages"]),
            "changed_file_count": workspace_evidence["changed_file_count"],
            "changed_source_characters": workspace_evidence["changed_source_characters"],
        }
        # A malfunctioning pass degrades to this record rather than failing the review: a
        # second-opinion malfunction must never outrank the primary's judgment or consume
        # a workstream attempt (77-, item 34).
        #
        # The gate was `approved` only, which left the one reviewer that picks its own evidence
        # unable to ever look at a rejection. That is backwards for one shape of blocker: a
        # blocking finding that names no file and reports a validation state is not checkable
        # against a diff, so nothing downstream can tell a real one from a false one, and the
        # retry loop will spend every attempt it has on it. AB-Feature-216 was stopped by two
        # such findings that were both false, and AB-Feature-225 by one the plan made
        # impossible -- neither was ever read a second time.
        #
        # It runs to MEASURE, never to overturn: the pass can only add findings (the merge is
        # additive and the verdict argument below only ever escalates), and a rejection stays a
        # rejection. What it adds is a record of whether an independent reviewer, choosing its
        # own evidence, could substantiate the blocker -- which is what a person arbitrating
        # the stop needs and had no way to get.
        unfiled_blockers = _unfiled_blocking_finding_ids(review)
        if review.verdict != "approved" and not unfiled_blockers:
            return _with_second_opinion(review, {"ran": False, "reason": "verdict", "gate": gate})
        if review.verdict != "approved":
            gate["reason"] = "unfiled_blocker"
            gate["unfiled_blocking_finding_ids"] = unfiled_blockers
        elif gate["channel_packages"]:
            gate["reason"] = "channel"
        elif (
            gate["changed_file_count"] >= _SECOND_OPINION_MIN_FILES
            or gate["changed_source_characters"] >= _SECOND_OPINION_MIN_CHARACTERS
        ):
            gate["reason"] = "size"
        else:
            return _with_second_opinion(review, {"ran": False, "reason": "gate", "gate": gate})

        workspace = resolve_workspace_root(state["workspace_descriptor"].root_path)
        file_tools = WorkspaceFileTools(workspace)
        inventory_text, inventory, inventory_truncated = _second_opinion_inventory(workspace)
        instructions = self._prompt_loader.render(
            _SECOND_OPINION_TEMPLATE,
            workflow_id=state["workflow_id"],
            max_requested_files=_SECOND_OPINION_MAX_REQUESTED_FILES,
            overruled_demands=self._overruled_demands,
        )
        response_ids: list[str] = []

        async def respond_validated(input_payload: dict[str, Any]) -> _SecondOpinionResponse:
            input_text = json.dumps(input_payload, indent=2, sort_keys=True)
            response = await self._llm_client.respond(
                instructions=instructions, input_text=input_text
            )
            await self._raise_if_cancelled()
            response_ids.append(response.response_id)
            try:
                return _second_opinion_payload(parse_model_json(response.output_text))
            except (AgentArtifactError, ValidationError) as error:
                repair = await self._llm_client.respond(
                    instructions=(
                        f"{instructions}\n\n"
                        "Your previous response was rejected by deterministic validation. "
                        "Here is your previous response:\n"
                        f"{response.output_text}\n\n"
                        f"Here is exactly what was wrong with it:\n{error}\n\n"
                        "Return the complete response again with only the fields named "
                        "above changed."
                    ),
                    input_text=input_text,
                )
                await self._raise_if_cancelled()
                response_ids.append(repair.response_id)
                # Still malformed after its one repair: the caller below degrades the pass
                # to "did not run" rather than failing the review. Degrading to *approval*
                # would report coverage the pass did not have; degrading to `ran: false`
                # reports exactly the coverage it had -- none.
                return _second_opinion_payload(parse_model_json(repair.output_text))

        withheld: list[dict[str, str]] = []
        shown: list[str] = []
        try:
            requested, findings, summary = await respond_validated(
                {
                    "turn": "ask",
                    "change_evidence": workspace_evidence["model_input"],
                    "repository_inventory": {
                        "paths": inventory_text,
                        "truncated": inventory_truncated,
                    },
                }
            )
            if requested:
                entries, withheld, shown = _second_opinion_entries(file_tools, requested, inventory)
                # Exactly one read round: whatever the second turn asks for is not fetched.
                _, findings, summary = await respond_validated(
                    {"turn": "read", "requested_files": entries}
                )
        except (AgentArtifactError, ValidationError):
            # A response still malformed after its one repair. A second-opinion malfunction
            # must never outrank the primary's judgment or consume a workstream attempt
            # (AB-Feature-208's frontend attempt 1 died as `platform_defect` to exactly
            # this), so the review completes on the primary verdict alone and the record
            # says honestly that the pass did not run. Cancellation and provider faults are
            # deliberately not caught: they are operation faults with their own retry
            # classification, not malformed responses.
            return _with_second_opinion(
                review,
                {
                    "ran": False,
                    "reason": "malformed_response",
                    "gate": gate,
                    "response_ids": response_ids,
                },
            )
        merged, duplicates, merged_ids = _merge_second_opinion(review, findings)
        blocking = sum(1 for item in merged if item.severity in {"critical", "high"})
        record = {
            "ran": True,
            "gate": gate,
            "requested_paths": list(requested),
            "shown_paths": shown,
            "withheld": withheld,
            "inventory_truncated": inventory_truncated,
            # Every merged finding survived the primary reviewer's approval: this is the
            # escaped-defect count the live matrix reads.
            "escaped_blocking": blocking,
            "escaped_advisory": len(merged) - blocking,
            "duplicate_count": duplicates,
            "escaped_finding_ids": merged_ids,
            "response_ids": response_ids,
            "prompt_template": _SECOND_OPINION_TEMPLATE,
            "summary": summary,
            # Which of the primary's unfiled blockers the independent read did not reproduce.
            # A measurement and nothing more -- the verdict below only ever escalates -- but it
            # is the only signal on this platform that distinguishes a blocker two reviewers
            # agree on from one that only the reviewer sharing the coder's evidence could see.
            # A person arbitrating a recurring-demand stop is deciding exactly that question.
            "unsubstantiated_blocking_finding_ids": [
                finding_id
                for finding_id in _unfiled_blocking_finding_ids(review)
                if finding_id not in {item.finding_id for item in findings}
            ],
        }
        return _with_second_opinion(
            review,
            record,
            merged_findings=merged,
            # Forced anyway by the approved-cannot-carry-blocking validator; set here so the
            # demotion is the platform's decision, not a validation accident.
            verdict="changes_requested" if blocking else None,
        )


def _unfiled_blocking_finding_ids(review: ReviewArtifact) -> list[str]:
    """The blocking findings that name no file and report a validation state.

    The shape nothing downstream can check. A blocking finding that cites a file can be judged
    against that file by the next attempt, by the retry planner and by a person reading the
    diff. One that cites none and says a command's result is missing or structural can only be
    judged against the validation plan -- and the demand it makes may be one no attempt is able
    to satisfy, which is indistinguishable from a real defect until somebody looks twice.

    Deliberately narrow. A file-less `requirement` finding is the ordinary way to say coverage
    is missing and is not this shape; the category is part of the test.
    """
    return [
        item.finding_id
        for item in review.findings
        if item.severity in {"critical", "high"}
        and item.file_path is None
        and item.finding_category == "validation_failure"
    ]


_SecondOpinionResponse = tuple[tuple[str, ...], list[ReviewFinding], str]


def _second_opinion_payload(payload: dict[str, Any]) -> _SecondOpinionResponse:
    """Validate one second-opinion response: what it asks to read, and what it found.

    Missing keys default to empty -- the response is a message, not an artifact -- but each
    finding is validated strictly: a malformed finding is a malformed response, and the
    caller's one-shot repair (then degradation to a pass that records it did not run)
    handles it. Everything raised here is `AgentArtifactError`, including a finding the
    schema rejects -- the pydantic `ValidationError` that escaped this path killed
    AB-Feature-208's frontend attempt as a platform defect.
    """
    requested = payload.get("requested_paths") or []
    if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
        msg = "second-opinion `requested_paths` must be a list of relative paths"
        raise AgentArtifactError(msg)
    raw_findings = payload.get("findings") or []
    if not isinstance(raw_findings, list):
        msg = "second-opinion `findings` must be a list"
        raise AgentArtifactError(msg)
    allowed = set(ReviewFinding.model_fields)
    findings: list[ReviewFinding] = []
    for index, item in enumerate(raw_findings):
        if not isinstance(item, dict):
            msg = "each second-opinion finding must be a JSON object"
            raise AgentArtifactError(msg)
        try:
            findings.append(
                ReviewFinding(**{key: value for key, value in item.items() if key in allowed})
            )
        except ValidationError as error:
            # Re-raised as the malformed-response type so the caller's one-shot repair
            # fires, naming the finding and its fields rather than dumping a pydantic
            # trace. Field names are schema-owned and safe to record; the rejected values
            # are not repeated, per `safe_error_diagnostics`' rule. AB-Feature-208's
            # frontend died to exactly this ValidationError escaping as a platform defect.
            fields = sorted({str(entry["loc"][0]) for entry in error.errors() if entry.get("loc")})
            named = ", ".join(fields) if fields else "unknown field"
            msg = (
                f"second-opinion finding {index} was rejected by schema validation "
                f"({named}); correct that finding and return the complete response"
            )
            raise AgentArtifactError(msg) from error
    summary = payload.get("summary")
    return (
        tuple(dict.fromkeys(requested)),
        findings,
        summary if isinstance(summary, str) else "",
    )


def _second_opinion_inventory(workspace: Path) -> tuple[str, set[str], bool]:
    """List the checkout for the ask turn, bounded, with the cut declared not silent."""
    paths = sorted(path.as_posix() for path in scan_directory(workspace).files)
    text = "\n".join(paths)
    truncated = len(text) > _SECOND_OPINION_INVENTORY_MAX_CHARACTERS
    if truncated:
        head = text[:_SECOND_OPINION_INVENTORY_MAX_CHARACTERS]
        cut = head.rfind("\n")
        text = head[:cut] if cut > 0 else head
    # The withheld-reason check below uses the full set either way: a path the listing cut
    # for size is still a real, readable file, not "outside_inventory".
    return text, set(paths), truncated


def _second_opinion_entries(
    file_tools: WorkspaceFileTools,
    requested: Sequence[str],
    inventory: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Read what the second opinion asked for; a withheld file is declared, never silent.

    The predicates are the primary evidence's own -- `_sensitive_review_path`,
    `carries_key_material`, `redact_source_credentials` -- one rule per obligation, no
    copies. Content is worktree bytes, never diff text, per the non-leak rule.
    """
    entries: list[dict[str, Any]] = []
    withheld: list[dict[str, str]] = []
    shown: list[str] = []
    remaining = _SECOND_OPINION_MAX_CHARACTERS

    def refuse(path: str, reason: str) -> None:
        withheld.append({"path": path, "reason": reason})
        entries.append({"path": path, "content": "[CONTENT WITHHELD]", "withheld_reason": reason})

    for index, path in enumerate(requested):
        if index >= _SECOND_OPINION_MAX_REQUESTED_FILES:
            refuse(path, "request_ceiling")
            continue
        if path not in inventory:
            refuse(path, "outside_inventory")
            continue
        if _sensitive_review_path(Path(path)):
            refuse(path, "sensitive_path")
            continue
        try:
            source = file_tools.read_file(path)
        except (UnicodeDecodeError, ValueError, OSError):
            source = None
        if source is None:
            refuse(path, "unreadable")
            continue
        if carries_key_material(source):
            refuse(path, "key_material")
            continue
        if remaining <= 0:
            refuse(path, "evidence_budget")
            continue
        redacted = redact_source_credentials(source)
        content = redacted[: min(_REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS, remaining)]
        remaining -= len(content)
        entries.append(
            {
                "path": path,
                "content": content,
                "truncated": len(content) < len(redacted),
                "redacted": redacted != source,
            }
        )
        shown.append(path)
    return entries, withheld, shown


def _merge_second_opinion(
    review: ReviewArtifact, findings: Sequence[ReviewFinding]
) -> tuple[list[ReviewFinding], int, list[str]]:
    """Keep the findings the primary approval let escape; drop restatements of what it saw.

    Identity is `fingerprint_for_text` over the description -- the ledger's and the 73-
    demotion's own identity, not a new one -- so "escaped" means exactly "would enter the
    loop as a defect the approval missed".
    """
    primary_fingerprints = {fingerprint_for_text(item.description) for item in review.findings}
    taken_ids = {item.finding_id for item in review.findings}
    merged: list[ReviewFinding] = []
    duplicates = 0
    seen: set[str] = set()
    for finding in findings:
        fingerprint = fingerprint_for_text(finding.description)
        if fingerprint in primary_fingerprints or fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        finding_id = f"{_SECOND_OPINION_FINDING_PREFIX}{finding.finding_id}"
        while finding_id in taken_ids:
            finding_id = f"{finding_id}-2"
        taken_ids.add(finding_id)
        merged.append(finding.model_copy(update={"finding_id": finding_id}))
    return merged, duplicates, [item.finding_id for item in merged]


def _with_second_opinion(
    review: ReviewArtifact,
    record: dict[str, Any],
    *,
    merged_findings: Sequence[ReviewFinding] = (),
    verdict: str | None = None,
) -> ReviewArtifact:
    """Rebuild the frozen artifact once, with the measurement and any merged findings.

    Reconstruction goes through `create_artifact`, so the schema validators run again: a
    merged blocking finding cannot ride on an `approved` verdict even if the demotion above
    were dropped by mistake.
    """
    payload = review.model_dump(mode="json")
    for field in (
        "schema_version",
        "workflow_id",
        "artifact_id",
        "producer",
        "timestamp",
        "metadata",
        "validation_status",
        "artifact_type",
    ):
        payload.pop(field, None)
    if merged_findings:
        payload["findings"] = [
            *payload["findings"],
            *[item.model_dump(mode="json") for item in merged_findings],
        ]
    if verdict is not None:
        payload["verdict"] = verdict
    return create_artifact(
        ReviewArtifact,
        workflow_id=review.workflow_id,
        artifact_id=review.artifact_id,
        producer=review.producer,
        payload=payload,
        metadata={**review.metadata, "second_opinion": record},
    )


async def reviewer_node(
    state: AgentState,
    *,
    prompt_loader: PromptLoader,
    llm_client: LLMClient,
    validation_tool: ValidationTool,
    validation_timeout_seconds: float = 120.0,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """Run a review node with explicitly injected model and validation boundaries."""
    return await ReviewerAgent(
        prompt_loader=prompt_loader,
        llm_client=llm_client,
        validation_tool=validation_tool,
        validation_timeout_seconds=validation_timeout_seconds,
        cancellation_token=cancellation_token,
    ).run(state)


async def _run_validation(
    tool: ValidationTool, method_name: str, *, timeout_seconds: float
) -> ValidationResult:
    """Prefer the new interruptible method while preserving existing synchronous test tools."""
    async_method = getattr(tool, f"{method_name}_async", None)
    result = (
        async_method(timeout_seconds=timeout_seconds)
        if async_method is not None
        else getattr(tool, method_name)(timeout_seconds=timeout_seconds)
    )
    resolved = await result if inspect.isawaitable(result) else result
    if not isinstance(resolved, ValidationResult):
        msg = "validation tool returned an invalid result"
        raise AgentArtifactError(msg)
    if resolved.cancelled:
        raise CancellationRequested("validation subprocess was cancelled")
    return resolved


async def _run_validations(
    tool: ValidationTool, *, timeout_seconds: float
) -> tuple[ValidationResult, ...]:
    """Prefer repository-native validation while keeping legacy injected tools compatible."""
    async_method = getattr(tool, "run_validations_async", None)
    sync_method = getattr(tool, "run_validations", None)
    if async_method is not None:
        result = async_method(timeout_seconds=timeout_seconds)
    elif sync_method is not None:
        result = sync_method(timeout_seconds=timeout_seconds)
    else:
        return (
            await _run_validation(tool, "run_ruff", timeout_seconds=timeout_seconds),
            await _run_validation(tool, "run_pytest", timeout_seconds=timeout_seconds),
        )
    resolved = await result if inspect.isawaitable(result) else result
    if not isinstance(resolved, tuple) or not resolved:
        msg = "validation tool must return a non-empty tuple of validation results"
        raise AgentArtifactError(msg)
    if not all(isinstance(item, ValidationResult) for item in resolved):
        msg = "validation tool returned an invalid validation result"
        raise AgentArtifactError(msg)
    if any(item.cancelled for item in resolved):
        raise CancellationRequested("validation subprocess was cancelled")
    return resolved


def _offerable_as_discovered(workspace: Path, relative_path: str) -> bool:
    """Whether a path Git found but the attempt never declared may enter review evidence.

    A discovered path is strictly-additional evidence: before this existed the review saw none
    of them, so anything that cannot be shown cleanly is dropped rather than reported. That is
    the whole rule, and it is load-bearing -- every limitation this function avoids is one that
    forbids approval, so a permissive filter here turns "the review reads more" into a new way
    to fail a change that was fine. The real-repository tier caught exactly that: a Python
    checkout with no ignore rule for `__pycache__` offered five `.pyc` files, none of which
    decode, and the resulting `REVIEW_EVIDENCE_INCOMPLETE` blocked an approved change.

    So: never a directory Git cannot be expected to have an opinion about (the seven names in
    `DEFAULT_IGNORED_DIRECTORIES` are never source in any repository), never a never-publish
    path, never a symlink, and never bytes that are not UTF-8 text. A declared path failing any
    of these still raises its limitation, exactly as before -- the attempt claimed that one.
    """
    parts = PurePosixPath(relative_path).parts
    if any(part in DEFAULT_IGNORED_DIRECTORIES for part in parts):
        return False
    if _sensitive_review_path(Path(relative_path)):
        return False
    try:
        candidate = resolve_workspace_path(workspace, relative_path)
    except (GitSafetyError, ValueError):
        return False
    raw = workspace / relative_path
    if raw.is_symlink() or not candidate.is_file():
        return False
    try:
        candidate.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError, OSError):
        return False
    return True


async def _workspace_change_evidence(
    state: AgentState,
    *,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
    expected_files_or_areas: Sequence[str] = (),
    evidence_budget: ReviewEvidenceBudget = _DEFAULT_REVIEW_EVIDENCE_BUDGET,
    previously_omitted_seam_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Capture bounded current source for every uncommitted attempt in this workspace.

    ``expected_files_or_areas`` is the workstream's own declared scope, read here for one
    purpose (87- Part C): saying which of the change's paths it never named. Defaulting to
    empty means a workflow with no declared scope asks nothing, which is the honest reading
    -- nothing was declared, so nothing is outside it.
    """
    changes: dict[str, Any] = {}
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
    declared_paths = list(changes)

    workspace = resolve_workspace_root(state["workspace_descriptor"].root_path)
    file_tools = WorkspaceFileTools(workspace)
    model_files: list[dict[str, Any]] = []
    oversized_paths: list[str] = []
    orphan_lockfile_paths: list[str] = []
    unshown_changed_regions: list[dict[str, Any]] = []
    # The change's own readable sources, for the channel-seam scan below, and the notes
    # naming each changed test that mocks a channel package and asserts on that mock. The
    # notes are detection over source this loop already read; they inform the review and are
    # never a limitation -- 47-D's law, asserted by test.
    changed_sources: list[tuple[str, str]] = []
    mock_notes: list[str] = []
    # Resolved on first use, not here: only a file too large to be shown whole needs it, which
    # is the rare case, and a review of ordinary files must not pay two subprocesses for a
    # question it never asks.
    baseline = LineageBaseRevision(
        workspace=workspace,
        # A revision child's lineage leaves its base branch, not the repository default;
        # `base_branch` is None everywhere else, which keeps this the default branch.
        default_branch=(
            state["workspace_descriptor"].base_branch
            or state["workspace_descriptor"].default_branch
        ),
        process_runner=process_runner,
        cancellation_token=cancellation_token,
    )
    # What the branch holds, beside what the attempt said it wrote. Until now this function's
    # whole universe was `file_changes` -- the coding executor's self-report -- so a file the
    # executor wrote and did not declare was invisible to every reviewer on this platform, and
    # the review then reasoned from its absence. AB-Feature-225's attempt 0 wrote a second
    # browser suite covering the pending-submit behaviour, never declared it, and was given a
    # finding saying that coverage was missing; the file was on disk the whole time and
    # `git ls-files --others` lists it. AB-Feature-216 lost an edited authentication middleware
    # the same way.
    #
    # Git is asked for the same reason the category evidence asks it (66-): it is the only
    # source that does not depend on a model choosing to mention something. `None` means the
    # question was not answered -- an unresolved branch point or a truncated read -- and the
    # declared set then stands alone, exactly as before.
    discovered = await branch_change_evidence(
        workspace=workspace,
        baseline=baseline,
        process_runner=process_runner,
        cancellation_token=cancellation_token,
        # Written into every child workspace by the platform on every attempt, so it is
        # permanently untracked here and belongs to no attempt. The prompt already forbids
        # reporting it; not offering it is the same rule one layer down.
        excluded_paths=(CONTRACT_PROJECTION_PATH,),
    )
    undeclared_by_executor: list[str] = []
    if discovered is not None:
        modified = set(discovered.modified_paths)
        for path in discovered.paths:
            if path in changes or not _offerable_as_discovered(workspace, path):
                continue
            undeclared_by_executor.append(path)
            changes[path] = FileChange(
                path=path,
                change_type="modified" if path in modified else "added",
                # Said plainly, because this text reaches the model beside the source: the
                # attempt did not claim this file, and the review is being shown it anyway.
                description="Found on the branch; this attempt's report did not declare it.",
            )
    # Declared first, so the bounded window never drops a claimed file to make room for a
    # discovered one, and the truncation stays deterministic across attempts.
    reviewed_paths = [*declared_paths, *undeclared_by_executor]
    # Keyed on the declared set alone, deliberately. This limitation forbids approval, and a
    # discovered path is a file the review could not see at all until now: letting one raise it
    # would turn a strict improvement in what gets read into a new way to block a change that
    # was fine. A discovered path pushed past the window is simply not shown, which is exactly
    # where it stood before.
    limitations: list[str] = (
        ["file_limit_exceeded"] if len(declared_paths) > _REVIEW_EVIDENCE_MAX_FILES else []
    )
    characters_used = 0
    for relative_path in reviewed_paths[:_REVIEW_EVIDENCE_MAX_FILES]:
        change = changes[relative_path]
        safe_path = resolve_workspace_path(workspace, relative_path)
        raw_path = workspace / relative_path
        entry: dict[str, Any] = {
            "path": relative_path,
            "change_type": change.change_type,
        }
        if _sensitive_review_path(Path(relative_path)):
            entry.update({"content": "[CONTENT WITHHELD]", "redacted": True})
            limitations.append("sensitive_path")
            model_files.append(entry)
            continue
        if raw_path.is_symlink():
            raw_target = str(raw_path.readlink())
            target = redact_source_credentials(raw_target)
            entry.update(
                {
                    "content": target,
                    "content_kind": "symlink_target",
                    "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                    "redacted": target != raw_target,
                }
            )
            if entry["redacted"]:
                limitations.append("redacted_content")
            model_files.append(entry)
            continue
        lockfile = await _generated_lockfile_evidence(
            workspace=workspace,
            safe_path=safe_path,
            relative_path=relative_path,
            reviewed_paths=reviewed_paths,
            baseline=baseline,
            process_runner=process_runner,
            cancellation_token=cancellation_token,
        )
        if lockfile is not None:
            entry.update(lockfile)
            characters_used += len(str(lockfile["content"]))
            if lockfile["manifest_changed"] is False:
                limitations.append(_ORPHAN_LOCKFILE_LIMITATION)
                orphan_lockfile_paths.append(relative_path)
            model_files.append(entry)
            continue
        source: str | None = None
        try:
            if safe_path.exists():
                source = file_tools.read_file(relative_path)
        except (UnicodeDecodeError, ValueError, OSError):
            source = None
        if source is not None:
            changed_sources.append((relative_path, source))
            if classify_file_change(relative_path) == "test":
                for note in seam_mock_notes(relative_path, source):
                    entry.setdefault("seam_mock_notes", []).append(note)
                    mock_notes.append(note)
        # The filename cleared `_sensitive_review_path`, which cannot see a service-account
        # key behind an ordinary name. Same policy as a credential-shaped path, because the
        # change touched this file and the reviewer still needs to know it exists.
        if source is not None and carries_key_material(source):
            entry.update({"content": "[CONTENT WITHHELD]", "redacted": True})
            limitations.append("sensitive_content")
            model_files.append(entry)
            continue
        remaining = max(0, evidence_budget.max_characters - characters_used)
        evidence = await _bounded_file_evidence(
            workspace=workspace,
            relative_path=relative_path,
            source=source,
            remaining_characters=remaining,
            per_file_max_characters=evidence_budget.per_file_max_characters,
            baseline=baseline,
            process_runner=process_runner,
            cancellation_token=cancellation_token,
        )
        hidden = evidence.pop("_hidden_changed_regions", ())
        entry.update(evidence)
        characters_used += len(str(evidence.get("content") or ""))
        if evidence.get("redacted") is True and not await _redaction_is_preexisting(
            workspace=workspace,
            relative_path=relative_path,
            source=source,
            process_runner=process_runner,
            cancellation_token=cancellation_token,
        ):
            limitations.append("redacted_content")
        if evidence.get("truncated") is True:
            limitations.append("truncated_content")
            oversized_paths.append(relative_path)
            # What the review could not see, in the terms the change itself is written in.
            # "Split this file into modules" was the old recommendation and it was unmeetable
            # for the case that produced it -- four changed lines in a shared constants file;
            # naming the hunk is what makes the (now rare) refusal actionable.
            unshown_changed_regions.extend(
                {"path": relative_path, "start_line": start, "end_line": end}
                for start, end in hidden
            )
        if evidence.get("content_kind") == "unavailable":
            limitations.append("unavailable_content")
        model_files.append(entry)

    # The seam the change stands on: every repository module that imports a channel package
    # the changed sources import, shown under the unchanged-source banner. Changed files kept
    # first claim on the budget above; seam context takes what remains, floored at
    # `seam_reserved_characters` so a change whose own diff is merely large cannot starve seam
    # evidence to zero before a single co-importer is even considered -- indistinguishable,
    # from the reviewer's side, from a change that never touched a shared client at all. A
    # seam file that still does not fit the (possibly reserved) remainder is omitted whole and
    # declared -- never trimmed (the 51-A rule).
    seam = _channel_seam_evidence(
        workspace=workspace,
        file_tools=file_tools,
        changed_sources=changed_sources,
        reviewed_paths=reviewed_paths,
        remaining_characters=max(
            evidence_budget.seam_reserved_characters,
            evidence_budget.max_characters - characters_used,
        ),
        per_file_max_characters=evidence_budget.per_file_max_characters,
        previously_omitted_seam_paths=previously_omitted_seam_paths,
    )
    model_files.extend(seam["entries"])
    if seam["budget_omitted"]:
        limitations.append(_SEAM_CONTEXT_LIMITATION)

    # 87- Part C: which of this change's paths the workstream never declared, and which
    # declared file carries so many separate changed regions that some of them are probably
    # not what the workstream asked for. A question for the reviewer, computed from the plan
    # and the diff, and it decides nothing on its own.
    scope_question = await _out_of_scope_change_question(
        workspace=workspace,
        reviewed_paths=reviewed_paths[:_REVIEW_EVIDENCE_MAX_FILES],
        expected_files_or_areas=expected_files_or_areas,
        baseline=baseline,
        process_runner=process_runner,
        cancellation_token=cancellation_token,
    )

    # Over the DECLARED paths, not the widened set. This fingerprint and the path list beside
    # it are a commit-time contract, not a record of what was read: `require_reviewed_workspace_
    # match` recomputes both from the same cumulative `file_changes` and refuses the publication
    # if they differ, and `ApprovedChangePublisher` stages exactly that list. Widening either
    # would put a file nobody declared into an approved commit -- `git ls-files --others` lists
    # whatever an attempt left in the worktree -- which is a worse failure than the one this
    # change fixes. What the review READS is widened; what the platform attests to and commits
    # is not.
    try:
        content_fingerprint = (
            reviewed_content_fingerprint(workspace, declared_paths)
            if declared_paths
            else hashlib.sha256(b"").hexdigest()
        )
    except GitSafetyError:
        # An implementation that touched a never-publish path still needs a durable rejected
        # review rather than an exception/retry loop. Bind only its path inventory here; its
        # withheld bytes are deliberately excluded from both model input and durable metadata.
        limitations.append("sensitive_path")
        content_fingerprint = hashlib.sha256(
            json.dumps(declared_paths, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    unique_limitations = list(dict.fromkeys(limitations))
    manual_review_required = bool(
        (_HUMAN_DECISION_LIMITATIONS | _UNREADABLE_LIMITATIONS) & set(unique_limitations)
    )
    # Recorded separately from `manual_review_required` because they lead somewhere else: the
    # attempt is told what exceeded the budget so the next one can come in under it.
    evidence_budget_exceeded = bool(_EVIDENCE_BUDGET_LIMITATIONS & set(unique_limitations))
    return {
        "model_input": {
            "evidence_version": "workspace-source-v1",
            "files": model_files,
            "limitations": unique_limitations,
            "manual_review_required": manual_review_required,
            # The seam summary beside the entries it explains: which channel packages the
            # change imports, which co-importing modules were shown, and which were omitted
            # with the reason each was. `truncated` says the co-importer set was cut at its
            # ceiling, which is itself something a reviewer should see.
            "channel_seam": {
                "packages": seam["packages"],
                "files": seam["seam_paths"],
                # 87- A2: which of those files is the one the repository *configured* -- the
                # base URL, the credentials, the interceptors -- and which channels this
                # checkout offered no visible configuration for. 218's two blockers were
                # both in a configuration module the review never read.
                "configuration_files": seam["configuration_paths"],
                "configuration_not_located": seam["configuration_unlocated"],
                "omitted": seam["omitted"],
                # Also present in `omitted` (reason `evidence_budget_unresolvable`), named
                # separately so the prompt can point at it directly: a repeat omission this
                # platform already knows cannot resolve by asking again, declared per 51-A
                # but not a finding -- see the matching prompt clause.
                "unresolvable": seam["unresolvable_omitted"],
                "co_importer_count": seam["co_importer_count"],
                "truncated": seam["truncated"],
            },
            # Each changed test that mocks a channel package and asserts on that same mock.
            # Facts for the reviewer to weigh, exactly like the lockfile host scan: no
            # limitation is raised and no verdict is forced over one.
            "seam_mock_notes": mock_notes,
            # The paths this workstream never declared, and the declared files that moved in
            # many separate places. Same citizenship as the notes above: a question, never a
            # finding, and empty for a change that stayed inside what the plan named.
            "out_of_scope_changes": scope_question,
        },
        # The declared set, which is the publication contract: the publisher recomputes this
        # list and this fingerprint from the same `file_changes` and stages exactly it.
        "reviewed_file_paths": declared_paths,
        # And what Git found beside it, which is the read set. Kept as its own key rather than
        # folded into the list above so widening what a review READS can never widen what an
        # approved commit CONTAINS. Recorded because a review shown a file nobody declared
        # should be answerable about it afterwards, and because a lineage that keeps producing
        # them is a coding executor under-reporting its own work -- not visible anywhere else.
        "paths_the_attempt_did_not_declare": [
            path
            for path in undeclared_by_executor
            if path in set(reviewed_paths[:_REVIEW_EVIDENCE_MAX_FILES])
        ],
        "reviewed_content_fingerprint": content_fingerprint,
        "limitations": unique_limitations,
        "manual_review_required": manual_review_required,
        "evidence_budget_exceeded": evidence_budget_exceeded,
        # The second-opinion gate's size measures (75-): how many files the change names and
        # how much readable changed source it carries. Readable only, deliberately -- a
        # withheld or lockfile entry is not source a review of either opinion can weigh.
        "changed_file_count": len(reviewed_paths),
        "changed_source_characters": sum(len(source) for _, source in changed_sources),
        # The channel packages this change's own sources import, for conditioning the
        # reviewer's seam clauses, and the seam evidence facts the artifact records.
        "channel_packages": seam["packages"],
        "seam_context_paths": seam["seam_paths"],
        "seam_configuration_paths": seam["configuration_paths"],
        "seam_configuration_unlocated": seam["configuration_unlocated"],
        "seam_omitted": seam["budget_omitted"],
        # Paths this round escalated to a summary rather than a whole omission (see
        # `_channel_seam_evidence`'s escalation). Persisted alongside `seam_omitted` so the
        # *next* attempt's `_previously_omitted_seam_paths` treats a summarized path the same
        # as a still-omitted one: once a file has been flagged this way, it stays on the
        # summary path on every later attempt too, rather than oscillating back to a whole
        # omission the one attempt it happens to fit the ordinary check again.
        "seam_summarized": seam["summarized"],
        # The other half of the same escalation, for a path with nothing to summarize
        # instead (AB-Feature-176). Persisted the same way, for the same reason: once flagged
        # unresolvable it must stay recognised as "previously omitted" on every later attempt,
        # not revert to a fresh, blocking omission.
        "seam_unresolvable": seam["unresolvable_omitted"],
        "seam_mock_notes": mock_notes,
        # 87- Part C's two lists, for the prompt clause and for the durable record: a review
        # that was asked about an undeclared path should be answerable about it afterwards.
        "undeclared_change_paths": scope_question["undeclared_paths"],
        "widely_changed_paths": scope_question["widely_changed_paths"],
        # Named, and in the order they were read, so the finding can say which file the review
        # could not see whole rather than asking the engineer to guess which of forty-eight.
        "oversized_paths": oversized_paths,
        # Generated lockfiles that moved with no manifest to explain them. Named for the same
        # reason as the oversized paths: the finding has to say which file it means, and this
        # one is asking the next attempt about a specific dependency change.
        "orphan_lockfile_paths": orphan_lockfile_paths,
        # The changed line ranges that did not fit, where the evidence knows them. Empty with
        # a non-empty `oversized_paths` means the excerpt could not be computed at all -- no
        # repository, or a diff Git would not answer -- which is a different sentence.
        "unshown_changed_regions": unshown_changed_regions,
    }


async def _redaction_is_preexisting(
    *,
    workspace: Path,
    relative_path: str,
    source: str | None,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> bool:
    """Report whether every credential-shaped literal here was already committed.

    Redaction keeps secrets out of the reviewer's context, but a literal the change never
    touched says nothing about the change. -067's backend was refused approval -- terminally,
    with the reviewer itself recording that "the visible feature source and required
    validations appear compliant" -- because it added two keys to a ``config`` module that
    already carried nine credential-shaped literals, every one of them committed long before.
    Any repository holding its settings in one such file could never have a change approved.

    Introducing new credential-shaped material still blocks: the comparison is against the
    committed version of the same file, so an added secret raises the count and is reported.
    """
    if source is None or not (workspace / ".git").exists():
        return False
    result = await process_runner.run(
        ("git", "show", f"HEAD:{relative_path}"),
        workspace,
        30.0,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not result.succeeded or result.output_truncated:
        return False
    return _redaction_count(source) <= _redaction_count(result.stdout)


def _redaction_count(value: str) -> int:
    """Count the credential literals redaction would withhold from the reviewer."""
    return redact_source_credentials(value).count("[REDACTED]")


async def _out_of_scope_change_question(
    *,
    workspace: Path,
    reviewed_paths: Sequence[str],
    expected_files_or_areas: Sequence[str],
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> dict[str, Any]:
    """Name the changed behaviour the workstream never asked for, as a question.

    AB-Feature-218's approved frontend diff added `.catch(() => null)` to `refetchData()` and
    `refetchPublishersData()` -- two effects that predate the feature and have nothing to do
    with bulk CSV upload -- so a failed app-list load now produces no error and no signal. It
    also loosened a query-match branch and added a pagination guard. All five edits came from
    the in-attempt source-repair pass satisfying the model's own new test mocks, and the
    review's `architecture_assessment` does not mention one of them.

    Two facts are computed, both cheap and both from things this platform already holds:

    * **Paths outside the declared scope.** Membership is exact: a declared token is a file
      when the path equals it and an area when the path sits under it. No fuzzy matching --
      `assigned_file_conformance` makes the same judgement the same way, and an area this
      check cannot interpret means the check says less, never something wrong. Each path
      carries its `change_kind`, because a test file the plan did not name is ordinary suite
      organisation and a production file it did not name is the 218 question.
    * **Declared files with many separate changed regions.** The spec asked for "hunks that
      touch lines the change did not otherwise need", and which lines a change *needed* is
      not something any deterministic inspection can know. What is knowable is how many
      disjoint regions of one file moved, and a declared file that moved in
      ``_WIDE_CHANGE_HUNK_COUNT`` or more places is worth asking about by region. One
      `git diff --unified=0` for the whole change, so the question costs one subprocess and
      only when a scope was declared at all.

    Returns empty lists when nothing qualifies, and the prompt then renders nothing: a review
    whose every path was declared is byte-identical to one taken before this existed.
    """
    areas = _declared_scope_areas(expected_files_or_areas)
    if not areas or not reviewed_paths:
        return {"undeclared_paths": [], "widely_changed_paths": []}
    undeclared = [
        {"path": path, "change_kind": classify_file_change(path)}
        for path in reviewed_paths
        if not _within_declared_scope(path, areas)
    ]
    declared = [path for path in reviewed_paths if _within_declared_scope(path, areas)]
    counts = await _changed_region_counts(
        workspace=workspace,
        paths=declared,
        baseline=baseline,
        process_runner=process_runner,
        cancellation_token=cancellation_token,
    )
    widely_changed = [
        {"path": path, "changed_region_count": counts[path]}
        for path in declared
        if counts.get(path, 0) >= _WIDE_CHANGE_HUNK_COUNT
    ]
    return {
        "undeclared_paths": undeclared[:_MAX_OUT_OF_SCOPE_PATHS],
        "widely_changed_paths": widely_changed[:_MAX_OUT_OF_SCOPE_PATHS],
    }


def _declared_scope_areas(expected_files_or_areas: Sequence[str]) -> tuple[str, ...]:
    """Return the workstream's declared tokens, normalised, in the plan's own order."""
    areas: list[str] = []
    for candidate in expected_files_or_areas:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip().lstrip("./").rstrip("/")
        if normalized:
            areas.append(normalized)
    return tuple(dict.fromkeys(areas))


def _within_declared_scope(path: str, areas: Sequence[str]) -> bool:
    """Say whether the plan named this path, as a file or as a directory holding it."""
    return any(path == area or path.startswith(f"{area}/") for area in areas)


async def _changed_region_counts(
    *,
    workspace: Path,
    paths: Sequence[str],
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> dict[str, int]:
    """Return how many disjoint regions of each path this change moved.

    One `git diff --unified=0` over the whole set rather than one per file, and unanswerable
    is empty: a diff Git would not produce, or one past the output cap, asks nothing rather
    than reporting a count nobody measured. A file the diff never mentions counts zero, which
    is right for an added file -- every line of it is the change, and "which regions did you
    not need" is not a question about a file that did not exist.
    """
    if not paths or not (workspace / ".git").exists():
        return {}
    result = await process_runner.run(
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--unified=0",
            await baseline.resolve(),
            "--",
            *paths,
        ),
        workspace,
        30.0,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not result.succeeded or result.output_truncated:
        return {}
    counts: dict[str, int] = {}
    current: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current = line[len("+++ b/") :]
            continue
        if line.startswith("+++ "):
            # `/dev/null`: the file was deleted, and a deletion has no regions to justify.
            current = None
            continue
        if current is not None and _DIFF_HUNK_HEADER.match(line):
            counts[current] = counts.get(current, 0) + 1
    return counts


def _channel_seam_evidence(
    *,
    workspace: Path,
    file_tools: WorkspaceFileTools,
    changed_sources: Sequence[tuple[str, str]],
    reviewed_paths: Sequence[str],
    remaining_characters: int,
    per_file_max_characters: int = _REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS,
    # Seam files the immediately preceding review's own evidence budget already omitted
    # whole. A first omission is the ordinary case the budget law below still governs; a
    # second consecutive one for the same path cannot be resolved by omitting it a third
    # time, because nothing about the file or the budget changes between attempts on that
    # account alone -- see the escalation carved out of that law just below.
    previously_omitted_seam_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Quote the unchanged modules that co-import a channel package this change imports.

    AB-Feature-206's blocker in one sentence: the change imported a channel package bare
    instead of the instance one unchanged module configures, every command gate passed, and
    the review artifact contained zero mention of the configuration module -- it was
    unchanged, so it never entered evidence, and a reviewer cannot flag a mismatch with a
    file it never read. `_workspace_change_evidence` iterates the change set and nothing
    else; this is the one deliberate exception, and it is derived from the diff (the
    co-importer set of the packages the change imports), never from reconnaissance prose,
    which is allowed to fail and may cite a file for a different reason.

    Every seam file shown is genuinely unchanged: a path in this lineage's change set is in
    ``reviewed_paths`` -- committed attempts included, per 51-A -- and is excluded here, so
    the unchanged-source banner is a fact and not a guess.

    Budget law (51-A): a seam file that does not fit what remains of the review budget, or
    exceeds the per-file bound, is omitted whole and declared with the seam limitation --
    never trimmed, because a reviewer shown half a configuration would attest to a seam it
    half-read. Sensitive, unreadable and key-material seam files are omitted and named without
    raising any limitation: the file is unchanged, so whatever it carries was committed long
    before this change, which is exactly the judgement `_redaction_is_preexisting` already
    encodes for changed files.

    Escalation, carved out of 51-A rather than replacing it: a file this same reason has
    already omitted once cannot be waited out -- a repository whose real shared-client module
    is simply larger than the per-file bound stays larger than it on every future attempt, and
    51-A applied to that case forever is what burned two live workstreams through their whole
    review-cycle budget on one unmoving sentence (`service.util.js` at 29,259 bytes,
    `CreateBatch.js` at 48,517 bytes -- both comfortably over the 24,000-character default).
    On the *second* consecutive omission for the same path, quote `configuration_calls`
    instead of the whole file: not a trim of the seam file's own content (51-A still forbids
    that), but the narrower, already-computed fact of *which lines make this module the seam
    at all*. A co-importer with no located configuration call has nothing this substitutes
    for and keeps the ordinary whole-omission path -- summarizing nothing would misrepresent
    the file as reviewed.
    """

    def read(path: str) -> str | None:
        try:
            return file_tools.read_file(path)
        except (FileNotFoundError, UnicodeDecodeError, ValueError, OSError):
            return None

    try:
        repository_paths = [item.as_posix() for item in scan_directory(workspace).files]
    except (NotADirectoryError, OSError):
        repository_paths = []
    scan = channel_seam_scan(changed_sources, repository_paths, read, exclude=set(reviewed_paths))
    entries: list[dict[str, Any]] = []
    seam_paths: list[str] = []
    configuration_paths: list[str] = []
    omitted: list[dict[str, Any]] = []
    summarized: list[dict[str, Any]] = []

    def omit(path: str, packages: Sequence[str], reason: str) -> None:
        omitted.append({"path": path, "channel_packages": list(packages), "reason": reason})

    budget = remaining_characters
    for item in scan.co_importers:
        if _sensitive_review_path(Path(item.path)):
            omit(item.path, item.channel_packages, "sensitive_path")
            continue
        source = read(item.path)
        if source is None:
            omit(item.path, item.channel_packages, "unreadable")
            continue
        if carries_key_material(source):
            omit(item.path, item.channel_packages, "key_material")
            continue
        redacted = redact_source_credentials(source)
        content = f"{_unchanged_source_banner(item.path)}\n{redacted}"
        if len(content) > per_file_max_characters or len(content) > budget:
            if item.path in previously_omitted_seam_paths:
                if item.configuration_calls:
                    summary_content = _channel_seam_summary_content(item)
                    if len(summary_content) <= budget:
                        entries.append(
                            {
                                "path": item.path,
                                "content_kind": "channel_seam_summary",
                                "channel_packages": list(item.channel_packages),
                                "configuration_calls": list(item.configuration_calls),
                                "content": summary_content,
                                "sha256": hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
                                "redacted": False,
                            }
                        )
                        seam_paths.append(item.path)
                        configuration_paths.append(item.path)
                        summarized.append(
                            {"path": item.path, "channel_packages": list(item.channel_packages)}
                        )
                        budget -= len(summary_content)
                        continue
                else:
                    # No configuration_calls to substitute -- a plain, oversized consumer of
                    # the channel wrapper (CreateBatch.js/Batches.js, AB-Feature-176), not its
                    # configurer. Its size cannot change between attempts (it is unchanged by
                    # construction), so blocking on it forever is 51-A's "waited out" failure
                    # the escalation exists to stop -- just the half configuration_calls never
                    # covers. Declared, not silently dropped (see `unresolvable_omitted`
                    # below), but no longer forced as a blocking finding every attempt.
                    omit(item.path, item.channel_packages, "evidence_budget_unresolvable")
                    continue
            omit(item.path, item.channel_packages, "evidence_budget")
            continue
        entries.append(
            {
                "path": item.path,
                "content_kind": "channel_seam_context",
                "channel_packages": list(item.channel_packages),
                # What makes this module the seam rather than one more call site, quoted as
                # it appears in the file. Empty for a co-importer that only uses the channel.
                "configuration_calls": list(item.configuration_calls),
                "content": content,
                "sha256": hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
                "redacted": redacted != source,
            }
        )
        seam_paths.append(item.path)
        if item.configuration_calls:
            configuration_paths.append(item.path)
        budget -= len(content)
    return {
        "packages": list(scan.packages),
        "entries": entries,
        "seam_paths": seam_paths,
        # Which of the shown seam files configure the channel, and which channels no
        # configuration module could be found for. The second is A2's recorded limitation and
        # it is deliberately not a `limitations` entry: "nothing in this checkout visibly
        # configures this channel" is a fact for the reviewer to weigh, not a platform bound
        # a retry can come in under, and 87-'s safety rule is that Parts A, B and C add
        # evidence and never a finding, a severity or a gate.
        "configuration_paths": configuration_paths,
        "configuration_unlocated": list(scan.configuration_unlocated),
        "omitted": omitted,
        "co_importer_count": scan.co_importer_count,
        "truncated": scan.truncated,
        # Only the budget omissions raise `seam_context_omitted`: they are the platform's own
        # bound, and the finding tells the next attempt which file and package it means.
        "budget_omitted": [item for item in omitted if item["reason"] == "evidence_budget"],
        # A repeat omission with nothing to summarize instead (AB-Feature-176: no
        # `configuration_calls`, so the branch above has nothing narrower to quote). Named
        # separately from `budget_omitted` for the same reason `summarized` is: this platform
        # already knows retrying will not change it, so it must not keep raising
        # `seam_context_omitted` as if it were a fresh, resolvable limitation.
        "unresolvable_omitted": [
            item for item in omitted if item["reason"] == "evidence_budget_unresolvable"
        ],
        # Paths shown as a summary this round because a prior attempt already omitted them
        # whole for the same reason. Deliberately not in `omitted`/`budget_omitted`: a
        # summarized path was shown something, so it does not raise `seam_context_omitted`
        # again -- the whole point of the escalation is to stop repeating a limitation the
        # next attempt cannot do anything about.
        "summarized": summarized,
    }


async def _generated_lockfile_evidence(
    *,
    workspace: Path,
    safe_path: Path,
    relative_path: str,
    reviewed_paths: Sequence[str],
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> dict[str, Any] | None:
    """Describe a lockfile too large to quote, or return ``None`` to leave it to the ladder.

    Run 189's frontend edited `package.json`, the platform's own dependency install churned
    `package-lock.json` exactly as designed, and the workstream terminated: 1,290,320 bytes is
    past the read cap, an unreadable file joins `_UNREADABLE_LIMITATIONS`, and that class is a
    human decision no retry may take. One machine-generated file, refused with the severity
    reserved for a credential. Any dependency-affecting change on a Node repository died the
    same way, deterministically.

    A lockfile is none of the things the unreadable class exists for. It is the package
    manager's own output, regenerated from the manifest, and this platform already models that
    relationship: `dependency_sync` runs a resolving install precisely when a manifest changed,
    "the one moment the lockfile is expected to move with it". The evidence layer had simply
    never learned what the dependency layer knows, so it is asked here rather than told.

    Two conditions, both narrow and both deterministic, and ``None`` -- today's behaviour,
    unchanged -- for everything else:

    * **Size, and size alone.** The cap is measured by `stat` *before* any read is attempted,
      never by reading an exception's message. Any other reason the bytes cannot be had is a
      real read failure, and the streaming pass below is what proves the difference: a
      permissions failure raises there, this returns ``None``, and the file takes the
      unreadable path exactly as it does today.
    * **A lockfile this platform knows.** Membership comes from `dependency_sync`'s own
      registry via `generated_lockfile_manifest`, not from a filename list restated here. A
      manager added there is understood here with no further change, and nothing about any
      particular repository enters the reviewer.

    What comes back is deterministic and computed without loading the file: its size, a
    streaming hash of its exact bytes, its numstat against the same lineage baseline the rest
    of this evidence is measured against, whether the manifest it is generated from moved in
    this same change set, and -- from `scan_resolved_hosts` -- which hosts it now resolves
    packages from that the committed lockfile at that same baseline did not. No content is
    quoted, so a registry URL carrying a token cannot reach the model through this path --
    which is also why the whole-file key-material check above has nothing to examine and is
    not asked to, and why the host scan reports authorities with any userinfo stripped.
    """
    manifest_path = generated_lockfile_manifest(relative_path)
    if manifest_path is None:
        return None
    try:
        size_bytes = safe_path.stat().st_size
    except OSError:
        return None
    if size_bytes <= DEFAULT_MAX_FILE_BYTES:
        # Small enough to read: the ordinary ladder applies, and this carve-out activates
        # only where the read cap would otherwise have ended the workstream.
        return None
    try:
        sha256 = streamed_file_digest(safe_path).hex()
    except OSError:
        # The read fails for something other than size after all. Not this carve-out.
        return None
    insertions, deletions = await _lockfile_numstat(
        workspace=workspace,
        safe_path=safe_path,
        relative_path=relative_path,
        baseline=baseline,
        process_runner=process_runner,
        cancellation_token=cancellation_token,
    )
    # The one thing size, line count and digest cannot say: where the packages now come from.
    # A tampered lockfile changes its hash exactly as a legitimate install does, so the facts
    # above are equally consistent with either. This states the difference as a fact and
    # nothing more -- no limitation is raised and no verdict is forced, because whether a new
    # registry host is a migration or an attack is a judgement, and judgement belongs to the
    # reviewer and to the person reading the review.
    host_scan = await scan_resolved_hosts(
        workspace=workspace,
        safe_path=safe_path,
        relative_path=relative_path,
        # ``None`` unless the lineage base is a real branch point. 51-A's rule, and it bites
        # harder here than it does for a numstat: an earlier approved attempt's commits are
        # already in `HEAD`, so a host that attempt introduced would read as one the
        # repository always used.
        baseline_revision=(await baseline.resolve()) if baseline.is_branch_point else None,
        process_runner=process_runner,
        cancellation_token=cancellation_token,
    )
    manifest_changed = manifest_path in set(reviewed_paths)
    counted = (
        f"{insertions:,} lines added and {deletions:,} removed"
        if insertions is not None and deletions is not None
        else "a line count this repository could not report"
    )
    content = "\n".join(
        (
            f"===== {relative_path}: generated lockfile, contents not shown =====",
            f"This file is dependency-manager output regenerated from {manifest_path}, not "
            "authored source. It is machine-written and was not read.",
            f"size: {size_bytes:,} bytes, past the "
            f"{DEFAULT_MAX_FILE_BYTES:,}-byte per-file read limit",
            f"sha256: {sha256}",
            f"change against the revision this change branched from: {counted}",
            f"{manifest_path} changed in this same change set: "
            f"{'yes' if manifest_changed else 'no'}",
            host_scan.sentence(),
        )
    )
    return {
        "content": content,
        "content_kind": "generated_lockfile",
        # The hash of the file's exact bytes, not of a redacted rendering: nothing was
        # rendered. It is the same value `reviewed_content_fingerprint` folds in for this
        # path, which is what lets the publication binding still speak for a file the review
        # deliberately did not quote.
        "sha256": sha256,
        "redacted": False,
        # Emphatically not truncated. Nothing was cut: the entry is complete evidence of a
        # different kind, and `truncated: true` would append a budget limitation that forces
        # `changes_requested` on a lockfile that did exactly what the manifest asked it to.
        "truncated": False,
        "size_bytes": size_bytes,
        "manifest_path": manifest_path,
        "manifest_changed": manifest_changed,
        "insertions": insertions,
        "deletions": deletions,
        # Facts about where the bytes came from, in the same shape as the facts above them.
        # `new_hosts` is `null` rather than `[]` wherever the comparison could not be made,
        # and the entry says which reason: an empty list is a claim, and the reassuring answer
        # is the one it would be worst to guess at.
        "resolved_host_scan": host_scan.as_evidence(),
    }


async def _lockfile_numstat(
    *,
    workspace: Path,
    safe_path: Path,
    relative_path: str,
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> tuple[int | None, int | None]:
    """Return how many lines this lockfile gained and lost, or ``None`` where Git cannot say.

    ``None`` rather than zero wherever the answer would be a guess, for the reason
    `LineageBaseRevision.is_branch_point` exists: an empty diff against the ``HEAD`` fallback
    means only that nothing is left *uncommitted*, which is a different fact from nothing
    having changed, and reporting it as "0 lines" would state the guess as a measurement. The
    entry's load-bearing claim -- whether the manifest moved with it -- comes from the reviewed
    path set and is unaffected either way.

    An untracked lockfile is the install having created one where the checkout had none, which
    `git diff` reports as nothing at all. Its every line is added, counted by streaming the
    file rather than by loading it.
    """
    if not (workspace / ".git").exists():
        return (None, None)
    tracked = await process_runner.run(
        ("git", "ls-files", "--error-unmatch", "--", relative_path),
        workspace,
        30.0,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not tracked.succeeded:
        try:
            return (_streamed_line_count(safe_path), 0)
        except OSError:
            return (None, None)
    result = await process_runner.run(
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--numstat",
            await baseline.resolve(),
            "--",
            relative_path,
        ),
        workspace,
        30.0,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not result.succeeded or result.output_truncated:
        return (None, None)
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        # `-\t-\t<path>` is Git calling the file binary, which is not a line count.
        if len(fields) >= 3 and fields[0].isdigit() and fields[1].isdigit():
            return (int(fields[0]), int(fields[1]))
    return (0, 0) if baseline.is_branch_point else (None, None)


def _streamed_line_count(path: Path) -> int:
    """Count a file's lines without holding it in memory, as its digest is taken."""
    newlines = 0
    trailing = b""
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            newlines += chunk.count(b"\n")
            trailing = chunk[-1:]
    # A final line with no newline after it is still a line; an empty file has none.
    return newlines + (1 if trailing not in {b"", b"\n"} else 0)


async def _bounded_file_evidence(
    *,
    workspace: Path,
    relative_path: str,
    source: str | None,
    remaining_characters: int,
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
    per_file_max_characters: int = _REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS,
) -> dict[str, Any]:
    """Prefer a complete source file, then this change's own hunks, else mark it truncated.

    The middle rung is what makes a large file reviewable. It used to be the *whole* unified
    diff against ``HEAD``, and that failed the case it mattered most for twice over: it is
    all-or-nothing, so a diff one byte over the budget contributed nothing, and it measures
    against ``HEAD``, which by the time a later attempt is reviewed already contains work an
    earlier approved attempt committed. Run 186's frontend died on exactly that -- four
    changed lines in a 31 KB constants file, already committed, so ``git diff HEAD`` was empty,
    the file was head-truncated, and the deterministic gate refused the truncation three times
    running. No retry could fix it: the file stays large.

    So the file's changed regions are quoted instead, hunk by hunk, read from the worktree and
    measured against the baseline the review is actually attesting. Selection scales where a
    cap does not: a late file in a large change still gets its hunks, because hunks are small.
    """
    budget = min(per_file_max_characters, remaining_characters)
    redacted_source = redact_source_credentials(source) if source is not None else None
    source_redacted = source is not None and redacted_source != source
    evidence_sha256 = (
        hashlib.sha256(redacted_source.encode("utf-8")).hexdigest()
        if redacted_source is not None
        else None
    )
    if redacted_source is not None and len(redacted_source) <= budget:
        return {
            "content": redacted_source,
            "content_kind": "utf8_source",
            "sha256": evidence_sha256,
            "redacted": source_redacted,
            "truncated": False,
        }

    if redacted_source is not None and budget > 0:
        excerpt = await _changed_region_evidence(
            workspace=workspace,
            relative_path=relative_path,
            redacted_source=redacted_source,
            source_redacted=source_redacted,
            evidence_sha256=evidence_sha256,
            budget=budget,
            baseline=baseline,
            process_runner=process_runner,
            cancellation_token=cancellation_token,
        )
        if excerpt is not None:
            return excerpt

    if redacted_source is None:
        return {
            "content": "[CONTENT UNAVAILABLE]",
            "content_kind": "unavailable",
            "sha256": None,
            "redacted": False,
            "truncated": True,
        }
    return {
        "content": redacted_source[:budget],
        "content_kind": "utf8_source",
        "sha256": evidence_sha256,
        "redacted": source_redacted,
        "truncated": True,
    }


def _unchanged_source_banner(relative_path: str) -> str:
    """The one sentence that marks source shown to the reviewer as non-diff content.

    One function rather than two spellings: the changed-region evidence uses it for a file
    whose worktree bytes are the baseline's bytes, and the channel-seam evidence uses it for
    a co-importing module the change never touched. A reviewer taught the vocabulary once
    must find the same bytes in both places.
    """
    return f"===== {relative_path}: unchanged from the revision this change branched from ====="


def _channel_seam_summary_content(item: SeamCoImporter) -> str:
    """Quote only what makes this co-importer the seam, once its whole file has proved too big.

    Never a trim of arbitrary source -- 51-A still forbids that, because a reviewer shown half
    a configuration would attest to a seam it half-read. This quotes a different, narrower
    fact instead: `configuration_calls` is `channel_seam_scan`'s own record of the exact lines
    that establish this module as the seam, computed the same way whether the file fits the
    budget or not. Reusing it here is what keeps the escalation honest about what it is -- a
    named, partial fact about an oversized file, not a disguised whole-file read.
    """
    banner = (
        f"===== {item.path}: unchanged from the revision this change branched from -- "
        "this file exceeded the review's evidence budget on a prior attempt, so only its "
        "channel-configuring call sites are quoted below, not the whole file ====="
    )
    calls = "\n".join(item.configuration_calls)
    return f"{banner}\n{calls}"


async def _changed_region_evidence(
    *,
    workspace: Path,
    relative_path: str,
    redacted_source: str,
    source_redacted: bool,
    evidence_sha256: str | None,
    budget: int,
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> dict[str, Any] | None:
    """Quote this file's changed regions from the worktree, or return ``None`` if it cannot.

    ``None`` means Git could not say where the change is -- no repository, a diff it refused,
    output past the runner's cap -- and the caller then falls back to head truncation exactly
    as before. That distinction is the whole safety argument for this path: an excerpt is
    allowed to claim the change is fully visible only when Git affirmatively said where the
    change is, and every one of those places was then quoted in full.

    ``sha256`` stays the hash of the full redacted source, as it was for a whole file: the hash
    names the file the excerpt came from, not the excerpt. Redaction needs no second
    application because the regions are sliced out of source already redacted whole, so
    excerpting cannot widen what a reviewer sees by one byte.
    """
    spans = await _changed_line_spans(
        workspace=workspace,
        relative_path=relative_path,
        line_count=len(redacted_source.splitlines()),
        baseline=baseline,
        process_runner=process_runner,
        cancellation_token=cancellation_token,
    )
    if spans is None:
        return None
    lines = redacted_source.splitlines()

    def heading(start: int, end: int) -> str:
        return f"===== {relative_path} lines {start}-{end} of {len(lines)} =====\n"

    shown, hidden = _fit_changed_regions(
        lines,
        spans,
        budget=budget,
        # Every region's heading is charged at the widest it could be -- the line numbers at
        # full width -- so the per-file bound holds on what is actually sent and not only on
        # the source inside it.
        region_overhead=len(heading(len(lines), len(lines))) + 1,
    )
    quoted = [heading(start, end) + "\n".join(lines[start - 1 : end]) for start, end in shown]
    if not spans:
        # Git said these bytes are the baseline's bytes. The path is in the change set because
        # an attempt in this lineage wrote it, and it now holds what was already there, so
        # there is nothing about it for the review to judge -- said in words, because an empty
        # content field reads as missing evidence rather than as an answer.
        quoted.append(_unchanged_source_banner(relative_path))
    if hidden:
        # Named even where nothing could be quoted, so the entry says which lines the review
        # is missing rather than arriving as an empty string the model has to interpret.
        quoted.append(
            f"===== {relative_path}: changed lines not shown, over the evidence budget =====\n"
            + "\n".join(f"lines {start}-{end}" for start, end in hidden)
        )
    content = "\n".join(quoted)
    return {
        "content": content,
        "content_kind": "changed_regions",
        "sha256": evidence_sha256,
        "redacted": source_redacted,
        # The new meaning of the word: not "the file was cut" but "changed content is not
        # fully visible". Every hunk quoted whole with the untouched remainder elided is not
        # a truncation of the change, and the deterministic gate must not fire on it.
        "truncated": bool(hidden),
        "changed_regions": [
            {"excerpt_lines": {"start": start, "end": end}} for start, end in shown
        ],
        "hunks_omitted": len(hidden),
        "region_order": _CHANGED_REGION_ORDER,
        "source_lines": len(lines),
        "_hidden_changed_regions": hidden,
    }


async def _changed_line_spans(
    *,
    workspace: Path,
    relative_path: str,
    line_count: int,
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> tuple[tuple[int, int], ...] | None:
    """Return this file's changed line spans against the baseline, or ``None`` if unanswerable.

    An empty tuple is Git's affirmative answer that these worktree bytes *are* the baseline's
    bytes for this path, and it is a different fact from ``None``: nothing changed means there
    is no changed content to hide, while an unanswerable question means the change could not
    be located and must not be attested to. Getting that distinction wrong in either direction
    is the only way this path can misbehave, so both are fenced explicitly below.

    A path Git does not track is reported as changed in its entirety, which is the case that
    would otherwise be silently wrong: ``git diff`` ignores untracked files, so a newly written
    file would come back with no hunks at all and an excerpt showing nothing would claim to
    show the whole change. An added file's every line is changed content.
    """
    if not (workspace / ".git").exists() or line_count == 0:
        return None
    tracked = await process_runner.run(
        ("git", "ls-files", "--error-unmatch", "--", relative_path),
        workspace,
        30.0,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not tracked.succeeded:
        return ((1, line_count),)
    result = await process_runner.run(
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--unified=0",
            await baseline.resolve(),
            "--",
            relative_path,
        ),
        workspace,
        30.0,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not result.succeeded or result.output_truncated:
        return None
    spans: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        header = _DIFF_HUNK_HEADER.match(line)
        if header is None:
            continue
        start = int(header.group("start"))
        count = 1 if header.group("count") is None else int(header.group("count"))
        if count == 0:
            # A hunk that only removed lines. Its `+start` is the line the removal happened
            # after, so the surviving neighbour is what a reviewer needs to see; `start` is 0
            # when the removal was at the head of the file.
            start, count = max(1, start), 1
        first = min(max(1, start), line_count)
        spans.append((first, min(first + count - 1, line_count)))
    if not spans and not baseline.is_branch_point:
        # No hunks against the ``HEAD`` fallback says only that nothing *uncommitted* is left,
        # which is exactly the state a committed earlier attempt leaves behind. Unanswerable,
        # not unchanged -- the caller falls back to truncation and the gate refuses, which is
        # today's behaviour and never a silent approval of source nobody saw.
        return None
    return tuple(spans)


def _fit_changed_regions(
    lines: Sequence[str],
    spans: Sequence[tuple[int, int]],
    *,
    budget: int,
    region_overhead: int = 0,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Choose which changed spans to quote inside the budget, and say which did not fit.

    Selection, not a cap, is what makes a large file reviewable, so this is where the policy
    lives -- and the policy is two orderings, both stated on the entry as `region_order`
    rather than left for a reader to infer from the line numbers.

    **Changed lines before context.** Every span claims its own bytes first, in one pass over
    all of them, before any span claims the unchanged source around it. A changed hunk the
    reviewer can see at all outranks a wider view of an earlier one: the alternative spends
    the whole budget giving the first few hunks their full window and hides the rest, which is
    the failure this selection exists to end rather than a smaller version of it.

    **Earlier hunks before later ones.** What remains after that is spent on context in file
    order, and never across a neighbouring hunk's changed lines -- so growing one region can
    cost another its context but can never cost it its visibility.

    The window itself is `source_region`'s, at `source_region`'s radius: the same helper the
    engineer's prompts use, so a quoted region and its `excerpt_lines` mean the same thing in
    both places. Windows that would overlap are merged, so no line is ever quoted twice.
    """
    content = "\n".join(lines)
    windows: list[tuple[int, int, list[tuple[int, int]]]] = []
    for span in sorted(spans):
        _region, start, end = source_region(
            content, span[0], max_characters=budget, end_line=span[1]
        )
        # `source_region` trims to the bound, so its window may no longer reach the span it was
        # asked about. Widen back to the span itself; what actually fits is decided below,
        # where the whole budget is accounted for at once.
        start, end = min(start, span[0]), max(end, span[1])
        if windows and start <= windows[-1][1] + 1:
            previous = windows[-1]
            windows[-1] = (previous[0], max(previous[1], end), [*previous[2], span])
            continue
        windows.append((start, end, [span]))

    def cost(start: int, end: int) -> int:
        # The heading that says which lines these are counts against the budget too. It is
        # part of what the model is sent, and a per-file bound that only measured the source
        # would be exceeded by every region it selected.
        return len("\n".join(lines[start - 1 : end])) + region_overhead

    # Pass one: the changed lines, and nothing else yet.
    reserved: list[list[int]] = []
    hidden: list[tuple[int, int]] = []
    used = 0
    for window_start, window_end, covered in windows:
        bare_start = min(item[0] for item in covered)
        bare_end = max(item[1] for item in covered)
        if used + cost(bare_start, bare_end) > budget:
            hidden.extend(covered)
            continue
        used += cost(bare_start, bare_end)
        reserved.append([bare_start, bare_end, window_start, window_end])

    # Pass two: whatever is left, spent on context. `source_region` is asked again with the
    # bound raised by the unspent budget, so how much of the window survives is its decision
    # and not a second trimming rule written here.
    for index, region in enumerate(reserved):
        bare_start, bare_end = region[0], region[1]
        floor = reserved[index - 1][1] + 1 if index else 1
        ceiling = reserved[index + 1][0] - 1 if index + 1 < len(reserved) else len(lines)
        bare_cost = cost(bare_start, bare_end)
        # The bound is in source bytes, so this region's own heading comes back off it: what
        # is available is the source it already holds plus whatever the budget has left.
        _region, start, end = source_region(
            content,
            bare_start,
            end_line=bare_end,
            max_characters=bare_cost - region_overhead + budget - used,
        )
        start = max(min(start, bare_start), floor)
        end = min(max(end, bare_end), ceiling)
        grown = cost(start, end)
        if used - bare_cost + grown > budget:
            continue
        used += grown - bare_cost
        region[0], region[1] = start, end
    return tuple((region[0], region[1]) for region in reserved), tuple(hidden)


def _sensitive_review_path(path: Path) -> bool:
    """Identify credential-shaped files that must never enter a reviewer model request.

    The predicate is `is_credential_shaped_path`, shared with engineer context so the two
    cannot answer "is this key material" differently. Only the policy is the reviewer's own:
    a file the change itself touched is evidence, so it is redacted with its limitation
    recorded rather than dropped, and the caller keeps that distinction.
    """
    return is_credential_shaped_path(path)


def _quoted_section_summary(contract_sections: dict[str, Any] | None) -> dict[str, Any] | None:
    """Record which contract sections this review was actually shown, without re-quoting them.

    The names and the omission reasons, never the schemas: the artifact is persisted, served to
    the console and read back by the batch reports, and a second copy of the contract in every
    review would be paid for on all three. What a measurement needs is which sections the judge
    held -- that is the whole variable this change introduces.
    """
    if contract_sections is None:
        return None
    return {
        "contract_version": contract_sections["contract_version"],
        "quoted": [item["name"] for item in contract_sections["sections"]],
        "absent": [item["name"] for item in contract_sections["sections_absent"]],
        "omitted": [
            {"name": item["name"], "reason": item["reason"]}
            for item in contract_sections["sections_omitted"]
        ],
        "characters_selected": contract_sections["characters_selected"],
    }


def _unresolved_contract_references(
    review_payload: dict[str, Any], names: frozenset[str]
) -> list[dict[str, str]]:
    """Name the findings whose `contract_reference` points at no section this contract defines.

    Deliberately a record and not a consequence. `contract_reference` is one of the four things
    that earns a finding blocking authority under a bounded review scope, so refusing that
    authority to an unresolved citation would turn some rejections into publications -- a new
    blocking authority rather than a filter, which is 00-todo item 27's open design question and
    not something to decide as a rider here. It also would not have caught -197, whose citation
    resolved perfectly: `BulkImportSuccess` exists, and the finding was wrong about its
    contents, which is what quoting the section fixes.

    What it does buy is the measurement that would justify a consequence. With the sections now
    in front of the judge there is no longer any excuse for a reference that names nothing, so
    how often one appears is worth counting before anything is built on it.
    """
    findings = review_payload.get("findings")
    if not isinstance(findings, list) or not names:
        return []
    unresolved: list[dict[str, str]] = []
    for item in findings:
        if not isinstance(item, dict) or item.get("finding_category") != "contract":
            continue
        reference = item.get("contract_reference")
        if isinstance(reference, str) and contract_reference_resolves(reference, names):
            continue
        finding_id = item.get("finding_id")
        if not isinstance(finding_id, str):
            continue
        unresolved.append(
            {
                "finding_id": finding_id,
                "contract_reference": reference if isinstance(reference, str) else "",
            }
        )
    return unresolved


def _finding_ids(review_payload: dict[str, Any]) -> set[str]:
    """Return every finding id currently on a review payload."""
    findings = review_payload.get("findings")
    if not isinstance(findings, list):
        return set()
    return {
        item["finding_id"]
        for item in findings
        if isinstance(item, dict) and isinstance(item.get("finding_id"), str)
    }


def _platform_finding_ids(
    review_payload: dict[str, Any],
    *,
    model_supplied: set[str],
    workspace_evidence: dict[str, Any],
) -> list[str]:
    """Name every finding this platform put on the review rather than the model.

    Anything present that the model did not write was appended by one of the deterministic
    injectors, so the difference is exact and the model cannot forge membership: an id it
    supplied is in ``model_supplied`` by construction.

    The evidence gate is the one injector that can *adopt* a finding instead of appending
    one -- if the response already carries its id, the gate sets the terminal verdict and
    returns -- so that id is named explicitly whenever the gate had a limitation to report.
    Without this a response that pre-emptively wrote `REVIEW_EVIDENCE_INCOMPLETE` would look
    like an ordinary model remark, and source that could not be read would stop blocking.
    """
    present = _finding_ids(review_payload)
    injected = present - model_supplied
    if workspace_evidence["limitations"] and _REVIEW_EVIDENCE_FINDING_ID in present:
        injected.add(_REVIEW_EVIDENCE_FINDING_ID)
    return sorted(injected)


def _rejection_contradiction(review: ReviewArtifact) -> str | None:
    """Say how a declining review contradicts itself, where it does.

    Two shapes, both of which decline the work while naming nothing that asks for a change:
    a review with no findings at all, and one whose findings are all `low` -- the severity
    reserved for wording the plan should correct, explicitly not a change to make.
    """
    if review.verdict == "approved":
        return None
    if not review.findings:
        return (
            f"The verdict is '{review.verdict}', but the review raised no findings at all, "
            "so it names nothing for the next attempt to change."
        )
    if all(item.severity == "low" for item in review.findings):
        return (
            f"The verdict is '{review.verdict}', but every finding is severity 'low', which "
            "this review records for wording the plan should correct and explicitly not for "
            "a change the implementation is expected to make. A review that declines the "
            "work must name at least one finding that does ask for a change."
        )
    return None


def _apply_workspace_evidence_findings(
    review_payload: dict[str, Any],
    workspace_evidence: dict[str, Any],
    *,
    evidence_budget: ReviewEvidenceBudget = _DEFAULT_REVIEW_EVIDENCE_BUDGET,
) -> None:
    """Forbid approval when source had to be hidden or truncated for model safety."""
    limitations = workspace_evidence["limitations"]
    if not limitations:
        return
    findings = review_payload.get("findings")
    if not isinstance(findings, list):
        return
    manual_review_required = workspace_evidence["manual_review_required"]
    terminal_verdict = "rejected" if manual_review_required else "changes_requested"
    if any(
        isinstance(item, dict) and item.get("finding_id") == _REVIEW_EVIDENCE_FINDING_ID
        for item in findings
    ):
        review_payload["verdict"] = terminal_verdict
        return
    security_limited = any(item in _HUMAN_DECISION_LIMITATIONS for item in limitations)
    oversized = list(workspace_evidence.get("oversized_paths") or ())
    orphan_lockfiles = list(workspace_evidence.get("orphan_lockfile_paths") or ())
    unshown = list(workspace_evidence.get("unshown_changed_regions") or ())
    seam_omitted = list(workspace_evidence.get("seam_omitted") or ())
    findings.append(
        ReviewFinding(
            finding_id=_REVIEW_EVIDENCE_FINDING_ID,
            severity="high",
            title=_evidence_limitation_title(
                orphan_lockfiles=orphan_lockfiles,
                budget_exceeded=bool(workspace_evidence.get("evidence_budget_exceeded")),
                oversized=oversized,
                unshown=unshown,
                seam_omitted=seam_omitted,
                manual_review_required=manual_review_required,
            ),
            description=_evidence_limitation_description(
                limitations=limitations,
                oversized=oversized,
                orphan_lockfiles=orphan_lockfiles,
                unshown=unshown,
                seam_omitted=seam_omitted,
                manual_review_required=manual_review_required,
                evidence_budget=evidence_budget,
            ),
            recommendation=_evidence_limitation_recommendation(
                oversized=oversized,
                orphan_lockfiles=orphan_lockfiles,
                unshown=unshown,
                seam_omitted=seam_omitted,
                manual_review_required=manual_review_required,
            ),
            # The first oversized file, or failing that the first orphan lockfile or omitted
            # seam module. A finding that names the file is one the next attempt can act on;
            # a finding that names nothing is noise the engineer cannot do anything with,
            # which is what this used to be.
            file_path=(
                (
                    [
                        *oversized,
                        *orphan_lockfiles,
                        *[item["path"] for item in seam_omitted],
                    ]
                    or [None]
                )[0]
                if not manual_review_required
                else None
            ),
            line_number=None,
            finding_category="security" if security_limited else "code_quality",
        ).model_dump(mode="json")
    )
    review_payload["verdict"] = terminal_verdict


def _evidence_limitation_title(
    *,
    orphan_lockfiles: Sequence[str],
    budget_exceeded: bool,
    oversized: Sequence[str],
    unshown: Sequence[dict[str, Any]],
    seam_omitted: Sequence[dict[str, Any]] = (),
    manual_review_required: bool,
) -> str:
    """Name the limitation in the terms of what actually happened.

    An orphan lockfile has its own sentence because it is not a fact about this platform's
    budget at all -- everything about the file was recorded -- but a fact about the change:
    dependency-manager output moved with no declaration to explain it. Titling that "changed
    source could not be shown in full" would describe evidence that was in fact complete, and
    send the next attempt looking for a file to split.

    An omitted seam module has its own sentence for the mirror-image reason: the file is not
    changed source at all -- it is the unchanged module this change shares a channel with --
    so "changed source could not be shown" would send the next attempt splitting a file the
    change never touched.
    """
    if manual_review_required:
        return "Workspace source could not be reviewed completely"
    if orphan_lockfiles and not oversized and not unshown and not seam_omitted:
        return "A generated lockfile changed without its manifest"
    if seam_omitted and not oversized and not unshown:
        return "A module this change shares a channel with could not be shown to the reviewer"
    return (
        "Changed source could not be shown to the reviewer in full"
        if budget_exceeded
        else "Workspace source could not be reviewed completely"
    )


def _orphan_lockfile_sentence(orphan_lockfiles: Sequence[str]) -> str:
    """Say which lockfiles moved alone, and why that is worth a question."""
    named = ", ".join(orphan_lockfiles[:5])
    more = f", and {len(orphan_lockfiles) - 5} more" if len(orphan_lockfiles) > 5 else ""
    return (
        f"These generated lockfiles changed but the manifest each is generated from did not: "
        f"{named}{more}. Their contents are dependency-manager output and were recorded as "
        "facts rather than quoted, but a lockfile that moves on its own is not explained by "
        "any declaration in this change."
    )


def _seam_omitted_sentence(seam_omitted: Sequence[dict[str, Any]]) -> str:
    """Say which seam modules were dropped whole, and which channel package required each."""
    named = "; ".join(
        f"{item['path']} (co-imports {', '.join(item['channel_packages'])})"
        for item in seam_omitted[:5]
    )
    more = f"; and {len(seam_omitted) - 5} more" if len(seam_omitted) > 5 else ""
    return (
        "This change imports a channel package, and these unchanged modules import the same "
        f"package -- the seam the change stands on -- but each is larger than what remained "
        f"of the evidence budget, so it was omitted whole rather than trimmed: {named}{more}. "
        "The review below never saw them, so nothing in it attests that the change is "
        "consistent with the channel configuration they hold."
    )


def _evidence_limitation_description(
    *,
    limitations: Sequence[str],
    oversized: Sequence[str],
    orphan_lockfiles: Sequence[str],
    unshown: Sequence[dict[str, Any]],
    seam_omitted: Sequence[dict[str, Any]] = (),
    manual_review_required: bool,
    evidence_budget: ReviewEvidenceBudget = _DEFAULT_REVIEW_EVIDENCE_BUDGET,
) -> str:
    """Say what the review could not see, in terms of the change rather than of the file.

    The budget sentence used to describe the file: too long, cut off before its end. It now
    describes the *change*, because that is what the evidence now selects and what a reader of
    this finding has to act on. A file being long is not a defect and was never actionable --
    run 186 was told three times to split a 31 KB constants file it had added four lines to.
    A changed hunk that will not fit is a fact about the change, and naming its lines is what
    the next attempt can do something about.

    The limitations compose, so the sentences do too. An orphan lockfile and a hunk that would
    not fit are unrelated problems that can arrive together, and reporting only the first would
    leave the next attempt fixing half of what blocked it and being refused again for the rest.
    """
    if manual_review_required:
        return (
            "One or more changed files required redaction, could not be read, or sit on a path "
            "this platform never publishes, so the changed source could not be inspected."
        )
    parts: list[str] = []
    if orphan_lockfiles:
        parts.append(_orphan_lockfile_sentence(orphan_lockfiles))
    if unshown:
        named = "; ".join(
            f"{item['path']} lines {item['start_line']}-{item['end_line']}" for item in unshown[:5]
        )
        more = f"; and {len(unshown) - 5} more" if len(unshown) > 5 else ""
        parts.append(
            "The review below did not see every changed line. These changed regions are each "
            f"larger than what remained of the "
            f"{evidence_budget.per_file_max_characters:,}-character per-file evidence budget, "
            f"so they were not quoted: {named}{more}."
        )
    elif oversized:
        named = ", ".join(oversized[:5])
        more = f", and {len(oversized) - 5} more" if len(oversized) > 5 else ""
        parts.append(
            "The review below did not see every changed line in these files, and this "
            "repository could not report where their changes are, so the reviewer was shown "
            f"only as much of each as the {evidence_budget.per_file_max_characters:,}-character"
            f" per-file evidence budget holds: {named}{more}."
        )
    if seam_omitted:
        parts.append(_seam_omitted_sentence(seam_omitted))
    if "file_limit_exceeded" in limitations or not parts:
        parts.append(
            f"The change contains more than the {_REVIEW_EVIDENCE_MAX_FILES} files the reviewer "
            "reads in one attempt, so some changed files were not inspected at all."
        )
    return " ".join(parts)


def _evidence_limitation_recommendation(
    *,
    oversized: Sequence[str],
    orphan_lockfiles: Sequence[str],
    unshown: Sequence[dict[str, Any]],
    seam_omitted: Sequence[dict[str, Any]] = (),
    manual_review_required: bool,
) -> str:
    """Say what the next attempt should do, where there is a next attempt."""
    if manual_review_required:
        return "Route this attempt to manual review without another coding retry."
    parts: list[str] = []
    if seam_omitted:
        # The omitted file is not this change's to split: it is unchanged repository source.
        # What the next attempt can change is how the change relates to it.
        parts.append(
            "The modules named above are not part of this change and cannot be made smaller "
            "by it. Import the configured instance those modules export instead of the bare "
            "channel package, or cover the call with a boundary-fake test that exercises the "
            "repository's real channel configuration, so the seam is verified by evidence "
            "the review can see."
        )
    if orphan_lockfiles:
        parts.append(
            "Either declare the dependency change in the manifest each lockfile above is "
            "generated from, so the two agree, or restore the lockfile to the state this "
            "change branched from. Do not hand-edit a lockfile: regenerate it with the "
            "repository's own package manager."
        )
    if unshown:
        parts.append(
            "Make the change to the regions named above in smaller edits, so each contiguous "
            "block of changed lines is small enough to quote, or deliver them across separate "
            "attempts. Restructuring the surrounding file is not what is being asked for, and "
            "do not delete behaviour to fit the limit."
        )
    elif oversized:
        parts.append(
            "Deliver the change to the files named above in smaller, separately reviewable "
            "edits so the reviewer can be shown each one whole. Do not delete behaviour to "
            "fit the limit."
        )
    if not parts:
        parts.append(
            "Deliver this change as fewer files, or split the work across attempts, so every "
            "changed file can be inspected."
        )
    return " ".join(parts)


def _apply_unreviewable_criteria_findings(
    review_payload: dict[str, Any], withheld: Sequence[UnverifiableCriterion]
) -> None:
    """Name every criterion this review set aside, without letting it block the work.

    Severity is `low` and the verdict is untouched deliberately. Blocking on a criterion no
    attempt can satisfy is the loop this withholding exists to end -- feature -047 spent five
    attempts in it. The finding exists so the criterion's absence from the judgement is
    visible to whoever wrote it, and so the plan can be corrected rather than the work failed.
    """
    if not withheld:
        return
    findings = review_payload.get("findings")
    if not isinstance(findings, list):
        return
    existing = {
        item.get("finding_id")
        for item in findings
        if isinstance(item, dict) and isinstance(item.get("finding_id"), str)
    }
    # One finding per requirement, not one for the lot: a `requirement` finding has to name
    # the requirement it belongs to, which is exactly the traceability an author needs to see
    # which of their criteria left the judgement.
    by_requirement: dict[str, list[str]] = {}
    for item in withheld:
        by_requirement.setdefault(item.requirement_id, []).append(item.criterion)
    for requirement_id, criteria in by_requirement.items():
        finding_id = f"ACCEPTANCE_CRITERIA_NOT_REVIEWED-{requirement_id}"
        if finding_id in existing:
            continue
        findings.append(
            ReviewFinding(
                finding_id=finding_id,
                severity="low",
                title="Acceptance criteria were not part of this review",
                description=(
                    f"Requirement {requirement_id} carries criteria that can only be proved by "
                    "measuring a running deployment, which no repository change and no "
                    f"repository command can demonstrate: {'; '.join(criteria)}"
                ),
                recommendation=(
                    "Restate each criterion as behaviour visible in the change, or verify it "
                    "outside this workflow. It was not checked here either way."
                ),
                file_path=None,
                line_number=None,
                requirement_id=requirement_id,
                # Categorised the way REVIEW_EVIDENCE_INCOMPLETE is, for the same reason:
                # this reports what the review could not cover, not a defect in the change.
                # `requirement` is reserved for scoped implementation findings and demands a
                # repository, responsibility, and revision, none of which this has -- the
                # criterion is unmeetable wherever it is reviewed from.
                finding_category="code_quality",
            ).model_dump(mode="json")
        )
        existing.add(finding_id)


def _review_scope(task_plan: TaskPlanArtifact) -> dict[str, Any]:
    """Return the explicit child assignment, or the full scope for legacy workflows."""
    raw_scope = task_plan.metadata.get("review_scope")
    if raw_scope is None:
        return {
            "scope_kind": "full_technical_prd",
            "task_ids": [task.task_id for task in task_plan.tasks],
            **_reviewable_scope_criteria(
                [criterion for task in task_plan.tasks for criterion in task.acceptance_criteria]
            ),
            "test_requirements": task_plan.test_strategy,
        }
    if not isinstance(raw_scope, dict):
        msg = "task plan review_scope metadata must be an object"
        raise AgentArtifactError(msg)
    scoped_references = _requirement_references(raw_scope, "scoped_requirements")
    shared_references = _requirement_references(raw_scope, "shared_requirements")
    requirement_ids = raw_scope.get("requirement_ids")
    if scoped_references:
        reference_ids = [item["requirement_id"] for item in scoped_references]
        requirement_ids = list(dict.fromkeys(reference_ids))
    if (
        not isinstance(requirement_ids, list)
        or not requirement_ids
        or any(not isinstance(item, str) or not item.strip() for item in requirement_ids)
        or len(set(requirement_ids)) != len(requirement_ids)
    ):
        msg = "task plan review_scope requirement_ids must be a non-empty unique string list"
        raise AgentArtifactError(msg)
    required_string_fields = (
        "repository_id",
        "workstream_id",
        "role",
    )
    if any(
        not isinstance(raw_scope.get(field), str) or not raw_scope[field].strip()
        for field in required_string_fields
    ):
        msg = "task plan review_scope must identify its repository, workstream, and role"
        raise AgentArtifactError(msg)
    return {
        "scope_kind": "repository_workstream",
        "repository_id": raw_scope["repository_id"],
        "workstream_id": raw_scope["workstream_id"],
        "role": raw_scope["role"],
        "requirement_ids": requirement_ids,
        "scoped_requirements": scoped_references,
        "out_of_scope_requirements": _string_list(raw_scope, "out_of_scope_requirements"),
        "shared_requirements": shared_references,
        "task_ids": [task.task_id for task in task_plan.tasks],
        "responsibilities": _string_list(raw_scope, "responsibilities"),
        **_reviewable_scope_criteria(_string_list(raw_scope, "acceptance_criteria")),
        "test_requirements": _string_list(raw_scope, "test_requirements"),
        "contract_sections_consumed": _string_list(raw_scope, "contract_sections_consumed"),
        "contract_sections_implemented": _string_list(raw_scope, "contract_sections_implemented"),
        "design_nodes": _string_list(raw_scope, "design_nodes"),
        "expected_files_or_areas": _string_list(raw_scope, "expected_files_or_areas"),
        "implementation_expectations": _implementation_expectations(raw_scope),
        "repository_revision": task_plan.metadata.get("repository_revision"),
    }


def _reviewable_scope_criteria(criteria: list[str]) -> dict[str, list[str]]:
    """Split a workstream's own acceptance criteria the way requirement criteria are split.

    The withholding applied only to PRD requirements, but the reviewer enforces *these* -- the
    planner's scoped criteria -- and they bypassed it entirely. Live feature -083 spent ten
    frontend attempts against a scoped criterion demanding manual verification of rendering,
    which no attempt could satisfy and which the requirement-level projection never saw.
    """
    return {
        "acceptance_criteria": [
            item for item in criteria if not requires_deployed_measurement(item)
        ],
        "acceptance_criteria_not_reviewable": [
            item for item in criteria if requires_deployed_measurement(item)
        ],
    }


def _string_list(scope: dict[str, Any], field: str) -> list[str]:
    """Retain only well-formed, non-secret textual workstream context."""
    value = scope.get(field, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        msg = f"task plan review_scope field '{field}' must be a string list"
        raise AgentArtifactError(msg)
    return value


def _requirement_references(scope: dict[str, Any], field: str) -> list[dict[str, str | list[str]]]:
    """Validate the serialised workstream ownership references before model review."""
    value = scope.get(field, [])
    if not isinstance(value, list):
        msg = f"task plan review_scope field '{field}' must be a list"
        raise AgentArtifactError(msg)
    references: list[dict[str, str | list[str]]] = []
    for item in value:
        if not isinstance(item, dict):
            msg = f"task plan review_scope field '{field}' contains a non-object"
            raise AgentArtifactError(msg)
        requirement_id = item.get("requirement_id")
        responsibility = item.get("responsibility")
        acceptance_ids = item.get("acceptance_criterion_ids")
        if (
            not isinstance(requirement_id, str)
            or not requirement_id.strip()
            or responsibility not in {"implements", "consumes", "validates", "documents"}
            or not isinstance(acceptance_ids, list)
            or not acceptance_ids
            or any(not isinstance(value, str) or not value.strip() for value in acceptance_ids)
        ):
            msg = f"task plan review_scope field '{field}' has an invalid requirement reference"
            raise AgentArtifactError(msg)
        references.append(
            {
                "requirement_id": requirement_id,
                "responsibility": responsibility,
                "acceptance_criterion_ids": list(acceptance_ids),
            }
        )
    return references


def _implementation_expectations(scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the planner's explicit production/test completion contract."""
    value = scope.get("implementation_expectations", [])
    if not isinstance(value, list):
        msg = "task plan review_scope implementation_expectations must be a list"
        raise AgentArtifactError(msg)
    expectations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            msg = "task plan implementation expectation must be an object"
            raise AgentArtifactError(msg)
        requirement_id = item.get("requirement_id")
        categories = item.get("expected_change_categories")
        areas = item.get("expected_source_areas", [])
        tests_required = item.get("tests_required")
        if (
            not isinstance(requirement_id, str)
            or not isinstance(categories, list)
            or not categories
            or any(not isinstance(category, str) for category in categories)
            or not isinstance(areas, list)
            or any(not isinstance(area, str) for area in areas)
            or not isinstance(tests_required, bool)
        ):
            msg = "task plan implementation expectation is invalid"
            raise AgentArtifactError(msg)
        expectations.append(
            {
                "requirement_id": requirement_id,
                "expected_change_categories": categories,
                "expected_source_areas": areas,
                "tests_required": tests_required,
            }
        )
    return expectations


def _reviewable_requirement(requirement: Any) -> dict[str, Any]:
    """Project one requirement without the criteria a repository change cannot demonstrate.

    A requirement is written before the checkout exists, and a planner asked for latency to be
    proved by "a staging or production-like load test measuring the 95th percentile". The
    reviewer is right to read that as unmet -- no diff can contain it -- so feature -047 spent
    five of its attempts being rejected for it. The requirement itself still stands and is
    still judged on whether the implementation is consistent with it; only its unreachable
    proof is withheld, and it is reported so the plan can be corrected rather than lost.
    """
    payload: dict[str, Any] = requirement.model_dump(mode="json")
    criteria = payload.get("acceptance_criteria")
    if not isinstance(criteria, list):
        return payload
    reviewable = [item for item in criteria if not requires_deployed_measurement(item)]
    withheld = [item for item in criteria if requires_deployed_measurement(item)]
    if not withheld:
        return payload
    payload["acceptance_criteria"] = reviewable
    payload["acceptance_criteria_not_reviewable"] = withheld
    return payload


def _review_technical_prd(
    technical_prd: TechnicalPRDArtifact, review_scope: dict[str, Any]
) -> dict[str, Any]:
    """Project the PRD to one child assignment so sibling work cannot block its review."""
    if review_scope["scope_kind"] == "full_technical_prd":
        # Unreviewable criteria are withheld here too. This path used to hand them straight to
        # the model, so the same criterion that could not be met in a scoped review was
        # enforced in full in an unscoped one.
        payload = technical_prd.model_dump(mode="json")
        payload["functional_requirements"] = [
            _reviewable_requirement(item) for item in technical_prd.functional_requirements
        ]
        payload["non_functional_requirements"] = [
            _reviewable_requirement(item) for item in technical_prd.non_functional_requirements
        ]
        return payload
    assigned_ids = set(review_scope["requirement_ids"])
    all_requirements = [
        *technical_prd.functional_requirements,
        *technical_prd.non_functional_requirements,
    ]
    known_ids = {requirement.requirement_id for requirement in all_requirements}
    unknown_ids = assigned_ids - known_ids
    if unknown_ids:
        msg = f"task plan review_scope references unknown requirements: {sorted(unknown_ids)}"
        raise AgentArtifactError(msg)
    return {
        "title": technical_prd.title,
        "solution_summary": technical_prd.solution_summary,
        "functional_requirements": [
            _reviewable_requirement(requirement)
            for requirement in technical_prd.functional_requirements
            if requirement.requirement_id in assigned_ids
        ],
        "non_functional_requirements": [
            _reviewable_requirement(requirement)
            for requirement in technical_prd.non_functional_requirements
            if requirement.requirement_id in assigned_ids
        ],
        # The rest of the feature is deliberately only non-actionable background.
        "background": {
            "data_requirements": technical_prd.data_requirements,
            "integration_requirements": technical_prd.integration_requirements,
            "security_requirements": technical_prd.security_requirements,
            "assumptions": technical_prd.assumptions,
        },
    }


def _validate_requirement_checks(
    review_payload: dict[str, Any], review_scope: dict[str, Any]
) -> None:
    """Keep child review checks traceable only to the requirements it owns."""
    if review_scope["scope_kind"] != "repository_workstream":
        return
    checks = review_payload.get("requirement_checks")
    if not isinstance(checks, list):
        msg = "review response field 'requirement_checks' must be a list"
        raise AgentArtifactError(msg)
    actual_ids = [
        check.get("requirement_id")
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("requirement_id"), str)
    ]
    expected_ids = set(review_scope["requirement_ids"])
    if len(actual_ids) != len(checks) or set(actual_ids) != expected_ids:
        msg = (
            "repository review requirement_checks must cover exactly its assigned requirements: "
            f"{sorted(expected_ids)}"
        )
        raise AgentArtifactError(msg)


_VOLATILE_REVIEW_INPUT_FIELDS = frozenset(
    {"timestamp", "response_id", "operation_id", "duration_seconds"}
)


def _review_input_signature(payload: dict[str, Any]) -> str:
    """Hash stable repository-review evidence without regenerated envelope data.

    The review is determined by the scoped requirements, code evidence, current validation,
    and exact reviewed workspace bytes. Task-plan preflight describes the checkout before the
    Engineer wrote those bytes, so it changes during crash recovery and is intentionally
    superseded here by the validation and workspace fingerprints produced afterwards.
    """
    stable = cast(dict[str, Any], json.loads(json.dumps(payload)))
    task_plan = stable.get("task_plan")
    if isinstance(task_plan, dict):
        metadata = task_plan.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("preflight_result", None)
            metadata.pop("repository_revision", None)
    review_scope = stable.get("review_scope")
    if isinstance(review_scope, dict):
        review_scope.pop("repository_revision", None)

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in sorted(value.items())
                if key not in _VOLATILE_REVIEW_INPUT_FIELDS
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    encoded = json.dumps(normalize(stable), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _apply_validation_findings(
    review_payload: dict[str, Any],
    validation_results: tuple[ValidationResult, ...],
    *,
    test_validation_required: bool = False,
    changed_test_paths: Sequence[str] = (),
    planned_commands: Sequence[ValidationCommand] = (),
) -> None:
    """Ensure failed lint or tests always block approval with a concrete review finding."""
    findings = review_payload.get("findings")
    if not isinstance(findings, list):
        msg = "review response field 'findings' must be a list"
        raise AgentArtifactError(msg)
    failed_results = [
        result
        for result in validation_results
        if result.status is ValidationStatus.FAILED and result.required
    ]
    optional_failed_results = [
        result
        for result in validation_results
        if result.status is ValidationStatus.FAILED and not result.required
    ]
    informational_results = [
        result
        for result in validation_results
        if result.status in {ValidationStatus.NOT_CONFIGURED, ValidationStatus.NO_TESTS_FOUND}
    ]
    if failed_results:
        review_payload["verdict"] = "changes_requested"
    existing_ids = {
        finding.get("finding_id")
        for finding in findings
        if isinstance(finding, dict) and isinstance(finding.get("finding_id"), str)
    }
    for result in failed_results:
        name = result.command[0] if result.command else result.validation_type
        # Keyed by validation type as well as executable: every Node command starts with
        # the same package manager, so an executable-only id collides across lint, test
        # and build and makes the whole review artifact fail its uniqueness rule.
        finding_id = f"validation-{result.validation_type}-{name}"
        if result.validation_type == "lint" and result.failure_classification in {
            "lint_configuration_error",
            "missing_dependency",
            "unsupported_runtime",
        }:
            finding_id = "LINT_CONFIGURATION_INVALID"
        # A command that ran out of time or memory is reported as its own category. It looks
        # identical to a rejection from here -- non-zero, with output -- but no coding retry
        # can answer it, which is what cost AB-Feature-108 six attempts.
        capacity = result.failure_classification in CAPACITY_FAILURE_CLASSIFICATIONS
        if capacity:
            finding_id = f"VALIDATION_CAPACITY_{result.validation_type.upper()}"
        if finding_id in existing_ids:
            continue
        findings.append(
            ReviewFinding(
                finding_id=finding_id,
                severity="high",
                title=(
                    "Lint configuration is invalid"
                    if finding_id == "LINT_CONFIGURATION_INVALID"
                    else f"{result.command[0]} validation could not complete"
                    if capacity
                    else f"{result.command[0]} validation failed"
                ),
                description=_validation_failure_description(result),
                recommendation=(
                    "Scope this command to the suites this change touches, or raise its "
                    "time and memory limits. Do not change application source in response "
                    "to it: it returned no verdict on the source."
                    if capacity
                    else f"Fix the reported {result.command[0]} validation failures and retry."
                ),
                file_path=None,
                line_number=None,
                repository_id=result.repository_id,
                validated_revision=result.repository_revision,
                finding_category="validation_capacity" if capacity else "validation_failure",
            ).model_dump(mode="json")
        )
        existing_ids.add(finding_id)
    for result in [*optional_failed_results, *informational_results]:
        assert result.status is not None
        name = result.command[0] if result.command else result.validation_type
        finding_id = (
            "TEST_COMMAND_NOT_CONFIGURED"
            if result.result_code == "TEST_COMMAND_NOT_CONFIGURED"
            else f"validation-{result.status.value}-{result.validation_type}-{name}"
        )
        if finding_id in existing_ids:
            continue
        # `no_tests_found` only. `not_configured` is the neighbouring status and the narrowing
        # below must never reach it: it says this repository has NO test command at all, so
        # nothing about the change was checked and no question about which files a runner could
        # collect even has a meaning -- there is no runner. That one blocks whenever tests were
        # required, exactly as it did, and a repository whose checks cannot run is a fact about
        # the checkout that an approving review does not get to override.
        empty_test_status = (
            result.validation_type == "test"
            and result.status is ValidationStatus.NO_TESTS_FOUND
            and test_validation_required
        )
        unconfigured_tests_block = (
            result.validation_type == "test"
            and result.status is ValidationStatus.NOT_CONFIGURED
            and test_validation_required
        )
        # Whether an empty suite is this change's fault or the plan's. The gate exists for the
        # attempt that was asked for tests and wrote none, and for that one it is unchanged: no
        # changed test files means the demand stands whatever the runner is.
        #
        # It becomes unsatisfiable the moment the attempt DID write tests and the planned
        # runner cannot collect them. AB-Feature-225 wrote two browser suites into the area its
        # own plan named; the planned command was the repository's unit runner, whose config
        # excludes that area, so `no_tests_found` was the only answer it could ever return. The
        # verdict was forced to `changes_requested` on every attempt, at `high`, for the whole
        # life of the workstream -- and the one action that would have cleared it, adding a
        # unit test to a suite the feature has no use for, is the one the review was separately
        # telling the attempt not to take. No attempt count reaches approval through that.
        plan_can_see_the_tests = _plan_could_collect(
            test_paths=changed_test_paths,
            commands=planned_commands,
            results=validation_results,
        )
        test_status_requires_blocking = unconfigured_tests_block or (
            empty_test_status and plan_can_see_the_tests
        )
        unrunnable_by_plan = empty_test_status and not plan_can_see_the_tests
        if test_status_requires_blocking:
            review_payload["verdict"] = "changes_requested"
        findings.append(
            ReviewFinding(
                finding_id=finding_id,
                severity="high" if test_status_requires_blocking else "low",
                title=(
                    f"{name} cannot collect this change's tests"
                    if unrunnable_by_plan
                    else f"{name} validation is {result.status.value}"
                ),
                description=(
                    _uncollectable_tests_description(result, changed_test_paths)
                    if unrunnable_by_plan
                    else _validation_failure_description(result)
                ),
                recommendation=(
                    "This is a defect in the assignment, not in the change: the workstream was "
                    "given a test requirement whose coverage the planned validation commands "
                    "cannot execute. Do not rewrite the change to satisfy it, and do not add "
                    "test infrastructure the repository does not otherwise want."
                    if unrunnable_by_plan
                    else "Configure or add repository-native tests when this workstream "
                    "requires them."
                ),
                file_path=None,
                line_number=None,
                repository_id=result.repository_id,
                validated_revision=result.repository_revision,
                finding_category="validation_failure",
            ).model_dump(mode="json")
        )
        existing_ids.add(finding_id)


def _plan_could_collect(
    *,
    test_paths: Sequence[str],
    commands: Sequence[ValidationCommand],
    results: Sequence[ValidationResult] = (),
) -> bool:
    """Say whether a required planned test command could have discovered these test files.

    Two defaults, both chosen so the platform never claims more than it knows:

    * **No changed test files at all** is `True`. The empty suite is then a fact about the
      change -- it was asked for tests and wrote none -- and the gate above should keep firing.
    * **A runner whose collectible suffixes are unknown** is `True`, the same convention
      `scoped_test_command` uses for the same reason: an unrecognised runner may well collect
      anything, and reading silence as "could not have" would quietly retire the gate.

    Only required test commands count. An advisory one carries no verdict, so its inability to
    see a suite is not grounds for holding a change that everything else accepted.
    """
    if not test_paths:
        return True
    # Measured beats guessed, so this is asked first. The suffix table below reads a runner's
    # *language*, and a runner that can parse a file may still be configured not to collect it:
    # a unit runner told to ignore the browser-test directory is the ordinary arrangement, and
    # by extension alone its specs are indistinguishable from the unit tests beside them. The
    # narrowed run already put the question to the runner itself at no extra cost -- the
    # platform hands it the attempt's own test paths -- and `no_tests_found` from that run is
    # the runner answering that it collects none of them.
    if _a_narrowed_run_collected_nothing(test_paths, results):
        return False
    for command in commands:
        if command.validation_type != "test" or not command.required:
            continue
        suffixes = {suffix.lower() for suffix in command.collectible_suffixes}
        if not suffixes:
            return True
        if any(PurePosixPath(path).suffix.lower() in suffixes for path in test_paths):
            return True
    return False


def _a_narrowed_run_collected_nothing(
    test_paths: Sequence[str], results: Sequence[ValidationResult]
) -> bool:
    """Whether the runner was handed this attempt's tests and reported collecting none.

    Only an advisory run counts, because only a narrowed one is advisory: `scoped_test_command`
    marks its output `required=False` precisely so the full suite stays the authoritative gate.
    A required command reporting an empty suite is the whole-repository statement and says
    nothing about these paths in particular.
    """
    wanted = set(test_paths)
    return any(
        result.validation_type == "test"
        and not result.required
        and result.status is ValidationStatus.NO_TESTS_FOUND
        and wanted.intersection(result.command)
        for result in results
    )


def _completion_test_paths(completion: CodeCompletionArtifact) -> list[str]:
    """The attempt's own test files, by the same reading `_apply_completion_findings` uses."""
    return list(completion.test_files_changed) or [
        item.path for item in completion.file_changes if classify_file_change(item.path) == "test"
    ]


def _planned_validation_commands(tool: ValidationTool) -> tuple[ValidationCommand, ...]:
    """Read the plan the results came from, without coupling test doubles to it."""
    plan = getattr(tool, "validation_plan", None)
    if not callable(plan):
        return ()
    resolved = plan()
    commands = getattr(resolved, "commands", None)
    if not isinstance(commands, Sequence):
        return ()
    return tuple(item for item in commands if isinstance(item, ValidationCommand))


def _uncollectable_tests_description(result: ValidationResult, test_paths: Sequence[str]) -> str:
    """State the mismatch as the plan's, and name both sides of it."""
    named = ", ".join(sorted(test_paths)[:5])
    command = " ".join(result.command)
    status = result.status.value if result.status is not None else "no result"
    return (
        f"{command} reports {status}, and it cannot collect the test files this change added "
        f"({named}). The workstream's test requirements ask for coverage that no planned "
        "validation command is able to execute, so no attempt can produce a passing result "
        "for it. Recorded as a question about the assignment rather than a defect in the "
        "change."
    )


def _validation_failure_description(result: ValidationResult) -> str:
    """Create an issue description that carries the failing command's own report.

    This description becomes a review finding, the finding becomes a blocking issue, and the
    blocking issue becomes the root cause in the retry plan the engineer is handed. It is the
    whole path by which a failing command tells the next attempt what to change, and for a
    plain non-zero exit it said only that the command exited non-zero.

    `ValidationResult` has captured the tail of that output for exactly this purpose --
    "a failing command's own report is the only thing that says what to change" -- and it was
    never read here. AB-Feature-123's console spent six attempts on `npm run test ... exited
    with code 1` while the 4kB sitting beside it in `stderr_summary` said:

        AllApps bulk deletion integration > passes the established full-access value
        mockConstructor(...): Nothing was returned from render. This usually means a
        return statement is missing. Or, to render nothing, return null.

    which names the defect, the test and the line. The excerpt is already tail-bounded and
    passed through the same redaction as any other subprocess output, and the pre-commit gate
    has always quoted its diagnostics this way, so nothing new becomes quotable here.
    """
    if result.timed_out:
        return f"{' '.join(result.command)} exceeded its configured timeout."
    if result.failure_classification == VALIDATION_RESOURCE_EXHAUSTED:
        return (
            f"{' '.join(result.command)} ran out of memory before it finished, so it "
            "reported no verdict on the source."
        )
    if result.status is ValidationStatus.NOT_CONFIGURED:
        return "TEST_COMMAND_NOT_CONFIGURED: no repository-native test command is configured."
    if result.status is ValidationStatus.NO_TESTS_FOUND:
        return f"{' '.join(result.command)} found no tests; this is not a test failure."
    headline = f"{' '.join(result.command)} exited with code {result.return_code}."
    excerpt = _validation_failure_excerpt(result)
    return f"{headline}\n{excerpt}" if excerpt else headline


# How much of a failing command's report to carry into the finding. Enough for a runner's
# failure block and its summary; short enough that three of them still leave room for the
# review's own reasoning in the next attempt's prompt.
_MAX_VALIDATION_EXCERPT_CHARACTERS = 1_500


def _validation_failure_excerpt(result: ValidationResult) -> str:
    """Return the failing command's own captured report, bounded for a retry prompt.

    Standard error first: every runner the platform drives reports its failures there, and
    standard output is usually the package manager's banner.
    """
    for summary in (result.stderr_summary, result.stdout_summary):
        text = (summary or "").strip()
        if not text:
            continue
        if len(text) > _MAX_VALIDATION_EXCERPT_CHARACTERS:
            # Tail again: `_bounded_summary` already kept the end of the output because that
            # is where the failure is, and trimming from the front here preserves that.
            text = f"[truncated]\n{text[-_MAX_VALIDATION_EXCERPT_CHARACTERS:]}"
        return text
    return ""


def _apply_completion_findings(
    review_payload: dict[str, Any],
    completion: CodeCompletionArtifact,
    review_scope: dict[str, Any],
) -> None:
    """Block approval when tests are substituted for expected production source work."""
    expectations = review_scope.get("implementation_expectations", [])
    findings = review_payload.get("findings")
    if not isinstance(expectations, list) or not isinstance(findings, list):
        return
    production_categories = {
        "production",
        "route",
        "controller",
        "service",
        "model",
        "migration",
        "frontend_component",
        "client",
    }
    existing_ids = {
        item.get("finding_id")
        for item in findings
        if isinstance(item, dict) and isinstance(item.get("finding_id"), str)
    }
    scoped = {
        item.get("requirement_id"): item
        for item in review_scope.get("scoped_requirements", [])
        if isinstance(item, dict) and isinstance(item.get("requirement_id"), str)
    }
    missing = set(completion.requirements_not_implemented)
    test_files = completion.test_files_changed or [
        item.path
        for item in completion.file_changes
        if "test" in item.path.lower() or "spec" in item.path.lower()
    ]
    production_files = completion.production_files_changed or [
        item.path
        for item in completion.file_changes
        if item.path not in test_files and not item.path.lower().endswith(".md")
    ]
    for expectation in expectations:
        if not isinstance(expectation, dict):
            continue
        requirement_id = expectation.get("requirement_id")
        categories = expectation.get("expected_change_categories", [])
        if not isinstance(requirement_id, str) or not isinstance(categories, list):
            continue
        expected_production = [item for item in categories if item in production_categories]
        tests_only = bool(expected_production) and bool(test_files) and not production_files
        if requirement_id not in missing and not tests_only:
            continue
        review_payload["verdict"] = "changes_requested"
        reference = scoped.get(requirement_id, {})
        responsibility = reference.get("responsibility", "implements")
        for category in expected_production:
            code = (
                f"BACKEND_{category.upper()}_NOT_IMPLEMENTED"
                if review_scope.get("role") == "backend"
                else f"{category.upper()}_NOT_IMPLEMENTED"
            )
            if code in existing_ids:
                continue
            findings.append(
                ReviewFinding(
                    finding_id=code,
                    severity="high",
                    title=f"Required {category.replace('_', ' ')} is not implemented",
                    description=(
                        f"Requirement {requirement_id} expects a production {category} change, "
                        "but its completion evidence does not satisfy that expectation."
                    ),
                    recommendation=(
                        f"Implement the required {category} in the scoped source area and retain "
                        "tests as supporting evidence."
                    ),
                    file_path=None,
                    line_number=None,
                    repository_id=review_scope.get("repository_id"),
                    requirement_id=requirement_id,
                    responsibility=responsibility,
                    validated_revision=review_scope.get("repository_revision") or "unvalidated",
                    evidence="Code completion does not include the required production change.",
                    recommended_fix=f"Add the required {category} implementation before review.",
                    finding_category="requirement",
                ).model_dump(mode="json")
            )
            existing_ids.add(code)
        if tests_only and "TESTS_ONLY_CHANGE" not in existing_ids:
            findings.append(
                ReviewFinding(
                    finding_id="TESTS_ONLY_CHANGE",
                    severity="high",
                    title="Tests-only change cannot satisfy production implementation",
                    description=(
                        "Only test files changed while the scoped requirement expects production "
                        "source implementation."
                    ),
                    recommendation=(
                        "Implement the missing production source before revising tests again."
                    ),
                    file_path=None,
                    line_number=None,
                    repository_id=review_scope.get("repository_id"),
                    requirement_id=requirement_id,
                    responsibility=responsibility,
                    validated_revision=review_scope.get("repository_revision") or "unvalidated",
                    evidence="Completion artifact lists test files but no production files.",
                    recommended_fix=(
                        "Add the missing production implementation in the repository's "
                        "established source area."
                    ),
                    finding_category="requirement",
                ).model_dump(mode="json")
            )
            existing_ids.add("TESTS_ONLY_CHANGE")


def _prior_findings(state: AgentState, review_scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only previous findings emitted by this child, never a sibling's review."""
    repository_id = review_scope.get("repository_id")
    findings: list[dict[str, Any]] = []
    for artifact in state["artifacts"]:
        if not isinstance(artifact, ReviewArtifact):
            continue
        for finding in artifact.findings:
            if repository_id is None or finding.repository_id in {None, repository_id}:
                findings.append(finding.model_dump(mode="json"))
    return findings


def _prior_reviews(state: AgentState) -> list[ReviewArtifact]:
    """Return this workstream's earlier reviews, oldest first."""
    return [
        artifact
        for artifact in state["artifacts"]
        if isinstance(artifact, ReviewArtifact)
        and artifact_id_matches_lineage(artifact.artifact_id, ARTIFACT_FILENAMES["review"])
    ]


def _previously_omitted_seam_paths(prior_reviews: Sequence[ReviewArtifact]) -> frozenset[str]:
    """Which unchanged seam files the immediately preceding review's own budget omitted whole.

    Read from the most recent review only, not every prior one: a file the budget could not
    show two rounds ago but showed last round is not a repeat, and treating it as one would
    escalate a limitation that already stopped recurring. `None` metadata (every review
    written before this existed) reads as "nothing recorded," the same answer a first attempt
    gives -- absence of the record is never treated as a repeat.
    """
    if not prior_reviews:
        return frozenset()
    recorded = prior_reviews[-1].metadata.get("seam_evidence_budget_omitted_paths")
    if not isinstance(recorded, list):
        return frozenset()
    return frozenset(path for path in recorded if isinstance(path, str))


def _prior_blocking_finding_summaries(reviews: list[ReviewArtifact]) -> list[dict[str, Any]]:
    """Summarise what earlier rounds blocked on, so a remediation round can hold it to that.

    Critical and high findings only, from the newest review backwards, because those are
    what actually stopped the work; the id is included so an unresolved repeat can carry it.
    """
    summaries: list[dict[str, Any]] = []
    for review in reversed(reviews):
        for finding in review.findings:
            if finding.severity not in {"critical", "high"}:
                continue
            summaries.append(
                {
                    "finding_id": finding.finding_id,
                    "severity": finding.severity,
                    "title": finding.title,
                    "description": finding.description,
                    "file_path": finding.file_path,
                }
            )
    return summaries


def _self_review_inconclusive_paths(code_completion: CodeCompletionArtifact) -> list[str]:
    """The changed production files the attempt's own self-review could not read.

    Read off the completion rather than recomputed, for the 61- reason: what one record
    states about another is a lookup, never a second derivation that can disagree with it.
    Absent, malformed or `clean` all answer the same empty list -- this reports a declared
    limitation and never infers one.
    """
    self_review = code_completion.metadata.get("self_review")
    if not isinstance(self_review, dict):
        return []
    paths = self_review.get("inconclusive_paths")
    if not isinstance(paths, list):
        return []
    return [item for item in paths if isinstance(item, str)]


def _validation_context_metadata(tool: ValidationTool) -> dict[str, Any]:
    """Expose safe detection/plan data in artifacts without coupling test doubles to it."""
    metadata: dict[str, Any] = {}
    profile = getattr(tool, "technology_profile", None)
    if callable(profile):
        resolved = profile()
        if hasattr(resolved, "model_dump"):
            metadata["technology_profile"] = resolved.model_dump(mode="json")
    plan = getattr(tool, "validation_plan", None)
    if callable(plan):
        resolved = plan()
        if hasattr(resolved, "model_dump"):
            metadata["validation_plan"] = resolved.model_dump(mode="json")
    return metadata
