"""Resolve the model role for an engineering attempt the retry policy already allowed.

Failure ownership and retry eligibility are intentionally upstream. This module cannot grant
an attempt, alter a budget, or turn a model change into progress. It receives the classified
failure and repair evidence, chooses either the primary coding role or the scoped mechanical
fix role, resolves that role through the central model configuration, and persists the answer.

There is no model ladder. The same substantive failure remains on ``CODING`` and the same
localized mechanical failure remains on ``SCOPED_FIX`` until the existing convergence and
retry policies stop it or changed evidence gives it a different repair scope.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import Field, field_validator

from configs.model_roles import AgentPlatform, ModelConfigService, ModelRole, PerformanceTier
from state.models import StateModel
from tools.retry_strategy import FailureClassification, requires_feature_engineer
from tools.review_fix_classification import (
    ReviewFixClassification,
    ReviewFixComplexity,
    highest_complexity,
)


class ModelExecutionMode(StrEnum):
    """What the Engineer is being asked to do."""

    INITIAL_IMPLEMENTATION = "INITIAL_IMPLEMENTATION"
    REVIEW_REMEDIATION = "REVIEW_REMEDIATION"


class RepairScope(StrEnum):
    """Whether the classified correction is safe for the scoped-fix role."""

    SCOPED = "scoped"
    SUBSTANTIVE = "substantive"
    AMBIGUOUS = "ambiguous"


# Role names written by the tiered router this module replaced. A record carrying one is read
# as the role that would execute the same work today. That is the mapping both consumers of a
# persisted role already applied before acting on it, so nothing about execution changes; the
# names are now translated on the way in rather than carried through as a second vocabulary.
_RETIRED_ROLE_NAMES: Mapping[str, ModelRole] = {
    "fix": ModelRole.SCOPED_FIX,
    "complex_fix": ModelRole.CODING,
    "escalation": ModelRole.CODING,
}

# What the retired ladder wrote alongside a decision. `StateModel` forbids undeclared fields,
# so a durable record written before this deletion would stop loading altogether unless they
# are dropped on the way in -- which is a worse outcome than the dead code was.
_RETIRED_DECISION_FIELDS = frozenset({"escalation_level", "escalated", "previous_role"})


class ModelRoutingInputs(StateModel):
    """The retry-policy-approved question whose answer is a configured model role."""

    feature_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    execution_mode: ModelExecutionMode
    attempt: int = Field(ge=0)
    failure_classification: FailureClassification | None = None
    repository_revision: str | None = None
    finding_fingerprints: list[str] = Field(default_factory=list)
    validation_fingerprint: str = ""
    contract_version: str = ""
    repair_version: str = ""

    @field_validator("execution_mode", mode="before")
    @classmethod
    def execution_mode_must_be_known(cls, value: ModelExecutionMode | str) -> ModelExecutionMode:
        """Rehydrate the execution mode from persisted JSON."""
        return value if isinstance(value, ModelExecutionMode) else ModelExecutionMode(value)

    @field_validator("failure_classification", mode="before")
    @classmethod
    def failure_classification_must_be_known(
        cls, value: FailureClassification | str | None
    ) -> FailureClassification | None:
        """Rehydrate the authoritative failure classification where one exists."""
        if value is None or isinstance(value, FailureClassification):
            return value
        return FailureClassification(value)

    def fingerprint(self) -> str:
        """Fingerprint the effective task, deliberately excluding role, model and effort."""
        canonical = {
            "feature_id": self.feature_id,
            "repository_id": self.repository_id,
            "execution_mode": self.execution_mode.value,
            "attempt": self.attempt,
            "failure_classification": (
                self.failure_classification.value if self.failure_classification is not None else ""
            ),
            "repository_revision": self.repository_revision or "",
            "finding_fingerprints": sorted(self.finding_fingerprints),
            "validation_fingerprint": self.validation_fingerprint,
            "contract_version": self.contract_version,
            "repair_version": self.repair_version,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


class ModelRoutingDecision(StateModel):
    """The resolved execution selection and short, non-chain-of-thought reason for it."""

    execution_mode: ModelExecutionMode
    role: ModelRole
    # Which provider executes this attempt. Recorded here for the reason the model is: the
    # engineer's client is rebuilt from this record on recovery rather than from
    # configuration, and a resumed attempt that switched SDK mid-workstream would make its
    # own routing record false.
    #
    # A decision persisted before this field existed carries no provider and is read as
    # `openai`, which is what it ran on.
    platform: AgentPlatform = AgentPlatform.OPENAI
    # The feature's pinned cost basis, recorded beside the provider so every attempt's cost
    # is attributable afterwards. Stamped by the workflow from feature state, never decided
    # here: the router chooses roles, not tiers. A decision persisted before the field
    # existed is read as `high`, which is what the unsuffixed configuration it resolved
    # against has always been.
    performance_tier: PerformanceTier = PerformanceTier.HIGH
    # Which model setup pinned a custom feature's selections, stamped by the workflow beside
    # the tier and None for a tier feature. Provenance only -- the values above are already
    # the record, so an edited or deleted setup cannot make this record dangle.
    model_setup_id: str | None = None
    model: str = ""
    reasoning: str | None = None
    model_variable: str | None = None
    classification: ReviewFixComplexity | None = None
    failure_classification: FailureClassification | None = None
    repair_scope: RepairScope | None = None
    attempt: int = Field(default=0, ge=0)
    finding_ids: list[str] = Field(default_factory=list)
    finding_fingerprints: list[str] = Field(default_factory=list)
    repository_revision: str | None = None
    routing_input_fingerprint: str = ""
    input_fingerprint: str = ""
    routing_reason: str = Field(min_length=1)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def execution_mode_must_be_known(cls, value: ModelExecutionMode | str) -> ModelExecutionMode:
        """Rehydrate the execution mode from persisted JSON."""
        return value if isinstance(value, ModelExecutionMode) else ModelExecutionMode(value)

    @field_validator("platform", mode="before")
    @classmethod
    def platform_must_be_known(cls, value: AgentPlatform | str | None) -> AgentPlatform:
        """Rehydrate the provider, reading a record written before it existed as OpenAI."""
        if value is None:
            return AgentPlatform.OPENAI
        return value if isinstance(value, AgentPlatform) else AgentPlatform(value)

    @field_validator("performance_tier", mode="before")
    @classmethod
    def performance_tier_must_be_known(cls, value: PerformanceTier | str | None) -> PerformanceTier:
        """Rehydrate the tier, reading a record written before it existed as `high`."""
        if value is None:
            return PerformanceTier.HIGH
        return value if isinstance(value, PerformanceTier) else PerformanceTier(value)

    @field_validator("role", mode="before")
    @classmethod
    def role_must_be_known(cls, value: ModelRole | str) -> ModelRole:
        """Rehydrate an active role, or the active equivalent of a retired one."""
        if isinstance(value, ModelRole):
            return value
        retired = _RETIRED_ROLE_NAMES.get(value)
        return retired if retired is not None else ModelRole(value)

    @field_validator("classification", mode="before")
    @classmethod
    def classification_must_be_known(
        cls, value: ReviewFixComplexity | str | None
    ) -> ReviewFixComplexity | None:
        """Rehydrate a finding complexity from persisted JSON."""
        if value is None or isinstance(value, ReviewFixComplexity):
            return value
        return ReviewFixComplexity(value)

    @field_validator("failure_classification", mode="before")
    @classmethod
    def failure_must_be_known(
        cls, value: FailureClassification | str | None
    ) -> FailureClassification | None:
        """Rehydrate the upstream failure class when a newer record carries it."""
        if value is None or isinstance(value, FailureClassification):
            return value
        return FailureClassification(value)

    @field_validator("repair_scope", mode="before")
    @classmethod
    def repair_scope_must_be_known(cls, value: RepairScope | str | None) -> RepairScope | None:
        """Rehydrate the conservative repair-scope verdict."""
        if value is None or isinstance(value, RepairScope):
            return value
        return RepairScope(value)

    @classmethod
    def from_persisted(cls, value: object) -> ModelRoutingDecision | None:
        """Rehydrate a decision, or return nothing for a child written before routing."""
        if isinstance(value, ModelRoutingDecision):
            return value
        if not isinstance(value, dict):
            return None
        return cls.model_validate(
            {key: item for key, item in value.items() if key not in _RETIRED_DECISION_FIELDS}
        )


class ModelRoutingOutcome(StateModel):
    """The resolved selection for one attempt."""

    decision: ModelRoutingDecision
    reused: bool = False


_SCOPED_FAILURES = frozenset(
    {
        FailureClassification.VALIDATION_SOURCE_FAILURE,
        FailureClassification.REVIEW_SCOPE_FAILURE,
    }
)


# One effort step per level. `low` has no step below it and `none` means disabled
# thinking, so neither degrades: for both, a truncation stays terminal. The vocabulary is
# the platform's (configs.model_roles normalizes it); the ladder lives here because
# stepping a selection down is a routing decision, not an adapter one.
_REASONING_STEP_DOWN = {
    "max": "xhigh",
    "xhigh": "high",
    "high": "medium",
    "medium": "low",
}

_TRUNCATION_RETRY_REASON = (
    "The previous response reached its configured max_tokens bound before finishing; the "
    "same selection is retried once at reduced reasoning effort -- the one request change "
    "the truncation diagnostic itself prescribes."
)


def degraded_truncation_retry(
    decision: ModelRoutingDecision | None,
) -> ModelRoutingDecision | None:
    """Return this selection one effort level lower, or nothing when no step remains.

    A truncated response is a deterministic provider answer, so it is never retried
    verbatim -- `is_transient_provider_fault` still refuses it, and that rule is untouched.
    What a truncation does prescribe is its own remedy: thinking shares the response's
    ``max_tokens`` bound, so on a bound already at the model's output ceiling (128000 was
    both the configured bound and claude-sonnet-5's ceiling when AB-Feature-181's coding
    call spent all of it at `xhigh`), lowering the effort is the only request change left.
    This returns the persisted selection with exactly that change applied, for the caller
    to persist and run once; a second truncation finds the fault terminal as before.

    Everything else on the decision is kept, the attempt number included: this is the same
    attempt asking the same question with a smaller thinking budget, not a new attempt, so
    it must not consume retry budget or disturb artifact lineage.
    """
    if decision is None or not decision.reasoning:
        return None
    stepped = _REASONING_STEP_DOWN.get(decision.reasoning)
    if stepped is None:
        return None
    return decision.model_copy(
        update={"reasoning": stepped, "routing_reason": _TRUNCATION_RETRY_REASON}
    )


def determine_repair_scope(
    classifications: Sequence[ReviewFixClassification],
) -> RepairScope:
    """Conservatively reduce classified findings to one repair scope.

    Every finding must be deterministically recognized as mechanical. A mixed set, any
    substantive finding, or any unplaced wording keeps the whole attempt on the primary coding
    model because the current workflow executes the findings as one task.
    """
    if not classifications or any(not item.deterministic for item in classifications):
        return RepairScope.AMBIGUOUS
    if all(item.classification is ReviewFixComplexity.STANDARD for item in classifications):
        return RepairScope.SCOPED
    return RepairScope.SUBSTANTIVE


class ModelRouter:
    """Resolve the configured model for one already-eligible engineering attempt."""

    def __init__(
        self,
        model_configs: ModelConfigService | None = None,
        *,
        platform: AgentPlatform = AgentPlatform.OPENAI,
    ) -> None:
        """Bind the role configuration service and the platform this feature runs on.

        The platform is the feature's, not the deployment's. A router built for one feature
        resolves every role it selects against that feature's provider, so the decision it
        persists names the model that will actually execute the attempt.
        """
        self._model_configs = model_configs
        self._platform = platform

    def route(
        self,
        *,
        inputs: ModelRoutingInputs,
        classifications: Sequence[ReviewFixClassification] = (),
        persisted_decision: object = None,
    ) -> ModelRoutingOutcome:
        """Select a role without deciding whether another attempt may exist."""
        question = inputs.fingerprint()
        existing = ModelRoutingDecision.from_persisted(persisted_decision)
        if existing is not None and existing.routing_input_fingerprint == question:
            return ModelRoutingOutcome(decision=existing, reused=True)
        if inputs.execution_mode is ModelExecutionMode.INITIAL_IMPLEMENTATION:
            return self._coding(
                inputs,
                question,
                classifications=(),
                scope=None,
                reason="Initial repository implementation uses the configured coding role.",
                execution_mode=ModelExecutionMode.INITIAL_IMPLEMENTATION,
            )

        scope = determine_repair_scope(classifications)
        failure = inputs.failure_classification
        scoped = (
            failure in _SCOPED_FAILURES
            and failure is not None
            and requires_feature_engineer(failure)
            and scope is RepairScope.SCOPED
        )
        if scoped:
            return self._outcome(
                inputs=inputs,
                question=question,
                classifications=classifications,
                scope=scope,
                role=ModelRole.SCOPED_FIX,
                reason=(
                    "The failure is engineer-remediable and every blocking finding is a "
                    "localized, independently verifiable mechanical correction."
                ),
                execution_mode=ModelExecutionMode.REVIEW_REMEDIATION,
            )

        if failure is not None and not requires_feature_engineer(failure):
            reason = (
                f"{failure.value} belongs to the existing non-engineer remediation path; "
                "no scoped-fix model is selected."
            )
            mode = ModelExecutionMode.INITIAL_IMPLEMENTATION
        elif failure not in _SCOPED_FAILURES:
            reason = (
                "The classified failure is not eligible for a mechanical scoped fix, so the "
                "primary coding role retains it."
            )
            mode = ModelExecutionMode.REVIEW_REMEDIATION
        elif scope is RepairScope.AMBIGUOUS:
            reason = "The repair scope is ambiguous, so the primary coding role retains the work."
            mode = ModelExecutionMode.REVIEW_REMEDIATION
        else:
            reason = (
                "The findings include substantive or mixed work, so the primary coding role "
                "retains the whole remediation."
            )
            mode = ModelExecutionMode.REVIEW_REMEDIATION
        return self._coding(
            inputs,
            question,
            classifications=classifications,
            scope=scope,
            reason=reason,
            execution_mode=mode,
        )

    def _coding(
        self,
        inputs: ModelRoutingInputs,
        question: str,
        *,
        classifications: Sequence[ReviewFixClassification],
        scope: RepairScope | None,
        reason: str,
        execution_mode: ModelExecutionMode,
    ) -> ModelRoutingOutcome:
        """Resolve the conservative primary-coding branch."""
        return self._outcome(
            inputs=inputs,
            question=question,
            classifications=classifications,
            scope=scope,
            role=ModelRole.CODING,
            reason=reason,
            execution_mode=execution_mode,
        )

    def _outcome(
        self,
        *,
        inputs: ModelRoutingInputs,
        question: str,
        classifications: Sequence[ReviewFixClassification],
        scope: RepairScope | None,
        role: ModelRole,
        reason: str,
        execution_mode: ModelExecutionMode,
    ) -> ModelRoutingOutcome:
        """Build the durable selection for one attempt."""
        # The platform this role actually resolves on. For every environment-backed service
        # `platform_for_role` answers nothing and the feature's own platform stands, byte for
        # byte the behaviour this router always had. A role-addressed service -- a custom
        # setup -- pins each role to its own provider, and both the lookup and the persisted
        # decision must name that one: a scoped fix pinned to Claude recorded as `openai`
        # would rebuild its client against the wrong API on recovery. The routing decisions
        # themselves -- which role answers -- are untouched.
        platform = (
            self._model_configs.platform_for_role(role) if self._model_configs is not None else None
        ) or self._platform
        config = (
            self._model_configs.get_model_config(role, platform=platform)
            if self._model_configs is not None
            else None
        )
        classification = (
            highest_complexity(item.classification for item in classifications)
            if classifications
            else None
        )
        decision = ModelRoutingDecision(
            execution_mode=execution_mode,
            role=role,
            platform=platform,
            model=config.model if config is not None else "",
            reasoning=config.reasoning if config is not None else None,
            model_variable=config.model_variable if config is not None else None,
            classification=classification,
            failure_classification=inputs.failure_classification,
            repair_scope=scope,
            attempt=inputs.attempt,
            finding_ids=[item.finding_id for item in classifications if item.finding_id],
            finding_fingerprints=sorted(inputs.finding_fingerprints),
            repository_revision=inputs.repository_revision,
            routing_input_fingerprint=question,
            # A role or model change alone is never new input and cannot bypass an operation
            # journal receipt or a retry fingerprint.
            input_fingerprint=question,
            routing_reason=reason,
        )
        return ModelRoutingOutcome(decision=decision)


__all__ = [
    "ModelExecutionMode",
    "ModelRouter",
    "ModelRoutingDecision",
    "ModelRoutingInputs",
    "ModelRoutingOutcome",
    "RepairScope",
    "degraded_truncation_retry",
    "determine_repair_scope",
]
