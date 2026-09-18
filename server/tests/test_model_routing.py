"""Regression coverage for four-role model routing and retry-safe scoped fixes."""

from __future__ import annotations

from typing import Any

import pytest

from adapters.llm_adapter import LLMAdapterError
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import FileChange, ReviewArtifact, ReviewFinding
from configs.model_roles import (
    MODEL_ROLE_VARIABLES,
    REASONING_LEVELS,
    AgentPlatform,
    ModelConfigService,
    ModelConfigurationError,
    ModelRole,
    PlatformModelSettings,
    build_model_config_service,
    normalize_reasoning_level,
    supported_reasoning,
)
from configs.settings import load_settings
from state.enums import ChildWorkflowStatus
from storage.external_operation_store import ExternalOperationError
from tools.model_routing import (
    ModelExecutionMode,
    ModelRouter,
    ModelRoutingDecision,
    ModelRoutingInputs,
    RepairScope,
    degraded_truncation_retry,
    determine_repair_scope,
)
from tools.retry_strategy import (
    FailureClassification,
    RemediationOwner,
    decide_child_retry,
    remediation_owner,
    requires_feature_engineer,
)
from tools.review_fix_classification import (
    ReviewFixClassifierVerdict,
    ReviewFixComplexity,
    apply_classifier_verdict,
    classify_blocking_issue,
    classify_findings,
    classify_review_finding,
    finding_fingerprint,
)
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowOrchestrator,
    MockChildWorkstreamExecutor,
)

_TEST_MODELS = {
    ModelRole.REASONING: "test-reasoning-model",
    ModelRole.CODING: "test-coding-model",
    ModelRole.REVIEW: "test-review-model",
    ModelRole.SCOPED_FIX: "test-scoped-fix-model",
}
_TEST_REASONING: dict[ModelRole, str | None] = {
    ModelRole.REASONING: "max",
    ModelRole.CODING: "max",
    ModelRole.REVIEW: "high",
    ModelRole.SCOPED_FIX: "high",
}


_TEST_MAX_TOKENS: dict[ModelRole, int | None] = {
    ModelRole.REASONING: 32_000,
    ModelRole.CODING: 64_000,
    ModelRole.REVIEW: 16_000,
    ModelRole.SCOPED_FIX: 16_000,
}


def model_configs(
    *,
    models: dict[ModelRole, str] | None = None,
    platform: AgentPlatform = AgentPlatform.OPENAI,
) -> ModelConfigService:
    """Build a complete central role configuration with distinct test model names."""
    return build_model_config_service(
        platforms={
            platform: PlatformModelSettings(
                models=models or _TEST_MODELS,
                reasoning=_TEST_REASONING,
                max_tokens=_TEST_MAX_TOKENS,
            )
        }
    )


def router(
    *,
    models: dict[ModelRole, str] | None = None,
    platform: AgentPlatform = AgentPlatform.OPENAI,
) -> ModelRouter:
    """Build the production router over one central configuration service."""
    return ModelRouter(model_configs(models=models, platform=platform), platform=platform)


def routing_inputs(**overrides: Any) -> ModelRoutingInputs:
    """Build a classified remediation question with only the fields a test varies."""
    values: dict[str, Any] = {
        "feature_id": "feature-routing",
        "repository_id": "backend",
        "execution_mode": ModelExecutionMode.REVIEW_REMEDIATION,
        "attempt": 1,
        "failure_classification": FailureClassification.REVIEW_SCOPE_FAILURE,
        "repository_revision": "rev-a",
    }
    values.update(overrides)
    return ModelRoutingInputs.model_validate(values)


def finding(**overrides: Any) -> ReviewFinding:
    """Build one structured reviewer finding with neutral fixture prose."""
    values: dict[str, Any] = {
        "finding_id": "REV-001",
        "severity": "high",
        "title": "Reviewer finding",
        "description": "There is an unused import in this file.",
        "recommendation": "Correct only the reported issue.",
        "file_path": "server/app.js",
        "line_number": 3,
        "finding_category": "code_quality",
    }
    values.update(overrides)
    return ReviewFinding.model_validate(values)


def feature_payload(feature_id: str, *, repositories: int = 1) -> dict[str, Any]:
    """Return a minimal feature request for deterministic orchestrator coverage."""
    names = ["backend", "frontend", "shared"][:repositories]
    return {
        "feature_id": feature_id,
        "prd": {
            "title": "Model routing",
            "problem_statement": "Different corrections need different model roles.",
            "goals": ["Route corrections only after classifying them."],
            "user_stories": [
                {
                    "story_id": "story-routing",
                    "persona": "Operator",
                    "need": "See which model corrected what",
                    "benefit": "Execution choices remain auditable",
                    "acceptance_criteria": ["Every routing decision is recorded."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": "requirement-routing",
                    "description": "Record the resolved model role for engineering work.",
                    "priority": "must",
                    "acceptance_criteria": ["Routing metadata is persisted."],
                    "dependencies": [],
                }
            ],
            "constraints": ["Use deterministic dependencies."],
            "out_of_scope": ["Manual model selection."],
            "stakeholders": ["Platform"],
        },
        "repositories": [
            {
                "repository_id": name,
                "name": name.title(),
                "role": name,
                "repository_url": f"https://github.com/example/{name}.git",
                "default_branch": "main",
            }
            for name in names
        ],
    }


def no_credentials() -> RequestScopedCredentials:
    """Return the empty credential envelope accepted by deterministic execution."""
    return RequestScopedCredentials(openai_api_key=None, github_token=None)


class RoutedFailureExecutor:
    """Fail one repository with fixed findings and record each persisted routing decision."""

    def __init__(
        self,
        repository_id: str,
        *,
        findings: list[ReviewFinding],
        failure_classification: str = "validation_source_failure",
        attempts_before_approval: int | None = None,
    ) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._repository_id = repository_id
        self._findings = findings
        self._failure_classification = failure_classification
        self._attempts_before_approval = attempts_before_approval
        self.attempts = 0
        self.routing: list[ModelRoutingDecision | None] = []
        self.routing_by_repository: dict[str, list[ModelRoutingDecision | None]] = {}

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Record the attempt and either return the delegate's approval or a changed failure."""
        child = kwargs["child"]
        repository_id = kwargs["repository"].repository_id
        decision = ModelRoutingDecision.from_persisted(child.model_routing)
        self.routing_by_repository.setdefault(repository_id, []).append(decision)
        execution = await self._delegate.run(**kwargs)
        if repository_id != self._repository_id:
            return execution
        self.attempts += 1
        self.routing.append(decision)
        if (
            self._attempts_before_approval is not None
            and self.attempts > self._attempts_before_approval
        ):
            return execution
        touched = [f"server/module{self.attempts}.js", "server/app.js"]
        completion = execution.code_completion
        review = execution.review
        assert completion is not None
        assert review is not None
        completion = completion.model_copy(
            update={
                "file_changes": [
                    FileChange(path=path, change_type="modified", description="attempt output")
                    for path in touched
                ],
                "production_files_changed": touched,
                "commit_sha": None,
            }
        )
        rejected = review.model_copy(
            update={"verdict": "changes_requested", "findings": self._findings}
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [item.description for item in self._findings],
                    "pull_request_readiness": False,
                    "production_files_changed": touched,
                    "failure_classification": self._failure_classification,
                    "production_diff_fingerprint": f"production-{self.attempts}",
                    "test_diff_fingerprint": "tests-unchanged",
                    # A deterministic gate may repeat while source continues changing. This
                    # keeps the test focused on bounded retry/model routing, not prose guards.
                    "metadata": {**execution.result.metadata, "deterministic_gate": True},
                    "review_artifact_id": rejected.artifact_id,
                }
            ),
            code_completion=completion,
            review=rejected,
        )


# -------------------------------------------------------------------------------------- config


def test_exactly_four_model_roles_resolve_independently() -> None:
    """The active public vocabulary is REASONING, CODING, REVIEW and SCOPED_FIX."""
    assert tuple(ModelRole) == (
        ModelRole.REASONING,
        ModelRole.CODING,
        ModelRole.REVIEW,
        ModelRole.SCOPED_FIX,
    )
    configs = model_configs()
    for role in ModelRole:
        resolved = configs.get_model_config(role, platform=AgentPlatform.OPENAI)
        assert resolved.model == _TEST_MODELS[role]
        assert resolved.reasoning == _TEST_REASONING[role]


def test_required_environment_names_resolve_the_intended_defaults() -> None:
    """All requested model names and effort levels remain deployment configuration."""
    settings = load_settings(
        openai_reasoning_model="gpt-5.6-sol",
        openai_reasoning_effort="max",
        openai_coding_model="gpt-5.6-sol",
        openai_coding_reasoning_effort="max",
        openai_review_model="gpt-5.6-terra",
        openai_review_reasoning_effort="high",
        openai_scoped_fix_model="gpt-5.3-codex",
        openai_scoped_fix_reasoning_effort="high",
    )

    openai = AgentPlatform.OPENAI
    assert settings.model_config_for_role(ModelRole.REASONING, platform=openai).model == (
        "gpt-5.6-sol"
    )
    assert settings.model_config_for_role(ModelRole.REASONING, platform=openai).reasoning == "max"
    assert settings.model_config_for_role(ModelRole.CODING, platform=openai).model == "gpt-5.6-sol"
    assert settings.model_config_for_role(ModelRole.CODING, platform=openai).reasoning == "max"
    assert settings.model_config_for_role(ModelRole.REVIEW, platform=openai).model == (
        "gpt-5.6-terra"
    )
    assert settings.model_config_for_role(ModelRole.SCOPED_FIX, platform=openai).model == (
        "gpt-5.3-codex"
    )


def test_an_environment_with_only_the_two_original_models_still_loads() -> None:
    """Review and scoped-fix fall back without silently choosing an unrelated third model."""
    settings = load_settings(
        openai_reasoning_model="legacy-reasoning",
        openai_coding_model="legacy-coding",
        openai_review_model="",
        openai_scoped_fix_model="",
        openai_fix_model="",
        openai_review_reasoning_effort="",
        openai_scoped_fix_reasoning_effort="",
    )

    openai = AgentPlatform.OPENAI
    assert settings.model_config_for_role(ModelRole.REASONING, platform=openai).model == (
        "legacy-reasoning"
    )
    assert settings.model_config_for_role(ModelRole.CODING, platform=openai).model == (
        "legacy-coding"
    )
    assert settings.model_config_for_role(ModelRole.REVIEW, platform=openai).model == (
        "legacy-reasoning"
    )
    assert settings.model_config_for_role(ModelRole.SCOPED_FIX, platform=openai).model == (
        "legacy-coding"
    )
    # Reported per platform now that two providers can each be half-migrated.
    assert set(settings.model_configs.roles_resolved_from_legacy_configuration()) == {
        (openai, ModelRole.REVIEW),
        (openai, ModelRole.SCOPED_FIX),
    }


def test_missing_and_invalid_role_configuration_fails_at_startup_by_variable_name() -> None:
    """Safe startup diagnostics identify configuration keys and never quote rejected values."""
    openai = AgentPlatform.OPENAI
    # Every platform empty is no usable deployment at all, and is named as one.
    with pytest.raises(ModelConfigurationError) as missing:
        build_model_config_service(
            platforms={
                openai: PlatformModelSettings(
                    models={role: "" for role in ModelRole},
                    reasoning=dict.fromkeys(ModelRole),
                    max_tokens=dict.fromkeys(ModelRole),
                )
            }
        )
    assert MODEL_ROLE_VARIABLES[(openai, ModelRole.REASONING)].model_variable in str(missing.value)

    # Some-but-not-all is a configuration mistake, and names the role that is missing.
    with pytest.raises(ModelConfigurationError) as partial:
        build_model_config_service(
            platforms={
                openai: PlatformModelSettings(
                    models={**_TEST_MODELS, ModelRole.SCOPED_FIX: ""},
                    reasoning=_TEST_REASONING,
                    max_tokens=_TEST_MAX_TOKENS,
                )
            }
        )
    assert MODEL_ROLE_VARIABLES[(openai, ModelRole.SCOPED_FIX)].model_variable in str(partial.value)

    with pytest.raises(ModelConfigurationError) as invalid:
        build_model_config_service(
            platforms={
                openai: PlatformModelSettings(
                    models=_TEST_MODELS,
                    reasoning={**_TEST_REASONING, ModelRole.SCOPED_FIX: "very-high"},
                    max_tokens=_TEST_MAX_TOKENS,
                )
            }
        )
    assert MODEL_ROLE_VARIABLES[(openai, ModelRole.SCOPED_FIX)].reasoning_variable in str(
        invalid.value
    )
    assert "very-high" not in str(invalid.value)


def test_reasoning_capability_normalization_is_central_and_never_swaps_models() -> None:
    """A deployment-declared unsupported effort is omitted in the one config layer."""
    configs = build_model_config_service(
        platforms={
            AgentPlatform.OPENAI: PlatformModelSettings(
                models=_TEST_MODELS,
                reasoning=_TEST_REASONING,
                max_tokens=_TEST_MAX_TOKENS,
            )
        },
        unsupported_reasoning={_TEST_MODELS[ModelRole.SCOPED_FIX]: ["high"]},
    )
    resolved = configs.get_model_config(ModelRole.SCOPED_FIX, platform=AgentPlatform.OPENAI)
    assert resolved.model == _TEST_MODELS[ModelRole.SCOPED_FIX]
    assert resolved.reasoning is None
    assert resolved.requested_reasoning == "high"
    assert supported_reasoning("m", "max", {"m": frozenset({"max"})}) is None
    for level in REASONING_LEVELS:
        assert normalize_reasoning_level("VAR", level) == level


def test_reasoning_coding_and_review_agents_are_bound_to_logical_roles() -> None:
    """Model invocation sites consume roles instead of embedding model identifiers."""
    settings = load_settings(openai_coding_model="c", openai_reasoning_model="r")
    assert settings.agents["product_manager"].model_role is ModelRole.REASONING
    assert settings.agents["planner"].model_role is ModelRole.REASONING
    assert settings.agents["engineer"].model_role is ModelRole.CODING
    assert settings.agents["reviewer"].model_role is ModelRole.REVIEW


# ----------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "description",
    [
        "Module not found: ../utils/foo in this file.",
        "Cannot resolve import '../utils/foo' in this module.",
        "Cannot resolve ../utils/foo during this build.",
        "A simple circular import introduced by recent change breaks this module.",
        "ESLint no-unused-vars at src/app.ts:3.",
        "ESLint reports one simple rule violation in this file.",
        "Remove the unused import from this file.",
        "Prettier reports a formatting failure in this file.",
        "Property 'status' is missing after a local interface rename.",
        "The recent change has a single interface mismatch.",
        "The small generic mismatch is confined to this module.",
        "This build uses an incorrect path in one import.",
        "The package has a missing small dependency declaration.",
        "A single broken assertion has the old deterministic fixture value.",
    ],
)
def test_known_localized_failures_are_classified_as_scoped(description: str) -> None:
    """Imports, lint, bounded types and deterministic tests are scoped candidates."""
    classified = classify_review_finding(finding(description=description))
    assert classified.classification is ReviewFixComplexity.STANDARD
    assert classified.deterministic is True
    assert determine_repair_scope((classified,)) is RepairScope.SCOPED


@pytest.mark.parametrize(
    "description",
    [
        "The authentication flow is architecturally incorrect.",
        "The backend and frontend api contract disagree.",
        "Tests show the business logic is incorrect for cancelled accounts.",
        "The database schema requires a migration and redesign.",
        "Please look at this again.",
        "A type error occurs somewhere in the feature.",
        "The response format needs to be reconsidered.",
        "We cannot resolve which product behavior is intended.",
    ],
)
def test_substantive_or_ambiguous_findings_stay_off_scoped_fix(description: str) -> None:
    """System behavior and uncertain prose never become cheap mechanical work."""
    classified = classify_review_finding(finding(description=description))
    assert classified.classification in {
        ReviewFixComplexity.COMPLEX,
        ReviewFixComplexity.CRITICAL,
    }
    assert determine_repair_scope((classified,)) in {
        RepairScope.SUBSTANTIVE,
        RepairScope.AMBIGUOUS,
    }


def test_mixed_mechanical_and_substantive_findings_are_substantive_as_one_task() -> None:
    """The workflow does not decompose findings, so the hardest member decides the scope."""
    classifications = classify_findings(
        [
            finding(description="Remove the unused import from this file."),
            finding(
                finding_id="REV-002",
                description="The authentication flow is architecturally incorrect.",
            ),
        ]
    )
    assert determine_repair_scope(classifications) is RepairScope.SUBSTANTIVE


def test_structured_security_and_contract_categories_override_mechanical_words() -> None:
    """Structured review evidence is preferred over a superficially small recommendation."""
    security = classify_review_finding(
        finding(finding_category="security", description="Remove the unused import.")
    )
    contract = classify_review_finding(
        finding(
            finding_category="contract",
            description="Rename this imported field.",
            requirement_id="REQ-1",
            contract_reference="listApps",
            repository_id="backend",
            responsibility="implements",
            validated_revision="rev-a",
            evidence="The consumer reads a field the producer does not send.",
        )
    )
    assert security.classification is ReviewFixComplexity.CRITICAL
    assert contract.classification is ReviewFixComplexity.COMPLEX


def test_bare_diagnostics_use_the_same_conservative_classifier() -> None:
    """Validation text and structured review findings share one repair-scope vocabulary."""
    assert (
        classify_blocking_issue("Cannot resolve import '../utils/foo'.").classification
        is ReviewFixComplexity.STANDARD
    )
    ambiguous = classify_blocking_issue("The behavior is wrong.")
    assert ambiguous.classification is ReviewFixComplexity.COMPLEX
    assert ambiguous.deterministic is False


def test_low_confidence_classifier_output_cannot_select_scoped_fix() -> None:
    """An optional classifier may raise confidence requirements but never route uncertainty down."""
    deterministic = classify_review_finding(finding(description="Please look at this again."))
    verdict = ReviewFixClassifierVerdict.model_validate(
        {"classification": "STANDARD", "confidence": 0.4, "reason": "Probably small."}
    )
    resolved = apply_classifier_verdict(deterministic, verdict, minimum_confidence=0.8)
    assert resolved.classification is ReviewFixComplexity.COMPLEX
    assert determine_repair_scope((resolved,)) is RepairScope.AMBIGUOUS


def test_reworded_diagnostic_fingerprints_remain_stable_for_convergence() -> None:
    """Model routing keeps the retry system's defect identity rather than inventing another."""
    first = finding(description="TypeError at app.route.js:117 while reading status.")
    reworded = finding(
        finding_id="REV-009",
        description="At app.route.js:117 reading status raises TypeError.",
    )
    assert finding_fingerprint(first) == finding_fingerprint(reworded)


# ------------------------------------------------------------------------------------ routing


def _route(description: str, **input_overrides: Any) -> ModelRoutingDecision:
    classifications = classify_findings([finding(description=description)])
    return (
        router()
        .route(
            inputs=routing_inputs(
                finding_fingerprints=[item.fingerprint for item in classifications],
                **input_overrides,
            ),
            classifications=classifications,
        )
        .decision
    )


def test_initial_implementation_routes_to_primary_coding() -> None:
    outcome = router().route(
        inputs=routing_inputs(
            execution_mode=ModelExecutionMode.INITIAL_IMPLEMENTATION,
            attempt=0,
            failure_classification=None,
        )
    )
    assert outcome.decision.role is ModelRole.CODING
    assert outcome.decision.model == _TEST_MODELS[ModelRole.CODING]
    assert outcome.decision.repair_scope is None


@pytest.mark.parametrize(
    "description",
    [
        "Cannot resolve import '../utils/foo' in this file.",
        "ESLint no-unused-vars at src/app.ts:3.",
        "Property 'status' is missing after a local interface rename.",
    ],
)
def test_localized_validation_or_review_failure_routes_to_scoped_fix(description: str) -> None:
    decision = _route(description)
    assert decision.role is ModelRole.SCOPED_FIX
    assert decision.model == _TEST_MODELS[ModelRole.SCOPED_FIX]
    assert decision.reasoning == _TEST_REASONING[ModelRole.SCOPED_FIX]
    assert decision.repair_scope is RepairScope.SCOPED


@pytest.mark.parametrize(
    "description",
    [
        "The authentication flow is architecturally incorrect.",
        "Tests show the business logic is incorrect for cancelled accounts.",
        "Please look at this again.",
    ],
)
def test_substantive_or_ambiguous_remediation_routes_to_primary_coding(
    description: str,
) -> None:
    decision = _route(description)
    assert decision.role is ModelRole.CODING
    assert decision.model == _TEST_MODELS[ModelRole.CODING]
    assert decision.repair_scope is not RepairScope.SCOPED


def test_mixed_findings_route_to_primary_coding() -> None:
    classifications = classify_findings(
        [
            finding(description="Remove the unused import from this file."),
            finding(
                finding_id="REV-002",
                description="The service boundary is architecturally incorrect.",
            ),
        ]
    )
    decision = (
        router()
        .route(
            inputs=routing_inputs(
                finding_fingerprints=[item.fingerprint for item in classifications]
            ),
            classifications=classifications,
        )
        .decision
    )
    assert decision.role is ModelRole.CODING
    assert decision.repair_scope is RepairScope.SUBSTANTIVE


@pytest.mark.parametrize(
    "failure",
    [
        FailureClassification.IMPLEMENTATION_MISSING,
        FailureClassification.CONTRACT_MISMATCH,
    ],
)
def test_non_mechanical_engineer_failure_classes_stay_on_primary_coding(
    failure: FailureClassification,
) -> None:
    decision = _route("Remove the unused import from this file.", failure_classification=failure)
    assert decision.role is ModelRole.CODING


def test_repository_configuration_failure_never_selects_scoped_fix() -> None:
    """A lint-shaped setup problem remains owned by repository repair."""
    decision = _route(
        "ESLint configuration references a dependency that is not installed.",
        failure_classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
    )
    assert decision.role is ModelRole.CODING
    assert decision.execution_mode is ModelExecutionMode.INITIAL_IMPLEMENTATION
    assert "non-engineer remediation path" in decision.routing_reason

    # Even if a flattened review failure reaches the ordinary review-scope class, wording that
    # identifies existing tooling configuration is not treated as a local lint edit.
    review_decision = _route(
        "eslint-config-airbnb-base is not installed in the repository setup.",
        failure_classification=FailureClassification.REVIEW_SCOPE_FAILURE,
    )
    assert review_decision.role is ModelRole.CODING
    assert review_decision.repair_scope is RepairScope.SUBSTANTIVE


@pytest.mark.parametrize(
    ("classification", "owner"),
    [
        (FailureClassification.IMPLEMENTATION_MISSING, RemediationOwner.REQUIRES_FEATURE_ENGINEER),
        (
            FailureClassification.VALIDATION_SOURCE_FAILURE,
            RemediationOwner.REQUIRES_FEATURE_ENGINEER,
        ),
        (FailureClassification.REVIEW_SCOPE_FAILURE, RemediationOwner.REQUIRES_FEATURE_ENGINEER),
        (FailureClassification.CONTRACT_MISMATCH, RemediationOwner.REQUIRES_FEATURE_ENGINEER),
        (
            FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
            RemediationOwner.AUTO_REPAIRABLE_REPOSITORY,
        ),
        (
            FailureClassification.TEST_INFRASTRUCTURE_MISSING,
            RemediationOwner.AUTO_REPAIRABLE_REPOSITORY,
        ),
        (FailureClassification.DEPENDENCY_INSTALLATION_FAILURE, RemediationOwner.INFRASTRUCTURE),
        (FailureClassification.VALIDATION_CAPACITY_FAILURE, RemediationOwner.INFRASTRUCTURE),
    ],
)
def test_failure_owner_is_decided_before_any_model(
    classification: FailureClassification, owner: RemediationOwner
) -> None:
    assert remediation_owner(classification) is owner
    assert requires_feature_engineer(classification) is (
        owner is RemediationOwner.REQUIRES_FEATURE_ENGINEER
    )


# ------------------------------------------------------------------------------- retry safety


def test_changing_only_configured_model_does_not_change_input_fingerprint() -> None:
    """A deployment edit cannot turn identical work into meaningful progress."""
    classifications = classify_findings(
        [finding(description="Remove the unused import from this file.")]
    )
    inputs = routing_inputs(finding_fingerprints=[item.fingerprint for item in classifications])
    first = router().route(inputs=inputs, classifications=classifications).decision
    changed_models = {
        **_TEST_MODELS,
        ModelRole.SCOPED_FIX: "a-different-scoped-model",
    }
    changed = (
        router(models=changed_models).route(inputs=inputs, classifications=classifications).decision
    )
    assert first.model != changed.model
    assert first.routing_input_fingerprint == changed.routing_input_fingerprint
    assert first.input_fingerprint == changed.input_fingerprint


def test_model_availability_cannot_make_an_exhausted_retry_eligible() -> None:
    """The router has no retry verdict, and the existing budget remains authoritative."""
    exhausted = decide_child_retry(
        classification=FailureClassification.VALIDATION_SOURCE_FAILURE,
        attempt_count=8,
        budget=8,
        retry_count=8,
        max_child_review_cycles=12,
        meaningful_change=True,
    )
    assert exhausted.should_retry is False
    outcome = router().route(inputs=routing_inputs(), classifications=())
    assert not hasattr(outcome, "should_retry")
    assert set(outcome.model_dump()) == {"decision", "reused"}


def test_repeated_failure_never_cycles_models() -> None:
    """Attempts may repeat only when retry policy allows; routing stays classification-based."""
    classifications = classify_findings(
        [finding(description="Remove the unused import from this file.")]
    )
    decisions = [
        router()
        .route(
            inputs=routing_inputs(
                attempt=attempt,
                repository_revision=f"rev-{attempt}",
                finding_fingerprints=[item.fingerprint for item in classifications],
            ),
            classifications=classifications,
        )
        .decision
        for attempt in range(1, 4)
    ]
    assert [item.role for item in decisions] == [ModelRole.SCOPED_FIX] * 3


def test_exact_same_question_reuses_its_persisted_selection() -> None:
    classifications = classify_findings(
        [finding(description="Remove the unused import from this file.")]
    )
    inputs = routing_inputs(finding_fingerprints=[item.fingerprint for item in classifications])
    first = router().route(inputs=inputs, classifications=classifications)
    resumed = router(models={**_TEST_MODELS, ModelRole.SCOPED_FIX: "changed"}).route(
        inputs=inputs,
        classifications=classifications,
        persisted_decision=first.decision.model_dump(mode="json"),
    )
    assert resumed.reused is True
    assert resumed.decision == first.decision


def test_a_record_written_by_the_retired_ladder_still_loads() -> None:
    """Deleting the ladder must not make an existing durable decision unreadable.

    Every field here was written by the tiered router that this module replaced, and
    `StateModel` forbids undeclared ones -- so without the read-time translation the two
    production rows carrying them would raise instead of loading.
    """
    historical = ModelRoutingDecision.from_persisted(
        {
            "execution_mode": "REVIEW_REMEDIATION",
            "role": "escalation",
            "model": "historical-model",
            "reasoning": "high",
            "attempt": 2,
            "escalation_level": 2,
            "escalated": True,
            "previous_role": "complex_fix",
            "routing_reason": "Historical escalated remediation.",
        }
    )

    assert historical is not None
    # Read as the role that would execute the same work today, which is the mapping both
    # consumers of a persisted role already applied before acting on one.
    assert historical.role is ModelRole.CODING
    assert historical.routing_reason == "Historical escalated remediation."
    assert "escalated" not in historical.model_dump(mode="json")


def test_a_retired_scoped_role_name_loads_as_the_scoped_fix_role() -> None:
    """The other half of the retired vocabulary, so the mapping is not asserted one-sided."""
    historical = ModelRoutingDecision.from_persisted(
        {
            "execution_mode": "REVIEW_REMEDIATION",
            "role": "fix",
            "model": "historical-model",
            "attempt": 1,
            "routing_reason": "Historical standard remediation.",
        }
    )

    assert historical is not None
    assert historical.role is ModelRole.SCOPED_FIX


# --------------------------------------------------------------- orchestrator integration


@pytest.mark.asyncio
async def test_real_retry_loop_routes_mechanical_correction_to_scoped_fix_then_reviews_it() -> None:
    request = StartFeatureRequest.model_validate(feature_payload("feature-scoped"))
    state = _initial_feature_state("feature-scoped", request)
    executor = RoutedFailureExecutor(
        "backend",
        findings=[finding(description="ESLint no-unused-vars at server/app.js:3.")],
        attempts_before_approval=1,
    )
    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    assert [item.role if item is not None else None for item in executor.routing] == [
        ModelRole.CODING,
        ModelRole.SCOPED_FIX,
    ]
    reviews = [item for item in result.artifacts if isinstance(item, ReviewArtifact)]
    assert [item.verdict for item in reviews] == ["changes_requested", "approved"]
    assert result.child_workflows["backend"].status in {
        ChildWorkflowStatus.APPROVED,
        ChildWorkflowStatus.COMPLETED,
    }


@pytest.mark.asyncio
async def test_real_retry_loop_keeps_substantive_review_work_on_primary_coding() -> None:
    request = StartFeatureRequest.model_validate(feature_payload("feature-substantive"))
    state = _initial_feature_state("feature-substantive", request)
    executor = RoutedFailureExecutor(
        "backend",
        findings=[finding(description="The authentication flow is architecturally incorrect.")],
        attempts_before_approval=1,
    )
    await FeatureWorkflowOrchestrator(child_executor=executor, model_router=router()).start(
        state, credentials=no_credentials()
    )
    assert [item.role if item is not None else None for item in executor.routing] == [
        ModelRole.CODING,
        ModelRole.CODING,
    ]


@pytest.mark.asyncio
async def test_model_routing_never_expands_the_existing_retry_budget() -> None:
    request = StartFeatureRequest.model_validate(feature_payload("feature-bounded"))
    state = _initial_feature_state("feature-bounded", request)
    state.max_validation_retries = 3
    state.max_child_review_cycles = 12
    executor = RoutedFailureExecutor(
        "backend",
        findings=[finding(description="ESLint no-unused-vars at server/app.js:3.")],
    )
    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    assert executor.attempts == state.max_validation_retries + 1
    assert result.child_workflows["backend"].validation_retry_count == 3
    assert [item.role for item in executor.routing[1:] if item is not None] == [
        ModelRole.SCOPED_FIX,
    ] * state.max_validation_retries


@pytest.mark.asyncio
async def test_repository_configuration_failure_starts_no_scoped_fix_attempt() -> None:
    request = StartFeatureRequest.model_validate(feature_payload("feature-config-failure"))
    state = _initial_feature_state("feature-config-failure", request)
    executor = RoutedFailureExecutor(
        "backend",
        findings=[
            finding(
                description="ESLint configuration references a dependency that is not installed."
            )
        ],
        failure_classification="validation_configuration_failure",
    )
    await FeatureWorkflowOrchestrator(child_executor=executor, model_router=router()).start(
        state, credentials=no_credentials()
    )
    assert all(
        decision is not None and decision.role is ModelRole.CODING for decision in executor.routing
    )


@pytest.mark.asyncio
async def test_routing_state_and_attempts_remain_repository_scoped() -> None:
    request = StartFeatureRequest.model_validate(
        feature_payload("feature-isolated", repositories=3)
    )
    state = _initial_feature_state("feature-isolated", request)
    executor = RoutedFailureExecutor(
        "backend",
        findings=[finding(description="Remove the unused import from this file.")],
        attempts_before_approval=1,
    )
    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    for repository_id in ("frontend", "shared"):
        decisions = executor.routing_by_repository[repository_id]
        assert len(decisions) == 1
        assert decisions[0] is not None and decisions[0].role is ModelRole.CODING
    backend_routing = result.child_workflows["backend"].model_routing
    assert backend_routing is not None
    assert backend_routing["role"] == "scoped_fix"


def test_configuration_summary_contains_only_safe_resolved_metadata() -> None:
    described = model_configs().describe()
    # Keyed by platform, then by role: two providers can each be configured.
    assert set(described) == {AgentPlatform.OPENAI.value}
    for roles in described.values():
        assert set(roles) == {role.value for role in ModelRole}
        for values in roles.values():
            assert set(values) <= {
                "platform",
                "role",
                "model",
                "reasoning",
                "max_tokens",
                "performance_tier",
                "requested_reasoning",
                "routing_reason",
            }
            # `token` is matched as a suffix rather than a substring, so an auth token --
            # `github_token`, `access_token` -- is still caught while `max_tokens`, which is
            # a count of tokens rather than one, is not.
            assert not any(
                forbidden in key
                for key in values
                for forbidden in ("api_key", "secret", "credential")
            )
            assert not any(key == "token" or key.endswith("_token") for key in values)


def test_incomplete_central_configuration_is_rejected() -> None:
    coding = model_configs().get_model_config(ModelRole.CODING, platform=AgentPlatform.OPENAI)
    with pytest.raises(ModelConfigurationError):
        ModelConfigService({(AgentPlatform.OPENAI, ModelRole.CODING): coding})


def test_a_mechanical_finding_atop_a_reset_path_fragment_cannot_classify_scoped() -> None:
    """A fragment is not a scoped fix (§3.4 of the retry-edits-in-place task).

    The classifier judges the finding; the job on the reset path is the reconstruction.
    AB-Feature-171 routed a twenty-file regeneration to the scoped-fix model because one
    import was mechanical. The same finding with the workspace intact still routes scoped,
    so this is a demotion for missing evidence, never a new ceiling on mechanical work.
    """
    mechanical = finding(description="There is an unused import in this file.")

    intact = classify_findings([mechanical], previous_attempt_intact=True)
    assert determine_repair_scope(intact) is RepairScope.SCOPED

    fragment = classify_findings([mechanical], previous_attempt_intact=False)
    assert determine_repair_scope(fragment) is RepairScope.AMBIGUOUS
    assert all(not item.deterministic for item in fragment)
    assert any("intact" in item.reason for item in fragment)


def test_the_reset_fragment_demotion_keeps_the_primary_coding_role() -> None:
    """Routed through the real router: the reconstruction stays on the coding model."""
    outcome = router().route(
        inputs=routing_inputs(
            failure_classification=FailureClassification.VALIDATION_SOURCE_FAILURE
        ),
        classifications=classify_findings(
            [finding(description="There is an unused import in this file.")],
            previous_attempt_intact=False,
        ),
    )
    assert outcome.decision.role is ModelRole.CODING
    assert outcome.decision.repair_scope is RepairScope.AMBIGUOUS


# ------------------------------------------------------------- truncation degraded retry


def _persisted_decision(reasoning: str | None) -> ModelRoutingDecision:
    """One durable coding selection with only the effort a test varies."""
    return ModelRoutingDecision.model_validate(
        {
            "execution_mode": "INITIAL_IMPLEMENTATION",
            "role": "coding",
            "model": "test-coding-model",
            "reasoning": reasoning,
            "attempt": 2,
            "routing_reason": "Initial repository implementation uses the configured coding role.",
        }
    )


def test_a_truncation_steps_the_persisted_effort_down_exactly_one_level() -> None:
    """The degraded retry changes the one thing the truncation diagnostic prescribes.

    Everything else -- the model, the role, the attempt number -- is the same selection:
    this is the same attempt asking the same question with a smaller thinking spend, not a
    new attempt, so it must not disturb artifact lineage or consume retry budget.
    """
    ladder = {"max": "xhigh", "xhigh": "high", "high": "medium", "medium": "low"}
    for level, expected in ladder.items():
        degraded = degraded_truncation_retry(_persisted_decision(level))
        assert degraded is not None, level
        assert degraded.reasoning == expected
        assert degraded.model == "test-coding-model"
        assert degraded.attempt == 2
        assert "reduced reasoning effort" in degraded.routing_reason


def test_a_selection_with_no_step_left_is_never_degraded() -> None:
    """`low` has no step below it, and absent or disabled effort has nothing to reduce."""
    assert degraded_truncation_retry(_persisted_decision("low")) is None
    assert degraded_truncation_retry(_persisted_decision(None)) is None
    assert degraded_truncation_retry(_persisted_decision("none")) is None
    assert degraded_truncation_retry(None) is None


class TruncationThenApproveExecutor:
    """Raise a wrapped `response_truncated` for the backend N times, then approve.

    The truncation arrives exactly as the live coding call delivers it: an
    `LLMAdapterError` carrying the classification, wrapped in `ExternalOperationError`, so
    the loop's cause walk -- not the top-level type -- is what recognizes it.
    """

    def __init__(self, *, truncations: int) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._remaining = truncations
        # (attempt number, persisted reasoning) per backend call, in order.
        self.calls: list[tuple[int, str | None]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        child = kwargs["child"]
        decision = ModelRoutingDecision.from_persisted(child.model_routing)
        self.calls.append((child.retry_count, decision.reasoning if decision is not None else None))
        if self._remaining > 0:
            self._remaining -= 1
            msg = "the model response reached its configured max_tokens bound and is truncated"
            truncated = LLMAdapterError(
                msg, diagnostics=(msg,), failure_classification="response_truncated"
            )
            wrapper_msg = "external operation failed before a confirmed result"
            raise ExternalOperationError(wrapper_msg) from truncated
        return await self._delegate.run(**kwargs)


@pytest.mark.asyncio
async def test_a_truncated_coding_call_is_re_asked_once_at_reduced_effort() -> None:
    """AB-Feature-181's shape, with the remedy applied instead of printed.

    Its coding call spent the entire 128000-token bound -- the model's output ceiling, so
    no larger bound existed -- at `xhigh`, and the workstream ended at attempt 0 with
    "lower the configured reasoning effort" sitting in its own failure record. The loop now
    applies that remedy once: the same attempt re-runs with the persisted selection one
    effort level lower, and nothing is spent from the retry budget.
    """
    request = StartFeatureRequest.model_validate(feature_payload("feature-truncated"))
    state = _initial_feature_state("feature-truncated", request)
    executor = TruncationThenApproveExecutor(truncations=1)

    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    # Same attempt number both times: the degraded re-ask is not a new attempt.
    assert executor.calls == [(0, "max"), (0, "xhigh")]
    assert result.child_workflows["backend"].status in {
        ChildWorkflowStatus.APPROVED,
        ChildWorkflowStatus.COMPLETED,
    }
    assert result.child_workflows["backend"].retry_count == 0


@pytest.mark.asyncio
async def test_a_second_truncation_at_the_lower_effort_is_terminal() -> None:
    """One step, not a ladder walked to the floor: the second truncation stops as before."""
    request = StartFeatureRequest.model_validate(feature_payload("feature-truncated-twice"))
    state = _initial_feature_state("feature-truncated-twice", request)
    executor = TruncationThenApproveExecutor(truncations=2)

    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    assert executor.calls == [(0, "max"), (0, "xhigh")]
    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    assert any("max_tokens" in issue for issue in child.blocking_issues)
