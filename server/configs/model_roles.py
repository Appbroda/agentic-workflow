"""Configured model roles: the one place a task is turned into a model and an effort level.

Business logic names a ``ModelRole`` and never a model. Which model a role resolves to, and
how hard that model is asked to think, is deployment configuration -- so a deployment can
change either without a code change, and a provider whose model identifiers look nothing like
OpenAI's can be configured for a role without rewriting orchestration.

A role now resolves *per platform*. The four roles are unchanged, because they are about the
authority of a call and not about a vendor: ``REVIEW`` never writes code and ``CODING`` never
certifies its own work, whichever provider executes them. What changed is that asking for a
role without saying which platform is asking is no longer a question this module can answer.

Capability normalization lives here too, for the reason the roles do: an agent that decided
for itself whether its model accepts ``max`` would be one of five places holding the same
belief, and the first one to be wrong would be wrong invisibly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# The effort names this platform accepts. Deliberately a vocabulary check and not a
# per-model capability table: which levels a given model accepts differs between models --
# `gpt-5.3-codex` takes `xhigh` but rejects `max`, `gpt-5.6-sol` takes both -- and a table of
# that in this repository would be wrong the day it is written. A misspelled level, though, is
# always a configuration mistake, and it is worth refusing at startup rather than discovering
# when a live feature reaches its first model call.
#
# The vocabulary is unchanged by the second provider. Anthropic's effort levels are `low`
# through `max` and overlap these exactly; `none` is the one this platform has that no
# provider spells the same way, and each adapter says what it means for its own API.
REASONING_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")


class AgentPlatform(StrEnum):
    """Which model provider executes a feature's agents, start to finish.

    Chosen by the person submitting a feature and fixed for that feature's life. It is not a
    deployment constant and is never re-read from configuration for a feature already
    running: a resumed attempt that switched SDK mid-workstream would make its own execution
    record false.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class PerformanceTier(StrEnum):
    """How expensively a feature's model roles resolve, chosen with the platform at start.

    A tier is a named preset over the four roles, not a routing input: the router still
    chooses *which role* answers, and the tier only decides what that role resolves to. Like
    the platform beside it, the tier is pinned when a feature starts and never re-read from
    configuration for a feature already running.

    ``HIGH`` is what every deployment had before tiers existed: the unsuffixed model-role
    variables resolve as the high tier, so a start request that never names a tier behaves
    exactly as it always has.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    # Above the unsuffixed configuration: a deployment that wants a preset more expensive
    # than its `high` names it here, through the same `{PROVIDER}_ULTRA_{ROLE}_...`
    # variables every other environment-backed tier uses. Nothing resolves it unless the
    # deployment configures it, per the completeness rule.
    ULTRA = "ultra"
    # A user-authored model setup rather than an environment-backed preset. It exists so the
    # NOT NULL tier columns can say the truth about a custom feature -- writing `high` there
    # would be a lie the cost-attribution records are built on. Nothing environment-shaped
    # resolves it: a custom feature resolves through its snapshot, and every loop that builds
    # presets from variable names iterates `env_backed()` instead of the whole enum, because
    # `anthropic_custom_coding_max_tokens` is a settings field that does not exist.
    CUSTOM = "custom"

    @classmethod
    def env_backed(cls) -> tuple[PerformanceTier, ...]:
        """Return the tiers a deployment configures through environment variables.

        Everything that derives variable or field names from a tier iterates this rather than
        the enum. ``CUSTOM`` has no variables to derive: it is resolved from a persisted
        role-map snapshot, never from the environment.
        """
        return (cls.LOW, cls.MEDIUM, cls.HIGH, cls.ULTRA)


# What a person is shown when asked to choose a tier. "Max" rather than "High" because it
# names today's maximum-effort behaviour; "Economy" rather than "Low" because its label must
# carry its own warning: bare minimum, mostly for testing, capable of very small changes.
TIER_LABELS: Mapping[PerformanceTier, str] = {
    PerformanceTier.LOW: "Economy",
    PerformanceTier.MEDIUM: "Standard",
    PerformanceTier.HIGH: "Max",
    PerformanceTier.ULTRA: "Ultra",
    PerformanceTier.CUSTOM: "Custom",
}


class ModelRole(StrEnum):
    """Which kind of work a configured model is selected for.

    Roles are logical, so two of them may legitimately resolve to the same model. What
    separates them is the prompt, the context and the authority of the call: ``REVIEW`` never
    writes code and ``CODING`` never certifies its own work, whatever they are configured with.
    """

    REASONING = "reasoning"
    CODING = "coding"
    REVIEW = "review"
    SCOPED_FIX = "scoped_fix"


# The environment variables each active role reads, and -- where one exists -- the variable a
# deployment that predates the role already set. The fallbacks let an environment containing
# only OPENAI_REASONING_MODEL and OPENAI_CODING_MODEL continue to start. Checked-in deployment
# configuration supplies the four intended defaults explicitly; no agent supplies one itself.
@dataclass(frozen=True, slots=True)
class ModelRoleVariables:
    """Which configuration a role reads, in the order it is consulted."""

    model_variable: str
    reasoning_variable: str
    max_tokens_variable: str
    legacy_model_variable: str | None = None
    legacy_reasoning_variable: str | None = None


MODEL_ROLE_VARIABLES: Mapping[tuple[AgentPlatform, ModelRole], ModelRoleVariables] = {
    (AgentPlatform.OPENAI, ModelRole.REASONING): ModelRoleVariables(
        model_variable="OPENAI_REASONING_MODEL",
        reasoning_variable="OPENAI_REASONING_EFFORT",
        max_tokens_variable="OPENAI_REASONING_MAX_TOKENS",
    ),
    (AgentPlatform.OPENAI, ModelRole.CODING): ModelRoleVariables(
        model_variable="OPENAI_CODING_MODEL",
        reasoning_variable="OPENAI_CODING_REASONING_EFFORT",
        max_tokens_variable="OPENAI_CODING_MAX_TOKENS",
        legacy_reasoning_variable="OPENAI_CODING_EFFORT",
    ),
    (AgentPlatform.OPENAI, ModelRole.REVIEW): ModelRoleVariables(
        model_variable="OPENAI_REVIEW_MODEL",
        reasoning_variable="OPENAI_REVIEW_REASONING_EFFORT",
        max_tokens_variable="OPENAI_REVIEW_MAX_TOKENS",
        legacy_model_variable="OPENAI_REASONING_MODEL",
        legacy_reasoning_variable="OPENAI_REASONING_EFFORT",
    ),
    (AgentPlatform.OPENAI, ModelRole.SCOPED_FIX): ModelRoleVariables(
        model_variable="OPENAI_SCOPED_FIX_MODEL",
        reasoning_variable="OPENAI_SCOPED_FIX_REASONING_EFFORT",
        max_tokens_variable="OPENAI_SCOPED_FIX_MAX_TOKENS",
        legacy_model_variable="OPENAI_CODING_MODEL",
        legacy_reasoning_variable="OPENAI_CODING_EFFORT",
    ),
    # No legacy fallbacks on this side. There is no deployment that predates these variables,
    # so a fallback would only be a way for a half-configured provider to look configured.
    (AgentPlatform.ANTHROPIC, ModelRole.REASONING): ModelRoleVariables(
        model_variable="ANTHROPIC_REASONING_MODEL",
        reasoning_variable="ANTHROPIC_REASONING_EFFORT",
        max_tokens_variable="ANTHROPIC_REASONING_MAX_TOKENS",
    ),
    (AgentPlatform.ANTHROPIC, ModelRole.CODING): ModelRoleVariables(
        model_variable="ANTHROPIC_CODING_MODEL",
        reasoning_variable="ANTHROPIC_CODING_REASONING_EFFORT",
        max_tokens_variable="ANTHROPIC_CODING_MAX_TOKENS",
    ),
    (AgentPlatform.ANTHROPIC, ModelRole.REVIEW): ModelRoleVariables(
        model_variable="ANTHROPIC_REVIEW_MODEL",
        reasoning_variable="ANTHROPIC_REVIEW_REASONING_EFFORT",
        max_tokens_variable="ANTHROPIC_REVIEW_MAX_TOKENS",
    ),
    (AgentPlatform.ANTHROPIC, ModelRole.SCOPED_FIX): ModelRoleVariables(
        model_variable="ANTHROPIC_SCOPED_FIX_MODEL",
        reasoning_variable="ANTHROPIC_SCOPED_FIX_REASONING_EFFORT",
        max_tokens_variable="ANTHROPIC_SCOPED_FIX_MAX_TOKENS",
    ),
}


# What a person is shown when they are asked to choose one. Brand spelling rather than the
# key: `anthropic` is the store key and "Claude" is what the models are called.
PLATFORM_LABELS: Mapping[AgentPlatform, str] = {
    AgentPlatform.ANTHROPIC: "Claude",
    AgentPlatform.OPENAI: "OpenAI",
}


def tier_model_role_variables(
    platform: AgentPlatform, tier: PerformanceTier, role: ModelRole
) -> ModelRoleVariables:
    """Name the configuration one role reads for one tier: the base name with a tier segment.

    ``OPENAI_CODING_MODEL`` becomes ``OPENAI_MEDIUM_CODING_MODEL`` and so on, so the tier
    scheme is the existing scheme and not a second one. Deliberately no legacy names: every
    tier-suffixed variable postdates the four-role convention, so a fallback here could only
    let a half-configured tier look configured.
    """
    base = MODEL_ROLE_VARIABLES[(platform, role)]
    prefix = f"{platform.name}_"
    tiered = f"{platform.name}_{tier.name}_"
    return ModelRoleVariables(
        model_variable=base.model_variable.replace(prefix, tiered, 1),
        reasoning_variable=base.reasoning_variable.replace(prefix, tiered, 1),
        max_tokens_variable=base.max_tokens_variable.replace(prefix, tiered, 1),
    )


_ROLE_ROUTING_REASONS: Mapping[ModelRole, str] = {
    ModelRole.REASONING: "Planning and high-level analysis use the configured reasoning role.",
    ModelRole.CODING: "Feature implementation and substantive fixes use the coding role.",
    ModelRole.REVIEW: "Independent implementation review uses the configured review role.",
    ModelRole.SCOPED_FIX: "A localized mechanical repair uses the scoped-fix role.",
}


class ModelConfigurationError(ValueError):
    """Raised when configured model roles cannot be resolved into a usable selection.

    Carries the name of the offending variable, never a value: a model identifier is safe to
    quote but this type is also raised about reasoning levels and thresholds, and one message
    format that is safe for every raise site is worth more than five that are each safe once.
    """


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One role's resolved model selection, with the provenance of both values.

    ``requested_reasoning`` and ``reasoning`` differ only when the deployment has declared the
    level unsupported for this model, in which case the request omits it and keeps the
    provider's default. That is a normalization, recorded as one -- not a silent substitution
    of a different model.

    ``max_tokens`` is required by the Messages API and unused by the Responses API, which is
    fine: this type already carries ``requested_reasoning`` for a case only one path reads.
    """

    platform: AgentPlatform
    role: ModelRole
    model: str
    reasoning: str | None
    requested_reasoning: str | None
    max_tokens: int | None
    model_variable: str
    reasoning_variable: str | None
    resolved_from_legacy_model_variable: bool
    routing_reason: str
    # Which performance tier resolved this selection. `HIGH` is the unsuffixed configuration
    # every deployment already has, so a config that predates tiers describes itself truthfully.
    tier: PerformanceTier = PerformanceTier.HIGH

    def as_metadata(self) -> dict[str, Any]:
        """Return the persistable, credential-free description of this selection."""
        return {
            "platform": self.platform.value,
            "role": self.role.value,
            "model": self.model,
            "reasoning": self.reasoning,
            "max_tokens": self.max_tokens,
            "performance_tier": self.tier.value,
            "routing_reason": self.routing_reason,
            **(
                {"requested_reasoning": self.requested_reasoning}
                if self.requested_reasoning != self.reasoning
                else {}
            ),
        }


def normalize_reasoning_level(
    variable: str, value: str | None, *, allow_empty: bool = True
) -> str | None:
    """Validate a configured effort name and turn an unset one into "send nothing"."""
    if value is None:
        return None
    level = value.strip()
    if not level:
        if allow_empty:
            return None
        msg = f"{variable} must name a reasoning level or be unset"
        raise ModelConfigurationError(msg)
    if level not in REASONING_LEVELS:
        msg = (
            f"{variable} must be one of {', '.join(REASONING_LEVELS)}; "
            "an unrecognised level is a configuration mistake"
        )
        raise ModelConfigurationError(msg)
    return level


def normalize_unsupported_reasoning(
    declared: Mapping[str, Iterable[str]] | None,
) -> Mapping[str, frozenset[str]]:
    """Validate the deployment's declaration of which levels a model does not accept."""
    normalized: dict[str, frozenset[str]] = {}
    for model, levels in (declared or {}).items():
        if not model.strip():
            msg = "MODEL_REASONING_UNSUPPORTED keys must name a model"
            raise ModelConfigurationError(msg)
        resolved = {
            level
            for item in levels
            if (level := normalize_reasoning_level("MODEL_REASONING_UNSUPPORTED", item)) is not None
        }
        normalized[model.strip()] = frozenset(resolved)
    return normalized


def supported_reasoning(
    model: str, reasoning: str | None, unsupported: Mapping[str, frozenset[str]]
) -> str | None:
    """Drop a configured effort the deployment has declared this model does not accept.

    Omitting leaves the provider's own default, which is what a deployment configuring nothing
    has always had. Nothing here guesses: a model with no declaration keeps whatever was
    configured and the provider remains the authority on whether it accepts it.
    """
    if reasoning is None:
        return None
    return None if reasoning in unsupported.get(model.strip(), frozenset()) else reasoning


# The smallest Anthropic `max_tokens` each configured effort can finish under. Thinking spends
# the same bound the answer does, so a bound sized only for the answer truncates exactly when
# the effort is doing its job: AB-Feature-175's planner produced nothing twice at 32k under
# `max`, and AB-Feature-180's reviewer died at 16k under `high`. These are floors, not
# sufficiency guarantees -- 180's *coding* call exhausted even 128k under `xhigh` -- but every
# truncation this platform has recorded would have been refused at startup by them.
#
# An unset effort still gets the floor for `medium`: adaptive thinking is the provider default
# for the models this platform can be configured with, so "nothing configured" is not "no
# thinking".
#
# They live here rather than in `configs.settings` because two authorities now enforce them:
# the settings validator over the environment-backed tiers, and `validate_model_setup` over a
# user-authored role map, which has no `Settings` instance to hang a validator on.
ANTHROPIC_EFFORT_MAX_TOKENS_FLOORS: Mapping[str, int] = {
    "none": 1,
    "low": 8_000,
    "medium": 16_000,
    "high": 32_000,
    "xhigh": 48_000,
    "max": 64_000,
}
ANTHROPIC_UNSET_EFFORT_FLOOR = ANTHROPIC_EFFORT_MAX_TOKENS_FLOORS["medium"]


def anthropic_effort_floor(effort: str | None) -> int:
    """Return the smallest bound one effort can finish under, or the unset-effort floor."""
    if effort is None:
        return ANTHROPIC_UNSET_EFFORT_FLOOR
    return ANTHROPIC_EFFORT_MAX_TOKENS_FLOORS[effort]


def require_anthropic_floor(
    *,
    role: ModelRole,
    effort: str | None,
    bound: int,
    bound_name: str,
    effort_name: str,
) -> None:
    """Raise when one Anthropic role's output bound is under its effort's floor.

    The pure body of the settings validator, extracted so a user-authored setup -- which has
    no ``Settings`` instance -- is refused by the same rule and the same sentence. ``effort``
    is already normalized; ``bound_name`` and ``effort_name`` are whatever the caller can
    honestly blame: an environment variable for a tier, the setup's own field for a setup.
    """
    floor = anthropic_effort_floor(effort)
    if bound >= floor:
        return
    effort_label = effort if effort is not None else "the provider's adaptive default"
    msg = (
        f"{bound_name}={bound} cannot hold a {role.value} response at reasoning effort "
        f"{effort_label!r}: thinking spends the same max_tokens bound the answer does, so "
        f"the response truncates before it completes (stop_reason=max_tokens, terminal). "
        f"Raise {bound_name} to at least {floor} or lower {effort_name}."
    )
    raise ModelConfigurationError(msg)


@dataclass(frozen=True, slots=True)
class AnthropicCeilingJudgement:
    """One Anthropic (model, effort, bound) pairing judged against the declared ceilings.

    Each field carries the finding's full sentence or ``None``. The judgement is the rule
    alone; what to do about a finding belongs to the caller, because the two authorities
    that share this rule owe different policies: ``validate_model_setup`` refuses everything
    refusable, since a person who authored a setup asked for that pairing explicitly, while
    the settings validator refuses only ``unsatisfiable`` and warns on the rest, because an
    environment preset may predate the ceiling declaration and refusing startup takes down
    a running deployment over configuration that used to be legal.
    """

    # The effort's floor exceeds the model's declared ceiling: no bound value can satisfy
    # both, so the pairing can never complete and even startup must refuse it.
    unsatisfiable: str | None
    # The bound exceeds the ceiling. Fixable by lowering the bound, so only the authoring
    # path refuses it.
    bound_over_ceiling: str | None
    # The AB-Feature-181 shape: coding at xhigh or above with the bound flush against the
    # ceiling -- valid, but the exact pairing that exhausted 181 with zero headroom.
    exhaustion_shape: str | None


def judge_anthropic_ceiling(
    *,
    role: ModelRole,
    model: str,
    effort: str | None,
    bound: int,
    ceilings: Mapping[str, int],
    subject: str,
    bound_name: str,
) -> AnthropicCeilingJudgement:
    """Judge one Anthropic pairing against the deployment's declared output ceilings.

    The one rule behind both authorities -- ``validate_model_setup`` over a user-authored
    role map and the settings validator over the environment-backed tiers -- extracted the
    way ``require_anthropic_floor`` was: the function owns the sentences, the caller
    supplies whatever it can honestly blame (``subject`` for the pairing, ``bound_name``
    for the bound's origin). ``effort`` arrives already normalized. A model absent from
    ``ceilings`` has no known ceiling and yields no finding: the deployment is the
    authority on what its models accept, and nothing here holds a ceiling table of its own.
    """
    ceiling = ceilings.get(model)
    if ceiling is None:
        return AnthropicCeilingJudgement(None, None, None)
    floor = anthropic_effort_floor(effort)
    unsatisfiable: str | None = None
    if floor > ceiling:
        effort_label = effort if effort is not None else "the provider's adaptive default"
        unsatisfiable = (
            f"{subject}: reasoning effort {effort_label!r} on {model} "
            f"needs at least {floor} output tokens and the model's declared ceiling is "
            f"{ceiling}; choose a lower effort or a different model for this role"
        )
    bound_over_ceiling: str | None = None
    if bound > ceiling:
        bound_over_ceiling = (
            f"{bound_name}={bound} exceeds {model}'s declared output "
            f"ceiling of {ceiling} (MODEL_MAX_OUTPUT_TOKENS); lower it to at most {ceiling}"
        )
    exhaustion_shape: str | None = None
    if role is ModelRole.CODING and effort in {"xhigh", "max"} and bound == ceiling:
        exhaustion_shape = (
            f"{role.value} at reasoning effort {effort!r} with max_tokens at {model}'s "
            f"declared ceiling ({ceiling}) is the exact shape that exhausted "
            "AB-Feature-181: thinking and the answer share this bound, and there is no "
            "headroom left to raise it. Consider a lower effort or a different model."
        )
    return AnthropicCeilingJudgement(
        unsatisfiable=unsatisfiable,
        bound_over_ceiling=bound_over_ceiling,
        exhaustion_shape=exhaustion_shape,
    )


def normalize_max_output_tokens(declared: Mapping[str, Any] | None) -> Mapping[str, int]:
    """Validate the deployment's declaration of each model's output ceiling.

    The same posture as ``normalize_unsupported_reasoning``: the deployment is the authority
    on what its models accept, a model absent from the declaration has no known ceiling, and
    nothing in this repository holds a ceiling table of its own -- it would be wrong the day a
    provider raises a limit.
    """
    normalized: dict[str, int] = {}
    for model, ceiling in (declared or {}).items():
        if not model.strip():
            msg = "MODEL_MAX_OUTPUT_TOKENS keys must name a model"
            raise ModelConfigurationError(msg)
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
            msg = "MODEL_MAX_OUTPUT_TOKENS values must be positive integers"
            raise ModelConfigurationError(msg)
        normalized[model.strip()] = ceiling
    return normalized


def normalize_context_window_tokens(declared: Mapping[str, Any] | None) -> Mapping[str, int]:
    """Validate the deployment's declaration of each model's context window.

    The same posture as ``normalize_max_output_tokens`` beside it: the deployment is the
    authority on what its models read, a model absent from the declaration has no known
    window and falls back to the default snapshot budget, and nothing in this repository
    holds a window table of its own -- it would be wrong the day a provider raises a limit.
    """
    normalized: dict[str, int] = {}
    for model, window in (declared or {}).items():
        if not model.strip():
            msg = "MODEL_CONTEXT_WINDOW_TOKENS keys must name a model"
            raise ModelConfigurationError(msg)
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            msg = "MODEL_CONTEXT_WINDOW_TOKENS values must be positive integers"
            raise ModelConfigurationError(msg)
        normalized[model.strip()] = window
    return normalized


@dataclass(frozen=True, slots=True)
class ModelSetupRole:
    """One role of a user-authored model setup, exactly as the person entered it."""

    platform: AgentPlatform
    model: str
    reasoning_effort: str | None
    max_tokens: int | None


def parse_model_setup_roles(raw: Mapping[str, Any]) -> dict[ModelRole, ModelSetupRole]:
    """Read a role map out of stored or submitted JSON, refusing a malformed one by name.

    Shape only -- values are judged by ``validate_model_setup``. Kept separate because two
    kinds of caller need the shape check alone: the store deserializing its own rows, and the
    response projection showing a setup back exactly as authored.
    """
    roles: dict[ModelRole, ModelSetupRole] = {}
    for key, value in raw.items():
        try:
            role = ModelRole(key)
        except ValueError as error:
            known = ", ".join(item.value for item in ModelRole)
            msg = f"a model setup role must be one of {known}; {key!r} is not"
            raise ModelConfigurationError(msg) from error
        if not isinstance(value, Mapping):
            msg = f"roles.{role.value} must be an object naming platform, model and effort"
            raise ModelConfigurationError(msg)
        platform_raw = str(value.get("platform") or "").strip()
        try:
            platform = AgentPlatform(platform_raw)
        except ValueError as error:
            known = ", ".join(item.value for item in AgentPlatform)
            msg = f"roles.{role.value}.platform must be one of {known}; {platform_raw!r} is not"
            raise ModelConfigurationError(msg) from error
        model = str(value.get("model") or "").strip()
        if not model:
            msg = f"roles.{role.value}.model must name a model"
            raise ModelConfigurationError(msg)
        effort_raw = value.get("reasoning_effort")
        effort = None if effort_raw is None else str(effort_raw)
        max_tokens_raw = value.get("max_tokens")
        if max_tokens_raw is not None and (
            not isinstance(max_tokens_raw, int) or isinstance(max_tokens_raw, bool)
        ):
            msg = f"roles.{role.value}.max_tokens must be an integer or absent"
            raise ModelConfigurationError(msg)
        roles[role] = ModelSetupRole(
            platform=platform,
            model=model,
            reasoning_effort=effort,
            max_tokens=max_tokens_raw,
        )
    missing = [role.value for role in ModelRole if role not in roles]
    if missing:
        msg = (
            "a model setup must name all four roles, exactly as a partial tier is refused; "
            f"missing: {', '.join(missing)}"
        )
        raise ModelConfigurationError(msg)
    return roles


def validate_model_setup(
    roles: Mapping[ModelRole, ModelSetupRole],
    *,
    unsupported: Mapping[str, frozenset[str]],
    ceilings: Mapping[str, int],
    configured_platforms: tuple[AgentPlatform, ...] | None = None,
) -> tuple[str, ...]:
    """Hold a user-authored role map to the checks an environment-backed tier passes.

    One predicate, three call sites -- save, feature start, and again when a snapshot is
    resolved into a ``ModelConfigService``, because a snapshot can outlive a change to the
    deployment's declarations. It raises ``ModelConfigurationError`` with the honest sentence
    and never normalizes: a person who authored a setup asked for that pairing explicitly, so
    every role is strict -- a declared-unsupported pairing is a refusal, never a
    silently-dropped effort. Nothing here lowers an effort, raises a bound, substitutes a
    model, or drops a role to make a setup work.

    Returns non-blocking warnings: the AB-Feature-181 shape (a coding role at ``xhigh`` or
    above whose bound sits exactly at the model's declared ceiling -- valid, but the exact
    pairing that exhausted 181 with zero headroom), and a role pinned to a platform this
    deployment configures no models for (allowed deliberately -- the setup supplies its own
    model and the credential check is the real gate -- but worth saying out loud).
    """
    warnings: list[str] = []
    missing = [role.value for role in ModelRole if role not in roles]
    if missing:
        msg = (
            "a model setup must name all four roles, exactly as a partial tier is refused; "
            f"missing: {', '.join(missing)}"
        )
        raise ModelConfigurationError(msg)
    for role in ModelRole:
        entry = roles[role]
        effort_name = f"roles.{role.value}.reasoning_effort"
        bound_name = f"roles.{role.value}.max_tokens"
        effort = normalize_reasoning_level(effort_name, entry.reasoning_effort)
        if effort is not None and supported_reasoning(entry.model, effort, unsupported) != effort:
            msg = (
                f"{effort_name} names {effort!r}, which MODEL_REASONING_UNSUPPORTED declares "
                f"{entry.model} does not accept; choose a supported effort for this role"
            )
            raise ModelConfigurationError(msg)
        if entry.platform is AgentPlatform.OPENAI:
            if entry.max_tokens is not None:
                msg = (
                    f"{bound_name} does not apply to an OpenAI-platform role; the Responses "
                    "API is not sent an output bound. Leave it unset."
                )
                raise ModelConfigurationError(msg)
            continue
        if entry.max_tokens is None or entry.max_tokens <= 0:
            msg = (
                f"{bound_name} is required for an Anthropic-platform role: the Messages API "
                "rejects a request without max_tokens"
            )
            raise ModelConfigurationError(msg)
        # One rule, two authorities: the same judgement the settings validator applies to
        # the environment-backed tiers. Only the policy differs, and an authored setup is
        # the strict side -- every refusable finding refuses, because the person asked for
        # this pairing explicitly.
        judgement = judge_anthropic_ceiling(
            role=role,
            model=entry.model,
            effort=effort,
            bound=entry.max_tokens,
            ceilings=ceilings,
            subject=f"roles.{role.value}",
            bound_name=bound_name,
        )
        if judgement.unsatisfiable is not None:
            raise ModelConfigurationError(judgement.unsatisfiable)
        if judgement.bound_over_ceiling is not None:
            raise ModelConfigurationError(judgement.bound_over_ceiling)
        require_anthropic_floor(
            role=role,
            effort=effort,
            bound=entry.max_tokens,
            bound_name=bound_name,
            effort_name=effort_name,
        )
        if judgement.exhaustion_shape is not None:
            warnings.append(judgement.exhaustion_shape)
    if configured_platforms is not None:
        for platform in dict.fromkeys(entry.platform for entry in roles.values()):
            if platform in configured_platforms:
                continue
            pinned = ", ".join(role.value for role in ModelRole if roles[role].platform is platform)
            warnings.append(
                f"{pinned} name{'s' if ',' not in pinned else ''} {platform.value}, which "
                "this deployment configures no models for; the setup supplies its own model, "
                f"so only a stored {PLATFORM_LABELS[platform]} credential is required to run it"
            )
    return tuple(warnings)


def model_config_service_for_setup(
    snapshot: Mapping[str, Any],
    *,
    unsupported: Mapping[str, frozenset[str]],
    ceilings: Mapping[str, int],
) -> ModelConfigService:
    """Resolve one feature's pinned setup snapshot into the service every caller asks.

    Validates again on the way -- belt and braces, because a snapshot can outlive a change to
    the deployment's declarations -- and refuses with the same sentence the save path uses,
    never with a clamp. The resolved ``ModelConfig``s carry the setup's name as their
    provenance: there is no environment variable to point a person at.
    """
    raw_roles = snapshot.get("roles")
    if not isinstance(raw_roles, Mapping):
        msg = "a model setup snapshot must carry its role map"
        raise ModelConfigurationError(msg)
    roles = parse_model_setup_roles(raw_roles)
    validate_model_setup(roles, unsupported=unsupported, ceilings=ceilings)
    name = str(snapshot.get("name") or snapshot.get("setup_id") or "custom setup")
    configs: dict[ModelRole, ModelConfig] = {}
    for role, entry in roles.items():
        effort = normalize_reasoning_level(
            f"roles.{role.value}.reasoning_effort", entry.reasoning_effort
        )
        configs[role] = ModelConfig(
            platform=entry.platform,
            role=role,
            model=entry.model,
            reasoning=effort,
            requested_reasoning=effort,
            max_tokens=(entry.max_tokens if entry.platform is AgentPlatform.ANTHROPIC else None),
            model_variable=f'model setup "{name}"',
            reasoning_variable=(f'model setup "{name}"' if effort is not None else None),
            routing_reason=(
                f"{_ROLE_ROUTING_REASONS[role]} Pinned by model setup {name!r}; the "
                "deployment's per-agent effort overrides do not apply."
            ),
            resolved_from_legacy_model_variable=False,
            tier=PerformanceTier.CUSTOM,
        )
    return ModelConfigService.for_role_map(configs)


@dataclass(frozen=True, slots=True)
class PlatformModelSettings:
    """One platform's configured models, efforts and output bounds, before validation."""

    models: Mapping[ModelRole, str]
    reasoning: Mapping[ModelRole, str | None]
    max_tokens: Mapping[ModelRole, int | None]
    legacy_models: Mapping[ModelRole, str] | None = None
    legacy_reasoning: Mapping[ModelRole, str | None] | None = None


class ModelConfigService:
    """Resolve a platform and role into the configured model and effort, for every caller.

    Agents ask this for a role and say which platform is asking. They do not read environment
    variables, and they do not decide what their model supports -- both of those, done per
    agent, are how five components come to disagree about one deployment's configuration.
    """

    def __init__(
        self,
        configs: Mapping[tuple[AgentPlatform, ModelRole], ModelConfig],
        *,
        tier_configs: Mapping[
            tuple[AgentPlatform, PerformanceTier], Mapping[ModelRole, ModelConfig]
        ]
        | None = None,
    ) -> None:
        """Bind an already validated selection for every role of every usable platform.

        Completeness is per platform, and at least one platform must be complete. A platform
        with no configuration at all is simply not offered; a platform missing some of its
        roles never reaches here, because ``build_model_config_service`` refuses it.

        ``tier_configs`` carries the tier-suffixed presets, already resolved. Every complete
        platform is its own ``HIGH`` tier whether or not the builder said so, because the
        unsuffixed configuration *is* the high tier and a service constructed without tiers
        must answer tier questions the way the deployment always behaved.
        """
        platforms = tuple(
            platform
            for platform in AgentPlatform
            if all((platform, role) in configs for role in ModelRole)
        )
        if not platforms:
            named = ", ".join(
                MODEL_ROLE_VARIABLES[(platform, ModelRole.REASONING)].model_variable
                for platform in AgentPlatform
            )
            msg = (
                "no agent platform is completely configured; configure one provider's four "
                f"model roles (start with one of {named})"
            )
            raise ModelConfigurationError(msg)
        self._configs = dict(configs)
        self._platforms = platforms
        self._tier_configs = {key: dict(value) for key, value in (tier_configs or {}).items()}
        # Empty for every environment-backed service. A role-addressed service built by
        # `for_role_map` fills it, and it is what lets a mixed setup say which platform
        # answers each role.
        self._role_platforms: dict[ModelRole, AgentPlatform] = {}
        for platform in platforms:
            self._tier_configs.setdefault(
                (platform, PerformanceTier.HIGH),
                {role: self._configs[(platform, role)] for role in ModelRole},
            )

    @classmethod
    def for_role_map(cls, configs: Mapping[ModelRole, ModelConfig]) -> ModelConfigService:
        """Bind a role-addressed selection: all four roles, each naming its own platform.

        The completeness rule the constructor enforces is per platform, and a mixed setup --
        three roles on one provider, coding on another -- has no complete platform at all. The
        rule here is the one a role map actually owes: all four roles present exactly once.
        Nothing fabricates a platform's missing rows from the deployment's environment to
        satisfy the per-platform rule -- a fabricated row is a model nobody chose.
        """
        missing = [role.value for role in ModelRole if role not in configs]
        if missing:
            msg = f"a model setup must resolve all four roles; missing: {', '.join(missing)}"
            raise ModelConfigurationError(msg)
        service = cls.__new__(cls)
        service._configs = {(config.platform, role): config for role, config in configs.items()}
        service._platforms = tuple(dict.fromkeys(config.platform for config in configs.values()))
        service._tier_configs = {}
        service._role_platforms = {role: config.platform for role, config in configs.items()}
        return service

    def platform_for_role(self, role: ModelRole) -> AgentPlatform | None:
        """Return the platform one role is pinned to, or nothing for an env-backed service.

        Only a role-addressed service has an answer: an environment-backed one resolves a
        role on whichever platform the caller asks for, so the caller's platform stands.
        """
        return self._role_platforms.get(role)

    def configured_platforms(self) -> tuple[AgentPlatform, ...]:
        """Return the platforms a feature may actually be submitted on.

        Reported to the client so a dropdown offers only what this deployment can run. A
        submission on an unconfigured platform would otherwise fail at its first model call,
        minutes later, on a worker.
        """
        return self._platforms

    def is_configured(self, platform: AgentPlatform) -> bool:
        """Return whether this deployment can run a feature on one platform."""
        return platform in self._platforms

    def is_tier_configured(self, platform: AgentPlatform, tier: PerformanceTier) -> bool:
        """Return whether one (platform, tier) pairing is selectable on this deployment."""
        return (platform, tier) in self._tier_configs

    def configured_options(self) -> tuple[tuple[AgentPlatform, PerformanceTier], ...]:
        """Return every (platform, tier) a feature may actually be submitted on.

        Ordered by platform and then from the cheapest tier up, which is the order a person
        is shown them. A tier with nothing configured is simply absent, the same preference
        the platform-level rule already expresses. Only the environment-backed tiers are
        candidates: ``custom`` is a user-authored setup, never a deployment pairing, and a
        (platform, custom) entry here would be a selectable-looking option that resolves
        nothing.
        """
        return tuple(
            (platform, tier)
            for platform in AgentPlatform
            for tier in PerformanceTier.env_backed()
            if (platform, tier) in self._tier_configs
        )

    def get_model_config(
        self,
        role: ModelRole,
        *,
        platform: AgentPlatform,
        tier: PerformanceTier = PerformanceTier.HIGH,
    ) -> ModelConfig:
        """Return the configured selection for one role on one platform and tier.

        ``platform`` is required rather than defaulted. A default here would quietly restore
        the deployment-wide provider this platform no longer has, and would do it in the one
        place every agent passes through. ``tier`` defaults to ``HIGH`` because that is the
        unsuffixed configuration: a caller that never heard of tiers gets exactly what it
        always got.

        A role-addressed service answers only on the role's own pinned platform, and the
        mismatch raises by name: answering a coding question asked of the wrong provider is
        how a Claude model would be sent to the Responses API.
        """
        if self._role_platforms:
            pinned = self._role_platforms[role]
            if pinned is not platform:
                msg = (
                    f"the {role.value} role of this model setup is pinned to {pinned.value}; "
                    f"it does not resolve on {platform.value}"
                )
                raise ModelConfigurationError(msg)
            return self._configs[(platform, role)]
        if tier is not PerformanceTier.HIGH:
            tier_roles = self._tier_configs.get((platform, tier))
            if tier_roles is None:
                msg = (
                    f"{platform.value} is not configured for the {tier.value} performance "
                    f"tier; configure "
                    f"{tier_model_role_variables(platform, tier, role).model_variable}"
                )
                raise ModelConfigurationError(msg)
            return tier_roles[role]
        try:
            return self._configs[(platform, role)]
        except KeyError as error:
            msg = (
                f"{platform.value} is not configured for this deployment; "
                f"configure {MODEL_ROLE_VARIABLES[(platform, role)].model_variable}"
            )
            raise ModelConfigurationError(msg) from error

    def for_tier(self, tier: PerformanceTier) -> ModelConfigService:
        """Return a view of this service scoped to one tier's resolved selections.

        The view is a plain ``ModelConfigService`` whose per-platform answers *are* the
        tier's, so everything that already resolves models through this type -- the router,
        the settings accessors, the adapters behind them -- resolves the tier without
        learning a new question. ``HIGH`` returns this service itself: the unsuffixed
        configuration is the high tier, byte for byte.
        """
        if tier is PerformanceTier.HIGH:
            return self
        configs: dict[tuple[AgentPlatform, ModelRole], ModelConfig] = {}
        for (platform, configured_tier), roles in self._tier_configs.items():
            if configured_tier is tier:
                for role, config in roles.items():
                    configs[(platform, role)] = config
        if not configs:
            msg = (
                f"no agent platform is configured for the {tier.value} performance tier; "
                f"a feature pinned to it cannot resolve a model"
            )
            raise ModelConfigurationError(msg)
        return ModelConfigService(configs)

    def roles_resolved_from_legacy_configuration(
        self,
    ) -> tuple[tuple[AgentPlatform, ModelRole], ...]:
        """Return the platform roles still running on a pre-roles variable.

        Reported at startup because a deployment where review or scoped fix falls back to one
        of the original two variables does not have independent role configuration yet. Named
        by platform as well as role now that two providers can each be half-migrated.
        """
        return tuple(
            key
            for key, config in self._configs.items()
            if config.resolved_from_legacy_model_variable
        )

    def describe(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return every configured selection for a startup log. Never includes a credential.

        Keyed by what is actually present rather than assuming four roles per platform: a
        role-addressed service legitimately holds a platform that answers one role only.
        """
        return {
            platform.value: {
                role.value: self._configs[(platform, role)].as_metadata()
                for role in ModelRole
                if (platform, role) in self._configs
            }
            for platform in self._platforms
        }


def build_model_config_service(
    *,
    platforms: Mapping[AgentPlatform, PlatformModelSettings],
    tiers: Mapping[tuple[AgentPlatform, PerformanceTier], PlatformModelSettings] | None = None,
    unsupported_reasoning: Mapping[str, Iterable[str]] | None = None,
) -> ModelConfigService:
    """Resolve, validate and normalize every configured platform's roles in one pass.

    An explicitly configured role always wins. Where one is unset, the pre-roles variable is
    used and recorded as such.

    Completeness is judged per platform, and **at least one** platform must resolve
    completely. Refusing to start because a second provider is unconfigured would be wrong: a
    deployment that only uses Claude should not need ``OPENAI_REVIEW_MODEL``. But a
    *partially* configured platform is a hard failure for that platform -- three of four roles
    set is a configuration mistake, not a preference, and it must be refused at startup rather
    than discovered when a feature reaches its scoped-fix path. Substituting an unrelated
    model for a missing one is exactly what a deployment cannot be allowed to discover from
    its output.

    ``tiers`` carries the tier-suffixed presets, and the completeness rule extends to them at
    tier granularity: all four roles set makes a (platform, tier) selectable, none set means
    the deployment does not offer it, and a partial tier is refused at startup by the missing
    variable's name. The ``HIGH`` entries are overlays -- ``OPENAI_HIGH_CODING_MODEL``, when
    set, overrides its unsuffixed twin -- because the unsuffixed variables *are* the high
    tier and no deployment breaks by upgrading.
    """
    unsupported = normalize_unsupported_reasoning(unsupported_reasoning)
    tier_settings = dict(tiers or {})
    configs: dict[tuple[AgentPlatform, ModelRole], ModelConfig] = {}
    tier_configs: dict[tuple[AgentPlatform, PerformanceTier], dict[ModelRole, ModelConfig]] = {}
    for platform, settings in platforms.items():
        high = tier_settings.get((platform, PerformanceTier.HIGH))
        merged, variables, strict_roles = _overlay_high_tier(platform, settings, high)
        resolved = _resolve_platform(
            platform,
            merged,
            unsupported,
            variables=variables,
            strict_capability_roles=strict_roles,
        )
        if resolved is None:
            continue
        configs.update(resolved)
    for (platform, tier), settings in tier_settings.items():
        if tier is PerformanceTier.HIGH:
            continue
        resolved = _resolve_platform(
            platform,
            settings,
            unsupported,
            tier=tier,
            variables={role: tier_model_role_variables(platform, tier, role) for role in ModelRole},
            strict_capability_roles=frozenset(ModelRole),
        )
        if resolved is None:
            continue
        tier_configs[(platform, tier)] = {role: config for (_, role), config in resolved.items()}
    return ModelConfigService(configs, tier_configs=tier_configs)


def _overlay_high_tier(
    platform: AgentPlatform,
    base: PlatformModelSettings,
    high: PlatformModelSettings | None,
) -> tuple[PlatformModelSettings, Mapping[ModelRole, ModelRoleVariables], frozenset[ModelRole]]:
    """Merge the ``HIGH``-suffixed overrides onto the unsuffixed configuration, per field.

    Returns the merged settings, the variable names each resolved value is attributable to,
    and which roles took any tier-suffixed value at all. Those roles get the strict capability
    check: a person who wrote a tier variable asked for that pairing explicitly, so an
    unsupported one is a mistake to refuse -- while an untouched unsuffixed pairing keeps its
    long-standing behaviour of dropping the level and recording the normalization.
    """
    if high is None:
        return (
            base,
            {role: MODEL_ROLE_VARIABLES[(platform, role)] for role in ModelRole},
            frozenset(),
        )
    models: dict[ModelRole, str] = {}
    reasoning: dict[ModelRole, str | None] = {}
    max_tokens: dict[ModelRole, int | None] = {}
    variables: dict[ModelRole, ModelRoleVariables] = {}
    strict: set[ModelRole] = set()
    for role in ModelRole:
        base_variables = MODEL_ROLE_VARIABLES[(platform, role)]
        high_variables = tier_model_role_variables(platform, PerformanceTier.HIGH, role)
        high_model = (high.models.get(role) or "").strip()
        high_reasoning = (high.reasoning.get(role) or "").strip()
        high_max_tokens = high.max_tokens.get(role)
        # Strictness follows the pairing the capability declaration is about. A tier that
        # names a model or an effort has asked for that pairing explicitly; a tier that only
        # bounds output has not.
        if high_model or high_reasoning:
            strict.add(role)
        models[role] = high_model or (base.models.get(role) or "")
        reasoning[role] = high_reasoning or base.reasoning.get(role)
        max_tokens[role] = (
            high_max_tokens if high_max_tokens is not None else base.max_tokens.get(role)
        )
        variables[role] = ModelRoleVariables(
            model_variable=(
                high_variables.model_variable if high_model else base_variables.model_variable
            ),
            reasoning_variable=(
                high_variables.reasoning_variable
                if high_reasoning
                else base_variables.reasoning_variable
            ),
            max_tokens_variable=(
                high_variables.max_tokens_variable
                if high_max_tokens is not None
                else base_variables.max_tokens_variable
            ),
            # A role the overlay named a model for no longer reads the pre-roles fallback:
            # the deployment has stated what it wants, so nothing older may fill it in.
            legacy_model_variable=(None if high_model else base_variables.legacy_model_variable),
            legacy_reasoning_variable=(
                None if high_reasoning else base_variables.legacy_reasoning_variable
            ),
        )
    merged = PlatformModelSettings(
        models=models,
        reasoning=reasoning,
        max_tokens=max_tokens,
        legacy_models=base.legacy_models,
        legacy_reasoning=base.legacy_reasoning,
    )
    return merged, variables, frozenset(strict)


def _resolve_platform(
    platform: AgentPlatform,
    settings: PlatformModelSettings,
    unsupported: Mapping[str, frozenset[str]],
    *,
    tier: PerformanceTier = PerformanceTier.HIGH,
    variables: Mapping[ModelRole, ModelRoleVariables] | None = None,
    strict_capability_roles: frozenset[ModelRole] = frozenset(),
) -> dict[tuple[AgentPlatform, ModelRole], ModelConfig] | None:
    """Resolve one platform's four roles, or return nothing when it names no model at all.

    The same function resolves a tier: the caller passes the tier's variable names and the
    completeness rule applies unchanged at tier granularity. ``strict_capability_roles`` are
    the roles whose (model, effort) pairing was named by a tier-suffixed variable; for those,
    a pairing the deployment has declared unsupported is a configuration error at startup
    rather than a normalization, because a preset that silently ran at the provider default
    would not be the preset the operator priced.
    """
    role_variables = variables or {
        role: MODEL_ROLE_VARIABLES[(platform, role)] for role in ModelRole
    }
    resolved: dict[tuple[AgentPlatform, ModelRole], ModelConfig] = {}
    named: list[str] = []
    for role in ModelRole:
        role_vars = role_variables[role]
        configured_model = (settings.models.get(role) or "").strip()
        legacy_model = ((settings.legacy_models or {}).get(role) or "").strip()
        model = configured_model or (legacy_model if role_vars.legacy_model_variable else "")
        if not model:
            name = role_vars.model_variable
            if role_vars.legacy_model_variable is not None:
                name = f"{name} (or {role_vars.legacy_model_variable})"
            named.append(name)
            continue
        requested = normalize_reasoning_level(
            role_vars.reasoning_variable, settings.reasoning.get(role)
        )
        reasoning_variable: str | None = role_vars.reasoning_variable
        if requested is None and role_vars.legacy_reasoning_variable is not None:
            requested = normalize_reasoning_level(
                role_vars.legacy_reasoning_variable, (settings.legacy_reasoning or {}).get(role)
            )
            if requested is not None:
                reasoning_variable = role_vars.legacy_reasoning_variable
        effective = supported_reasoning(model, requested, unsupported)
        if role in strict_capability_roles and requested is not None and effective != requested:
            msg = (
                f"{reasoning_variable} names a reasoning level MODEL_REASONING_UNSUPPORTED "
                "declares the configured model does not accept; a performance tier must not "
                "name an unsupported pairing"
            )
            raise ModelConfigurationError(msg)
        resolved[(platform, role)] = ModelConfig(
            platform=platform,
            role=role,
            model=model,
            reasoning=effective,
            requested_reasoning=requested,
            max_tokens=settings.max_tokens.get(role),
            model_variable=(
                role_vars.model_variable
                if configured_model
                else (role_vars.legacy_model_variable or role_vars.model_variable)
            ),
            reasoning_variable=reasoning_variable if requested is not None else None,
            resolved_from_legacy_model_variable=not configured_model,
            routing_reason=_ROLE_ROUTING_REASONS[role],
            tier=tier,
        )
    if not resolved:
        # Nothing at all is set for this platform, so the deployment has simply not chosen to
        # offer it. That is a preference; the half-configured case below is a mistake.
        return None
    if named:
        msg = f"missing required configuration: {', '.join(named)}"
        raise ModelConfigurationError(msg)
    return resolved


__all__ = [
    "ANTHROPIC_EFFORT_MAX_TOKENS_FLOORS",
    "ANTHROPIC_UNSET_EFFORT_FLOOR",
    "MODEL_ROLE_VARIABLES",
    "PLATFORM_LABELS",
    "REASONING_LEVELS",
    "TIER_LABELS",
    "AgentPlatform",
    "AnthropicCeilingJudgement",
    "ModelConfig",
    "ModelConfigService",
    "ModelConfigurationError",
    "ModelRole",
    "ModelRoleVariables",
    "ModelSetupRole",
    "PerformanceTier",
    "PlatformModelSettings",
    "anthropic_effort_floor",
    "build_model_config_service",
    "judge_anthropic_ceiling",
    "model_config_service_for_setup",
    "normalize_max_output_tokens",
    "normalize_reasoning_level",
    "normalize_unsupported_reasoning",
    "parse_model_setup_roles",
    "require_anthropic_floor",
    "supported_reasoning",
    "tier_model_role_variables",
    "validate_model_setup",
]
