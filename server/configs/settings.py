"""Environment-backed platform settings with version-controlled YAML policy sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from configs.model_roles import (
    MODEL_ROLE_VARIABLES,
    AgentPlatform,
    ModelConfig,
    ModelConfigService,
    ModelConfigurationError,
    ModelRole,
    ModelRoleVariables,
    PerformanceTier,
    PlatformModelSettings,
    build_model_config_service,
    judge_anthropic_ceiling,
    model_config_service_for_setup,
    normalize_context_window_tokens,
    normalize_max_output_tokens,
    normalize_reasoning_level,
    normalize_unsupported_reasoning,
    require_anthropic_floor,
    supported_reasoning,
    tier_model_role_variables,
)

CONFIG_DIRECTORY = Path(__file__).resolve().parent
SERVER_ROOT = CONFIG_DIRECTORY.parent
# The deployment `.env` sits at the repository root beside `docker-compose.yml`, which reads
# the same file. Resolving it from this module rather than the working directory keeps a
# command run from `server/` and one run from the repository root reading the same settings;
# a bare relative name silently found neither when the project moved under `server/`.
REPOSITORY_ROOT = SERVER_ROOT.parent
CONFIG_FILE = CONFIG_DIRECTORY / "config.yaml"
AGENT_CONFIG_FILE = CONFIG_DIRECTORY / "agent.yaml"
REQUIRED_AGENT_NAMES = frozenset({"product_manager", "planner", "engineer", "reviewer", "github"})

# The effort floors moved to `configs.model_roles` when user-authored setups arrived: a setup
# has no `Settings` instance to hang a validator on, and two copies of the table is how two
# authorities come to disagree. The validator below still enforces them here, over the
# environment-backed tiers, through the same extracted function the setup predicate calls.


def _anthropic_effort_field(prefix: str, role: ModelRole) -> str:
    """Name the effort field one role reads, mirroring `_tier_platform_settings`' rule.

    The reasoning role's effort has no role segment (`ANTHROPIC_REASONING_EFFORT`, not
    `ANTHROPIC_REASONING_REASONING_EFFORT`), and every other role carries one.
    """
    if role is ModelRole.REASONING:
        return f"{prefix}reasoning_effort"
    return f"{prefix}{role.value}_reasoning_effort"


class AgentSettings(BaseModel):
    """Operational settings for one specialized workflow agent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # Which configured model role this agent's work belongs to, where one applies. Absent only
    # for non-model policy entries such as the GitHub publisher, which makes no model call at
    # all. Every live LLM boundary either reads this role or receives an explicit routed role
    # at construction.
    #
    # There is no per-agent model variable beside this any more. It was a `Literal` of two
    # OpenAI variable names, consulted only when an agent declared no role, and naming one
    # provider's variables in an agent's policy is exactly what stopped the platform being
    # able to run on a second one.
    model_role: ModelRole | None = None
    temperature: float = Field(ge=0, le=2)
    # How hard a reasoning model is asked to think. Absent means "send nothing", which
    # leaves the provider's own default and is what every deployment did before this
    # existed. Named per agent because the coder and the reviewer are different jobs: the
    # levels a model accepts differ between models, so an unsupported value is rejected by
    # the provider rather than silently ignored, and it belongs beside the model choice.
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0)

    @field_validator("model_role", mode="before")
    @classmethod
    def model_role_must_be_known(cls, value: ModelRole | str | None) -> ModelRole | None:
        """Read the role from its YAML name, rejecting one this platform does not define."""
        if value is None or isinstance(value, ModelRole):
            return value
        try:
            return ModelRole(value)
        except ValueError as error:
            known = ", ".join(role.value for role in ModelRole)
            msg = f"model_role must be one of {known}"
            raise ValueError(msg) from error


# Which deployment variable carries each agent's own effort. Named here rather than derived
# from the agent name so a variable this platform reads is always greppable, and so adding an
# agent does not silently start reading an environment variable nobody declared.
_AGENT_REASONING_EFFORT_VARIABLES: Mapping[str, str] = {
    "planner": "OPENAI_PLANNER_REASONING_EFFORT",
    "product_manager": "OPENAI_PRODUCT_MANAGER_REASONING_EFFORT",
    "recon": "OPENAI_RECON_REASONING_EFFORT",
}

# Whose account the startup bootstrap sets a password on when the deployment names nobody.
#
# A constant rather than a literal on the field, because `blank_string_is_default` has to
# restore the same value when Compose hands the setting an empty string -- and a second copy
# of it would be the next thing to drift apart. Migration 0031 writes this same address as
# the administrator's `subject`; the two must agree or the bootstrap finds nobody.
DEFAULT_BOOTSTRAP_ADMIN_EMAIL = "akhilesh@appbroda.com"


class Settings(BaseSettings):
    """Validated platform configuration from environment, `.env`, and YAML policy files."""

    model_config = SettingsConfigDict(
        # Later files win, so a server-local override beats the shared deployment file.
        env_file=(REPOSITORY_ROOT / ".env", SERVER_ROOT / ".env"),
        env_file_encoding="utf-8",
        # Compose-only database and port variables may share the deployment .env file.
        # They are intentionally ignored by the API settings model.
        extra="ignore",
        # The model-routing configuration is named for what an operator sets in `.env`:
        # `MODEL_ROUTER_MIN_CONFIDENCE` and friends. Pydantic reserves the `model_` prefix for
        # its own attributes, and none of the fields below collide with one.
        protected_namespaces=(),
    )

    openai_reasoning_model: str = Field(min_length=1)
    openai_coding_model: str = Field(min_length=1)
    # Maximum-effort reasoning responses can legitimately take several minutes before the
    # provider returns the first complete artifact. Keep that deadline in deployment
    # configuration: a 120-second agent-policy default caused the SDK to abandon and replay the
    # same Product Manager request four times without ever allowing one attempt to finish.
    openai_reasoning_timeout_seconds: int = Field(default=1_800, gt=0)
    # How long a streamed call may stay silent before it has said anything at all. Not a
    # thinking budget and deliberately not per-model: the Responses API emits
    # `response.created` before the model reasons, so this measures whether the provider
    # began answering. The deadline above still governs how long it may then think, applied
    # between events, and nothing here shortens it.
    #
    # In deployment configuration rather than as a constant because it is the one number in
    # this design chosen by judgement instead of measurement -- no instrument records
    # time-to-first-event yet. The journal bounds it only indirectly: the fastest complete
    # `gpt-5.6-sol` coding call is 7 seconds, which caps that call's first event at 7, while
    # `gpt-6-astra`'s fastest is 114 and so caps nothing useful. Until the measurement exists,
    # an operator who sees re-issues in the log must be able to raise this without a deploy.
    first_event_timeout_seconds: int = Field(default=180, gt=0)
    # The four authoritative per-role effort settings. Empty keeps compatibility with an
    # environment that predates role-specific configuration; legacy names below are consulted
    # centrally and never by an agent.
    openai_reasoning_effort: str = ""
    openai_coding_reasoning_effort: str = ""
    openai_review_reasoning_effort: str = ""
    openai_scoped_fix_reasoning_effort: str = ""
    # Per-agent effort, for the case a role cannot express: two agents sharing one role that
    # are not the same job. The product manager and the planner are both `reasoning`, and the
    # planner emits the architecture, the contract and the execution plan in a single
    # response, so easing it off `max` must not ease the product manager off with it.
    #
    # These outrank the checked-in policy file deliberately. The file travels inside the
    # image, so changing it costs a commit and a rebuild, while these are read from the
    # deployment at start-up -- and an operator comparing effort levels on a live deployment
    # must not have to rebuild to do it. The policy file supplies the default; the deployment
    # has the last word.
    openai_planner_reasoning_effort: str = ""
    openai_product_manager_reasoning_effort: str = ""
    openai_recon_reasoning_effort: str = ""
    openai_review_model: str = ""
    openai_scoped_fix_model: str = ""

    # The second platform's four roles. Empty means this deployment does not offer Anthropic,
    # which is a preference and starts normally; some-but-not-all set is a configuration
    # mistake and is refused at startup. There are deliberately no legacy fallbacks here --
    # no deployment predates these names, so a fallback could only let a half-configured
    # provider look configured.
    anthropic_reasoning_model: str = ""
    anthropic_coding_model: str = ""
    anthropic_review_model: str = ""
    anthropic_scoped_fix_model: str = ""
    anthropic_reasoning_effort: str = ""
    anthropic_coding_reasoning_effort: str = ""
    anthropic_review_reasoning_effort: str = ""
    anthropic_scoped_fix_reasoning_effort: str = ""
    # How many tokens one Messages API response may produce. Required by that API and unused
    # by the Responses API, and deliberately per role rather than one platform-wide number:
    # the reviewer's output is a findings list, the engineer's is a set of complete file
    # bodies for a multi-file feature. A number generous enough for the second wastes nothing
    # on the first, but a number sized for the first truncates the second -- and a truncation
    # arrives as `stop_reason: "max_tokens"`, which reads as a malformed model response rather
    # than as a budget that was too small. That is the defect at `agent.yaml:44-53` again,
    # where a deadline shorter than the work turned "slow" into "impossible" and hid it as a
    # retryable fault.
    #
    # Defaulted rather than required, like the reasoning deadline above: a bound on one call
    # is an operational limit, and what a deployment must state explicitly is which *models*
    # execute a role.
    #
    # Sized for models whose thinking spends the same bound the answer does. The previous
    # defaults (32k/64k/16k/16k) predated adaptive thinking and killed every Anthropic run of
    # the 175-180 matrix before a single response could be judged: AB-Feature-175's planner
    # truncated twice at 32k under effort `max`, and AB-Feature-180's reviewer died at 16k
    # under effort `high` after its Engineer had already produced a working implementation.
    # `anthropic_bounds_must_hold_a_thinking_response` below enforces the floor these numbers
    # clear, so a deployment that lowers one is told at startup instead of at review time.
    anthropic_reasoning_max_tokens: int = Field(default=64_000, gt=0)
    anthropic_coding_max_tokens: int = Field(default=128_000, gt=0)
    anthropic_review_max_tokens: int = Field(default=48_000, gt=0)
    anthropic_scoped_fix_max_tokens: int = Field(default=32_000, gt=0)

    # The performance-tier presets: `{PROVIDER}_{TIER}_{ROLE}_MODEL` and friends, the existing
    # naming with a tier segment. The unsuffixed variables above *are* the `high` tier, so a
    # deployment that never sets one of these behaves exactly as it always has; a `HIGH_`
    # variable, when set, overrides its unsuffixed twin. Completeness is judged per
    # (platform, tier): all four roles set makes the pair selectable, none set means this
    # deployment does not offer it, and a partial tier is refused at startup by the missing
    # variable's name. Efforts follow the platform convention -- empty sends nothing and
    # keeps the provider's default. The Anthropic `MAX_TOKENS` bounds fall back to the
    # platform's per-role bounds above when a tier does not name its own.
    openai_low_reasoning_model: str = ""
    openai_low_reasoning_effort: str = ""
    openai_low_coding_model: str = ""
    openai_low_coding_reasoning_effort: str = ""
    openai_low_review_model: str = ""
    openai_low_review_reasoning_effort: str = ""
    openai_low_scoped_fix_model: str = ""
    openai_low_scoped_fix_reasoning_effort: str = ""
    openai_medium_reasoning_model: str = ""
    openai_medium_reasoning_effort: str = ""
    openai_medium_coding_model: str = ""
    openai_medium_coding_reasoning_effort: str = ""
    openai_medium_review_model: str = ""
    openai_medium_review_reasoning_effort: str = ""
    openai_medium_scoped_fix_model: str = ""
    openai_medium_scoped_fix_reasoning_effort: str = ""
    openai_high_reasoning_model: str = ""
    openai_high_reasoning_effort: str = ""
    openai_high_coding_model: str = ""
    openai_high_coding_reasoning_effort: str = ""
    openai_high_review_model: str = ""
    openai_high_review_reasoning_effort: str = ""
    openai_high_scoped_fix_model: str = ""
    openai_high_scoped_fix_reasoning_effort: str = ""
    openai_ultra_reasoning_model: str = ""
    openai_ultra_reasoning_effort: str = ""
    openai_ultra_coding_model: str = ""
    openai_ultra_coding_reasoning_effort: str = ""
    openai_ultra_review_model: str = ""
    openai_ultra_review_reasoning_effort: str = ""
    openai_ultra_scoped_fix_model: str = ""
    openai_ultra_scoped_fix_reasoning_effort: str = ""
    anthropic_low_reasoning_model: str = ""
    anthropic_low_reasoning_effort: str = ""
    anthropic_low_coding_model: str = ""
    anthropic_low_coding_reasoning_effort: str = ""
    anthropic_low_review_model: str = ""
    anthropic_low_review_reasoning_effort: str = ""
    anthropic_low_scoped_fix_model: str = ""
    anthropic_low_scoped_fix_reasoning_effort: str = ""
    anthropic_low_reasoning_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_low_coding_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_low_review_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_low_scoped_fix_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_medium_reasoning_model: str = ""
    anthropic_medium_reasoning_effort: str = ""
    anthropic_medium_coding_model: str = ""
    anthropic_medium_coding_reasoning_effort: str = ""
    anthropic_medium_review_model: str = ""
    anthropic_medium_review_reasoning_effort: str = ""
    anthropic_medium_scoped_fix_model: str = ""
    anthropic_medium_scoped_fix_reasoning_effort: str = ""
    anthropic_medium_reasoning_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_medium_coding_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_medium_review_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_medium_scoped_fix_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_high_reasoning_model: str = ""
    anthropic_high_reasoning_effort: str = ""
    anthropic_high_coding_model: str = ""
    anthropic_high_coding_reasoning_effort: str = ""
    anthropic_high_review_model: str = ""
    anthropic_high_review_reasoning_effort: str = ""
    anthropic_high_scoped_fix_model: str = ""
    anthropic_high_scoped_fix_reasoning_effort: str = ""
    anthropic_high_reasoning_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_high_coding_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_high_review_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_high_scoped_fix_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_ultra_reasoning_model: str = ""
    anthropic_ultra_reasoning_effort: str = ""
    anthropic_ultra_coding_model: str = ""
    anthropic_ultra_coding_reasoning_effort: str = ""
    anthropic_ultra_review_model: str = ""
    anthropic_ultra_review_reasoning_effort: str = ""
    anthropic_ultra_scoped_fix_model: str = ""
    anthropic_ultra_scoped_fix_reasoning_effort: str = ""
    anthropic_ultra_reasoning_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_ultra_coding_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_ultra_review_max_tokens: int | None = Field(default=None, gt=0)
    anthropic_ultra_scoped_fix_max_tokens: int | None = Field(default=None, gt=0)

    # Pre-four-role names retained only as fallbacks while deployments migrate. They are not
    # active roles and are never emitted by a new routing decision.
    openai_coding_effort: str = ""
    openai_coding_reasoning: str = ""
    openai_review_reasoning: str = ""
    openai_fix_model: str = ""
    openai_fix_reasoning: str = ""
    # Which effort levels a configured model does not accept, declared by the deployment that
    # knows: `{"gpt-5.3-codex": ["max"]}`. A declared level is omitted from the request, which
    # leaves the provider's default; anything undeclared is sent and the provider remains the
    # authority. Kept as configuration rather than a table in code, because the accepted set
    # differs per model and changes without this repository being told.
    #
    # Held as text and parsed below rather than declared as a mapping. A mapping field is
    # JSON-parsed by pydantic-settings before any validator of ours runs, so the entirely
    # ordinary `MODEL_REASONING_UNSUPPORTED=` in a hand-edited `.env` failed the whole process
    # with an opaque source error instead of meaning "nothing is declared".
    model_reasoning_unsupported: str = ""
    # Which configured models reject the `temperature` parameter outright, declared by the
    # deployment that knows: `["gpt-6-astra"]`. AB-Feature-210 died on its first provider
    # call because a prefix test in the adapter guessed for a family it had never seen; the
    # accepted set changes without this repository being told, so it is configuration, not a
    # table in code. The gpt-5 Responses family is already known to the adapter and needs no
    # declaration here. Same held-as-text posture as MODEL_REASONING_UNSUPPORTED above, for
    # the same `.env` parse reason.
    model_temperature_unsupported: str = ""
    # How many output tokens each model can produce at most, declared by the deployment that
    # knows: `{"claude-sonnet-5": 128000}`. The one input the AB-Feature-181 refusal needs and
    # nothing in this repository can know for itself -- the truncation diagnostic says
    # outright that the adapter does not know the ceiling. A model absent from the declaration
    # has no known ceiling and the ceiling checks do not fire for it; a table in code would
    # refuse a valid configuration the day a provider raises a limit. Same held-as-text
    # posture as MODEL_REASONING_UNSUPPORTED above, for the same `.env` parse reason.
    model_max_output_tokens: str = ""
    # How many context tokens each model can read, declared by the deployment that knows:
    # `{"gpt-6-astra": 400000}`. The one input the repository-snapshot budget derivation
    # needs and nothing in this repository can know for itself -- AB-Feature-212 died at r1
    # because the snapshot budget was a constant sized for one model era while the required
    # union scaled with the model that wrote it. A model absent from the declaration falls
    # back to the default budget (absence of a declaration is never a punishment, the
    # temperature-pattern posture). Same held-as-text posture as MODEL_REASONING_UNSUPPORTED
    # above, for the same `.env` parse reason.
    model_context_window_tokens: str = ""
    # Which models can be shown an image, declared by the deployment that knows:
    # `["gpt-6-astra", "claude-opus-5"]`. Exactly the kind of fact this repository cannot
    # know and is not told when it changes -- the same argument the three declarations above
    # make.
    #
    # And the one whose default points the other way. An undeclared model is treated as *not*
    # vision-capable, where an undeclared model is treated as supporting every effort level
    # and having no output ceiling. That asymmetry is deliberate: for the others, absence of
    # a declaration means "the provider remains the authority", and the cost of guessing
    # wrong is a request the provider answers. Here the cost of guessing "probably fine" is a
    # provider 400 in the middle of a run, or worse -- a model that accepts the blocks,
    # ignores them, and answers from the prose, which is indistinguishable from a model that
    # read them. Same held-as-text posture as the three above, for the same `.env` parse
    # reason.
    model_vision_capable: str = ""
    # Below this, a classification model's answer is not trusted and the finding routes to the
    # primary coding role instead. Uncertainty never selects the scoped-fix role.
    model_router_min_confidence: float = Field(default=0.80, ge=0, le=1)
    # The deployment's shared administrative credential. Optional now, and unset is the
    # target state: with per-user passwords and per-user tokens there is an account for every
    # person, and anybody holding this key is an unnamed administrator with every permission.
    #
    # Not removed. Deployments, the disaster-recovery runbook and the canary all use it, and
    # break-glass access when the *database* is the thing that is broken is a real
    # requirement -- a per-user token cannot be resolved without the directory. Where it used
    # to be kept, keep an admin token issued in advance instead.
    #
    # When unset, only per-user tokens authenticate. `PlatformAuthenticator` already answers
    # 503 for "this deployment cannot check" and 401 for "wrong", and that distinction stays
    # correct: with no directory *and* no key it can check nothing.
    platform_api_key: SecretStr | None = None
    # Seals provider credentials at rest. Optional: without it the platform keeps its
    # original behaviour of never persisting a provider secret, and the credential endpoints
    # answer "this deployment does not store credentials". It lives in the environment and
    # never in the database it protects -- a key stored beside the ciphertext protects
    # nothing. Generate one with `services.secrets.generate_encryption_key`.
    secret_encryption_key: SecretStr | None = None
    # Versioned keys retained only while rows sealed by an older deployment are lazily
    # re-encrypted. Supply as a JSON list, then remove a key after every row has been used or
    # explicitly rotated. Duplicate versions are rejected by the store.
    secret_encryption_previous_keys: list[SecretStr] = Field(default_factory=list)
    # How long a password login's token stays usable. Bounded, unlike an administrator-issued
    # API token, because a session is a browser tab somebody may walk away from.
    # `token_is_usable` enforces it; nothing else has to know.
    session_token_ttl_hours: int = Field(default=12, ge=1, le=720)
    # Which account the startup bootstrap sets a password on, and what to set. The password is
    # a deploy-time secret: it is never in version control, never logged, and the bootstrap
    # refuses to overwrite an existing hash so a container restart with a stale value cannot
    # silently reset the administrator's password. Remove it from the environment after the
    # first boot; `must_change_password` is what makes forgetting survivable.
    #
    # The default is a module constant rather than a literal here, because
    # `blank_string_is_default` below has to restore the same value and a second copy of it
    # would be the next thing to drift. Bare `= "..."` was not enough on its own: Compose
    # passes this through as `BOOTSTRAP_ADMIN_EMAIL=${BOOTSTRAP_ADMIN_EMAIL:-}`, so a
    # variable absent from `.env` arrives as an empty *string*, which pydantic accepts as a
    # perfectly valid `str` and which then overrides the default. The first deploy of this
    # change looked the administrator up by `""`, found nobody, logged `no_such_account` and
    # set no password -- leaving a console nobody could sign in to.
    bootstrap_admin_email: str = DEFAULT_BOOTSTRAP_ADMIN_EMAIL
    bootstrap_admin_password: SecretStr | None = None
    # How many login attempts one source address, or one email, may make in a window. Rate
    # limiting rather than account lockout, deliberately: lockout stops an attacker and hands
    # anybody a denial of service against a known address. Counted in Redis, because an
    # in-process counter is worthless behind more than one worker.
    login_rate_limit_attempts: int = Field(default=10, ge=1, le=1000)
    login_rate_limit_window_seconds: int = Field(default=900, ge=1, le=86_400)
    environment: Literal["test", "development", "production"] = "development"
    build_revision: str = "local"
    expected_build_revision: str | None = None
    expected_workflow_schema_version: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    database_url: str = Field(min_length=1)
    # Above SQLAlchemy's default of five. A live feature run holds connections for as long
    # as it is flushing state for several concurrent workstreams, and the lease-renewal loop
    # that proves the run is still alive needs one of its own. On the default pool those
    # compete, and losing that race is what abandoned AB-Feature-108 seventy minutes in.
    database_pool_size: int = Field(default=20, ge=1, le=200)
    database_max_overflow: int = Field(default=20, ge=0, le=200)
    redis_url: str = Field(min_length=1)
    workspace_root: Path = Field(default=Path("/workspaces"))
    max_request_body_bytes: int = Field(default=1_048_576, ge=1_024, le=10_485_760)
    max_workspace_file_bytes: int = Field(default=1_048_576, ge=1_024, le=10_485_760)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    max_clarification_rounds: int = Field(ge=1)
    max_engineer_review_cycles: int = Field(ge=1)
    max_child_review_cycles: int = Field(default=5, ge=1)
    max_implementation_retries: int = Field(default=4, ge=0)
    max_validation_retries: int = Field(default=2, ge=0)
    max_repository_setup_retries: int = Field(default=1, ge=0)
    max_integration_review_cycles: int = Field(default=5, ge=1)
    max_contract_revision_cycles: int = Field(default=3, ge=1)
    # Whether a blocking review finding has to name what it derives from: a scoped
    # requirement, a contract section, a failed required validation command, or an
    # implementation expectation. A finding that names none of those becomes advisory --
    # recorded on the result and published in the pull-request body for a person to judge,
    # rather than spending one of a bounded number of coding attempts.
    #
    # Off by default, and deliberately temporary. It exists so the behaviour change can be
    # measured against the same fixtures with it on and off; once the numbers are in, the
    # winning behaviour becomes the only one and this flag is deleted. A permanent flag is a
    # permanent second code path.
    bounded_review_scope: bool = False
    max_parallel_workstreams: int = Field(default=4, ge=1, le=64)
    # How long a feature may claim to be running, with nothing holding a live lease on it,
    # before the recovery sweep gives it a terminal status. Longer than the slowest single
    # step by a wide margin: being wrong here ends work that was progressing normally.
    feature_run_stale_after_seconds: float = Field(default=2_700.0, gt=0)
    # How long one feature may execute before it is stopped, however well it is progressing.
    #
    # Deliberately not the same question as `feature_run_stale_after_seconds` above, which
    # asks whether an executor has stopped *writing*. A feature can be writing a checkpoint
    # every few minutes, be visibly making progress, and still be nine and a half hours into
    # a run nobody wants to pay for -- the longest observed here reached 571 minutes inside a
    # single coroutine. That is what this bounds. Both are needed and neither subsumes the
    # other.
    #
    # Six hours: comfortably above the slowest healthy run measured (roughly ninety minutes
    # average, with a long tail), so this is a backstop rather than a scheduling policy.
    # Clocked from when a worker claimed the feature, not from when it was submitted, and
    # only against actively-executing statuses -- a feature waiting for a person is resting,
    # and may legitimately rest for days. Zero disables it.
    feature_runtime_limit_seconds: float = Field(default=21_600.0, ge=0)
    # How long one repository's workstream may execute before no further attempt is
    # scheduled for it. Checked between attempts, never during one: a ceiling that
    # interrupted an attempt could interrupt a push, and this must only stop scheduling.
    # Ninety minutes, which is above the whole retry budget of a healthy workstream.
    # Zero disables it.
    repository_runtime_limit_seconds: float = Field(default=5_400.0, ge=0)
    # How long a deferred remote operation on a feature that has ended may wait for the
    # credentialed request that would confirm it, before a person is asked instead.
    #
    # An interrupted push or pull request is deliberately left `awaiting_reconciliation` so
    # the next credentialed request through the same code path can settle it, which for a
    # live feature is the ordinary course of events. A feature that reached a terminal status
    # makes no further requests, so without this the row waits for ever -- and since a
    # deferred operation is not returned by the incomplete-operation scan, no later sweep
    # revisits it either. A day, because a failed feature can still be resumed by somebody
    # returning to it, and asking early would ask about an effect about to be confirmed.
    deferred_operation_settle_after_seconds: float = Field(default=86_400.0, gt=0)
    github_default_draft_pull_requests: bool = True
    default_agent_timeout_seconds: int = Field(gt=0)
    # A repository's commit gate lints the whole project, not the handful of generated
    # paths, so it is bounded separately from a single agent call. Sharing the agent
    # timeout made a normal project-wide lint look like a hung process.
    commit_gate_timeout_seconds: int = Field(default=900, gt=0)
    # How long one `npm ci` -- or its pip/poetry equivalent -- may take. Its own number
    # because a package manager resolving a dependency tree over the network is not an agent
    # call, and because a timeout here is terminal: `install_failure_is_transient` refuses to
    # retry it on purpose, so a budget that does not fit the work ends a workstream before a
    # single coding call is made. AB-Feature-203, 204 and 205 all died exactly there.
    #
    # 600 rather than the agent call's 120 because the observed distribution is bimodal and
    # 120 sits between its two modes. Measured on one machine inside one hour, across two
    # repositories and one container: 37s, 48s, 54s with a warm cache; 224s, 274s, 277s,
    # 301s, 303s, 378s cold or contended. A budget between the modes makes the outcome a coin
    # flip; 600 clears the slowest observed install by a further 59%, which is the margin a
    # contended host needs, and stays far enough above the warm mode that a cold-cache
    # regression still shows up as a slow install rather than as a passing one.
    #
    # This bounds one subprocess run, not the whole operation: `_install_with_retries` may ask
    # again when the evidence names the registry, and the journal allows the operation a
    # second attempt. Raising this raises that product too.
    dependency_install_timeout_seconds: int = Field(default=600, gt=0)
    # How long a repository's own test or build command may take. Distinct from the reviewer
    # agent's timeout, which bounds a model call: sharing one value meant a passing test suite
    # was recorded as timed out, and the reviewer then blocked a change whose tests it could
    # see had passed.
    validation_command_timeout_seconds: int = Field(default=900, gt=0)
    # How many finished features keep their workspace. Each one retains an installed
    # dependency tree, so an unbounded set filled the volume after roughly thirty runs and
    # every later install failed with an opaque package-manager exit code.
    workspace_retention_features: int = Field(default=5, ge=1)
    # How long a feature that ended needing a human keeps its checkout. That checkout is the
    # only copy of what the attempt actually wrote, so an operator must be able to open it --
    # but retaining it forever is what filled this volume: a pilot whose runs mostly ended
    # this way reclaimed nothing, and the capacity preflight then refused to start -071 at
    # all. The database keeps the artifacts, events and operation journal either way.
    workspace_failed_retention_hours: int = Field(default=72, ge=0)
    # How many of those checkouts may be retained at once, whatever their age. The window
    # above is what a person needs in order to open the newest one; without a count beside
    # it, a day of failures fills the volume before any of them is old enough to reclaim.
    # That is run 193 exactly: every retained workspace was from the same day, the volume
    # reached 99% with 1.84 GB free, and the capacity preflight then failed a feature whose
    # repositories had both been approved. Generous on purpose -- ten is more than an
    # operator reviews in a sitting -- because the bound exists to stop unbounded growth,
    # not to ration evidence. Zero disables it, as zero hours disables the window.
    workspace_failed_retention_features: int = Field(default=10, ge=0)
    # How many live features may be in flight at once. Each one clones every repository it
    # touches, installs their dependency trees and holds provider quota, so without a limit
    # the first thing a burst meets is the disk or the provider rather than a policy. Zero is
    # unbounded, which is what this was before.
    max_concurrent_live_features: int = Field(default=3, ge=0)
    # How many queued features this process will execute at once. A feature is accepted and
    # committed before anything runs, so this bounds the workers that drain that queue rather
    # than what a caller may submit; `max_concurrent_live_features` still bounds live work.
    # More than one because a live feature occupies its worker for minutes, and one worker
    # would make the queue strictly serial behind it.
    feature_queue_workers: int = Field(default=2, ge=1, le=32)
    # How long a claimed queue entry stays claimed without being renewed. A worker renews at
    # a third of this while it runs, so in principle it only decides how long a crashed
    # process's work waits before another one picks it up.
    #
    # A claim covers one step now rather than a whole feature, so this is re-derived from the
    # longest step rather than the longest feature -- and it lands on the same number, which
    # is worth stating rather than quietly leaving alone. The longest step is a repository
    # wave, and a wave is bounded by `repository_runtime_limit_seconds` at ninety minutes, so
    # no plausible lease covers it outright and the renewal loop stays load-bearing for that
    # one step. What the lease actually has to exceed is how long the event loop can go
    # without running the renewal task, and this platform's work blocks it: a clone and a
    # full-tree scan of a real repository are not fast, and none of that changed when the
    # execution unit did. At 120s a live feature outlived its own lease inside fifteen
    # minutes, a second worker took the entry, and the collision retired a feature that was
    # running perfectly well.
    #
    # What did change is the cost of being wrong. A lapse used to hand a second worker a
    # feature with an hour of unreconciled work in flight; it now hands over one step, and the
    # claim that takes it re-runs only that step. Shortening this would buy faster crash
    # detection for the four cheap steps at the price of collisions on the expensive one, and
    # the expensive one is where a collision costs something -- so it stays where the
    # measurement put it.
    feature_queue_lease_seconds: int = Field(default=900, ge=30)
    # Refuse a new mutation before clone/install when cleanup cannot leave enough capacity.
    # This is a deployment limit rather than a repository-validation failure.
    workspace_minimum_free_bytes: int = Field(default=2_147_483_648, ge=0)
    default_agent_max_retries: int = Field(ge=0)
    confidence_threshold: float = Field(ge=0, le=1)
    agents: dict[str, AgentSettings]

    # Resolved once, when settings load, so a deployment discovers a missing or misspelled
    # model role at startup rather than when a live feature first reaches its coding stage.
    _model_configs: ModelConfigService = PrivateAttr()
    # True only on a clone `for_model_setup` returned. It is what makes a user's pinned role
    # efforts authoritative for a custom feature: the per-agent overrides are deployment knobs
    # over deployment presets, and a recorded personal choice is neither.
    _model_setup_scoped: bool = PrivateAttr(default=False)
    # Fixable ceiling findings over the environment-backed tiers, collected at validation and
    # logged by the production lifespan as named warnings. Deliberately not refusals: item 29's
    # warn/refuse split is the design -- only an unsatisfiable pairing may stop a deployment.
    _model_configuration_warnings: tuple[str, ...] = PrivateAttr(default=())

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load environment variables before the immutable project policy defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, CONFIG_FILE),
            YamlConfigSettingsSource(settings_cls, AGENT_CONFIG_FILE),
            file_secret_settings,
        )

    @field_validator(
        "anthropic_low_reasoning_max_tokens",
        "anthropic_low_coding_max_tokens",
        "anthropic_low_review_max_tokens",
        "anthropic_low_scoped_fix_max_tokens",
        "anthropic_medium_reasoning_max_tokens",
        "anthropic_medium_coding_max_tokens",
        "anthropic_medium_review_max_tokens",
        "anthropic_medium_scoped_fix_max_tokens",
        "anthropic_high_reasoning_max_tokens",
        "anthropic_high_coding_max_tokens",
        "anthropic_high_review_max_tokens",
        "anthropic_high_scoped_fix_max_tokens",
        "anthropic_ultra_reasoning_max_tokens",
        "anthropic_ultra_coding_max_tokens",
        "anthropic_ultra_review_max_tokens",
        "anthropic_ultra_scoped_fix_max_tokens",
        mode="before",
    )
    @classmethod
    def an_unset_tier_bound_is_no_bound(cls, value: object) -> object:
        """Read an empty deployment variable as "this tier names no bound of its own".

        Every other optional setting here is a string, for which "unset" and "" are the same
        thing. These are the first optional *integers*, and the substrate that supplies them
        cannot express absence: `docker-compose.yml` interpolates an unset variable to the
        empty string, so the container receives `ANTHROPIC_LOW_REVIEW_MAX_TOKENS=` rather
        than nothing at all. Without this the API refuses to start with twelve parse errors
        the moment the tier variables are passed through -- which is exactly what it did.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("platform_api_key", "bootstrap_admin_password", mode="before")
    @classmethod
    def blank_secret_is_absent(cls, value: object) -> object:
        """Treat an empty secret as unset rather than as a credential nobody can guess.

        `docker-compose.yml` passes variables through as `NAME=${NAME}`, so a variable that
        is not exported in the environment arrives as an empty string rather than as nothing
        at all. Before this, an empty `PLATFORM_API_KEY` was a validation error and the API
        refused to start; now it means what an operator means by it, which is "not
        configured".

        The distinction matters for exactly one reason: an empty *string* would otherwise be
        a live shared key that `secrets.compare_digest` matches for any caller who sends an
        empty bearer token. Absent is absent.
        """
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value().strip():
            return None
        return value

    @field_validator("bootstrap_admin_email", mode="before")
    @classmethod
    def blank_string_is_default(cls, value: object) -> object:
        """Treat a blank value as unset, so the field's own default is reachable.

        Same Compose pass-through as `blank_secret_is_absent` above, and a worse consequence
        for being on a plain `str`: pydantic accepts `""` as a valid value and it silently
        *wins* over the declared default, where a `SecretStr | None` field at least ends up
        `None`. Returning `None` here makes pydantic apply the default instead.

        This is not hypothetical. The first deploy of multi-user login passed
        `BOOTSTRAP_ADMIN_EMAIL=` from Compose, the bootstrap looked the administrator up by
        the empty string, found nobody, and set no password -- leaving a console nobody could
        sign in to on a deployment whose only other credential the login page no longer
        accepts. The bootstrap itself behaved correctly and did nothing; the setting had
        already lied to it.

        The default is restored explicitly rather than by returning `None`: a `mode="before"`
        validator runs after the field is populated, so `None` on a non-optional `str` is a
        validation error rather than a request for the default.
        """
        if isinstance(value, str) and not value.strip():
            return DEFAULT_BOOTSTRAP_ADMIN_EMAIL
        return value

    @field_validator("workspace_root")
    @classmethod
    def workspace_root_must_be_absolute(cls, value: Path) -> Path:
        """Keep API-requested workspaces inside one explicitly configured absolute root."""
        if not value.is_absolute():
            msg = "workspace_root must be absolute"
            raise ValueError(msg)
        return value

    @field_validator("allowed_hosts")
    @classmethod
    def allowed_hosts_must_be_explicit(cls, value: list[str]) -> list[str]:
        """Reject blank or catch-all hosts before the HTTP middleware trusts them."""
        if any(not host.strip() or host.strip() == "*" for host in value):
            msg = "allowed_hosts entries must be explicit non-empty host patterns"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def anthropic_bounds_must_hold_a_thinking_response(self) -> Self:
        """Refuse an Anthropic output bound its own configured effort cannot finish under.

        A truncation arrives as `stop_reason: "max_tokens"` on the first live call and is
        correctly deterministic (`response_truncated`, never retried verbatim; the child
        loop may re-ask once at reduced effort, and only there). It is otherwise
        terminal for whatever stage it hits -- which in the 175-180 matrix was every
        Anthropic stage that ran: two planners, a reviewer, and a coding call. All four were
        this configuration mistake, and every one of them was checkable here, before any
        feature existed. Like `_required_max_tokens` in the adapter: refused at construction,
        not discovered as a dead feature run.

        Validated per (scope, role) for exactly the scopes a request can resolve through: the
        unsuffixed platform -- which *is* the high tier -- and each tier preset, using the same
        fall-back-to-unsuffixed bound rule `_tier_platform_settings` applies. The error names
        the variable whose value was actually read, so an inherited bound blames the unsuffixed
        variable and a tier's own bound blames the tier's.

        The same scopes are also judged against the `MODEL_MAX_OUTPUT_TOKENS` ceilings, by the
        same rule `validate_model_setup` holds an authored setup to -- but with the split
        policy 00-todo item 29 prescribes: only an *unsatisfiable* pairing (the effort's floor
        exceeds the model's declared ceiling, so no bound value could ever work) refuses
        startup; a fixable finding -- a bound over the ceiling, or the AB-Feature-181 shape of
        a coding bound flush against it -- becomes a named warning the production lifespan
        logs, because an environment preset may predate the declaration and refusing startup
        would take down a running deployment over configuration that used to be legal.
        """
        # Parsed here as well as at custom-setup use, so malformed JSON refuses startup the
        # way MODEL_REASONING_UNSUPPORTED already does -- a ceiling declaration that silently
        # fails to apply is exactly the blindspot the tier checks close.
        ceilings = normalize_max_output_tokens(self.declared_max_output_tokens())
        # Same rule for the temperature declaration: malformed JSON refuses startup rather
        # than failing the first live provider call of a run, which is how AB-Feature-210
        # died -- a capability the adapter learns mid-run is a capability learned too late.
        self.declared_temperature_unsupported()
        # And for the context-window declaration, whose consumer is the snapshot budget
        # derivation: a declaration that silently fails to parse would run every attempt at
        # the default budget while the deployment believes the declared one is in force.
        normalize_context_window_tokens(self.declared_context_window_tokens())
        warnings: list[str] = []
        for role in ModelRole:
            self._judge_anthropic_bounds(
                role=role,
                model=getattr(self, f"anthropic_{role.value}_model"),
                effort_raw=getattr(self, _anthropic_effort_field("anthropic_", role)),
                bound=getattr(self, f"anthropic_{role.value}_max_tokens"),
                variables=MODEL_ROLE_VARIABLES[(AgentPlatform.ANTHROPIC, role)],
                bound_variable=None,
                ceilings=ceilings,
                scope_label=f"the anthropic {role.value} role",
                warnings=warnings,
            )
        # Only the environment-backed tiers: `custom` derives no field names, and
        # `anthropic_custom_reasoning_max_tokens` is an attribute that does not exist.
        for tier in PerformanceTier.env_backed():
            prefix = f"anthropic_{tier.value}_"
            for role in ModelRole:
                declared = getattr(self, f"{prefix}{role.value}_max_tokens")
                self._judge_anthropic_bounds(
                    role=role,
                    model=getattr(self, f"{prefix}{role.value}_model"),
                    effort_raw=getattr(self, _anthropic_effort_field(prefix, role)),
                    bound=(
                        declared
                        if declared is not None
                        else getattr(self, f"anthropic_{role.value}_max_tokens")
                    ),
                    variables=tier_model_role_variables(AgentPlatform.ANTHROPIC, tier, role),
                    bound_variable=(
                        None
                        if declared is not None
                        else MODEL_ROLE_VARIABLES[
                            (AgentPlatform.ANTHROPIC, role)
                        ].max_tokens_variable
                    ),
                    ceilings=ceilings,
                    scope_label=f"the anthropic {tier.value} tier's {role.value} role",
                    warnings=warnings,
                )
        self._model_configuration_warnings = tuple(warnings)
        return self

    def _judge_anthropic_bounds(
        self,
        *,
        role: ModelRole,
        model: str,
        effort_raw: str,
        bound: int,
        variables: ModelRoleVariables,
        bound_variable: str | None,
        ceilings: Mapping[str, int],
        scope_label: str,
        warnings: list[str],
    ) -> None:
        """Hold one configured Anthropic role to the floor and ceiling rules, split by fixability.

        A role with no model configured in this scope is skipped: nothing can construct a
        client through it, so its bound is never read. Both checks are the extracted rules
        shared with `validate_model_setup` -- `require_anthropic_floor` and
        `judge_anthropic_ceiling` -- so the two authorities judge with one rule and one
        sentence, and only the policy differs: an unsatisfiable pairing and an under-floor
        bound raise, a fixable ceiling finding is appended to ``warnings`` for the startup
        log. ``scope_label`` names the tier so the sentences blame something an operator can
        find; the bound keeps the same variable attribution the floor check always had.
        """
        if not str(model).strip():
            return
        effort = normalize_reasoning_level(variables.reasoning_variable, str(effort_raw))
        bound_name = bound_variable or variables.max_tokens_variable
        judgement = judge_anthropic_ceiling(
            role=role,
            model=str(model).strip(),
            effort=effort,
            bound=bound,
            ceilings=ceilings,
            subject=f"{scope_label} ({variables.reasoning_variable})",
            bound_name=bound_name,
        )
        if judgement.unsatisfiable is not None:
            raise ModelConfigurationError(judgement.unsatisfiable)
        require_anthropic_floor(
            role=role,
            effort=effort,
            bound=bound,
            bound_name=bound_name,
            effort_name=variables.reasoning_variable,
        )
        if judgement.bound_over_ceiling is not None:
            warnings.append(judgement.bound_over_ceiling)
        if judgement.exhaustion_shape is not None:
            warnings.append(judgement.exhaustion_shape)

    @model_validator(mode="after")
    def required_agents_must_be_configured(self) -> Self:
        """Ensure every workflow role has explicit operational policy before startup."""
        missing_agents = REQUIRED_AGENT_NAMES - self.agents.keys()
        if missing_agents:
            msg = f"missing agent configuration: {sorted(missing_agents)}"
            raise ValueError(msg)
        if self.environment == "production" and not self.allowed_hosts:
            msg = "production requires at least one configured allowed_host"
            raise ValueError(msg)
        self._model_configs = build_model_config_service(
            platforms={
                AgentPlatform.OPENAI: PlatformModelSettings(
                    models={
                        ModelRole.REASONING: self.openai_reasoning_model,
                        ModelRole.CODING: self.openai_coding_model,
                        ModelRole.REVIEW: self.openai_review_model,
                        ModelRole.SCOPED_FIX: self.openai_scoped_fix_model,
                    },
                    reasoning={
                        ModelRole.REASONING: self.openai_reasoning_effort,
                        ModelRole.CODING: self.openai_coding_reasoning_effort,
                        ModelRole.REVIEW: self.openai_review_reasoning_effort,
                        ModelRole.SCOPED_FIX: self.openai_scoped_fix_reasoning_effort,
                    },
                    # The Responses API needs no output bound and this platform has never
                    # sent one, so nothing is declared for it here.
                    max_tokens=dict.fromkeys(ModelRole, None),
                    legacy_models={
                        ModelRole.REVIEW: self.openai_reasoning_model,
                        ModelRole.SCOPED_FIX: self.openai_fix_model or self.openai_coding_model,
                    },
                    legacy_reasoning={
                        ModelRole.CODING: self.openai_coding_reasoning
                        or self.openai_coding_effort
                        or self._agent_reasoning("engineer"),
                        ModelRole.REVIEW: self.openai_review_reasoning
                        or self.openai_reasoning_effort
                        or self._agent_reasoning("reviewer"),
                        ModelRole.SCOPED_FIX: self.openai_fix_reasoning
                        or self.openai_coding_reasoning
                        or self.openai_coding_effort,
                    },
                ),
                AgentPlatform.ANTHROPIC: PlatformModelSettings(
                    models={
                        ModelRole.REASONING: self.anthropic_reasoning_model,
                        ModelRole.CODING: self.anthropic_coding_model,
                        ModelRole.REVIEW: self.anthropic_review_model,
                        ModelRole.SCOPED_FIX: self.anthropic_scoped_fix_model,
                    },
                    reasoning={
                        ModelRole.REASONING: self.anthropic_reasoning_effort,
                        ModelRole.CODING: self.anthropic_coding_reasoning_effort,
                        ModelRole.REVIEW: self.anthropic_review_reasoning_effort,
                        ModelRole.SCOPED_FIX: self.anthropic_scoped_fix_reasoning_effort,
                    },
                    max_tokens={
                        ModelRole.REASONING: self.anthropic_reasoning_max_tokens,
                        ModelRole.CODING: self.anthropic_coding_max_tokens,
                        ModelRole.REVIEW: self.anthropic_review_max_tokens,
                        ModelRole.SCOPED_FIX: self.anthropic_scoped_fix_max_tokens,
                    },
                ),
            },
            tiers=self._tier_platform_settings(),
            unsupported_reasoning=self.declared_unsupported_reasoning(),
        )
        return self

    def _tier_platform_settings(
        self,
    ) -> dict[tuple[AgentPlatform, PerformanceTier], PlatformModelSettings]:
        """Collect every tier-suffixed preset a deployment has named anything in.

        Assembled from the field names rather than written out six more times: every variable
        this reads is declared explicitly above, so each one stays greppable, and the names
        are exactly `tier_model_role_variables`' -- the declared field list and this assembly
        cannot drift apart without startup failing on the missing attribute.
        """
        tiers: dict[tuple[AgentPlatform, PerformanceTier], PlatformModelSettings] = {}
        for platform in AgentPlatform:
            # Environment-backed tiers only: `custom` names no variables and has no preset.
            for tier in PerformanceTier.env_backed():
                prefix = f"{platform.value}_{tier.value}_"
                models: dict[ModelRole, str] = {}
                reasoning: dict[ModelRole, str | None] = {}
                max_tokens: dict[ModelRole, int | None] = {}
                for role in ModelRole:
                    models[role] = str(getattr(self, f"{prefix}{role.value}_model"))
                    effort_field = (
                        f"{prefix}reasoning_effort"
                        if role is ModelRole.REASONING
                        else f"{prefix}{role.value}_reasoning_effort"
                    )
                    reasoning[role] = str(getattr(self, effort_field)) or None
                    if platform is AgentPlatform.ANTHROPIC:
                        declared = getattr(self, f"{prefix}{role.value}_max_tokens")
                        fallback = getattr(self, f"anthropic_{role.value}_max_tokens")
                        max_tokens[role] = int(declared) if declared is not None else int(fallback)
                    else:
                        # The Responses API needs no output bound; parity with the platform
                        # configuration above, which declares none for OpenAI either.
                        max_tokens[role] = None
                if (
                    not any(models.values())
                    and not any(reasoning.values())
                    and not (
                        platform is AgentPlatform.ANTHROPIC
                        and any(
                            getattr(self, f"{prefix}{role.value}_max_tokens") is not None
                            for role in ModelRole
                        )
                    )
                ):
                    # Nothing named for this pairing: the deployment does not offer it, which
                    # is a preference and must not even reach resolution as an empty preset.
                    continue
                tiers[(platform, tier)] = PlatformModelSettings(
                    models=models, reasoning=reasoning, max_tokens=max_tokens
                )
        return tiers

    def for_performance_tier(self, tier: PerformanceTier) -> Settings:
        """Return this configuration viewed through one feature's pinned tier.

        ``HIGH`` is this object itself, byte for byte -- the unsuffixed variables are the
        high tier, so a feature that predates tiers resolves through the very same instance.
        Any other tier gets a copy whose resolved model roles are the tier's; everything
        else -- timeouts, budgets, credentials policy -- is untouched, because a tier prices
        model calls and changes nothing about how the platform runs.
        """
        if tier is PerformanceTier.CUSTOM:
            # A custom feature resolves through its snapshot -- `for_model_setup` below -- so
            # a caller that reached the tier path with CUSTOM has lost the snapshot. Raising
            # here surfaces that bug at the first model call instead of as a feature that
            # quietly ran at some other tier.
            msg = (
                "a custom feature resolves through its pinned model setup snapshot, not "
                "through the tier path; the snapshot was not carried to this call"
            )
            raise ModelConfigurationError(msg)
        if tier is PerformanceTier.HIGH:
            return self
        clone = self.model_copy()
        clone._model_configs = self._model_configs.for_tier(tier)
        return clone

    def for_model_setup(self, snapshot: Mapping[str, Any]) -> Settings:
        """Return this configuration viewed through one feature's pinned custom setup.

        The same kind of clone `for_performance_tier` returns: the resolved model roles are
        the snapshot's, resolved through a role-addressed `ModelConfigService`, and everything
        else -- timeouts, budgets, credentials policy -- is untouched. The snapshot is
        re-validated on the way (it can outlive a change to the deployment's declarations),
        and a snapshot that no longer passes raises the honest refusal rather than clamping.

        The clone also marks itself setup-scoped: a user's pinned role efforts are
        authoritative for a custom feature, so the per-agent deployment overrides
        (`OPENAI_PLANNER_REASONING_EFFORT` and friends) and the policy-file per-agent efforts
        do not apply -- a pinned, recorded choice must not be silently defeated by a
        deployment knob.
        """
        clone = self.model_copy()
        clone._model_configs = model_config_service_for_setup(
            snapshot,
            unsupported=normalize_unsupported_reasoning(self.declared_unsupported_reasoning()),
            ceilings=normalize_max_output_tokens(self.declared_max_output_tokens()),
        )
        clone._model_setup_scoped = True
        return clone

    def declared_unsupported_reasoning(self) -> dict[str, list[str]]:
        """Parse the deployment's declaration of levels a model does not accept."""
        raw = self.model_reasoning_unsupported.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            msg = "MODEL_REASONING_UNSUPPORTED must be a JSON object of model to level list"
            raise ModelConfigurationError(msg) from error
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(value, list) for key, value in parsed.items()
        ):
            msg = "MODEL_REASONING_UNSUPPORTED must be a JSON object of model to level list"
            raise ModelConfigurationError(msg)
        return {key: [str(item) for item in value] for key, value in parsed.items()}

    def declared_temperature_unsupported(self) -> frozenset[str]:
        """Parse the deployment's declaration of models that reject ``temperature``."""
        raw = self.model_temperature_unsupported.strip()
        if not raw:
            return frozenset()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            msg = "MODEL_TEMPERATURE_UNSUPPORTED must be a JSON array of model names"
            raise ModelConfigurationError(msg) from error
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            msg = "MODEL_TEMPERATURE_UNSUPPORTED must be a JSON array of model names"
            raise ModelConfigurationError(msg)
        return frozenset(item.strip() for item in parsed)

    def declared_vision_capable(self) -> frozenset[str]:
        """Parse the deployment's declaration of models that can be shown an image.

        Empty means no model is, which is the fail-closed reading the field's comment
        argues for. A malformed declaration raises rather than degrading to empty: silently
        reading a typo as "nothing accepts images" would refuse every submission with a
        screenshot and say nothing about why.
        """
        raw = self.model_vision_capable.strip()
        if not raw:
            return frozenset()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            msg = "MODEL_VISION_CAPABLE must be a JSON array of model names"
            raise ModelConfigurationError(msg) from error
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            msg = "MODEL_VISION_CAPABLE must be a JSON array of model names"
            raise ModelConfigurationError(msg)
        return frozenset(item.strip() for item in parsed if item.strip())

    def declared_max_output_tokens(self) -> dict[str, int]:
        """Parse the deployment's declaration of each model's output ceiling."""
        raw = self.model_max_output_tokens.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            msg = "MODEL_MAX_OUTPUT_TOKENS must be a JSON object of model to output ceiling"
            raise ModelConfigurationError(msg) from error
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
            for key, value in parsed.items()
        ):
            msg = "MODEL_MAX_OUTPUT_TOKENS must be a JSON object of model to output ceiling"
            raise ModelConfigurationError(msg)
        return dict(parsed)

    def declared_context_window_tokens(self) -> dict[str, int]:
        """Parse the deployment's declaration of each model's context window."""
        raw = self.model_context_window_tokens.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            msg = (
                "MODEL_CONTEXT_WINDOW_TOKENS must be a JSON object of model to "
                "context window in tokens"
            )
            raise ModelConfigurationError(msg) from error
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
            for key, value in parsed.items()
        ):
            msg = (
                "MODEL_CONTEXT_WINDOW_TOKENS must be a JSON object of model to "
                "context window in tokens"
            )
            raise ModelConfigurationError(msg)
        return dict(parsed)

    def _agent_reasoning(self, agent_name: str) -> str | None:
        """Read one agent's policy-file effort, so a role inherits what that agent had."""
        agent = self.agents.get(agent_name)
        if agent is None:
            return None
        return normalize_reasoning_level(
            f"agents.{agent_name}.reasoning_effort", agent.reasoning_effort
        )

    @property
    def model_configs(self) -> ModelConfigService:
        """Return the one resolved model-role configuration for this deployment."""
        return self._model_configs

    @property
    def model_configuration_warnings(self) -> tuple[str, ...]:
        """Return the fixable ceiling findings the tier validation chose to warn about.

        Empty for a clean environment. The production lifespan logs each as a
        `model_configuration_warning` event so the named warning reaches the startup log
        next to the resolved model table.
        """
        return self._model_configuration_warnings

    def model_config_for_role(self, role: ModelRole, *, platform: AgentPlatform) -> ModelConfig:
        """Resolve one model role, for a caller that has no reason to hold the service."""
        return self._model_configs.get_model_config(role, platform=platform)

    def model_for_agent(self, agent_name: str, *, platform: AgentPlatform) -> str:
        """Resolve an agent's model from its role on the platform executing this feature."""
        agent = self._require_agent(agent_name)
        if agent.model_role is None:
            msg = (
                f"agent {agent_name} declares no model_role and makes no model call; "
                "it has no configured model"
            )
            raise ModelConfigurationError(msg)
        return self._model_configs.get_model_config(agent.model_role, platform=platform).model

    def _agent_effort_override(self, agent_name: str) -> str:
        """Read one agent's deployment-supplied effort, or empty when it supplies none."""
        return {
            "planner": self.openai_planner_reasoning_effort,
            "product_manager": self.openai_product_manager_reasoning_effort,
            "recon": self.openai_recon_reasoning_effort,
        }.get(agent_name, "")

    def reasoning_effort_for_agent(
        self,
        agent_name: str,
        *,
        platform: AgentPlatform,
        model_role: ModelRole | None = None,
    ) -> str | None:
        """Resolve an agent's configured reasoning effort, or nothing to use the default.

        Four statements, most specific first. An explicitly routed role wins outright, for the
        reason the timeout does: a caller that routes a call has chosen that role deliberately
        and must not have the choice overridden by whichever agent abstraction it reused.

        Otherwise the deployment's per-agent variable wins, then the policy file's per-agent
        declaration, then the role's own level. The two per-agent statements exist because two
        agents can share a role and still be different jobs -- the product manager and the
        planner are both `reasoning` -- so before them the only way to ease one off `max` was
        to ease both. Environment above file throughout: the file ships inside the image and
        costs a rebuild to change, and configuration an operator cannot change on a running
        deployment is not really configuration.

        Every level returned here goes through the one central capability check, so a
        deployment that declares a level unsupported for a model still has that respected
        however the level was chosen.
        """
        agent = self._require_agent(agent_name)
        if model_role is not None:
            return self._model_configs.get_model_config(model_role, platform=platform).reasoning
        if agent.model_role is None:
            # An agent with no role makes no model call, so there is no effort to resolve.
            return None
        role_config = self._model_configs.get_model_config(agent.model_role, platform=platform)
        if self._model_setup_scoped:
            # A custom setup's role effort is authoritative. The per-agent statements exist to
            # let an operator tune deployment presets; a user's pinned, recorded choice being
            # silently defeated by one would be the broken promise G2 forbids, and the role's
            # routing reason says which statement was followed.
            return role_config.reasoning
        declared = normalize_reasoning_level(
            _AGENT_REASONING_EFFORT_VARIABLES.get(agent_name, ""),
            self._agent_effort_override(agent_name),
        ) or normalize_reasoning_level(
            f"agents.{agent_name}.reasoning_effort", agent.reasoning_effort
        )
        if declared is not None:
            return supported_reasoning(
                role_config.model,
                declared,
                normalize_unsupported_reasoning(self.declared_unsupported_reasoning()),
            )
        return role_config.reasoning

    def timeout_seconds_for_agent(
        self, agent_name: str, *, model_role: ModelRole | None = None
    ) -> int:
        """Resolve the deadline for an agent call without scattering environment reads.

        Reasoning is the one role whose maximum-effort latency has exceeded the legacy agent
        policy in production. Other roles retain their established per-agent deadlines; an
        explicitly routed role is honored so a caller cannot accidentally inherit a reasoning
        timeout merely because it reuses an agent abstraction.
        """
        agent = self._require_agent(agent_name)
        resolved_role = model_role or agent.model_role
        if resolved_role is ModelRole.REASONING:
            return self.openai_reasoning_timeout_seconds
        return agent.timeout_seconds

    def _require_agent(self, agent_name: str) -> AgentSettings:
        """Return one configured agent, refusing an unknown name by that name."""
        try:
            return self.agents[agent_name]
        except KeyError as error:
            msg = f"unknown agent: {agent_name}"
            raise KeyError(msg) from error


def load_settings(config_directory: Path = CONFIG_DIRECTORY, **overrides: Any) -> Settings:
    """Load settings from an explicit YAML directory while retaining env-variable precedence."""
    config_file = config_directory / "config.yaml"
    agent_file = config_directory / "agent.yaml"
    # Read both documents eagerly so a missing or malformed policy file fails with this
    # module's diagnostic instead of an opaque validation error about absent fields.
    _load_yaml_mapping(config_file)
    _load_yaml_mapping(agent_file)

    class _DirectoryScopedSettings(Settings):
        """Settings bound to one policy directory and ranked below the deployment."""

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            """Rank the selected policy directory beneath environment and `.env` values."""
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                YamlConfigSettingsSource(settings_cls, config_file),
                YamlConfigSettingsSource(settings_cls, agent_file),
                file_secret_settings,
            )

    # The policy documents are defaults, not overrides. Passing them as init keyword
    # arguments ranked them above every environment variable, so a container exporting
    # ENVIRONMENT=production kept the `development` value config.yaml names, and the
    # production-only identity and allowed-hosts gates never ran. Explicit `overrides`
    # stay authoritative because a caller passes them deliberately.
    return _DirectoryScopedSettings(**overrides)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML document that must contain a top-level mapping."""
    try:
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        msg = f"configuration file does not exist: {path}"
        raise FileNotFoundError(msg) from error
    if raw_data is None:
        return {}
    if not isinstance(raw_data, dict):
        msg = f"configuration file must contain a mapping: {path}"
        raise ValueError(msg)
    return raw_data
