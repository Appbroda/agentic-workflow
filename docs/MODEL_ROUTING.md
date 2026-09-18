# Model routing

The platform exposes four logical model roles. Agents and workflows ask for a role; the central
configuration service resolves the model and reasoning effort from environment settings.
Model identifiers do not appear in agent or workflow decisions.

A role resolves **per agent platform**. The roles are unchanged by that — they are about the
authority of a call and not about a vendor: `REVIEW` never writes code and `CODING` never
certifies its own work, whichever provider executes them.

| Role | Work | OpenAI | Anthropic |
|---|---|---|---|
| `REASONING` | product analysis, planning, architecture, repository analysis and diagnosis | `gpt-5.6-sol` @ `max` | `claude-opus-5` @ `max` |
| `CODING` | feature implementation and substantive or ambiguous corrections | `gpt-5.6-sol` @ `max` | `claude-opus-5` @ `xhigh` |
| `REVIEW` | independent repository and integration review | `gpt-5.6-terra` @ `high` | `claude-opus-5` @ `high` |
| `SCOPED_FIX` | localized import, lint, formatting, type, build or deterministic test corrections | `gpt-5.3-codex` @ `high` | `claude-sonnet-5` @ `high` |

That table is the **high** tier — the unsuffixed configuration every deployment already had.
What the other two tiers resolve to is below, under [Performance tiers](#performance-tiers).

These are defaults to be swept, not settled facts — they are configuration precisely so a
deployment can move them without a rebuild. `xhigh` rather than `max` for Anthropic coding
matches a judgement this repository has already made once: `agent.yaml` eased the planner off
`max` because reasoning cost rises far faster than the quality of the output does, and because
a twenty to forty-five minute attempt interacts badly with a retry budget.

The checked-in deployment example configures both:

```env
OPENAI_REASONING_MODEL=gpt-5.6-sol
OPENAI_REASONING_EFFORT=max
OPENAI_REASONING_TIMEOUT_SECONDS=1800

OPENAI_CODING_MODEL=gpt-5.6-sol
OPENAI_CODING_REASONING_EFFORT=max

OPENAI_REVIEW_MODEL=gpt-5.6-terra
OPENAI_REVIEW_REASONING_EFFORT=high

OPENAI_SCOPED_FIX_MODEL=gpt-5.3-codex
OPENAI_SCOPED_FIX_REASONING_EFFORT=high

ANTHROPIC_REASONING_MODEL=claude-opus-5
ANTHROPIC_REASONING_EFFORT=max

ANTHROPIC_CODING_MODEL=claude-opus-5
ANTHROPIC_CODING_REASONING_EFFORT=xhigh

ANTHROPIC_REVIEW_MODEL=claude-opus-5
ANTHROPIC_REVIEW_REASONING_EFFORT=high

ANTHROPIC_SCOPED_FIX_MODEL=claude-sonnet-5
ANTHROPIC_SCOPED_FIX_REASONING_EFFORT=high
```

## Which platform runs a feature

The provider is a property of a **feature**, chosen by the person submitting it and fixed for
that feature's life. A feature submitted on Claude is planned, implemented, reviewed and
remediated on Claude, across every restart and every recovered attempt, whatever the deployment
happens to be configured with by the time it resumes. There is no endpoint that changes a
running feature's platform and no code path that re-resolves it from configuration — the same
rule `execution_mode` has.

Configure a platform completely or not at all:

* **all four models set** — the platform is offered on the submission form;
* **none set** — the deployment does not offer it, which is a preference and starts normally;
* **some set** — a configuration mistake, refused at startup by the missing variable's name
  rather than discovered when a feature reaches its scoped-fix path.

At least one platform must resolve completely. A deployment that only uses Claude does not need
`OPENAI_REVIEW_MODEL`.

`/setup` reports which (platform, tier) pairings are configured and which models each resolves
to, so the form offers only what this deployment can actually run and states the models without
holding a second copy of deployment configuration. A platform that is not configured stays
visible and disabled rather than disappearing.

## Performance tiers

A feature also pins a **performance tier** — `low`, `medium` or `high` — beside its platform.
The tier is a named preset over the same four roles, not a routing input: the router still
chooses *which role* answers, and the tier only decides what that role resolves to. Like the
platform, it is fixed at submission and never re-read from configuration for a feature already
running, and every attempt's `model_routing` record and `model_routing_decision` log names it,
so a run's cost basis is attributable afterwards.

The environment naming is the scheme above with a tier segment:

```text
{PROVIDER}_{TIER}_{ROLE}_MODEL            OPENAI_LOW_CODING_MODEL=gpt-5.3-codex
{PROVIDER}_{TIER}_{ROLE}_REASONING_EFFORT ANTHROPIC_MEDIUM_CODING_REASONING_EFFORT=xhigh
{PROVIDER}_{TIER}_{ROLE}_MAX_TOKENS       ANTHROPIC_LOW_CODING_MAX_TOKENS=128000
```

* **The unsuffixed variables are the `high` tier.** `OPENAI_CODING_MODEL` and friends keep
  their meaning exactly; a `HIGH_`-suffixed variable, when set, overrides its unsuffixed twin,
  per field. No deployment changes behaviour by upgrading, and the API's default tier is
  `high` so a caller that names none gets what it always got.
* **Completeness applies per (platform, tier)**, the same "configure completely or not at all"
  contract the platforms already have: all four roles set makes the pairing selectable, none
  set means the deployment does not offer it, and a partial tier is refused at startup by the
  missing variable's name.
* **An Anthropic tier that names no `MAX_TOKENS`** for a role inherits that platform's
  per-role bound above.
* **The capability declaration is stricter for a tier.** `MODEL_REASONING_UNSUPPORTED` still
  normalizes an unsuffixed pairing by omitting the level and recording both values; a
  *tier-suffixed* pairing that names a declared-unsupported level is a configuration error at
  startup instead. `gpt-5.3-codex` takes `xhigh` but not `max`, and a preset that asked for
  `max` anyway would quietly run at the provider's default rather than at the effort it was
  priced for.
* **Per-agent efforts are not multiplied by tier.** `OPENAI_PLANNER_REASONING_EFFORT` and its
  siblings keep today's semantics: when set, they beat the resolved tier's role-level effort.
  They are a deployment-wide tuning knob.

The recommended presets for both providers ship in `.env.example`. Effort is the dominant
lever and model class changes only where a cheaper model is already proven in that role — the
middle tier cuts effort before model class, because a proven model at `high` beats an
unproven one at `xhigh`. The cost percentages there are estimates rather than measurements,
and tiers change spend *per attempt*, not attempts per feature: a low-tier run that retries
more can out-spend a medium-tier run.

The client shows the pairings as six labeled options — `OpenAI — Economy / Standard / Max` and
the same for Claude — preselects Standard on whichever provider was last used, and states the
Economy tier's scope where the choice is made: it is for small, well-bounded changes, and
larger work costs more there, not less, through retries. An option this deployment cannot run
renders disabled with the reason and the form refuses to submit rather than substituting a
different pairing.

The platform does not pick tiers by task size, and a feature's tier cannot be changed
mid-run. The human picks; the platform records and obeys.

## Custom model setups

A **custom model setup** is a user-authored tier: a persisted role map that selects, per role,
the platform, model, reasoning effort and output bound — including mixing providers in one
flow (Anthropic reasoning, OpenAI coding, Anthropic review). It is authored on
**Settings → Models** and selected on the submission form under "Your setups". The
environment-naming scheme above deliberately does not apply: a setup lives in the database,
scoped to the identity that authored it (`/model-setups`, `Permission.MODEL_SETUP_MANAGE`,
granted to operators and above), and its provenance strings name the setup — there is no
variable to grep for.

* **Resolution goes through the same seam a tier does.** A feature pinned to a setup records
  `performance_tier: "custom"`, and the runtime resolves it through
  `Settings.for_model_setup(snapshot)` — the same kind of scoped clone
  `for_performance_tier` returns, carrying a role-addressed `ModelConfigService` built from
  the snapshot. The router and the adapters are unchanged; each role answers on the platform
  it is pinned to, and the routing record names that platform, the tier `custom` and the
  setup id.
* **Validated at save AND at feature start, never clamped.** One predicate
  (`validate_model_setup`) holds a setup to the checks a tier passes — the effort vocabulary,
  the capability declaration (strict: a declared-unsupported pairing is refused, never
  normalized away), the Anthropic effort floors — plus the per-model output ceilings from
  `MODEL_MAX_OUTPUT_TOKENS`. A bound over a declared ceiling, a bound under its effort's
  floor, and an effort whose floor exceeds the model's ceiling are three distinct refusals,
  each with its own sentence. The third is AB-Feature-181 refused at authoring time. A model
  with no declared ceiling has no ceiling checks: the deployment is the authority.
* **The snapshot is the execution authority.** A feature copies the setup's role map at
  acceptance onto `feature_workflows.model_setup_snapshot` (and the queue row); editing or
  deleting the setup never changes what a running or historical feature means. The feature's
  Overview shows the pinned snapshot and says when the setup has been edited or deleted
  since. `model_setup_id` is provenance only and is never resolved through.
* **Credentials for every platform the roles name, plus GitHub.** Refused at setup save and
  at feature start with the roles that pinned the missing platform, and checked again by the
  worker before it loads any state.
* **A custom feature's `agent_platform` is the coding role's platform** — the platform the
  implementation work runs on — so every existing reader of that column keeps working;
  everything credential-shaped asks the setup's platform *set* instead.
* **A setup's efforts are authoritative.** The per-agent deployment overrides
  (`OPENAI_PLANNER_REASONING_EFFORT` and friends) tune deployment presets; a pinned, recorded
  personal choice is not silently defeated by one.

## `max_tokens`, which only one platform needs

The Messages API rejects a request without `max_tokens`; the Responses API does not need one and
this platform has never sent one. It is configured **per role**, beside the model and the effort,
because the reviewer's output is a findings list and the engineer's is a set of complete file
bodies for a multi-file feature. A number generous enough for the second wastes nothing on the
first, but a number sized for the first truncates the second — and a truncation arrives as
`stop_reason: "max_tokens"`, which reads as a malformed model response rather than as a budget
that was too small. The adapter classifies it as `response_truncated` for exactly that reason.

| Role | Default bound |
|---|---|
| `REASONING` | 32,000 |
| `CODING` | 64,000 |
| `REVIEW` | 16,000 |
| `SCOPED_FIX` | 16,000 |

Above roughly 16K the Messages API requires streaming, so the coding role's calls are issued
over a stream. That threshold is expressed in tokens rather than reusing the deadline that
decides transport for the Responses client: one says "this call will be slow", the other says
"this call is too large to answer in one response".

## `MODEL_CONTEXT_WINDOW_TOKENS`, which sizes the engineer's snapshot

The repository snapshot the engineer reads is character-budgeted, and the correct budget is a
function of the model that reads the prompt — a stronger model both writes wider first drafts
and reads bigger prompts, so a constant budget meets a wall exactly when the model outgrows it
(AB-Feature-212: both ultra retries were refused before the model was called because the
required union exceeded a constant sized for a smaller era).

`MODEL_CONTEXT_WINDOW_TOKENS` declares each model's context window in tokens, byte-shaped like
`MODEL_MAX_OUTPUT_TOKENS`:

```
MODEL_CONTEXT_WINDOW_TOKENS={"gpt-6-astra": 400000, "claude-fable-5": 200000}
```

One derivation (`repository_snapshot_budget` in the engineer, applied once at the executor's
construction site) turns a window into the snapshot budget: 20% of the window, at four
characters per token, with a tenth of that as the per-file bound. The key is the **resolved
model identifier** the routing decision carries — never the tier — so the CUSTOM tier and any
future tier inherit correct budgets with no new variables. A declared 200,000-token window
derives exactly the default constants, and an undeclared model falls back to exactly those
defaults: absence of a declaration is never a punishment. Malformed JSON refuses startup, like
the other declarations. Every completed attempt records `context_budget_characters`,
`context_budget_per_file_characters` and `context_budget_source` on its completion metadata,
so whether a declaration was actually in force is a lookup, never a reconstruction. Declare
only observed windows — an invented number would misbudget a valid configuration — and
remember the compose allowlist: the variable must appear in both `api` and `api-dev`
environment blocks, and `deploy.sh` verifies it crossed.

## Credentials

Which credentials a feature needs depends on the feature: an `anthropic` feature needs
`anthropic` + `github`, an `openai` feature needs `openai` + `github`, and neither needs the
other's key. Keys arrive as `X-OpenAI-Api-Key` and `X-Anthropic-Api-Key`, or from the submitting
identity's stored credentials — one header per provider, because a single generic header lets a
caller send a key for the wrong provider and get an authentication failure that names nothing
useful.

`OPENAI_REASONING_TIMEOUT_SECONDS` bounds each reasoning-role provider attempt. It is longer
than the legacy 120-second agent default because a maximum-effort planning response can take
several minutes; the provider SDK's retry count remains a separate, unchanged policy.

## Decision order

Model selection happens only after the platform has established what failed and whether another
attempt is allowed:

```text
failure
  → existing structured failure classification
  → existing remediation-owner decision
  → existing retry eligibility and convergence checks
  → repair-scope classification
  → logical model role
  → centrally resolved model and effort
  → execution
```

The router cannot grant an attempt, change a budget, or schedule work. Repository configuration,
dependency installation, validation capacity and missing test infrastructure keep using their
existing repository-repair, infrastructure or human paths before a coding model is considered.

## Scoped-fix eligibility

`SCOPED_FIX` is selected only when both conditions hold:

1. the existing failure classifier says the failure belongs to the feature Engineer and is a
   source-validation or review-scope failure; and
2. every blocking finding is deterministically recognized as localized, mechanical and
   independently verifiable.

Examples include a missing or incorrect import, an explicit ESLint rule violation, formatting,
a bounded local TypeScript mismatch, one deterministic fixture/assertion correction, or a simple
module-resolution problem.

The whole attempt remains on `CODING` when any finding is substantive or uncertain. This includes
business behavior, authentication and authorization, API or integration contracts, schema and
database design, architecture, cross-service state, concurrency, significant tests, and mixed
mechanical/substantive finding sets. The workflow currently executes a repository's findings as
one task, so it does not split the inexpensive subset away from a substantive finding.

Classification uses structured review categories before prose. A `security` or `contract`
category therefore cannot be reduced to a scoped fix merely because its recommendation includes
words such as “rename” or “import.” Unplaced wording is `AMBIGUOUS` and stays on `CODING`.

## Retry safety

Model routing is not a retry strategy. There is no model ladder and no model-cycling retry:

```text
same failure + same evidence
  ≠ new input merely because a configured model changed
```

Role, model name and effort are excluded from the effective input fingerprint. The existing retry
policy remains the only source of retry eligibility, and existing no-progress, repeated-diagnostic,
budget, revision and external-operation idempotency checks remain authoritative. A mechanical
failure stays on `SCOPED_FIX`; a substantive or ambiguous failure stays on `CODING` until changed
evidence produces a different classification or the existing policy stops the workstream.

The former `fix`, `complex_fix` and `escalation` values no longer exist as a vocabulary. A
historical durable record naming one is read as the active role that would execute the same work
today -- `fix` as `SCOPED_FIX`, the other two as `CODING` -- which is the mapping both consumers of
a persisted role already applied before acting on one. The ladder's own fields
(`escalation_level`, `escalated`, `previous_role`, and the per-finding ledger) are dropped on the
way in, so such a record still loads.

## Resolved execution metadata

Every successful model-backed artifact records the values that were true for its model call:

- provider;
- performance tier;
- agent type;
- logical model role;
- provider-returned model identifier;
- reasoning effort actually requested after central capability normalization;
- configuration variable used; and
- a short platform routing reason.

Engineering attempts also persist their classification, repair scope, revision and input
fingerprint in `model_routing`. These are execution facts, not chain-of-thought.

Historical records are read only from the artifact or per-attempt routing decision that produced
them. The execution-record service does not read current model configuration, so editing `.env`
cannot relabel an earlier feature.

The workflow graph consumes those backend execution records. Normal implementation arrows show
the persisted coding model and effort; review arrows show the persisted review model; retry arrows
show the persisted scoped-fix or coding model selected for that attempt. React never reconstructs
or predicts routing.

## Reading what a deployment actually resolved

`GET /model-configuration` serves the resolved table: for every configured (platform, tier)
pairing, each of the four roles with its model, reasoning effort, `max_tokens`, routing reason,
and the configuration variable the value came from. A role still resolving through a pre-roles
variable is flagged as such, and where capability normalization dropped a configured effort both
the requested and the effective value are published.

This is the `model_roles_resolved` startup log, answered on request and extended across tiers —
that log names the unsuffixed configuration only, and a deployment offering three tiers resolves
three tables. Both read the same `ModelConfigService` every agent resolves a model through, so
neither can disagree with what a feature runs on. Only pairings this deployment can actually run
appear; an unconfigured one is absent rather than published empty. `GET /setup` remains the
answer to which pairings *exist* and which are selectable.

The endpoint is read-only and there is no companion write route: what a role resolves to is
deployment configuration, which is the entire point of the role indirection. The payload states
that with an `editable` flag rather than leaving a client to infer it from a `405`. The web
client renders it under Settings → Models, per pairing, with each row's variable name shown so
the place to change a value is visible from the place it is read.

## Validation and compatibility

Settings resolve and validate all four roles at startup, for every configured tier. An unknown
effort is rejected by variable name. `MODEL_REASONING_UNSUPPORTED` may declare unsupported
model/effort pairs as JSON; for the unsuffixed configuration the central resolver then omits that
effort and records both the requested and effective values rather than silently changing models,
and for a tier-suffixed pairing it refuses startup instead.

An older environment containing only `OPENAI_REASONING_MODEL` and `OPENAI_CODING_MODEL` still
starts: `REVIEW` falls back to the reasoning model and `SCOPED_FIX` falls back to the coding model.
The new variables should be set explicitly to obtain independent review and scoped-fix routing.
Legacy effort and former `OPENAI_FIX_*` names remain fallback inputs during migration, but new
configuration and documentation should use the four authoritative names above.
