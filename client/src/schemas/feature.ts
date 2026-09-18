import { z } from 'zod';

/**
 * Zod mirrors of the server response models in `server/api/feature_schemas.py`.
 *
 * Statuses are deliberately `z.string()` rather than enums. The server owns the lifecycle and
 * adds to it -- `inspecting_repositories` was added this week -- and a client that rejects an
 * unknown status turns a backend improvement into a broken page. Display text comes from
 * `/console/status-vocabulary`, which the server already owns and tests.
 */

/** An ISO-8601 instant from the API, kept as a string until something needs to format it. */
const isoTimestamp = z.string();

const failureSummarySchema = z.object({
  stage: z.string(),
  agent: z.string(),
  repository_id: z.string().nullable(),
  root_classification: z.string(),
  attempt: z.number().nullable(),
  command: z.array(z.string()).default([]),
  exit_code: z.number().nullable(),
  diagnostics: z.array(z.string()).default([]),
  retryable: z.boolean(),
  next_action: z.string(),
  recorded_at: isoTimestamp,
});

export const repositorySummarySchema = z.object({
  repository_id: z.string(),
  name: z.string(),
  role: z.string(),
  repository_url: z.string(),
  default_branch: z.string(),
  required: z.boolean(),
  implementation_order: z.number().nullable().optional(),
});

/** One role of the setup a custom feature pinned at submission. */
export const pinnedModelSetupRoleSchema = z.object({
  role: z.string(),
  platform: z.string(),
  model: z.string(),
  reasoning_effort: z.string().nullish(),
  max_tokens: z.number().nullish(),
});

/**
 * The setup a custom feature was pinned to, in full, plus what became of the setup since.
 * The roles here are the snapshot — what actually runs — not the setup's current values.
 */
export const pinnedModelSetupSchema = z.object({
  setup_id: z.string(),
  name: z.string(),
  roles: z.array(pinnedModelSetupRoleSchema).default([]),
  setup_state: z.enum(['unchanged', 'edited', 'deleted']).default('unchanged'),
});

export const featureSchema = z.object({
  feature_id: z.string(),
  workflow_id: z.string(),
  status: z.string(),
  /**
   * Queue-aware live phase. It differs from `status` while an accepted resume or retry is
   * running against that durable checkpoint, and is optional for older API deployments.
   */
  effective_status: z.string().optional(),
  title: z.string(),
  /**
   * The identity people use: `AB-Feature-42`. Allocated by the server at creation and never
   * reassigned. Nullable because a feature created before references existed has none, so the
   * client falls back to the internal id rather than inventing one.
   */
  reference: z.string().nullable().optional(),
  /**
   * How many times a person has revised this feature after completion. 0 is the original
   * run; optional because a server that predates revisions publishes none.
   */
  revision: z.number().optional(),
  current_agent: z.string().nullable(),
  repository_count: z.number(),
  required_repository_count: z.number(),
  repositories: z.array(repositorySummarySchema).default([]),
  clarification_rounds: z.number(),
  integration_review_cycles: z.number(),
  /**
   * The budgets this feature is being run under, published by the server from the feature's
   * own state. They are optional here because a feature written before the server published
   * them has none: a counter with no denominator is shown as a bare count rather than as a
   * ratio the client invented.
   */
  max_clarification_rounds: z.number().optional(),
  max_integration_review_cycles: z.number().optional(),
  max_child_review_cycles: z.number().optional(),
  max_implementation_retries: z.number().optional(),
  max_validation_retries: z.number().optional(),
  max_repository_setup_retries: z.number().optional(),
  merge_strategy: z.string().nullable(),
  deployment_strategy: z.string().nullable(),
  execution_mode: z.string(),
  // Optional for the same reason the budgets above are: a feature read back from a server
  // that predates the field has none, and the detail view shows nothing rather than
  // inventing a provider it cannot know.
  agent_platform: z.string().optional(),
  performance_tier: z.string().optional(),
  /**
   * The setup a custom feature pinned at submission, in full — the snapshot, not the setup's
   * current values — with whether that setup has since been edited or deleted. Null for a
   * tier feature and absent from older servers.
   */
  model_setup: pinnedModelSetupSchema.nullish(),
  cancellation_status: z.string(),
  cancellation_requested_at: isoTimestamp.nullable(),
  cancellation_reason: z.string().nullable(),
  cleanup_requirements: z.array(z.unknown()).default([]),
  failure_summary: failureSummarySchema.nullable().optional(),
  /**
   * The pre-coding stages' two clocks: what the planning calls took, and how much of that
   * classified provider faults consumed. Null where nothing was measured.
   */
  planning_wall_seconds: z.number().nullable().optional(),
  planning_provider_fault_seconds: z.number().nullable().optional(),
  available_actions: z.array(z.string()).optional(),
  created_at: isoTimestamp,
  updated_at: isoTimestamp,
});

export const startFeatureResponseSchema = featureSchema.extend({ created: z.boolean() });

export const featureSummarySchema = z.object({
  feature_id: z.string(),
  workflow_id: z.string(),
  title: z.string(),
  reference: z.string().nullable().optional(),
  status: z.string(),
  execution_mode: z.string(),
  agent_platform: z.string().optional(),
  performance_tier: z.string().optional(),
  created_at: isoTimestamp,
  updated_at: isoTimestamp,
  repository_count: z.number().default(0),
  pull_request_count: z.number().default(0),
  human_action_required: z.boolean().default(false),
  dashboard_group: z.enum(['queued', 'running', 'waiting', 'failed', 'completed', 'cancelled']),
});

export const featureListSchema = z.object({
  features: z.array(featureSummarySchema),
  next_cursor: z.string().nullable().optional(),
});

/**
 * The workstream response carries far more than a list view needs. Only fields the client
 * actually renders are declared; `passthrough` keeps the rest reachable without this schema
 * having to track every diagnostic field the server adds.
 */
export const workstreamSchema = z
  .object({
    repository_id: z.string(),
    repository_name: z.string().nullable().optional(),
    repository_role: z.string().nullable().optional(),
    repository_url: z.string().nullable().optional(),
    repository_required: z.boolean().nullable().optional(),
    child_workflow_id: z.string(),
    workstream_id: z.string(),
    status: z.string(),
    branch_name: z.string(),
    /**
     * The branch this workstream's checkout was created from; null means the repository
     * default. A feature revision sets it to the branch it superseded.
     */
    base_branch: z.string().nullable().optional(),
    workspace_path: z.string(),
    retry_count: z.number(),
    code_completion_artifact_id: z.string().nullable(),
    review_artifact_id: z.string().nullable(),
    blocking_issues: z.array(z.string()).default([]),
    pull_request_artifact_id: z.string().nullable(),
    current_revision: z.string().nullable().optional(),
    current_validation_results: z.array(z.record(z.unknown())).default([]),
    preflight_status: z.string().nullable().optional(),
    blocking_setup_issues: z.array(z.record(z.unknown())).default([]),
    selected_package_manager: z.string().nullable().optional(),
    technology_profile: z.record(z.unknown()).nullable().optional(),
    production_files_changed: z.array(z.string()).default([]),
    test_files_changed: z.array(z.string()).default([]),
    configuration_files_changed: z.array(z.string()).default([]),
    requirements_implemented: z.array(z.string()).default([]),
    requirements_not_implemented: z.array(z.string()).default([]),
    failure_classification: z.string().nullable().optional(),
    /**
     * Which authority demanded the attempt this repository is on — `integration_remediation`,
     * `operator_retry`, or null for a workstream retry, which is also what every attempt
     * recorded before the platform stamped it reads as. The latest attempt's and no other's.
     * Optional: a server that predates the field publishes none.
     */
    targeted_attempt_kind: z.string().nullable().optional(),
    retry_refusal_reason: z.string().nullable().optional(),
    implementation_retry_count: z.number().default(0),
    validation_retry_count: z.number().default(0),
    repository_setup_retry_count: z.number().default(0),
    integration_retry_count: z.number().default(0),
    granted_extra_attempts: z.number().default(0),
    retry_grants: z.array(z.record(z.unknown())).default([]),
    meaningful_change: z.boolean().nullable().optional(),
    meaningful_change_reason: z.string().nullable().optional(),
    scoped_requirements: z.array(z.record(z.unknown())).default([]),
    implementation_expectations: z.array(z.record(z.unknown())).default([]),
    configured_validation_commands: z.array(z.record(z.unknown())).default([]),
    retry_strategy: z.record(z.unknown()).nullable().optional(),
    // Which configured model role this repository's next attempt runs with, and why. Declared
    // in full rather than passed through untyped because the detail panel renders it: a reader
    // asking why a repository is on its third attempt needs to see that the routing moved up a
    // tier, not only that a counter went up.
    model_routing: z
      .object({
        execution_mode: z.string().optional(),
        role: z.string().optional(),
        model: z.string().optional(),
        reasoning: z.string().nullable().optional(),
        classification: z.string().nullable().optional(),
        attempt: z.number().optional(),
        finding_ids: z.array(z.string()).optional(),
        routing_reason: z.string().optional(),
      })
      .passthrough()
      .nullable()
      .optional(),
    test_availability: z.string().nullable().optional(),
    out_of_scope_requirements: z.array(z.string()).default([]),
    /**
     * Reconnaissance failed for this repository and its plan was written without checkout
     * evidence -- the workstream most likely to fail, flagged so an operator deciding
     * whether to trust the plan does not need the server logs.
     */
    planned_blind: z.boolean().default(false),
    planned_blind_reason: z.string().nullable().optional(),
    available_actions: z.array(z.string()).optional(),
    /**
     * Whether pressing `PUBLISH_FEATURE` would open a pull request for this repository, and
     * under which class: `reviewed` for work that passed review and has none yet,
     * `unreviewed` for work whose every required check passed and whose review rejected it.
     * Null means it would not, and `publication_refusal` is the server's sentence saying why.
     * The server decides both; nothing here re-derives them.
     */
    publication_class: z.string().nullable().optional(),
    publication_refusal: z.string().nullable().optional(),
  })
  .passthrough();

export const workstreamsSchema = z.object({
  feature_id: z.string(),
  workstreams: z.array(workstreamSchema),
});

export const artifactSchema = z.object({
  artifact_id: z.string(),
  artifact_type: z.string(),
  workflow_id: z.string().optional(),
  schema_version: z.string(),
  producer: z.string(),
  timestamp: isoTimestamp,
  metadata: z.record(z.unknown()).default({}),
  validation_status: z.string(),
  payload: z.record(z.unknown()).default({}),
});

export const artifactsSchema = z.object({
  feature_id: z.string(),
  artifacts: z.array(artifactSchema),
});

export const pullRequestsSchema = z.object({
  feature_id: z.string(),
  pull_requests: z.array(artifactSchema),
});

export const timelineEventSchema = z.object({
  timestamp: isoTimestamp,
  event_type: z.string(),
  source: z.string(),
  event: z.string(),
  details: z.record(z.unknown()).default({}),
});

export const timelineSchema = z.object({
  feature_id: z.string(),
  events: z.array(timelineEventSchema),
});

export const proposedActionSchema = z.object({
  type: z.string(),
  arguments: z.record(z.unknown()).default({}),
  summary: z.string(),
});

export const chatMessageSchema = z.object({
  id: z.number(),
  role: z.string(),
  content: z.string(),
  proposed_action: proposedActionSchema.nullable().optional(),
  action_status: z.string().nullable().optional(),
  action_result: z.string().nullable().optional(),
  /** The durable action a confirmed proposal became, so a reload can follow it. */
  action_id: z.string().nullable().optional(),
  created_at: isoTimestamp.nullable().optional(),
});

/**
 * A workflow-changing action somebody asked for, and what became of it.
 *
 * This is what makes a confirmation survive a reload: the page follows the action rather
 * than the request that started it, so closing the tab mid-execution loses nothing.
 */
export const featureActionSchema = z.object({
  action_id: z.string(),
  feature_id: z.string(),
  repository_id: z.string().nullable().optional(),
  action_type: z.string(),
  actor_id: z.string(),
  actor_display_name: z.string().nullable().optional(),
  origin: z.string(),
  origin_message_id: z.number().nullable().optional(),
  status: z.string(),
  attempt: z.number(),
  max_attempts: z.number(),
  created_at: isoTimestamp,
  started_at: isoTimestamp.nullable().optional(),
  completed_at: isoTimestamp.nullable().optional(),
  /** Decided by the server against its own clock, never by the browser against its own. */
  in_progress: z.boolean().default(false),
  result_summary: z.string().nullable().optional(),
  error_code: z.string().nullable().optional(),
  error_message: z.string().nullable().optional(),
  external_operation_ids: z.array(z.string()).default([]),
  reconciled_by: z.string().nullable().optional(),
  reconciliation_reason: z.string().nullable().optional(),
  reconciled_at: isoTimestamp.nullable().optional(),
});

export const featureActionsSchema = z.object({
  feature_id: z.string(),
  actions: z.array(featureActionSchema),
});

const repairCommandSchema = z.object({
  command: z.array(z.string()),
  working_directory: z.string().nullable().optional(),
  purpose: z.string(),
});

/** A repository the platform will not change without being told to, and what it proposes. */
export const repositoryRepairSchema = z.object({
  repair_id: z.string(),
  feature_id: z.string(),
  repository_id: z.string(),
  originating_stage: z.string(),
  failure_classification: z.string(),
  detected_problem: z.string(),
  evidence: z.array(z.string()).default([]),
  proposed_repair: z.string(),
  affected_files: z.array(z.string()).default([]),
  affected_dependencies: z.array(z.string()).default([]),
  commands: z.array(repairCommandSchema).default([]),
  expected_impact: z.string(),
  risk: z.string(),
  changes_source_logic: z.boolean(),
  proposed_at_revision: z.string().nullable().optional(),
  status: z.string(),
  approved_by: z.string().nullable().optional(),
  approved_at: isoTimestamp.nullable().optional(),
  rejected_by: z.string().nullable().optional(),
  rejection_reason: z.string().nullable().optional(),
  execution_result: z.string().nullable().optional(),
  resulting_revision: z.string().nullable().optional(),
  /** Whether the repository moved since the diagnosis. A stale repair is shown as stale. */
  stale: z.boolean().default(false),
  created_at: isoTimestamp.nullable().optional(),
});

export const repositoryRepairsSchema = z.object({
  feature_id: z.string(),
  repairs: z.array(repositoryRepairSchema),
});

/** Who this browser is acting as, and what the server will let it do. */
export const actorSchema = z.object({
  actor_id: z.string(),
  display_name: z.string(),
  authentication: z.string(),
  roles: z.array(z.string()).default([]),
  /**
   * Published so the page can hide what it cannot do. Hiding is a courtesy; the server
   * checks every one of these again on the route that performs the action.
   */
  permissions: z.array(z.string()).default([]),
  /** The login identifier. Empty for the shared platform key, which is not an account. */
  subject: z.string().default(''),
  /**
   * Whether this session must replace its password before doing anything else. The client
   * gates the whole application on it; the server does not depend on the client for that.
   */
  must_change_password: z.boolean().default(false),
});

/**
 * A session token, returned exactly once by a successful login.
 *
 * The only response in this client that carries a credential. The platform stores a digest,
 * so there is no endpoint that can show it again -- it goes straight to `writeToken` and is
 * never held anywhere else.
 */
export const loginSchema = z.object({
  token: z.string(),
  expires_at: z.string().nullable().default(null),
  actor: actorSchema,
  must_change_password: z.boolean().default(false),
});

/**
 * One account, as the administrator's list shows it.
 *
 * There is no field here a password could occupy, which is deliberate on both sides: the
 * server's `UserResponse` has none either. `has_password` is the whole of what this surface
 * says about one.
 */
export const platformUserSchema = z.object({
  user_id: z.string(),
  subject: z.string(),
  display_name: z.string(),
  roles: z.array(z.string()).default([]),
  disabled: z.boolean(),
  created_at: z.string(),
  has_password: z.boolean().default(false),
  must_change_password: z.boolean().default(false),
  last_login_at: z.string().nullable().default(null),
});

export const platformUsersSchema = z.object({
  users: z.array(platformUserSchema).default([]),
});

/** A newly minted API token, shown once. */
export const issuedTokenSchema = z.object({
  token_id: z.string(),
  user_id: z.string(),
  label: z.string(),
  token: z.string(),
  expires_at: z.string().nullable().default(null),
});

/** What can safely be shown about a token that already exists: never the token. */
export const platformTokenSchema = z.object({
  token_id: z.string(),
  user_id: z.string(),
  label: z.string(),
  created_at: z.string(),
  expires_at: z.string().nullable().default(null),
  revoked_at: z.string().nullable().default(null),
  last_used_at: z.string().nullable().default(null),
  kind: z.string().default('api'),
});

export const platformTokensSchema = z.object({
  tokens: z.array(platformTokenSchema).default([]),
});

/**
 * Whether a provider credential is configured. There is no field here for the credential
 * itself, and the server has none either.
 */
/**
 * What GitHub said one token reaches, in counts and notes rather than repositories.
 *
 * Rides back on saving a GitHub token, because that is the one moment somebody is looking at
 * the answer. Absent everywhere else — listing and removing credentials ask GitHub nothing.
 */
export const githubAccessSchema = z.object({
  verified: z.enum(['accepted', 'refused', 'unknown']),
  token_kind: z.enum(['classic', 'fine_grained', 'unknown']).default('unknown'),
  scopes: z.array(z.string()).default([]),
  /**
   * False when the listing itself failed. Distinct from a count of zero: "GitHub says you can
   * reach nothing" is an answer, and "the listing did not happen" is the absence of one.
   */
  repositories_listed: z.boolean().default(false),
  repository_count: z.number().default(0),
  writable_count: z.number().default(0),
  truncated: z.boolean().default(false),
  /** Worth reading, and never the reason anything was refused. */
  advisories: z.array(z.string()).default([]),
});

export const credentialSchema = z.object({
  provider: z.string(),
  configured: z.boolean(),
  hint: z.string().default(''),
  created_at: isoTimestamp.nullable().optional(),
  updated_at: isoTimestamp.nullable().optional(),
  expires_at: isoTimestamp.nullable().optional(),
  last_used_at: isoTimestamp.nullable().optional(),
  // Only ever set by saving a GitHub token, and only when a probe answered. A deployment
  // without one, or a GitHub that did not answer, sends null — which is not a bad verdict.
  access: githubAccessSchema.nullish(),
});

export const credentialsSchema = z.object({
  credentials: z.array(credentialSchema),
});

/** One provider the platform requires, and whether this identity has configured it. */
export const providerRequirementSchema = z.object({
  provider: z.string(),
  label: z.string(),
  configured: z.boolean(),
});

/**
 * What still has to be configured before a feature can be created.
 *
 * Server-owned on purpose: which providers are required, and whether this deployment stores
 * credentials at all, are both its decisions. A deployment that stores none reports ready.
 */
/**
 * One (platform, performance tier) a feature could be submitted on.
 *
 * `configured` is a deployment fact only the server knows: it is true when all four of that
 * pairing's model roles resolve. `models` is published with it so the form can say which
 * models a choice runs on without holding a second copy of the deployment's configuration.
 * `performance_tier` defaults to `high` because a server that predates tiers serves entries
 * without it, and everything those servers offer is the high tier.
 *
 * `reasoning_efforts` is keyed like `models`: the effort each role is *sent* at, `null` where
 * the provider's own default applies. A role missing from it is not a role with no effort —
 * it is a server that did not answer, which is what a client reading one that predates the
 * field sees. The two are distinguished at the point they are rendered.
 */
export const agentPlatformSchema = z.object({
  platform: z.string(),
  performance_tier: z.string().default('high'),
  label: z.string(),
  configured: z.boolean(),
  models: z.record(z.string()).default({}),
  reasoning_efforts: z.record(z.string().nullable()).default({}),
  /**
   * Whether this pairing's reasoning model — the one role that reads the submitted PRD — is
   * declared able to be shown an image.
   *
   * Defaulted to `false`, which is both what a server predating the field publishes and the
   * fail-closed answer the server itself would give. The submission form uses it to say so at
   * the selector rather than letting a submission with screenshots meet a 422.
   */
  vision_capable: z.boolean().default(false),
});

export const setupStateSchema = z.object({
  credentials_ready: z.boolean(),
  providers: z.array(providerRequirementSchema).default([]),
  // Defaulted rather than required: a server that predates the field publishes none, and a
  // form with no answer falls back to offering what it was given.
  agent_platforms: z.array(agentPlatformSchema).default([]),
  repositories_ready: z.boolean(),
  saved_repository_count: z.number().default(0),
  credential_storage_available: z.boolean().default(true),
});

/** One repository somebody saved so it is not retyped for every feature. */
export const savedRepositorySchema = z.object({
  configuration_id: z.string(),
  name: z.string(),
  repository_url: z.string(),
  default_branch: z.string(),
  repository_type: z.string(),
  created_at: isoTimestamp,
  updated_at: isoTimestamp,
});

export const savedRepositoriesSchema = z.object({
  repositories: z.array(savedRepositorySchema).default([]),
  suggested_types: z.array(z.string()).default([]),
});

/** One repository a stored GitHub token can reach, as the picker offers it. */
export const githubRepositoryOptionSchema = z.object({
  full_name: z.string(),
  repository_url: z.string(),
  default_branch: z.string(),
  private: z.boolean().default(false),
  archived: z.boolean().default(false),
  /**
   * The account's role in the repository, not the token's own grant — GitHub does not publish
   * the latter for a fine-grained token. A false is conclusive; a true is not a promise.
   */
  can_push: z.boolean().default(false),
  already_saved: z.boolean().default(false),
});

/**
 * The repositories a stored GitHub token can reach, and why there are none when there are.
 *
 * `available` false with a `detail` saying which — no token, no probe, GitHub silent — is why
 * an empty menu never has to be interpreted. The form shows the sentence instead of a menu.
 */
export const githubRepositoriesSchema = z.object({
  available: z.boolean().default(false),
  detail: z.string().default(''),
  access: githubAccessSchema,
  repositories: z.array(githubRepositoryOptionSchema).default([]),
});

export const credentialCheckSchema = z.object({
  provider: z.string(),
  configured: z.boolean(),
  usable: z.boolean(),
  detail: z.string(),
  /**
   * What the provider itself said, when it was asked. Three-valued and absent by default.
   *
   * `usable` answers a different question — whether this deployment can still open what it
   * sealed — and a credential can be readable and refused at the same time. Nullable rather
   * than defaulted to a verdict: no answer is not `unknown` arriving from a provider, it is
   * nobody having asked.
   */
  verified: z.enum(['accepted', 'refused', 'unknown']).nullish(),
});

/**
 * The deployment's Slack delivery configuration. Mirrors `SlackConfigurationResponse`.
 *
 * There is deliberately no field a token could occupy: the bot token lives in the credential
 * store and is described here by `credential_hint` only — the same rule the credentials panel
 * follows.
 */
export const slackConfigurationSchema = z.object({
  configured: z.boolean(),
  enabled: z.boolean().default(false),
  workspace_id: z.string().nullable().optional(),
  workspace_name: z.string().nullable().optional(),
  channel_id: z.string().nullable().optional(),
  channel_name: z.string().nullable().optional(),
  token_owner_id: z.string().nullable().optional(),
  verbosity: z.string().default('milestones'),
  status: z.string().default('disabled'),
  status_reason: z.string().nullable().optional(),
  console_base_url: z.string().nullable().optional(),
  credential_configured: z.boolean().default(false),
  credential_hint: z.string().default(''),
  updated_by: z.string().nullable().optional(),
  updated_at: isoTimestamp.nullable().optional(),
});

/**
 * Where this deployment's designs come from. Mirrors `DesignSourceConfigurationResponse`.
 *
 * There is deliberately no field a token could occupy: the Figma personal access token lives
 * in the credential store and is described here by `credential_hint` only.
 *
 * `file_allowlist_permits_any_file` is carried rather than derived from `file_allowlist.length`
 * because the emptiness is consequential -- empty means any file the configured token can read
 * -- and the server is the authority on what its own control means.
 */
export const designSourceSchema = z.object({
  configured: z.boolean(),
  enabled: z.boolean().default(false),
  token_owner_id: z.string().nullable().optional(),
  file_allowlist: z.array(z.string()).default([]),
  file_allowlist_permits_any_file: z.boolean().default(true),
  status: z.string().default('disabled'),
  status_reason: z.string().nullable().optional(),
  credential_configured: z.boolean().default(false),
  credential_hint: z.string().default(''),
  updated_by: z.string().nullable().optional(),
  updated_at: isoTimestamp.nullable().optional(),
});

/** One person's own Slack link and opt-in scope. Mirrors `SlackLinkResponse`. */
export const slackLinkSchema = z.object({
  user_id: z.string(),
  slack_user_id: z.string().nullable().optional(),
  // A closed set on the server too: opting into mentions is a defined choice, not a status.
  notify_scope: z.enum(['none', 'human_interaction', 'all']).default('none'),
});

/**
 * One role of one (platform, tier) pairing, and everything it resolved to.
 *
 * `requested_reasoning_effort` is present only when it differs from the effective one, which
 * happens when the deployment declared the model does not accept the level configured. Both
 * are shown there, because a normalization presented as a preference is a lie about a setting
 * somebody wrote.
 */
export const modelRoleConfigurationSchema = z.object({
  role: z.string(),
  /**
   * Which provider answers this role. Absent on deployment rows, whose whole pairing is one
   * platform; present on a custom setup's rows, where a mixed setup pins each role's own.
   */
  platform: z.string().nullish(),
  model: z.string(),
  reasoning_effort: z.string().nullish(),
  requested_reasoning_effort: z.string().nullish(),
  max_tokens: z.number().nullish(),
  routing_reason: z.string(),
  /**
   * The environment variable this selection reads — or, on a custom row, the setup that
   * authored it. The panel decides which to say from `origin`, never from this string.
   */
  model_variable: z.string(),
  reasoning_variable: z.string().nullish(),
  resolved_from_legacy_variable: z.boolean().default(false),
});

/** One row of the resolved table: a deployment (platform, tier) pairing, or a custom setup. */
export const modelSetupSchema = z.object({
  platform: z.string(),
  platform_label: z.string(),
  performance_tier: z.string(),
  tier_label: z.string(),
  label: z.string(),
  roles: z.array(modelRoleConfigurationSchema).default([]),
  /** Deployment rows keep their meaning; a custom row is the caller's own authored setup. */
  origin: z.enum(['deployment', 'custom']).default('deployment'),
  setup_id: z.string().nullish(),
  editable: z.boolean().default(false),
});

/**
 * The resolved model-roles table this deployment runs on, plus the caller's own setups.
 *
 * The top-level `editable` means "you may author a setup here" — the deployment persists
 * setups and the caller holds the permission — not "these rows are editable"; each row says
 * that for itself.
 */
export const modelConfigurationSchema = z.object({
  editable: z.boolean().default(false),
  setups: z.array(modelSetupSchema).default([]),
});

/** One role of a setup as its author entered it: platform, model, effort, output bound. */
export const modelSetupRoleInputSchema = z.object({
  platform: z.string(),
  model: z.string(),
  reasoning_effort: z.string().nullish(),
  max_tokens: z.number().nullish(),
});

/**
 * One saved setup, raw as entered, with whether it can currently run.
 *
 * Raw rather than resolved: the authoring form re-populates itself from these values, and the
 * resolved view drops what it normalized. `validation_error` and `missing_credentials` are
 * the server's own sentences — rendered verbatim, because this is where a person meets the
 * AB-Feature-181 refusal and a paraphrase would soften it.
 */
export const savedModelSetupSchema = z.object({
  setup_id: z.string(),
  name: z.string(),
  roles: z.record(modelSetupRoleInputSchema),
  created_at: z.string(),
  updated_at: z.string(),
  platforms: z.array(z.string()).default([]),
  usable: z.boolean().default(true),
  validation_error: z.string().nullish(),
  warnings: z.array(z.string()).default([]),
  missing_credentials: z.array(z.string()).default([]),
  /** Whether this setup's reasoning role reads images. Same rule as a pairing's. */
  vision_capable: z.boolean().default(false),
});

export const modelSetupsSchema = z.object({
  setups: z.array(savedModelSetupSchema).default([]),
});


/**
 * One image uploaded for a submission, as the upload endpoint answers.
 *
 * No URL, by design. Every attachment read is authenticated, so the browser fetches by id
 * with its own token; a field a browser could follow without one is exactly what this
 * response must not carry.
 */
export const attachmentSchema = z.object({
  attachment_id: z.string(),
  filename: z.string(),
  media_type: z.string(),
  byte_size: z.number(),
  sha256: z.string(),
});

/**
 * The caps, as the server enforces them.
 *
 * Read rather than restated so the form's helper text and its client-side refusals cannot
 * drift from the server's — the same reason `schema.ts` mirrors the server's URL rules
 * instead of leaving somebody to read a 422. Defaulted so a form still works against a
 * server that does not publish them.
 */
export const attachmentLimitsSchema = z.object({
  max_attachment_bytes: z.number().default(5 * 1024 * 1024),
  max_attachments_per_feature: z.number().default(8),
  max_attachment_bytes_per_feature: z.number().default(20 * 1024 * 1024),
  accepted_media_types: z.array(z.string()).default(['image/png', 'image/jpeg', 'image/webp']),
});

export const chatHistorySchema = z.object({
  feature_id: z.string(),
  messages: z.array(chatMessageSchema),
});

export const healthSchema = z.object({
  status: z.string(),
  build_revision: z.string(),
  workflow_schema_version: z.string(),
  runtime_compatible: z.boolean(),
});

export const unresolvedOperationsSchema = z.object({
  operations: z.array(
    z.object({
      operation_id: z.string(),
      workflow_id: z.string(),
      feature_id: z.string().nullable(),
      repository_id: z.string().nullable(),
      operation_type: z.string(),
      status: z.string(),
      attempt: z.number(),
      max_attempts: z.number(),
      heartbeat_at: isoTimestamp.nullable(),
    }),
  ),
});

/**
 * One journal row for a repository workstream: what is running, since when, and is it alive.
 *
 * Timestamps arrive raw and never as ages — the server's response is deterministic, and
 * "2s ago" is computed here against this browser's clock. There is deliberately no
 * `safe_metadata` field: the endpoint publishes one typed integer out of it (`child_attempt`)
 * and nothing else, so the client must not expect the dictionary itself.
 */
/**
 * Why one row is one of several of its type inside a single attempt.
 *
 * Two `run_linter` rows in one attempt read as a bug; on run 201's backend attempt 3 both
 * were real — the same command at a new revision, after an in-attempt repair changed the
 * tree. Server-computed from the operation's own `repository_revision` and
 * `command_fingerprint` columns, so this client parses no strings and counts no rows: an
 * ordinal computed here would silently change meaning when the response's row budget
 * truncates the attempt.
 */
export const workstreamOperationRepeatSchema = z.object({
  /** `new_revision` | `different_command` | `same_step`, and unknown values render as served. */
  kind: z.string(),
  detail: z.string().nullable().default(null),
});

export const workstreamOperationSchema = z.object({
  operation_id: z.string(),
  operation_type: z.string(),
  /** The server's stage name for the operation type: setup, coding, validation, review, … */
  stage: z.string(),
  status: z.string(),
  attempt: z.number(),
  max_attempts: z.number(),
  /**
   * Which of the child's attempts this row belongs to, or null where the journal cannot say —
   * a row written before the platform stamped attempts, or one belonging to no attempt at all.
   *
   * Distinct from `attempt` above, which counts this one operation's own retries: a coding
   * call on its third provider attempt inside the child's first attempt is `attempt: 3`,
   * `child_attempt: 0`.
   *
   * Defaulted rather than required, because a deployment serving the client ahead of the
   * server would otherwise fail the whole response over a field whose absence has a meaning
   * this client already handles: unstamped.
   */
  child_attempt: z.number().nullable().default(null),
  started_at: isoTimestamp.nullable(),
  heartbeat_at: isoTimestamp.nullable(),
  completed_at: isoTimestamp.nullable(),
  error_code: z.string().nullable(),
  /**
   * Present only where this attempt holds more than one row of this operation type, and only
   * where the row is not simply the first run of its command. Defaulted rather than required,
   * following `child_attempt`.
   */
  repeat: workstreamOperationRepeatSchema.nullable().default(null),
  /**
   * How many times this row's model call had to be issued again because the stream it opened
   * said nothing inside the first-event budget.
   *
   * `null` means no measurement exists — a row that made no model call, a call whose
   * transport was a plain POST, or a row written before the field. `0` is a measurement and
   * means the stream spoke on the first issue; the two must not be rendered the same way,
   * because the question this answers is whether the budget is close to binding.
   *
   * Defaulted rather than required, following `child_attempt`.
   */
  stream_reissues: z.number().nullable().default(null),
});

/**
 * Where one finished attempt ended, as the server assembled it from records that exist.
 *
 * The journal says what ran and succeeded; it cannot say where an attempt *ended*. Run 201's
 * backend attempt 0 ran 57 minutes with every journaled row ticked and was stopped by the
 * implementation self-review before the reviewer was ever called — and the drawer drew a wall
 * of green. This block is that missing fact, and it is a served fact: nothing in this client
 * derives an ending from operation rows.
 *
 * Present only for attempts the child has moved past. The in-flight attempt has no ending, and
 * an old run that predates the block has none either — both render in words that claim nothing.
 */
export const workstreamAttemptSchema = z.object({
  attempt: z.number(),
  /**
   * The record kind that ended this attempt's own cycle. `z.string()` rather than an enum for
   * the reason every status here is: the server owns the vocabulary and adds to it, and a
   * client that rejected an unknown kind would turn a new ending into a blank response.
   */
  ended_by: z.string(),
  /** The stage the ending landed in, in the same vocabulary the rows carry. */
  stage: z.string(),
  detail: z.string().nullable().default(null),
  /** fresh_checkout | preserved | reset | recovered_coding_output, as recorded. */
  workspace: z.string().nullable().default(null),
  /** The self-review's own outcome, or null where the attempt recorded no self-review. */
  self_review_outcome: z.string().nullable().default(null),
  /** Files a correction rewrote — never a count of correction rounds. */
  self_review_corrected_files: z.number().nullable().default(null),
  source_repair_passes: z.number().nullable().default(null),
  /**
   * Re-issues spent by this attempt's unjournaled in-attempt passes. `null` means nothing
   * measured it; `0` means every stream spoke on the first issue. Only a positive count is
   * worth drawing — but the two zero-ish readings are different facts and stay distinct.
   */
  stream_reissues: z.number().nullable().default(null),
});

export const workstreamOperationsSchema = z.object({
  feature_id: z.string(),
  repository_id: z.string(),
  operations: z.array(workstreamOperationSchema),
  /**
   * One entry per finished attempt, oldest first. Defaulted rather than required, following
   * `child_attempt`: a deployment serving this client ahead of the server must not lose the
   * whole journal over a block whose absence already has a meaning this client handles.
   */
  attempts: z.array(workstreamAttemptSchema).default([]),
});

/**
 * One bubble of a feature's run, and the durable record it was read from.
 *
 * Mirrors `LogbookEntryResponse`. Every field here is composed server-side by
 * `services/logbook.py` from records the platform already wrote — there is no sentence in
 * this client that a logbook bubble could be built out of, deliberately: the whole point of
 * the tab is that nothing between the record and the screen can invent anything.
 *
 * `agent` and `template` are `z.string()` rather than enums for the reason every status here
 * is: the server owns those vocabularies and adds to them, and a client that rejected an
 * unknown chip would turn a new agent into a blank page.
 */
export const logbookEntrySchema = z.object({
  sequence: z.number(),
  /**
   * Which emission of this record this is — a record whose sentence is re-composed keeps its
   * sequence and bumps this. Defaulted rather than required, following `child_attempt`: an
   * older server serving this client must not break the whole tab over one counter.
   */
  emission: z.number().default(0),
  timestamp: isoTimestamp,
  agent: z.string(),
  tone: z.string(),
  /** The registry key the sentence came from. Stable, and shared with Slack delivery. */
  template: z.string(),
  text: z.string(),
  detail: z.string().nullable().optional(),
  /** The agent's own persisted words, clipped at a whole word by the server. */
  quote: z.string().nullable().optional(),
  /** Which persisted field the quote was read from. */
  quote_source: z.string().nullable().optional(),
  record: z.object({
    kind: z.string(),
    id: z.string(),
    repository_id: z.string().nullable().optional(),
  }),
  repository_id: z.string().nullable().optional(),
});

export const logbookSchema = z.object({
  feature_id: z.string(),
  entries: z.array(logbookEntrySchema),
  next_cursor: z.number().nullable().optional(),
  agents: z.array(z.string()).default([]),
});

export const featureEventSchema = z.object({
  id: z.number(),
  timestamp: isoTimestamp,
  event_type: z.string(),
  source: z.string(),
  event: z.string(),
  details: z.record(z.unknown()).default({}),
});

export const featureEventsSchema = z.object({
  feature_id: z.string(),
  events: z.array(featureEventSchema),
  last_event_id: z.number().nullable(),
});

export const clarificationQuestionSchema = z.object({
  question_id: z.string(),
  question: z.string(),
  rationale: z.string(),
  required: z.boolean(),
  /**
   * What the platform would answer, and where it read that. Empty when it has no grounded
   * answer — the client must never fill this in itself, because a suggestion the browser
   * invented is indistinguishable from one the platform justified from a repository.
   */
  suggested_answer: z.string().default(''),
  suggestion_source: z.string().default(''),
  suggestion_confidence: z.string().nullable().optional(),
});

export const designConflictPositionSchema = z.object({
  /**
   * Which review holds this position. Named as the review rather than the agent: what a
   * person arbitrates is which review is right, not which model wrote it.
   */
  authority: z.enum(['repository_review', 'integration_review']),
  statement: z.string(),
  /**
   * What the attempt that stopped reporting this said it had done. Present only on the
   * satisfied side, and only where the lineage held a completion summary for that cycle.
   */
  grounds: z.string().default(''),
});

export const designConflictSchema = z.object({
  conflict_id: z.string(),
  repository_id: z.string(),
  /**
   * Which shape of re-litigation this question is. A `reversal` carries both positions; a
   * `recurring_demand` was never satisfied, so it has no second position and its evidence
   * quotes every wording the review used across the attempts it blocked.
   */
  kind: z.enum(['reversal', 'recurring_demand']).default('reversal'),
  question: z.string(),
  evidence: z.array(z.string()).default([]),
  /**
   * The positions, each in the words it was actually put in. A reversal renders both: what
   * makes it a decision rather than a defect is that the same thing was required, delivered,
   * and then let go, and only the pair shows that. A recurring demand has only the current
   * one — nothing was ever satisfied.
   */
  demanded: designConflictPositionSchema,
  satisfied: designConflictPositionSchema.nullable().optional(),
  cross_authority: z.boolean().default(false),
  removals: z.number(),
  attempts_spent: z.number(),
  attempts_remaining: z.number(),
  /** Whether the server would accept a verdict now. Never inferred here from the counters. */
  answerable: z.boolean().default(true),
});

export const clarificationSchema = z.object({
  feature_id: z.string(),
  awaiting_answers: z.boolean(),
  /**
   * Which of the clarification surface's states this feature is in, serialized by the
   * server rather than re-derived here. Optional so a payload from before the field
   * existed still parses; the fallback is derived from `awaiting_answers` at the render.
   */
  clarification_state: z
    .enum([
      'idle',
      'investigating',
      'awaiting_answers',
      'asked_after_grounding_failure',
      'awaiting_design_verdict',
    ])
    .optional(),
  technical_prd_artifact_id: z.string().nullable(),
  clarification_rounds: z.number(),
  max_clarification_rounds: z.number(),
  questions: z.array(clarificationQuestionSchema).default([]),
  previous_answers: z.record(z.string()).default({}),
  /**
   * Design decisions one workstream stopped on, waiting on a person. A different kind of
   * question from `questions`: about one repository rather than the requirement, arriving
   * after coding rather than before it, and answered through its own endpoint — submitting
   * one as a clarification answer would reopen planning.
   */
  design_conflicts: z.array(designConflictSchema).default([]),
});

/**
 * One execution transition: who moved the feature from one stage to the next.
 *
 * Mirrors `ExecutionRecord` in `server/services/execution_records.py`. Every field the graph
 * shows about a model comes from here, and there is deliberately nothing in this client that
 * can produce a model name any other way — the server reads what each execution recorded, so
 * a completed transition keeps naming the model that actually ran it.
 *
 * `null` means "the platform did not record this" and is rendered as an absence, never as a
 * plausible-looking default.
 */
export const executionRecordSchema = z.object({
  execution_id: z.string(),
  from_stage: z.string(),
  to_stage: z.string(),
  repository_id: z.string().nullable().optional(),
  is_retry: z.boolean().default(false),

  handler_type: z.string(),
  handler: z.string(),
  agent_type: z.string().nullable().optional(),

  provider: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
  reasoning_effort: z.string().nullable().optional(),
  model_role: z.string().nullable().optional(),
  model_variable: z.string().nullable().optional(),
  /** False where the backend has not resolved a model. The client must not predict one. */
  model_resolved: z.boolean().default(false),

  status: z.string(),
  attempt: z.number().nullable().optional(),
  max_attempts: z.number().nullable().optional(),
  /** The server's own sentence for what the ratio counts. Never reworded here. */
  attempt_meaning: z.string().nullable().optional(),
  counter_label: z.string().nullable().optional(),
  counter_value: z.number().nullable().optional(),
  counter_limit: z.number().nullable().optional(),

  failure_classification: z.string().nullable().optional(),
  failure_summary: z.string().nullable().optional(),
  failure_severity: z.string().nullable().optional(),
  remediation_summary: z.string().nullable().optional(),
  routing_reason: z.string().nullable().optional(),

  review_verdict: z.string().nullable().optional(),
  review_finding_count: z.number().nullable().optional(),
  validation_passed: z.number().nullable().optional(),
  validation_total: z.number().nullable().optional(),
  command: z.array(z.string()).default([]),
  exit_code: z.number().nullable().optional(),

  revision_before: z.string().nullable().optional(),
  revision_after: z.string().nullable().optional(),
  contract_version: z.string().nullable().optional(),
  input_fingerprint: z.string().nullable().optional(),

  artifact_id: z.string().nullable().optional(),
  code_completion_artifact_id: z.string().nullable().optional(),
  review_artifact_id: z.string().nullable().optional(),
  result_artifact_id: z.string().nullable().optional(),
  pull_request_artifact_id: z.string().nullable().optional(),
  repair_id: z.string().nullable().optional(),
  previous_execution_id: z.string().nullable().optional(),

  human_action: z.string().nullable().optional(),
  human_requirement: z.string().nullable().optional(),

  /**
   * Operation-derived records (the journaled planning-stage model calls) carry a real
   * start and last heartbeat; artifact-derived records carry neither.
   */
  started_at: isoTimestamp.nullable().optional(),
  heartbeat_at: isoTimestamp.nullable().optional(),
  completed_at: isoTimestamp.nullable().optional(),
  duration_seconds: z.number().nullable().optional(),
  execution_mode: z.string(),
});

export const featureExecutionsSchema = z.object({
  feature_id: z.string(),
  executions: z.array(executionRecordSchema),
});

export type ExecutionRecord = z.infer<typeof executionRecordSchema>;
export type FeatureExecutions = z.infer<typeof featureExecutionsSchema>;

export type WorkstreamOperation = z.infer<typeof workstreamOperationSchema>;
export type WorkstreamAttempt = z.infer<typeof workstreamAttemptSchema>;
export type WorkstreamOperations = z.infer<typeof workstreamOperationsSchema>;

export type LogbookEntry = z.infer<typeof logbookEntrySchema>;
export type Logbook = z.infer<typeof logbookSchema>;

export type RepositorySummary = z.infer<typeof repositorySummarySchema>;
export type Feature = z.infer<typeof featureSchema>;
export type FeatureSummary = z.infer<typeof featureSummarySchema>;
export type FeatureList = z.infer<typeof featureListSchema>;
export type Workstream = z.infer<typeof workstreamSchema>;
export type Artifact = z.infer<typeof artifactSchema>;
export type TimelineEvent = z.infer<typeof timelineEventSchema>;
export type Clarification = z.infer<typeof clarificationSchema>;
export type FeatureEvent = z.infer<typeof featureEventSchema>;
export type ChatMessage = z.infer<typeof chatMessageSchema>;
export type FeatureAction = z.infer<typeof featureActionSchema>;
export type RepositoryRepair = z.infer<typeof repositoryRepairSchema>;
export type Actor = z.infer<typeof actorSchema>;
export type Credential = z.infer<typeof credentialSchema>;
export type ClarificationQuestion = z.infer<typeof clarificationQuestionSchema>;
export type DesignConflict = z.infer<typeof designConflictSchema>;
export type SetupState = z.infer<typeof setupStateSchema>;
export type AgentPlatformOption = z.infer<typeof agentPlatformSchema>;
export type ProviderRequirement = z.infer<typeof providerRequirementSchema>;
export type SavedRepository = z.infer<typeof savedRepositorySchema>;
export type GitHubAccess = z.infer<typeof githubAccessSchema>;
export type GitHubRepositoryOption = z.infer<typeof githubRepositoryOptionSchema>;
export type GitHubRepositories = z.infer<typeof githubRepositoriesSchema>;
export type CredentialCheck = z.infer<typeof credentialCheckSchema>;
export type SlackConfiguration = z.infer<typeof slackConfigurationSchema>;
export type DesignSource = z.infer<typeof designSourceSchema>;
export type SlackLink = z.infer<typeof slackLinkSchema>;
export type ModelRoleConfiguration = z.infer<typeof modelRoleConfigurationSchema>;
export type ModelSetup = z.infer<typeof modelSetupSchema>;
export type ModelConfiguration = z.infer<typeof modelConfigurationSchema>;
export type ModelSetupRoleInput = z.infer<typeof modelSetupRoleInputSchema>;
export type SavedModelSetup = z.infer<typeof savedModelSetupSchema>;
export type ModelSetups = z.infer<typeof modelSetupsSchema>;
export type Attachment = z.infer<typeof attachmentSchema>;
export type AttachmentLimits = z.infer<typeof attachmentLimitsSchema>;
export type PinnedModelSetup = z.infer<typeof pinnedModelSetupSchema>;
export type PinnedModelSetupRole = z.infer<typeof pinnedModelSetupRoleSchema>;
export type Login = z.infer<typeof loginSchema>;
export type PlatformUser = z.infer<typeof platformUserSchema>;
export type IssuedToken = z.infer<typeof issuedTokenSchema>;
export type PlatformToken = z.infer<typeof platformTokenSchema>;
