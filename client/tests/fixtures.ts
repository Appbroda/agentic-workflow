import type { FeatureApi } from '@/api/features';
import { setupStateSchema, type SetupState } from '@/schemas/feature';
import setupPerformanceTiers from './fixtures/setup.performance-tiers.json';

/**
 * Fixtures shaped like the real responses, taken from features that actually ran against the
 * pilot repositories. Invented shapes would let a test pass while the client cannot read what
 * the server sends.
 */

export const STATUS_VOCABULARY = {
  feature: {
    completed: { headline: 'Finished', detail: 'All work is done.', next_step: '', tone: 'done' },
    failed_requires_human: {
      headline: 'Partly done — needs an engineer',
      detail: 'Some repositories finished; one needs a person.',
      next_step: 'Open the feature and read the blocking issues.',
      tone: 'attention',
    },
    waiting_for_human: {
      headline: 'Waiting on you',
      detail: 'The platform asked a question.',
      next_step: 'Answer the open questions.',
      tone: 'attention',
    },
    resuming: {
      headline: 'Resuming',
      detail: 'Your answers were accepted and the workflow is continuing in the background.',
      next_step: 'Nothing to do. Do not submit the clarification form again.',
      tone: 'working',
    },
    retrying: {
      headline: 'Retrying',
      detail: 'The requested retry is running in the background.',
      next_step: 'Nothing to do yet.',
      tone: 'working',
    },
    running_child_workflows: {
      headline: 'Building',
      detail: 'Repositories are being implemented.',
      next_step: 'Nothing to do yet.',
      tone: 'working',
    },
    cancelled: { headline: 'Cancelled', detail: '', next_step: '', tone: 'stopped' },
  },
  workstream: {
    completed: { headline: 'Done', detail: '', next_step: '', tone: 'done' },
    failed: { headline: 'Stopped', detail: '', next_step: '', tone: 'stopped' },
    running: { headline: 'Working', detail: '', next_step: '', tone: 'working' },
  },
} as const;

export function featureSummary(overrides: Partial<Summary> = {}): Summary {
  return {
    feature_id: 'adunit-deactivate-live-086',
    workflow_id: 'adunit-deactivate-live-086',
    reference: 'AB-Feature-86',
    title: 'Deactivate ad units from the console',
    status: 'completed',
    execution_mode: 'live',
    created_at: '2026-08-24T22:40:00Z',
    updated_at: '2026-08-24T23:20:00Z',
    repository_count: 2,
    pull_request_count: 2,
    human_action_required: false,
    dashboard_group: 'completed',
    ...overrides,
  };
}

type Summary = Awaited<ReturnType<FeatureApi['listFeatures']>>['features'][number];

export const FEATURE_PAGE = {
  features: [
    featureSummary(),
    featureSummary({
      feature_id: 'admin-server-status-live-082',
      reference: 'AB-Feature-82',
      title: 'Server uptime history on the admin console',
      status: 'failed_requires_human',
      pull_request_count: 0,
      human_action_required: true,
      dashboard_group: 'waiting',
    }),
    featureSummary({
      feature_id: 'feature-in-flight',
      reference: 'AB-Feature-90',
      title: 'Something still running',
      status: 'running_child_workflows',
      pull_request_count: 0,
      dashboard_group: 'running',
    }),
  ],
  next_cursor: null,
};

/**
 * One saved repository, as Settings and the New Feature picker read them.
 *
 * The shape a real `/repositories` response has: the name and the identifier are the server's,
 * derived and generated respectively, so neither is something a test should invent freely.
 */
export function savedRepository(
  overrides: Partial<SavedRepositoryFixture> = {},
): SavedRepositoryFixture {
  return {
    configuration_id: 'repo-1',
    name: 'admanager_console-2.0',
    repository_url: 'https://github.com/Appbroda/admanager_console-2.0',
    default_branch: 'master',
    repository_type: 'Backend',
    created_at: '2026-08-26T09:00:00Z',
    updated_at: '2026-08-26T09:00:00Z',
    ...overrides,
  };
}

type SavedRepositoryFixture = Awaited<
  ReturnType<FeatureApi['listSavedRepositories']>
>['repositories'][number];

/**
 * What a GitHub token reaches, as the repository picker reads it.
 *
 * The second entry is deliberately one the account cannot push to: a menu where every option
 * is selectable never exercises the case the picker exists to make legible.
 */
export function githubRepositories(
  overrides: Partial<GitHubRepositoriesFixture> = {},
): GitHubRepositoriesFixture {
  return {
    available: true,
    detail: '',
    access: {
      verified: 'accepted',
      token_kind: 'classic',
      scopes: ['repo', 'workflow'],
      repositories_listed: true,
      repository_count: 2,
      writable_count: 1,
      truncated: false,
      advisories: [],
    },
    repositories: [
      {
        full_name: 'Appbroda/admanager_console-2.0',
        repository_url: 'https://github.com/Appbroda/admanager_console-2.0',
        default_branch: 'master',
        private: true,
        archived: false,
        can_push: true,
        already_saved: false,
      },
      {
        full_name: 'Appbroda/read-only-service',
        repository_url: 'https://github.com/Appbroda/read-only-service',
        default_branch: 'main',
        private: false,
        archived: false,
        can_push: false,
        already_saved: false,
      },
    ],
    ...overrides,
  };
}

type GitHubRepositoriesFixture = Awaited<ReturnType<FeatureApi['listGitHubRepositories']>>;

/**
 * A deployment where both prerequisites are satisfied, which is the ordinary case.
 *
 * The saved response of a real `GET /setup` from the platform with both providers' tier
 * presets configured -- six (platform, tier) options -- parsed through the same schema the
 * client parses production responses with. Saved rather than hand-written because a fixture
 * written by hand contains exactly the fields its author remembered; refresh it by re-running
 * the capture against a configured server (see the fixture file's provenance in the report
 * for `42-performance-tiers`). The interesting cases -- a selection the deployment cannot
 * run, an option that must render disabled -- are the ones a test states for itself.
 */
export const SETUP_READY = setupStateSchema.parse(
  setupPerformanceTiers,
) as SetupState;

/**
 * A real 1x1 PNG, as bytes.
 *
 * Real rather than a placeholder string, because it is handed to `URL.createObjectURL` and
 * rendered as an image: a blob of arbitrary bytes would work here and would stop working the
 * day anything asked what it was.
 */
export const ONE_PIXEL_PNG = Uint8Array.from(
  atob(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==',
  ),
  (character) => character.charCodeAt(0),
);

/** A minimal API stub; every test declares only the operations it exercises. */
export function stubApi(overrides: Partial<FeatureApi>): FeatureApi {
  return {
    getStatusVocabulary: async () => STATUS_VOCABULARY,
    // Both prerequisites satisfied by default: a test about anything other than the setup
    // gates should not have to describe them.
    getSetupState: async () => SETUP_READY,
    listSavedRepositories: async () => ({
      repositories: [savedRepository()],
      suggested_types: ['Frontend', 'Backend', 'Service', 'Other'],
    }),
    // The repository form asks GitHub what the account's token reaches before it offers a
    // menu, so a test about anything else does not have to describe that answer.
    listGitHubRepositories: async () => githubRepositories(),
    listActions: async (featureId: string) => ({ feature_id: featureId, actions: [] }),
    // A feature with no recorded executions draws exactly the graph it drew before this
    // existed, so a test about anything else does not have to describe them.
    getExecutions: async (featureId: string) => ({ feature_id: featureId, executions: [] }),
    // No journal rows means no liveness indicator and no sub-stage strip, which is the
    // pre-endpoint graph.
    listWorkstreamOperations: async (featureId: string, repositoryId: string) => ({
      feature_id: featureId,
      repository_id: repositoryId,
      operations: [],
      attempts: [],
    }),
    // A feature with no story yet renders the logbook's empty state, so a test about
    // anything else does not have to describe one.
    getLogbook: async (featureId: string) => ({
      feature_id: featureId,
      entries: [],
      next_cursor: null,
      agents: [],
    }),
    // A deployment that resolved no models renders the panel's own empty state, so a test
    // about anything other than model configuration does not have to describe a table.
    getModelConfiguration: async () => ({ editable: false, setups: [] }),
    // No authored setups by default: the submission form offers only the pairings, and a
    // test about custom setups states its own list.
    listModelSetups: async () => ({ setups: [] }),
    // Slack delivery unconfigured and no personal link by default: the settings panel and the
    // profile form render their blank states, so a test about anything else ignores them.
    getSlackConfiguration: async () => ({
      configured: false,
      enabled: false,
      workspace_id: null,
      workspace_name: null,
      channel_id: null,
      channel_name: null,
      token_owner_id: null,
      verbosity: 'milestones',
      status: 'disabled',
      status_reason: null,
      console_base_url: null,
      credential_configured: false,
      credential_hint: '',
      updated_by: null,
      updated_at: null,
    }),
    getSlackLink: async () => ({
      user_id: 'platform-admin',
      slack_user_id: null,
      notify_scope: 'none',
    }),
    // The caps the shipped server publishes, so the submission form's helper text and its
    // client-side refusals are the real numbers in every test that renders the form -- not
    // just in the ones about attachments.
    getAttachmentLimits: async () => ({
      max_attachment_bytes: 5 * 1024 * 1024,
      max_attachments_per_feature: 8,
      max_attachment_bytes_per_feature: 20 * 1024 * 1024,
      accepted_media_types: ['image/png', 'image/jpeg', 'image/webp'],
    }),
    // A one-pixel PNG for any thumbnail a test renders. Present by default because
    // `AttachmentImage` calls it the moment an attachment row exists, and a stub missing it
    // takes the surrounding document down rather than failing the one assertion about it.
    getAttachmentBlob: async () => new Blob([ONE_PIXEL_PNG], { type: 'image/png' }),
    deleteAttachment: async () => undefined,
    // No design source by default, for the reason Slack has none: the panel renders its own
    // blank state and a test about anything else ignores it. This is also the honest default
    // -- most deployments cite no designs, and none of this is reachable without a citation.
    getDesignSource: async () => ({
      configured: false,
      enabled: false,
      token_owner_id: null,
      file_allowlist: [],
      file_allowlist_permits_any_file: true,
      status: 'disabled',
      status_reason: null,
      credential_configured: false,
      credential_hint: '',
      updated_by: null,
      updated_at: null,
    }),
    getMe: async () => ({
      actor_id: 'platform-admin',
      display_name: 'Platform operator',
      authentication: 'platform_key',
      roles: ['admin'],
      permissions: ['feature:read', 'action:reconcile'],
    }),
    ...overrides,
  } as unknown as FeatureApi;
}
