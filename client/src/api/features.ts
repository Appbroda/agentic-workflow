import { z } from 'zod';
import type { ApiClient, ProviderCredentials, RequestOptions } from './client';
import { readEventStream, type StreamEvent } from './stream';
import { readToken } from './token';
import { clearPendingActionKey, pendingActionKey } from './action-idempotency';
import { ApiError } from './errors';
import {
  actorSchema,
  artifactSchema,
  artifactsSchema,
  attachmentLimitsSchema,
  attachmentSchema,
  chatHistorySchema,
  chatMessageSchema,
  clarificationSchema,
  credentialCheckSchema,
  credentialSchema,
  credentialsSchema,
  featureActionSchema,
  featureActionsSchema,
  featureEventsSchema,
  featureExecutionsSchema,
  featureListSchema,
  featureSchema,
  githubRepositoriesSchema,
  healthSchema,
  issuedTokenSchema,
  loginSchema,
  logbookSchema,
  modelConfigurationSchema,
  modelSetupsSchema,
  platformTokensSchema,
  platformUserSchema,
  platformUsersSchema,
  savedModelSetupSchema,
  pullRequestsSchema,
  repositoryRepairsSchema,
  savedRepositoriesSchema,
  savedRepositorySchema,
  setupStateSchema,
  designSourceSchema,
  slackConfigurationSchema,
  slackLinkSchema,
  startFeatureResponseSchema,
  timelineSchema,
  unresolvedOperationsSchema,
  workstreamOperationsSchema,
  workstreamsSchema,
} from '@/schemas/feature';

/**
 * Domain operations, one per real backend capability.
 *
 * Every name here corresponds to an endpoint that exists in `server/api/feature_routes.py`.
 * A capability the backend does not have is deliberately absent rather than stubbed, so a
 * missing feature is a compile error at the call site instead of a request that fails at
 * runtime.
 *
 * The two streaming calls return async generators rather than promises, because a stream is
 * something you read until you stop rather than something that arrives.
 */

/**
 * Shape accepted by `POST /features/start`. Mirrors `StartFeatureRequest`.
 *
 * A submission names either the pairing (`agent_platform` + `performance_tier`) or a custom
 * setup (`model_setup_id`) — never both. The server refuses two answers to one question, so
 * the fields are optional here and `toStartFeatureInput` sends exactly one selection.
 */
export interface StartFeatureInput {
  feature_id?: string;
  prd: unknown;
  repositories: unknown[];
  execution_mode: 'mock' | 'live';
  /** Which model provider runs this feature's agents. Fixed once it starts. */
  agent_platform?: 'openai' | 'anthropic';
  /** How expensively that provider's model roles resolve. Fixed with the platform. */
  performance_tier?: 'low' | 'medium' | 'high' | 'ultra';
  /** A user-authored setup to pin instead of a (platform, tier) pairing. */
  model_setup_id?: string;
}

/** One image reference travelling with a submission. Mirrors `PRDAttachmentRef`. */
export interface AttachmentReference {
  attachment_id: string;
  marker: string;
  caption: string;
}

/** Shape accepted by `POST /model-setups` and `PUT /model-setups/{id}`. */
export interface SaveModelSetupInput {
  name: string;
  roles: Record<
    string,
    {
      platform: string;
      model: string;
      reasoning_effort?: string | null;
      max_tokens?: number | null;
    }
  >;
}

/** Shape accepted by `PUT /slack-configuration`. Mirrors `SlackConfigurationRequest`. */
export interface SaveSlackConfigurationInput {
  enabled: boolean;
  channel_id: string;
  channel_name?: string | null;
  /** The detailed feed is defined but not delivered yet, so the server accepts only this. */
  verbosity: 'milestones';
  console_base_url?: string | null;
  /** Which identity's `slack` credential the dispatcher resolves. Null means the saver. */
  token_owner_id?: string | null;
}

/** Shape accepted by `PUT /design-source`. Mirrors `DesignSourceConfigurationRequest`. */
export interface SaveDesignSourceInput {
  enabled: boolean;
  /** File keys, never URLs: the server refuses a pasted URL by name, and says why. */
  file_allowlist: string[];
  /** Whose stored `figma` credential the resolver opens. Null means the saver. */
  token_owner_id?: string | null;
}

/** Shape accepted by `PUT /account/slack-link`. Mirrors `SlackLinkRequest`. */
export interface SaveSlackLinkInput {
  slack_user_id?: string | null;
  notify_scope: 'none' | 'human_interaction' | 'all';
}

export interface ClarificationAnswerInput {
  question_id: string;
  answer: string;
}

const statusVocabularySchema = z.record(z.record(z.object({
  headline: z.string(),
  detail: z.string(),
  next_step: z.string(),
  tone: z.string(),
})));

export type StatusVocabulary = z.infer<typeof statusVocabularySchema>;
export const FEATURE_EVENT_PAGE_SIZE = 200;
/** One screenful of a run's story, several times over. The server caps this at 500. */
export const LOGBOOK_PAGE_SIZE = 200;

async function durableAction<T>(scope: string, operation: (key: string) => Promise<T>): Promise<T> {
  const key = pendingActionKey(scope);
  try {
    const result = await operation(key);
    clearPendingActionKey(scope);
    return result;
  } catch (error) {
    // A timeout, disconnect, aborted navigation, or 5xx does not prove the backend stopped.
    // Keep the key through refresh so the next request observes the same durable action.
    if (!(error instanceof ApiError) || (!error.isRetryable && error.kind !== 'aborted')) {
      clearPendingActionKey(scope);
    }
    throw error;
  }
}

export function createFeatureApi(client: ApiClient) {
  const call = <S extends z.ZodTypeAny>(path: string, schema: S, options?: RequestOptions) =>
    client.request(path, schema, options);

  return {
    listFeatures: (params: { limit?: number; cursor?: string | null } = {}, signal?: AbortSignal) =>
      call('/features', featureListSchema, {
        query: { limit: params.limit ?? 20, cursor: params.cursor ?? undefined },
        signal,
      }),

    getFeature: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}`, featureSchema, { signal }),

    getWorkstreams: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/workstreams`, workstreamsSchema, { signal }),

    /**
     * One repository's external-operation journal rows, newest first and bounded.
     *
     * This is the liveness read: while an operation runs, its row is the only record with a
     * start time and a heartbeat, so this is polled rather than keyed by the event cursor —
     * a heartbeat renewal writes no lifecycle event.
     */
    listWorkstreamOperations: (featureId: string, repositoryId: string, signal?: AbortSignal) =>
      call(
        `/features/${encodeURIComponent(featureId)}/workstreams/${encodeURIComponent(repositoryId)}/operations`,
        workstreamOperationsSchema,
        { signal },
      ),

    /**
     * Lifecycle events after a cursor. This is what a client polls to stay current: it reads
     * the indexed event table, where the timeline endpoint hydrates every artifact.
     */
    listEvents: (featureId: string, after: number | null, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/events`, featureEventsSchema, {
        query: { after: after ?? undefined, limit: FEATURE_EVENT_PAGE_SIZE },
        signal,
      }),

    getTimeline: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/timeline`, timelineSchema, { signal }),

    /**
     * The feature's run as a conversation, composed server-side from its durable records.
     *
     * Like the timeline it reads parent state, so it is opened rather than polled, and it
     * pages by entry: a twelve-attempt run has several hundred bubbles and the thread asks
     * for the next page when a reader gets to the end of one.
     */
    getLogbook: (featureId: string, after: number | null, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/logbook`, logbookSchema, {
        query: { after: after ?? undefined, limit: LOGBOOK_PAGE_SIZE },
        signal,
      }),

    /**
     * Who performed each transition in the execution graph, for the whole feature at once.
     *
     * One request rather than one per arrow: a five-repository feature on its fourth attempt
     * has well over a hundred transitions, and the graph must not cost a request for each.
     */
    getExecutions: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/executions`, featureExecutionsSchema, {
        signal,
      }),

    getPullRequests: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/pull-requests`, pullRequestsSchema, { signal }),

    /**
     * Envelopes by default. A completed two-repository feature returns roughly 400 KB with every
     * payload included, so a list view asks for shape and opens one artifact on demand.
     */
    listArtifacts: (
      featureId: string,
      params: { artifactType?: string; includePayload?: boolean } = {},
      signal?: AbortSignal,
    ) =>
      call(`/features/${encodeURIComponent(featureId)}/artifacts`, artifactsSchema, {
        query: {
          artifact_type: params.artifactType,
          include_payload: params.includePayload ?? false,
        },
        signal,
      }),

    getArtifact: (featureId: string, artifactId: string, signal?: AbortSignal) =>
      call(
        `/features/${encodeURIComponent(featureId)}/artifacts/${encodeURIComponent(artifactId)}`,
        artifactSchema,
        { signal },
      ),

    getClarification: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/clarification`, clarificationSchema, { signal }),

    createFeature: (
      input: StartFeatureInput,
      options: { credentials?: ProviderCredentials; idempotencyKey: string },
    ) =>
      call('/features/start', startFeatureResponseSchema, {
        method: 'POST',
        body: input,
        credentials: options.credentials,
        idempotencyKey: options.idempotencyKey,
        // A live submission runs reconnaissance across every repository before it answers.
        timeoutMs: 300_000,
      }),

    /**
     * Answers a clarification, or recovers an interrupted feature when `answers` is empty.
     * Both are the same endpoint on the server; the distinction is the payload.
     */
    resumeFeature: (
      featureId: string,
      answers: ClarificationAnswerInput[],
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`resume:${featureId}`, (idempotencyKey) =>
        call(`/features/${encodeURIComponent(featureId)}/resume`, featureSchema, {
          method: 'POST',
          body: { answers },
          credentials: options.credentials,
          idempotencyKey,
          timeoutMs: 300_000,
        }),
      ),

    /**
     * Open the pull requests a feature that did not land is holding.
     *
     * A feature that fully landed publishes itself. One that did not publishes nothing until
     * somebody decides it should, because a pull request whose sibling repository does not
     * exist is not reviewable work. This is that decision, so the server requires a reason
     * and records it -- and for a repository whose review rejected the work, the pull request
     * it opens carries code nothing approved.
     *
     * A push and a provider call per repository run on a worker, so this takes the long
     * timeout even though it is accepted rather than completed inside the request.
     */
    publishFeature: (
      featureId: string,
      input: { requested_by: string; reason: string },
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`publish:${featureId}`, (idempotencyKey) =>
        call(`/features/${encodeURIComponent(featureId)}/publish`, featureSchema, {
          method: 'POST',
          body: input,
          credentials: options.credentials,
          idempotencyKey,
          timeoutMs: 300_000,
        }),
      ),

    /**
     * Ask for changes to a completed feature's published work.
     *
     * The request text becomes the revision run's requirement set verbatim: the work is
     * revised on a new `-V{n}` branch created from the branch it published, a replacement
     * pull request opens, and the superseded one is closed with a cross-link comment. The
     * revision is applied synchronously and the run is queued, so this takes the long
     * timeout even though it is accepted rather than completed inside the request.
     */
    reviseFeature: (
      featureId: string,
      input: { request: string; requested_by?: string },
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`revise:${featureId}`, (idempotencyKey) =>
        call(`/features/${encodeURIComponent(featureId)}/revise`, featureSchema, {
          method: 'POST',
          body: input,
          credentials: options.credentials,
          idempotencyKey,
          timeoutMs: 300_000,
        }),
      ),

    /**
     * Buy one stopped repository more attempts.
     *
     * This overrides a decision the platform made on purpose, so the operator and the reason
     * are required by the server and are recorded against the repository. The request runs a
     * real attempt -- clone, code, validate, push -- and so takes the long timeout.
     */
    retryWorkstream: (
      featureId: string,
      repositoryId: string,
      input: { additional_attempts: number; requested_by: string; reason: string },
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`retry:${featureId}:${repositoryId}`, (idempotencyKey) =>
        call(
          `/features/${encodeURIComponent(featureId)}/workstreams/${encodeURIComponent(repositoryId)}/retry`,
          featureSchema,
          {
            method: 'POST',
            body: input,
            credentials: options.credentials,
            idempotencyKey,
            timeoutMs: 300_000,
          },
        ),
      ),

    /**
     * Settle one re-litigated design decision, and let the repository act on it.
     *
     * `decision` is not optional and is not a comment: the server hands it to the next attempt
     * verbatim as the invariant it may not argue, which is the difference between an
     * instruction an engineer follows and a preference it reverses. `additional_attempts`
     * defaults to zero because the stop this answers returned the workstream's budget unspent;
     * anything above zero is an ordinary grant and is recorded as one.
     *
     * Accepted and queued, then a real attempt runs on a worker, so this takes the long
     * timeout for the same reason a retry grant does.
     */
    answerDesignConflict: (
      featureId: string,
      conflictId: string,
      input: {
        verdict: 'requirement_holds' | 'removal_holds';
        decision: string;
        additional_attempts?: number;
      },
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`design-verdict:${featureId}:${conflictId}`, (idempotencyKey) =>
        call(
          `/features/${encodeURIComponent(featureId)}/design-conflicts/${encodeURIComponent(conflictId)}/answer`,
          featureSchema,
          {
            method: 'POST',
            body: input,
            credentials: options.credentials,
            idempotencyKey,
            timeoutMs: 300_000,
          },
        ),
      ),

    cancelFeature: (featureId: string, reason?: string) =>
      durableAction(`cancel:${featureId}`, (idempotencyKey) =>
        call(`/features/${encodeURIComponent(featureId)}/cancel`, featureSchema, {
          method: 'POST',
          body: { reason: reason ?? null },
          idempotencyKey,
        }),
      ),

    retireFeature: (featureId: string, input: { operator: string; reason: string }) =>
      durableAction(`retire:${featureId}`, (idempotencyKey) =>
        call(`/features/${encodeURIComponent(featureId)}/retire`, featureSchema, {
          method: 'POST',
          body: input,
          idempotencyKey,
        }),
      ),

    approveContractChange: (
      featureId: string,
      requestId: string,
      body: { resolution: string; updated_contract: unknown },
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`contract-approve:${featureId}:${requestId}`, (idempotencyKey) =>
        call(
          `/features/${encodeURIComponent(featureId)}/contract-change-requests/${encodeURIComponent(requestId)}/approve`,
          featureSchema,
          {
            method: 'POST',
            body,
            credentials: options.credentials,
            idempotencyKey,
            timeoutMs: 300_000,
          },
        ),
      ),

    rejectContractChange: (
      featureId: string,
      requestId: string,
      body: { resolution: string },
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`contract-reject:${featureId}:${requestId}`, (idempotencyKey) =>
        call(
          `/features/${encodeURIComponent(featureId)}/contract-change-requests/${encodeURIComponent(requestId)}/reject`,
          featureSchema,
          { method: 'POST', body, credentials: options.credentials, idempotencyKey },
        ),
      ),

    /** Readiness, and the build the API is actually running. */
    getReadiness: (signal?: AbortSignal) => call('/readyz', healthSchema, { signal }),

    /**
     * External operations the platform could not resolve. Real operator data, not a setting:
     * these need a person to look at the repository or the provider.
     */
    listUnresolvedOperations: (signal?: AbortSignal) =>
      call('/features/operations/unresolved', unresolvedOperationsSchema, { signal }),

    getChatHistory: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/chat`, chatHistorySchema, { signal }),

    /**
     * Ask the assistant one question.
     *
     * The provider key travels as a request header, exactly as it does for the agents that do
     * the work: this platform keeps provider credentials request-scoped and the deployment
     * holds none. Without one the server answers 503 and records nothing.
     */
    sendChatMessage: (
      featureId: string,
      message: string,
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      call(`/features/${encodeURIComponent(featureId)}/chat`, chatHistorySchema, {
        method: 'POST',
        body: { message },
        credentials: options.credentials,
        // The assistant reads the feature's record before answering.
        timeoutMs: 120_000,
      }),

    /**
     * Runs a proposal a person confirmed. The server executes it through the same control
     * plane the ordinary buttons use, and refuses it on the same terms.
     */
    confirmChatAction: (
      featureId: string,
      messageId: number,
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      call(
        `/features/${encodeURIComponent(featureId)}/chat/${messageId}/confirm`,
        chatMessageSchema,
        { method: 'POST', credentials: options.credentials, timeoutMs: 300_000 },
      ),

    rejectChatAction: (featureId: string, messageId: number) =>
      call(
        `/features/${encodeURIComponent(featureId)}/chat/${messageId}/reject`,
        chatMessageSchema,
        { method: 'POST' },
      ),

    /**
     * Plain-language wording for every feature and workstream state, owned and tested by the
     * server in `api/console.py`. Reused rather than restated so status copy has one home.
     */
    getStatusVocabulary: (signal?: AbortSignal) =>
      call('/console/status-vocabulary', statusVocabularySchema, { signal }),

    /** Who this browser is acting as, and what the server will let it do. */
    getMe: (signal?: AbortSignal) => call('/me', actorSchema, { signal }),

    /**
     * Exchange an email and a password for a session token.
     *
     * The one call in this client that both needs no credential and returns one. The caller
     * writes the token; this does not, so that "who stores the session" stays one decision
     * made in one place -- see `LoginPage`.
     */
    login: (email: string, password: string, signal?: AbortSignal) =>
      call('/auth/login', loginSchema, {
        method: 'POST',
        body: { email, password },
        signal,
      }),

    /**
     * Revoke the token this browser is holding.
     *
     * Called before the token is cleared locally, and its failure is not allowed to stop the
     * local clear -- a person who pressed sign out must end up signed out of this browser
     * whether or not the server was reachable to hear about it.
     */
    logout: (signal?: AbortSignal) =>
      call('/auth/logout', z.object({}).passthrough(), { method: 'POST', signal }),

    /** Replace this account's own password, having proved the current one. */
    changePassword: (currentPassword: string, newPassword: string, signal?: AbortSignal) =>
      call('/auth/password', actorSchema, {
        method: 'POST',
        body: { current_password: currentPassword, new_password: newPassword },
        signal,
      }),

    /** Every account this deployment knows about. Administrators only, server-side. */
    listUsers: (signal?: AbortSignal) => call('/users', platformUsersSchema, { signal }),

    /**
     * Register one account, optionally with a first password.
     *
     * A password given here is a handover credential: the server marks the account
     * `must_change_password`, so the person replaces it before doing anything else.
     */
    createUser: (
      input: {
        subject: string;
        display_name: string;
        roles: string[];
        password?: string;
      },
      signal?: AbortSignal,
    ) =>
      call('/users', platformUserSchema, {
        method: 'POST',
        body: {
          subject: input.subject,
          display_name: input.display_name,
          roles: input.roles,
          ...(input.password ? { password: input.password } : {}),
        },
        signal,
      }),

    /**
     * Change one account's name, roles, or whether it works.
     *
     * Only the fields supplied are sent, which is what the endpoint expects: an
     * administrator adjusting a role must not have to restate a display name and risk
     * clobbering a rename somebody else made.
     */
    updateUser: (
      userId: string,
      changes: { display_name?: string; roles?: string[]; disabled?: boolean },
      signal?: AbortSignal,
    ) =>
      call(`/users/${encodeURIComponent(userId)}`, platformUserSchema, {
        method: 'PATCH',
        body: changes,
        signal,
      }),

    /** Hand one account a new password, ending every session it had. Returns no password. */
    setUserPassword: (userId: string, password: string, signal?: AbortSignal) =>
      call(`/users/${encodeURIComponent(userId)}/password`, platformUserSchema, {
        method: 'POST',
        body: { password },
        signal,
      }),

    /** One account's tokens, with nothing in the response that could authenticate. */
    listUserTokens: (userId: string, signal?: AbortSignal) =>
      call(`/users/${encodeURIComponent(userId)}/tokens`, platformTokensSchema, { signal }),

    /**
     * Mint an API token for one account, and return it the only time it will be readable.
     *
     * The value is shown once and never stored by this client. There is no endpoint that can
     * show it again, which is the property the dialog has to state plainly.
     */
    issueUserToken: (userId: string, label: string, signal?: AbortSignal) =>
      call(`/users/${encodeURIComponent(userId)}/tokens`, issuedTokenSchema, {
        method: 'POST',
        body: { label },
        signal,
      }),

    /** Stop one token authenticating anything, keeping the record that it existed. */
    revokeUserToken: (tokenId: string, signal?: AbortSignal) =>
      call(`/users/tokens/${encodeURIComponent(tokenId)}`, z.object({}).passthrough(), {
        method: 'DELETE',
        signal,
      }),

    /**
     * Repairs proposed for this feature. A repair is a change to somebody's repository that
     * the platform will not make on its own, so it is proposed and waits for a decision.
     */
    listRepairs: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/repairs`, repositoryRepairsSchema, {
        signal,
      }),

    /**
     * Approve one repair. The acknowledgement is not a formality the client can skip: the
     * server requires it, because this endpoint is reachable without the button that asks.
     * Approval applies the repair and runs a real attempt, so it takes the long timeout.
     */
    approveRepair: (
      featureId: string,
      repairId: string,
      options: { credentials?: ProviderCredentials } = {},
    ) =>
      durableAction(`repair-approve:${featureId}:${repairId}`, (idempotencyKey) =>
        call(
          `/features/${encodeURIComponent(featureId)}/repairs/${encodeURIComponent(repairId)}/approve`,
          featureSchema,
          {
            method: 'POST',
            body: { acknowledge_repository_change: true },
            credentials: options.credentials,
            idempotencyKey,
            timeoutMs: 300_000,
          },
        ),
      ),

    rejectRepair: (featureId: string, repairId: string, reason: string) =>
      durableAction(`repair-reject:${featureId}:${repairId}`, (idempotencyKey) =>
        call(
          `/features/${encodeURIComponent(featureId)}/repairs/${encodeURIComponent(repairId)}/reject`,
          featureSchema,
          { method: 'POST', body: { reason }, idempotencyKey },
        ),
      ),

    /**
     * The durable actions people have asked this feature to perform.
     *
     * This is what makes a confirmation survive a reload: the page follows the action rather
     * than the request that started it.
     */
    listActions: (featureId: string, signal?: AbortSignal) =>
      call(`/features/${encodeURIComponent(featureId)}/actions`, featureActionsSchema, { signal }),

    getAction: (featureId: string, actionId: string, signal?: AbortSignal) =>
      call(
        `/features/${encodeURIComponent(featureId)}/actions/${encodeURIComponent(actionId)}`,
        featureActionSchema,
        { signal },
      ),

    reconcileAction: (
      featureId: string,
      actionId: string,
      outcome: 'succeeded' | 'failed',
      reason: string,
    ) =>
      call(
        `/features/${encodeURIComponent(featureId)}/actions/${encodeURIComponent(actionId)}/reconcile`,
        featureActionSchema,
        { method: 'POST', body: { outcome, reason } },
      ),

    /**
     * What this identity still has to configure before a feature can be created.
     *
     * One call rather than three, because the New Feature page needs the whole answer before
     * it decides what to render: the form, a credential prompt, or a repository prompt.
     */
    getSetupState: (signal?: AbortSignal) => call('/setup', setupStateSchema, { signal }),

    /** The repositories this identity has saved for reuse across features. */
    listSavedRepositories: (signal?: AbortSignal) =>
      call('/repositories', savedRepositoriesSchema, { signal }),

    /**
     * The repositories this identity's stored GitHub token can reach.
     *
     * What the repository form offers instead of asking for a URL. Answered live on every
     * call, deliberately: a token's reach changes when somebody is granted a repository, and
     * that is exactly the moment they come here to add it.
     */
    listGitHubRepositories: (signal?: AbortSignal) =>
      call('/credentials/github/repositories', githubRepositoriesSchema, { signal }),

    /**
     * Save one repository. Only three fields are sent: the name is derived from the URL by
     * the server and the stable identifier is the server's, so neither is asked for.
     */
    saveRepository: (input: {
      repository_url: string;
      default_branch: string;
      repository_type: string;
    }) => call('/repositories', savedRepositorySchema, { method: 'POST', body: input }),

    updateSavedRepository: (
      configurationId: string,
      input: { repository_url: string; default_branch: string; repository_type: string },
    ) =>
      call(`/repositories/${encodeURIComponent(configurationId)}`, savedRepositorySchema, {
        method: 'PUT',
        body: input,
      }),

    /** Forget one saved repository. Features already created from it are untouched. */
    deleteSavedRepository: (configurationId: string) =>
      // A 204 has no body; the client normalises that to `{}`, so the schema has to accept it.
      client.request(`/repositories/${encodeURIComponent(configurationId)}`, z.unknown(), {
        method: 'DELETE',
      }),

    /** Provider credentials this identity has configured. Never the credentials themselves. */
    listCredentials: (signal?: AbortSignal) => call('/credentials', credentialsSchema, { signal }),

    storeCredential: (provider: string, secret: string) =>
      client.request(`/credentials/${encodeURIComponent(provider)}`, credentialSchema, {
        method: 'PUT',
        body: { secret },
      }),

    removeCredential: (provider: string) =>
      client.request(`/credentials/${encodeURIComponent(provider)}`, credentialSchema, {
        method: 'DELETE',
      }),

    /**
     * Whether one stored credential can still be opened, and — when asked — still works.
     *
     * `verify` is opt-in for the reason the endpoint's own docstring gives: it sends
     * somebody's credential to the provider on a button press. It is a separate question from
     * readability and never turns `usable` false, so the caller decides which one it wants.
     */
    checkCredential: (provider: string, options: { verify?: boolean } = {}) =>
      call(`/credentials/${encodeURIComponent(provider)}/check`, credentialCheckSchema, {
        method: 'POST',
        query: options.verify ? { verify: true } : undefined,
      }),

    /**
     * Where feature threads go, and whether the bot token is stored — by hint, never by
     * value. A deployment without the directory answers 503, which the panel reads as a fact
     * about the deployment rather than a failure.
     */
    getSlackConfiguration: (signal?: AbortSignal) =>
      call('/slack-configuration', slackConfigurationSchema, { signal }),

    saveSlackConfiguration: (input: SaveSlackConfigurationInput) =>
      call('/slack-configuration', slackConfigurationSchema, { method: 'PUT', body: input }),

    /**
     * Ask Slack whose token this is and whether the channel is reachable. Operator-initiated,
     * so it may fail loudly — a check is not a notification — and it answers in the same
     * verdict shape a credential check does.
     */
    checkSlackConfiguration: () =>
      call('/slack-configuration/check', credentialCheckSchema, { method: 'POST' }),

    /**
     * Where this deployment's designs come from, and whether a Figma token is stored — by
     * hint, never by value. A deployment without the directory answers 503, which the panel
     * reads as a fact about the deployment rather than a failure.
     */
    getDesignSource: (signal?: AbortSignal) =>
      call('/design-source', designSourceSchema, { signal }),

    saveDesignSource: (input: SaveDesignSourceInput) =>
      call('/design-source', designSourceSchema, { method: 'PUT', body: input }),

    /**
     * Ask Figma whether the configured owner's token is still accepted. Operator-initiated, and
     * it answers in the same verdict shape a credential check does. A refusal degrades the
     * source on the server, so the caller re-reads the configuration afterwards.
     */
    checkDesignSource: () =>
      call('/design-source/check', credentialCheckSchema, { method: 'POST' }),

    /**
     * One cited frame, rendered on demand at the file version the snapshot recorded.
     *
     * Read as bytes through this platform rather than as a URL the browser fetches, because
     * the console authenticates with a bearer token and an `<img src>` cannot carry one. No
     * render URL is stored anywhere: Figma's expire, and the snapshot they belong to is frozen.
     */
    getDesignPreview: (featureId: string, nodeId: string, signal?: AbortSignal) =>
      client.requestBlob(
        `/features/${encodeURIComponent(featureId)}/design-preview?node_id=${encodeURIComponent(nodeId)}`,
        { signal },
      ),

    /** This identity's own Slack link and opt-in scope. Nobody reads anybody else's. */
    getSlackLink: (signal?: AbortSignal) => call('/account/slack-link', slackLinkSchema, { signal }),

    saveSlackLink: (input: SaveSlackLinkInput) =>
      call('/account/slack-link', slackLinkSchema, { method: 'PUT', body: input }),

    /**
     * The resolved model-roles table this deployment runs on, plus the caller's own setups.
     *
     * The deployment rows stay read-only per row; which model each of their roles resolves to
     * is deployment configuration. The write calls below act on `/model-setups` — a custom
     * setup — never on this table.
     */
    getModelConfiguration: (signal?: AbortSignal) =>
      call('/model-configuration', modelConfigurationSchema, { signal }),

    /** This identity's authored setups, raw as entered, each with whether it can run now. */
    listModelSetups: (signal?: AbortSignal) =>
      call('/model-setups', modelSetupsSchema, { signal }),

    /** The caps the server enforces, so the form states the same numbers it will be held to. */
    getAttachmentLimits: (signal?: AbortSignal) =>
      call('/attachments/limits', attachmentLimitsSchema, { signal }),

    /**
     * Upload one image. Multipart, through the client's `FormData` branch: the JSON funnel
     * would set a content type the browser has to set itself.
     */
    uploadAttachment: (file: File, signal?: AbortSignal) => {
      const body = new FormData();
      body.append('file', file, file.name);
      // A generous deadline: this is bytes over somebody's uplink, not a JSON round trip.
      return client.upload('/attachments', attachmentSchema, body, { signal, timeoutMs: 120_000 });
    },

    /**
     * Read one attachment's bytes, authenticated.
     *
     * Returned as a `Blob` for the caller to turn into an object URL, because every read is
     * authenticated: there is no URL an `<img src>` could follow, which is the point.
     */
    getAttachmentBlob: (attachmentId: string, signal?: AbortSignal) =>
      client.fetchBlob(`/attachments/${encodeURIComponent(attachmentId)}`, { signal }),

    /** Forget one unbound upload. A bound one is refused; retiring the feature removes it. */
    deleteAttachment: (attachmentId: string) =>
      call(`/attachments/${encodeURIComponent(attachmentId)}`, z.unknown(), {
        method: 'DELETE',
      }),

    /**
     * Author one setup. A 422 carries the server's own refusal sentence — the 181 footgun
     * refused at save — and the caller renders it verbatim.
     */
    createModelSetup: (input: SaveModelSetupInput) =>
      call('/model-setups', savedModelSetupSchema, { method: 'POST', body: input }),

    /** Replace one setup. Features already pinned to it keep their snapshots. */
    updateModelSetup: (setupId: string, input: SaveModelSetupInput) =>
      call(`/model-setups/${encodeURIComponent(setupId)}`, savedModelSetupSchema, {
        method: 'PUT',
        body: input,
      }),

    /** Forget one setup. Features already created from it keep running on their snapshots. */
    deleteModelSetup: (setupId: string) =>
      // A 204 has no body; the client normalises that to `{}`, so the schema has to accept it.
      call(`/model-setups/${encodeURIComponent(setupId)}`, z.unknown(), { method: 'DELETE' }),

    /**
     * Ask the assistant one question and read the answer as it is written.
     *
     * The provider key and the platform token both travel as headers, which is the whole
     * reason this reads the stream with `fetch` rather than using `EventSource`.
     */
    streamChatMessage: (
      featureId: string,
      message: string,
      options: { credentials?: ProviderCredentials; signal?: AbortSignal } = {},
    ): AsyncGenerator<StreamEvent, void, void> =>
      readEventStream(streamUrl(client, `/features/${encodeURIComponent(featureId)}/chat/stream`), {
        method: 'POST',
        body: { message },
        headers: streamHeaders(options.credentials),
        signal: options.signal,
      }),

    /**
     * Lifecycle events as they are recorded, resuming from a cursor.
     *
     * `after` is the last event this client has already applied. Passing it is what makes a
     * reconnect pick up rather than replay, and what makes a duplicate harmless.
     */
    streamFeatureEvents: (
      featureId: string,
      after: number | null,
      options: { signal?: AbortSignal } = {},
    ): AsyncGenerator<StreamEvent, void, void> => {
      const path = `/features/${encodeURIComponent(featureId)}/events/stream`;
      const query = after === null ? '' : `?after=${after}`;
      return readEventStream(streamUrl(client, `${path}${query}`), {
        headers: streamHeaders(undefined),
        signal: options.signal,
      });
    },
  };
}

/**
 * Build an absolute URL for a stream.
 *
 * `ApiClient.request` owns URL construction for ordinary calls, but a stream is read with a
 * bare `fetch` -- it has to be, to hold the response open -- so the base URL is applied here.
 */
function streamUrl(client: ApiClient, path: string): string {
  return client.absoluteUrl(path);
}

/**
 * The headers a streamed request carries.
 *
 * Deliberately the same ones every other call sends. A stream that authenticated differently
 * from the rest of the client would be a second way in, and the point of reading streams with
 * `fetch` was to avoid needing one.
 */
function streamHeaders(credentials: ProviderCredentials | undefined): Record<string, string> {
  const headers: Record<string, string> = {};
  const token = readToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (credentials?.openaiApiKey) headers['X-OpenAI-Api-Key'] = credentials.openaiApiKey;
  if (credentials?.anthropicApiKey) headers['X-Anthropic-Api-Key'] = credentials.anthropicApiKey;
  if (credentials?.githubToken) headers['X-GitHub-Token'] = credentials.githubToken;
  return headers;
}

export type FeatureApi = ReturnType<typeof createFeatureApi>;
