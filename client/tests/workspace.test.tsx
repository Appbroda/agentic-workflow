import { describe, expect, it, vi } from 'vitest';
import { render, renderHook, screen, waitFor, within } from '@testing-library/react';
import type { ReactNode } from 'react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import { ApiError } from '@/api/errors';
import { useWorkstreams } from '@/features/feature-workspace/hooks';
import { artifactsSchema, featureSchema } from '@/schemas/feature';
import { stubApi } from './fixtures';
import live185 from './fixtures/integration-reviews.bulk-apps-live-185.json';
import pinnedFeature from './fixtures/feature.pinned-model-setup.json';

/** Run 185's first integration review, as the API returned it: `changes_requested`. */
const LOOPED_BACK_REVIEW = artifactsSchema.parse(live185).artifacts[0]!;

const FEATURE = {
  feature_id: 'adunit-deactivate-live-086',
  workflow_id: 'workflow-086',
  status: 'completed',
  title: 'Deactivate ad units from the console',
  current_agent: 'feature_workflow',
  repository_count: 2,
  required_repository_count: 2,
  repositories: [
    {
      repository_id: 'admanager-server',
      name: 'Ad Manager Server',
      role: 'backend',
      repository_url: 'https://github.com/cryn3t/admanager_console-2.0.git',
      default_branch: 'master',
      required: true,
      implementation_order: null,
    },
  ],
  clarification_rounds: 0,
  integration_review_cycles: 0,
  merge_strategy: null,
  deployment_strategy: null,
  execution_mode: 'live',
  cancellation_status: 'not_requested',
  cancellation_requested_at: null,
  cancellation_reason: null,
  cleanup_requirements: [],
  available_actions: [],
  created_at: '2026-08-24T22:40:00Z',
  updated_at: '2026-08-24T23:20:00Z',
};

function workstream(overrides: Record<string, unknown> = {}) {
  return {
    repository_id: 'admanager-server',
    repository_name: 'Ad Manager Server',
    repository_role: 'backend',
    repository_url: 'https://github.com/cryn3t/admanager_console-2.0.git',
    repository_required: true,
    child_workflow_id: 'adunit-deactivate-live-086:admanager-server',
    workstream_id: 'admanager-server',
    status: 'completed',
    branch_name: 'ai/adunit/admanager-server/deactivate',
    workspace_path: '/workspaces/adunit/admanager-server',
    retry_count: 5,
    code_completion_artifact_id: '006_code_completion.json',
    review_artifact_id: '007_review.json',
    blocking_issues: [],
    pull_request_artifact_id: '008_pull_request.admanager-server.json',
    current_validation_results: [],
    production_files_changed: ['server/services/adunit/status.service.js'],
    test_files_changed: [],
    configuration_files_changed: [],
    requirements_implemented: ['REQ-1'],
    requirements_not_implemented: [],
    implementation_retry_count: 0,
    validation_retry_count: 5,
    repository_setup_retry_count: 0,
    integration_retry_count: 0,
    granted_extra_attempts: 0,
    retry_grants: [],
    implementation_expectations: [],
    configured_validation_commands: [],
    blocking_setup_issues: [],
    scoped_requirements: [],
    out_of_scope_requirements: [],
    planned_blind: false,
    ...overrides,
  };
}

function renderWorkspace(api: Partial<FeatureApi>, path = '/features/adunit-deactivate-live-086') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[path]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId" element={<FeatureWorkspacePage />} />
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

const BASE: Partial<FeatureApi> = {
  listEvents: async () => ({ feature_id: FEATURE.feature_id, events: [], last_event_id: null }),
  getFeature: async () => FEATURE,
  getWorkstreams: async () => ({ feature_id: FEATURE.feature_id, workstreams: [workstream()] }),
  getTimeline: async () => ({ feature_id: FEATURE.feature_id, events: [] }),
  getClarification: async () => ({
    feature_id: FEATURE.feature_id,
    awaiting_answers: false,
    technical_prd_artifact_id: '002_technical_prd.json',
    clarification_rounds: 0,
    max_clarification_rounds: 10,
    questions: [],
    previous_answers: {},
    design_conflicts: [],
  }),
};

/**
 * A feature stopped on one repository-informed question.
 *
 * The suggestion is the one the platform would actually produce: what the checkout does,
 * quoted, with the file it was read from. A hand-written "Yes." would test the prefill
 * without testing that the provenance travels with it.
 */
const WAITING_WITH_SUGGESTION: Partial<FeatureApi> = {
  getFeature: async () => ({
    ...FEATURE,
    status: 'waiting_for_human',
    available_actions: ['ANSWER_CLARIFICATION', 'CANCEL_WORKFLOW'],
  }),
  getClarification: async () => ({
    feature_id: FEATURE.feature_id,
    awaiting_answers: true,
    technical_prd_artifact_id: '002_technical_prd.revision-2.json',
    clarification_rounds: 0,
    max_clarification_rounds: 10,
    questions: [
      {
        question_id: 'recon-admanager-server-3',
        question: 'Rely on the global middleware, or introduce per-route auth?',
        rationale:
          'In admanager-server, the requirements assume per-route authentication. The checkout applies it once, globally. Evidence: server/config/express.js',
        required: true,
        suggested_answer:
          'Follow what admanager-server already does: authentication is applied once, globally.',
        suggestion_source:
          'Suggested from repository analysis of admanager-server (server/config/express.js)',
        suggestion_confidence: 'high',
      },
    ],
    previous_answers: {},
    design_conflicts: [],
  }),
};

describe('FeatureWorkspace', () => {
  it('shows the feature with the server’s own status explanation', async () => {
    renderWorkspace(BASE);

    expect(await screen.findByRole('heading', { name: FEATURE.title })).toBeInTheDocument();
    // The status headline appears as a badge beside the title and again as the phase; both
    // are the server's wording, never this client's.
    expect(screen.getAllByText('Finished').length).toBeGreaterThan(0);
    expect(await screen.findByText('All work is done.')).toBeInTheDocument();
  });

  it('writes the platform a feature ran on with the shared provider spelling', async () => {
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({ ...FEATURE, agent_platform: 'anthropic' }),
    });

    // Brand casing, from the same helper the execution records use, so one feature's provider
    // is spelled the same everywhere it appears rather than sentence-cased into "Anthropic"
    // in one place and "anthropic" in another.
    expect(await screen.findByText('Agent platform')).toBeInTheDocument();
    expect(screen.getByText('Anthropic')).toBeInTheDocument();
  });

  it('says nothing about the platform for a feature written before the choice existed', async () => {
    renderWorkspace({ ...BASE, getFeature: async () => FEATURE });

    await screen.findByRole('heading', { name: FEATURE.title });
    // A field the server did not publish is absent rather than invented.
    expect(screen.queryByText('Agent platform')).not.toBeInTheDocument();
  });

  it('writes the performance tier a feature was submitted at, beside its platform', async () => {
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({
        ...FEATURE,
        agent_platform: 'anthropic',
        performance_tier: 'medium',
      }),
    });

    // The name the submission form offered, not the wire value: somebody reading a feature
    // back should see the choice they made spelled the way they made it.
    expect(await screen.findByText('Performance tier')).toBeInTheDocument();
    expect(screen.getByText('Standard')).toBeInTheDocument();
    // And once in the header, as the one place the tier is a chip: the graph's edge labels
    // write reasoning effort as "effort: high" so it is never read as this.
    expect(screen.getByText('Standard tier')).toBeInTheDocument();
  });

  it('says nothing about the tier for a feature written before tiers existed', async () => {
    renderWorkspace({ ...BASE, getFeature: async () => FEATURE });

    await screen.findByRole('heading', { name: FEATURE.title });
    expect(screen.queryByText('Performance tier')).not.toBeInTheDocument();
    expect(screen.queryByText(/tier$/)).not.toBeInTheDocument();
  });

  it('shows and reconciles a durable action whose crash outcome needs checking', async () => {
    const reconcileAction = vi.fn().mockResolvedValue({});
    renderWorkspace({
      ...BASE,
      reconcileAction,
      listActions: async () => ({
        feature_id: FEATURE.feature_id,
        actions: [
          {
            action_id: 'action-uncertain',
            feature_id: FEATURE.feature_id,
            repository_id: 'admanager-server',
            action_type: 'APPROVE_REPOSITORY_REPAIR',
            actor_id: 'user-alex',
            actor_display_name: 'Alex',
            origin: 'rest',
            origin_message_id: null,
            status: 'requires_reconciliation',
            attempt: 1,
            max_attempts: 1,
            created_at: '2026-08-24T23:00:00Z',
            started_at: '2026-08-24T23:00:01Z',
            completed_at: '2026-08-24T23:04:00Z',
            in_progress: false,
            result_summary: null,
            error_code: 'unconfirmed_external_effect',
            error_message: 'The provider result could not be confirmed.',
            external_operation_ids: ['operation-1'],
          },
        ],
      }),
    });
    const user = userEvent.setup();

    expect(await screen.findByText('APPROVE_REPOSITORY_REPAIR')).toBeInTheDocument();
    await user.type(
      screen.getByLabelText('Evidence checked and conclusion'),
      'The branch and workflow snapshot show no applied repair.',
    );
    await user.click(screen.getByRole('button', { name: 'Mark verified not completed' }));

    await waitFor(() =>
      expect(reconcileAction).toHaveBeenCalledWith(
        FEATURE.feature_id,
        'action-uncertain',
        'failed',
        'The branch and workflow snapshot show no applied repair.',
      ),
    );
  });

  it('renders repositories from backend data, naming them rather than showing only an id', async () => {
    renderWorkspace(BASE);

    const repositories = await screen.findByRole('table', { name: 'Repository workstreams' });
    expect(within(repositories).getByText('Ad Manager Server')).toBeInTheDocument();
    expect(within(repositories).getByText('backend')).toBeInTheDocument();
  });

  it('renders any number of repositories with repeated roles', async () => {
    renderWorkspace({
      ...BASE,
      getWorkstreams: async () => ({
        feature_id: FEATURE.feature_id,
        workstreams: [
          workstream(),
          workstream({ repository_id: 'worker-a', repository_name: 'Worker A', status: 'running' }),
          workstream({ repository_id: 'worker-b', repository_name: 'Worker B', status: 'failed' }),
          workstream({ repository_id: 'sdk', repository_name: 'SDK', repository_role: 'shared' }),
        ],
      }),
    });

    const repositories = await screen.findByRole('table', { name: 'Repository workstreams' });
    // Four repositories, three sharing a role: identity is the id, never the role.
    expect(within(repositories).getAllByRole('row')).toHaveLength(5);
    expect(within(repositories).getByText('Worker A')).toBeInTheDocument();
    expect(within(repositories).getByText('SDK')).toBeInTheDocument();
  });

  it('surfaces the operator question a stopped workstream ended on', async () => {
    const question =
      '8 attempts were spent on this repository without reaching an approved review. Read the blocking issues above: should the requirement change, should the repository be repaired, or is this work simply larger than the attempt budget it was given?';
    renderWorkspace({
      ...BASE,
      getWorkstreams: async () => ({
        feature_id: FEATURE.feature_id,
        workstreams: [
          workstream({
            status: 'failed',
            blocking_issues: ['npm run test found no tests; this is not a test failure.', question],
          }),
        ],
      }),
    });

    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Show detail' }));

    // The actionable part of a failure, shown in full rather than truncated.
    expect(await screen.findByText(question)).toBeInTheDocument();
  });

  it('renders the backend failure summary and cancellation cleanup evidence', async () => {
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({
        ...FEATURE,
        status: 'failed_requires_human',
        failure_summary: {
          stage: 'feature_planner',
          agent: 'planner',
          repository_id: 'admanager-server',
          root_classification: 'provider_timeout',
          attempt: 2,
          command: ['npm', 'test'],
          exit_code: 1,
          diagnostics: ['The provider did not answer before the deadline.'],
          retryable: true,
          next_action: 'Resume from the last safe checkpoint.',
          recorded_at: '2026-08-25T09:00:00Z',
        },
        cleanup_requirements: [
          {
            resource_type: 'pull_request',
            repository_id: 'admanager-server',
            external_reference: 'https://github.com/example/repo/pull/7',
            reason: 'Cancellation retained an open pull request.',
            recommended_action: 'Review it before cleanup.',
          },
        ],
      }),
    });

    expect(await screen.findByText('Why the feature stopped')).toBeInTheDocument();
    expect(screen.getByText('provider_timeout')).toBeInTheDocument();
    expect(screen.getByText('Resume from the last safe checkpoint.')).toBeInTheDocument();
    expect(screen.getByText('External resources need checking')).toBeInTheDocument();
    expect(screen.getByText('Review it before cleanup.')).toBeInTheDocument();
  });

  it('asks the clarification questions with the evidence attached', async () => {
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({
        ...FEATURE,
        status: 'waiting_for_human',
        available_actions: ['ANSWER_CLARIFICATION', 'CANCEL_WORKFLOW'],
      }),
      getClarification: async () => ({
        feature_id: FEATURE.feature_id,
        awaiting_answers: true,
        technical_prd_artifact_id: '002_technical_prd.revision-2.json',
        clarification_rounds: 0,
        max_clarification_rounds: 10,
        questions: [
          {
            question_id: 'recon-admanager-server-3',
            question: 'Rely on the global middleware, or introduce per-route auth?',
            rationale:
              'In admanager-server, the requirements assume per-route authentication. The checkout applies it once, globally. Evidence: server/config/express.js',
            required: true,
            suggested_answer:
              'Follow what admanager-server already does: authentication is applied once, globally.',
            suggestion_source:
              'Suggested from repository analysis of admanager-server (server/config/express.js)',
            suggestion_confidence: 'high',
          },
        ],
        previous_answers: {},
        design_conflicts: [],
      }),
    });

    expect(await screen.findByText('This feature is waiting on you')).toBeInTheDocument();
    expect(
      screen.getByLabelText('Rely on the global middleware, or introduce per-route auth?'),
    ).toBeInTheDocument();
    // The evidence is what lets a person check the answer rather than guess it.
    expect(screen.getByText(/Evidence: server\/config\/express\.js/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Resume' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel feature' })).toBeInTheDocument();
  });

  it('does not ask the same questions again while their accepted resume is running', async () => {
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({
        ...FEATURE,
        status: 'waiting_for_human',
        effective_status: 'resuming',
        available_actions: ['CANCEL_WORKFLOW', 'RETIRE_FEATURE'],
      }),
      getClarification: async () => ({
        feature_id: FEATURE.feature_id,
        awaiting_answers: false,
        technical_prd_artifact_id: '002_technical_prd.revision-2.json',
        clarification_rounds: 0,
        max_clarification_rounds: 10,
        // The durable checkpoint keeps the questions until the worker reaches its next safe
        // boundary. Their presence is evidence, not permission to submit them a second time.
        questions: [
          {
            question_id: 'recon-admanager-server-3',
            question: 'Rely on the global middleware, or introduce per-route auth?',
            rationale: 'The checkout applies authentication globally.',
            required: true,
            suggested_answer: 'Keep the global middleware.',
            suggestion_source: 'Repository analysis',
          },
        ],
        previous_answers: {},
        design_conflicts: [],
      }),
    });

    expect(
      await screen.findByText(
        'Your answers were accepted and the workflow is continuing in the background.',
      ),
    ).toBeInTheDocument();
    expect(screen.getAllByText('Resuming').length).toBeGreaterThan(0);
    expect(screen.queryByText('This feature is waiting on you')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Submit answers' })).not.toBeInTheDocument();
    expect(screen.queryByText('Action required')).not.toBeInTheDocument();
  });

  it('a queued feature counts the repositories it was submitted with', async () => {
    // Workstreams exist only once the planner has agreed the contract, so a queued feature
    // has none — and reading the count from them made a feature submitted against two
    // repositories announce "0 repositories" for as long as it sat in the queue.
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({ ...FEATURE, status: 'pending', reference: 'AB-Feature-97' }),
      getWorkstreams: async () => ({ feature_id: FEATURE.feature_id, workstreams: [] }),
      // Nothing produced yet, which is what makes it queued rather than analysing.
      listArtifacts: async () => ({ feature_id: FEATURE.feature_id, artifacts: [] }),
    });

    await screen.findByRole('heading', { name: FEATURE.title });
    // The header and the Repositories panel both count them, and they must agree.
    expect(screen.getAllByText('2 repositories').length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText('0 repositories')).not.toBeInTheDocument();
    // And the first stage says the platform has the work rather than that it is doing it.
    // The strip beside the title and the map on the Overview are the same stages, so both
    // have to say it. Neither may read "in progress" for work nothing has begun.
    const progress = screen.getAllByLabelText('Workflow progress');
    expect(progress).toHaveLength(2);
    for (const view of progress) expect(view).toHaveTextContent(/Technical PRD.*queued/);
    expect(progress[1]).not.toHaveTextContent('in progress');
  });

  it('says the integration review asked for changes, and holds the pull requests', async () => {
    // Run 185 at 09:22 (2026-09-01): the integration review had returned changes_requested,
    // one repository was three attempts into fixing it, and both surfaces said "Integration
    // review: done" over "Pull requests: in progress". The review is served here exactly as
    // the API serves it -- payload only when asked for, which is the reason the verdict was
    // invisible to these surfaces in the first place.
    const envelope = (artifact_type: string, artifact_id: string) => ({
      artifact_id,
      artifact_type,
      schema_version: '1.0',
      producer: 'feature_workflow',
      timestamp: '2026-09-01T09:20:59Z',
      metadata: {},
      validation_status: 'valid',
      payload: {},
    });
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({ ...FEATURE, status: 'running_child_workflows', integration_review_cycles: 1 }),
      getWorkstreams: async () => ({
        feature_id: FEATURE.feature_id,
        workstreams: [
          workstream({ repository_id: 'AB-console-admin-2.0', status: 'completed' }),
          workstream({ repository_id: 'admanager_console-2.0', status: 'running' }),
        ],
      }),
      listArtifacts: async (_featureId, params) => ({
        feature_id: FEATURE.feature_id,
        artifacts:
          params?.artifactType === 'integration_review'
            ? params.includePayload
              ? [LOOPED_BACK_REVIEW]
              : [envelope('integration_review', LOOPED_BACK_REVIEW.artifact_id)]
            : params?.artifactType
              ? []
              : [
                  envelope('technical_prd', '002_technical_prd.json'),
                  envelope('integration_contract', '009_integration_contract.json'),
                  envelope('repository_execution_plan', '010_repository_execution_plan.json'),
                  envelope('integration_review', LOOPED_BACK_REVIEW.artifact_id),
                ],
      }),
    });

    await screen.findByRole('heading', { name: FEATURE.title });
    // The strip beside the title and the map on the Overview are the same stages, and a
    // reader who sees one and not the other must not be told two different things.
    const progress = screen.getAllByLabelText('Workflow progress');
    expect(progress).toHaveLength(2);
    for (const view of progress) {
      expect(view).toHaveTextContent(/Integration review.*changes requested/);
    }
    // The map states each stage in words: the gate did not open, so the stage after it has
    // not started. It used to read "done" and "in progress" respectively.
    expect(within(progress[1]!).getByText('Pull requests').closest('li')).toHaveTextContent(
      'not started',
    );
    expect(progress[1]).not.toHaveTextContent(/Integration review\s*done/);
  });

  it('prefills the answer the platform read in the repository, and says where from', async () => {
    const resumeFeature = vi.fn().mockResolvedValue(FEATURE);
    renderWorkspace({ ...BASE, resumeFeature, ...WAITING_WITH_SUGGESTION });
    const user = userEvent.setup();

    const field = await screen.findByLabelText(
      'Rely on the global middleware, or introduce per-route auth?',
    );
    // Prefilled: accepting the suggestion is doing nothing, which is the point.
    expect(field).toHaveValue(
      'Follow what admanager-server already does: authentication is applied once, globally.',
    );
    // And shown separately, with its provenance, so it can be checked rather than trusted.
    expect(
      screen.getByText(/Suggested from repository analysis of admanager-server/),
    ).toBeInTheDocument();
    expect(screen.getByText(/high confidence/)).toBeInTheDocument();
    // Nothing was submitted on the author's behalf.
    expect(resumeFeature).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Submit answers' }));

    await waitFor(() => expect(resumeFeature).toHaveBeenCalledTimes(1));
    expect(resumeFeature.mock.calls[0]![1]).toEqual([
      {
        question_id: 'recon-admanager-server-3',
        answer:
          'Follow what admanager-server already does: authentication is applied once, globally.',
      },
    ]);
  });

  it('lets a suggested answer be replaced, and submits what was typed', async () => {
    const resumeFeature = vi.fn().mockResolvedValue(FEATURE);
    renderWorkspace({ ...BASE, resumeFeature, ...WAITING_WITH_SUGGESTION });
    const user = userEvent.setup();

    const field = await screen.findByLabelText(
      'Rely on the global middleware, or introduce per-route auth?',
    );
    await user.clear(field);
    await user.type(field, 'No. Add per-route auth for this endpoint only.');
    await user.click(screen.getByRole('button', { name: 'Submit answers' }));

    await waitFor(() => expect(resumeFeature).toHaveBeenCalledTimes(1));
    expect(resumeFeature.mock.calls[0]![1]).toEqual([
      {
        question_id: 'recon-admanager-server-3',
        answer: 'No. Add per-route auth for this endpoint only.',
      },
    ]);
  });

  it('offers the suggestion back once it has been edited away', async () => {
    renderWorkspace({ ...BASE, resumeFeature: vi.fn(), ...WAITING_WITH_SUGGESTION });
    const user = userEvent.setup();

    const field = await screen.findByLabelText(
      'Rely on the global middleware, or introduce per-route auth?',
    );
    // Not offered before an edit: the fields already hold the suggestions, so it would do
    // nothing.
    expect(
      screen.queryByRole('button', { name: 'Restore suggested answers' }),
    ).not.toBeInTheDocument();

    await user.clear(field);
    await user.type(field, 'Something else');
    await user.click(screen.getByRole('button', { name: 'Restore suggested answers' }));

    expect(field).toHaveValue(
      'Follow what admanager-server already does: authentication is applied once, globally.',
    );
  });

  it('requires every open question to be answered before submitting', async () => {
    const resumeFeature = vi.fn().mockResolvedValue(FEATURE);
    renderWorkspace({
      ...BASE,
      resumeFeature,
      getFeature: async () => ({
        ...FEATURE,
        status: 'waiting_for_human',
        available_actions: ['ANSWER_CLARIFICATION', 'CANCEL_WORKFLOW'],
      }),
      getClarification: async () => ({
        feature_id: FEATURE.feature_id,
        awaiting_answers: true,
        technical_prd_artifact_id: '002_technical_prd.json',
        clarification_rounds: 0,
        max_clarification_rounds: 10,
        questions: [
          {
            question_id: 'q1',
            question: 'First question?',
            rationale: 'because',
            required: true,
            suggested_answer: '',
            suggestion_source: '',
          },
          {
            question_id: 'q2',
            question: 'Second question?',
            rationale: 'because',
            required: true,
            suggested_answer: '',
            suggestion_source: '',
          },
        ],
        previous_answers: {},
        design_conflicts: [],
      }),
    });
    const user = userEvent.setup();

    const submit = await screen.findByRole('button', { name: 'Submit answers' });
    // The server rejects a partial answer set, so the client does not let one be sent.
    expect(submit).toBeDisabled();

    await user.type(screen.getByLabelText('First question?'), 'Use the global middleware.');
    expect(submit).toBeDisabled();
    await user.type(screen.getByLabelText('Second question?'), 'A new page.');
    expect(submit).toBeEnabled();

    await user.click(submit);
    const [, answers] = resumeFeature.mock.calls[0] as [string, { question_id: string }[]];
    expect(answers.map((item) => item.question_id)).toEqual(['q1', 'q2']);
  });

  it('confirms before cancelling and explains a refusal from the platform', async () => {
    const cancelFeature = vi
      .fn()
      .mockRejectedValue(new ApiError('conflict', 'no', 409, 'a cancellation is already in flight'));
    // A feature still running: cancelling it is a legal thing to ask, and the platform is
    // still the one that decides whether this particular request may proceed.
    renderWorkspace({
      ...BASE,
      cancelFeature,
      getFeature: async () => ({
        ...FEATURE,
        status: 'running_child_workflows',
        available_actions: ['RESUME_WORKFLOW', 'CANCEL_WORKFLOW'],
      }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Cancel feature' }));
    // Significant actions confirm first.
    expect(screen.getByRole('dialog', { name: 'Cancel this feature?' })).toBeInTheDocument();
    expect(cancelFeature).not.toHaveBeenCalled();

    await user.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel feature' }));

    // The platform decides legality; the client reports what it said.
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The platform refused this action in the feature’s current state.',
    );
  });

  it('asks for no credentials when resuming a live feature', async () => {
    // A resume is queued and the worker resolves the feature owner's stored keys; the server
    // discards any header the request carried. Asking for keys here collected credentials
    // nothing would ever send, so the dialog says where the keys come from instead.
    const resumeFeature = vi.fn().mockResolvedValue({ ...FEATURE, status: 'running_child_workflows' });
    renderWorkspace({
      ...BASE,
      resumeFeature,
      getFeature: async () => ({
        ...FEATURE,
        status: 'failed',
        available_actions: ['RESUME_WORKFLOW', 'CANCEL_WORKFLOW'],
      }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Resume' }));
    const dialog = screen.getByRole('dialog', { name: 'Resume this feature?' });
    expect(within(dialog).queryByLabelText('OpenAI API key')).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText('GitHub token')).not.toBeInTheDocument();
    expect(dialog).toHaveTextContent('Runs with the credentials stored in Settings.');
    await user.click(within(dialog).getByRole('button', { name: 'Resume' }));

    expect(resumeFeature).toHaveBeenCalledWith(FEATURE.feature_id, []);
  });

  it('moves focus into the confirmation and closes it on Escape', async () => {
    renderWorkspace({
      ...BASE,
      getFeature: async () => ({
        ...FEATURE,
        status: 'running_child_workflows',
        available_actions: ['RESUME_WORKFLOW', 'CANCEL_WORKFLOW'],
      }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Cancel feature' }));
    const dialog = screen.getByRole('dialog', { name: 'Cancel this feature?' });

    // Without this, somebody navigating by keyboard tabs from the button they pressed straight
    // past the confirmation into the rest of the page -- which for "cancel this feature" is
    // exactly the wrong thing to do by accident.
    expect(dialog.contains(document.activeElement)).toBe(true);

    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('does not offer an action the platform can only refuse', async () => {
    // A completed or cancelled feature cannot be resumed or cancelled again. Offering the
    // buttons would be controls whose only outcome is a 409 -- the same reason the retry
    // control is hidden on a repository with nothing to retry. Feature -076, cancelled in
    // August, was still showing both.
    renderWorkspace({ ...BASE, getFeature: async () => ({ ...FEATURE, status: 'cancelled' }) });

    await screen.findByRole('table', { name: 'Repository workstreams' });
    expect(screen.queryByRole('button', { name: 'Cancel feature' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Resume' })).not.toBeInTheDocument();
  });

  it('lists artifact envelopes and loads a payload only when one is opened', async () => {
    const getArtifact = vi.fn().mockResolvedValue({
      artifact_id: '009_integration_contract.json',
      artifact_type: 'integration_contract',
      schema_version: '1.0',
      producer: 'feature_planner',
      timestamp: '2026-08-24T22:45:00Z',
      metadata: {},
      validation_status: 'valid',
      payload: { contract_version: '1.0.0', api_style: 'rest', owning_workstreams: ['admanager-server'] },
    });
    renderWorkspace(
      {
        ...BASE,
        getArtifact,
        listArtifacts: async () => ({
          feature_id: FEATURE.feature_id,
          artifacts: [
            {
              artifact_id: '009_integration_contract.json',
              artifact_type: 'integration_contract',
              schema_version: '1.0',
              producer: 'feature_planner',
              timestamp: '2026-08-24T22:45:00Z',
              metadata: {},
              validation_status: 'valid',
              payload: {},
            },
          ],
        }),
      },
      '/features/adunit-deactivate-live-086/artifacts',
    );
    const user = userEvent.setup();

    const list = await screen.findByRole('list', { name: 'Artifacts' });
    // Envelopes only: nothing has been fetched in full yet.
    expect(getArtifact).not.toHaveBeenCalled();

    await user.click(within(list).getByRole('button'));

    expect(await screen.findByText('1.0.0', { exact: false })).toBeInTheDocument();
    expect(getArtifact).toHaveBeenCalledWith(
      'adunit-deactivate-live-086',
      '009_integration_contract.json',
      expect.anything(),
    );
  });

  it('falls back to a raw view for an artifact type it has no renderer for', async () => {
    renderWorkspace(
      {
        ...BASE,
        listArtifacts: async () => ({
          feature_id: FEATURE.feature_id,
          artifacts: [
            {
              artifact_id: '099_something_new.json',
              artifact_type: 'something_new',
              schema_version: '1.0',
              producer: 'future_agent',
              timestamp: '2026-08-24T22:45:00Z',
              metadata: {},
              validation_status: 'valid',
              payload: {},
            },
          ],
        }),
        getArtifact: async () => ({
          artifact_id: '099_something_new.json',
          artifact_type: 'something_new',
          schema_version: '1.0',
          producer: 'future_agent',
          timestamp: '2026-08-24T22:45:00Z',
          metadata: {},
          validation_status: 'valid',
          payload: { interesting: 'value' },
        }),
      },
      '/features/adunit-deactivate-live-086/artifacts',
    );
    const user = userEvent.setup();

    await user.click(within(await screen.findByRole('list', { name: 'Artifacts' })).getByRole('button'));

    // The platform grows artifact types; an unknown one must still be readable.
    expect(await screen.findByText(/No formatted view for this artifact type yet/)).toBeInTheDocument();
    expect(screen.getByText(/"interesting": "value"/)).toBeInTheDocument();
  });

  it('shows pull requests with their branches and an external link', async () => {
    renderWorkspace(
      {
        ...BASE,
        getPullRequests: async () => ({
          feature_id: FEATURE.feature_id,
          pull_requests: [
            {
              artifact_id: '008_pull_request.admanager-server.json',
              artifact_type: 'pull_request',
              schema_version: '1.0',
              producer: 'github',
              timestamp: '2026-08-24T23:10:00Z',
              metadata: {},
              validation_status: 'valid',
              payload: {
                repository: 'cryn3t/admanager_console-2.0',
                url: 'https://github.com/cryn3t/admanager_console-2.0/pull/16',
                title: 'Deactivate ad units',
                source_branch: 'ai/adunit/admanager-server/deactivate',
                target_branch: 'master',
                state: 'open',
              },
            },
          ],
        }),
      },
      '/features/adunit-deactivate-live-086/pull-requests',
    );

    const table = await screen.findByRole('table', { name: 'Pull requests' });
    expect(within(table).getByText('cryn3t/admanager_console-2.0')).toBeInTheDocument();
    expect(within(table).getByText('ai/adunit/admanager-server/deactivate')).toBeInTheDocument();
    const link = within(table).getByRole('link', { name: 'Open' });
    expect(link).toHaveAttribute('href', 'https://github.com/cryn3t/admanager_console-2.0/pull/16');
    // External links must not hand the opener a window reference.
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('reports a failure to load the feature instead of rendering an empty screen', async () => {
    renderWorkspace({
      ...BASE,
      getFeature: async () => {
        throw new ApiError('not_found', 'missing', 404, 'feature not found');
      },
    });

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The platform has nothing by that identifier.',
    );
  });
});

describe('live updates', () => {
  it('re-reads a query when a newer event id arrives', async () => {
    let reads = 0;
    const api = stubApi({
      ...BASE,
      getWorkstreams: async () => {
        reads += 1;
        return { feature_id: FEATURE.feature_id, workstreams: [workstream()] };
      },
    });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <AppProviders api={api} queryClient={queryClient}>
        {children}
      </AppProviders>
    );

    const view = renderHook(({ at }: { at: number | null }) => useWorkstreams('f', at), {
      wrapper,
      initialProps: { at: null as number | null },
    });
    await waitFor(() => expect(reads).toBe(1));

    // A newer event id is a different question -- "workstreams as of event 7" -- so the data
    // is read again rather than served from the answer to the previous one.
    view.rerender({ at: 7 });
    await waitFor(() => expect(reads).toBe(2));

    // The same id asks the same question and must not cost another request.
    view.rerender({ at: 7 });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(reads).toBe(2);
  });

  it('keeps the previous data on screen while a refresh is in flight', async () => {
    renderWorkspace(BASE);

    const table = await screen.findByRole('table', { name: 'Repository workstreams' });
    // A refresh must not blank the screen it is refreshing.
    expect(within(table).getByText('Ad Manager Server')).toBeInTheDocument();
  });

  it('resumes from its cursor so a reconnect does not replay the history', async () => {
    const seen: (number | null)[] = [];
    renderWorkspace({
      ...BASE,
      listEvents: async (_id: string, after: number | null) => {
        seen.push(after);
        return {
          feature_id: FEATURE.feature_id,
          events: [
            {
              id: 12,
              timestamp: '2026-08-24T23:00:00Z',
              event_type: 'lifecycle',
              source: 'api',
              event: 'pull_request_created',
              details: {},
            },
          ],
          last_event_id: 12,
        };
      },
    });

    await screen.findByRole('table', { name: 'Repository workstreams' });
    // The first read starts from nothing; the cursor is what a later read continues from.
    expect(seen[0]).toBe(null);
  });
});

/**
 * A custom feature's pinned setup on the Overview, from a captured real `/features/{id}`
 * payload (refreshed by `server/tests/capture_model_setup_fixtures.py`). The capture edited
 * the setup after submission, so the payload carries the edited-since state — G3 made
 * visible: the feature keeps running on its snapshot, and the page says so.
 */
describe('the pinned model setup on the overview', () => {
  const CUSTOM_FEATURE = featureSchema.parse(pinnedFeature);

  it('shows the snapshot in full and says the setup was edited since', async () => {
    renderWorkspace(
      {
        ...BASE,
        listEvents: async () => ({
          feature_id: CUSTOM_FEATURE.feature_id,
          events: [],
          last_event_id: null,
        }),
        getFeature: async () => CUSTOM_FEATURE,
        getWorkstreams: async () => ({
          feature_id: CUSTOM_FEATURE.feature_id,
          workstreams: [],
        }),
      },
      `/features/${CUSTOM_FEATURE.feature_id}`,
    );

    expect(await screen.findByText(/Mixed pilot setup — pinned at submission/)).toBeInTheDocument();
    expect(screen.getByText(/The setup has been edited since/)).toBeInTheDocument();
    // The four snapshotted roles, with each role's own platform — the mixed shape whole.
    const rows = screen.getByText(/Mixed pilot setup — pinned at submission/).closest('div')!;
    const pinned = (rows as HTMLElement).textContent ?? '';
    expect(pinned).toContain('OpenAI · GPT-5.6 Sol');
    expect(pinned).toContain('Anthropic · Claude Opus 5 · effort: max · max 64,000');
    expect(pinned).toContain('scoped_fix');
  });
});
