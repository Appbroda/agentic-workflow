import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Navigate, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { writeToken } from '@/api/token';
import { AppLayout } from '@/app/AppLayout';
import { DashboardPage } from '@/pages/DashboardPage';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import { RepositoryPage } from '@/pages/RepositoryPage';
import type { FeatureApi } from '@/api/features';
import { stubApi } from './fixtures';

/**
 * Getting around the control plane.
 *
 * These are the journeys the redesign is for: a product manager finding what is waiting on
 * them, and an engineer walking from a feature down to the exact command that failed. They
 * exercise the real shell -- sidebar, breadcrumbs, tabs -- rather than a page in isolation,
 * because the point of each is where it sits relative to the others.
 */

const FEATURE = {
  feature_id: 'f-1',
  workflow_id: 'w-1',
  status: 'failed_requires_human',
  title: 'Server uptime history on the admin console',
  current_agent: 'child_workflows',
  repository_count: 2,
  required_repository_count: 2,
  repositories: [],
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
  created_at: '2026-08-25T08:00:00Z',
  updated_at: '2026-08-25T09:00:00Z',
};

type Summary = Awaited<ReturnType<FeatureApi['listFeatures']>>['features'][number];

function summary(overrides: Partial<Summary> = {}): Summary {
  return {
    feature_id: 'f-1',
    workflow_id: 'w-1',
    title: 'Server uptime history on the admin console',
    status: 'failed_requires_human',
    execution_mode: 'live',
    created_at: '2026-08-25T08:00:00Z',
    updated_at: '2026-08-25T09:00:00Z',
    repository_count: 2,
    pull_request_count: 0,
    human_action_required: true,
    dashboard_group: 'waiting',
    ...overrides,
  };
}

function workstream(overrides: Record<string, unknown> = {}) {
  return {
    repository_id: 'admanager-server',
    repository_name: 'Ad Manager Server',
    repository_role: 'backend',
    repository_url: 'https://github.com/example/admanager.git',
    repository_required: true,
    child_workflow_id: 'f-1:admanager-server',
    workstream_id: 'ws-1',
    status: 'failed',
    branch_name: 'ai/f-1/admanager-server/server-uptime-history-on-the-admin-console',
    workspace_path: '/workspaces/f-1/admanager-server',
    retry_count: 2,
    code_completion_artifact_id: null,
    review_artifact_id: null,
    blocking_issues: [],
    pull_request_artifact_id: null,
    current_revision: '76eab2f3aa11bb22cc33dd44ee55ff6600112233',
    current_validation_results: [
      {
        name: 'npm',
        command: 'npm run lint',
        passed: true,
        validation_type: 'lint',
        status: 'passed',
        exit_code: 0,
        duration_seconds: 3.02,
        repository_revision: '76eab2f3aa11bb22cc33dd44ee55ff6600112233',
        working_directory: '.',
        stdout_summary: '',
        stderr_summary: '',
        required: true,
        is_current: true,
      },
      {
        name: 'npm',
        command: 'npm run test',
        passed: false,
        validation_type: 'test',
        status: 'failed',
        exit_code: 1,
        duration_seconds: 12.5,
        repository_revision: '76eab2f3aa11bb22cc33dd44ee55ff6600112233',
        working_directory: '.',
        stdout_summary: 'Tests:       1 failed, 12 passed, 13 total',
        stderr_summary: 'healthHistory.test.js › retains an unhealthy sample',
        result_code: 'TEST_VALIDATION_FAILED',
        failure_classification: 'implementation_defect',
        required: true,
        is_current: true,
      },
    ],
    preflight_status: 'ready',
    blocking_setup_issues: [],
    selected_package_manager: 'npm',
    technology_profile: { primary_language: 'JavaScript', frameworks: ['Express'] },
    production_files_changed: [],
    test_files_changed: [],
    configuration_files_changed: [],
    requirements_implemented: ['REQ-1'],
    requirements_not_implemented: [],
    implementation_retry_count: 0,
    validation_retry_count: 2,
    repository_setup_retry_count: 0,
    integration_retry_count: 0,
    granted_extra_attempts: 0,
    retry_grants: [],
    scoped_requirements: [{ requirement_id: 'REQ-1', responsibility: 'implements' }],
    implementation_expectations: [],
    configured_validation_commands: [],
    out_of_scope_requirements: [],
    planned_blind: false,
    available_actions: [],
    ...overrides,
  };
}

const PRD_ARTIFACT = {
  artifact_id: '001_prd.json',
  artifact_type: 'prd',
  schema_version: '1.0',
  producer: 'product_manager',
  validation_status: 'valid',
  timestamp: '2026-08-25T08:00:00Z',
  metadata: {},
  payload: {
    title: 'Server uptime history',
    problem_statement: 'Operators cannot see whether a server has been flapping.',
    goals: ['Expose a rolling health history'],
    requirements: [
      {
        requirement_id: 'REQ-1',
        description: 'Expose a health-history endpoint.',
        priority: 'must',
        acceptance_criteria: ['Returns a JSON array of samples.'],
      },
      {
        requirement_id: 'REQ-9',
        description: 'Something nobody was assigned.',
        priority: 'should',
        acceptance_criteria: [],
      },
    ],
  },
};

const TIMELINE = [
  {
    timestamp: '2026-08-25T08:00:00Z',
    event_type: 'lifecycle',
    source: 'api',
    event: 'feature_started',
    details: {},
  },
  {
    timestamp: '2026-08-25T08:30:00Z',
    event_type: 'lifecycle',
    source: 'api',
    event: 'child_workflow_failed',
    details: { repository_id: 'admanager-server' },
  },
  {
    timestamp: '2026-08-25T08:45:00Z',
    event_type: 'lifecycle',
    source: 'api',
    event: 'pull_request_created',
    details: { repository_id: 'admanager-server' },
  },
];

const BASE: Partial<FeatureApi> = {
  listFeatures: async () => ({ features: [summary()], next_cursor: null }),
  getFeature: async () => FEATURE,
  getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [workstream()] }),
  listEvents: async () => ({ feature_id: 'f-1', events: [], last_event_id: null }),
  getTimeline: async () => ({ feature_id: 'f-1', events: TIMELINE }),
  getPullRequests: async () => ({ feature_id: 'f-1', pull_requests: [] }),
  listArtifacts: async (_id, params) => ({
    feature_id: 'f-1',
    artifacts:
      params?.artifactType === undefined || params.artifactType === 'prd' ? [PRD_ARTIFACT] : [],
  }),
  getArtifact: async () => PRD_ARTIFACT,
  listRepairs: async () => ({ feature_id: 'f-1', repairs: [] }),
  getClarification: async () => ({
    feature_id: 'f-1',
    awaiting_answers: false,
    technical_prd_artifact_id: null,
    clarification_rounds: 0,
    max_clarification_rounds: 10,
    questions: [],
    previous_answers: {},
    design_conflicts: [],
  }),
  getChatHistory: async () => ({ feature_id: 'f-1', messages: [] }),
};

// The shell asks for a platform token before it renders anything, which is the right
// behaviour and not what these tests are about.
beforeEach(() => writeToken('test-token'));

/** The whole shell, so the sidebar and breadcrumbs are part of what is under test. */
function renderApp(path: string, api: Partial<FeatureApi> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi({ ...BASE, ...api })} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[path]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<DashboardPage />} />
            {/* The old activity routes are redirects now; the filter lives in the query
                string on the one Features page. */}
            <Route path="needs-attention" element={<Navigate to="/?group=waiting" replace />} />
            <Route path="features/:featureId" element={<FeatureWorkspacePage />} />
            <Route
              path="features/:featureId/repositories/:repositoryId"
              element={<RepositoryPage />}
            />
            <Route path="features/:featureId/:tab" element={<FeatureWorkspacePage />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('a product manager finding what needs them', () => {
  it('offers the two things you can do, and Settings apart from them', async () => {
    renderApp('/');

    await screen.findByRole('table', { name: 'Features' });
    const navigation = screen.getByRole('navigation', { name: 'Main' });
    expect(within(navigation).getAllByRole('link').map((item) => item.textContent)).toEqual([
      'Features',
      'New feature',
      'Settings',
    ]);
  });

  it('counts the features waiting on a person, as a filter you can press', async () => {
    renderApp('/');

    // The count and the table read the same query, so they cannot disagree -- and the count
    // appears only once there is an answer.
    await screen.findByRole('table', { name: 'Features' });
    const groups = screen.getByRole('tablist', { name: 'Feature groups' });
    expect(within(groups).getByRole('tab', { name: 'Needs attention (1)' })).toBeInTheDocument();
  });

  it('opens a feature from the table and says where you are', async () => {
    renderApp('/');
    const user = userEvent.setup();

    const table = await screen.findByRole('table', { name: 'Features' });
    await user.click(within(table).getByRole('link', { name: FEATURE.title }));

    expect(await screen.findByRole('heading', { name: FEATURE.title, level: 1 })).toBeInTheDocument();
    // The trail names the feature rather than its identifier once the title is known.
    const trail = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(within(trail).getByText('Features')).toBeInTheDocument();
    expect(within(trail).getByText(FEATURE.title)).toBeInTheDocument();
  });

  it('keeps the old needs-attention link working, as a filter on the Features page', async () => {
    renderApp('/needs-attention');

    // One screen, not four. The URL still says which set the sender was looking at.
    expect(await screen.findByRole('heading', { name: 'Features', level: 1 })).toBeInTheDocument();
    const groups = await screen.findByRole('tablist', { name: 'Feature groups' });
    expect(within(groups).getByRole('tab', { name: 'Needs attention (1)' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    const table = await screen.findByRole('table', { name: 'Features' });
    expect(within(table).getByText(FEATURE.title)).toBeInTheDocument();
    expect(within(table).getByText('Action required')).toBeInTheDocument();
  });
});

describe('an engineer reaching the exact failure', () => {
  it('walks feature → repository → validation → command and error', async () => {
    renderApp('/features/f-1');
    const user = userEvent.setup();

    // Feature → repository.
    const repositories = await screen.findByRole('table', { name: 'Repository workstreams' });
    await user.click(within(repositories).getByRole('link', { name: 'Ad Manager Server' }));

    // Repository → validation.
    const validationTab = await screen.findByRole('link', { name: /Validation/ });
    await user.click(validationTab);

    // Validation → the exact command and how it ended.
    const checks = await screen.findByRole('table', { name: 'Validation checks' });
    expect(within(checks).getByText('npm run test')).toBeInTheDocument();
    expect(within(checks).getByText('failed')).toBeInTheDocument();

    // → the exact error.
    await user.click(within(checks).getAllByRole('row')[2]!);
    const drawer = await screen.findByRole('dialog');
    // The captured error, and the platform's own classification of it. It appears twice:
    // once as the summary, and again inside the record the platform kept.
    expect(within(drawer).getAllByText(/healthHistory.test.js/).length).toBeGreaterThan(0);
    expect(within(drawer).getByText('implementation_defect')).toBeInTheDocument();
  });

  it('filters the history to the errors without leaving the page', async () => {
    renderApp('/features/f-1/history');
    const user = userEvent.setup();

    expect(await screen.findByText('Pull request opened')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /^All/ })).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: /^Errors/ }));

    expect(screen.getByText('Repository workstream stopped')).toBeInTheDocument();
    expect(screen.queryByText('Pull request opened')).not.toBeInTheDocument();
  });
});

describe('requirements, joined to what was built', () => {
  it('says which repository owns each requirement and whether it was implemented', async () => {
    renderApp('/features/f-1/prd');

    const table = await screen.findByRole('table', { name: 'Requirements' });
    const implemented = within(table).getByText('REQ-1').closest('tr')!;
    // The repository reported implementing REQ-1, so the row says so and names the repository.
    expect(within(implemented).getByText('Implemented')).toBeInTheDocument();
    expect(within(implemented).getByText('admanager-server')).toBeInTheDocument();

    // REQ-9 was never scoped to a repository. Nothing is invented for it.
    const unassigned = within(table).getByText('REQ-9').closest('tr')!;
    expect(within(unassigned).getByText('Unassigned')).toBeInTheDocument();
  });
});

describe('the assistant', () => {
  it('opens beside the feature and puts that in the URL, so the view is linkable', async () => {
    renderApp('/features/f-1');
    const user = userEvent.setup();

    await screen.findByRole('heading', { name: FEATURE.title, level: 1 });
    expect(screen.queryByRole('complementary', { name: 'Feature assistant' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Ask AI' }));
    const panel = await screen.findByRole('complementary', { name: 'Feature assistant' });

    // It says what it is looking at rather than dumping the context it was given.
    expect(within(panel).getByText('Context')).toBeInTheDocument();
    expect(within(panel).getAllByText(FEATURE.title).length).toBeGreaterThan(0);
    expect(within(panel).getByText('admanager-server')).toBeInTheDocument();
    // And it suggests the questions somebody opens it to ask.
    expect(within(panel).getByRole('button', { name: 'Why is this blocked?' })).toBeInTheDocument();

    await user.click(within(panel).getByRole('button', { name: 'Close assistant' }));
    expect(screen.queryByRole('complementary', { name: 'Feature assistant' })).not.toBeInTheDocument();
  });

  it('opens from a link that asked for it', async () => {
    renderApp('/features/f-1?chat=open');

    expect(await screen.findByRole('complementary', { name: 'Feature assistant' })).toBeInTheDocument();
  });
});

describe('links people already sent each other', () => {
  it('keeps the sections this workspace used to have as their own tabs', async () => {
    // `requirements` became `prd`. A link in somebody's chat history must not 404.
    renderApp('/features/f-1/requirements');

    expect(await screen.findByText('Product requirements')).toBeInTheDocument();
  });
});

describe('the workflow graph', () => {
  it('is a tab of its own and draws a node per stage and per repository step', async () => {
    renderApp('/features/f-1/workflow');

    const graph = await screen.findByRole('group', { name: 'Feature execution graph' });
    // Named for what it is and how it is going, so the graph reads without being seen.
    expect(within(graph).getByRole('button', { name: /^Stage, Product manager/ })).toBeInTheDocument();
    expect(
      within(graph).getByRole('button', { name: /^Repository, Ad Manager Server/ }),
    ).toBeInTheDocument();
    expect(
      within(graph).getByRole('button', { name: /^Validation, Validation, admanager-server/ }),
    ).toBeInTheDocument();
    expect(
      within(graph).getByRole('button', { name: /^Pull requests, Pull requests/ }),
    ).toBeInTheDocument();
  });

  it('says a repository is retrying, and against which budget', async () => {
    renderApp('/features/f-1/workflow', {
      // A feature still running, under the budgets the server publishes with it.
      getFeature: async () => ({
        ...FEATURE,
        status: 'running_child_workflows',
        max_child_review_cycles: 12,
        max_validation_retries: 8,
        max_implementation_retries: 8,
        max_clarification_rounds: 10,
        max_integration_review_cycles: 5,
        max_repository_setup_retries: 1,
      }),
      getWorkstreams: async () => ({
        feature_id: 'f-1',
        workstreams: [workstream({ status: 'running', code_completion_artifact_id: 'c' })],
      }),
    });

    const graph = await screen.findByRole('group', { name: 'Feature execution graph' });
    // Two validation retries against the feature's own published budget of eight. "Still
    // running" and "on its third attempt" are different situations, and the graph says which.
    expect(
      within(graph).getByRole('button', { name: /Validation.*Retry 2 of 8/ }),
    ).toBeInTheDocument();
    // The repository lane and the step it is stuck on both say so.
    expect(within(graph).getAllByRole('button', { name: /retrying/ }).length).toBeGreaterThan(0);
  });

  it('opens what a node stands for, and the evidence stays one link away', async () => {
    renderApp('/features/f-1/workflow');
    const user = userEvent.setup();

    const graph = await screen.findByRole('group', { name: 'Feature execution graph' });
    await user.click(
      within(graph).getByRole('button', { name: /^Validation, Validation, admanager-server/ }),
    );

    // A lane node opens what its agents are doing (56); the evidence the node used to jump
    // straight to is the drawer's onward link, so nothing the click reached is lost.
    const drawer = await screen.findByRole('dialog', { name: 'Agent work' });
    await user.click(within(drawer).getByRole('link', { name: 'Open validation' }));
    expect(await screen.findByRole('table', { name: 'Validation checks' })).toBeInTheDocument();
  });

  it('offers the graph from the Overview, with the step the feature is on', async () => {
    renderApp('/features/f-1', {
      getFeature: async () => ({ ...FEATURE, status: 'running_child_workflows' }),
    });
    const user = userEvent.setup();

    expect(await screen.findByText('Currently')).toBeInTheDocument();
    await user.click(screen.getByRole('link', { name: 'View workflow' }));

    expect(
      await screen.findByRole('group', { name: 'Feature execution graph' }),
    ).toBeInTheDocument();
  });
});
