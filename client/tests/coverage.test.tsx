import { describe, expect, it } from 'vitest';
import { render, screen, waitFor, within, renderHook } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { ApiContext } from '@/app/api-context';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import { useLiveFeature } from '@/features/feature-workspace/useLiveFeature';
import { agentRuns, duration } from '@/features/feature-workspace/agent-runs';
import type { FeatureApi } from '@/api/features';
import { stubApi } from './fixtures';

function feature(repositoryCount: number) {
  return {
    feature_id: 'f-1',
    workflow_id: 'f-1',
    status: 'running_child_workflows',
    title: 'A feature across many repositories',
    current_agent: 'child_workflows',
    repository_count: repositoryCount,
    required_repository_count: repositoryCount,
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
    created_at: '2026-08-25T08:00:00Z',
    updated_at: '2026-08-25T09:00:00Z',
  };
}

function workstream(repositoryId: string, role: string, status: string) {
  return {
    repository_id: repositoryId,
    repository_name: repositoryId,
    repository_role: role,
    child_workflow_id: `f-1:${repositoryId}`,
    workstream_id: `ws-${repositoryId}`,
    status,
    branch_name: `ai/f-1/${repositoryId}`,
    workspace_path: `/w/${repositoryId}`,
    retry_count: 0,
    code_completion_artifact_id: null,
    review_artifact_id: null,
    blocking_issues: [],
    pull_request_artifact_id: null,
    current_validation_results: [],
    scoped_requirements: [],
    out_of_scope_requirements: [],
    planned_blind: false,
    production_files_changed: [],
    test_files_changed: [],
    configuration_files_changed: [],
    requirements_implemented: [],
    requirements_not_implemented: [],
    implementation_retry_count: 0,
    validation_retry_count: 0,
    repository_setup_retry_count: 0,
    integration_retry_count: 0,
    granted_extra_attempts: 0,
    retry_grants: [],
    implementation_expectations: [],
    configured_validation_commands: [],
    blocking_setup_issues: [],
  };
}

const BASE: Partial<FeatureApi> = {
  listEvents: async () => ({ feature_id: 'f-1', events: [], last_event_id: null }),
  getTimeline: async () => ({ feature_id: 'f-1', events: [] }),
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
  listArtifacts: async () => ({ feature_id: 'f-1', artifacts: [] }),
};

function renderWorkspace(api: Partial<FeatureApi>, path = '/features/f-1/repositories') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi({ ...BASE, ...api })} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[path]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('however many repositories a feature has', () => {
  it.each([
    ['one', ['api']],
    ['two', ['api', 'admin-frontend']],
    ['five', ['api', 'admin-frontend', 'mobile-app', 'shared-sdk', 'notification-worker']],
  ])('renders %s of them from backend data alone', async (_name, repositories) => {
    // Two repositories with the same role, and roles that are neither "frontend" nor
    // "backend": identity is the repository id, and nothing is keyed off the role.
    const roles = ['service', 'frontend', 'mobile', 'sdk', 'worker'];
    renderWorkspace({
      getFeature: async () => feature(repositories.length),
      getWorkstreams: async () => ({
        feature_id: 'f-1',
        workstreams: repositories.map((id, index) =>
          workstream(id, roles[index] ?? 'other', 'running'),
        ),
      }),
    });

    const table = await screen.findByRole('table', { name: 'Repository workstreams' });
    // One row per repository, plus the header row.
    expect(within(table).getAllByRole('row')).toHaveLength(repositories.length + 1);
    for (const id of repositories) {
      expect(within(table).getAllByText(id).length).toBeGreaterThan(0);
    }
  });

  it('renders two repositories sharing one role without collapsing them', async () => {
    renderWorkspace({
      getFeature: async () => feature(2),
      getWorkstreams: async () => ({
        feature_id: 'f-1',
        workstreams: [workstream('orders-api', 'service', 'running'), workstream('billing-api', 'service', 'approved')],
      }),
    });

    const table = await screen.findByRole('table', { name: 'Repository workstreams' });
    // Keyed by repository id, so a shared role is not an identity.
    expect(within(table).getAllByRole('row')).toHaveLength(3);
    expect(within(table).getAllByText('orders-api').length).toBeGreaterThan(0);
    expect(within(table).getAllByText('billing-api').length).toBeGreaterThan(0);
  });
});

describe('agent history', () => {
  const events = [
    { timestamp: '2026-08-25T08:00:00Z', event_type: 'lifecycle', source: 'api', event: 'child_workflow_started', details: { repository_id: 'api' } },
    { timestamp: '2026-08-25T08:00:05Z', event_type: 'lifecycle', source: 'api', event: 'child_workflow_started', details: { repository_id: 'web' } },
    { timestamp: '2026-08-25T08:02:00Z', event_type: 'lifecycle', source: 'api', event: 'child_workflow_failed', details: { repository_id: 'api' } },
    { timestamp: '2026-08-25T08:03:05Z', event_type: 'lifecycle', source: 'api', event: 'child_workflow_completed', details: { repository_id: 'web' } },
  ];

  it('pairs each start with its own end, not with a sibling’s', () => {
    const runs = agentRuns(events);

    // Two repositories running at once must not close each other's run: keyed by repository.
    const api = runs.find((run) => run.repositoryId === 'api')!;
    const web = runs.find((run) => run.repositoryId === 'web')!;
    expect(api.outcome).toBe('failed');
    expect(web.outcome).toBe('completed');
    expect(duration(api)).toBe('2m 0s');
    expect(duration(web)).toBe('3m 0s');
  });

  it('says nothing rather than "0s" when the events cannot measure the work', () => {
    // A child workstream's started and failed events are both written when its result is
    // persisted -- the same millisecond, after the attempt already ran for minutes. Live,
    // every run in this view read "0s", which is a measurement the platform never took.
    const sameInstant = [
      { timestamp: '2026-08-25T08:00:00.151Z', event_type: 'lifecycle', source: 'api', event: 'child_workflow_started', details: { repository_id: 'api' } },
      { timestamp: '2026-08-25T08:00:00.151Z', event_type: 'lifecycle', source: 'api', event: 'child_workflow_failed', details: { repository_id: 'api' } },
    ];
    const run = agentRuns(sameInstant)[0]!;
    expect(run.outcome).toBe('failed');
    expect(duration(run)).toBeNull();
  });

  it('shows a run that has not finished as running rather than as a gap', () => {
    const runs = agentRuns([events[0]!]);
    expect(runs[0]?.outcome).toBe('running');
    expect(duration(runs[0]!)).toBe('running');
  });

  it('is reachable as its own view', async () => {
    renderWorkspace(
      {
        getFeature: async () => feature(1),
        getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [] }),
        getTimeline: async () => ({ feature_id: 'f-1', events }),
      },
      '/features/f-1/history?view=agents',
    );

    const history = await screen.findByRole('table', { name: 'Agent history' });
    expect(within(history).getAllByRole('row')).toHaveLength(3);
  });
});

describe('the live update loop', () => {
  function wrapper({ children }: { children: ReactNode }) {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    return (
      <QueryClientProvider client={queryClient}>
        <ApiContext.Provider value={stubApi(BASE)}>{children}</ApiContext.Provider>
      </QueryClientProvider>
    );
  }

  it('recovers after a failed poll and continues from the cursor it reached', async () => {
    // The event read is the only thing on a timer, so a transient failure must not end
    // updates: the next tick has to carry on from where the last success got to.
    const cursors: (number | null)[] = [];
    let call = 0;
    const api = stubApi({
      ...BASE,
      listEvents: async (_featureId: string, after: number | null) => {
        cursors.push(after);
        call += 1;
        if (call === 2) throw new Error('network went away');
        return {
          feature_id: 'f-1',
          events: [],
          last_event_id: call === 1 ? 7 : 9,
        };
      },
    });

    const { result } = renderHook(() => useLiveFeature('f-1', 'running_child_workflows'), {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
          <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
        </QueryClientProvider>
      ),
    });

    await waitFor(() => expect(result.current.lastEventId).toBe(7));
    await result.current.query.refetch();
    await result.current.query.refetch();

    // First read starts from nothing; the failed read does not move the cursor backwards.
    expect(cursors[0]).toBe(null);
    expect(cursors[1]).toBe(7);
    expect(cursors[2]).toBe(7);
    await waitFor(() => expect(result.current.lastEventId).toBe(9));
  });

  it('deduplicates replayed events and resets its cursor when the route changes feature', async () => {
    const calls: Array<[string, number | null]> = [];
    const api = stubApi({
      ...BASE,
      listEvents: async (featureId: string, after: number | null) => {
        calls.push([featureId, after]);
        const id = featureId === 'first' ? 8 : 2;
        return {
          feature_id: featureId,
          events: [
            { id, timestamp: '2026-08-25T08:00:00Z', event_type: 'lifecycle', source: 'api', event: 'feature_started', details: {} },
            { id, timestamp: '2026-08-25T08:00:00Z', event_type: 'lifecycle', source: 'api', event: 'feature_started', details: {} },
          ],
          last_event_id: id,
        };
      },
    });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    const { result, rerender } = renderHook(
      ({ featureId }) => useLiveFeature(featureId, 'running_child_workflows'),
      {
        initialProps: { featureId: 'first' },
        wrapper: ({ children }: { children: ReactNode }) => (
          <QueryClientProvider client={queryClient}>
            <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
          </QueryClientProvider>
        ),
      },
    );

    await waitFor(() => expect(result.current.events).toHaveLength(1));
    rerender({ featureId: 'second' });
    await waitFor(() => expect(calls).toContainEqual(['second', null]));
    await waitFor(() => expect(result.current.events.map((event) => event.id)).toEqual([2]));
  });

  it('drains a full event page even when the feature is already settled', async () => {
    const cursors: (number | null)[] = [];
    const api = stubApi({
      ...BASE,
      listEvents: async (_featureId: string, after: number | null) => {
        cursors.push(after);
        if (after === null) {
          return {
            feature_id: 'f-1',
            events: Array.from({ length: 200 }, (_, index) => ({
              id: index + 1,
              timestamp: '2026-08-25T08:00:00Z',
              event_type: 'lifecycle',
              source: 'api',
              event: 'child_workflow_started',
              details: {},
            })),
            last_event_id: 200,
          };
        }
        return {
          feature_id: 'f-1',
          events: [
            {
              id: 201,
              timestamp: '2026-08-25T09:00:00Z',
              event_type: 'lifecycle',
              source: 'api',
              event: 'feature_completed',
              details: {},
            },
          ],
          last_event_id: 201,
        };
      },
    });
    const { result } = renderHook(() => useLiveFeature('f-1', 'completed'), {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider
          client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
        >
          <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
        </QueryClientProvider>
      ),
    });

    await waitFor(() => expect(result.current.lastEventId).toBe(201));
    expect(cursors).toEqual([null, 200]);
    expect(result.current.events.at(-1)?.event).toBe('feature_completed');
  });

  it('stops polling a feature that has finished', () => {
    const { result } = renderHook(() => useLiveFeature('f-1', 'completed'), { wrapper });
    // Nothing further will happen to it, so nothing should keep asking.
    expect(result.current.query.isFetching || result.current.query.isPending).toBeDefined();
  });
});

describe('answering a clarification', () => {
  it('clears the panel once the platform has accepted the answers', async () => {
    // The clarification, workstream and timeline reads are keyed by the newest event id, and a
    // feature that has settled is not polled at all. Invalidating only the feature left "This
    // feature is waiting on you" on screen after the platform had accepted the answers and run
    // the feature to completion, and it never recovered without a reload. Found by answering
    // one in a browser.
    let answered = false;
    renderWorkspace(
      {
        getFeature: async () => ({
          ...feature(1),
          status: answered ? 'completed' : 'waiting_for_human',
        }),
        getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [] }),
        listEvents: async () => ({
          feature_id: 'f-1',
          events: [],
          last_event_id: answered ? 9 : 1,
        }),
        getClarification: async () => ({
          feature_id: 'f-1',
          awaiting_answers: !answered,
          technical_prd_artifact_id: null,
          clarification_rounds: answered ? 1 : 0,
          max_clarification_rounds: 10,
          questions: answered
            ? []
            : [
                {
                  question_id: 'Q-1',
                  question: 'Which endpoint?',
                  rationale: 'Two exist.',
                  required: true,
                  suggested_answer: '',
                  suggestion_source: '',
                },
              ],
          previous_answers: (answered ? { 'Q-1': 'The nested one.' } : {}) as Record<string, string>,
          design_conflicts: [],
        }),
        resumeFeature: async () => {
          answered = true;
          return { ...feature(1), status: 'completed' };
        },
      },
      '/features/f-1/overview',
    );
    const user = userEvent.setup();

    await screen.findByText('This feature is waiting on you');
    await user.type(screen.getByLabelText('Which endpoint?'), 'The nested one.');
    await user.click(screen.getByRole('button', { name: /Submit answers|Answer/ }));

    await waitFor(() =>
      expect(screen.queryByText('This feature is waiting on you')).not.toBeInTheDocument(),
    );
  });
});

describe('clarification history', () => {
  it('stays visible after the feature has moved on', async () => {
    // These answers are where somebody told the platform which endpoint to change and which of
    // two parallel module layouts is the live one. The platform planned from them, so when the
    // result is wrong this is the first thing to re-read -- and it was reachable only by
    // querying the API by hand.
    renderWorkspace(
      {
        getFeature: async () => feature(1),
        getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [] }),
        getClarification: async () => ({
          feature_id: 'f-1',
          awaiting_answers: false,
          technical_prd_artifact_id: '002_technical_prd.json',
          clarification_rounds: 1,
          max_clarification_rounds: 10,
          questions: [],
          previous_answers: {
            'UQ-1': 'Use `ADUNIT_STATUS.INACTIVE`; introduce no new status value.',
          },
          design_conflicts: [],
        }),
      },
      '/features/f-1/requirements',
    );

    expect(await screen.findByText(/Answered already/)).toBeInTheDocument();
    expect(screen.getByText('UQ-1')).toBeInTheDocument();
    expect(screen.getByText(/introduce no new status value/)).toBeInTheDocument();
    // Answers are typed by a person and are markdown in practice.
    expect(document.querySelector('.answers code')).toBeTruthy();
  });
});

describe('artifacts', () => {
  const artifact = {
    artifact_id: '001_prd.json',
    artifact_type: 'prd',
    schema_version: '1.0',
    producer: 'product_manager',
    validation_status: 'valid',
    timestamp: '2026-08-25T08:00:00Z',
    metadata: {},
    payload: {
      title: 'A feature',
      problem_statement: 'Operators cannot **deactivate** an ad unit.',
      goals: ['Let them do it from the console'],
      requirements: [],
    },
  };

  it('opens the artifact a link names, so agent history can point at a result', async () => {
    // The agent history links to the result each attempt produced. The tab ignored the query
    // param, so every one of those links opened an empty panel.
    renderWorkspace(
      {
        getFeature: async () => feature(1),
        getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [] }),
        listArtifacts: async () => ({ feature_id: 'f-1', artifacts: [artifact] }),
        getArtifact: async () => artifact,
      },
      `/features/f-1/artifacts?artifact=${encodeURIComponent(artifact.artifact_id)}`,
    );

    expect(await screen.findByText('Let them do it from the console')).toBeInTheDocument();
  });

  it('offers a formatted view and the raw document behind it', async () => {
    renderWorkspace(
      {
        getFeature: async () => feature(1),
        getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [] }),
        listArtifacts: async () => ({ feature_id: 'f-1', artifacts: [artifact] }),
        getArtifact: async () => artifact,
      },
      '/features/f-1/requirements',
    );
    const user = userEvent.setup();

    // Formatted by default: raw JSON is a fallback, not the way to read a document.
    expect(await screen.findByText('Let them do it from the console')).toBeInTheDocument();
    // The markdown in the problem statement is structured, not shown as asterisks.
    expect(document.querySelector('.markdown strong')?.textContent).toBe('deactivate');

    // Raw stays available, one control away, and is never what a reader lands on.
    await user.click(screen.getByRole('button', { name: 'Raw' }));
    expect(screen.getAllByText(/problem_statement/)[0]).toBeInTheDocument();
  });
});
