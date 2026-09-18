// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import type { ExecutionRecord, WorkstreamAttempt, WorkstreamOperation } from '@/schemas/feature';
import { workstreamOperationsSchema } from '@/schemas/feature';
import {
  attemptMarkers,
  attemptViews,
  groupByStage,
  partitionAttempts,
  planningNodeOperations,
  stageElapsed,
  unjournaledPhase,
} from '@/features/feature-workspace/agent-work';
import { stubApi } from './fixtures';
import live201Backend from './fixtures/operations.bulk-apps-live-201.backend.json';

/**
 * The agent shows its work (56): a repository lane node opens into the journal's operations,
 * grouped by the agent that owns each stage, with the 55- liveness treatment on the running
 * row. The view a person needed four times during the 185-194 cycle and got only by querying
 * Postgres.
 *
 * The two tests that decide correctness are the geometry one -- the drill-in must not move a
 * single node -- and the divider one: an attempt boundary is drawn only where the data proves
 * it, because a wrong divider tells a false story about which attempt did what.
 */

const FEATURE_ID = 'adunit-deactivate-live-186';
const REPOSITORY = 'admanager-server';

const FEATURE = {
  feature_id: FEATURE_ID,
  workflow_id: 'workflow-186',
  status: 'running_child_workflows',
  title: 'The 186 story',
  current_agent: 'engineer',
  repository_count: 1,
  required_repository_count: 1,
  repositories: [],
  clarification_rounds: 0,
  integration_review_cycles: 0,
  max_clarification_rounds: 10,
  max_integration_review_cycles: 5,
  max_child_review_cycles: 8,
  max_implementation_retries: 4,
  max_validation_retries: 2,
  max_repository_setup_retries: 1,
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
    repository_id: REPOSITORY,
    repository_name: 'Ad Manager Server',
    repository_role: 'backend',
    child_workflow_id: `${FEATURE_ID}:${REPOSITORY}`,
    workstream_id: REPOSITORY,
    status: 'running',
    branch_name: 'ai/x',
    workspace_path: '/w',
    retry_count: 0,
    code_completion_artifact_id: null,
    review_artifact_id: null,
    blocking_issues: [],
    pull_request_artifact_id: null,
    current_validation_results: [],
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
    scoped_requirements: [],
    out_of_scope_requirements: [],
    planned_blind: false,
    ...overrides,
  };
}

let sequence = 0;

/**
 * One journal row as the endpoint serves it. Times are stated per test, never invented.
 *
 * `child_attempt` is null by default -- the shape of every row written before the platform
 * stamped attempts, which is what keeps the divider tests above testing the clone heuristic
 * rather than the stamp that replaced it.
 */
function operation(overrides: Partial<WorkstreamOperation> = {}): WorkstreamOperation {
  sequence += 1;
  return {
    operation_id: `op-${sequence}`,
    operation_type: 'run_coding_executor',
    stage: 'coding',
    status: 'succeeded',
    attempt: 1,
    max_attempts: 1,
    child_attempt: null,
    started_at: '2026-08-24T22:41:00Z',
    heartbeat_at: null,
    completed_at: '2026-08-24T22:42:00Z',
    error_code: null,
    repeat: null,
    stream_reissues: null,
    ...overrides,
  };
}

/**
 * One served per-attempt ending, in the shape the endpoint answers with.
 *
 * Built inline like the rows above, and for the same reason: these tests are about the words
 * a person reads. The block's *shape* is pinned separately, by a real captured payload parsed
 * through the production schema.
 */
function ending(overrides: Partial<WorkstreamAttempt> = {}): WorkstreamAttempt {
  return {
    attempt: 0,
    ended_by: 'review_rejected',
    stage: 'review',
    detail: 'review returned changes requested with 4 findings, blocking on F-1',
    workspace: 'preserved',
    self_review_outcome: 'clean',
    self_review_corrected_files: 0,
    source_repair_passes: null,
    stream_reissues: null,
    ...overrides,
  };
}

function renderWorkflow(
  operations: WorkstreamOperation[],
  overrides: Partial<FeatureApi> = {},
  endings: WorkstreamAttempt[] = [],
) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const api: Partial<FeatureApi> = {
    listEvents: async () => ({ feature_id: FEATURE_ID, events: [], last_event_id: null }),
    getFeature: async () => FEATURE,
    getWorkstreams: async () => ({ feature_id: FEATURE_ID, workstreams: [workstream()] }),
    getTimeline: async () => ({ feature_id: FEATURE_ID, events: [] }),
    listArtifacts: async () => ({ feature_id: FEATURE_ID, artifacts: [] }),
    getExecutions: async () => ({ feature_id: FEATURE_ID, executions: [] }),
    listWorkstreamOperations: async (featureId: string, repositoryId: string) => ({
      feature_id: featureId,
      repository_id: repositoryId,
      operations,
      attempts: endings,
    }),
    ...overrides,
  } as Partial<FeatureApi>;
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[`/features/${FEATURE_ID}/workflow`]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

/** Open the drill-in from the lane's Implementation node. */
async function openDrillIn(name = /^Implementation, Implementation, admanager-server/) {
  const node = await screen.findByRole('button', { name });
  await userEvent.click(node);
  return screen.findByRole('dialog', { name: 'Agent work' });
}

function stageSection(drawer: HTMLElement, name: string): HTMLElement {
  const section = within(drawer)
    .getAllByRole('region')
    .find((item) => item.getAttribute('aria-label') === name);
  expect(section, `stage group "${name}"`).toBeDefined();
  return section!;
}

describe('attempt partitioning (the divider rules)', () => {
  it('splits confidently at a fresh clone_repository', () => {
    const rows = [
      operation({ operation_type: 'run_tests', stage: 'validation' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    const { attempts } = partitionAttempts(rows);
    expect(attempts).toHaveLength(2);
    expect(attempts[0]!.map((item) => item.operation_type)).toEqual([
      'clone_repository',
      'run_coding_executor',
    ]);
    expect(attempts[1]!.map((item) => item.operation_type)).toEqual([
      'clone_repository',
      'run_tests',
    ]);
  });

  it('never splits at install_dependencies: the server journals custom validation commands and repairs under that type', () => {
    // A5's trap, as data: an install row between two coding bursts is a custom validation
    // command, not a new attempt. Splitting here would be the wrong divider.
    const rows = [
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    expect(partitionAttempts(rows).attempts).toHaveLength(1);
  });

  it('keeps an in-place regeneration (40-) inside its attempt rather than guessing a boundary', () => {
    const rows = [
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'run_reviewer', stage: 'review' }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    expect(partitionAttempts(rows).attempts).toHaveLength(1);
  });
});

describe('stage grouping', () => {
  it('renders a stage name the order has never heard of after publication rather than dropping it', () => {
    const groups = groupByStage(
      // Chronological, as partitionAttempts hands them over.
      [
        operation({ operation_type: 'clone_repository', stage: 'setup' }),
        operation({ operation_type: 'run_quantum_gate', stage: 'quantum' }),
      ],
      { includePending: false },
    );
    expect(groups.map((group) => group.stage)).toEqual(['setup', 'quantum']);
    expect(groups[1]!.agent).toBeNull();
  });

  it('shows the five canonical stages of the current attempt even before rows exist', () => {
    const groups = groupByStage([operation({ operation_type: 'clone_repository', stage: 'setup' })], {
      includePending: true,
    });
    expect(groups.map((group) => group.stage)).toEqual([
      'setup',
      'coding',
      'validation',
      'review',
      'publication',
    ]);
  });
});

describe('A1 — the 186 story renders', () => {
  it('shows setup complete, the coding row alive on call 3, and review/publication pending', async () => {
    const rows = [
      // Newest first, exactly as the endpoint answers.
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        status: 'running',
        attempt: 3,
        max_attempts: 3,
        started_at: new Date(Date.now() - 60_000).toISOString(),
        heartbeat_at: new Date(Date.now() - 2_000).toISOString(),
        completed_at: null,
      }),
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        status: 'failed_retryable',
        attempt: 2,
        max_attempts: 3,
        error_code: 'provider_read_error',
      }),
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        status: 'failed_retryable',
        attempt: 1,
        max_attempts: 3,
        error_code: 'provider_read_error',
      }),
      operation({ operation_type: 'run_linter', stage: 'validation' }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    renderWorkflow(rows);
    const drawer = await openDrillIn();

    // The setup group is complete: both rows succeeded, said in words as well as glyphs.
    const setup = stageSection(drawer, 'Engineer, setup');
    expect(within(setup).getByText('clone_repository')).toBeInTheDocument();
    expect(within(setup).getByText('install_dependencies')).toBeInTheDocument();
    expect(within(setup).getAllByText('succeeded')).toHaveLength(2);

    // The coding group tells the truncation→fault→third-call story as it happens: two
    // failed calls with their raw status and error code, and call 3 alive right now.
    const coding = stageSection(drawer, 'Engineer, coding');
    expect(
      within(coding).getByText(/call 1 · failed_retryable · provider_read_error/),
    ).toBeInTheDocument();
    expect(
      within(coding).getByText(/call 2 · failed_retryable · provider_read_error/),
    ).toBeInTheDocument();
    const running = within(coding).getByText(/call 3 · running 1m \d+s · heartbeat \d+s ago/);
    expect(running.closest('.agent-work__row')).toHaveClass('agent-work__row--alive');

    // Review and publication have not been seen; the groups still render, saying so.
    expect(within(stageSection(drawer, 'Reviewer, review')).getByText('not yet seen')).toBeInTheDocument();
    expect(
      within(stageSection(drawer, 'Publisher, publication')).getByText('not yet seen'),
    ).toBeInTheDocument();
  });
});

describe('A2 — a completed feature renders its history', () => {
  it('groups in execution order, collapses the earlier attempt behind a divider, publication ✓', async () => {
    const rows = [
      // Newest first: a delivered second attempt, then the failed first attempt below it.
      operation({ operation_type: 'create_pull_request', stage: 'publication' }),
      operation({ operation_type: 'push_branch', stage: 'publication' }),
      operation({ operation_type: 'create_commit', stage: 'publication' }),
      operation({ operation_type: 'run_reviewer', stage: 'review' }),
      operation({ operation_type: 'run_tests', stage: 'validation' }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        status: 'failed_terminal',
      }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    renderWorkflow(rows, {
      getFeature: async () => ({ ...FEATURE, status: 'completed' }),
      getWorkstreams: async () => ({
        feature_id: FEATURE_ID,
        workstreams: [
          workstream({
            status: 'completed',
            code_completion_artifact_id: 'code.json',
            review_artifact_id: 'review.json',
            pull_request_artifact_id: 'pr.json',
          }),
        ],
      }),
    });
    const drawer = await openDrillIn();

    // The earlier attempt is behind its divider, collapsed, with the current attempt open.
    const divider = within(drawer).getByText('Earlier attempt · 3 operations');
    expect(divider.closest('details')).not.toHaveAttribute('open');
    expect(within(drawer).getByText('Current attempt')).toBeInTheDocument();

    // Groups render in execution order, labelled by the agent that owns each stage with the
    // stage word beside it -- both vocabularies, one line.
    const names = within(drawer)
      .getAllByRole('region')
      .map((section) => section.getAttribute('aria-label'));
    const currentAttemptNames = names.slice(names.length - 5);
    expect(currentAttemptNames).toEqual([
      'Engineer, setup',
      'Engineer, coding',
      'Engineer, validation',
      'Reviewer, review',
      'Publisher, publication',
    ]);

    const publication = stageSection(drawer, 'Publisher, publication');
    expect(within(publication).getByText('create_pull_request')).toBeInTheDocument();
    expect(within(publication).getAllByText('succeeded')).toHaveLength(3);
  });
});

describe('A3 — no geometry drift', () => {
  it('moves no node and resizes no canvas when the drill-in opens', async () => {
    const rows = [
      operation({ operation_type: 'run_coding_executor', stage: 'coding', status: 'running' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    const { container } = renderWorkflow(rows);
    await screen.findByRole('button', { name: /^Implementation, Implementation/ });

    const geometry = () => {
      const canvas = container.querySelector<HTMLElement>('.graph__canvas')!;
      const nodes = [...container.querySelectorAll<HTMLElement>('.graph__node')];
      return {
        canvas: { width: canvas.style.width, height: canvas.style.height },
        nodes: nodes.map((node) => ({
          label: node.getAttribute('aria-label'),
          left: node.style.left,
          top: node.style.top,
          width: node.style.width,
          height: node.style.height,
        })),
      };
    };

    const closed = geometry();
    expect(closed.nodes.length).toBeGreaterThan(0);
    const drawer = await openDrillIn();
    expect(drawer).toBeInTheDocument();
    // Open: every node in the same place, the canvas the same size. The drill-in is a
    // drawer beside the page, not a re-layout.
    expect(geometry()).toEqual(closed);
  });
});

describe('A4 — unknown operation types survive', () => {
  it('renders an unrecognised type raw, under the stage the server assigned', async () => {
    const rows = [
      operation({ operation_type: 'run_quantum_linter', stage: 'validation' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    renderWorkflow(rows);
    const drawer = await openDrillIn();
    const validation = stageSection(drawer, 'Engineer, validation');
    expect(within(validation).getByText('run_quantum_linter')).toBeInTheDocument();
  });
});

describe('A5 — no wrong dividers', () => {
  it('renders an ambiguous history without a divider rather than with a misplaced one', async () => {
    // An in-place regeneration plus a mid-attempt install: two patterns that look like
    // boundaries and are not. Everything renders in one attempt, chronologically within its
    // stage group, and no divider claims to know better.
    const rows = [
      operation({ operation_type: 'run_linter', stage: 'validation' }),
      operation({ operation_id: 'coding-2', operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
      operation({ operation_type: 'run_reviewer', stage: 'review' }),
      operation({ operation_id: 'coding-1', operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    renderWorkflow(rows);
    const drawer = await openDrillIn();

    expect(within(drawer).queryByText(/Earlier attempt/)).not.toBeInTheDocument();
    expect(within(drawer).queryByText('Current attempt')).not.toBeInTheDocument();
    // Both coding bursts are present in the coding group, oldest first.
    const coding = stageSection(drawer, 'Engineer, coding');
    expect(within(coding).getAllByText('run_coding_executor')).toHaveLength(2);
  });
});

describe('21 — a write is what its position says it is', () => {
  /** A whole attempt: clone, branch, the pre-coding write, the coding call, its own write. */
  function attemptWithBothWrites() {
    return [
      operation({ operation_type: 'run_linter', stage: 'validation' }),
      operation({ operation_id: 'write-coding', operation_type: 'write_file_changes', stage: 'coding' }),
      operation({ operation_id: 'coding', operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_id: 'write-pre', operation_type: 'write_file_changes', stage: 'coding' }),
      operation({ operation_type: 'create_branch', stage: 'setup' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
  }

  it('files the pre-coding write under setup and the coding call’s own write under coding', async () => {
    // Run 198 FE: the deterministic write the platform makes before the Engineer runs was
    // rendered as "Engineer · coding ✓" one second after create_branch, which reads as code
    // written while dependencies were still installing.
    renderWorkflow(attemptWithBothWrites());
    const drawer = await openDrillIn();

    const setup = stageSection(drawer, 'Engineer, setup');
    expect(within(setup).getByText('write_file_changes')).toBeInTheDocument();
    expect(
      within(setup).getByText(/workspace prepared, before the coding call/),
    ).toBeInTheDocument();

    const coding = stageSection(drawer, 'Engineer, coding');
    expect(within(coding).getAllByText('write_file_changes')).toHaveLength(1);
    expect(within(coding).getByText(/the coding call’s file changes/)).toBeInTheDocument();
  });

  it('reclassifies nothing in a slice whose clone the row budget cut off', async () => {
    // The endpoint serves a bounded window. In a slice that does not start at its clone, the
    // coding call a write followed may simply be missing -- filing that write under setup
    // would be the same misreading in the other direction, so the server's stage stands.
    renderWorkflow(attemptWithBothWrites().slice(0, -1));
    const drawer = await openDrillIn();

    const coding = stageSection(drawer, 'Engineer, coding');
    expect(within(coding).getAllByText('write_file_changes')).toHaveLength(2);
    expect(within(drawer).queryByText(/workspace prepared/)).not.toBeInTheDocument();
  });
});

describe('22a — every row says how long it took', () => {
  it('counts an overlap once: the implementation write is journaled inside the coding call', () => {
    const [group] = groupByStage(
      [
        operation({
          operation_type: 'run_coding_executor',
          stage: 'coding',
          started_at: '2026-09-02T10:00:00Z',
          completed_at: '2026-09-02T10:05:00Z',
        }),
        operation({
          operation_type: 'write_file_changes',
          stage: 'coding',
          started_at: '2026-09-02T10:04:58Z',
          completed_at: '2026-09-02T10:04:59Z',
        }),
      ],
      { includePending: false },
    );
    // Five minutes of wall clock, not five minutes and the nested second twice over.
    expect(stageElapsed(group!, Date.parse('2026-09-02T10:06:00Z'))).toBe('5m 0s');
  });

  it('shows no duration for a row the journal never timestamped both ends of', () => {
    const [group] = groupByStage(
      [
        operation({
          operation_type: 'clone_repository',
          stage: 'setup',
          started_at: '2026-09-02T10:00:00Z',
          completed_at: null,
          status: 'failed_terminal',
        }),
      ],
      { includePending: false },
    );
    expect(stageElapsed(group!, Date.parse('2026-09-02T10:06:00Z'))).toBeNull();
  });

  it('renders each row’s elapsed and the stage subtotal from the endpoint’s timestamps', async () => {
    renderWorkflow([
      operation({
        operation_type: 'run_tests',
        stage: 'validation',
        started_at: '2026-09-02T09:40:00Z',
        completed_at: '2026-09-02T09:42:30Z',
      }),
      operation({
        operation_type: 'run_linter',
        stage: 'validation',
        started_at: '2026-09-02T09:39:00Z',
        completed_at: '2026-09-02T09:39:20Z',
      }),
      operation({
        operation_type: 'clone_repository',
        stage: 'setup',
        started_at: '2026-09-02T09:38:00Z',
        completed_at: '2026-09-02T09:38:45Z',
      }),
    ]);
    const drawer = await openDrillIn();

    const validation = stageSection(drawer, 'Engineer, validation');
    expect(within(validation).getByText('20s')).toBeInTheDocument();
    expect(within(validation).getByText('2m 30s')).toBeInTheDocument();
    // The subtotal sits on the stage name: 20s of linting plus 2m 30s of tests.
    expect(
      within(validation)
        .getByRole('heading')
        .textContent?.replace(/\s+/g, ' '),
    ).toContain('2m 50s');
    // A stage with one row: the row's elapsed and the stage's subtotal are the same 45 seconds.
    expect(within(stageSection(drawer, 'Engineer, setup')).getAllByText('45s')).toHaveLength(2);
  });
});

describe('22b — the unjournaled phase is a row, not silence', () => {
  const settled = operation({
    operation_type: 'run_coding_executor',
    stage: 'coding',
    started_at: '2026-09-02T09:40:00Z',
    completed_at: '2026-09-02T09:47:42Z',
  });

  it('says nothing while a journaled operation is running: that row already speaks', () => {
    expect(
      unjournaledPhase(
        [operation({ operation_type: 'clone_repository', stage: 'setup' }), { ...settled, status: 'running', completed_at: null }],
        { childRunning: true },
      ),
    ).toBeNull();
  });

  it('says nothing for a settled workstream, which is quiet because it finished', () => {
    expect(
      unjournaledPhase(
        [operation({ operation_type: 'clone_repository', stage: 'setup' }), settled],
        { childRunning: false },
      ),
    ).toBeNull();
  });

  it('names the phase after a completed coding call, and never a sub-step of it', () => {
    const phase = unjournaledPhase(
      [operation({ operation_type: 'clone_repository', stage: 'setup' }), settled],
      { childRunning: true },
    );
    expect(phase).toMatchObject({ stage: 'coding', label: 'in-attempt checks' });
    expect(phase!.since).toBe('2026-09-02T09:47:42Z');
    // The checks are named as the phase's content, with the reason they have no rows. Nothing
    // claims which of them is running now -- the platform does not record that.
    expect(phase!.detail).toMatch(/journals none of them individually/);
  });

  it('renders it under the coding rows, with its start time and how long it has been going', async () => {
    // Run 197 FE: all ticks, last op ✓ 09:47:42, and seven minutes of in-attempt checks that
    // the drawer drew as silence -- "idk what is running right now".
    const completedAt = new Date(Date.now() - 7 * 60_000).toISOString();
    renderWorkflow([
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        started_at: new Date(Date.now() - 12 * 60_000).toISOString(),
        completed_at: completedAt,
      }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ]);
    const drawer = await openDrillIn();

    const coding = stageSection(drawer, 'Engineer, coding');
    expect(within(coding).getByText('in-attempt checks')).toBeInTheDocument();
    expect(within(coding).getByText(/since .* · 7m \d+s/)).toBeInTheDocument();
    // Derived, so it carries no heartbeat treatment: there is no heartbeat behind it.
    const row = within(coding).getByText('in-attempt checks').closest('.agent-work__row');
    expect(row).toHaveClass('agent-work__row--phase');
    expect(row).not.toHaveClass('agent-work__row--alive');
    // Validation has not been seen; the phase row did not invent rows for the checks it names.
    expect(
      within(stageSection(drawer, 'Engineer, validation')).getByText('not yet seen'),
    ).toBeInTheDocument();
  });

  it('falls back to naming no work at all where the phase is not one it knows', () => {
    const phase = unjournaledPhase(
      [
        operation({ operation_type: 'clone_repository', stage: 'setup' }),
        operation({
          operation_type: 'run_reviewer',
          stage: 'review',
          completed_at: '2026-09-02T09:47:42Z',
        }),
      ],
      { childRunning: true },
    );
    expect(phase).toMatchObject({ stage: 'review', label: 'not journaled' });
    expect(phase!.detail).toBe(
      'the workstream is recorded as running and no operation is journaled',
    );
  });
});

describe('23 — the attempt dropdown', () => {
  /**
   * Run 197 BE, as the endpoint now serves it: three attempts, and only attempt 0 cloned.
   * Attempts 1 and 2 preserved the workspace and never re-cloned, which is exactly the shape
   * the clone heuristic could not divide -- it rendered all three as one CURRENT ATTEMPT.
   */
  function threeAttempts() {
    return [
      // Newest first, as the endpoint answers.
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        status: 'running',
        child_attempt: 2,
        started_at: '2026-09-02T10:30:00Z',
        completed_at: null,
      }),
      operation({
        operation_type: 'run_tests',
        stage: 'validation',
        status: 'failed_terminal',
        child_attempt: 1,
        error_code: 'repository_tests_failed',
        started_at: '2026-09-02T10:20:00Z',
        completed_at: '2026-09-02T10:29:00Z',
      }),
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        child_attempt: 1,
        started_at: '2026-09-02T10:11:00Z',
        completed_at: '2026-09-02T10:20:00Z',
      }),
      operation({
        operation_type: 'run_reviewer',
        stage: 'review',
        child_attempt: 0,
        started_at: '2026-09-02T10:05:00Z',
        completed_at: '2026-09-02T10:10:00Z',
      }),
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        child_attempt: 0,
        started_at: '2026-09-02T10:01:00Z',
        completed_at: '2026-09-02T10:05:00Z',
      }),
      operation({
        operation_type: 'clone_repository',
        stage: 'setup',
        child_attempt: 0,
        started_at: '2026-09-02T10:00:00Z',
        completed_at: '2026-09-02T10:00:30Z',
      }),
    ];
  }

  it('splits the stamped attempts the clone heuristic merged into one', () => {
    const views = attemptViews(threeAttempts());
    expect(views.map((view) => view.number)).toEqual([0, 1, 2]);
    expect(views.map((view) => view.operations.length)).toEqual([3, 2, 1]);
    // Chronological within each attempt, which is the order the write-role rule reads.
    expect(views[0]!.operations.map((item) => item.operation_type)).toEqual([
      'clone_repository',
      'run_coding_executor',
      'run_reviewer',
    ]);
    // The heuristic on the same rows is the defect: one group, three attempts merged.
    expect(partitionAttempts(threeAttempts()).attempts).toHaveLength(1);
  });

  it('reads each marker as number, glyph with its word, duration and failure class', () => {
    const markers = attemptMarkers(attemptViews(threeAttempts()), {
      childRunning: true,
      nowMs: Date.parse('2026-09-02T10:37:00Z'),
    });
    expect(markers.map((marker) => marker.text)).toEqual([
      // Attempt 0's rows all succeeded, but a later attempt followed it -- so no tick, which
      // would read as "this attempt delivered".
      '· Attempt 0 · 9m 30s · a later attempt followed',
      // 8m 60s would be wrong: 10:11→10:20 and 10:20→10:29 abut, and the overlap-once rule
      // measures the 18 minutes of clock they actually cover.
      '✗ Attempt 1 · 18m 0s · failed: repository_tests_failed',
      '● Attempt 2 · running · 7m 0s so far',
    ]);
    expect(markers.map((marker) => marker.state)).toEqual(['other', 'failed', 'running']);
  });

  it('takes the latest attempt’s outcome from the workstream, which is whose it is', () => {
    // Every operation in the attempt succeeded and the workstream is approved: the tick and
    // the word both come from the record, and neither is inferred from the rows.
    const approved = attemptMarkers(attemptViews(threeAttempts().slice(3)), {
      childRunning: false,
      outcome: { status: 'approved', failureClassification: null },
      nowMs: Date.parse('2026-09-02T10:37:00Z'),
    });
    expect(approved.at(-1)!.text).toBe('✓ Attempt 0 · 9m 30s · approved');

    // A workstream that failed with every operation in it succeeded -- an attempt rejected at
    // review. The rows cannot say that; the workstream's own class can, and does.
    const failed = attemptMarkers(attemptViews(threeAttempts().slice(3)), {
      childRunning: false,
      outcome: { status: 'failed', failureClassification: 'review_findings_unresolved' },
      nowMs: Date.parse('2026-09-02T10:37:00Z'),
    });
    expect(failed.at(-1)!.text).toBe('✗ Attempt 0 · 9m 30s · failed: review_findings_unresolved');
  });

  it('collects unstamped rows into one entry and never splits them across the stamped ones', () => {
    const views = attemptViews([
      ...threeAttempts(),
      // Older than everything above and stamped by nothing: the reconnaissance clone, plus a
      // row from a build that predates the stamp.
      operation({ operation_type: 'run_repository_recon', stage: 'planning' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ]);
    expect(views.map((view) => view.number)).toEqual([null, 0, 1, 2]);
    expect(views[0]!.operations).toHaveLength(2);
    const markers = attemptMarkers(views, {
      childRunning: true,
      nowMs: Date.parse('2026-09-02T10:37:00Z'),
    });
    expect(markers[0]!.label).toBe('Earlier history');
  });

  it('renders one attempt at a time, defaulting to the latest, and switches on choosing', async () => {
    renderWorkflow(threeAttempts());
    const drawer = await openDrillIn();

    // Latest by default: attempt 2's running coding row, and nothing from attempt 1 or 0.
    const picker = within(drawer).getByLabelText('Attempt');
    expect((picker as HTMLSelectElement).value).toBe('attempt-2');
    expect(within(stageSection(drawer, 'Engineer, coding')).getAllByText('run_coding_executor'))
      .toHaveLength(1);
    expect(within(drawer).queryByText('run_tests')).not.toBeInTheDocument();
    expect(within(drawer).queryByText('clone_repository')).not.toBeInTheDocument();

    // The dropdown itself is the run history, newest first. The live entry's duration is read
    // against this browser's clock, so it is matched rather than stated -- everything settled
    // comes from the endpoint's own timestamps and is exact.
    const options = within(picker)
      .getAllByRole('option')
      .map((option) => option.textContent);
    expect(options[0]).toMatch(/^● Attempt 2 · running · .+ so far$/);
    expect(options.slice(1)).toEqual([
      '✗ Attempt 1 · 18m 0s · failed: repository_tests_failed',
      '· Attempt 0 · 9m 30s · a later attempt followed',
    ]);

    // Choosing attempt 1 shows attempt 1's rows and only those. No divider drawn anywhere:
    // one attempt per view is the whole point.
    await userEvent.selectOptions(picker, 'attempt-1');
    expect(within(stageSection(drawer, 'Engineer, validation')).getByText('run_tests')).toBeInTheDocument();
    expect(within(drawer).queryByText(/Earlier attempt/)).not.toBeInTheDocument();
    expect(within(drawer).queryByText('Current attempt')).not.toBeInTheDocument();
    // A finished attempt does not advertise the stages it never reached: "not yet seen" there
    // would read as work still to come on an attempt that is over.
    expect(within(drawer).queryByText('not yet seen')).not.toBeInTheDocument();
  });
});

describe('the drill-in reads what the graph already polls', () => {
  it('fetches the journal once for the lane, not once per node', async () => {
    let reads = 0;
    const rows = [operation({ operation_type: 'clone_repository', stage: 'setup' })];
    renderWorkflow(rows, {
      // Settled, so the 55- cadence stops polling and the read count is exact.
      getFeature: async () => ({ ...FEATURE, status: 'completed' }),
      getWorkstreams: async () => ({
        feature_id: FEATURE_ID,
        workstreams: [workstream({ status: 'completed' })],
      }),
      listWorkstreamOperations: async (featureId: string, repositoryId: string) => {
        reads += 1;
        return {
          feature_id: featureId,
          repository_id: repositoryId,
          operations: rows,
          attempts: [],
        };
      },
    });
    const drawer = await openDrillIn();
    await waitFor(() => expect(reads).toBe(1));

    // Opening a second node in the same lane shows the same journal without another read.
    await userEvent.click(within(drawer).getByRole('button', { name: 'Close' }));
    await openDrillIn(/^Validation, Validation, admanager-server/);
    expect(reads).toBe(1);
  });
});

describe('the served per-attempt block (65 A.2)', () => {
  /**
   * One captured `GET /features/{id}/workstreams/{repo}/operations` from AB-Feature-201's
   * backend workstream — ten finished attempts, eleven endings, three of the four workspace
   * values and both self-review outcomes the run produced. Captured through the real router
   * and response model by `server/tests/capture_operations_fixture.py`, because a
   * hand-written fixture is how this client silently lost a dozen fields once.
   *
   * The rendering tests above build rows inline, which is right: they are about words on a
   * screen. This one is about the shape, which is the part a hand-written file gets wrong.
   */
  it('parses through the production schema with every ending field the server sends', () => {
    const parsed = workstreamOperationsSchema.parse(live201Backend);

    expect(parsed.repository_id).toBe('admanager_console-2.0');
    expect(parsed.operations.length).toBeGreaterThan(50);
    // Oldest attempt first, one entry per finished attempt, and the in-flight attempt absent.
    const attempts = parsed.attempts.map((item) => item.attempt);
    expect(attempts).toEqual([...attempts].sort((left, right) => left - right));

    const byAttempt = new Map(parsed.attempts.map((item) => [item.attempt, item]));
    // Attempt 0 is the case this whole item exists for: stopped at the self-review gate,
    // with the ✗ belonging on `review` and no `run_reviewer` row anywhere beneath it.
    expect(byAttempt.get(0)).toMatchObject({
      ended_by: 'self_review',
      stage: 'review',
      workspace: 'fresh_checkout',
      self_review_outcome: 'corrections_failed',
    });
    expect(byAttempt.get(0)!.detail).toContain('self-review corrections failed');
    expect(
      parsed.operations.some(
        (item) => item.child_attempt === 0 && item.operation_type === 'run_reviewer',
      ),
    ).toBe(false);

    // Attempt 2 is `approved` and still carries `review_scope_failure`. Its ending names
    // neither a failure nor a classification: a later attempt reopened it.
    expect(byAttempt.get(2)).toMatchObject({ ended_by: 'superseded', stage: 'publication' });

    // The workspace value travels verbatim, so "carried over" has a fact to rest on.
    expect(new Set(parsed.attempts.map((item) => item.workspace))).toEqual(
      new Set(['fresh_checkout', 'preserved', 'recovered_coding_output']),
    );
    // Every entry carries the words its glyph will stand for.
    for (const attempt of parsed.attempts) {
      expect(attempt.detail).toBeTruthy();
      expect(attempt.ended_by).toBeTruthy();
      expect(attempt.stage).toBeTruthy();
    }
  });

  it('reads a response from a server that has never heard of the block', () => {
    // A deployment serving this client ahead of the server. The absence has a meaning this
    // client already handles — no ending recorded — and must not fail the whole journal.
    const withoutBlock = Object.fromEntries(
      Object.entries(live201Backend as Record<string, unknown>).filter(
        ([key]) => key !== 'attempts',
      ),
    );

    expect(workstreamOperationsSchema.parse(withoutBlock).attempts).toEqual([]);
  });
});

/**
 * Where a finished attempt stopped, and what never ran (65 A.1).
 *
 * Four of these would otherwise ship broken while everything else passed. A1 is the case the
 * whole item exists for: the ✗ has to land on `review` from a self-review ending with no
 * `run_reviewer` row anywhere to hang it on. A1c is its mirror: an `approved` attempt that
 * still carries a classification must take no ✗ at all — 201 has four of those. A8 is the
 * publication rows that read as shipping when nothing shipped. A9's second half guards a lie
 * the platform can tell rather than a shape anybody has seen.
 */
describe('A1 — a self-review ending puts the ✗ on review with no review row beneath it', () => {
  it('renders setup/coding/validation ✓, review ✗ with the finding named, publication never ran', async () => {
    // 201 backend attempt 0, as the records hold it: clone + branch + install, the baseline
    // commands, the coding call and its write — every row succeeded — and no run_reviewer row
    // at all, because the self-review gate stopped the attempt before the reviewer was called.
    const rows = [
      operation({ operation_type: 'write_file_changes', stage: 'coding', child_attempt: 0 }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 0 }),
      operation({ operation_type: 'run_build', stage: 'validation', child_attempt: 0 }),
      operation({ operation_type: 'run_tests', stage: 'validation', child_attempt: 0 }),
      operation({ operation_type: 'run_linter', stage: 'validation', child_attempt: 0 }),
      operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 0 }),
      operation({ operation_type: 'create_branch', stage: 'setup', child_attempt: 0 }),
      operation({ operation_type: 'clone_repository', stage: 'setup', child_attempt: 0 }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 1 }),
    ];
    renderWorkflow(rows, {}, [
      ending({
        attempt: 0,
        ended_by: 'self_review',
        stage: 'review',
        detail: 'self-review corrections failed against 1 finding (FR-009, localized)',
        workspace: 'fresh_checkout',
        self_review_outcome: 'corrections_failed',
        source_repair_passes: 0,
        stream_reissues: null,
      }),
    ]);
    const drawer = await openDrillIn();
    await userEvent.selectOptions(
      within(drawer).getByLabelText('Attempt'),
      within(drawer).getByRole('option', { name: /Attempt 0/ }).getAttribute('value')!,
    );

    // No reviewer row exists, so nothing in the journal could have carried this cross.
    expect(within(drawer).queryByText('run_reviewer')).not.toBeInTheDocument();

    const review = stageSection(drawer, 'Reviewer, review');
    expect(review).toHaveClass('agent-work__stage--failed');
    expect(
      within(review).getByText(/self-review corrections failed against 1 finding \(FR-009, localized\)/),
    ).toBeInTheDocument();
    // The word, not only the glyph: the state has to be readable without the symbol.
    expect(within(review).getByText(/stopped at the self-review gate/)).toBeInTheDocument();

    // The stages that ran, and the one that never did.
    expect(stageSection(drawer, 'Engineer, setup')).toHaveClass('agent-work__stage--succeeded');
    expect(stageSection(drawer, 'Engineer, coding')).toHaveClass('agent-work__stage--succeeded');
    expect(stageSection(drawer, 'Engineer, validation')).toHaveClass(
      'agent-work__stage--succeeded',
    );
    const publication = stageSection(drawer, 'Publisher, publication');
    expect(publication).toHaveClass('agent-work__stage--skipped');
    expect(within(publication).getByText('never ran')).toBeInTheDocument();
    expect(within(publication).queryByText('not yet seen')).not.toBeInTheDocument();

    // The dropdown marker names the ending rather than "a later attempt followed".
    const marker = within(drawer).getByRole('option', { name: /Attempt 0/ });
    expect(marker).toHaveTextContent('✗');
    expect(marker).toHaveTextContent('self-review corrections failed against 1 finding');
    expect(marker).not.toHaveTextContent('a later attempt followed');
  });

  it('A6 — the stepper reads as a list, every glyph with its word', async () => {
    renderWorkflow(
      [operation({ operation_type: 'clone_repository', stage: 'setup', child_attempt: 0 })],
      {},
      [ending({ attempt: 0, ended_by: 'fault', stage: 'coding', detail: 'classified as implementation missing' })],
    );
    const drawer = await openDrillIn();
    const stepper = within(drawer).getByRole('list', { name: 'stage lifecycle' });

    expect(within(stepper).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
      '✓setupevery step succeeded',
      '✗codingthe platform classified a fault',
      '○validationnever ran',
      '○reviewnever ran',
      '○publicationnever ran',
    ]);
  });
});

describe('A1b — a real review rejection keeps the reviewer row ticked', () => {
  it('crosses the stage and leaves the row ✓: the call answered, the answer was a rejection', async () => {
    const rows = [
      operation({ operation_type: 'run_reviewer', stage: 'review', child_attempt: 1 }),
      operation({ operation_type: 'run_tests', stage: 'validation', child_attempt: 1 }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 1 }),
      operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 1 }),
    ];
    renderWorkflow(rows, {}, [
      ending({
        attempt: 1,
        ended_by: 'review_rejected',
        stage: 'review',
        detail:
          'review returned changes requested with 4 findings, blocking on bulk-validation-schema-duplication',
      }),
    ]);
    const drawer = await openDrillIn();

    const review = stageSection(drawer, 'Reviewer, review');
    // The contradiction that is not one, asserted in one place so neither half can drift.
    expect(review).toHaveClass('agent-work__stage--failed');
    expect(within(review).getByText('run_reviewer').closest('.agent-work__row')).toHaveClass(
      'agent-work__row--succeeded',
    );
    expect(within(review).getByText(/blocking on bulk-validation-schema-duplication/)).toBeInTheDocument();
  });
});

describe('A1c — an approved attempt carrying a classification takes no ✗ anywhere', () => {
  it('reads as superseded, with the publication rows named as what they are (A8)', async () => {
    // 201 backend attempt 2: every row ✓ including run_reviewer, create_commit and
    // push_branch — and a later attempt still followed, because integration review asked for
    // more. The classification it still carries is not its ending.
    const rows = [
      // Attempt 3's first row, so the picker renders and its marker can be read.
      operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 3 }),
      operation({ operation_type: 'push_branch', stage: 'publication', child_attempt: 2 }),
      operation({ operation_type: 'create_commit', stage: 'publication', child_attempt: 2 }),
      operation({ operation_type: 'run_reviewer', stage: 'review', child_attempt: 2 }),
      operation({ operation_type: 'run_tests', stage: 'validation', child_attempt: 2 }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 2 }),
      operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 2 }),
    ];
    renderWorkflow(rows, {}, [
      ending({
        attempt: 2,
        ended_by: 'superseded',
        stage: 'publication',
        detail: 'passed its own review and was reopened by a later attempt',
      }),
    ]);
    const drawer = await openDrillIn();
    await userEvent.selectOptions(
      within(drawer).getByLabelText('Attempt'),
      within(drawer).getByRole('option', { name: /Attempt 2/ }).getAttribute('value')!,
    );

    // Not one cross in the whole attempt.
    for (const stage of [
      'Engineer, setup',
      'Engineer, coding',
      'Engineer, validation',
      'Reviewer, review',
      'Publisher, publication',
    ]) {
      expect(stageSection(drawer, stage)).not.toHaveClass('agent-work__stage--failed');
    }
    const publication = stageSection(drawer, 'Publisher, publication');
    // The rows plainly exist and are not greyed; what they are is said in words, because
    // "publication ✓" over a commit and a push claims an acceptance that never happened.
    expect(within(publication).getByText('create_commit')).toBeInTheDocument();
    expect(within(publication).getByText(/work committed to the branch/)).toBeInTheDocument();
    expect(within(publication).queryByText(/published/)).not.toBeInTheDocument();
    // The marker takes the `other` glyph and the word, never ✗ and never a tick.
    const marker = within(drawer).getByRole('option', { name: /Attempt 2/ });
    expect(marker).toHaveTextContent('·');
    expect(marker).toHaveTextContent('reopened by a later attempt');
  });
});

describe('A2/A3 — the endings that grey everything after them', () => {
  it('A2 — a validation ending names the command and greys review and publication', async () => {
    const rows = [
      operation({ operation_type: 'run_tests', stage: 'validation', status: 'failed_terminal', child_attempt: 5 }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 5 }),
      operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 5 }),
    ];
    renderWorkflow(rows, {}, [
      ending({
        attempt: 5,
        ended_by: 'validation_failed',
        stage: 'validation',
        detail: 'the required test command failed: npm run test (exit 1)',
      }),
    ]);
    const drawer = await openDrillIn();

    const validation = stageSection(drawer, 'Engineer, validation');
    expect(validation).toHaveClass('agent-work__stage--failed');
    expect(within(validation).getByText(/npm run test \(exit 1\)/)).toBeInTheDocument();
    for (const stage of ['Reviewer, review', 'Publisher, publication']) {
      expect(within(stageSection(drawer, stage)).getByText('never ran')).toBeInTheDocument();
    }
  });

  it('A3 — a refusal ending stops at coding, and nothing after it says anything else', async () => {
    renderWorkflow(
      [operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 4 })],
      {},
      [
        ending({
          attempt: 4,
          ended_by: 'refusal',
          stage: 'coding',
          detail: 'The Engineer refused the attempt: two required context files were dropped.',
        }),
      ],
    );
    const drawer = await openDrillIn();

    const coding = stageSection(drawer, 'Engineer, coding');
    expect(coding).toHaveClass('agent-work__stage--failed');
    expect(within(coding).getByText(/two required context files were dropped/)).toBeInTheDocument();
    for (const stage of ['Engineer, validation', 'Reviewer, review', 'Publisher, publication']) {
      const section = stageSection(drawer, stage);
      expect(within(section).getByText('never ran')).toBeInTheDocument();
      expect(within(section).queryByText('not yet seen')).not.toBeInTheDocument();
      expect(within(section).queryByText('not recorded')).not.toBeInTheDocument();
    }
  });
});

describe('A5 — an attempt with no served ending claims nothing', () => {
  it('renders the scaffold reading "not recorded", and falls back to the outcome marker', async () => {
    // An old stamped attempt from a deployment that predates the block: the scaffold renders,
    // but the platform has no ending to anchor "never ran" against, so it does not claim one.
    const rows = [
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 0 }),
      operation({ operation_type: 'clone_repository', stage: 'setup', child_attempt: 0 }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 1 }),
    ];
    renderWorkflow(rows, {}, []);
    const drawer = await openDrillIn();
    await userEvent.selectOptions(
      within(drawer).getByLabelText('Attempt'),
      within(drawer).getByRole('option', { name: /Attempt 0/ }).getAttribute('value')!,
    );

    for (const stage of ['Engineer, validation', 'Reviewer, review', 'Publisher, publication']) {
      const section = stageSection(drawer, stage);
      expect(within(section).getByText('not recorded')).toBeInTheDocument();
      expect(within(section).queryByText('never ran')).not.toBeInTheDocument();
    }
    // Today's fallback text, unchanged: nothing here invents an ending.
    expect(within(drawer).getByRole('option', { name: /Attempt 0/ })).toHaveTextContent(
      'a later attempt followed',
    );
  });

  it('leaves the unstamped merged history exactly as it renders today', async () => {
    // Every row unstamped: the gate is `isLatest || view.number !== null`, so this view keeps
    // the scaffold it has today — and gets none of the new state words, because nothing here
    // can say which attempt these rows belong to.
    const rows = [
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    renderWorkflow(rows, {}, [ending({ attempt: 0, ended_by: 'fault', stage: 'coding' })]);
    const drawer = await openDrillIn();

    // No picker at all: one view, nothing to choose between — byte-identical to today.
    expect(within(drawer).queryByLabelText('Attempt')).not.toBeInTheDocument();
    expect(
      within(stageSection(drawer, 'Reviewer, review')).getByText('not yet seen'),
    ).toBeInTheDocument();
    expect(stageSection(drawer, 'Engineer, coding')).not.toHaveClass('agent-work__stage--failed');
  });
});

describe('A9 — the ordinary retry, and the carried-over exception', () => {
  it('a preserved retry renders its one install row and claims nothing about carrying over', async () => {
    // The shape 201 actually has, nine times over: every backend retry journals exactly one
    // `install_dependencies` row and no clone. An earlier draft of this design called that
    // stage "carried over"; the journal disproves it.
    const rows = [
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 1 }),
      operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 1 }),
    ];
    renderWorkflow(rows, {}, [
      ending({ attempt: 1, ended_by: 'review_rejected', stage: 'review', workspace: 'preserved' }),
    ]);
    const drawer = await openDrillIn();

    const setup = stageSection(drawer, 'Engineer, setup');
    expect(within(setup).getByText('install_dependencies')).toBeInTheDocument();
    expect(within(setup).queryByText(/carried over/)).not.toBeInTheDocument();
    expect(setup).toHaveClass('agent-work__stage--succeeded');
  });

  it('CONSTRUCTED — an empty coding stage on a recovered_coding_output attempt reads "carried over"', async () => {
    // Deliberately constructed rather than replayed: it guards a lie the platform can tell.
    // The coding output was recovered rather than regenerated, so `coding` can be empty on an
    // attempt that demonstrably produced code — and "never ran" about that is the lie.
    renderWorkflow(
      [operation({ operation_type: 'run_reviewer', stage: 'review', child_attempt: 3 })],
      {},
      [
        ending({
          attempt: 3,
          ended_by: 'review_rejected',
          stage: 'review',
          workspace: 'recovered_coding_output',
        }),
      ],
    );
    const drawer = await openDrillIn();

    for (const stage of ['Engineer, setup', 'Engineer, coding']) {
      const section = stageSection(drawer, stage);
      expect(within(section).getByText('carried over from the previous attempt')).toBeInTheDocument();
      expect(within(section).queryByText('never ran')).not.toBeInTheDocument();
    }
    // And the precedence holds in the other direction: a stage after the ending is still
    // "never ran", because the workspace inherited coding and setup and nothing else.
    expect(
      within(stageSection(drawer, 'Publisher, publication')).getByText('never ran'),
    ).toBeInTheDocument();
  });
});

describe('A12 — unknown stages and unknown types survive the new states', () => {
  it('renders a stage name and an operation type this client has never heard of', async () => {
    const rows = [
      operation({ operation_type: 'run_quantum_gate', stage: 'quantum', child_attempt: 0 }),
      operation({ operation_type: 'clone_repository', stage: 'setup', child_attempt: 0 }),
    ];
    renderWorkflow(rows, {}, [
      ending({ attempt: 0, ended_by: 'fault', stage: 'quantum', detail: 'classified as a new thing' }),
    ]);
    const drawer = await openDrillIn();

    // The row renders as itself, under the stage the server assigned.
    const unknown = stageSection(drawer, 'quantum');
    expect(within(unknown).getByText('run_quantum_gate')).toBeInTheDocument();
    expect(within(unknown).getByText(/classified as a new thing/)).toBeInTheDocument();
    // A stage the order has never heard of is never called "after" anything, so the known
    // stages read `not recorded` rather than being greyed on a guess about where it sits.
    expect(
      within(stageSection(drawer, 'Reviewer, review')).getByText('not recorded'),
    ).toBeInTheDocument();
  });

  it('takes no ✗ from an ended_by value it does not know', async () => {
    // A later server's new ending kind. A cross this client cannot justify looks exactly like
    // one that was earned, so it renders the served name and the served sentence instead.
    renderWorkflow(
      [operation({ operation_type: 'clone_repository', stage: 'setup', child_attempt: 0 })],
      {},
      [
        ending({
          attempt: 0,
          ended_by: 'quarantined_by_policy',
          stage: 'review',
          detail: 'held by a policy this client has never heard of',
        }),
      ],
    );
    const drawer = await openDrillIn();

    const review = stageSection(drawer, 'Reviewer, review');
    expect(review).not.toHaveClass('agent-work__stage--failed');
    expect(within(review).getByText('Quarantined by policy')).toBeInTheDocument();
    expect(within(review).getByText(/held by a policy/)).toBeInTheDocument();
  });
});

describe('A7 — a repeated row says why it repeats', () => {
  it('names the re-run and the sibling command, and never reads as an unexplained duplicate', async () => {
    // 201 backend attempt 3's shape: the linter ran at the revision the attempt started from
    // and again after an in-attempt repair changed the tree, and two test commands ran at one
    // revision — the change's own scoped test beside the full suite.
    const rows = [
      operation({
        operation_type: 'run_tests',
        stage: 'validation',
        child_attempt: 3,
        repeat: { kind: 'different_command', detail: '897d0436' },
      }),
      operation({
        operation_type: 'run_tests',
        stage: 'validation',
        child_attempt: 3,
        repeat: { kind: 'different_command', detail: '890b8bbf' },
      }),
      operation({
        operation_type: 'run_linter',
        stage: 'validation',
        child_attempt: 3,
        repeat: { kind: 'new_revision', detail: '9e773a4d' },
      }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 3 }),
      operation({ operation_type: 'run_linter', stage: 'validation', child_attempt: 3 }),
      operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 3 }),
    ];
    renderWorkflow(rows);
    const drawer = await openDrillIn();
    const validation = stageSection(drawer, 'Engineer, validation');

    // The second linter run says what changed; the first says nothing, because it is not a
    // re-run of anything.
    expect(
      within(validation).getByText(/re-run at new revision 9e773a4d \(after an in-attempt repair\)/),
    ).toBeInTheDocument();
    expect(within(validation).getAllByText('run_linter')).toHaveLength(2);
    // Both test rows are told apart by their own command, and neither is called a re-run.
    expect(within(validation).getByText(/a different command · 897d0436/)).toBeInTheDocument();
    expect(within(validation).getByText(/a different command · 890b8bbf/)).toBeInTheDocument();
    expect(within(validation).queryByText(/2nd|second run/)).not.toBeInTheDocument();
  });

  it('says only that a row is another run of the same step where nothing distinguishes it', async () => {
    // Coding and git rows carry flat steps and neither column, so the server names no reason.
    const rows = [
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        child_attempt: 2,
        repeat: { kind: 'same_step', detail: null },
      }),
      operation({
        operation_type: 'run_coding_executor',
        stage: 'coding',
        child_attempt: 2,
        repeat: { kind: 'same_step', detail: null },
      }),
    ];
    renderWorkflow(rows);
    const drawer = await openDrillIn();

    expect(
      within(stageSection(drawer, 'Engineer, coding')).getAllByText(
        /another run of the same step/,
      ),
    ).toHaveLength(2);
  });

  it('lets the write role answer instead, where the role is the more specific answer', async () => {
    // 201 attempt 0 really does hold two `write_file_changes` rows, and the server marks them
    // `same_step` because neither carries a distinguishing column. They are two callers, not
    // two runs of one step, and the position already says which is which — so the vaguer
    // answer is withheld rather than printed beside the specific one.
    const rows = [
      operation({
        operation_type: 'write_file_changes',
        stage: 'coding',
        child_attempt: 0,
        repeat: { kind: 'same_step', detail: null },
      }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 0 }),
      operation({
        operation_type: 'write_file_changes',
        stage: 'coding',
        child_attempt: 0,
        repeat: { kind: 'same_step', detail: null },
      }),
      operation({ operation_type: 'clone_repository', stage: 'setup', child_attempt: 0 }),
    ];
    renderWorkflow(rows);
    const drawer = await openDrillIn();

    expect(
      within(drawer).getByText(/workspace prepared, before the coding call/),
    ).toBeInTheDocument();
    expect(within(drawer).getByText(/the coding call’s file changes/)).toBeInTheDocument();
    expect(within(drawer).queryByText(/another run of the same step/)).not.toBeInTheDocument();
  });
});

/**
 * The spec's own live check, run against the records rather than against a screenshot.
 *
 * Its Verification section asks for the drawer to be opened on run 201's backend attempt 0
 * and for the ✗ to land on the stage its ending record actually names — read from the
 * artifacts first and asserted against them, not against an assumption about where it
 * stopped. This does exactly that: the payload is the captured response of the real endpoint
 * over 201's own records, rendered by the real drawer.
 */
describe('the 201 replay, from the captured payload', () => {
  function renderCaptured() {
    const payload = workstreamOperationsSchema.parse(live201Backend);
    return renderWorkflow(payload.operations, {}, payload.attempts);
  }

  async function chooseAttempt(drawer: HTMLElement, number: number) {
    await userEvent.selectOptions(
      within(drawer).getByLabelText('Attempt'),
      within(drawer).getByRole('option', { name: new RegExp(`Attempt ${number} `) }).getAttribute('value')!,
    );
  }

  it('attempt 0: the ✗ is on review, with no run_reviewer row beneath it', async () => {
    const drawer = await (renderCaptured(), openDrillIn());
    await chooseAttempt(drawer, 0);

    // Every journaled row of this attempt succeeded — clone, branch, install, the three
    // baseline commands, the coding call and its writes — which is the wall of green the
    // drawer used to draw and stop at.
    expect(within(drawer).getAllByText('succeeded').length).toBeGreaterThan(5);
    expect(within(drawer).queryByText('run_reviewer')).not.toBeInTheDocument();

    const review = stageSection(drawer, 'Reviewer, review');
    expect(review).toHaveClass('agent-work__stage--failed');
    expect(within(review).getByText(/self-review corrections failed/)).toBeInTheDocument();
    // Not a review rejection, which is what the spec's own author believed until the records
    // said otherwise — and which the drawer's silence is what caused.
    expect(within(review).queryByText(/review returned/)).not.toBeInTheDocument();
    expect(
      within(stageSection(drawer, 'Publisher, publication')).getByText('never ran'),
    ).toBeInTheDocument();
  });

  it('attempt 1: a preserved workspace, one install row, and nothing carried over', async () => {
    const drawer = await (renderCaptured(), openDrillIn());
    await chooseAttempt(drawer, 1);

    const setup = stageSection(drawer, 'Engineer, setup');
    expect(within(setup).getAllByText('install_dependencies')).toHaveLength(1);
    expect(within(setup).queryByText('clone_repository')).not.toBeInTheDocument();
    // The shape 201 has nine times over. An earlier draft called this "carried over"; the
    // journal disproves it, and the stage renders the row it actually holds.
    expect(within(drawer).queryByText(/carried over/)).not.toBeInTheDocument();
    // Its two linter runs are told apart rather than reading as a duplicate.
    const validation = stageSection(drawer, 'Engineer, validation');
    expect(within(validation).getByText(/re-run at new revision/)).toBeInTheDocument();
  });

  it('attempt 2: approved while still carrying a classification, and no ✗ anywhere', async () => {
    const drawer = await (renderCaptured(), openDrillIn());
    await chooseAttempt(drawer, 2);

    for (const stage of [
      'Engineer, setup',
      'Engineer, coding',
      'Engineer, validation',
      'Reviewer, review',
      'Publisher, publication',
    ]) {
      expect(stageSection(drawer, stage)).not.toHaveClass('agent-work__stage--failed');
    }
    // Its commit and push are named for what they are, because this attempt did not ship.
    expect(
      within(stageSection(drawer, 'Publisher, publication')).getByText(/work committed to the branch/),
    ).toBeInTheDocument();
  });
});

describe('the planning nodes list their journaled calls', () => {
  /**
   * One planning-call execution as `/executions` serves it: `execution_records.py` derives
   * these from the journal rows 41- Part A introduced, stamps the `planning_call:` prefix,
   * and carries the journal's `logical_step` as `agent_type`.
   */
  function planningCall(
    overrides: Partial<ExecutionRecord> & { execution_id: string },
  ): ExecutionRecord {
    return {
      from_stage: 'request',
      to_stage: 'technical_prd',
      repository_id: null,
      is_retry: false,
      handler_type: 'model',
      handler: 'Product manager',
      agent_type: 'product_manager',
      model_resolved: false,
      status: 'completed',
      command: [],
      execution_mode: 'live',
      started_at: '2026-08-24T22:41:00Z',
      completed_at: '2026-08-24T22:41:39Z',
      duration_seconds: 39,
      ...overrides,
    } as ExecutionRecord;
  }

  const NOW = Date.parse('2026-08-24T23:00:00Z');

  it('routes each call to the node its to_stage names, oldest first, and drops the rest', () => {
    const lines = planningNodeOperations(
      [
        // Served newest first, like every other list; the planner started last.
        planningCall({
          execution_id: 'planning_call:planner',
          to_stage: 'integration_contract',
          handler: 'Technical planner',
          agent_type: 'feature_planner',
          started_at: '2026-08-24T22:45:00Z',
        }),
        planningCall({
          execution_id: 'planning_call:recon',
          to_stage: 'integration_contract',
          handler: 'Repository reconnaissance',
          agent_type: 'repository_reconnaissance',
          repository_id: 'admanager-server',
          started_at: '2026-08-24T22:42:00Z',
        }),
        planningCall({ execution_id: 'planning_call:pm' }),
        // Artifact-derived executions carry no planning prefix and render on no node.
        planningCall({ execution_id: 'implementation:admanager-server:0' }),
        // A stage this client has never heard of renders nowhere rather than being guessed.
        planningCall({ execution_id: 'planning_call:odd', to_stage: 'quantum_stage' }),
      ],
      { nowMs: NOW },
    );
    expect(lines.get('technical_prd')?.map((line) => line.name)).toEqual(['product_manager']);
    expect(lines.get('integration_contract')?.map((line) => line.name)).toEqual([
      'repository_reconnaissance',
      'feature_planner',
    ]);
    expect([...lines.keys()]).toEqual(['technical_prd', 'integration_contract']);
  });

  it('says which repository a reconnaissance call read, beside its duration', () => {
    const lines = planningNodeOperations(
      [
        planningCall({
          execution_id: 'planning_call:recon',
          to_stage: 'integration_contract',
          agent_type: 'repository_reconnaissance',
          repository_id: 'admanager-server',
          duration_seconds: 154,
        }),
      ],
      { nowMs: NOW },
    );
    const line = lines.get('integration_contract')?.[0];
    expect(line?.meta).toBe('admanager-server · 2m 34s');
    expect(line?.state).toBe('succeeded');
    expect(line?.word).toBe('succeeded');
  });

  it('names the model a call recorded, and never invents one for a call that did not', () => {
    const lines = planningNodeOperations(
      [
        planningCall({
          execution_id: 'planning_call:pm',
          model: 'gpt-5.6-terra',
          duration_seconds: 39,
        }),
        planningCall({
          execution_id: 'planning_call:recon',
          to_stage: 'integration_contract',
          agent_type: 'repository_reconnaissance',
          repository_id: 'admanager-server',
          model: 'claude-opus-5',
          duration_seconds: 154,
        }),
        // An older row, a mock run, a deterministic composition: no model was recorded, and
        // the line says nothing rather than reconstructing one from today's configuration.
        planningCall({
          execution_id: 'planning_call:reconcile',
          agent_type: 'requirement_reconciliation',
          duration_seconds: 40,
          started_at: '2026-08-24T22:47:00Z',
        }),
      ],
      { nowMs: NOW },
    );
    const [manager, reconcile] = lines.get('technical_prd') ?? [];
    expect(manager?.meta).toBe('GPT-5.6 Terra · 39s');
    expect(reconcile?.meta).toBe('40s');
    expect(lines.get('integration_contract')?.[0]?.meta).toBe(
      'admanager-server · Claude Opus 5 · 2m 34s',
    );
  });

  it('shows a running call with its elapsed so far, and a failed one without inventing one', () => {
    const lines = planningNodeOperations(
      [
        planningCall({
          execution_id: 'planning_call:pm',
          status: 'running',
          started_at: '2026-08-24T22:53:00Z',
          completed_at: null,
          duration_seconds: null,
        }),
        planningCall({
          execution_id: 'planning_call:reconcile',
          agent_type: 'requirement_reconciliation',
          status: 'failed',
          started_at: '2026-08-24T22:55:00Z',
          completed_at: null,
          duration_seconds: null,
        }),
      ],
      { nowMs: NOW },
    );
    const [running, failed] = lines.get('technical_prd') ?? [];
    expect(running?.state).toBe('running');
    expect(running?.meta).toBe('7m 0s');
    expect(failed?.state).toBe('failed');
    // A failed row with no completion measured nothing; "now minus start" would invent it.
    expect(failed?.meta).toBeNull();
  });

  it('gives a status it has never heard of no tick and no cross, only its own name', () => {
    const lines = planningNodeOperations(
      [planningCall({ execution_id: 'planning_call:pm', status: 'quantum_hold' })],
      { nowMs: NOW },
    );
    const line = lines.get('technical_prd')?.[0];
    expect(line?.state).toBe('other');
    expect(line?.word).toBe('Quantum hold');
  });
});
