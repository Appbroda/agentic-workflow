// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { buildFeatureGraph, planView } from '@/features/feature-workspace/graph';
import {
  DEFAULT_HEARTBEAT_THRESHOLD_SECONDS,
  heartbeatThresholdSeconds,
  livenessByNode,
  livenessForOperation,
  runningOperation,
} from '@/features/feature-workspace/operations';
import { WorkflowGraph } from '@/features/feature-workspace/WorkflowGraph';
import {
  featureSchema,
  workstreamSchema,
  type WorkstreamOperation,
} from '@/schemas/feature';

/**
 * The liveness indicator, fed by the external-operation journal.
 *
 * During the 185-194 verification cycle the graph said "Running" for a 51-minute attempt
 * while the journal knew the heartbeat was seconds old, and run 194's frontend edge said
 * "Model pending / Running" eighteen minutes after the coding call had completed while the
 * reviewer was actively working. The indicator reads the journal, not the node states, so
 * it can tell both truths without reworking how node states are derived.
 */

const NOW = Date.parse('2026-09-01T12:00:00Z');

function operation(overrides: Partial<WorkstreamOperation> = {}): WorkstreamOperation {
  return {
    operation_id: 'operation-1',
    operation_type: 'run_coding_executor',
    stage: 'coding',
    status: 'running',
    attempt: 3,
    max_attempts: 3,
    child_attempt: null,
    started_at: '2026-09-01T11:20:00Z',
    heartbeat_at: new Date(NOW - 30_000).toISOString(),
    completed_at: null,
    error_code: null,
    repeat: null,
    stream_reissues: null,
    ...overrides,
  };
}

function agedHeartbeat(seconds: number): string {
  return new Date(NOW - seconds * 1000).toISOString();
}

describe('per-operation-type heartbeat thresholds', () => {
  it('gives a coding call and a lint run different definitions of "quiet too long"', () => {
    expect(heartbeatThresholdSeconds('run_linter')).toBeLessThan(
      heartbeatThresholdSeconds('run_coding_executor'),
    );
    // A type the table does not know falls back to the journal's own stale window rather
    // than crashing or borrowing another type's cadence.
    expect(heartbeatThresholdSeconds('some_future_operation')).toBe(
      DEFAULT_HEARTBEAT_THRESHOLD_SECONDS,
    );
  });

  it('renders a running coding call alive, with its type, attempt and heartbeat age', () => {
    const placed = livenessForOperation('backend', operation(), NOW);
    expect(placed).not.toBeNull();
    expect(placed!.nodeId).toBe('implement:backend');
    expect(placed!.liveness.state).toBe('alive');
    expect(placed!.liveness.summary).toBe('coding running — 30s');
    expect(placed!.liveness.detail).toContain('run_coding_executor');
    expect(placed!.liveness.detail).toContain('call 3 of 3');
    expect(placed!.liveness.detail).toContain('heartbeat 30s ago');
  });

  it('flips a coding call to the warning state past the coding threshold', () => {
    const threshold = heartbeatThresholdSeconds('run_coding_executor');
    // Frozen clock, just past the threshold: same operation, different reading.
    const placed = livenessForOperation(
      'backend',
      operation({ heartbeat_at: agedHeartbeat(threshold + 10) }),
      NOW,
    );
    expect(placed!.liveness.state).toBe('stale');
    expect(placed!.liveness.summary).toMatch(/^coding: no heartbeat for /);
  });

  it('flips a lint run at the lint threshold, not the coding one', () => {
    const lintThreshold = heartbeatThresholdSeconds('run_linter');
    const age = lintThreshold + 30;
    const lint = livenessForOperation(
      'backend',
      operation({
        operation_type: 'run_linter',
        stage: 'validation',
        heartbeat_at: agedHeartbeat(age),
      }),
      NOW,
    );
    const coding = livenessForOperation(
      'backend',
      operation({ heartbeat_at: agedHeartbeat(age) }),
      NOW,
    );
    expect(age).toBeLessThan(heartbeatThresholdSeconds('run_coding_executor'));
    expect(lint!.liveness.state).toBe('stale');
    expect(coding!.liveness.state).toBe('alive');
  });

  it('only a live operation carries an indicator', () => {
    expect(
      runningOperation([
        operation({ status: 'succeeded', completed_at: agedHeartbeat(0) }),
        operation({ operation_id: 'operation-0', status: 'failed_terminal' }),
      ]),
    ).toBeNull();
  });
});

describe('the run-194 replay: node states lag, the indicator does not', () => {
  it('reports review with its age while the node states still say implementation', () => {
    const feature = featureSchema.parse({
      feature_id: 'feature-194',
      workflow_id: 'feature-194',
      status: 'running_child_workflows',
      title: 'The 194 replay',
      current_agent: 'engineer',
      repository_count: 1,
      required_repository_count: 1,
      clarification_rounds: 0,
      integration_review_cycles: 0,
      merge_strategy: null,
      deployment_strategy: null,
      execution_mode: 'live',
      cancellation_status: 'not_requested',
      cancellation_requested_at: null,
      cancellation_reason: null,
      created_at: '2026-09-01T10:00:00Z',
      updated_at: '2026-09-01T12:00:00Z',
    });
    // Mid-implementation as far as durable state knows: no completion artifact yet, so the
    // Implementation node is active and the Review node is grey.
    const workstream = workstreamSchema.parse({
      repository_id: 'frontend',
      child_workflow_id: 'feature-194-frontend',
      workstream_id: 'frontend',
      status: 'running',
      branch_name: 'ai/feature-194',
      workspace_path: '/workspaces/frontend',
      retry_count: 0,
      code_completion_artifact_id: null,
      review_artifact_id: null,
      pull_request_artifact_id: null,
    });
    const graph = buildFeatureGraph({
      feature,
      workstreams: [workstream],
      artifactTypes: new Set(['technical_prd', 'integration_contract', 'repository_execution_plan']),
      plan: planView(undefined),
      featureId: 'feature-194',
    });
    const reviewNode = graph.nodes.find((node) => node.id === 'review:frontend');
    expect(reviewNode?.state).toBe('pending');

    // ...while the journal says the reviewer has been running for three minutes.
    const liveness = livenessByNode(
      new Map([
        [
          'frontend',
          [
            operation({
              operation_type: 'run_reviewer',
              stage: 'review',
              heartbeat_at: agedHeartbeat(180),
            }),
          ],
        ],
      ]),
      NOW,
    );

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <WorkflowGraph graph={graph} label="Feature execution graph" liveness={liveness} />
      </MemoryRouter>,
    );

    const indicator = screen.getByRole('img', { name: /review running — 3m 0s/ });
    expect(indicator).toBeInTheDocument();
    // The truthful overlay sits on the Review node the journal names, not on the node the
    // stage statuses claim is running.
    const reviewButton = screen.getByRole('button', { name: /^Review, Review, frontend/ });
    expect(reviewButton).toContainElement(indicator);
  });
});

/**
 * The live edge moves, and everything else holds still (65 E).
 *
 * A person could not see which path a run was currently flowing through at a glance:
 * `.graph__edge--active` existed only on retry edges, and a static treatment among static
 * edges does not read as "flowing".
 *
 * E1 is the test that decides correctness, and the tie-break is why. The record builder emits
 * an in-flight attempt's retry record *alongside* its implementation record on purpose, "so
 * an in-flight attempt has the same pair of arrows a completed one does", and both carry the
 * same status — so "the edge whose newest execution is running" names two. One live edge per
 * lane is an invariant to enforce, not an observation about the data.
 */
describe('E — the live edge', () => {
  function liveFeature(status = 'running_child_workflows') {
    return featureSchema.parse({
      feature_id: 'feature-201',
      workflow_id: 'feature-201',
      status,
      title: 'Allow an admin to bulk add apps',
      current_agent: 'engineer',
      repository_count: 2,
      required_repository_count: 2,
      clarification_rounds: 0,
      integration_review_cycles: 1,
      max_integration_review_cycles: 5,
      max_child_review_cycles: 12,
      merge_strategy: null,
      deployment_strategy: null,
      execution_mode: 'live',
      cancellation_status: 'not_requested',
      cancellation_requested_at: null,
      cancellation_reason: null,
      created_at: '2026-09-03T06:00:00Z',
      updated_at: '2026-09-03T09:00:00Z',
    });
  }

  function lane(repositoryId: string, overrides: Record<string, unknown> = {}) {
    return workstreamSchema.parse({
      repository_id: repositoryId,
      child_workflow_id: `feature-201-${repositoryId}`,
      workstream_id: repositoryId,
      status: 'running',
      branch_name: 'ai/feature-201',
      workspace_path: `/workspaces/${repositoryId}`,
      retry_count: 0,
      code_completion_artifact_id: null,
      review_artifact_id: null,
      pull_request_artifact_id: null,
      ...overrides,
    });
  }

  function record(overrides: Record<string, unknown>) {
    return {
      from_stage: 'repository',
      to_stage: 'implementation',
      is_retry: false,
      handler_type: 'model',
      handler: 'Engineer',
      agent_type: 'Engineer',
      model_resolved: true,
      status: 'running',
      escalated: false,
      command: [],
      execution_mode: 'live',
      ...overrides,
    } as never;
  }

  function build(
    workstreams: ReturnType<typeof lane>[],
    executions: ReturnType<typeof record>[],
    status?: string,
  ) {
    return buildFeatureGraph({
      feature: liveFeature(status),
      workstreams,
      artifactTypes: new Set([
        'technical_prd',
        'integration_contract',
        'repository_execution_plan',
      ]),
      plan: planView(undefined),
      featureId: 'feature-201',
      executions,
    });
  }

  it('E1 — the loop wins the tie-break, and the entry edge beneath it stays static', () => {
    // Mid-remediation: the builder emits both records for the in-flight attempt, both running.
    const graph = build(
      [lane('backend', { retry_count: 3, integration_retry_count: 1 }), lane('frontend')],
      [
        record({
          execution_id: 'implementation:backend:3',
          repository_id: 'backend',
          attempt: 4,
        }),
        record({
          execution_id: 'retry:backend:3',
          repository_id: 'backend',
          from_stage: 'integration_review',
          is_retry: true,
          attempt: 4,
        }),
      ],
    );

    const live = graph.edges.filter((edge) => edge.live);
    expect(live).toHaveLength(1);
    expect(live[0]).toMatchObject({ from: 'integration_review', to: 'implement:backend' });
    // The entry edge carries the same running attempt's implementation record and is not live.
    expect(graph.edges.find((edge) => edge.from === 'repo:backend')?.live).toBeUndefined();
    // And nothing in the other lane moves.
    expect(graph.edges.some((edge) => edge.live && edge.to === 'implement:frontend')).toBe(false);
  });

  it('E1 — a lane on its first attempt animates the entry edge instead', () => {
    const graph = build(
      [lane('backend')],
      [record({ execution_id: 'implementation:backend:0', repository_id: 'backend', attempt: 1 })],
    );

    const live = graph.edges.filter((edge) => edge.live);
    expect(live).toHaveLength(1);
    expect(live[0]).toMatchObject({ from: 'repo:backend', to: 'implement:backend' });
  });

  it('E1 — a queued attempt is live, and a completed one is not', () => {
    const queued = build(
      [lane('backend', { status: 'pending' })],
      [
        record({
          execution_id: 'implementation:backend:0',
          repository_id: 'backend',
          status: 'queued',
        }),
      ],
    );
    const done = build(
      [lane('backend', { status: 'approved' })],
      [
        record({
          execution_id: 'implementation:backend:0',
          repository_id: 'backend',
          status: 'completed',
        }),
      ],
    );

    expect(queued.edges.filter((edge) => edge.live)).toHaveLength(1);
    expect(done.edges.filter((edge) => edge.live)).toHaveLength(0);
  });

  it('E1 — when the child goes terminal, nothing animates', () => {
    // The record's own status is what decides, so a stale `running` record on a stopped
    // feature does not keep an arrow moving.
    const graph = build(
      [lane('backend', { status: 'failed' })],
      [record({ execution_id: 'implementation:backend:0', repository_id: 'backend' })],
      'failed_requires_human',
    );

    expect(graph.edges.filter((edge) => edge.live)).toHaveLength(0);
  });

  it('E2 — the live edge keeps the --active static treatment, on a flow edge', () => {
    const graph = build(
      [lane('backend')],
      [record({ execution_id: 'implementation:backend:0', repository_id: 'backend' })],
    );
    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <WorkflowGraph graph={graph} label="Feature execution graph" />
      </MemoryRouter>,
    );

    const edge = document.querySelector('.graph__edge--live')!;
    // `--active` is the still frame: solid, thicker, toned. The motion is added by `--live`
    // inside the reduced-motion query and is the only thing that disappears when motion is
    // off, so the reduced-motion frame is this class list minus an animation.
    expect(edge).toHaveClass('graph__edge--active');
    // A flow edge, which never received `--active` before this.
    expect(edge).toHaveClass('graph__edge--flow');
    // Distinguishable from a static retry edge in the same frame, which is dashed and thin.
    expect(document.querySelectorAll('.graph__edge--retry.graph__edge--live')).toHaveLength(0);
  });

  it('E3 — a feature waiting on a person animates nothing', () => {
    // The stillness is itself the signal. A child left queued behind a human gate is not
    // motion, and this is the case that would otherwise animate on a `queued` record.
    const graph = build(
      [lane('backend', { status: 'pending' })],
      [
        record({
          execution_id: 'implementation:backend:0',
          repository_id: 'backend',
          status: 'queued',
        }),
      ],
      'waiting_for_human',
    );

    expect(graph.edges.filter((edge) => edge.live)).toHaveLength(0);
  });

  it('E — the restraint: neither of the middle flow edges can ever be live', () => {
    // The `implement→validate` record is completed, failed or pending only, and the
    // `validate→review` record's status is hard-coded completed. Asserted so a later change
    // that made either of them running would fail here rather than quietly animating two
    // edges in one lane.
    const graph = build(
      [lane('backend')],
      [
        record({ execution_id: 'implementation:backend:0', repository_id: 'backend' }),
        record({
          execution_id: 'validation:backend:0',
          repository_id: 'backend',
          from_stage: 'implementation',
          to_stage: 'validation',
          status: 'running',
        }),
      ],
    );

    const live = graph.edges.filter((edge) => edge.live);
    expect(live).toHaveLength(1);
    expect(live[0]!.to).toBe('implement:backend');
  });
});
