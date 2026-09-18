import { describe, expect, it } from 'vitest';
import { buildFeatureGraph, currentActivity, planView } from '@/features/feature-workspace/graph';
import { latestIntegrationReview } from '@/features/feature-workspace/integration-review';
import type { IntegrationReviewOutcome } from '@/features/feature-workspace/integration-review';
import {
  artifactsSchema,
  type ExecutionRecord,
  type Feature,
  type Workstream,
} from '@/schemas/feature';
import live185 from './fixtures/integration-reviews.bulk-apps-live-185.json';

/**
 * The execution graph, as a model.
 *
 * Drawing is one thing; what the drawing claims is another. These are about the claims: that
 * a count is only ever shown against a limit the server published, that a loop appears only
 * where work actually went round one, that parallelism and dependencies come from the
 * planner's own plan, and that the number of repositories is whatever the feature has.
 */

function feature(overrides: Partial<Feature> = {}): Feature {
  return {
    feature_id: 'f-1',
    workflow_id: 'w-1',
    status: 'running_child_workflows',
    title: 'A feature',
    current_agent: 'engineer',
    repository_count: 1,
    required_repository_count: 1,
    repositories: [],
    clarification_rounds: 0,
    integration_review_cycles: 0,
    max_clarification_rounds: 10,
    max_integration_review_cycles: 5,
    max_child_review_cycles: 12,
    max_implementation_retries: 8,
    max_validation_retries: 8,
    max_repository_setup_retries: 1,
    merge_strategy: null,
    deployment_strategy: null,
    execution_mode: 'live',
    cancellation_status: 'not_requested',
    cancellation_requested_at: null,
    cancellation_reason: null,
    cleanup_requirements: [],
    created_at: '2026-08-25T08:00:00Z',
    updated_at: '2026-08-25T09:00:00Z',
    ...overrides,
  } as Feature;
}

function workstream(overrides: Partial<Workstream> & { repository_id: string }): Workstream {
  return {
    repository_name: overrides.repository_id,
    repository_role: 'service',
    child_workflow_id: 'c',
    workstream_id: overrides.repository_id,
    status: 'running',
    branch_name: 'ai/x',
    workspace_path: '/w',
    retry_count: 0,
    code_completion_artifact_id: null,
    review_artifact_id: null,
    blocking_issues: [],
    pull_request_artifact_id: null,
    current_validation_results: [],
    blocking_setup_issues: [],
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
    scoped_requirements: [],
    implementation_expectations: [],
    configured_validation_commands: [],
    out_of_scope_requirements: [],
    planned_blind: false,
    ...overrides,
  } as Workstream;
}

const EMPTY_PLAN = planView(undefined);

function build(
  workstreams: Workstream[],
  overrides: Partial<Feature> = {},
  plan = EMPTY_PLAN,
  artifactTypes = new Set(['technical_prd', 'integration_contract', 'repository_execution_plan']),
  integrationReview: IntegrationReviewOutcome | null = null,
  executions: ExecutionRecord[] = [],
) {
  return buildFeatureGraph({
    feature: feature({ repository_count: workstreams.length, ...overrides }),
    workstreams,
    artifactTypes,
    plan,
    featureId: 'f-1',
    integrationReview,
    executions,
  });
}

/** One retry execution, as the server records it, on whichever transition it names. */
function retryRecord(
  repositoryId: string,
  attempt: number,
  from: 'review' | 'integration_review',
): ExecutionRecord {
  return {
    execution_id: `retry:${repositoryId}:${attempt}`,
    from_stage: from,
    to_stage: 'implementation',
    repository_id: repositoryId,
    is_retry: true,
    handler_type: 'model',
    handler: 'Engineer',
    agent_type: 'Engineer',
    model_resolved: true,
    status: 'completed',
    escalated: false,
    command: [],
    execution_mode: 'live',
    attempt: attempt + 1,
    max_attempts: 12,
  } as ExecutionRecord;
}

describe('the execution graph', () => {
  it.each([1, 2, 5])('draws a lane per repository, for %i of them', (repositories) => {
    const graph = build(
      Array.from({ length: repositories }, (_, index) =>
        workstream({ repository_id: `repo-${index}` }),
      ),
    );

    const lanes = graph.nodes.filter((node) => node.kind === 'repository');
    expect(lanes).toHaveLength(repositories);
    // Every lane is a full chain, and every chain converges on the same two stages.
    for (const lane of lanes) {
      for (const kind of ['implement', 'validate', 'review'] as const) {
        expect(graph.nodes.some((node) => node.kind === kind && node.repositoryId === lane.repositoryId)).toBe(true);
      }
      expect(
        graph.edges.some(
          (edge) => edge.from === `review:${lane.repositoryId}` && edge.to === 'integration_review',
        ),
      ).toBe(true);
    }
    expect(graph.edges.some((edge) => edge.from === 'integration_review' && edge.to === 'pull_requests')).toBe(true);
  });

  it('takes parallelism and dependencies from the planner, not from the roles', () => {
    const plan = planView({
      workstreams: [
        { workstream_id: 'sdk-ws', repository_id: 'shared-sdk', dependency_workstream_ids: [] },
        {
          workstream_id: 'api-ws',
          repository_id: 'backend-api',
          dependency_workstream_ids: ['sdk-ws'],
        },
        {
          workstream_id: 'web-ws',
          repository_id: 'admin-web',
          dependency_workstream_ids: ['sdk-ws'],
        },
      ],
      parallel_groups: [['sdk-ws'], ['api-ws', 'web-ws']],
    });
    const graph = build(
      [
        workstream({ repository_id: 'backend-api' }),
        workstream({ repository_id: 'admin-web' }),
        workstream({ repository_id: 'shared-sdk' }),
      ],
      {},
      plan,
    );

    // The dependency lands in the earlier layer, so it is drawn above what waits for it.
    const row = (id: string) => graph.nodes.find((node) => node.id === `repo:${id}`)!.row;
    expect(row('shared-sdk')).toBeLessThan(row('backend-api'));
    expect(row('shared-sdk')).toBeLessThan(row('admin-web'));
    // The two dependents are independent of each other, and neither has an edge to the other.
    const dependencies = graph.edges.filter((edge) => edge.kind === 'dependency');
    expect(dependencies.map((edge) => [edge.from, edge.to])).toEqual([
      ['repo:shared-sdk', 'repo:backend-api'],
      ['repo:shared-sdk', 'repo:admin-web'],
    ]);
  });

  it('draws the retry loop only where work actually went round one', () => {
    const graph = build([
      workstream({ repository_id: 'first', retry_count: 0 }),
      workstream({ repository_id: 'second', retry_count: 3 }),
    ]);

    const loops = graph.edges.filter((edge) => edge.kind === 'retry');
    expect(loops).toHaveLength(1);
    expect(loops[0]).toMatchObject({
      from: 'review:second',
      to: 'implement:second',
      label: 'Retry 4 / 12',
    });
  });

  it('shows a backend no-progress refusal as needs attention', () => {
    const reason =
      'The previous attempt produced no meaningful production change, so repeating it would submit identical inputs.';
    const graph = build(
      [
        workstream({
          repository_id: 'api',
          status: 'failed',
          retry_count: 1,
          review_artifact_id: 'review-r1',
          retry_refusal_reason: reason,
        }),
      ],
      { status: 'failed_requires_human', max_child_review_cycles: 5 },
    );

    expect(graph.nodes.find((node) => node.id === 'repo:api')).toMatchObject({
      state: 'attention',
      detail: reason,
      counter: { label: 'Attempt', current: 2, limit: 5 },
    });
    expect(currentActivity(graph)?.id).toBe('repo:api');
    expect(graph.nodes.find((node) => node.id === 'review:api')?.state).toBe('attention');
  });

  it('shows the exact exhausted attempt count from backend retry state', () => {
    const reason = 'The child review cycle limit of 5 is reached.';
    const graph = build(
      [
        workstream({
          repository_id: 'api',
          status: 'failed',
          retry_count: 4,
          review_artifact_id: 'review-r5',
          retry_refusal_reason: reason,
        }),
      ],
      { status: 'failed_requires_human', max_child_review_cycles: 5 },
    );

    expect(graph.nodes.find((node) => node.id === 'repo:api')).toMatchObject({
      state: 'attention',
      detail: reason,
      counter: { label: 'Attempt', current: 5, limit: 5 },
    });
    expect(graph.edges.find((edge) => edge.kind === 'retry')?.label).toBe('Retry 5 / 5');
    expect(graph.nodes.some((node) => node.state === 'active')).toBe(false);
  });

  it('measures every count against a limit the server published', () => {
    const graph = build(
      [workstream({ repository_id: 'api', retry_count: 3, validation_retry_count: 2 })],
      { clarification_rounds: 1, integration_review_cycles: 2 },
    );

    // The same reading the repository table shows, against the limit that bounds it.
    expect(graph.nodes.find((node) => node.id === 'repo:api')?.counter).toEqual({
      label: 'Attempt',
      current: 4,
      limit: 12,
    });
    expect(graph.nodes.find((node) => node.id === 'validate:api')?.counter).toEqual({
      label: 'Retry',
      current: 2,
      limit: 8,
    });
    expect(graph.nodes.find((node) => node.id === 'technical_prd')?.counter).toEqual({
      label: 'Clarification',
      current: 1,
      limit: 10,
    });
    expect(graph.nodes.find((node) => node.id === 'integration_review')?.counter).toEqual({
      label: 'Cycle',
      current: 2,
      limit: 5,
    });
  });

  it('adds a granted attempt to that repository’s ceiling and to nobody else’s', () => {
    const graph = build([
      workstream({ repository_id: 'granted', validation_retry_count: 9, granted_extra_attempts: 3 }),
      workstream({ repository_id: 'plain', validation_retry_count: 1 }),
    ]);

    expect(graph.nodes.find((node) => node.id === 'validate:granted')?.counter?.limit).toBe(11);
    expect(graph.nodes.find((node) => node.id === 'validate:plain')?.counter?.limit).toBe(8);
  });

  it('shows a bare count rather than an invented ratio when the server published no limit', () => {
    // A feature recorded before the server published its budgets. A denominator the client
    // made up would be worse than none.
    const graph = build([workstream({ repository_id: 'api', retry_count: 2 })], {
      max_child_review_cycles: undefined,
      max_validation_retries: undefined,
    });

    expect(graph.nodes.find((node) => node.id === 'repo:api')?.counter).toBeUndefined();
  });

  it('reads each step from the evidence the repository produced', () => {
    const graph = build([
      workstream({
        repository_id: 'api',
        status: 'review_rejected',
        code_completion_artifact_id: 'c',
        review_artifact_id: 'r',
        current_validation_results: [{ passed: true }, { passed: false }],
      }),
    ]);

    expect(graph.nodes.find((node) => node.id === 'implement:api')?.state).toBe('done');
    expect(graph.nodes.find((node) => node.id === 'validate:api')?.state).toBe('stopped');
    expect(graph.nodes.find((node) => node.id === 'validate:api')?.detail).toBe('1/2 passed');
    expect(graph.nodes.find((node) => node.id === 'review:api')?.state).toBe('stopped');
    expect(graph.nodes.find((node) => node.id === 'review:api')?.detail).toBe('Changes requested');
  });

  it('does not draw a repository as running when the feature itself has stopped', () => {
    // A child left at `running` in the record when its feature was cancelled is not running:
    // nothing is.
    const graph = build([workstream({ repository_id: 'api', status: 'running' })], {
      status: 'cancelled',
    });

    expect(graph.nodes.find((node) => node.id === 'repo:api')?.state).toBe('stopped');
  });

  it('points every node at the thing it stands for', () => {
    const graph = build([workstream({ repository_id: 'api' })]);
    const href = (id: string) => graph.nodes.find((node) => node.id === id)?.href;

    expect(href('technical_prd')).toBe('/features/f-1/prd');
    expect(href('repo:api')).toBe('/features/f-1/repositories/api');
    expect(href('validate:api')).toBe('/features/f-1/repositories/api?view=validation');
    expect(href('review:api')).toBe('/features/f-1/repositories/api?view=review');
    expect(href('pull_requests')).toBe('/features/f-1/pull-requests');
  });
});

describe('what is happening now', () => {
  it('prefers a repository going round the loop again to one running for the first time', () => {
    const graph = build([
      workstream({ repository_id: 'first-try', code_completion_artifact_id: 'c' }),
      workstream({
        repository_id: 'retrying',
        retry_count: 2,
        code_completion_artifact_id: 'c',
        current_validation_results: [{ passed: false }],
      }),
    ]);

    const now = currentActivity(graph);
    expect(now?.repositoryId).toBe('retrying');
    expect(now?.retrying).toBe(true);
  });

  it('reports the human gate ahead of anything still running', () => {
    const graph = build([workstream({ repository_id: 'api', code_completion_artifact_id: 'c' })], {
      status: 'waiting_for_human',
    });

    expect(currentActivity(graph)?.id).toBe('technical_prd');
  });

  it('draws the review as unfinished and the pull requests as unreached while it asks for changes', () => {
    // Run 185's first cycle, as the graph saw it: the review returned changes_requested and a
    // repository went back to coding. The node said "done" and the one after it said "active",
    // which is the same misreading the progress strip made -- and the two surfaces must not
    // now disagree, since the Overview draws both.
    const graph = build(
      [
        workstream({ repository_id: 'AB-console-admin-2.0', status: 'completed' }),
        workstream({ repository_id: 'admanager_console-2.0', status: 'running' }),
      ],
      { status: 'running_child_workflows', integration_review_cycles: 1 },
      EMPTY_PLAN,
      new Set(['technical_prd', 'integration_contract', 'repository_execution_plan', 'integration_review']),
      latestIntegrationReview(artifactsSchema.parse(live185).artifacts.slice(0, 1)),
    );
    const node = (id: string) => graph.nodes.find((item) => item.id === id)!;

    expect(node('integration_review').state).toBe('active');
    expect(node('integration_review').detail).toBe('Changes requested');
    expect(node('pull_requests').state).toBe('pending');
  });

  it('says nothing rather than something wrong when a feature is finished', () => {
    const graph = build(
      [
        workstream({
          repository_id: 'api',
          status: 'completed',
          code_completion_artifact_id: 'c',
          review_artifact_id: 'r',
          current_validation_results: [{ passed: true }],
        }),
      ],
      { status: 'completed' },
      EMPTY_PLAN,
      new Set([
        'technical_prd',
        'integration_contract',
        'repository_execution_plan',
        'integration_review',
        'pull_request',
      ]),
    );

    expect(currentActivity(graph)).toBeNull();
  });
});

/**
 * A retry's arrow starts at the authority that demanded it (65 D).
 *
 * Observed live on AB-Feature-201: both children finished, the integration review requested
 * changes, and the remediation attempts drew as orange loops out of each lane's own Review
 * node — attributing to the repository reviewer a rework the integration review had demanded.
 *
 * D4 is the test that decides correctness. `TargetedAttempt`'s own docstring says both kinds
 * of targeted attempt increment `retry_count`, and the lane label was built from it, so once
 * remediation has an arrow of its own a label derived from the shared counter double-counts
 * every remediation. D1 and D2 each test one arrow in isolation and would both pass with that
 * shipped.
 */
describe('D — the remediation loop', () => {
  it('D1 — draws from the integration review, labelled from the remediation allowance', () => {
    const graph = build(
      [workstream({ repository_id: 'api', retry_count: 3, integration_retry_count: 2 })],
      { max_integration_review_cycles: 5, max_child_review_cycles: 12 },
      undefined,
      undefined,
      null,
      [retryRecord('api', 2, 'integration_review'), retryRecord('api', 3, 'integration_review')],
    );

    const remediation = graph.edges.filter((edge) => edge.kind === 'remediation');
    expect(remediation).toHaveLength(1);
    expect(remediation[0]).toMatchObject({
      from: 'integration_review',
      to: 'implement:api',
      label: 'Remediation 2 / 5',
    });
    // The records land on it, because the arrow the server named is the arrow that exists.
    expect(remediation[0]!.executions).toHaveLength(2);
  });

  it('D2 — a lane with no remediation draws exactly what it drew before', () => {
    const graph = build([workstream({ repository_id: 'api', retry_count: 3 })]);

    expect(graph.edges.filter((edge) => edge.kind === 'remediation')).toHaveLength(0);
    // Byte-identical to today: with nothing subtracted, the lane label is the attempt ratio
    // it has always printed.
    expect(graph.edges.find((edge) => edge.kind === 'retry')).toMatchObject({
      from: 'review:api',
      to: 'implement:api',
      label: 'Retry 4 / 12',
    });
  });

  it('D3 — a pre-fix record attaches to the lane loop, and nothing draws twice', () => {
    // Its origin is `review`, because that is what the platform stamped before the fix. The
    // counter still says a remediation happened; the arrow it belongs to does not exist, and
    // inventing an empty one beside the populated lane loop is the failure mode.
    const graph = build(
      [workstream({ repository_id: 'api', retry_count: 3, integration_retry_count: 1 })],
      {},
      undefined,
      undefined,
      null,
      [retryRecord('api', 2, 'review'), retryRecord('api', 3, 'review')],
    );

    expect(graph.edges.filter((edge) => edge.kind === 'remediation')).toHaveLength(0);
    const lane = graph.edges.filter((edge) => edge.kind === 'retry');
    expect(lane).toHaveLength(1);
    expect(lane[0]!.executions).toHaveLength(2);
  });

  it('D4 — both arrows live at once, and neither label double-counts', () => {
    // Four attempts after the first: two the lane's own review asked for, two the integration
    // review demanded. `retry_count` is 4 and counts both kinds.
    const graph = build(
      [workstream({ repository_id: 'api', retry_count: 4, integration_retry_count: 2 })],
      { max_integration_review_cycles: 5, max_child_review_cycles: 12 },
      undefined,
      undefined,
      null,
      [
        retryRecord('api', 1, 'review'),
        retryRecord('api', 2, 'integration_review'),
        retryRecord('api', 3, 'review'),
        retryRecord('api', 4, 'integration_review'),
      ],
    );

    const lane = graph.edges.find((edge) => edge.kind === 'retry')!;
    const remediation = graph.edges.find((edge) => edge.kind === 'remediation')!;
    // Two remediations on the remediation arrow, and the lane arrow counting only what is
    // left of `retry_count` after them — 4 less 2, plus the first attempt the ratio has
    // always included.
    expect(remediation.label).toBe('Remediation 2 / 5');
    expect(lane.label).toBe('Retry 3 / 12');
    // And no attempt is on both arrows.
    const laneIds = lane.executions.map((item) => item.execution_id);
    const remediationIds = remediation.executions.map((item) => item.execution_id);
    expect(laneIds).toEqual(['retry:api:1', 'retry:api:3']);
    expect(remediationIds).toEqual(['retry:api:2', 'retry:api:4']);
    expect(laneIds.filter((id) => remediationIds.includes(id))).toEqual([]);
  });

  it('D4 — a lane whose only retries were remediations draws no lane loop', () => {
    // `retry_count > 0` is no longer the test. It counts remediations too, so a lane like
    // this would otherwise draw an empty review loop beside the populated integration one.
    const graph = build(
      [workstream({ repository_id: 'api', retry_count: 2, integration_retry_count: 2 })],
      { max_integration_review_cycles: 5 },
      undefined,
      undefined,
      null,
      [retryRecord('api', 1, 'integration_review'), retryRecord('api', 2, 'integration_review')],
    );

    expect(graph.edges.filter((edge) => edge.kind === 'retry')).toHaveLength(0);
    expect(graph.edges.filter((edge) => edge.kind === 'remediation')).toHaveLength(1);
  });

  it('D4 — "Changes requested" still draws on the status, not on a counter', () => {
    // The one lane-loop case that never depended on `retry_count`, so it is unaffected.
    const graph = build([
      workstream({ repository_id: 'api', status: 'review_rejected', retry_count: 0 }),
    ]);

    expect(graph.edges.find((edge) => edge.kind === 'retry')?.label).toBe('Changes requested');
  });
});
