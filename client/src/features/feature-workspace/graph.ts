import type { Artifact, ExecutionRecord, Feature, Workstream } from '@/schemas/feature';
import { count } from '@/utils/count';
import { humanise } from '@/utils/text';
import { edgeKey, executionsByEdge } from './executions';
import { effectiveFeatureStatus } from './feature-status';
import type { IntegrationReviewOutcome } from './integration-review';
import { attempts, validationSummary } from './workstream-view';

/**
 * The shape of one feature's execution, derived from what the platform recorded.
 *
 * Everything here is read from state: which artifacts exist, what each workstream's status
 * and counters say, and what the planner's own execution plan declared about order and
 * dependencies. Nothing is inferred beyond that, and no limit is invented -- a ratio appears
 * only where the server published the denominator for it.
 *
 * The result is a layered graph: the stages before the fan-out, one lane per repository, and
 * the convergence back into integration review and pull requests. It is computed rather than
 * drawn, so a feature with one repository and a feature with five are the same code.
 */

export type NodeState = 'done' | 'active' | 'pending' | 'stopped' | 'attention';

export type NodeKind =
  | 'stage'
  | 'repository'
  | 'implement'
  | 'validate'
  | 'review'
  | 'integration'
  | 'pull-requests';

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  state: NodeState;
  /** The repository this node belongs to, where it belongs to one. */
  repositoryId?: string;
  /** A short second line: what is happening, or what was counted. */
  detail?: string;
  /** A counter and the published limit it is measured against. Never one without the other. */
  counter?: { label: string; current: number; limit: number };
  /** The agent responsible, when the platform says one is running. */
  agent?: string;
  /** Where clicking this node goes. */
  href?: string;
  /** Whether this node is part of a loop the feature is going round right now. */
  retrying?: boolean;
  column: number;
  row: number;
}

export interface GraphEdge {
  /**
   * Stable, and never just source-and-target: retries produce several executions between the
   * same two nodes, and the drawer, the realtime update and React's own reconciliation all
   * need to tell them apart.
   */
  id: string;
  from: string;
  to: string;
  /**
   * A loop edge is drawn backwards, around the outside, and labelled. `remediation` is its
   * own kind rather than a `retry` reused: the retry geometry computes its dip from one
   * node's row and assumes both ends share it, so a cross-column, cross-row back edge drawn
   * as a retry would run straight through the lanes.
   */
  kind: 'flow' | 'retry' | 'dependency' | 'remediation';
  label?: string;
  active?: boolean;
  /**
   * The one edge per lane the work is flowing along right now: the edge whose newest attached
   * execution is running or queued. At most one per lane, ever — an invariant this graph
   * enforces rather than an observation about the data, because the record builder emits the
   * retry record *alongside* the implementation record on purpose, so during a retry attempt
   * two edges carry the same running status.
   */
  live?: boolean;
  /**
   * The executions this arrow represents, oldest first. An arrow is a transition, and the
   * transition was performed by something -- these records say by what, on which attempt, and
   * against what evidence. Empty where the platform has not recorded an execution for it.
   */
  executions: ExecutionRecord[];
  /**
   * Whether this arrow draws its own label. False only where several arrows share one
   * execution -- every repository's review leads into the same integration review -- so the
   * label appears once and each arrow still opens it.
   */
  showLabel: boolean;
}

export interface FeatureGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  columns: number;
  rows: number;
}

const FEATURE_STOPPED = new Set([
  'failed',
  'failed_requires_human',
  'cancelled',
  'cancelled_with_external_side_effects',
]);

const CHILD_DONE = new Set(['approved', 'completed']);
const CHILD_STOPPED = new Set(['failed', 'review_rejected', 'cancelled']);
const CHILD_ATTENTION = new Set(['blocked', 'waiting_for_contract_change']);
const CHILD_PENDING = new Set(['pending']);

/** A repository's own overall state, in the same vocabulary the graph draws. */
function repositoryState(workstream: Workstream, featureStopped: boolean): NodeState {
  if (CHILD_DONE.has(workstream.status)) return 'done';
  // A persisted retry refusal is an operator handoff, not a generic dead end. The backend
  // owns both the decision and its reason; the graph only projects that durable fact.
  if (workstream.retry_refusal_reason) return 'attention';
  if (CHILD_STOPPED.has(workstream.status)) return 'stopped';
  if (CHILD_ATTENTION.has(workstream.status)) return 'attention';
  if (CHILD_PENDING.has(workstream.status)) return 'pending';
  // A child left at `running` when the feature itself has stopped is not running: nothing
  // is. The feature's terminal state decides, because it is the thing that stopped.
  return featureStopped ? 'stopped' : 'active';
}

/**
 * Order the repositories into dependency layers.
 *
 * The planner publishes both `parallel_groups` (already layered) and per-workstream
 * `dependency_workstream_ids`. The groups are preferred because they are the planner's own
 * answer; the dependency lists are the fallback and the source of the edges either way.
 */
function layerRepositories(
  workstreams: Workstream[],
  plan: PlanView,
): { order: string[]; layer: Map<string, number> } {
  const layer = new Map<string, number>();
  const byRepository = new Map(workstreams.map((item) => [item.repository_id, item]));

  if (plan.parallelGroups.length > 0) {
    plan.parallelGroups.forEach((group, index) => {
      for (const workstreamId of group) {
        const repositoryId = plan.repositoryOf.get(workstreamId) ?? workstreamId;
        if (byRepository.has(repositoryId)) layer.set(repositoryId, index);
      }
    });
  }
  // Anything the plan did not mention -- a repository added later, or a feature that never
  // reached planning -- starts in the first layer rather than being dropped from the graph.
  for (const workstream of workstreams) {
    if (!layer.has(workstream.repository_id)) layer.set(workstream.repository_id, 0);
  }

  const order = [...workstreams]
    .sort((left, right) => {
      const difference =
        (layer.get(left.repository_id) ?? 0) - (layer.get(right.repository_id) ?? 0);
      return difference !== 0
        ? difference
        : left.repository_id.localeCompare(right.repository_id);
    })
    .map((item) => item.repository_id);

  return { order, layer };
}

export interface PlanView {
  parallelGroups: string[][];
  /** Which repository each planner workstream id belongs to. */
  repositoryOf: Map<string, string>;
  /** Repository id → the repository ids it must wait for. */
  dependencies: Map<string, string[]>;
}

/** Read the planner's execution plan into the two things the graph needs from it. */
export function planView(payload: Record<string, unknown> | undefined): PlanView {
  const repositoryOf = new Map<string, string>();
  const dependencyWorkstreams = new Map<string, string[]>();

  const workstreams = Array.isArray(payload?.workstreams) ? payload.workstreams : [];
  for (const item of workstreams) {
    if (typeof item !== 'object' || item === null) continue;
    const record = item as Record<string, unknown>;
    const workstreamId = typeof record.workstream_id === 'string' ? record.workstream_id : null;
    const repositoryId = typeof record.repository_id === 'string' ? record.repository_id : null;
    if (!workstreamId || !repositoryId) continue;
    repositoryOf.set(workstreamId, repositoryId);
    dependencyWorkstreams.set(
      workstreamId,
      Array.isArray(record.dependency_workstream_ids)
        ? record.dependency_workstream_ids.filter((value): value is string => typeof value === 'string')
        : [],
    );
  }

  const dependencies = new Map<string, string[]>();
  for (const [workstreamId, required] of dependencyWorkstreams) {
    const repositoryId = repositoryOf.get(workstreamId);
    if (!repositoryId) continue;
    dependencies.set(
      repositoryId,
      required.map((item) => repositoryOf.get(item) ?? item).filter((item) => item !== repositoryId),
    );
  }

  const groups = Array.isArray(payload?.parallel_groups) ? payload.parallel_groups : [];
  const parallelGroups = groups
    .filter((group): group is unknown[] => Array.isArray(group))
    .map((group) => group.filter((item): item is string => typeof item === 'string'))
    .filter((group) => group.length > 0);

  return { parallelGroups, repositoryOf, dependencies };
}

/** A counter is only ever shown beside the limit the platform measures it against. */
function counter(label: string, current: number, limit: number | undefined) {
  return limit === undefined ? undefined : { label, current, limit };
}

export function buildFeatureGraph({
  feature,
  workstreams,
  artifactTypes,
  plan,
  featureId,
  describeWorkstream,
  executions = [],
  integrationReview = null,
}: {
  feature: Feature;
  workstreams: Workstream[];
  artifactTypes: Set<string>;
  plan: PlanView;
  featureId: string;
  /**
   * The newest integration review's verdict, where the caller read it. A review can hand the
   * work back, so the existence of the artifact does not finish that node. Optional: a graph
   * built without it draws exactly what it drew before.
   */
  integrationReview?: IntegrationReviewOutcome | null;
  /** The server's plain-language headline for a workstream status, where the caller has it. */
  describeWorkstream?: (status: string) => string;
  /**
   * Who performed each transition, as the server recorded it. Optional: a graph built without
   * it draws exactly what it drew before, and every arrow is simply unlabelled.
   */
  executions?: ExecutionRecord[];
}): FeatureGraph {
  const effectiveStatus = effectiveFeatureStatus(feature);
  const stopped = FEATURE_STOPPED.has(effectiveStatus);
  const attention = effectiveStatus === 'waiting_for_human';
  const base = `/features/${encodeURIComponent(featureId)}`;
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const attached = executionsByEdge(
    executions,
    workstreams.map((item) => item.repository_id),
  );
  /**
   * Add one arrow, carrying whatever executions the server recorded between those two stages.
   * The identity is the pair of nodes, which is what an arrow *is*; the several executions
   * along it are distinguished by their own ids inside the drawer.
   */
  const connect = (
    from: string,
    to: string,
    kind: GraphEdge['kind'],
    extra: Partial<GraphEdge> = {},
  ) => {
    const id = edgeKey(from, to);
    edges.push({
      id,
      from,
      to,
      kind,
      executions: attached.get(id) ?? [],
      showLabel: true,
      ...extra,
    });
  };

  const stageState = (produced: boolean, reached: boolean): NodeState => {
    if (produced) return 'done';
    if (!reached) return 'pending';
    return stopped ? 'stopped' : 'active';
  };

  const hasTechnicalPrd = artifactTypes.has('technical_prd');
  const hasContract = artifactTypes.has('integration_contract');
  const hasPlan = artifactTypes.has('repository_execution_plan');
  // Accepted and durably queued, and no worker has picked it up yet. Worth its own reading:
  // the graph used to draw the first stage as active for a feature nothing had touched, which
  // said the product manager was working when it was not.
  const queued = effectiveStatus === 'pending';

  // --- the stages before the fan-out ------------------------------------------------------
  nodes.push({
    id: 'request',
    kind: 'stage',
    // Always done. The request exists — that is what the rest of the page is about — so the
    // first thing the graph says is the one thing that is certainly true.
    state: 'done',
    label: 'Request',
    detail: 'Accepted',
    href: `${base}/prd`,
    column: 0,
    row: 0,
  });
  nodes.push({
    id: 'technical_prd',
    kind: 'stage',
    // The clarification gate belongs to this stage: the questions are asked once the
    // technical PRD has been drafted, so a feature waiting on a person is waiting here even
    // though the artifact already exists.
    state: attention ? 'attention' : queued ? 'pending' : stageState(hasTechnicalPrd, true),
    label: 'Product manager',
    detail: attention ? 'Waiting on your answers' : queued ? 'Queued' : undefined,
    counter:
      feature.clarification_rounds > 0 || attention
        ? counter('Clarification', feature.clarification_rounds, feature.max_clarification_rounds)
        : undefined,
    href: `${base}/prd`,
    column: 1,
    row: 0,
  });
  nodes.push({
    id: 'integration_contract',
    kind: 'stage',
    label: 'Shared contract',
    state: stageState(hasContract, hasTechnicalPrd),
    href: `${base}/plan?section=contract`,
    column: 2,
    row: 0,
  });
  nodes.push({
    id: 'execution_plan',
    kind: 'stage',
    label: 'Execution plan',
    state: stageState(hasPlan, hasContract),
    detail:
      workstreams.length > 0
        ? count(workstreams.length, 'repository', 'repositories')
        : undefined,
    href: `${base}/plan?section=execution`,
    column: 3,
    row: 0,
  });
  connect('request', 'technical_prd', 'flow');
  connect('technical_prd', 'integration_contract', 'flow');
  connect('integration_contract', 'execution_plan', 'flow');

  // --- one lane per repository ------------------------------------------------------------
  const { order } = layerRepositories(workstreams, plan);
  const byRepository = new Map(workstreams.map((item) => [item.repository_id, item]));
  const laneRow = new Map<string, number>();

  order.forEach((repositoryId, index) => {
    const workstream = byRepository.get(repositoryId);
    if (!workstream) return;
    const row = index;
    laneRow.set(repositoryId, row);
    const state = repositoryState(workstream, stopped);
    const repositoryPath = `${base}/repositories/${encodeURIComponent(repositoryId)}`;
    const label = workstream.repository_name ?? repositoryId;
    // The platform's own phrase, against the limit it actually measures `retry_count` by,
    // and the same reading the repository table shows.
    const attempt = counter('Attempt', attempts(workstream), feature.max_child_review_cycles);
    const retrying = state === 'active' && workstream.retry_count > 0;

    const implemented = Boolean(workstream.code_completion_artifact_id);
    const validation = validationSummary(workstream);
    const reviewed = Boolean(workstream.review_artifact_id);
    const setupBlocked = workstream.blocking_setup_issues.length > 0;

    const implementState: NodeState = setupBlocked
      ? 'attention'
      : implemented || validation || reviewed
        ? 'done'
        : state === 'pending'
          ? 'pending'
          : state;
    const validateState: NodeState = !validation
      ? implemented && state === 'active'
        ? 'active'
        : state === 'stopped' && implemented
          ? 'stopped'
          : 'pending'
      : validation.passed === validation.total
        ? 'done'
        : 'stopped';
    const reviewState: NodeState = reviewed
      ? workstream.status === 'review_rejected'
        ? 'stopped'
        : CHILD_DONE.has(workstream.status)
          ? 'done'
          : state === 'attention' || state === 'stopped'
            ? state
            : 'active'
      : validateState === 'done' && state === 'active'
        ? 'active'
        : 'pending';

    nodes.push({
      id: `repo:${repositoryId}`,
      kind: 'repository',
      label,
      state,
      repositoryId,
      // The server's own words for a repository that needs something, and the role
      // otherwise. "Needs a decision about the shared contract" is what somebody has to
      // read; "frontend" is what they already know.
      detail:
        state === 'attention' || state === 'stopped'
          ? (workstream.retry_refusal_reason ??
            describeWorkstream?.(workstream.status) ??
            workstream.repository_role ??
            undefined)
          : (workstream.repository_role ?? undefined),
      counter: workstream.retry_count > 0 || state === 'active' ? attempt : undefined,
      retrying,
      href: repositoryPath,
      column: 4,
      row,
    });
    nodes.push({
      id: `implement:${repositoryId}`,
      kind: 'implement',
      label: 'Implementation',
      state: implementState,
      repositoryId,
      detail: setupBlocked ? 'Repository setup needs a decision' : undefined,
      counter:
        workstream.implementation_retry_count > 0
          ? counter(
              'Retry',
              workstream.implementation_retry_count,
              addGrant(feature.max_implementation_retries, workstream),
            )
          : undefined,
      retrying: retrying && implementState === 'active',
      href: `${repositoryPath}?view=changes`,
      column: 5,
      row,
    });
    nodes.push({
      id: `validate:${repositoryId}`,
      kind: 'validate',
      label: 'Validation',
      state: validateState,
      repositoryId,
      detail: validation ? `${validation.passed}/${validation.total} passed` : undefined,
      counter:
        workstream.validation_retry_count > 0
          ? counter(
              'Retry',
              workstream.validation_retry_count,
              addGrant(feature.max_validation_retries, workstream),
            )
          : undefined,
      retrying: retrying && validateState !== 'done',
      href: `${repositoryPath}?view=validation`,
      column: 6,
      row,
    });
    nodes.push({
      id: `review:${repositoryId}`,
      kind: 'review',
      label: 'Review',
      state: reviewState,
      repositoryId,
      detail:
        workstream.status === 'review_rejected'
          ? 'Changes requested'
          : reviewState === 'done'
            ? 'Approved'
            : undefined,
      href: `${repositoryPath}?view=review`,
      column: 7,
      row,
    });

    connect('execution_plan', `repo:${repositoryId}`, 'flow');
    connect(`repo:${repositoryId}`, `implement:${repositoryId}`, 'flow');
    connect(`implement:${repositoryId}`, `validate:${repositoryId}`, 'flow');
    connect(`validate:${repositoryId}`, `review:${repositoryId}`, 'flow');

    // The two loops, and the two budgets they draw on. Each label comes from its own
    // counter, never from the shared `retry_count`: `TargetedAttempt`'s own docstring says
    // both kinds of targeted attempt increment it, so a lane label built from it counts every
    // remediation twice over once remediation has an arrow of its own.
    //
    // Workstream retries are what is left of `retry_count` after the remediations, which are
    // counted separately and incremented only for them.
    const workstreamRetries = Math.max(
      0,
      workstream.retry_count - workstream.integration_retry_count,
    );
    const loopExecutions = attached.get(
      edgeKey(`review:${repositoryId}`, `implement:${repositoryId}`),
    );
    const remediationExecutions = attached.get(
      edgeKey('integration_review', `implement:${repositoryId}`),
    );
    // The lane loop: drawn only where it has work of its own. `retry_count` alone is no
    // longer the test -- a lane whose only retries were remediations would draw an empty
    // review loop beside the populated integration one. The `review_rejected` "Changes
    // requested" case is unaffected: it draws on its status, not on a counter.
    if (
      workstreamRetries > 0 ||
      workstream.status === 'review_rejected' ||
      (loopExecutions?.length ?? 0) > 0
    ) {
      // The ratio this loop has always printed -- the attempt number against the review-cycle
      // budget -- with the remediations taken back out of it. With no remediations that is
      // `attempts(workstream)` exactly, so a lane that never left its own review loop reads
      // as it always did; with two of four attempts demanded by the integration review, the
      // two it did not ask for are the only ones counted here.
      const laneAttempt = workstreamRetries + 1;
      const retryLabel =
        workstreamRetries > 0
          ? feature.max_child_review_cycles === undefined
            ? `Retry ${laneAttempt}`
            : `Retry ${laneAttempt} / ${feature.max_child_review_cycles}`
          : 'Changes requested';
      connect(`review:${repositoryId}`, `implement:${repositoryId}`, 'retry', {
        label: retryLabel,
        active: retrying,
      });
    }
    // The integration loop: the rework the integration review demanded, drawn from the
    // authority that demanded it. Drawn only where records actually attach to it, so a run
    // that predates the origin stamp keeps every retry on the lane loop exactly as today
    // rather than gaining an empty second arrow beside it.
    if ((remediationExecutions?.length ?? 0) > 0) {
      const label =
        feature.max_integration_review_cycles === undefined
          ? `Remediation ${workstream.integration_retry_count}`
          : `Remediation ${workstream.integration_retry_count} / ${feature.max_integration_review_cycles}`;
      connect('integration_review', `implement:${repositoryId}`, 'remediation', {
        label,
        active: retrying && workstream.integration_retry_count > 0,
      });
    }
  });

  // Dependency edges between repositories, as the plan declared them.
  for (const [repositoryId, required] of plan.dependencies) {
    if (!byRepository.has(repositoryId)) continue;
    for (const dependency of required) {
      if (!byRepository.has(dependency)) continue;
      // Lane to lane, not step to lane: the plan says this repository waits for that one,
      // and an edge from the far end of another lane crossed every row between them.
      connect(`repo:${dependency}`, `repo:${repositoryId}`, 'dependency', {
        label: 'waits for',
      });
    }
  }

  // --- convergence ------------------------------------------------------------------------
  const allDone =
    workstreams.length > 0 &&
    workstreams.every((item) => CHILD_DONE.has(item.status));
  const middleRow = workstreams.length > 0 ? (workstreams.length - 1) / 2 : 0;

  // The same reading the progress strip makes: a `changes_requested` verdict sent the work
  // back to a repository, so this node is not done and the pull requests after it have not
  // been reached. Only an approved review opens that gate.
  const unapprovedReview =
    integrationReview && !integrationReview.approved ? integrationReview : null;
  const reviewApproved = artifactTypes.has('integration_review') && unapprovedReview === null;

  nodes.push({
    id: 'integration_review',
    kind: 'integration',
    label: 'Integration review',
    state: unapprovedReview
      ? stopped
        ? 'stopped'
        : 'active'
      : stageState(reviewApproved, allDone),
    // The verdict in the record's own word, so the node says why it is not finished.
    detail: unapprovedReview ? humanise(unapprovedReview.status) : undefined,
    counter:
      feature.integration_review_cycles > 0
        ? counter(
            'Cycle',
            feature.integration_review_cycles,
            feature.max_integration_review_cycles,
          )
        : undefined,
    href: `${base}/artifacts`,
    column: 8,
    row: middleRow,
  });
  nodes.push({
    id: 'pull_requests',
    kind: 'pull-requests',
    label: 'Pull requests',
    state: stageState(artifactTypes.has('pull_request'), reviewApproved),
    href: `${base}/pull-requests`,
    column: 9,
    row: middleRow,
  });
  connect('integration_review', 'pull_requests', 'flow');
  // Every repository's review leads into the same integration review, and one integration
  // review performed that transition for all of them. Each arrow opens it; the label is drawn
  // once, on the lane that points straight at the node, so five repositories do not produce
  // five copies of the same sentence.
  const lanes = order.filter((repositoryId) => byRepository.has(repositoryId));
  const nearestLane = lanes.reduce<string | null>((closest, repositoryId) => {
    if (closest === null) return repositoryId;
    const distance = Math.abs((laneRow.get(repositoryId) ?? 0) - middleRow);
    return distance < Math.abs((laneRow.get(closest) ?? 0) - middleRow) ? repositoryId : closest;
  }, null);
  for (const repositoryId of lanes) {
    connect(`review:${repositoryId}`, 'integration_review', 'flow', {
      showLabel: repositoryId === nearestLane,
    });
  }

  markLiveEdges(edges, {
    repositoryIds: order.filter((repositoryId) => byRepository.has(repositoryId)),
    // Nothing is flowing while the platform is waiting on a person, or after it stopped. The
    // stillness is itself the signal, and a queued attempt behind a human gate is not motion.
    still: stopped || attention,
  });

  return {
    nodes,
    edges,
    columns: 10,
    rows: Math.max(1, workstreams.length),
  };
}

/** The execution statuses in which work is flowing along an arrow rather than recorded on it. */
const LIVE_EXECUTION_STATUSES = new Set(['running', 'queued']);

/**
 * Mark the one edge per lane the work is flowing along, and no more than one.
 *
 * Derived from the executions already attached to each edge -- no new server field, and no
 * second reading of what is running. Only two edges can ever qualify: the lane's entry edge
 * and its loops. The `implement→validate` record is completed, failed or pending only, and
 * the `validate→review` record's status is hard-coded completed, so neither can be running --
 * which is consistent with the restraint this function exists to keep rather than a gap in it.
 *
 * **The tie-break, because two edges qualify at once.** The record builder emits an
 * in-flight attempt's retry record *alongside* its implementation record on purpose, "so an
 * in-flight attempt has the same pair of arrows a completed one does", and both carry the
 * same status. During a retry attempt the lane's entry edge and its loop are therefore both
 * running. **The loop wins**: it is the more specific statement of what is happening, and the
 * entry edge stays static beneath it. Where no loop is live, the entry edge is the live one.
 */
function markLiveEdges(
  edges: GraphEdge[],
  { repositoryIds, still }: { repositoryIds: string[]; still: boolean },
): void {
  if (still) return;
  const isLive = (edge: GraphEdge) => {
    const newest = edge.executions.at(-1);
    return newest !== undefined && LIVE_EXECUTION_STATUSES.has(newest.status);
  };
  for (const repositoryId of repositoryIds) {
    const target = `implement:${repositoryId}`;
    const loop = edges.find(
      (edge) => edge.to === target && edge.kind !== 'flow' && isLive(edge),
    );
    const entry = edges.find(
      (edge) => edge.to === target && edge.kind === 'flow' && isLive(edge),
    );
    const winner = loop ?? entry;
    if (winner) winner.live = true;
  }
}

/**
 * A granted attempt raises this repository's ceiling and nobody else's, so the limit shown
 * beside its counter is the feature's budget plus whatever a person granted here.
 */
function addGrant(limit: number | undefined, workstream: Workstream): number | undefined {
  return limit === undefined ? undefined : limit + workstream.granted_extra_attempts;
}

/**
 * The single line that answers "what is happening right now".
 *
 * Read from the graph rather than computed a second time, so the summary on the Overview and
 * the highlighted node in the graph cannot disagree.
 */
export function currentActivity(graph: FeatureGraph): GraphNode | null {
  const steps = graph.nodes.filter((node) => node.kind !== 'repository');
  // A repository going round the loop again outranks one running for the first time: "still
  // running" and "on its fourth attempt" are different situations, and only one of them is
  // worth interrupting somebody about.
  const retrying = steps.find((node) => node.retrying && node.state !== 'done');
  if (retrying) return retrying;
  const needsSomebody = graph.nodes.find((node) => node.state === 'attention');
  if (needsSomebody) return needsSomebody;
  const active = steps.find((node) => node.state === 'active');
  if (active) return active;
  return steps.find((node) => node.state === 'stopped') ?? null;
}

export function isArtifactTypeSet(artifacts: Artifact[]): Set<string> {
  return new Set(artifacts.map((item) => item.artifact_type));
}
