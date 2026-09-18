import { useCallback, useMemo, useState } from 'react';
import { EmptyState, TableSkeleton } from '@/components/common/States';
import { Panel } from '@/components/ui/Layout';
import { Badge, RepositoryBadge } from '@/components/ui/Badge';
import { DetailList, DetailRow } from '@/components/ui/Value';
import {
  useArtifact,
  useArtifactList,
  useExecutions,
  useFeature,
  useIntegrationReview,
  useWording,
  useWorkstreamOperations,
  useWorkstreams,
} from './hooks';
import { buildFeatureGraph, currentActivity, planView, type GraphEdge, type GraphNode } from './graph';
import { livenessByNode } from './operations';
import { GraphLegend, WorkflowGraph } from './WorkflowGraph';
import { ExecutionDrawer } from './ExecutionDrawer';
import { AgentWorkDrawer } from './AgentWorkDrawer';
import {
  nodeOperations,
  planningNodeOperations,
  type NodeOpLine,
  type WorkstreamOutcome,
} from './agent-work';

/**
 * Which stages each lane node carries, by the node-id prefix `graph.ts` builds.
 *
 * Setup rides Implementation because the lane has no setup node and clone/branch/install are
 * the Engineer preparing to write the change. Publication is absent on purpose: there is no
 * publication node in a lane, and `create_commit`/`push_branch` would have to claim a stage
 * the graph does not model. They stay in the drawer, where the stepper already names them.
 */
const NODE_STAGES: Record<string, readonly string[]> = {
  implement: ['setup', 'coding'],
  validate: ['validation'],
  review: ['review'],
};
import type { Workstream } from '@/schemas/feature';

/**
 * The workflow, as the graph it actually is.
 *
 * It reads the same queries as every other view -- the feature, its workstreams, its
 * artifacts, the planner's execution plan and the executions behind each transition -- all
 * keyed by the event cursor, so it updates from the existing live-update loop rather than from
 * a second state system of its own. The backend stays the only source of truth about where
 * execution is and about which model performed it.
 *
 * Nodes and edges are both ways in. A node opens the stage it stands for; an edge opens the
 * execution that moved the feature along it.
 */
export function WorkflowTab({ featureId, at }: { featureId: string; at: number | null }) {
  const feature = useFeature(featureId);
  const workstreams = useWorkstreams(featureId, at);
  const artifacts = useArtifactList(featureId, undefined, at);
  const plans = useArtifactList(featureId, 'repository_execution_plan', at);
  const planArtifact = useArtifact(featureId, plans.data?.artifacts.at(-1)?.artifact_id ?? null);
  const executions = useExecutions(featureId, at);
  const integrationReview = useIntegrationReview(featureId, at);
  // The journal's liveness rows, polled on the feature's cadence rather than keyed by the
  // event cursor: a heartbeat renewal writes no lifecycle event, and the heartbeat is the
  // one thing this read exists to see.
  const journalByRepository = useWorkstreamOperations(
    featureId,
    (workstreams.data?.workstreams ?? []).map((item) => item.repository_id),
    feature.data?.status,
  );
  // One clock reading for everything derived from the poll, so the node indicator and the
  // drill-in cannot say two different ages for the same heartbeat. The rows are lifted out
  // of the responses once: the liveness reading is about rows alone, and the per-attempt
  // endings beside them belong to the drawer and the strip.
  const { liveness, nowMs } = useMemo(() => {
    const now = Date.now();
    const rows = new Map(
      [...journalByRepository].map(([repositoryId, journal]) => [repositoryId, journal.operations]),
    );
    return { liveness: livenessByNode(rows, now), nowMs: now };
  }, [journalByRepository]);
  // The operations each node lists, keyed by node id. The lane nodes carry the newest
  // attempt's journal rows — setup and coding under Implementation, the commands under
  // Validation, the reviewer call under Review — from the same polled response the drawer
  // reads, so the graph and the drill-in cannot disagree about what the agent did. The
  // planning nodes carry the journaled pre-coding calls from the executions read the edge
  // chips already consume. Neither costs a request of its own, and the key spaces cannot
  // collide: lane keys are `prefix:repository`, planning keys are the stage node ids.
  const nodeOps = useMemo(() => {
    const byNode = new Map<string, NodeOpLine[]>();
    for (const [repositoryId, journal] of journalByRepository) {
      for (const [prefix, stages] of Object.entries(NODE_STAGES)) {
        byNode.set(
          `${prefix}:${repositoryId}`,
          nodeOperations(journal.operations, journal.attempts, stages, { nowMs }),
        );
      }
    }
    for (const [nodeId, lines] of planningNodeOperations(executions.data?.executions ?? [], {
      nowMs,
    })) {
      byNode.set(nodeId, lines);
    }
    return byNode;
  }, [journalByRepository, nowMs, executions.data]);
  // The arrow whose execution is open. Held by edge id rather than by index: retries produce
  // several executions between the same two nodes, and the graph is rebuilt on every poll.
  const [openEdgeId, setOpenEdgeId] = useState<string | null>(null);
  // The lane node whose agent-work drill-in is open, held by node id for the same reason.
  const [openNodeId, setOpenNodeId] = useState<string | null>(null);
  // The server's own wording for a workstream state, so a repository that needs something
  // says what rather than repeating its role.
  const wordingFor = useWording('workstream');
  const describeWorkstream = useCallback(
    (status: string) => wordingFor(status).headline,
    [wordingFor],
  );

  const graph = useMemo(() => {
    if (!feature.data || !workstreams.data || !artifacts.data) return null;
    return buildFeatureGraph({
      feature: feature.data,
      workstreams: workstreams.data.workstreams,
      artifactTypes: new Set(artifacts.data.artifacts.map((item) => item.artifact_type)),
      plan: planView(planArtifact.data?.payload),
      featureId,
      describeWorkstream,
      // Absent while the read is in flight: the graph draws, and its arrows are labelled as
      // soon as the executions arrive rather than the whole view waiting for them.
      executions: executions.data?.executions,
      integrationReview,
    });
  }, [
    feature.data,
    workstreams.data,
    artifacts.data,
    planArtifact.data,
    executions.data,
    integrationReview,
    featureId,
    describeWorkstream,
  ]);

  const openEdge = useMemo(
    () => graph?.edges.find((edge) => edge.id === openEdgeId) ?? null,
    [graph, openEdgeId],
  );
  const openNode = useMemo(
    () => graph?.nodes.find((node): node is GraphNode => node.id === openNodeId) ?? null,
    [graph, openNodeId],
  );

  if (!graph) {
    return (
      <Panel title="Workflow">
        <TableSkeleton rows={6} label="Loading the workflow…" />
      </Panel>
    );
  }

  const now = currentActivity(graph);
  const repositories = workstreams.data?.workstreams ?? [];

  return (
    <div className="stack">
      <Panel
        title="Workflow"
        meta={
          repositories.length > 0
            ? `${repositories.length} ${repositories.length === 1 ? 'repository' : 'repositories'}`
            : undefined
        }
        actions={<GraphLegend />}
      >
        {now ? (
          <DetailList narrow>
            <DetailRow label="Currently">
              <span className="row" style={{ gap: 'var(--space-2)' }}>
                <span>{now.label}</span>
                {now.repositoryId ? <RepositoryBadge repositoryId={now.repositoryId} /> : null}
                {now.retrying ? <Badge tone="attention">Retrying</Badge> : null}
              </span>
            </DetailRow>
            {now.counter ? (
              <DetailRow label={now.counter.label}>
                {/* The counter names the repository it counts for. "Retry 1 of 8" on its own
                    row is ambiguous the moment two lanes are retrying. */}
                <span className="row" style={{ gap: 'var(--space-2)' }}>
                  <span>
                    {now.counter.current} of {now.counter.limit}
                  </span>
                  {now.repositoryId ? <RepositoryBadge repositoryId={now.repositoryId} /> : null}
                </span>
              </DetailRow>
            ) : null}
            {feature.data?.current_agent ? (
              <DetailRow label="Agent">{feature.data.current_agent}</DetailRow>
            ) : null}
          </DetailList>
        ) : null}
        {repositories.length === 0 ? (
          <EmptyState
            title="No repository workstreams yet"
            detail="The graph fans out once the planner has agreed the shared contract and assigned the repositories."
          />
        ) : null}
        <WorkflowGraph
          graph={graph}
          label="Feature execution graph"
          selectedEdgeId={openEdgeId}
          onSelectEdge={(edge: GraphEdge) => {
            setOpenNodeId(null);
            setOpenEdgeId(edge.id);
          }}
          selectedNodeId={openNodeId}
          onSelectNode={(node: GraphNode) => {
            setOpenEdgeId(null);
            setOpenNodeId(node.id);
          }}
          operations={nodeOps}
          liveness={liveness}
        />
        <p className="subtle">
          Every node opens what it stands for &mdash; a repository lane node opens what its
          agents are doing, operation by operation &mdash; and every arrow opens the execution
          that moved the feature along it: which model or which handler performed it, on which
          attempt, and what a retry was asked to change. Attempt and cycle counts are the
          platform&rsquo;s own, measured against the budgets this feature is being run under.
        </p>
      </Panel>
      {openEdge && openEdge.executions.length > 0 ? (
        <ExecutionDrawer
          featureId={featureId}
          executions={openEdge.executions}
          onDismiss={() => setOpenEdgeId(null)}
        />
      ) : null}
      {openNode?.repositoryId ? (
        <AgentWorkDrawer
          featureId={featureId}
          repositoryId={openNode.repositoryId}
          node={openNode}
          // The same rows the liveness indicator polls, on the same 55- cadence: opening the
          // drill-in costs no request of its own, and one fetch serves every node in the lane.
          operations={journalByRepository.get(openNode.repositoryId)?.operations ?? []}
          // Where each finished attempt ended, from that same poll. The drawer renders these
          // and never derives one: an ending is a record, and a client that guessed at one is
          // the defect this replaced.
          endings={journalByRepository.get(openNode.repositoryId)?.attempts ?? []}
          // The workstream's own recorded state, and the only thing that licenses the drawer to
          // say an unjournaled phase is in progress: a settled workstream with no running
          // operation is silent because it finished, not because it went quiet.
          childRunning={
            repositories.find((item) => item.repository_id === openNode.repositoryId)?.status ===
            'running'
          }
          // How the workstream stands now, which is the latest attempt's outcome and no
          // other's -- one status and one failure class describe where it is, not where each
          // earlier attempt ended. The drawer marks only the latest attempt with it; the rest
          // are read from their own journal rows.
          outcome={workstreamOutcome(
            repositories.find((item) => item.repository_id === openNode.repositoryId),
          )}
          nowMs={nowMs}
          onDismiss={() => setOpenNodeId(null)}
        />
      ) : null}
    </div>
  );
}

/**
 * The workstream's own status and failure class, or nothing when the workstream is not in this
 * read. Nothing is a real answer: without a record, the drawer marks the latest attempt from
 * its journal rows alone rather than from a guessed status.
 */
function workstreamOutcome(workstream: Workstream | undefined): WorkstreamOutcome | undefined {
  return workstream
    ? {
        status: workstream.status,
        failureClassification: workstream.failure_classification ?? null,
      }
    : undefined;
}
