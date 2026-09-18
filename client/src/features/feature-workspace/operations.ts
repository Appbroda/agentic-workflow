import type { WorkstreamOperation } from '@/schemas/feature';
import { absoluteTime, formatDuration } from '@/utils/time';

/**
 * Liveness, read from the external-operation journal.
 *
 * While an operation runs, its journal row is the only record with a start time and a
 * heartbeat — stage statuses update at attempt boundaries, which is how run 194's graph said
 * "Model pending / Running" eighteen minutes after the coding call had completed while the
 * reviewer was actually working. Everything here is derived from the rows the server serves;
 * nothing is inferred from node states, and nothing here reworks how node states are derived.
 */

/**
 * How long a heartbeat may be quiet before the indicator turns from alive to warning,
 * per operation type — one table, because a 40-minute coding call and a 20-second lint run
 * have different definitions of "quiet too long". The server's stage mapping travels on the
 * row itself (`stage`); these thresholds are the client's reading cadence for each type.
 */
export const HEARTBEAT_THRESHOLD_SECONDS: Record<string, number> = {
  // Setup: a clone or install streams work and heartbeats while it runs, but a large
  // repository or a cold package cache legitimately goes quiet for stretches.
  clone_repository: 180,
  create_branch: 60,
  install_dependencies: 300,
  // Model calls: heartbeats renew on the provider polling loop, so minutes of silence mean
  // the call — not the model — has gone quiet. 186's ReadError showed as exactly this.
  run_coding_executor: 240,
  run_reviewer: 240,
  write_file_changes: 60,
  // Validation commands: short, chatty subprocesses. A lint run quiet for a minute is stuck
  // in a way a coding call quiet for a minute is not.
  run_formatter: 60,
  run_linter: 60,
  run_typecheck: 120,
  run_tests: 300,
  run_build: 300,
  // Publication: small remote calls against the git host.
  create_commit: 60,
  push_branch: 120,
  create_pull_request: 120,
  update_pull_request: 120,
  add_labels: 120,
  add_reviewers: 120,
  // The pre-coding model calls, journaled since 41-: same cadence as the other model calls.
  run_product_manager: 240,
  run_repository_recon: 240,
  run_clarification_grounding: 240,
  run_feature_planner: 240,
};

/** Matches the server journal's own stale-detection window, for types not named above. */
export const DEFAULT_HEARTBEAT_THRESHOLD_SECONDS = 300;

export function heartbeatThresholdSeconds(operationType: string): number {
  return HEARTBEAT_THRESHOLD_SECONDS[operationType] ?? DEFAULT_HEARTBEAT_THRESHOLD_SECONDS;
}

/** The statuses in which an operation is live work rather than a record of finished work. */
const ACTIVE_STATUSES = new Set(['starting', 'running', 'cancellation_requested']);

/** The newest live operation, if any. Rows arrive newest first from the server. */
export function runningOperation(operations: WorkstreamOperation[]): WorkstreamOperation | null {
  return operations.find((operation) => ACTIVE_STATUSES.has(operation.status)) ?? null;
}

export interface NodeLiveness {
  state: 'alive' | 'stale';
  /** The truthful one-liner: "review running — 3m", or "review: no heartbeat for 4m". */
  summary: string;
  /** The tooltip: operation type, attempt number, started-at, and heartbeat age. */
  detail: string;
  repositoryId: string;
}

/**
 * Which node the indicator belongs on: the node of the stage the journal says is running,
 * even when the node states have not caught up — the indicator is the truthful overlay for
 * run 194's lie, and the node-status derivation itself is deliberately untouched (item 7).
 * Stages without a node of their own in the lane (setup, publication) sit on the lane's
 * repository node.
 */
function nodeIdForStage(stage: string, repositoryId: string): string {
  switch (stage) {
    case 'coding':
      return `implement:${repositoryId}`;
    case 'validation':
      return `validate:${repositoryId}`;
    case 'review':
      return `review:${repositoryId}`;
    default:
      return `repo:${repositoryId}`;
  }
}

/** A short age for reading beside a stage name: "2s", "3m 10s". */
function ageText(milliseconds: number): string {
  return formatDuration(milliseconds) ?? `${Math.max(0, Math.round(milliseconds / 1000))}s`;
}

/**
 * One running row's heartbeat, read against this browser's clock with the same per-type
 * thresholds the node indicator uses -- the drill-in and the indicator must not disagree
 * about whether the same operation is quiet too long.
 */
export function heartbeatReading(
  operation: WorkstreamOperation,
  nowMs: number,
): { age: string; stale: boolean } | null {
  const heartbeat = operation.heartbeat_at ?? operation.started_at;
  if (!heartbeat) return null;
  const heartbeatAgeMs = Math.max(0, nowMs - Date.parse(heartbeat));
  return {
    age: ageText(heartbeatAgeMs),
    stale: heartbeatAgeMs / 1000 > heartbeatThresholdSeconds(operation.operation_type),
  };
}

export function livenessForOperation(
  repositoryId: string,
  operation: WorkstreamOperation,
  nowMs: number,
): { nodeId: string; liveness: NodeLiveness } | null {
  const heartbeat = operation.heartbeat_at ?? operation.started_at;
  if (!heartbeat) return null;
  const heartbeatAgeMs = Math.max(0, nowMs - Date.parse(heartbeat));
  const stale = heartbeatAgeMs / 1000 > heartbeatThresholdSeconds(operation.operation_type);
  const age = ageText(heartbeatAgeMs);
  const summary = stale
    ? `${operation.stage}: no heartbeat for ${age}`
    : `${operation.stage} running — ${age}`;
  const detail = [
    operation.operation_type,
    `call ${operation.attempt} of ${operation.max_attempts}`,
    operation.started_at ? `started ${absoluteTime(operation.started_at)}` : null,
    `heartbeat ${age} ago`,
  ]
    .filter(Boolean)
    .join(' · ');
  return {
    nodeId: nodeIdForStage(operation.stage, repositoryId),
    liveness: { state: stale ? 'stale' : 'alive', summary, detail, repositoryId },
  };
}

/** The indicator for every repository with a live operation, keyed by graph node id. */
export function livenessByNode(
  operationsByRepository: ReadonlyMap<string, WorkstreamOperation[]>,
  nowMs: number,
): Map<string, NodeLiveness> {
  const byNode = new Map<string, NodeLiveness>();
  for (const [repositoryId, operations] of operationsByRepository) {
    const operation = runningOperation(operations);
    if (!operation) continue;
    const placed = livenessForOperation(repositoryId, operation, nowMs);
    if (placed) byNode.set(placed.nodeId, placed.liveness);
  }
  return byNode;
}
