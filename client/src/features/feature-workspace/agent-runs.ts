import type { TimelineEvent } from '@/schemas/feature';

/**
 * What each agent did, in order, with how long it took.
 *
 * The timeline is every event the platform recorded; this is the subset that answers "which
 * agent ran, on what, and how did it end". Both are wanted: the timeline for what happened,
 * this for who did it. It is derived from the same events rather than from a second endpoint,
 * because the platform records agent work as those events and a parallel source would drift.
 *
 * Duration is measured between an agent's start event and its next terminal event, so a run
 * still going shows as running rather than as a gap.
 */
export interface AgentRun {
  key: string;
  agent: string;
  repositoryId: string | null;
  startedAt: string;
  endedAt: string | null;
  outcome: 'running' | 'completed' | 'failed';
  /** Which attempt this was, when the platform's own artifact id says so. */
  attempt: number | null;
  /** The result artifact, so a person can open what the run actually produced. */
  artifactId: string | null;
}

/**
 * The attempt number, read from the result artifact's identifier.
 *
 * The platform names these `..._child_workflow_result.<repository>.attempt-N.json`, so the
 * number is its own rather than something inferred by counting rows. Without it, a repository
 * that tried nine times showed nine identical lines.
 */
function attemptFrom(artifactId: string | null): number | null {
  const match = artifactId ? /\.attempt-(\d+)\./.exec(artifactId) : null;
  return match ? Number(match[1]) : null;
}

const STARTS: Record<string, string> = {
  child_workflow_started: 'Repository workstream',
  integration_review_started: 'Integration reviewer',
  feature_started: 'Product manager',
};

const ENDS: Record<string, 'completed' | 'failed'> = {
  child_workflow_completed: 'completed',
  child_workflow_failed: 'failed',
  integration_review_completed: 'completed',
  contract_created: 'completed',
  contract_approved: 'completed',
  feature_completed: 'completed',
  feature_failed: 'failed',
};

export function agentRuns(events: TimelineEvent[]): AgentRun[] {
  const runs: AgentRun[] = [];
  // Keyed by repository so two repositories running at once do not close each other's run.
  const open = new Map<string, AgentRun>();

  events.forEach((event, index) => {
    const repositoryId =
      typeof event.details?.repository_id === 'string' ? event.details.repository_id : null;
    const scope = repositoryId ?? '';

    const artifactId =
      typeof event.details?.artifact_id === 'string' ? event.details.artifact_id : null;

    const started = STARTS[event.event];
    if (started) {
      const run: AgentRun = {
        key: `${event.timestamp}-${index}`,
        agent: started,
        repositoryId,
        startedAt: event.timestamp,
        endedAt: null,
        outcome: 'running',
        attempt: attemptFrom(artifactId),
        artifactId,
      };
      open.set(scope, run);
      runs.push(run);
      return;
    }

    // Every artifact carries its producer. These are the platform's most precise agent
    // history records: product-manager, planner, engineer, reviewer and GitHub outputs remain
    // visible even when the lifecycle table only has an aggregate child-workstream event.
    if (event.event_type === 'artifact' && artifactId && event.source !== 'api') {
      runs.push({
        key: `${event.timestamp}-${index}-artifact`,
        agent: event.source,
        repositoryId,
        startedAt: event.timestamp,
        endedAt: event.timestamp,
        outcome: 'completed',
        attempt: attemptFrom(artifactId),
        artifactId,
      });
      return;
    }

    const ended = ENDS[event.event];
    if (ended) {
      const run = open.get(scope);
      if (run) {
        run.endedAt = event.timestamp;
        run.outcome = ended;
        run.artifactId = artifactId ?? run.artifactId;
        run.attempt = run.attempt ?? attemptFrom(artifactId);
        open.delete(scope);
      }
    }
  });

  return runs.reverse();
}


/**
 * How long a run took, or nothing when the events cannot say.
 *
 * Not every pair brackets real work. A child workstream's `started` and `failed` events are
 * both written when its result is persisted -- the same millisecond, after the attempt has
 * already finished -- so subtracting them yields `0s` for an attempt that ran for minutes.
 * Printing that would be inventing a measurement the platform never took, which is worse than
 * leaving the column empty: a reader would believe it.
 *
 * So a difference under a second is reported as unknown rather than as zero.
 */
export function duration(run: AgentRun): string | null {
  if (!run.endedAt) return 'running';
  const ms = Date.parse(run.endedAt) - Date.parse(run.startedAt);
  if (!Number.isFinite(ms) || ms < MINIMUM_MEASURABLE_MS) return null;
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return minutes < 60
    ? `${minutes}m ${seconds % 60}s`
    : `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

const MINIMUM_MEASURABLE_MS = 1000;
