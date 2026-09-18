import type { ExecutionRecord } from '@/schemas/feature';
import { modelDisplayName, modelLabel } from '@/utils/model';

/**
 * Execution records, attached to the arrows they describe.
 *
 * The server answers "who moved this feature from one stage to the next" in its own stage
 * vocabulary. The graph draws nodes with its own identifiers. This is the only place the two
 * are joined, and it is the only place in the client that decides what an arrow says.
 *
 * Two things are deliberately not done here. Nothing derives a model: an arrow shows the model
 * the record carries, or says that no model has been resolved. And nothing re-counts an
 * attempt: `attempt`, `max_attempts` and the sentence explaining what they count all come from
 * the backend, which is the thing that enforces them.
 */

/** The server's stage vocabulary, mapped onto the node identifiers this graph draws. */
const STAGE_NODES: Record<string, (repositoryId: string | null) => string | null> = {
  request: () => 'request',
  technical_prd: () => 'technical_prd',
  integration_contract: () => 'integration_contract',
  execution_plan: () => 'execution_plan',
  repository: (repositoryId) => (repositoryId ? `repo:${repositoryId}` : null),
  implementation: (repositoryId) => (repositoryId ? `implement:${repositoryId}` : null),
  validation: (repositoryId) => (repositoryId ? `validate:${repositoryId}` : null),
  review: (repositoryId) => (repositoryId ? `review:${repositoryId}` : null),
  integration_review: () => 'integration_review',
  pull_requests: () => 'pull_requests',
};

/** Stages that exist once per repository, so a record without one spans every lane. */
const PER_REPOSITORY = new Set(['repository', 'implementation', 'validation', 'review']);

export function nodeIdFor(stage: string, repositoryId: string | null): string | null {
  return STAGE_NODES[stage]?.(repositoryId) ?? null;
}

export function edgeKey(from: string, to: string): string {
  return `${from}->${to}`;
}

/**
 * Group every execution by the arrow it belongs to.
 *
 * A record with no repository whose stage is per-repository -- the integration review, which
 * happens once for a feature but leaves every repository's review node -- is attached to each
 * lane's arrow. It is one execution seen from several lanes, not several executions, so the
 * renderer shows its label once.
 */
export function executionsByEdge(
  executions: ExecutionRecord[],
  repositoryIds: string[],
): Map<string, ExecutionRecord[]> {
  const byEdge = new Map<string, ExecutionRecord[]>();
  const add = (from: string | null, to: string | null, record: ExecutionRecord) => {
    if (!from || !to) return;
    const key = edgeKey(from, to);
    const existing = byEdge.get(key);
    if (existing) existing.push(record);
    else byEdge.set(key, [record]);
  };

  for (const record of executions) {
    const repositoryId = record.repository_id ?? null;
    const spansLanes =
      repositoryId === null &&
      (PER_REPOSITORY.has(record.from_stage) || PER_REPOSITORY.has(record.to_stage));
    const lanes = spansLanes ? repositoryIds : [repositoryId];
    for (const lane of lanes) {
      add(nodeIdFor(record.from_stage, lane), nodeIdFor(record.to_stage, lane), record);
    }
  }
  return byEdge;
}

/** Statuses in which an arrow has something to say about a model that has been chosen. */
const RESOLVED_STATUSES = new Set(['queued', 'running', 'completed', 'failed', 'needs_human']);

const STATUS_WORDS: Record<string, string> = {
  pending: 'Not started',
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  blocked: 'Blocked',
  needs_human: 'Needs a person',
  cancelled: 'Cancelled',
};

export function statusWord(status: string): string {
  return STATUS_WORDS[status] ?? status.replace(/_/g, ' ');
}

/**
 * The handler line: which model, or which deterministic handler, or that neither is known yet.
 *
 * A model-backed transition with no recorded model is two different situations and says so.
 * Before it runs, the platform has not chosen one; afterwards, its record does not carry one --
 * a mock run, or an artifact written before the platform recorded execution metadata. Neither
 * is an invitation to print a plausible model name.
 */
export function handlerLine(record: ExecutionRecord): string | null {
  if (record.handler_type !== 'model') return record.handler;
  const named = modelLabel(record.model, record.reasoning_effort);
  if (named) return named;
  if (!RESOLVED_STATUSES.has(record.status)) return null;
  // Short enough to sit in the gap between two columns without truncating. The drawer says it
  // in full -- "Model selected at execution" -- and the accessible name says it in words.
  return record.status === 'queued' || record.status === 'running'
    ? 'Model pending'
    : 'Model not recorded';
}

/**
 * The same fact, at the length a detail panel has room for.
 *
 * Without the effort: the drawer gives that its own row, and repeating it here made one value
 * look like two.
 */
export function handlerDetail(record: ExecutionRecord): string | null {
  if (record.handler_type !== 'model') return record.handler;
  const named = modelDisplayName(record.model);
  if (named) return named;
  return record.status === 'queued' || record.status === 'running' || record.status === 'pending'
    ? 'Model selected at execution'
    : 'Model not recorded';
}

/**
 * One row of an arrow's label.
 *
 * The kind decides how it is set, and the distinction that matters is `model`: a model
 * identifier must never be broken across lines, because half an identifier is a different
 * model. Prose -- a handler name, a status -- may wrap.
 */
export interface EdgeLabelLine {
  text: string;
  kind: 'attempt' | 'model' | 'handler' | 'tail';
  /**
   * The effort the model was asked for, kept separate from its name so the two can fall onto
   * two lines in a narrow gap. A model identifier must not break in the middle; "GPT-5.6 Sol"
   * above "effort: max" is readable, and "GPT-5.6 So…" is not.
   */
  effort?: string;
}

/**
 * One word for why an arrow failed, from the platform's recorded classification.
 *
 * Two vocabularies arrive in `failure_classification`, one per kind of record. A deterministic
 * validation record carries the gate that failed -- the failing command's own `validation_type`
 * -- because "which gate" is the word that saves a click there. A remediation record carries
 * the retry policy's classification, because "which budget this consumed" is that arrow's
 * question. A remediation record can also carry a feature-level classification instead, for
 * the attempts that stopped on something no retry budget covers -- an external service that
 * did not answer. Only values the platform actually writes are mapped: an unknown value stays
 * off the arrow -- the drawer still says it in full -- rather than being abbreviated into a
 * guess.
 */
const FAILURE_CLASS_WORDS: Record<string, string> = {
  // The gate that failed, on a validator arrow.
  build: 'build',
  test: 'own tests',
  lint: 'lint',
  typecheck: 'typecheck',
  format: 'format',
  custom: 'custom check',
  // The retry policy's classification, on a remediation arrow.
  implementation_missing: 'incomplete',
  validation_source_failure: 'own checks',
  validation_capacity_failure: 'capacity',
  validation_configuration_failure: 'validation config',
  dependency_installation_failure: 'dependencies',
  test_infrastructure_missing: 'tests missing',
  review_scope_failure: 'review scope',
  contract_mismatch: 'contract',
  // The three external services, on the same arrow. A workstream that spent its fault
  // allowance carries one of these rather than a retry-policy class, and an operator retry
  // afterwards makes it the previous attempt a remediation arrow reads -- so the arrow had a
  // classification and showed nothing. The word names the service, which is the whole reason
  // these are three classifications and not one. A missing word is worse than a vague one:
  // `design_source_unavailable` had no entry here until 2026-09-12 and the arrow that ended
  // AB-Feature-228 rendered blank.
  provider_unavailable: 'provider',
  git_remote_unavailable: 'git remote',
  design_source_unavailable: 'design source',
};

/** The class word an arrow shows, only where the arrow already reads as a failure. */
export function failureClassWord(record: ExecutionRecord): string | null {
  if (record.status !== 'failed' && record.status !== 'needs_human') return null;
  if (!record.failure_classification) return null;
  return FAILURE_CLASS_WORDS[record.failure_classification] ?? null;
}

export interface EdgeLabel {
  /** The rows drawn on the arrow, in order. Never more than three. */
  lines: EdgeLabelLine[];
  /** How many earlier executions this arrow also holds. */
  previous: number;
  /**
   * The record the lines describe: the newest execution on this arrow. Absent only for the
   * fallback label below, which has no execution behind it and therefore opens nothing.
   */
  record?: ExecutionRecord;
  /** One sentence carrying everything the lines and the colour carry. */
  description: string;
}

/**
 * What an arrow says when the platform recorded no execution for it.
 *
 * The graph has always been able to draw a retry loop from the workstream's own counters, and
 * it keeps doing so: a feature whose executions have not loaded yet, or one recorded before
 * this endpoint existed, still shows "Retry 3 / 8" rather than losing the loop's meaning. It
 * simply has nothing to open, because there is no execution record to show.
 */
export function fallbackEdgeLabel(text: string | undefined): EdgeLabel | null {
  if (!text) return null;
  return { lines: [{ text, kind: 'attempt' }], previous: 0, description: text };
}

/**
 * What one arrow says, from the executions attached to it.
 *
 * Kept short on purpose. A retry is worth two lines -- which attempt, and what is answering it
 * -- and a status word is added only where the arrow is not simply finished. Everything else
 * is in the drawer.
 */
export function edgeLabel(records: ExecutionRecord[]): EdgeLabel | null {
  const record = records.at(-1);
  if (!record) return null;
  const previous = records.length - 1;
  const lines: EdgeLabelLine[] = [];

  if (record.is_retry) {
    if (record.status === 'needs_human') {
      lines.push({ text: 'No further retry', kind: 'attempt' });
    } else if (record.attempt !== null && record.attempt !== undefined) {
      lines.push({
        text: record.max_attempts
          ? `Retry ${record.attempt} / ${record.max_attempts}`
          : `Retry ${record.attempt}`,
        kind: 'attempt',
      });
    }
  }

  // "Human action" beside "No further retry" and "Needs a person" is the same fact three times,
  // and a four-line label on a loop edge overlapped the lane below it.
  if (record.handler_type === 'model' && record.model) {
    lines.push({
      text: modelDisplayName(record.model) ?? record.model,
      kind: 'model',
      ...(record.reasoning_effort ? { effort: record.reasoning_effort } : {}),
    });
  } else if (record.handler_type !== 'human' || lines.length === 0) {
    const handler = handlerLine(record);
    if (handler) lines.push({ text: handler, kind: 'handler' });
  }

  // Three rows at most, so the height of a loop label is bounded and the graph's row spacing
  // can be relied on. The status, the failure class and the attempt history share the last one.
  const tail = [
    record.status !== 'completed' && record.status !== 'pending' ? statusWord(record.status) : '',
    failureClassWord(record) ?? '',
    previous > 0 ? `+${previous} earlier` : '',
  ]
    .filter(Boolean)
    .join(' · ');
  if (tail) lines.push({ text: tail, kind: 'tail' });
  if (lines.length === 0) return null;

  return {
    lines,
    previous,
    record,
    description: edgeDescription(record, previous),
  };
}

/**
 * The accessible name for an arrow.
 *
 * Says in words everything the shape, the colour and the two short lines say, in the order a
 * person would ask for it: what kind of transition, between which stages, on which attempt,
 * performed by what, and how it is going.
 */
export function edgeDescription(record: ExecutionRecord, previous = 0): string {
  const stage = (value: string) => value.replace(/_/g, ' ');
  const attempt =
    record.attempt === null || record.attempt === undefined
      ? ''
      : record.max_attempts
        ? `${record.is_retry ? 'Retry attempt' : 'Attempt'} ${record.attempt} of ${record.max_attempts}`
        : `${record.is_retry ? 'Retry attempt' : 'Attempt'} ${record.attempt}`;
  const by =
    record.handler_type === 'model'
      ? (modelLabel(record.model, record.reasoning_effort) ?? 'no model resolved yet')
      : record.handler;
  const failureClass = failureClassWord(record);
  return [
    record.is_retry ? 'Retry' : 'Execution',
    `from ${stage(record.from_stage)} to ${stage(record.to_stage)}`,
    record.repository_id ? `in ${record.repository_id}` : '',
    attempt,
    `handled by ${by}`,
    statusWord(record.status).toLowerCase(),
    failureClass ? `failure class ${failureClass}` : '',
    previous > 0 ? `${previous} earlier ${previous === 1 ? 'execution' : 'executions'}` : '',
  ]
    .filter(Boolean)
    .join(', ');
}

/** The tone an arrow is drawn in, which is never the only thing saying what it means. */
export function edgeTone(record: ExecutionRecord): string {
  if (record.status === 'failed') return 'failed';
  if (record.status === 'needs_human' || record.status === 'blocked') return 'attention';
  if (record.status === 'running' || record.status === 'queued') return 'working';
  if (record.status === 'cancelled') return 'stopped';
  if (record.status === 'completed') return 'done';
  return 'pending';
}
