import type { ExecutionRecord, WorkstreamAttempt, WorkstreamOperation } from '@/schemas/feature';
import { modelDisplayName } from '@/utils/model';
import { humanise } from '@/utils/text';
import { formatDuration } from '@/utils/time';

/**
 * The agent's work, derived from the journal rows the 55- endpoint serves.
 *
 * Everything here is a pure reading of `WorkstreamOperation[]` as the server returns them --
 * newest first, bounded, each row carrying a server-owned `stage` name. Nothing is fetched
 * here and nothing re-derives node states: this is the drill-in's data shape, the view a
 * person needed four times during the 185-194 cycle and got only by querying Postgres.
 */

/**
 * The stages in execution order. `planning` precedes the fan-out and `other` is the server's
 * name for a type it has not mapped; both render only when rows exist for them, where the
 * five in the middle always render so "not yet seen" is visible as such.
 */
export const STAGE_ORDER = [
  'planning',
  'setup',
  'coding',
  'validation',
  'review',
  'publication',
  'other',
] as const;

/** The five stages a repository attempt always passes through, shown even before rows exist. */
const ALWAYS_SHOWN_STAGES = new Set(['setup', 'coding', 'validation', 'review', 'publication']);

/**
 * The agent that owns each stage, so the box reads as "what is this agent doing". Stages
 * without an owner here (planning, other, anything the server names later) keep their stage
 * word as the primary label rather than having an agent guessed for them.
 */
export const AGENT_FOR_STAGE: Record<string, string> = {
  setup: 'Engineer',
  coding: 'Engineer',
  validation: 'Engineer',
  review: 'Reviewer',
  publication: 'Publisher',
};

/** The statuses in which an operation is live work rather than a record of finished work. */
const RUNNING_STATUSES = new Set(['starting', 'running', 'cancellation_requested']);
const FAILED_STATUSES = new Set(['failed_retryable', 'failed_terminal']);

export type RowState = 'succeeded' | 'failed' | 'running' | 'other';

/** How one row reads: the glyph, and the word the glyph stands for. Never one without the other. */
export function rowState(operation: WorkstreamOperation): RowState {
  if (operation.status === 'succeeded') return 'succeeded';
  if (FAILED_STATUSES.has(operation.status)) return 'failed';
  if (RUNNING_STATUSES.has(operation.status)) return 'running';
  return 'other';
}

export const ROW_GLYPHS: Record<RowState, string> = {
  succeeded: '✓',
  failed: '✗',
  running: '●',
  other: '·',
};

/**
 * The word each glyph stands for, so a line carrying only a glyph can still be read aloud.
 *
 * The drawer's rows can lean on their note for this -- a failed row names its error code
 * beside the cross -- but a graph node's line has room for the name and little else, so the
 * word has to come from somewhere that is not the note.
 */
export const ROW_WORDS: Record<RowState, string> = {
  succeeded: 'succeeded',
  failed: 'failed',
  running: 'running',
  other: 'recorded',
};

/**
 * What one `write_file_changes` row is, decided by its position in the attempt rather than by
 * its type.
 *
 * The server maps that one operation type to the coding stage, and two different callers write
 * under it. `feature_runtime.py::_write_contract_projection` writes the approved contract's
 * OpenAPI projection into the checkout on every attempt, before the Engineer is called at all;
 * the implementation write is journaled by `llm_adapter.py` inside the coding call itself, so
 * its row is always created after that call's `run_coding_executor` row. Run 198 FE showed the
 * first of those filed as "Engineer · coding ✓" one second after `create_branch`, which reads
 * as code written while dependencies were still installing.
 *
 * The position is the fact: a write with no coding call before it in this attempt cannot be
 * the implementation. The stage moves; the raw type still renders as itself.
 */
export type WriteRole = 'pre_coding' | 'implementation';

/**
 * Which writes this attempt can speak about, keyed by operation id.
 *
 * Only an attempt whose first row is its `clone_repository` is judged. The endpoint's row
 * budget truncates the oldest history it serves, and in a truncated slice the coding call a
 * write followed may simply be missing -- reclassifying there would file the implementation
 * write under setup, which is the same lie in the other direction.
 */
export function writeRoles(chronological: WorkstreamOperation[]): Map<string, WriteRole> {
  const roles = new Map<string, WriteRole>();
  if (chronological[0]?.operation_type !== 'clone_repository') return roles;
  let codingCalled = false;
  for (const operation of chronological) {
    if (operation.operation_type === 'run_coding_executor') codingCalled = true;
    else if (operation.operation_type === 'write_file_changes') {
      roles.set(operation.operation_id, codingCalled ? 'implementation' : 'pre_coding');
    }
  }
  return roles;
}

/** Where a pre-coding write renders instead: it is workspace preparation, and setup owns that. */
const PRE_CODING_WRITE_STAGE = 'setup';

/**
 * The phase an attempt is in when its child is running and no journaled operation is.
 *
 * Run 197 FE: every row ticked, the last at 09:47:42, nothing running -- while the Engineer had
 * been seven minutes inside the checks it runs after the coding call. Those checks are
 * unjournaled by construction (`feature_runtime.py` builds the scoped test runner, typecheck
 * runner, reachability and assigned-file checkers and the self-reviewer as non-authoritative
 * in-attempt tools), so the drawer rendered the busiest window of an attempt as silence.
 *
 * One derived row, naming the phase and its start time. It never names a sub-step: which check
 * is running is not journaled, and 57-'s granularity rule is that the screen may only say what
 * a record says.
 */
export interface UnjournaledPhase {
  /** The stage group this row renders at the end of -- the stage of the last journaled row. */
  stage: string;
  /** What the phase is. */
  label: string;
  /** What it contains, and why it has no rows of its own. */
  detail: string;
  /** The last journaled operation's completion: the phase's start, as the journal has it. */
  since: string;
}

/** The checks the Engineer runs after its coding call, none of them journaled individually. */
const IN_ATTEMPT_CHECK_DETAIL =
  'commit gate, typecheck, scoped tests, wiring and placement checks, self-review and repair ' +
  'passes — the platform journals none of them individually';

export type StageRow =
  | { kind: 'operation'; operation: WorkstreamOperation; write: WriteRole | null }
  | { kind: 'phase'; phase: UnjournaledPhase };

export interface StageGroup {
  stage: string;
  /** The agent that owns the stage, where one does. */
  agent: string | null;
  /** Chronological, oldest first -- the order the work happened in. */
  rows: StageRow[];
}

export interface AttemptPartition {
  /**
   * Chronological groups, oldest first; the last one is the current attempt. A single group
   * means either one attempt or an ambiguous history -- either way, no divider is drawn.
   */
  attempts: WorkstreamOperation[][];
}

/** One entry of the attempt dropdown: the rows of one attempt, and which attempt that is. */
export interface AttemptView {
  /** Stable across polls, so an open selection survives the next refresh. */
  key: string;
  /** The number the journal stamped, or null for the rows that predate the stamp. */
  number: number | null;
  /** Chronological, oldest first -- the order `groupByStage` and `writeRoles` both need. */
  operations: WorkstreamOperation[];
}

/** The attempt one row is stamped with, or null when it carries no usable stamp. */
function stampOf(operation: WorkstreamOperation): number | null {
  const value = operation.child_attempt;
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null;
}

/**
 * The attempts this journal contains, oldest first, from the stamp the server now writes.
 *
 * The stamp is the fact: `feature_runtime.py` builds one operation executor per attempt and it
 * labels every row it journals with that attempt's number, so clone, installs, the coding call
 * and its write, the validation commands, the commit and the push all say which attempt they
 * belong to. Run 197 BE is what that replaces -- attempts 0, 1 and 2 rendered as one CURRENT
 * ATTEMPT, because the only confident divider was a fresh `clone_repository` and a
 * retry-edits-in-place retry never re-clones.
 *
 * Rows with no stamp collect into exactly one entry rather than being assigned to a
 * neighbouring attempt: a row written before the stamp existed, and a row belonging to no
 * attempt at all (reconnaissance is scoped to the feature, not to an attempt), are both
 * genuinely unattributable. Inside that one entry the clone heuristic still draws whatever
 * dividers it can prove, which is exactly the job it had before and no more -- so a wholly
 * pre-stamp journal reads as it always did, and the heuristic never competes with the stamp.
 */
export function attemptViews(operations: WorkstreamOperation[]): AttemptView[] {
  // The server answers newest first; the reading order here is the order the work happened.
  const chronological = [...operations].reverse();
  const byNumber = new Map<number, WorkstreamOperation[]>();
  const unstamped: WorkstreamOperation[] = [];
  for (const operation of chronological) {
    const number = stampOf(operation);
    if (number === null) {
      unstamped.push(operation);
      continue;
    }
    const existing = byNumber.get(number);
    if (existing) existing.push(operation);
    else byNumber.set(number, [operation]);
  }
  const views: AttemptView[] = [];
  if (unstamped.length > 0) {
    views.push({ key: 'earlier-history', number: null, operations: unstamped });
  }
  // By number, not by first appearance: the endpoint's row budget can truncate an attempt's
  // opening rows, and an attempt whose remaining rows happen to be interleaved must still sit
  // where its number puts it.
  for (const number of [...byNumber.keys()].sort((left, right) => left - right)) {
    views.push({ key: `attempt-${number}`, number, operations: byNumber.get(number)! });
  }
  return views;
}

/**
 * Partition journal rows into attempts, confidently or not at all.
 *
 * Since the server stamps each row with its attempt (`attemptViews` above), this reads only the
 * rows that carry no stamp: whatever a deployment journaled before the stamp existed. That is
 * the whole of its remaining job, and the reason it survives rather than being deleted -- those
 * rows are still served, and the heuristic is still the best that can be said about them.
 *
 * The one reliable marker is a fresh `clone_repository`: a granted retry runs a real attempt
 * from clone onwards, and a crash-continued run that lost its workspace clones again.
 * `install_dependencies` is deliberately NOT a boundary even though setup owns it -- the
 * server journals custom validation commands and repository repairs under that same type
 * (`validation_tools.py`, `feature_runtime.py`), so an install row can appear mid-attempt and
 * would plant a divider where no attempt boundary is. A review-cycle retry edits in place
 * without cloning, so it stays inside its group rather than being guessed at. A wrong divider
 * is worse than none.
 */
export function partitionAttempts(operations: WorkstreamOperation[]): AttemptPartition {
  // The server answers newest first; the reading order here is the order the work happened.
  const chronological = [...operations].reverse();
  const attempts: WorkstreamOperation[][] = [];
  let current: WorkstreamOperation[] = [];
  for (const operation of chronological) {
    if (operation.operation_type === 'clone_repository' && current.length > 0) {
      attempts.push(current);
      current = [];
    }
    current.push(operation);
  }
  if (current.length > 0) attempts.push(current);
  return { attempts };
}

/**
 * The stage one row renders under: the server's, unless this attempt's own order corrects it.
 *
 * The correction is the pre-coding write only. A client must never work a stage out from an
 * operation type's spelling -- that mapping is the server's -- but which of two callers wrote
 * a row is a question the server's type table cannot answer and the row order can.
 */
function stageOf(operation: WorkstreamOperation, write: WriteRole | null): string {
  return write === 'pre_coding' ? PRE_CODING_WRITE_STAGE : operation.stage;
}

/**
 * The unjournaled phase to render, or nothing.
 *
 * Nothing unless the workstream is recorded as running with no journaled operation running:
 * a settled workstream is silent because it is finished, and a running operation already has
 * its own row with its own heartbeat. The phase starts where the journal stops, so a last row
 * that never recorded a completion produces no row rather than a guessed start time.
 */
export function unjournaledPhase(
  chronological: WorkstreamOperation[],
  { childRunning }: { childRunning: boolean },
): UnjournaledPhase | null {
  if (!childRunning) return null;
  if (chronological.some((operation) => RUNNING_STATUSES.has(operation.status))) return null;
  const last = chronological.at(-1);
  if (!last?.completed_at) return null;
  const stage = stageOf(last, writeRoles(chronological).get(last.operation_id) ?? null);
  // The one phase whose content is known: the checks that follow a completed coding call.
  // Anywhere else the honest answer is that the platform is between journaled operations,
  // which is what the row says rather than naming work it cannot see.
  const inAttemptChecks = stage === 'coding' && last.status === 'succeeded';
  return {
    stage,
    label: inAttemptChecks ? 'in-attempt checks' : 'not journaled',
    detail: inAttemptChecks
      ? IN_ATTEMPT_CHECK_DETAIL
      : 'the workstream is recorded as running and no operation is journaled',
    since: last.completed_at,
  };
}

/**
 * Group one attempt's rows by stage, in execution order.
 *
 * Rows arrive chronologically, as `partitionAttempts` returns them: the write-role rule reads
 * that order, and a reversed slice would file every write under setup.
 *
 * When `includePending` is set (the current attempt), the five stages a repository attempt
 * always passes through render even without rows, so "not yet seen" is a visible state rather
 * than an absence. Stages the order does not know -- a server-side addition -- render after
 * publication rather than being dropped, in the order they first appear. `phase` is the derived
 * row for an attempt whose work has left the journal; it joins the group of the stage it names,
 * last, and creates that group when the stage has no rows of its own.
 */
export function groupByStage(
  operations: WorkstreamOperation[],
  { includePending, phase }: { includePending: boolean; phase?: UnjournaledPhase | null },
): StageGroup[] {
  const roles = writeRoles(operations);
  const byStage = new Map<string, StageRow[]>();
  const append = (stage: string, row: StageRow) => {
    const existing = byStage.get(stage);
    if (existing) existing.push(row);
    else byStage.set(stage, [row]);
  };
  for (const operation of operations) {
    const write = roles.get(operation.operation_id) ?? null;
    append(stageOf(operation, write), { kind: 'operation', operation, write });
  }
  if (phase) append(phase.stage, { kind: 'phase', phase });

  const groups: StageGroup[] = [];
  for (const stage of STAGE_ORDER) {
    const rows = byStage.get(stage);
    byStage.delete(stage);
    if (rows) groups.push({ stage, agent: AGENT_FOR_STAGE[stage] ?? null, rows });
    else if (includePending && ALWAYS_SHOWN_STAGES.has(stage)) {
      groups.push({ stage, agent: AGENT_FOR_STAGE[stage] ?? null, rows: [] });
    }
  }
  // Anything left is a stage name this order has never heard of. Served, so shown.
  for (const [stage, rows] of byStage) {
    groups.push({ stage, agent: AGENT_FOR_STAGE[stage] ?? null, rows });
  }
  return groups;
}


/**
 * What a row's re-issue count says, in words — or nothing, which is the usual answer.
 *
 * A re-issue happens when the provider accepted the request and its stream said nothing
 * inside the first-event budget: the adapter closes it and asks again. It costs one request,
 * writes no journal row of its own, and the caller sees an ordinary answer — so this note is
 * the only place a person can see it happened at all.
 *
 * `0` and `null` both render nothing, deliberately, even though they mean different things:
 * "the stream spoke first time" is the overwhelmingly common case and a badge on every
 * healthy row would be noise. The distinction survives in the data for whoever queries it.
 */
export function reissueNote(count: number | null): string | null {
  if (count === null || count <= 0) return null;
  if (count === 1) return 're-issued once after a silent stream';
  if (count === 2) return 're-issued twice after a silent stream';
  return `re-issued ${count} times after a silent stream`;
}

/**
 * What one repeated row's served differentiator says, in words.
 *
 * The server decided *why* the row repeats and published a typed field; this decides only how
 * to say it. A `kind` this client has never heard of falls back to its served detail, because
 * a later server's new answer is still an answer.
 */
export function repeatNote(repeat: WorkstreamOperation['repeat']): string | null {
  if (!repeat) return null;
  if (repeat.kind === 'new_revision') {
    return repeat.detail
      ? `re-run at new revision ${repeat.detail} (after an in-attempt repair)`
      : 're-run at a new revision (after an in-attempt repair)';
  }
  if (repeat.kind === 'different_command') {
    return repeat.detail ? `a different command · ${repeat.detail}` : 'a different command';
  }
  // Nothing recorded distinguishes it, so nothing is claimed about why. Relative wording, not
  // an ordinal: the response is bounded, and "2nd" changes meaning when the window truncates.
  if (repeat.kind === 'same_step') return 'another run of the same step';
  return repeat.detail ?? humanise(repeat.kind);
}

/* ------------------------------------------------------------------ where an attempt ended */

/**
 * How one stage of one attempt reads: the glyph, the word it stands for, and what the record
 * licenses beside it.
 *
 * Six states, and the two greyed ones are the point of the whole thing. Before this, a
 * finished attempt showed only the stages that ran, all of them ticked — run 201's backend
 * attempt 0 rendered as an unbroken column of green while the self-review gate had stopped it
 * before the reviewer was called. A person could not see where it stopped or why.
 */
export type StageState =
  | 'succeeded'
  | 'failed'
  | 'running'
  | 'other'
  /** The latest attempt has not reached this stage yet. Today's wording, unchanged. */
  | 'pending'
  /** This stage never ran, was inherited, or was never recorded. Greyed, and it says which. */
  | 'skipped';

export const STAGE_GLYPHS: Record<StageState, string> = {
  succeeded: '✓',
  failed: '✗',
  running: '●',
  other: '·',
  pending: '○',
  skipped: '○',
};

export interface StageReading {
  state: StageState;
  /** The word the glyph stands for. Never one without the other. */
  word: string;
  /** Sentences the record licenses beside the word, in reading order. May be empty. */
  notes: string[];
}

/**
 * The endings that are not failures, so no ✗ is drawn for them.
 *
 * `superseded` is the one worth naming: the attempt completed its own cycle and a later one
 * reopened it. Its last stage keeps its rows and the publication header carries the
 * supersession wording — a cross there would say the attempt failed, which it did not.
 */
const DELIVERED_ENDINGS = new Set(['approved', 'published']);

/**
 * The endings that DO put a ✗ on the stage they name, listed rather than inferred.
 *
 * Deliberately a positive list. An `ended_by` this client has never heard of -- a value a
 * later server added -- takes no cross: a cross this client cannot justify looks exactly like
 * one that was earned, which is the whole class of mistake this item exists to remove.
 */
const FAILURE_ENDINGS = new Set([
  'self_review',
  'review_rejected',
  'validation_failed',
  'refusal',
  'fault',
]);

/**
 * The word each ending kind stands for.
 *
 * The server's vocabulary, said in words a person reads. An `ended_by` this client has never
 * heard of — a value a later server added — renders its own name humanised and takes no ✗:
 * a cross this client cannot justify is worse than no cross, because it looks the same as
 * one that was earned.
 */
const ENDING_WORDS: Record<string, string> = {
  self_review: 'stopped at the self-review gate',
  review_rejected: 'review sent the attempt back',
  validation_failed: 'validation failed',
  refusal: 'the attempt refused to proceed',
  fault: 'the platform classified a fault',
  superseded: 'superseded',
  approved: 'approved',
  published: 'published',
};

/**
 * Which stages a workspace carried over from the previous attempt.
 *
 * The mechanism, stated carefully because two plausible-sounding versions of it are false.
 * A `preserved` retry runs against the checkout the previous attempt left, so an empty
 * `setup` there is work that was inherited rather than work that never happened. A
 * `recovered_coding_output` attempt goes further: the coding output was recovered rather
 * than regenerated, so its `coding` stage can be empty on an attempt that demonstrably
 * produced code, and "never ran" about that is the lie this state exists to prevent.
 *
 * `fresh_checkout` and `reset` inherit nothing and appear here as nothing. The attempt
 * NUMBER decides none of this — a retry whose checkout was unusable provisions a
 * replacement exactly as attempt 0 would. The recorded workspace value decides.
 *
 * It applies only to an EMPTY stage. 201's nine backend retries each journal exactly one
 * `install_dependencies` row under `setup`, and a stage with rows renders them and claims
 * nothing about carrying anything over.
 */
const INHERITED_STAGES: Record<string, ReadonlySet<string>> = {
  preserved: new Set(['setup']),
  recovered_coding_output: new Set(['setup', 'coding']),
};

/** Whether a served ending puts a ✗ anywhere. Only the named failure kinds do. */
export function endingIsFailure(ending: WorkstreamAttempt): boolean {
  return FAILURE_ENDINGS.has(ending.ended_by);
}

/** The word one ending stands for, the server's own where this client knows it. */
export function endingWord(ending: WorkstreamAttempt): string {
  return ENDING_WORDS[ending.ended_by] ?? humanise(ending.ended_by);
}

/**
 * How one stage reads, given the ending the server served for this attempt.
 *
 * The four-way precedence for an empty stage, tested in this order so that a stage never has
 * two possible words:
 *
 * 1. `carried over` — the recorded workspace inherited this stage's work. It outranks both
 *    of the next two, because an inherited stage is neither still coming nor skipped.
 * 2. `never ran` — the attempt has a served ending and this stage comes after it.
 * 3. `not yet seen` — the latest attempt, mid-run, has not reached this stage. Today's
 *    wording, unchanged, and the state an in-flight attempt is always in: it has no ending
 *    by definition, so it has no workspace fact either and rule 1 cannot fire for it.
 * 4. `not recorded` — anything else. The platform must not claim "never ran" about a stage
 *    it has no ending to anchor against.
 */
export function stageReading(
  group: StageGroup,
  { ending, isLatest }: { ending: WorkstreamAttempt | null; isLatest: boolean },
): StageReading {
  const endedHere = ending !== null && ending.stage === group.stage;
  const notes: string[] = [];
  // The publication rows are the attempt persisting its work to the branch, which is not the
  // same thing as shipping. `create_commit ✓ / push_branch ✓` under a stage called
  // "publication" reads as an acceptance the attempt never got, so where the ending is not
  // `published` the header says what those rows actually are.
  if (
    group.stage === 'publication' &&
    group.rows.length > 0 &&
    (ending === null || ending.ended_by !== 'published')
  ) {
    notes.push('work committed to the branch');
  }
  if (endedHere && ending.detail) notes.push(ending.detail);

  if (endedHere && endingIsFailure(ending)) {
    // Even when every journaled row inside it succeeded. 201's attempt 2 ticked
    // `run_reviewer`, `create_commit` and `push_branch` and a later attempt still followed:
    // `run_reviewer` succeeding means the call answered, and the answer may be a rejection.
    // The row keeps its ✓; the stage says why the attempt still ended here.
    return { state: 'failed', word: endingWord(ending), notes };
  }
  if (endedHere) {
    return {
      state: DELIVERED_ENDINGS.has(ending.ended_by) ? 'succeeded' : 'other',
      word: endingWord(ending),
      notes,
    };
  }
  if (group.rows.length > 0) return { ...rowsReading(group), notes };
  return { ...emptyReading(group.stage, { ending, isLatest }), notes };
}

/** How a stage with rows reads, summarising the rows without contradicting any of them. */
function rowsReading(group: StageGroup): Omit<StageReading, 'notes'> {
  const operations = group.rows.flatMap((row) => (row.kind === 'operation' ? [row.operation] : []));
  if (operations.some((operation) => rowState(operation) === 'running')) {
    return { state: 'running', word: 'running' };
  }
  if (operations.length > 0 && operations.every((operation) => rowState(operation) === 'succeeded')) {
    return { state: 'succeeded', word: 'every step succeeded' };
  }
  // A stage holding a `failed_retryable` row that a later call answered did not fail, and a
  // stage whose only row is the derived unjournaled phase has nothing to summarise. Neither
  // is a tick and neither is a cross; the rows themselves say which is which.
  return { state: 'other', word: 'not every step succeeded' };
}

/** How an empty stage reads: the four-way precedence, in order. */
function emptyReading(
  stage: string,
  { ending, isLatest }: { ending: WorkstreamAttempt | null; isLatest: boolean },
): Omit<StageReading, 'notes'> {
  const inherited = ending?.workspace ? INHERITED_STAGES[ending.workspace] : undefined;
  if (inherited?.has(stage)) {
    return { state: 'skipped', word: 'carried over from the previous attempt' };
  }
  if (ending !== null && isAfter(stage, ending.stage)) {
    return { state: 'skipped', word: 'never ran' };
  }
  if (isLatest && ending === null) return { state: 'pending', word: 'not yet seen' };
  return { state: 'skipped', word: 'not recorded' };
}

/**
 * Whether one stage comes after another in the lifecycle.
 *
 * A stage name neither of them knows — a server-side addition — is never called "after"
 * anything, so it renders as `not recorded` rather than being greyed on a guess.
 */
function isAfter(stage: string, other: string): boolean {
  const at = STAGE_ORDER.indexOf(stage as (typeof STAGE_ORDER)[number]);
  const ended = STAGE_ORDER.indexOf(other as (typeof STAGE_ORDER)[number]);
  return at >= 0 && ended >= 0 && at > ended;
}

/** The endings the server served, keyed by the attempt each describes. */
export function endingsByAttempt(
  endings: readonly WorkstreamAttempt[],
): ReadonlyMap<number, WorkstreamAttempt> {
  return new Map(endings.map((item) => [item.attempt, item]));
}

/**
 * The window one row measures: what the journal timestamped, and nothing else.
 *
 * A row with no start time measures nothing. A row that has not completed measures only while
 * it is running -- a failed or deferred row with no completion is a row whose duration the
 * platform never recorded, and printing "now minus its start" would invent one.
 */
function operationWindow(
  operation: WorkstreamOperation,
  nowMs: number,
): { startMs: number; endMs: number } | null {
  const startMs = operation.started_at ? Date.parse(operation.started_at) : Number.NaN;
  if (!Number.isFinite(startMs)) return null;
  const completedMs = operation.completed_at ? Date.parse(operation.completed_at) : Number.NaN;
  if (Number.isFinite(completedMs)) {
    return { startMs, endMs: Math.max(startMs, completedMs) };
  }
  // Elapsed so far, against this browser's clock -- the same reading the heartbeat age uses,
  // clamped because the server's clock and this one are not the same clock.
  if (rowState(operation) === 'running') return { startMs, endMs: Math.max(startMs, nowMs) };
  return null;
}

/** How long one row took, or how long it has been running. Null when the journal cannot say. */
export function rowElapsed(operation: WorkstreamOperation, nowMs: number): string | null {
  const window = operationWindow(operation, nowMs);
  return window ? formatDuration(window.endMs - window.startMs) : null;
}

/** How long the derived phase has been going, read from the last completion the journal holds. */
export function phaseElapsed(phase: UnjournaledPhase, nowMs: number): string | null {
  const startMs = Date.parse(phase.since);
  if (!Number.isFinite(startMs)) return null;
  return formatDuration(Math.max(0, nowMs - startMs));
}

/**
 * One stage's subtotal: the time this stage's journaled operations were running.
 *
 * Overlapping windows count once rather than twice. The implementation write is journaled
 * inside the coding call, so summing the rows would report a stage that ran longer than the
 * clock allows; gaps between rows are excluded for the same reason -- the unjournaled phase
 * above has its own row and its own elapsed, and folding it in here would attribute it to
 * operations that were not running.
 */
export function stageElapsed(group: StageGroup, nowMs: number): string | null {
  const operations = group.rows.flatMap((row) => (row.kind === 'operation' ? [row.operation] : []));
  return mergedElapsed(operations, nowMs);
}

/**
 * The time a set of journaled operations was running, overlaps counted once.
 *
 * The one duration rule in this module, shared by the stage subtotals and the attempt markers.
 * Summing the rows would over-count: the implementation write is journaled inside the coding
 * call, so a stage that ran five minutes would report five minutes plus the nested second
 * twice over. Gaps between rows are excluded for the same reason -- nothing was running in
 * them, and the unjournaled phase that fills them has a row and an elapsed of its own.
 *
 * Null when no row in the set has a window the journal timestamped, which is different from
 * zero: "the platform never recorded this" is not "it took no time".
 */
function mergedElapsed(operations: WorkstreamOperation[], nowMs: number): string | null {
  const windows = operations
    .map((operation) => operationWindow(operation, nowMs))
    .filter((window): window is { startMs: number; endMs: number } => window !== null)
    .sort((left, right) => left.startMs - right.startMs);
  if (windows.length === 0) return null;
  let total = 0;
  let openStart: number | null = null;
  let openEnd = 0;
  for (const window of windows) {
    if (openStart === null) {
      openStart = window.startMs;
      openEnd = window.endMs;
    } else if (window.startMs <= openEnd) {
      openEnd = Math.max(openEnd, window.endMs);
    } else {
      total += openEnd - openStart;
      openStart = window.startMs;
      openEnd = window.endMs;
    }
  }
  if (openStart !== null) total += openEnd - openStart;
  return formatDuration(total);
}

/**
 * The workstream's own recorded state, for the one attempt it is the state of.
 *
 * Only the latest attempt: a workstream row holds one status and one failure class, and they
 * describe where it stands now. Reading them onto an earlier attempt would label attempt 1
 * with attempt 4's outcome, which is the class of mistake this whole item exists to remove.
 */
export interface WorkstreamOutcome {
  /** The workstream's status word, as the server spells it: running, completed, failed, … */
  status: string;
  /** The platform's own class for why it stopped, where it recorded one. */
  failureClassification: string | null;
}

/** One dropdown entry: what happened in that attempt, at a glance. */
export interface AttemptMarker {
  key: string;
  number: number | null;
  state: RowState;
  /** "Attempt 4", or "Earlier history" for the rows that predate the stamp. */
  label: string;
  /** The merged running time of the attempt's operations, or null where none was recorded. */
  elapsed: string | null;
  /** The word the glyph stands for. Always the record's own: never a verdict inferred here. */
  note: string;
  /** The whole entry as one line -- "✓ Attempt 4 · 12m 8s · approved". */
  text: string;
}

/**
 * What one attempt's marker says, read from the attempt's own rows and, for the latest attempt
 * only, from the workstream's recorded outcome.
 *
 * The split matters. A running row, or a workstream recorded as running, makes the marker live
 * -- and the second case is load-bearing, because the busiest window of an attempt is the
 * in-attempt checks, which are journaled by nothing (22). A settled latest attempt takes the
 * workstream's own status word and failure class, because that is the record of how the attempt
 * ended and the journal rows are not: an attempt can fail at review with every operation in it
 * succeeded.
 *
 * An earlier attempt has no such record of its own, so its marker is journal-only: the failed
 * row it stopped at, named by the class the platform recorded for it, or by the operation that
 * failed where it recorded no class. Where an earlier attempt's rows all succeeded the marker
 * says only that a later attempt followed -- a tick there would read as "this attempt
 * delivered", which is the one thing the journal has just disproved.
 */
function attemptMarkerState(
  view: AttemptView,
  { latest, outcome, childRunning, ending }: {
    latest: boolean;
    outcome?: WorkstreamOutcome;
    childRunning: boolean;
    ending: WorkstreamAttempt | null;
  },
): { state: RowState; note: string } {
  if (view.operations.some((operation) => RUNNING_STATUSES.has(operation.status))) {
    return { state: 'running', note: 'running' };
  }
  if (latest && childRunning) return { state: 'running', note: 'running' };
  // One authority per attempt. Where the server served an ending for this attempt, it
  // supersedes `outcome` — which is the latest attempt's state and no other's — for both the
  // marker and the stages. `outcome` remains the fallback for an attempt with no served
  // ending, which is every attempt of a run that predates the block. Two sources for one
  // fact is the shape this ordering exists to prevent.
  if (ending !== null) {
    return {
      state: endingIsFailure(ending)
        ? 'failed'
        : DELIVERED_ENDINGS.has(ending.ended_by)
          ? 'succeeded'
          : 'other',
      // The record's own clause where it has one, else the word the glyph stands for. The
      // glyph never travels alone: a native option carries text only.
      note: ending.detail ?? endingWord(ending),
    };
  }
  if (latest && outcome) {
    if (FAILED_WORKSTREAM_STATUSES.has(outcome.status)) {
      // The class the platform recorded, else the operation that failed, else the status word
      // itself -- and never "failed: failed", which says nothing twice.
      const named =
        outcome.failureClassification ??
        failedRowName(view) ??
        (outcome.status === 'failed' ? null : humanStatus(outcome.status));
      return { state: 'failed', note: named ? `failed: ${named}` : 'failed' };
    }
    return {
      state: SUCCEEDED_WORKSTREAM_STATUSES.has(outcome.status) ? 'succeeded' : 'other',
      note: humanStatus(outcome.status),
    };
  }
  const failedName = failedRowName(view);
  if (failedName) return { state: 'failed', note: `failed: ${failedName}` };
  const last = view.operations.at(-1);
  if (last && rowState(last) === 'succeeded') {
    return latest
      ? { state: 'succeeded', note: 'every operation succeeded' }
      : { state: 'other', note: 'a later attempt followed' };
  }
  return { state: 'other', note: last ? humanStatus(last.status) : 'nothing recorded' };
}

/** The workstream statuses that mean this attempt delivered, and the ones that mean it did not. */
const SUCCEEDED_WORKSTREAM_STATUSES = new Set(['approved', 'completed']);
const FAILED_WORKSTREAM_STATUSES = new Set(['failed', 'review_rejected']);

/** What the attempt's last failed row was, by the class the platform gave it or by its type. */
function failedRowName(view: AttemptView): string | null {
  const failed = view.operations.filter((operation) => rowState(operation) === 'failed').at(-1);
  return failed ? (failed.error_code ?? failed.operation_type) : null;
}

/** A server status word, spaced for a sentence. Never reworded -- the vocabulary is the server's. */
function humanStatus(status: string): string {
  return status.replace(/_/g, ' ');
}

/**
 * The dropdown, as text: one marker per attempt, the run history at a glance.
 *
 * The glyph never travels alone. Native options carry text only, so the word the glyph stands
 * for is in the line itself -- "✓ Attempt 4 · 12m 8s · approved" reads the same to a screen
 * reader as it does to an eye, which "✓" on its own does not.
 */
export function attemptMarkers(
  views: AttemptView[],
  {
    childRunning,
    outcome,
    nowMs,
    endings,
  }: {
    childRunning: boolean;
    outcome?: WorkstreamOutcome;
    nowMs: number;
    /**
     * Where each finished attempt ended, as served. Keyed by attempt number, so the
     * unstamped merged history — whose number is null — gets none of them: five confident
     * stage states over a pre-stamp history would be the lie this whole item removes.
     */
    endings?: ReadonlyMap<number, WorkstreamAttempt>;
  },
): AttemptMarker[] {
  return views.map((view, index) => {
    const latest = index === views.length - 1;
    const ending = (view.number !== null ? endings?.get(view.number) : null) ?? null;
    const { state, note } = attemptMarkerState(view, { latest, outcome, childRunning, ending });
    const label = view.number === null ? 'Earlier history' : `Attempt ${view.number}`;
    const elapsed = mergedElapsed(view.operations, nowMs);
    const parts =
      state === 'running'
        ? [label, 'running', elapsed ? `${elapsed} so far` : null]
        : [label, elapsed, note];
    return {
      key: view.key,
      number: view.number,
      state,
      label,
      elapsed,
      note,
      text: `${ROW_GLYPHS[state]} ${parts.filter(Boolean).join(' · ')}`,
    };
  });
}

/* -------------------------------------------------- the graph's miniature of the lifecycle */

/**
 * One cell of the graph's sub-stage strip: the drawer's lifecycle, miniaturised.
 *
 * The graph draws Implementation and Validation as two nodes and has nowhere to say that one
 * agent owns both, nor where inside that pair the work currently is. These are that, and the
 * two things the full nodes cannot express at all: whether the implementation self-review
 * passed, and how many in-attempt repair passes ran.
 */
export interface SubStageCell {
  key: string;
  /** What the cell is, in the stage vocabulary where it has one. */
  label: string;
  state: StageState;
  /** The word the glyph stands for. Never one without the other. */
  word: string;
}

/**
 * The self-review's own vocabulary, mapped one for one.
 *
 * Six values the code enumerates -- `unavailable`, `clean`, `substantive_problem`,
 * `corrections_failed`, `correction_rejected`, `corrected` -- and a seventh the records show
 * and the code does not: absent. 201's backend attempt 4 carries no `self_review` key at all,
 * and a cell that renders nothing there reads as clean.
 *
 * Mapping every one is the point. A cell that rendered only `corrected` and blanked the rest
 * is how `corrections_failed` -- the gate that actually stopped 201's attempt 0 -- becomes
 * invisible, which is the defect Part A exists to fix, one surface over.
 */
const SELF_REVIEW_WORDS: Record<string, { state: StageState; word: string }> = {
  clean: { state: 'succeeded', word: 'clean' },
  corrected: { state: 'succeeded', word: 'corrected' },
  unavailable: { state: 'other', word: 'composed but could not run' },
  substantive_problem: { state: 'failed', word: 'substantive problem' },
  corrections_failed: { state: 'failed', word: 'corrections failed' },
  correction_rejected: { state: 'failed', word: 'correction rejected' },
};

/** Where the strip's self-review cell sits: after coding, before validation. */
const SELF_REVIEW_KEY = 'self-review';

/**
 * The newest attempt's sub-stage progression, from the same polled response the drawer reads.
 *
 * One data source, two surfaces, never a second derivation. The strip shows the newest
 * attempt only -- history lives in the drawer -- so an in-flight attempt is the ordinary case
 * here, and it has no served ending: its self-review cell says "not yet seen" rather than
 * claiming a pass nobody recorded.
 */
export function subStageStrip(
  operations: WorkstreamOperation[],
  endings: readonly WorkstreamAttempt[],
): SubStageCell[] {
  const views = attemptViews(operations);
  const latest = views.at(-1);
  if (!latest) return [];
  const served = endingsByAttempt(endings);
  const ending = (latest.number !== null ? served.get(latest.number) : null) ?? null;
  const groups = groupByStage(latest.operations, { includePending: true });
  const byStage = new Map(groups.map((group) => [group.stage, group]));
  const cells: SubStageCell[] = [];
  for (const stage of ['setup', 'coding', 'validation'] as const) {
    const group = byStage.get(stage) ?? { stage, agent: AGENT_FOR_STAGE[stage] ?? null, rows: [] };
    const reading = stageReading(group, { ending, isLatest: true });
    cells.push({ key: stage, label: stage, state: reading.state, word: reading.word });
    // Between coding and validation, which is where it runs: the gate composes over the
    // change the coding call wrote, and its findings are re-validated after a correction.
    if (stage === 'coding') cells.push(selfReviewCell(ending));
  }
  // The in-attempt repair passes, on the cell they repaired. Absent renders nothing and `0`
  // renders nothing either -- across 201's nine backend completions the field is present
  // twice, both times zero, so a "0 repairs" badge would be noise on every healthy attempt.
  const repairs = ending?.source_repair_passes ?? 0;
  if (repairs > 0) {
    cells.push({
      key: 'repairs',
      label: repairs === 1 ? 'repair pass' : 'repair passes',
      state: 'other',
      word: `${repairs} in-attempt repair ${repairs === 1 ? 'pass' : 'passes'}`,
    });
  }
  // Beside them, because they happened inside them: the in-attempt passes are unjournaled,
  // so a stream of theirs that stalled and then answered leaves no row anywhere. Positive
  // counts only, on the same reasoning as the repairs above.
  const reissues = ending?.stream_reissues ?? 0;
  if (reissues > 0) {
    cells.push({
      key: 'reissues',
      label: 'provider re-issues',
      state: 'other',
      word: `${reissues} provider re-${reissues === 1 ? 'issue' : 'issues'} in-attempt`,
    });
  }
  return cells;
}

/** One line inside a lane node: an operation the agent ran, or the word for a stage with none. */
export interface NodeOpLine {
  key: string;
  state: StageState;
  /** The name the line stands for: an operation type, or a stage's own word. */
  name: string;
  /** The word the glyph stands for, for the line's accessible text. Never one without the other. */
  word: string;
  /** What qualifies it -- `call 1`, an elapsed time, a write's role. May be absent. */
  meta: string | null;
}

/**
 * How many lines a lane node shows before it starts counting the rest.
 *
 * Seven, because that is what 201's Implementation node actually holds: clone, create_branch,
 * the pre-coding write, install, the coding call, the implementation write, and the
 * self-review. A node sized for the common case and honest about the remainder beats one that
 * grows without bound and pushes the next lane off the canvas.
 */
export const MAX_NODE_OPS = 7;

/**
 * The operations to render inside one lane node, from the same polled response the drawer reads.
 *
 * The graph used to show only which stage the work was in; the operations lived one drawer
 * away, so "the Engineer does a lot of this" was invisible on the surface a person watches.
 * Each lane node now carries the operations of the stages it owns -- setup and coding under
 * Implementation, the commands under Validation, the reviewer call under Review -- because an
 * operation belongs to the stage that ran it and needs no legend to say so.
 *
 * Publication rows (`create_commit`, `push_branch`) are deliberately absent: the lane has no
 * publication node to hang them on, and inventing one would claim a stage the graph does not
 * model. They stay in the drawer, which is where the stage stepper already names them.
 *
 * A stage with no rows renders its own word rather than nothing -- `never ran` after an
 * ending, `not yet seen` mid-run, `carried over` where the workspace was inherited -- so the
 * A.1 vocabulary reaches the graph instead of stopping at the drawer.
 */
export function nodeOperations(
  operations: WorkstreamOperation[],
  endings: readonly WorkstreamAttempt[],
  stages: readonly string[],
  { nowMs }: { nowMs: number },
): NodeOpLine[] {
  const views = attemptViews(operations);
  const latest = views.at(-1);
  if (!latest) return [];
  const served = endingsByAttempt(endings);
  const ending = (latest.number !== null ? served.get(latest.number) : null) ?? null;
  const roles = writeRoles(latest.operations);
  const groups = groupByStage(latest.operations, { includePending: true });
  const byStage = new Map(groups.map((group) => [group.stage, group]));
  const lines: NodeOpLine[] = [];
  for (const stage of STAGE_ORDER) {
    if (!stages.includes(stage)) continue;
    const group = byStage.get(stage) ?? { stage, agent: AGENT_FOR_STAGE[stage] ?? null, rows: [] };
    if (group.rows.length === 0) {
      const reading = stageReading(group, { ending, isLatest: true });
      lines.push({
        key: `stage:${stage}`,
        state: reading.state,
        name: stage,
        word: reading.word,
        meta: reading.word,
      });
    }
    for (const row of group.rows) {
      if (row.kind === 'phase') {
        lines.push({
          key: `phase:${row.phase.stage}`,
          state: 'running',
          name: row.phase.stage,
          word: 'running',
          meta: phaseElapsed(row.phase, nowMs),
        });
        continue;
      }
      const state = rowState(row.operation);
      const write = roles.get(row.operation.operation_id) ?? null;
      lines.push({
        key: row.operation.operation_id,
        state,
        name: row.operation.operation_type,
        word: ROW_WORDS[state],
        meta: nodeOpMeta(row.operation, write, nowMs),
      });
    }
    // The gate that stopped 201's attempt 0, on the stage it composes over. Without a line of
    // its own a failed self-review is a stage of ticks and an attempt that ended anyway.
    if (stage === 'coding') {
      const cell = selfReviewCell(ending, { movedOn: movedPastCoding(latest.operations) });
      lines.push({
        key: cell.key,
        state: cell.state,
        name: cell.label,
        word: cell.word,
        meta: cell.word,
      });
    }
  }
  const repairs = ending?.source_repair_passes ?? 0;
  if (repairs > 0 && stages.includes('coding')) {
    lines.push({
      key: 'repairs',
      state: 'other',
      name: repairs === 1 ? 'repair pass' : 'repair passes',
      word: `${repairs} in-attempt repair ${repairs === 1 ? 'pass' : 'passes'}`,
      meta: `${repairs}`,
    });
  }
  const reissues = ending?.stream_reissues ?? 0;
  if (reissues > 0 && stages.includes('coding')) {
    lines.push({
      key: 'reissues',
      state: 'other',
      name: 'provider re-issues',
      word: `${reissues} provider re-${reissues === 1 ? 'issue' : 'issues'} in-attempt`,
      meta: `${reissues}`,
    });
  }
  return lines;
}

/** What qualifies one operation line: which call it is, and how long it has been running. */
function nodeOpMeta(
  operation: WorkstreamOperation,
  write: WriteRole | null,
  nowMs: number,
): string | null {
  const parts = [
    operation.operation_type === 'run_coding_executor' ? `call ${operation.attempt}` : null,
    write === 'pre_coding' ? 'workspace' : null,
    rowElapsed(operation, nowMs),
  ].filter((part): part is string => part !== null);
  return parts.length > 0 ? parts.join(' · ') : null;
}

/**
 * Whether this attempt has run anything the self-review precedes.
 *
 * The gate composes over what the coding call wrote and its findings are re-validated after a
 * correction, so validation, review and publication all come after it. A row in any of them
 * is proof the attempt is past the point where the self-review would have run -- whatever the
 * completion does or does not say about it.
 */
function movedPastCoding(operations: WorkstreamOperation[]): boolean {
  const after = new Set(['validation', 'review', 'publication']);
  const roles = writeRoles(operations);
  return operations.some((operation) => after.has(stageOf(operation, roles.get(operation.operation_id) ?? null)));
}

/* ---------------------------------------------------- the planning nodes' journaled calls */

/**
 * How the server marks an execution derived from a journaled pre-coding call: the
 * `planning_call:` prefix `execution_records.py` stamps on `execution_id`. Every other
 * execution is derived from artifacts and has no start or heartbeat to show.
 */
const PLANNING_CALL_PREFIX = 'planning_call:';

/**
 * The graph nodes a planning call may render on, by the server's own `to_stage`.
 *
 * A positive list, matched against node ids `graph.ts` builds: the product-manager calls
 * land on `technical_prd` and reconnaissance, grounding and the planner on
 * `integration_contract`, exactly as `_PLANNING_CALL_STAGES` assigns them server-side. A
 * stage this client has never heard of renders nowhere rather than being guessed onto a
 * neighbouring node -- the same rule the unknown-`ended_by` handling follows.
 *
 * `execution_plan` is deliberately absent: the planner writes the contract and the plan in
 * one call, and that call's row sits on the contract node the server assigns it to. Two
 * renderings of one row would claim two calls ran.
 */
const PLANNING_NODES = new Set(['technical_prd', 'integration_contract']);

/**
 * The execution-status vocabulary, mapped positively onto the stage glyphs.
 *
 * An unknown status -- a value a later server added -- takes `other` and its own name
 * humanised, never a tick or a cross this client cannot justify.
 */
const PLANNING_CALL_STATES: Record<string, { state: StageState; word: string }> = {
  completed: { state: 'succeeded', word: 'succeeded' },
  running: { state: 'running', word: 'running' },
  failed: { state: 'failed', word: 'failed' },
  pending: { state: 'pending', word: 'queued' },
  queued: { state: 'pending', word: 'queued' },
  cancelled: { state: 'other', word: 'cancelled' },
  blocked: { state: 'other', word: 'blocked' },
  needs_human: { state: 'other', word: 'needs a person' },
};

/** How long one planning call took, or has been running. Null when the record cannot say. */
function planningCallElapsed(call: ExecutionRecord, nowMs: number): string | null {
  if (typeof call.duration_seconds === 'number') {
    return formatDuration(call.duration_seconds * 1000);
  }
  // Elapsed so far, against this browser's clock, and only while the record says the call is
  // still going: printing "now minus start" for a failed row would invent a duration the
  // platform never recorded -- the same rule `operationWindow` applies to journal rows.
  if (call.status !== 'running' || !call.started_at) return null;
  const startMs = Date.parse(call.started_at);
  if (!Number.isFinite(startMs)) return null;
  return formatDuration(Math.max(0, nowMs - startMs));
}

/**
 * The planning nodes' operation lines, from the executions read the edge chips already show.
 *
 * The Engineer's lane nodes list their journaled operations; the planning nodes said only
 * whether an artifact existed, so the calls that produce those artifacts -- the
 * product-manager draft, the requirement reconciliation, per-repository reconnaissance,
 * clarification grounding, the planner -- were visible on the graph only as edge chips.
 * These are the same `planning_call:` records, rendered as lines on the node whose stage the
 * server assigns each call to. One data source, two surfaces: a node and the arrow beside it
 * can never disagree about what ran.
 *
 * The line's name is the journal's own `logical_step` (served as `agent_type`), so the node
 * speaks journal vocabulary the way lane nodes speak `operation_type`; the handler's human
 * name is the fallback for a record that carries none. Reconnaissance runs once per
 * repository, so its lines carry the repository in their meta -- without it, two ticked
 * `repository_reconnaissance` lines read as one call that ran twice.
 */
export function planningNodeOperations(
  executions: readonly ExecutionRecord[],
  { nowMs }: { nowMs: number },
): ReadonlyMap<string, NodeOpLine[]> {
  const startMs = (item: ExecutionRecord): number => {
    const value = item.started_at ? Date.parse(item.started_at) : Number.NaN;
    // A call with no start yet sorts last, in served order: it has not happened.
    return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
  };
  const chronological = executions
    .filter((item) => item.execution_id.startsWith(PLANNING_CALL_PREFIX))
    .sort((left, right) => startMs(left) - startMs(right));
  const byNode = new Map<string, NodeOpLine[]>();
  for (const call of chronological) {
    if (!PLANNING_NODES.has(call.to_stage)) continue;
    const reading = PLANNING_CALL_STATES[call.status] ?? {
      state: 'other' as StageState,
      word: humanise(call.status),
    };
    // The model between the repository and the duration: what ran, where, for how long. Only
    // the model the record itself carries -- a row that recorded none shows none, exactly as
    // the arrow beside the node does, rather than this line inventing a plausible name.
    const meta = [
      call.repository_id ?? null,
      modelDisplayName(call.model),
      planningCallElapsed(call, nowMs),
    ]
      .filter((part): part is string => part !== null)
      .join(' · ');
    const line: NodeOpLine = {
      key: call.execution_id,
      state: reading.state,
      name: call.agent_type ?? call.handler,
      word: reading.word,
      meta: meta.length > 0 ? meta : null,
    };
    const existing = byNode.get(call.to_stage);
    if (existing) existing.push(line);
    else byNode.set(call.to_stage, [line]);
  }
  return byNode;
}

/** The self-review cell, over all seven states its record can be in. */
function selfReviewCell(
  ending: WorkstreamAttempt | null,
  { movedOn }: { movedOn: boolean } = { movedOn: false },
): SubStageCell {
  const base = { key: SELF_REVIEW_KEY, label: SELF_REVIEW_KEY };
  if (ending === null) {
    // No ending yet, so there is no record of a self-review either way -- but "not yet seen"
    // is a claim about the future, and it is false the moment a later stage has rows. 202's
    // frontend showed exactly that: lint, tests and build all succeeded beside a self-review
    // reported as still to come, on a cancelled run whose attempt never recorded one. Where
    // the attempt has moved past coding the honest word is that nothing recorded it.
    return movedOn
      ? { ...base, state: 'skipped', word: 'no record' }
      : { ...base, state: 'pending', word: 'not yet seen' };
  }
  const outcome = ending.self_review_outcome;
  if (outcome === null) {
    // The seventh state: an attempt that recorded no self-review at all. Said as such,
    // because an empty cell here reads as a clean pass.
    return { ...base, state: 'skipped', word: 'no self-review recorded' };
  }
  const known = SELF_REVIEW_WORDS[outcome];
  if (!known) return { ...base, state: 'other', word: humanise(outcome) };
  // "corrected · 2 files", never "corrected ×2": `corrections_applied` is a list of paths, so
  // the count it yields is files corrected, and "×2" reads as two correction rounds.
  const files = ending.self_review_corrected_files;
  const word =
    outcome === 'corrected' && files !== null && files > 0
      ? `${known.word} · ${files} ${files === 1 ? 'file' : 'files'}`
      : known.word;
  return { ...base, state: known.state, word };
}
