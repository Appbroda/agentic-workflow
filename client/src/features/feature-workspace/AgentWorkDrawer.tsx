import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Drawer } from '@/components/ui/Drawer';
import { RepositoryBadge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/common/States';
import { humanise } from '@/utils/text';
import { eventTime } from '@/utils/time';
import type { WorkstreamAttempt, WorkstreamOperation } from '@/schemas/feature';
import {
  attemptMarkers,
  attemptViews,
  endingsByAttempt,
  groupByStage,
  partitionAttempts,
  phaseElapsed,
  reissueNote,
  repeatNote,
  ROW_GLYPHS,
  rowElapsed,
  rowState,
  STAGE_GLYPHS,
  stageElapsed,
  stageReading,
  type AttemptView,
  type StageGroup,
  type StageReading,
  type UnjournaledPhase,
  type WorkstreamOutcome,
  type WriteRole,
  unjournaledPhase,
} from './agent-work';
import { heartbeatReading } from './operations';
import type { GraphNode } from './graph';

/**
 * One repository lane, opened: the agent's sub-operations for one attempt, in order, with live
 * status -- `install_dependencies ✓ → run_linter ✓ → run_coding_executor (call 3, heartbeat 2s
 * ago)`. The view a person needed four times during the 185-194 cycle and got only by querying
 * Postgres.
 *
 * Which attempt is chosen at the top, defaulting to the latest, and the dropdown is itself the
 * run history: "✓ Attempt 4 · 12m 8s · approved", "✗ Attempt 2 · 9m 2s · failed: own tests",
 * "● Attempt 5 · running · 7m 0s so far". One attempt per view, never dividers in a merged scroll
 * -- run 197 BE showed attempts 0, 1 and 2 as one CURRENT ATTEMPT, because a retry that
 * preserves its workspace never re-clones and a fresh clone was the only divider there was.
 *
 * Everything shown is a journal row the 55- endpoint served, with two derivations named as
 * such: a write that no coding call preceded renders as the workspace preparation it is (21),
 * and an attempt whose child is running with nothing journaled renders one row naming that
 * phase (22) instead of a column of ticks that reads as silence. Every duration is computed
 * from the endpoint's own timestamps against this browser's clock. Operation names are the raw
 * types on purpose: a type this client has never heard of renders as itself under the stage
 * the server assigned, never dropped and never guessed into a friendlier label. The drawer
 * reads the rows the graph already polls -- opening it costs no request of its own.
 *
 * Every attempt shows its whole lifecycle, not only the parts of it that ran. A finished
 * attempt renders ✓ on the stages it completed, ✗ on the stage its ending record names -- with
 * that record's own sentence beside it -- and a greyed word on the stages after the stop that
 * says which kind of absence each one is: never ran, carried over from the previous attempt,
 * or not recorded. Run 201's backend attempt 0 is why: 57 minutes, every journaled row ticked,
 * stopped by the self-review gate before the reviewer was called, and this drawer drew it as an
 * unbroken column of green. The ending is always a served fact; nothing here infers one.
 */

export function AgentWorkDrawer({
  featureId,
  repositoryId,
  node,
  operations,
  endings,
  childRunning,
  outcome,
  nowMs,
  onDismiss,
}: {
  featureId: string;
  /** The repository whose journal is being read. */
  repositoryId: string;
  /** The lane node that was opened, for the onward link to what it stands for. */
  node: GraphNode;
  /** The repository's journal rows exactly as served: newest first, bounded. */
  operations: WorkstreamOperation[];
  /**
   * Where each finished attempt ended, from the same polled response. One per attempt the
   * child has moved past; the in-flight attempt has none, because it has not ended.
   */
  endings?: readonly WorkstreamAttempt[];
  /** Whether the workstream's own state says this child is executing right now. */
  childRunning: boolean;
  /** The workstream's own recorded outcome, which is the latest attempt's and no other's. */
  outcome?: WorkstreamOutcome;
  nowMs: number;
  onDismiss: () => void;
}) {
  const views = useMemo(() => attemptViews(operations), [operations]);
  const byAttempt = useMemo(() => endingsByAttempt(endings ?? []), [endings]);
  const markers = attemptMarkers(views, { childRunning, outcome, nowMs, endings: byAttempt });
  // Latest by default, and it keeps following the latest until somebody chooses otherwise: a
  // person watching a run wants the attempt that is happening, and a person who has picked an
  // earlier one must not be dragged off it when the next poll adds a row. A chosen key the
  // journal no longer serves -- the row budget can drop an attempt off the end -- falls back
  // to the latest rather than rendering nothing.
  const [chosenKey, setChosenKey] = useState<string | null>(null);
  const selected =
    views.find((view) => view.key === chosenKey) ?? views.at(-1) ?? { key: '', number: null, operations: [] };
  const isLatest = selected.key === views.at(-1)?.key;
  // Only the latest attempt can be mid-phase. The stages it has not reached still render for
  // every stamped attempt now -- what changes is the word, not whether they appear: "not yet
  // seen" on a finished attempt read as work still to come, and hiding the stage instead read
  // as an attempt that had no review at all.
  const phase = isLatest ? unjournaledPhase(selected.operations, { childRunning }) : null;
  // Endings key on stamped attempt numbers only. The unstamped merged history spans whatever
  // happened before the platform stamped attempts, and it gets none of the new state words.
  const ending = (selected.number !== null ? byAttempt.get(selected.number) : null) ?? null;
  const repositoryPath = `/features/${encodeURIComponent(featureId)}/repositories/${encodeURIComponent(repositoryId)}`;

  return (
    <Drawer
      title="Agent work"
      subtitle={
        <span className="row" style={{ gap: 'var(--space-2)', flexWrap: 'wrap' }}>
          <RepositoryBadge repositoryId={repositoryId} />
          <span>what each agent is doing, from the operation journal</span>
        </span>
      }
      onDismiss={onDismiss}
    >
      <div className="stack">
        {operations.length === 0 ? (
          <EmptyState
            title="No operations recorded yet"
            detail="This repository's journal is empty. Rows appear as soon as the first operation begins."
          />
        ) : (
          <>
            {/* One attempt per view, chosen here -- never dividers in a merged scroll. The
                dropdown itself is the run history: each entry carries its number, its glyph
                with the word the glyph stands for, how long it took and how it ended. With a
                single entry there is nothing to choose between, so the control does not
                appear and the drawer reads exactly as it did before attempts were stamped. */}
            {markers.length > 1 ? (
              <div className="field agent-work__attempt-picker">
                <label className="field__label" htmlFor="agent-work-attempt">
                  Attempt
                </label>
                <select
                  id="agent-work-attempt"
                  value={selected.key}
                  onChange={(event) => setChosenKey(event.target.value)}
                >
                  {/* Newest first in the list, because the newest is the one being watched,
                      while the views themselves stay in the order the work happened. */}
                  {[...markers].reverse().map((marker) => (
                    <option key={marker.key} value={marker.key}>
                      {marker.text}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}
            <AttemptRows
              view={selected}
              isLatest={isLatest}
              ending={ending}
              phase={phase}
              nowMs={nowMs}
            />
          </>
        )}
        <div className="row" style={{ gap: 'var(--space-2)', flexWrap: 'wrap' }}>
          {node.kind !== 'repository' && node.href ? (
            <Link className="button button--small" to={node.href}>
              Open {node.label.toLowerCase()}
            </Link>
          ) : null}
          <Link className="button button--small" to={repositoryPath}>
            Open repository
          </Link>
        </div>
      </div>
    </Drawer>
  );
}

/**
 * One chosen attempt's rows, grouped by stage.
 *
 * A stamped attempt renders as one sequence: the server said which attempt every row belongs
 * to, so there is nothing left to divide.
 *
 * The unstamped view is the exception, and it is where the clone heuristic keeps its old job
 * and no more. Those rows span whatever happened before the platform stamped attempts, so the
 * fresh `clone_repository` marker still divides them into the attempts it can prove, behind
 * the same collapsed dividers -- a wholly pre-stamp journal therefore reads exactly as it did.
 * The heuristic never competes with a stamp, so it can no longer merge a retry into its
 * predecessor: run 197 BE's rows would now all be stamped and never reach this path.
 */
function AttemptRows({
  view,
  isLatest,
  ending,
  phase,
  nowMs,
}: {
  view: AttemptView;
  isLatest: boolean;
  /** This attempt's served ending, or null for the unstamped history and the in-flight attempt. */
  ending: WorkstreamAttempt | null;
  phase: UnjournaledPhase | null;
  nowMs: number;
}) {
  const slices = view.number === null ? partitionAttempts([...view.operations].reverse()).attempts : [view.operations];
  const current = slices.at(-1) ?? [];
  const earlier = slices.slice(0, -1);
  // The scaffold applies to every STAMPED attempt, not only the latest -- that is the whole
  // of A.1's change, and the gate is deliberately not `true`. The unstamped merged history is
  // the latest view when a journal wholly predates the stamp, so forcing the scaffold off
  // would change today's rendering as surely as forcing it on: `isLatest` keeps that case
  // exactly as it reads now, and `view.number !== null` adds it to every stamped attempt.
  const scaffold = isLatest || view.number !== null;
  const groups = groupByStage(current, { includePending: scaffold, phase });
  const readings = new Map(
    groups.map((group) => [group.stage, stageReading(group, { ending, isLatest })]),
  );
  return (
    <>
      {earlier.map((slice, index) => (
        <details key={`slice-${index}`} className="agent-work__attempt">
          <summary className="agent-work__divider">
            Earlier attempt · {slice.length} {slice.length === 1 ? 'operation' : 'operations'}
          </summary>
          <div className="stack stack--tight">
            {/* The pre-stamp slices keep exactly the rendering they have: no scaffold, no
                endings, no new state words. Nothing here can say which attempt these rows
                belong to, so nothing here says where one ended. */}
            {groupByStage(slice, { includePending: false }).map((group) => (
              <Stage
                key={group.stage}
                group={group}
                reading={stageReading(group, { ending: null, isLatest: false })}
                nowMs={nowMs}
              />
            ))}
          </div>
        </details>
      ))}
      <div className="stack stack--tight">
        {/* Named only where there is something above it to distinguish it from. "Current" is
            claimed only by the latest view; inside an older slice of history the honest word
            is that this is the most recent of these, not that it is what is happening. */}
        {earlier.length > 0 ? (
          <h4 className="details__label">
            {isLatest ? 'Current attempt' : 'Most recent in this history'}
          </h4>
        ) : null}
        {/* The lifecycle at a glance, above the rows it summarises. Only where the scaffold
            renders: over a merged pre-stamp history a stepper would claim a single lifecycle
            for several attempts at once. */}
        {scaffold ? <StageStepper groups={groups} readings={readings} /> : null}
        {groups.map((group) => (
          <Stage
            key={group.stage}
            group={group}
            reading={readings.get(group.stage)!}
            nowMs={nowMs}
          />
        ))}
      </div>
    </>
  );
}

/**
 * The lifecycle as one compact row: a glyph per stage, in execution order.
 *
 * A list rather than a decoration, and every glyph travels with the word it stands for, so
 * the stepper reads the same to a screen reader as it does to an eye. It says nothing the
 * stage sections below do not; it says it in one line, which is what a person scanning for
 * "where did this stop" needs.
 */
function StageStepper({
  groups,
  readings,
}: {
  groups: StageGroup[];
  readings: ReadonlyMap<string, StageReading>;
}) {
  return (
    <ol className="agent-work__stepper" aria-label="stage lifecycle">
      {groups.map((group) => {
        const reading = readings.get(group.stage)!;
        return (
          <li
            key={group.stage}
            className={`agent-work__step agent-work__step--${reading.state}`}
            title={`${group.stage}: ${reading.word}`}
          >
            <span aria-hidden="true">{STAGE_GLYPHS[reading.state]}</span>
            <span className="agent-work__step-name">{group.stage}</span>
            <span className="sr-only">{reading.word}</span>
          </li>
        );
      })}
    </ol>
  );
}

/**
 * One stage's operations under the agent that owns it -- "Engineer · setup" -- both
 * vocabularies on one line, agent first because "what is this agent doing" is the question
 * that motivated the box. A stage without an owning agent keeps its stage word as the label
 * rather than having an agent invented for it.
 */
function Stage({
  group,
  reading,
  nowMs,
}: {
  group: StageGroup;
  reading: StageReading;
  nowMs: number;
}) {
  const elapsed = stageElapsed(group, nowMs);
  return (
    <section
      className={`agent-work__stage agent-work__stage--${reading.state}`}
      aria-label={groupName(group)}
    >
      <h5 className="agent-work__stage-name">
        {/* The state of the stage, before the name of it: "where did this stop" is answered by
            scanning down this column. The glyph never travels without its word -- spoken here
            where the stage has rows, and read from the visible line below where it has none,
            so the word is announced exactly once either way. */}
        <span aria-hidden="true">{STAGE_GLYPHS[reading.state]}</span>
        {group.rows.length > 0 ? <span className="sr-only">{reading.word}, </span> : null}
        {group.agent ? (
          <>
            {group.agent} <span className="agent-work__stage-word">· {group.stage}</span>
          </>
        ) : (
          humanise(group.stage)
        )}
        {elapsed ? (
          // The subtotal, and what it is a subtotal of: the time this stage's operations were
          // running. Not the wall clock across the stage, which would silently include the
          // phase between its rows -- that phase has a row of its own.
          <span
            className="agent-work__stage-elapsed"
            title="time this stage's journaled operations were running"
          >
            {elapsed}
          </span>
        ) : null}
      </h5>
      {/* What the state means, in the words the record licenses. `not yet seen` is today's
          wording for the latest attempt mid-run; the rest are the words a finished attempt
          needs and never had -- `never ran`, `carried over from the previous attempt`,
          `not recorded` -- plus the ending's own sentence where this is the stage it named. */}
      {reading.notes.length > 0 ? (
        <p className="agent-work__stage-note">{reading.notes.join(' · ')}</p>
      ) : null}
      {group.rows.length === 0 ? (
        <p className={`agent-work__row agent-work__row--${reading.state}`}>
          <span aria-hidden="true">{STAGE_GLYPHS[reading.state]}</span> {reading.word}
        </p>
      ) : (
        <ul className="agent-work__rows">
          {group.rows.map((row) =>
            row.kind === 'phase' ? (
              <PhaseRow key={`phase-${row.phase.stage}`} phase={row.phase} nowMs={nowMs} />
            ) : (
              <Row
                key={row.operation.operation_id}
                operation={row.operation}
                write={row.write}
                nowMs={nowMs}
              />
            ),
          )}
        </ul>
      )}
    </section>
  );
}

function groupName(group: StageGroup): string {
  return group.agent ? `${group.agent}, ${group.stage}` : group.stage;
}

/**
 * The one row nothing journaled: the phase the attempt is in between its records.
 *
 * Deliberately not a liveness row -- there is no heartbeat behind it, so it carries neither
 * the pulse nor the stale treatment, and it never claims an operation is running. It says what
 * the phase is, that its steps are not journaled individually, and since when. Run 197 FE is
 * the case it exists for: seven minutes of in-attempt checks that the drawer drew as silence.
 */
function PhaseRow({ phase, nowMs }: { phase: UnjournaledPhase; nowMs: number }) {
  const elapsed = phaseElapsed(phase, nowMs);
  return (
    <li className="agent-work__row agent-work__row--phase">
      <span aria-hidden="true">⋯</span>
      <span className="sr-only">no journaled operation</span>
      {/* Not `agent-work__type`: that class is the monospace an operation type reads in, and
          this label is derived. A derived phase dressed as an identifier reads as a row the
          journal holds, which is the misreading this row exists to end. */}
      <span className="agent-work__phase">{phase.label}</span>
      <span className="agent-work__note">
        {phase.detail} · since {eventTime(phase.since, new Date(nowMs))}
        {elapsed ? ` · ${elapsed}` : ''}
      </span>
    </li>
  );
}

/**
 * One journal row. The glyph always travels with its word, so the status is readable without
 * the symbol: "✓" alone is not an answer a screen reader can give.
 */
function Row({
  operation,
  write,
  nowMs,
}: {
  operation: WorkstreamOperation;
  write: WriteRole | null;
  nowMs: number;
}) {
  const state = rowState(operation);
  const heartbeat = state === 'running' ? heartbeatReading(operation, nowMs) : null;
  // The coding row carries its call count -- this is where 186's truncation→fault→third-call
  // story becomes visible as it happens.
  const call = operation.operation_type === 'run_coding_executor' ? `call ${operation.attempt}` : null;
  const elapsed = rowElapsed(operation, nowMs);
  const parts = [
    call,
    // Why this row is one of several of its type in the attempt. Two linter rows in one
    // attempt read as a bug; on 201's backend attempt 3 both were real, and the journal
    // already held the reason in two columns nothing served.
    //
    // Withheld for the two `write_file_changes` rows of one attempt, which the role below
    // already tells apart: the projection write and the coding call's write are two callers,
    // not two runs of one step, and neither column can distinguish them. Saying both would
    // contradict the more specific answer with the vaguer one.
    write === null || operation.repeat?.kind !== 'same_step'
      ? repeatNote(operation.repeat)
      : null,
    // Beside the call it belongs to: a stream that never spoke was closed and asked again,
    // and this row's duration is the only other trace it left.
    reissueNote(operation.stream_reissues),
    // What this write is, said in the words the position licenses: before the coding call, or
    // the change the coding call wrote. Never "workspace prepared" for a write that came after
    // the Engineer, and never "implementation" for one that came before it.
    write === 'pre_coding' ? 'workspace prepared, before the coding call' : null,
    write === 'implementation' ? 'the coding call’s file changes' : null,
    // A running row says how long it has been running before it says how fresh its heartbeat
    // is: 186's question was "how long has this call been going", and the heartbeat answers a
    // different one. Both are read against this browser's clock, from the same reading.
    state === 'running' ? (elapsed ? `running ${elapsed}` : 'running') : null,
    state === 'running' && heartbeat ? `heartbeat ${heartbeat.age} ago` : null,
    // Neither success nor liveness: the raw status word is the honest label. failed_retryable
    // and failed_terminal are different situations and are not merged into "failed".
    state === 'failed' || state === 'other' ? operation.status : null,
    operation.error_code,
    // Last, so a finished row reads "what happened, then how long it took". A row the journal
    // never timestamped both ends of shows no duration at all rather than an estimated one.
    state === 'running' ? null : elapsed,
  ].filter(Boolean);

  const classes = [
    'agent-work__row',
    `agent-work__row--${state}`,
    heartbeat ? (heartbeat.stale ? 'agent-work__row--stale' : 'agent-work__row--alive') : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <li className={classes}>
      <span aria-hidden="true">{ROW_GLYPHS[state]}</span>
      {state === 'succeeded' ? <span className="sr-only">succeeded</span> : null}
      <span className="agent-work__type">{operation.operation_type}</span>
      {parts.length > 0 ? (
        <span className="agent-work__note">{parts.join(' · ')}</span>
      ) : null}
    </li>
  );
}
