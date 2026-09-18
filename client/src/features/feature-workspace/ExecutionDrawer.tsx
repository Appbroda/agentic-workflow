import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Badge, RepositoryBadge, SeverityBadge } from '@/components/ui/Badge';
import { CodeBlock } from '@/components/ui/Code';
import { Drawer } from '@/components/ui/Drawer';
import { CopyValue, DetailList, DetailRow, Revision } from '@/components/ui/Value';
import type { ExecutionRecord } from '@/schemas/feature';
import { humanise } from '@/utils/text';
import { absoluteTime, formatDuration } from '@/utils/time';
import { modelDisplayName, providerDisplayName } from '@/utils/model';
import { handlerDetail, statusWord } from './executions';

/**
 * One arrow, opened.
 *
 * A node answers "what stage is this". This answers the other half: what moved the feature
 * here, which model or which handler did it, on which attempt, and -- for a retry -- why the
 * previous attempt failed and what the next one has to change. Those last two are separate
 * questions with separate answers: a repository whose checked-in lint configuration is broken
 * *failed* as a lint error and is *remediated* by repairing the repository, not by writing the
 * same file again.
 *
 * Everything shown is a field the server published. Nothing here summarises a failure itself,
 * and nothing here shows model reasoning: the platform does not record any, and this would not
 * be the place for it if it did. Where a value is missing it is absent, not filled in.
 *
 * The drawer routes rather than reimplements. Every piece of real evidence -- the validation
 * output, the review findings, the repository diff, the artifact, the repair, the pull request
 * -- already has a screen, and the links go there.
 *
 * Which attempt is chosen at the top, defaulting to the newest -- the one the arrow was
 * labelled with. The mechanism was always here; what shipped was a list of subtle buttons at
 * the BOTTOM, below the detail it drove, with nothing saying the list was a selector. The
 * operator watching run 201 read it as a log. It is now the same labelled "Attempt" dropdown
 * the agent-work drawer has, in the one place a control that drives a page belongs.
 */

export function ExecutionDrawer({
  featureId,
  executions,
  onDismiss,
}: {
  featureId: string;
  /** Every execution on the arrow, oldest first. */
  executions: ExecutionRecord[];
  onDismiss: () => void;
}) {
  // The newest attempt is what the arrow was labelled with, so it is what opens.
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const current = executions.at(-1);
  if (!current) return null;
  const record = executions.find((item) => item.execution_id === selectedId) ?? current;

  const from = humanise(record.from_stage);
  const to = humanise(record.to_stage);

  return (
    <Drawer
      title={record.is_retry ? 'Retry execution' : 'Execution'}
      subtitle={
        <span className="row" style={{ gap: 'var(--space-2)', flexWrap: 'wrap' }}>
          <span>
            {from} → {to}
          </span>
          {record.repository_id ? <RepositoryBadge repositoryId={record.repository_id} /> : null}
          <Badge tone={tone(record.status)}>{statusWord(record.status)}</Badge>
        </span>
      }
      onDismiss={onDismiss}
    >
      <div className="stack">
        {/* The selector, above the detail it drives. With a single execution there is nothing
            to choose between, so the control does not appear and the drawer reads exactly as
            it did before. */}
        {executions.length > 1 ? (
          <div className="field agent-work__attempt-picker">
            <label className="field__label" htmlFor="execution-attempt">
              Attempt
            </label>
            <select
              id="execution-attempt"
              value={record.execution_id}
              onChange={(event) => setSelectedId(event.target.value)}
            >
              {/* Newest first, because the newest is the one the arrow was labelled with. */}
              {[...executions].reverse().map((item) => (
                <option key={item.execution_id} value={item.execution_id}>
                  {attemptOptionText(item)}
                </option>
              ))}
            </select>
          </div>
        ) : null}
        <Summary record={record} />
        {/* The platform's own sentence for what the ratio counts, in its own line rather than
            in a column of the grid above: it is a sentence, and the grid is for values. Shown
            rather than reworded, because the backend enforces these budgets and two clients
            must not explain them differently. */}
        {record.attempt_meaning ? <p className="subtle">{record.attempt_meaning}</p> : null}
        {record.is_retry ||
        record.failure_summary ||
        record.remediation_summary ||
        record.human_requirement ||
        record.routing_reason ? (
          <Explanation record={record} />
        ) : null}
        <Evidence record={record} />
        <Links featureId={featureId} record={record} />
        <Technical record={record} />
      </div>
    </Drawer>
  );
}

function tone(status: string): 'done' | 'working' | 'attention' | 'stopped' | 'neutral' {
  if (status === 'completed') return 'done';
  if (status === 'running' || status === 'queued') return 'working';
  if (status === 'needs_human' || status === 'blocked') return 'attention';
  if (status === 'failed' || status === 'cancelled') return 'stopped';
  return 'neutral';
}

/**
 * Who ran this, and against what.
 *
 * The agent and the model are separate rows on purpose. The agent is the role the platform
 * asked to do the work -- what makes a review a review is that it never writes code, whatever
 * it is configured with -- and the model is what answered.
 */
function Summary({ record }: { record: ExecutionRecord }) {
  return (
    <DetailList narrow>
      <DetailRow label="Handler">{handlerDetail(record)}</DetailRow>
      {record.agent_type ? <DetailRow label="Agent">{record.agent_type}</DetailRow> : null}
      {record.handler_type === 'model' && record.provider ? (
        <DetailRow label="Provider">{providerDisplayName(record.provider)}</DetailRow>
      ) : null}
      {record.reasoning_effort ? (
        <DetailRow label="Reasoning">{record.reasoning_effort}</DetailRow>
      ) : null}
      {record.model_role ? (
        <DetailRow label="Model role">{humanise(record.model_role)}</DetailRow>
      ) : null}
      {record.attempt !== null && record.attempt !== undefined ? (
        <DetailRow label={record.is_retry ? 'Remediation attempt' : 'Attempt'}>
          {record.max_attempts ? `${record.attempt} / ${record.max_attempts}` : record.attempt}
        </DetailRow>
      ) : null}
      {record.counter_label && record.counter_value !== null && record.counter_value !== undefined ? (
        <DetailRow label={record.counter_label}>
          {record.counter_limit === null || record.counter_limit === undefined
            ? record.counter_value
            : `${record.counter_value} / ${record.counter_limit}`}
        </DetailRow>
      ) : null}
      {record.completed_at ? (
        <DetailRow label="Finished">{absoluteTime(record.completed_at)}</DetailRow>
      ) : null}
      {record.duration_seconds !== null && record.duration_seconds !== undefined ? (
        <DetailRow label="Duration">
          {formatDuration(record.duration_seconds * 1000) ?? `${record.duration_seconds.toFixed(1)}s`}
        </DetailRow>
      ) : null}
      {record.execution_mode === 'mock' ? (
        <DetailRow label="Execution mode">
          Mock — no provider was contacted for this feature
        </DetailRow>
      ) : null}
    </DetailList>
  );
}

/** Why the previous attempt failed, and what the next one has to change. Never merged. */
function Explanation({ record }: { record: ExecutionRecord }) {
  return (
    <div className="stack stack--tight">
      {record.failure_summary ? (
        <section className="callout callout--attention">
          <p className="callout__title">
            {record.is_retry ? 'Why the previous attempt failed' : 'Why this failed'}
          </p>
          <p className="prose">
            {record.failure_severity ? (
              <>
                <SeverityBadge severity={record.failure_severity} />{' '}
              </>
            ) : null}
            {record.failure_summary}
          </p>
          {record.failure_classification ? (
            <p className="subtle">
              Classified by the platform as {humanise(record.failure_classification)}.
            </p>
          ) : null}
        </section>
      ) : null}
      {record.remediation_summary ? (
        <section className="callout callout--info">
          <p className="callout__title">What needs to be fixed</p>
          <p className="prose">{record.remediation_summary}</p>
        </section>
      ) : null}
      {record.human_requirement ? (
        <section className="callout callout--attention">
          <p className="callout__title">
            {record.human_action === 'retry_refused'
              ? 'No further retry is scheduled'
              : 'What is required'}
          </p>
          <p className="prose">{record.human_requirement}</p>
        </section>
      ) : null}
      {record.routing_reason ? (
        <section className="callout callout--info">
          <p className="callout__title">Why this model was selected</p>
          <p className="prose">{record.routing_reason}</p>
        </section>
      ) : null}
    </div>
  );
}

/** The measurable outcome: revisions, review verdict, validation, the failing command. */
function Evidence({ record }: { record: ExecutionRecord }) {
  const hasRevisions = Boolean(record.revision_before || record.revision_after);
  const hasValidation =
    record.validation_total !== null && record.validation_total !== undefined;
  if (
    !hasRevisions &&
    !hasValidation &&
    !record.review_verdict &&
    record.command.length === 0 &&
    !record.contract_version
  ) {
    return null;
  }
  return (
    <DetailList narrow>
      {record.revision_before ? (
        <DetailRow label="Previous revision">
          <Revision value={record.revision_before} />
        </DetailRow>
      ) : null}
      {record.revision_after ? (
        <DetailRow label="Current revision">
          <Revision value={record.revision_after} />
        </DetailRow>
      ) : null}
      {record.review_verdict ? (
        <DetailRow label="Review result">{humanise(record.review_verdict)}</DetailRow>
      ) : null}
      {record.review_finding_count !== null && record.review_finding_count !== undefined ? (
        <DetailRow label="Findings">{record.review_finding_count}</DetailRow>
      ) : null}
      {hasValidation ? (
        <DetailRow label="Validation">
          {record.validation_passed ?? 0} of {record.validation_total} passed
        </DetailRow>
      ) : null}
      {record.exit_code !== null && record.exit_code !== undefined ? (
        <DetailRow label="Exit code">{record.exit_code}</DetailRow>
      ) : null}
      {record.contract_version ? (
        <DetailRow label="Contract version">{record.contract_version}</DetailRow>
      ) : null}
      {record.command.length > 0 ? (
        <DetailRow label="Command">
          <CopyValue value={record.command.join(' ')} label="Copy command" />
        </DetailRow>
      ) : null}
    </DetailList>
  );
}

/** The glyph each status reads as, matching the agent-work drawer's own vocabulary. */
const STATUS_GLYPHS: Record<string, string> = {
  completed: '✓',
  failed: '✗',
  cancelled: '✗',
  running: '●',
  queued: '●',
};

/**
 * One option of the attempt dropdown: everything the removed list showed, on one line.
 *
 * The list carried a number, a handler, a status and a failure summary. The first three are
 * here; the fourth is rendered in full in the Explanation block for whichever attempt is
 * selected, which is where a sentence belongs. Nothing was dropped silently.
 *
 * `item.attempt` can be null -- the list rendered it '—' and so does this. And the glyph never
 * travels without its word: a native option carries text only, so "✗" alone would be nothing
 * at all to a screen reader.
 */
function attemptOptionText(item: ExecutionRecord): string {
  const ratio =
    item.attempt === null || item.attempt === undefined
      ? '—'
      : item.max_attempts
        ? `${item.attempt} / ${item.max_attempts}`
        : `${item.attempt}`;
  const handler =
    item.handler_type === 'model'
      ? (modelDisplayName(item.model) ?? 'model not recorded')
      : item.handler;
  const glyph = STATUS_GLYPHS[item.status] ?? '·';
  return `${glyph} ${[ratio, handler, statusWord(item.status)].filter(Boolean).join(' · ')}`;
}

/**
 * Where to go for the evidence itself.
 *
 * Links only. The validation output, the review findings, the diff and the repair proposal all
 * have screens already, and rebuilding any of them inside a graph drawer would give the
 * platform two places that could disagree about the same artifact.
 */
function Links({ featureId, record }: { featureId: string; record: ExecutionRecord }) {
  const base = `/features/${encodeURIComponent(featureId)}`;
  const repository = record.repository_id
    ? `${base}/repositories/${encodeURIComponent(record.repository_id)}`
    : null;
  const artifactLink = (artifactId: string) =>
    `${base}/artifacts?artifact=${encodeURIComponent(artifactId)}`;

  const links: { to: string; label: string }[] = [];
  if (record.review_artifact_id && repository) {
    links.push({ to: `${repository}?view=review`, label: 'Open review' });
  }
  if (record.to_stage === 'validation' && repository) {
    links.push({ to: `${repository}?view=validation`, label: 'Open validation' });
  }
  if (record.to_stage === 'implementation' && repository) {
    links.push({ to: `${repository}?view=changes`, label: 'Open changes' });
  }
  if (repository) {
    links.push({ to: repository, label: 'Open repository' });
  }
  if (record.human_action === 'clarification') {
    links.push({ to: `${base}/prd`, label: 'Answer the questions' });
  }
  if (record.repair_id && repository) {
    links.push({ to: repository, label: 'Open repair' });
  }
  if (record.to_stage === 'pull_requests' || record.pull_request_artifact_id) {
    links.push({ to: `${base}/pull-requests`, label: 'Open pull requests' });
  }
  if (record.artifact_id) {
    links.push({ to: artifactLink(record.artifact_id), label: 'Open artifact' });
  }
  if (record.result_artifact_id && record.result_artifact_id !== record.artifact_id) {
    links.push({ to: artifactLink(record.result_artifact_id), label: 'Open attempt record' });
  }

  // The same destination can be reached two ways -- a review artifact and the review view --
  // and a row of duplicate buttons reads as a bug.
  const unique = links.filter(
    (item, index) => links.findIndex((other) => other.to === item.to) === index,
  );
  if (unique.length === 0) return null;

  return (
    <div className="row" style={{ gap: 'var(--space-2)', flexWrap: 'wrap' }}>
      {unique.map((item) => (
        <Link key={item.to + item.label} className="button button--small" to={item.to}>
          {item.label}
        </Link>
      ))}
    </div>
  );
}

/** Fingerprints and identifiers. Real, useful, and not what a first look should be made of. */
function Technical({ record }: { record: ExecutionRecord }) {
  const rows: { label: string; value: string }[] = [
    { label: 'Execution id', value: record.execution_id },
    ...(record.model ? [{ label: 'Model identifier', value: record.model }] : []),
    ...(record.model_variable
      ? [{ label: 'Configuration key', value: record.model_variable }]
      : []),
    ...(record.input_fingerprint
      ? [{ label: 'Input fingerprint', value: record.input_fingerprint }]
      : []),
    ...(record.code_completion_artifact_id
      ? [{ label: 'Code completion artifact', value: record.code_completion_artifact_id }]
      : []),
    ...(record.review_artifact_id
      ? [{ label: 'Review artifact', value: record.review_artifact_id }]
      : []),
    ...(record.result_artifact_id
      ? [{ label: 'Attempt record', value: record.result_artifact_id }]
      : []),
    ...(record.repair_id ? [{ label: 'Repair', value: record.repair_id }] : []),
  ];
  return (
    <details>
      <summary className="muted">Technical details</summary>
      <DetailList narrow>
        {rows.map((row) => (
          <DetailRow key={row.label} label={row.label}>
            <CopyValue value={row.value} label={`Copy ${row.label.toLowerCase()}`} />
          </DetailRow>
        ))}
      </DetailList>
      {record.command.length > 0 ? (
        <CodeBlock copyValue={record.command.join(' ')}>{record.command.join(' ')}</CodeBlock>
      ) : null}
    </details>
  );
}
