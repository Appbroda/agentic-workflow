import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import { CredentialFields, type Credentials } from '@/components/common/CredentialFields';
import { modelProviderFor } from '@/utils/model';
import { Markdown } from '@/components/common/Markdown';
import { Badge, RepositoryBadge, SeverityBadge } from '@/components/ui/Badge';
import { CodeBlock } from '@/components/ui/Code';
import { DetailList, DetailRow, Revision } from '@/components/ui/Value';
import type { RepositoryRepair as Repair, Workstream } from '@/schemas/feature';
import { useFeature, useRefreshFeature } from './hooks';

/**
 * A repository the platform could not work in, and the change that would let it.
 *
 * This is a different kind of failure from code that was wrong: the checked-in setup does not
 * let the repository run its own checks, so nothing written there could ever have been
 * validated. The platform stops rather than reporting an implementation it could not check,
 * and writes down a diagnosis precise enough to act on -- but it will not change somebody's
 * repository on its own. That decision is made here.
 *
 * These are rendered as action cards at the top of the page, not buried in a repository row.
 * A feature waiting on a person for two days because nobody expanded the right row is the
 * failure mode this placement exists to prevent.
 */

export function RepositoryRepairs({
  featureId,
  workstreams,
}: {
  featureId: string;
  workstreams: Workstream[];
}) {
  const api = useApi();
  const repairs = useQuery({
    queryKey: ['repairs', featureId],
    queryFn: ({ signal }) => api.listRepairs(featureId, signal),
  });

  const blocked = workstreams.filter((item) => item.blocking_setup_issues.length > 0);
  const all = repairs.data?.repairs ?? [];
  if (all.length === 0 && blocked.length === 0) return null;

  return (
    <>
      {workstreams.map((workstream) => (
        <RepositorySetupCard
          key={workstream.repository_id}
          featureId={featureId}
          workstream={workstream}
          repairs={all.filter((repair) => repair.repository_id === workstream.repository_id)}
        />
      ))}
      {/* A repair for a repository this feature no longer lists still belongs on the page. */}
      {all
        .filter((repair) => !workstreams.some((item) => item.repository_id === repair.repository_id))
        .map((repair) => (
          <section className="action-card" key={repair.repair_id}>
            <RepairCard featureId={featureId} repair={repair} />
          </section>
        ))}
    </>
  );
}

/** One repository's setup problem, with whatever the platform proposed for it. */
export function RepositorySetupCard({
  featureId,
  workstream,
  repairs,
}: {
  featureId: string;
  workstream: Workstream;
  repairs: Repair[];
}) {
  const issues = workstream.blocking_setup_issues;
  if (repairs.length === 0 && issues.length === 0) return null;

  return (
    <section className="action-card" aria-label={`Repository setup: ${workstream.repository_id}`}>
      <div className="action-card__header">
        <h2 className="action-card__title">This repository could not run its own checks</h2>
        <RepositoryBadge repositoryId={workstream.repository_id} />
        {workstream.preflight_status ? <Badge tone="attention">{workstream.preflight_status}</Badge> : null}
      </div>
      <p className="muted">
        Nothing written here could have been validated, so the platform stopped rather than
        reporting an implementation it could not check.
      </p>

      {repairs.map((repair) => (
        <RepairCard key={repair.repair_id} featureId={featureId} repair={repair} />
      ))}

      {/* The raw findings, for a repository the platform found nothing approvable in. Shown
          only then: beside a proposal they are the same problem stated twice. */}
      {repairs.length === 0 ? <BlockingIssues issues={issues} /> : null}
    </section>
  );
}

function RepairCard({ featureId, repair }: { featureId: string; repair: Repair }) {
  const api = useApi();
  const refresh = useRefreshFeature(featureId);
  const feature = useFeature(featureId);
  const [confirming, setConfirming] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [credentials, setCredentials] = useState<Credentials>({});

  const approve = useMutation({
    mutationFn: () => api.approveRepair(featureId, repair.repair_id, { credentials }),
    onSettled: () => {
      setConfirming(false);
      refresh();
    },
  });
  const reject = useMutation({
    mutationFn: () => api.rejectRepair(featureId, repair.repair_id, reason.trim()),
    onSettled: () => {
      setRejecting(false);
      refresh();
    },
  });

  const decided = repair.status !== 'proposed';
  const busy = approve.isPending || reject.isPending;
  // A repair that touches files or dependencies is a change to somebody's repository. The
  // server refuses an approval that does not acknowledge that, so asking here is not the
  // control -- it is making sure the person pressing the button knows what it does.
  const changesRepository =
    repair.changes_source_logic ||
    repair.affected_files.length > 0 ||
    repair.affected_dependencies.length > 0 ||
    repair.commands.length > 0;

  return (
    <article className="card" aria-label={`Repair ${repair.repair_id}`}>
      <div className="card__header">
        <strong>{repair.detected_problem}</strong>
        <Badge>{repair.status}</Badge>
        <Badge outline>{repair.failure_classification}</Badge>
        <Badge outline>risk: {repair.risk}</Badge>
        {repair.stale ? <Badge tone="attention">out of date</Badge> : null}
      </div>

      <DetailList narrow>
        <DetailRow label="Repository">{repair.repository_id}</DetailRow>
        <DetailRow label="Found by">{repair.originating_stage}</DetailRow>
        {repair.proposed_at_revision ? (
          <DetailRow label="Diagnosed at">
            <Revision value={repair.proposed_at_revision} />
          </DetailRow>
        ) : null}
        {repair.resulting_revision ? (
          <DetailRow label="Now at">
            <Revision value={repair.resulting_revision} />
          </DetailRow>
        ) : null}
      </DetailList>

      <p className="prose">
        <strong>Suggested repair:</strong> {repair.proposed_repair}
      </p>
      <p className="muted">{repair.expected_impact}</p>

      <Listing label="Affected files" items={repair.affected_files} />
      <Listing label="Affected dependencies" items={repair.affected_dependencies} />
      <Listing label="Evidence" items={repair.evidence} />

      {repair.commands.length > 0 ? (
        <details>
          <summary className="muted">Commands this would run</summary>
          <div className="stack stack--tight" style={{ marginTop: 'var(--space-3)' }}>
            {repair.commands.map((command, index) => (
              <CodeBlock
                key={index}
                title={command.purpose}
                copyValue={command.command.join(' ')}
              >
                {command.command.join(' ')}
              </CodeBlock>
            ))}
          </div>
        </details>
      ) : null}

      {repair.stale && !decided ? (
        <p className="field__hint" role="status">
          This repository has changed since the diagnosis was written, so the repair may no
          longer describe it. The platform will refuse to apply it and diagnose the current
          checkout instead.
        </p>
      ) : null}

      {decided ? <Outcome repair={repair} /> : null}

      {[approve.error, reject.error].map((error, index) =>
        error ? (
          <p className="field__error" role="alert" key={index}>
            {error instanceof ApiError ? userMessage(error) : 'That did not work.'}
            {error instanceof ApiError && error.detail ? ` — ${error.detail}` : ''}
          </p>
        ) : null,
      )}

      {decided ? null : confirming ? (
        <div className="callout callout--attention">
          <p className="callout__title">Approve this repair?</p>
          <p className="prose">
            {changesRepository
              ? 'This changes the repository’s checked-in files or dependencies, then runs the repository again.'
              : 'This runs the repository again with the repair applied.'}
          </p>
          <details>
            <summary className="muted">Use different keys for this attempt (optional)</summary>
            <CredentialFields
              value={credentials}
              onChange={setCredentials}
              need={[modelProviderFor(feature.data?.agent_platform), 'github']}
              note="Left blank, the attempt runs with the credentials stored in Settings. Keys typed here override them for this request only and are never stored."
            />
          </details>
          <div className="form__actions">
            <button
              type="button"
              className="button button--primary"
              disabled={busy}
              onClick={() => approve.mutate()}
            >
              {approve.isPending ? 'Applying…' : 'Yes, approve and retry'}
            </button>
            <button
              type="button"
              className="button button--quiet"
              disabled={busy}
              onClick={() => setConfirming(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : rejecting ? (
        <div className="callout">
          <div className="field">
            <label className="field__label" htmlFor={`reject-${repair.repair_id}`}>
              Why is this repair not the right change?
            </label>
            {/* Required by the server too. A stop somebody chose to leave in place, with no
                recorded reason, explains nothing to whoever picks the feature up next. */}
            <textarea
              id={`reject-${repair.repair_id}`}
              rows={3}
              value={reason}
              onChange={(event) => setReason(event.target.value)}
            />
          </div>
          <div className="form__actions">
            <button
              type="button"
              className="button button--danger"
              disabled={busy || !reason.trim()}
              onClick={() => reject.mutate()}
            >
              {reject.isPending ? 'Recording…' : 'Reject repair'}
            </button>
            <button
              type="button"
              className="button button--quiet"
              disabled={busy}
              onClick={() => setRejecting(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="form__actions">
          <button
            type="button"
            className="button button--primary"
            disabled={busy}
            onClick={() => setConfirming(true)}
          >
            Approve repair
          </button>
          <button
            type="button"
            className="button button--danger"
            disabled={busy}
            onClick={() => setRejecting(true)}
          >
            Reject repair
          </button>
          <Link
            className="button button--quiet"
            to={`/features/${encodeURIComponent(featureId)}/chat?prompt=${encodeURIComponent(
              `Explain repository repair ${repair.repair_id} for ${repair.repository_id}, its risks, and whether its evidence supports approval.`,
            )}`}
          >
            Ask AI
          </Link>
        </div>
      )}
    </article>
  );
}

/** What became of a repair somebody already decided about. */
function Outcome({ repair }: { repair: Repair }) {
  if (repair.status === 'rejected') {
    return (
      <p className="muted">
        Rejected{repair.rejected_by ? ` by ${repair.rejected_by}` : ''}
        {repair.rejection_reason ? `: ${repair.rejection_reason}` : '.'}
      </p>
    );
  }
  if (repair.status === 'superseded') {
    return <p className="muted">{repair.execution_result ?? 'Superseded before it was applied.'}</p>;
  }
  return (
    <div className="callout callout--info">
      <p className="callout__title">
        {repair.status === 'succeeded'
          ? 'Repair applied'
          : repair.status === 'executing'
            ? 'Applying the repair…'
            : 'The repair did not fix it'}
      </p>
      {repair.approved_by ? <p className="muted">Approved by {repair.approved_by}.</p> : null}
      {repair.execution_result ? <Markdown>{repair.execution_result}</Markdown> : null}
      {repair.resulting_revision ? (
        <p className="muted">The repository is now at {repair.resulting_revision.slice(0, 12)}.</p>
      ) : null}
    </div>
  );
}

/** The preflight findings, when the platform found nothing it could propose a repair for. */
function BlockingIssues({ issues }: { issues: Workstream['blocking_setup_issues'] }) {
  if (issues.length === 0) return null;
  return (
    <ul className="cards">
      {issues.map((issue, index) => (
        <li key={text(issue.issue_id) ?? index} className="card">
          <div className="card__header">
            <strong>{text(issue.issue_id) ?? `Issue ${index + 1}`}</strong>
            {text(issue.severity) ? <SeverityBadge severity={text(issue.severity)!} /> : null}
            {text(issue.category) ? <Badge outline>{text(issue.category)}</Badge> : null}
          </div>
          <p className="prose">{text(issue.description) ?? ''}</p>
          {text(issue.evidence) ? <p className="muted">Evidence: {text(issue.evidence)}</p> : null}
          {text(issue.recommended_action) ? (
            <p className="prose">
              <strong>Suggested repair:</strong> {text(issue.recommended_action)}
            </p>
          ) : null}
          <p className="muted">
            The platform proposed no repair for this, so it is not something a fixed command can
            put right. Somebody has to make the change, then grant this repository another
            attempt.
          </p>
        </li>
      ))}
    </ul>
  );
}

function Listing({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <p className="muted">
      {label}: {items.join(', ')}
    </p>
  );
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}
