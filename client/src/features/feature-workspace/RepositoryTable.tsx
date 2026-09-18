import { Link, useNavigate } from 'react-router-dom';
import type { Workstream } from '@/schemas/feature';
import type { StatusWording } from '@/hooks/useStatusVocabulary';
import { Badge, RepositoryBadge, StatusBadge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { CopyValue, DetailList, DetailRow, ExternalLink, Revision } from '@/components/ui/Value';
import { EmptyState } from '@/components/common/States';
import type { PullRequestView } from './workstream-view';
import { attempts, reached, reviewOutcome, validationSummary } from './workstream-view';
import { RetryWorkstream } from './RetryWorkstream';

/**
 * The repositories a feature is being built across.
 *
 * Identity is `repository_id`. The role is shown because it is useful, never switched on: a
 * feature can name five repositories and three of them can share a role, and nothing here
 * assumes a frontend/backend pair or a fixed count.
 *
 * Each row opens the repository's own page. The expander gives the two or three facts that
 * usually settle the question without going there.
 */
export function RepositoryTable({
  featureId,
  workstreams,
  wordingFor,
  pullRequests,
  onRetried,
}: {
  featureId: string;
  workstreams: Workstream[];
  wordingFor: (status: string) => StatusWording;
  pullRequests: PullRequestView[];
  onRetried?: () => void;
}) {
  const navigate = useNavigate();
  const path = (repositoryId: string) =>
    `/features/${encodeURIComponent(featureId)}/repositories/${encodeURIComponent(repositoryId)}`;

  const pullRequestFor = (workstream: Workstream) =>
    pullRequests.find(
      (item) =>
        item.artifactId === workstream.pull_request_artifact_id ||
        item.repositoryId === workstream.repository_id,
    );

  const columns: Column<Workstream>[] = [
    {
      key: 'repository',
      header: 'Repository',
      sortValue: (item) => (item.repository_name ?? item.repository_id).toLowerCase(),
      render: (item) => (
        <div className="stack" style={{ gap: '2px', minWidth: '14rem' }}>
          <Link className="table__primary truncate" to={path(item.repository_id)}>
            {item.repository_name ?? item.repository_id}
          </Link>
          <span className="table__secondary">
            <CopyValue value={item.repository_id} label="Copy repository ID" />
          </span>
        </div>
      ),
    },
    {
      key: 'role',
      header: 'Role',
      shrink: true,
      sortValue: (item) => item.repository_role ?? '',
      render: (item) => (item.repository_role ? <Badge outline>{item.repository_role}</Badge> : <Dash />),
    },
    {
      key: 'status',
      header: 'Status',
      shrink: true,
      sortValue: (item) => wordingFor(item.status).headline,
      render: (item) => <StatusBadge wording={wordingFor(item.status)} />,
    },
    {
      key: 'reached',
      header: 'Reached',
      shrink: true,
      sortValue: (item) => reached(item),
      render: (item) => (
        <span className="nowrap" title="The furthest evidence this repository produced.">
          {reached(item)}
        </span>
      ),
    },
    {
      key: 'validation',
      header: 'Validation',
      shrink: true,
      sortValue: (item) => {
        const summary = validationSummary(item);
        return summary ? summary.passed - summary.total : null;
      },
      render: (item) => {
        const summary = validationSummary(item);
        if (!summary) return <Dash />;
        const failed = summary.total - summary.passed;
        return (
          <Link to={`${path(item.repository_id)}?view=validation`}>
            <Badge tone={failed === 0 ? 'done' : 'stopped'}>
              {summary.passed}/{summary.total} passed
            </Badge>
          </Link>
        );
      },
    },
    {
      key: 'review',
      header: 'Review',
      shrink: true,
      sortValue: (item) => reviewOutcome(item) ?? '',
      render: (item) => {
        const outcome = reviewOutcome(item);
        if (!outcome) return <Dash />;
        return (
          <Link to={`${path(item.repository_id)}?view=review`}>
            <Badge tone={outcome === 'approved' ? 'done' : 'stopped'}>
              {outcome === 'approved' ? 'Approved' : 'Rejected'}
            </Badge>
          </Link>
        );
      },
    },
    {
      key: 'attempts',
      header: 'Attempt',
      shrink: true,
      align: 'right',
      sortValue: (item) => attempts(item),
      // Nothing has been attempted yet for a repository that has not started, so this says
      // so rather than claiming a first attempt that has not happened.
      render: (item) => (item.status === 'pending' ? <Dash /> : <span>{attempts(item)}</span>),
    },
    {
      key: 'pull-request',
      header: 'PR',
      shrink: true,
      sortValue: (item) => pullRequestFor(item)?.number ?? null,
      render: (item) => {
        const pullRequest = pullRequestFor(item);
        if (!pullRequest) return <Dash />;
        const label = pullRequest.number ? `#${pullRequest.number}` : 'Open';
        return pullRequest.url ? (
          <ExternalLink href={pullRequest.url}>{label}</ExternalLink>
        ) : (
          <span>{label}</span>
        );
      },
    },
  ];

  return (
    <DataTable
      label="Repository workstreams"
      columns={columns}
      rows={workstreams}
      rowKey={(item) => item.repository_id}
      onRowClick={(item) => navigate(path(item.repository_id))}
      expandLabel="Show detail"
      expand={(item) => (
        <RowSummary
          featureId={featureId}
          workstream={item}
          pullRequest={pullRequestFor(item)}
          onRetried={onRetried}
        />
      )}
      empty={
        <EmptyState
          title="No repository workstreams yet"
          detail="The planner assigns repositories once it has agreed the shared contract."
        />
      }
    />
  );
}

function Dash() {
  return <span className="subtle">—</span>;
}

function RowSummary({
  featureId,
  workstream,
  pullRequest,
  onRetried,
}: {
  featureId: string;
  workstream: Workstream;
  pullRequest: PullRequestView | undefined;
  onRetried?: () => void;
}) {
  const profile = workstream.technology_profile ?? {};
  const stack = [profile.primary_language, ...(Array.isArray(profile.frameworks) ? profile.frameworks : [])]
    .filter((item): item is string => typeof item === 'string')
    .join(' · ');

  return (
    <div className="stack stack--tight">
      <DetailList>
        {stack ? <DetailRow label="Technology">{stack}</DetailRow> : null}
        {workstream.selected_package_manager ? (
          <DetailRow label="Package manager">{workstream.selected_package_manager}</DetailRow>
        ) : null}
        <DetailRow label="Branch">
          <CopyValue value={workstream.branch_name} label="Copy branch" />
        </DetailRow>
        {workstream.current_revision ? (
          <DetailRow label="Revision">
            <Revision value={workstream.current_revision} />
          </DetailRow>
        ) : null}
        <DetailRow label="Files changed">
          {workstream.production_files_changed.length +
            workstream.test_files_changed.length +
            workstream.configuration_files_changed.length}
        </DetailRow>
        {workstream.repository_url ? (
          <DetailRow label="Repository">
            <ExternalLink href={workstream.repository_url}>{workstream.repository_url}</ExternalLink>
          </DetailRow>
        ) : null}
      </DetailList>

      {workstream.planned_blind ? (
        <div className="callout callout--warn">
          {/* The plan for this repository was written without reconnaissance evidence, so it
              is the one most likely to have invented conventions the checkout does not
              have. Shown beside the workstream because that is where somebody deciding
              whether to trust the plan is already looking. */}
          <p className="callout__title">Planned without reconnaissance</p>
          <p className="prose">
            Reading this repository failed before planning
            {workstream.planned_blind_reason ? `: ${workstream.planned_blind_reason}` : '.'}{' '}
            Its first attempt re-reads the checkout to recover registry evidence.
          </p>
        </div>
      ) : null}

      {workstream.blocking_issues.length > 0 ? (
        <div className="callout callout--stopped">
          <p className="callout__title">Blocking issues</p>
          <ul className="bullets">
            {/* The last entry is usually the operator question the platform ended on; it is
                shown in full rather than truncated, because it is the actionable part. */}
            {workstream.blocking_issues.map((issue, index) => (
              <li key={index} className="prose">
                {issue}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {workstream.retry_refusal_reason ? (
        <div className="callout callout--warn">
          <p className="callout__title">Why the platform stopped retrying</p>
          <p className="prose">{workstream.retry_refusal_reason}</p>
        </div>
      ) : null}

      {/* Whether pressing Publish would open a pull request for this repository, and under
          which class -- or the server's own sentence saying why it would not. Rendered here
          because "2 ready to open, 1 held for lint" is a per-repository fact, and a bare
          button on the feature cannot say it. Only while there is no pull request yet. */}
      {workstream.pull_request_artifact_id === null && workstream.publication_class ? (
        <div className={
          workstream.publication_class === 'unreviewed'
            ? 'callout callout--attention'
            : 'callout'
        }>
          <p className="callout__title">
            {workstream.publication_class === 'unreviewed'
              ? 'Rejected by review, publishable on request'
              : 'Passed review, waiting to be published'}
          </p>
          <p className="prose">
            {workstream.publication_class === 'unreviewed'
              ? 'Every check this repository requires passed and the review still said no. Publishing it overrules that judgement; its pull request is titled [REVIEW REJECTED] and lists what the reviewer asked for.'
              : 'This work passed its review. The feature did not land, so its pull request opens when somebody publishes this feature.'}
          </p>
        </div>
      ) : null}
      {workstream.pull_request_artifact_id === null && workstream.publication_refusal ? (
        <div className="callout callout--warn">
          <p className="callout__title">Not offered for publication</p>
          <p className="prose">{workstream.publication_refusal}</p>
        </div>
      ) : null}


      {workstream.retry_grants.length > 0 ? (
        <div className="callout">
          {/* An override of the platform's own stop, shown as an override: without this the
              only trace is an attempt count past the configured limit, which reads as a
              platform defect rather than as a decision somebody made. */}
          <p className="callout__title">Attempts granted by a person</p>
          <ul className="bullets">
            {workstream.retry_grants.map((grant, index) => (
              <li key={index} className="prose">
                {String(grant.granted_by ?? 'unknown')} granted {String(grant.attempts ?? '?')} —{' '}
                {String(grant.reason ?? '')}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="form__actions">
        <Link
          className="button button--small"
          to={`/features/${encodeURIComponent(featureId)}/repositories/${encodeURIComponent(workstream.repository_id)}`}
        >
          Open repository
        </Link>
        {/* Offered only where the platform says it would accept one. A control whose only
            outcome is a 409 is worse than no control. */}
        {workstream.available_actions?.includes('RETRY_WORKSTREAM') ? (
          <RetryWorkstream
            featureId={featureId}
            workstream={workstream}
            onDone={() => onRetried?.()}
          />
        ) : null}
        {pullRequest?.url ? (
          <a className="button button--small" href={pullRequest.url} target="_blank" rel="noopener noreferrer">
            Open pull request
          </a>
        ) : null}
        <span className="subtle">
          <RepositoryBadge repositoryId={workstream.repository_id} />
        </span>
      </div>
    </div>
  );
}
