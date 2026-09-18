import { Fragment } from 'react';
import { Async, EmptyState, TableSkeleton } from '@/components/common/States';
import { Badge, RepositoryBadge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { Panel } from '@/components/ui/Layout';
import { CopyValue, DetailList, DetailRow, ExternalLink, Revision } from '@/components/ui/Value';
import type { Tone } from '@/components/ui/tone';
import { useArtifact, useArtifactList, useFeature, usePullRequests, useWorkstreams } from './hooks';
import { pullRequestView, type PullRequestView } from './workstream-view';

/**
 * The pull requests this feature opened, and the order they should merge in.
 *
 * The order is read from the execution plan the platform wrote. It is never inferred here and
 * never assumed to be backend-first: a feature's repositories are whatever it has, and merging
 * a consumer before the thing it consumes is exactly the mistake this table exists to prevent.
 */
export function PullRequestsTab({ featureId, at }: { featureId: string; at: number | null }) {
  const feature = useFeature(featureId);
  const pullRequests = usePullRequests(featureId, at);
  const workstreams = useWorkstreams(featureId, at);
  const plan = useArtifactList(featureId, 'repository_execution_plan', at);
  const planArtifact = useArtifact(featureId, plan.data?.artifacts.at(-1)?.artifact_id ?? null);
  const payload = planArtifact.data?.payload;

  const order = mergeOrder(payload);
  const dependencies = dependencyMap(payload);
  const list = workstreams.data?.workstreams ?? [];

  const columns: Column<PullRequestView>[] = [
    {
      key: 'repository',
      header: 'Repository',
      sortValue: (item) => item.repositoryId ?? '',
      render: (item) => (
        <div className="stack" style={{ gap: '2px', minWidth: '13rem' }}>
          <span className="table__primary truncate" title={item.repository ?? undefined}>
            {item.repository ?? item.repositoryId}
          </span>
          {item.repositoryId ? (
            <span className="table__secondary">
              <RepositoryBadge repositoryId={item.repositoryId} />
            </span>
          ) : null}
        </div>
      ),
    },
    {
      key: 'title',
      header: 'Title',
      sortValue: (item) => item.title ?? '',
      // The platform titles every pull request with the feature's reference, so this column is
      // where somebody looking at a list of pull requests can see which feature each belongs
      // to. Held to a width: titles run long and the merge-order columns are the point here.
      render: (item) =>
        item.title ? (
          <span className="truncate" style={{ maxWidth: '22rem' }} title={item.title}>
            {item.title}
          </span>
        ) : (
          <span className="subtle">—</span>
        ),
    },
    {
      key: 'number',
      header: 'PR',
      shrink: true,
      sortValue: (item) => item.number,
      render: (item) => {
        const label = item.number ? `#${item.number}` : 'Open';
        return item.url ? <ExternalLink href={item.url}>{label}</ExternalLink> : <span>{label}</span>;
      },
    },
    {
      key: 'state',
      header: 'Status',
      shrink: true,
      sortValue: (item) => item.state ?? '',
      render: (item) => (
        <span className="row" style={{ gap: 'var(--space-2)' }}>
          {item.state ? (
            <Badge tone={pullRequestTone(item.state)}>{item.state}</Badge>
          ) : (
            <span className="subtle">—</span>
          )}
          {item.draft ? <Badge outline>draft</Badge> : null}
        </span>
      ),
    },
    {
      key: 'branch',
      header: 'Branch',
      sortValue: (item) => item.sourceBranch ?? '',
      render: (item) =>
        item.sourceBranch ? (
          // Branch names here run to ninety characters. Held to a width so the columns after
          // it -- dependency and merge order, which are the point of this table -- stay on
          // screen; the whole name is in the tooltip and on the clipboard.
          <div style={{ maxWidth: '20rem' }}>
            <CopyValue value={item.sourceBranch} label="Copy branch" />
          </div>
        ) : (
          <span className="subtle">—</span>
        ),
    },
    {
      key: 'target',
      header: 'Target',
      shrink: true,
      sortValue: (item) => item.targetBranch ?? '',
      render: (item) => <span className="mono">{item.targetBranch ?? '—'}</span>,
    },
    {
      key: 'dependency',
      header: 'Depends on',
      shrink: true,
      render: (item) => {
        const depends = item.repositoryId ? (dependencies[item.repositoryId] ?? []) : [];
        return depends.length > 0 ? (
          <span className="row" style={{ gap: 'var(--space-1)' }}>
            {depends.map((id) => (
              <RepositoryBadge key={id} repositoryId={id} />
            ))}
          </span>
        ) : (
          <span className="subtle">—</span>
        );
      },
    },
    {
      key: 'order',
      header: 'Merge order',
      shrink: true,
      align: 'right',
      sortValue: (item) => positionIn(order, item.repositoryId),
      render: (item) => {
        const position = positionIn(order, item.repositoryId);
        return position === null ? <span className="subtle">—</span> : <span>{position}</span>;
      },
    },
  ];

  return (
    <div className="stack">
      {order.length > 1 ? (
        <Panel title="Merge order" meta="From the platform's execution plan">
          <div className="progress-strip">
            {order.map((item, index) => (
              <Fragment key={item}>
                {index > 0 ? (
                  <span className="progress-strip__sep" aria-hidden="true">
                    →
                  </span>
                ) : null}
                <RepositoryBadge repositoryId={item} />
              </Fragment>
            ))}
          </div>
          {typeof payload?.merge_strategy === 'string' || typeof payload?.deployment_strategy === 'string' ? (
            <DetailList narrow>
              <DetailRow label="Merge strategy">{String(payload?.merge_strategy ?? 'unspecified')}</DetailRow>
              <DetailRow label="Deployment">{String(payload?.deployment_strategy ?? 'unspecified')}</DetailRow>
            </DetailList>
          ) : null}
        </Panel>
      ) : null}

      <Panel
        title="Pull requests"
        meta={
          feature.data?.reference
            ? `Every pull request opened for ${feature.data.reference}`
            : undefined
        }
        flush
      >
        <Async query={pullRequests} skeleton={<TableSkeleton rows={3} />}>
          {(data) => (
            <DataTable
              label="Pull requests"
              columns={columns}
              rows={data.pull_requests.map((artifact) => pullRequestView(artifact, list))}
              rowKey={(item) => item.artifactId}
              expandLabel="Show pull request detail"
              expand={(item) => <PullRequestDetail item={item} />}
              empty={
                <EmptyState
                  title="No pull requests yet"
                  detail="They are opened once a repository passes its own review."
                />
              }
            />
          )}
        </Async>
      </Panel>
    </div>
  );
}

function PullRequestDetail({ item }: { item: PullRequestView }) {
  return (
    <div className="stack stack--tight">
      {item.title ? <p className="prose">{item.title}</p> : null}
      <DetailList>
        {item.commitSha ? (
          <DetailRow label="Commit">
            <Revision value={item.commitSha} />
          </DetailRow>
        ) : null}
        {item.reviewers.length > 0 ? (
          <DetailRow label="Reviewers">{item.reviewers.join(', ')}</DetailRow>
        ) : null}
        {item.labels.length > 0 ? <DetailRow label="Labels">{item.labels.join(', ')}</DetailRow> : null}
      </DetailList>
      {item.url ? (
        <div className="form__actions">
          <a className="button button--small" href={item.url} target="_blank" rel="noopener noreferrer">
            Open on GitHub
          </a>
        </div>
      ) : null}
    </div>
  );
}

/**
 * GitHub's own word for a pull request's state, coloured.
 *
 * `open` is in progress rather than good news -- nothing merges itself here, so an open pull
 * request is still work waiting on a person.
 */
function pullRequestTone(state: string): Tone {
  const value = state.trim().toLowerCase();
  if (value === 'merged') return 'done';
  if (value === 'closed') return 'stopped';
  if (value === 'open') return 'working';
  return 'neutral';
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function mergeOrder(plan: Record<string, unknown> | undefined): string[] {
  const recommended = strings(plan?.recommended_merge_order);
  return recommended.length > 0 ? recommended : strings(plan?.execution_order);
}

/** Which repositories each workstream depends on, as the plan recorded it. */
function dependencyMap(plan: Record<string, unknown> | undefined): Record<string, string[]> {
  const workstreams = Array.isArray(plan?.workstreams) ? plan.workstreams : [];
  const map: Record<string, string[]> = {};
  for (const item of workstreams) {
    if (typeof item !== 'object' || item === null) continue;
    const record = item as Record<string, unknown>;
    const id = typeof record.repository_id === 'string' ? record.repository_id : null;
    if (!id) continue;
    map[id] = strings(record.dependency_workstream_ids);
  }
  return map;
}

/**
 * A repository's position in the merge order.
 *
 * The plan lists workstream ids, which are usually but not always the repository id, so a
 * prefix match catches `admanager-server-health-history` for `admanager-server` without
 * inventing an order for a repository the plan never named.
 */
function positionIn(order: string[], repositoryId: string | null): number | null {
  if (!repositoryId) return null;
  const index = order.findIndex((item) => item === repositoryId || item.startsWith(`${repositoryId}-`));
  return index === -1 ? null : index + 1;
}
