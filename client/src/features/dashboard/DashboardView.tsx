import { useMemo } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { ErrorState, EmptyState, TableSkeleton } from '@/components/common/States';
import { StatusBadge, Badge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { FilterChips, PageHeader, Panel, type FilterChip } from '@/components/ui/Layout';
import { CopyValue } from '@/components/ui/Value';
import { count } from '@/utils/count';
import { IconPlus, IconSearch } from '@/components/ui/icons';
import { statusWording, useStatusVocabulary, type StatusWording } from '@/hooks/useStatusVocabulary';
import { absoluteTime, elapsed, relativeTime } from '@/utils/time';
import type { FeatureSummary } from '@/schemas/feature';
import {
  GROUP_LABELS,
  GROUP_ORDER,
  GROUP_TONES,
  filterFeatures,
  isFeatureGroup,
  type FeatureGroup,
} from './grouping';
import { useFeatureList } from './useFeatures';

/**
 * The operational home: everything the platform has been asked to build.
 *
 * A table rather than a card per feature — ninety-odd features exist in this deployment, and
 * the question somebody opens this page to answer, "is anything waiting on me", is a scanning
 * question. The counts along the top are the filters, not decoration, so noticing a number and
 * acting on it are the same gesture.
 *
 * The filter and the search live in the URL. That is what replaced the three separate activity
 * routes: one screen, and a link that still says which rows the sender was looking at.
 */

export function DashboardView({ group }: { group?: FeatureGroup } = {}) {
  const navigate = useNavigate();
  const vocabulary = useStatusVocabulary();
  const { query, features: loaded, counts } = useFeatureList();
  const [params, setParams] = useSearchParams();

  const fromUrl = params.get('group');
  const activeGroup: FeatureGroup | 'all' = group ?? (isFeatureGroup(fromUrl) ? fromUrl : 'all');
  const search = params.get('q') ?? '';
  const status = params.get('status') ?? 'all';

  const setParam = (key: string, value: string | null) => {
    const next = new URLSearchParams(params);
    if (value === null || value === '' || value === 'all') next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true });
  };

  const wordingFor = useMemo(
    () => (statusValue: string) => statusWording(vocabulary.data, 'feature', statusValue),
    [vocabulary.data],
  );

  const visible = useMemo(
    () => filterFeatures(loaded, { search, status, group: activeGroup }),
    [loaded, search, status, activeGroup],
  );
  const statuses = useMemo(() => [...new Set(loaded.map((item) => item.status))].sort(), [loaded]);
  const filtered = search !== '' || status !== 'all' || activeGroup !== 'all';

  const chips: FilterChip<FeatureGroup | 'all'>[] = [
    { value: 'all', label: 'All', count: loaded.length },
    ...GROUP_ORDER.map((item) => ({
      value: item,
      label: GROUP_LABELS[item],
      count: counts[item],
      tone: GROUP_TONES[item],
    })),
  ];

  return (
    <div className="page stack">
      <PageHeader
        title="Features"
        subtitle="Everything this platform has been asked to build, from the moment it was accepted."
        actions={
          <Link className="button button--primary" to="/features/new">
            <IconPlus />
            New feature
          </Link>
        }
      />

      {/* Counts are a claim about what exists. They appear once there is an answer, rather
          than reading zero for everything while the first page is still in flight. */}
      {query.data ? (
        <FilterChips
          label="Feature groups"
          value={activeGroup}
          chips={chips}
          metric
          onChange={(value) => setParam('group', value === 'all' ? null : value)}
        />
      ) : null}

      <Panel flush>
        <div className="toolbar">
          <label className="filters__field" style={{ position: 'relative' }}>
            <span className="sr-only">Search</span>
            <input
              type="search"
              value={search}
              placeholder="Search AB-Feature-42, a title, or an ID"
              onChange={(event) => setParam('q', event.target.value)}
              aria-label="Search"
            />
          </label>
          <label className="filters__field">
            <span className="sr-only">Status</span>
            <select
              aria-label="Status"
              value={status}
              onChange={(event) => setParam('status', event.target.value)}
            >
              <option value="all">All statuses</option>
              {statuses.map((item) => (
                <option key={item} value={item}>
                  {wordingFor(item).headline}
                </option>
              ))}
            </select>
          </label>
          <span className="toolbar__spacer" />
          {/* Said plainly: the list endpoint is cursor-paged and has no search, so filtering
              applies to what has been loaded rather than to everything that exists. */}
          <span className="subtle">
            Searching {count(loaded.length, 'loaded feature')}.
          </span>
          {query.hasNextPage ? (
            <button
              type="button"
              className="button button--small"
              onClick={() => void query.fetchNextPage()}
              disabled={query.isFetchingNextPage}
            >
              {query.isFetchingNextPage ? 'Loading…' : 'Load more'}
            </button>
          ) : null}
        </div>

        {query.isPending ? (
          <TableSkeleton rows={6} label="Loading features…" />
        ) : query.isError ? (
          <ErrorState error={query.error} onRetry={query.refetch} />
        ) : (
          <DataTable
            label="Features"
            columns={featureColumns(wordingFor)}
            rows={visible}
            rowKey={(feature) => feature.feature_id}
            onRowClick={(feature) => navigate(`/features/${encodeURIComponent(feature.feature_id)}`)}
            empty={
              loaded.length === 0 ? (
                <EmptyState
                  title="No features yet"
                  detail="Submit a PRD and the platform will plan it, split it across your repositories, and open pull requests."
                  action={
                    <Link className="button button--primary" to="/features/new">
                      <IconPlus />
                      New feature
                    </Link>
                  }
                />
              ) : (
                <EmptyState
                  title="Nothing matches those filters"
                  detail={`${count(loaded.length, 'feature')} loaded. Clear the search or choose another category.`}
                  action={
                    filtered ? (
                      <button
                        type="button"
                        className="button"
                        onClick={() => setParams(new URLSearchParams(), { replace: true })}
                      >
                        <IconSearch />
                        Clear filters
                      </button>
                    ) : undefined
                  }
                />
              )
            }
          />
        )}
      </Panel>
    </div>
  );
}

function featureColumns(wordingFor: (status: string) => StatusWording): Column<FeatureSummary>[] {
  return [
    {
      key: 'reference',
      header: 'ID',
      shrink: true,
      sortValue: (feature) => feature.reference ?? feature.feature_id,
      // The identity somebody arrives holding — it is in the pull request title and in
      // whatever message sent them here — so it leads the row and is copyable in one click.
      render: (feature) => (
        <span className="mono nowrap">
          <CopyValue
            value={feature.reference ?? feature.feature_id}
            label="Copy feature ID"
          />
        </span>
      ),
    },
    {
      key: 'feature',
      header: 'Feature',
      sortValue: (feature) => feature.title.toLowerCase(),
      render: (feature) => (
        // A floor rather than a fixed width: without it the shrink columns take everything and
        // a long feature title wraps to three lines in a 200px cell.
        <div className="stack" style={{ gap: '2px', minWidth: '20rem' }}>
          <Link
            className="table__primary"
            to={`/features/${encodeURIComponent(feature.feature_id)}`}
          >
            {feature.title}
          </Link>
        </div>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      shrink: true,
      sortValue: (feature) => wordingFor(feature.status).headline,
      render: (feature) => (
        <span className="row" style={{ gap: 'var(--space-2)' }}>
          <StatusBadge wording={wordingFor(feature.status)} />
        </span>
      ),
    },
    {
      key: 'attention',
      header: 'Needs you',
      shrink: true,
      sortValue: (feature) => (feature.human_action_required ? 0 : 1),
      render: (feature) =>
        feature.human_action_required ? (
          <Badge tone="attention">Action required</Badge>
        ) : (
          <span className="subtle">—</span>
        ),
    },
    {
      key: 'repositories',
      header: 'Repos',
      shrink: true,
      align: 'right',
      sortValue: (feature) => feature.repository_count,
      render: (feature) => <span title={count(feature.repository_count, 'repository', 'repositories')}>{feature.repository_count}</span>,
    },
    {
      key: 'pull-requests',
      header: 'PRs',
      shrink: true,
      align: 'right',
      sortValue: (feature) => feature.pull_request_count,
      render: (feature) =>
        feature.pull_request_count > 0 ? (
          <Link to={`/features/${encodeURIComponent(feature.feature_id)}/pull-requests`}>
            {feature.pull_request_count}
          </Link>
        ) : (
          <span className="subtle">—</span>
        ),
    },
    {
      key: 'mode',
      header: 'Mode',
      shrink: true,
      sortValue: (feature) => feature.execution_mode,
      render: (feature) => (
        <Badge outline title={feature.execution_mode === 'live' ? 'Ran against real repositories' : 'Ran without touching a repository'}>
          {feature.execution_mode}
        </Badge>
      ),
    },
    {
      key: 'started',
      header: 'Started',
      shrink: true,
      sortValue: (feature) => Date.parse(feature.created_at),
      render: (feature) => (
        <span className="nowrap" title={absoluteTime(feature.created_at)}>
          {relativeTime(feature.created_at)}
        </span>
      ),
    },
    {
      key: 'activity',
      header: 'Last activity',
      shrink: true,
      sortValue: (feature) => Date.parse(feature.updated_at),
      render: (feature) => (
        <span className="nowrap" title={absoluteTime(feature.updated_at)}>
          {relativeTime(feature.updated_at)}
        </span>
      ),
    },
    {
      key: 'duration',
      header: 'Duration',
      shrink: true,
      align: 'right',
      sortValue: (feature) => Date.parse(feature.updated_at) - Date.parse(feature.created_at),
      render: (feature) => (
        <span className="nowrap subtle">{elapsed(feature.created_at, feature.updated_at) ?? '—'}</span>
      ),
    },
  ];
}
