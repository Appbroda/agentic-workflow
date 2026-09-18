import { useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import type { TimelineEvent } from '@/schemas/feature';
import { AgentBadge, Badge, RepositoryBadge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { EmptyState } from '@/components/common/States';
import { absoluteTime, relativeTime } from '@/utils/time';
import { humanise } from '@/utils/text';
import { agentRuns, duration, type AgentRun } from './agent-runs';

/**
 * Which agent ran, on what, and how it ended.
 *
 * Derived from the same events as the timeline rather than from a second endpoint, because
 * the platform records agent work as those events and a parallel source would drift. What it
 * shows is what an agent produced -- never any account of how it reasoned, which the platform
 * does not record and this would not display if it did.
 */
export function AgentHistory({ events, featureId }: { events: TimelineEvent[]; featureId: string }) {
  const navigate = useNavigate();
  const runs = useMemo(() => agentRuns(events), [events]);

  const columns: Column<AgentRun>[] = [
    {
      key: 'agent',
      header: 'Agent',
      sortValue: (run) => run.agent,
      render: (run) => <AgentBadge agent={run.agent} />,
    },
    {
      key: 'repository',
      header: 'Repository',
      shrink: true,
      sortValue: (run) => run.repositoryId ?? '',
      render: (run) =>
        run.repositoryId ? (
          <RepositoryBadge repositoryId={run.repositoryId} />
        ) : (
          <span className="subtle">Feature-wide</span>
        ),
    },
    {
      key: 'produced',
      header: 'Produced',
      sortValue: (run) => run.artifactId ?? '',
      render: (run) =>
        run.artifactId ? (
          <Link
            className="truncate"
            to={`/features/${encodeURIComponent(featureId)}/artifacts?artifact=${encodeURIComponent(run.artifactId)}`}
            title={run.artifactId}
          >
            {artifactLabel(run.artifactId)}
          </Link>
        ) : (
          <span className="subtle">—</span>
        ),
    },
    {
      key: 'status',
      header: 'Status',
      shrink: true,
      sortValue: (run) => run.outcome,
      render: (run) => (
        <Badge tone={run.outcome === 'completed' ? 'done' : run.outcome === 'failed' ? 'stopped' : 'working'}>
          {run.outcome}
        </Badge>
      ),
    },
    {
      key: 'attempt',
      header: 'Attempt',
      shrink: true,
      align: 'right',
      sortValue: (run) => run.attempt,
      render: (run) => (run.attempt === null ? <span className="subtle">—</span> : run.attempt + 1),
    },
    {
      key: 'duration',
      header: 'Duration',
      shrink: true,
      align: 'right',
      sortValue: (run) => (run.endedAt ? Date.parse(run.endedAt) - Date.parse(run.startedAt) : null),
      // Not every start/end pair brackets real work, so an unmeasurable run says nothing
      // rather than claiming zero seconds.
      render: (run) => <span className="subtle">{duration(run) ?? '—'}</span>,
    },
    {
      key: 'started',
      header: 'Started',
      shrink: true,
      sortValue: (run) => Date.parse(run.startedAt),
      render: (run) => (
        <span className="nowrap" title={absoluteTime(run.startedAt)}>
          {relativeTime(run.startedAt)}
        </span>
      ),
    },
  ];

  return (
    <DataTable
      label="Agent history"
      columns={columns}
      rows={runs}
      rowKey={(run) => run.key}
      onRowClick={(run) =>
        run.artifactId
          ? navigate(
              `/features/${encodeURIComponent(featureId)}/artifacts?artifact=${encodeURIComponent(run.artifactId)}`,
            )
          : undefined
      }
      empty={
        <EmptyState
          title="No agent activity yet"
          detail="Each agent's run appears here once it produces its first result."
        />
      }
    />
  );
}

/** The artifact's type, read from the identifier the platform gave it. */
function artifactLabel(artifactId: string): string {
  const withoutIndex = artifactId.replace(/^\d+_/, '');
  return humanise(withoutIndex.split('.')[0] ?? artifactId);
}
