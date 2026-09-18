import { Async, TableSkeleton } from '@/components/common/States';
import { Panel } from '@/components/ui/Layout';
import { count } from '@/utils/count';
import { usePullRequests, useWording, useWorkstreams } from './hooks';
import { RepositoryTable } from './RepositoryTable';
import { RepositoryRepairs } from './RepositoryRepair';
import { pullRequestView } from './workstream-view';

/** Every repository this feature touches, and the decisions any of them are waiting on. */
export function RepositoriesTab({ featureId, at }: { featureId: string; at: number | null }) {
  const workstreams = useWorkstreams(featureId, at);
  const pullRequests = usePullRequests(featureId, at);
  const workstreamWording = useWording('workstream');
  const list = workstreams.data?.workstreams ?? [];

  return (
    <div className="stack">
      <RepositoryRepairs featureId={featureId} workstreams={list} />
      <Panel
        title="Repository workstreams"
        meta={workstreams.data ? count(list.length, 'repository', 'repositories') : undefined}
        flush
      >
        <Async query={workstreams} skeleton={<TableSkeleton rows={4} />}>
          {(data) => (
            <RepositoryTable
              featureId={featureId}
              workstreams={data.workstreams}
              wordingFor={workstreamWording}
              pullRequests={(pullRequests.data?.pull_requests ?? []).map((artifact) =>
                pullRequestView(artifact, data.workstreams),
              )}
              onRetried={() => void workstreams.refetch()}
            />
          )}
        </Async>
      </Panel>
    </div>
  );
}
