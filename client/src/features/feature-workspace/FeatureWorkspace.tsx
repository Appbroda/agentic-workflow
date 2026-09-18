import { useEffect, useState } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { ErrorState, TableSkeleton } from '@/components/common/States';
import { StatusBadge, Badge } from '@/components/ui/Badge';
import { LinkTabs, PageHeader, type TabDefinition } from '@/components/ui/Layout';
import { CopyValue, DetailList, DetailRow } from '@/components/ui/Value';
import { count } from '@/utils/count';
import { performanceTierName } from '@/utils/model';
import { IconChat } from '@/components/ui/icons';
import { absoluteTime, elapsed, relativeTime } from '@/utils/time';
import { useFeature, usePullRequests, useWording, useWorkstreams } from './hooks';
import { useLiveFeature } from './useLiveFeature';
import { Overview } from './Overview';
import { WorkflowTab } from './WorkflowTab';
import { RepositoriesTab } from './RepositoriesTab';
import { PrdTab, PlanTab } from './DocumentTab';
import { ArtifactsTab } from './ArtifactsTab';
import { HistoryTab } from './HistoryTab';
import { LogbookTab } from './LogbookTab';
import { PullRequestsTab } from './PullRequestsTab';
import { ChatPanel } from './ChatPanel';
import { Notifications } from './Notifications';
import { ProgressStrip } from './ProgressStrip';
import { FeatureControls } from './HumanActions';
import { PublishFeature } from './PublishFeature';
import { ReviseFeature } from './ReviseFeature';
import { effectiveFeatureStatus } from './feature-status';

/**
 * One feature, for both audiences.
 *
 * The header answers a product manager's questions without a click: what is being built, where
 * it is, whether it is waiting on somebody, how many repositories and pull requests it has.
 * Everything an engineer needs is a tab or a row away, and nothing has been removed to get
 * there -- the progression is summary, then detail, then the raw evidence underneath it.
 */

const TABS = [
  'overview',
  'workflow',
  'prd',
  'plan',
  'repositories',
  'artifacts',
  'logbook',
  'history',
  'pull-requests',
] as const;
type Tab = (typeof TABS)[number];

/**
 * Sections this workspace used to have as their own tabs, and where their content now lives.
 * Links people already sent each other keep working.
 */
const MOVED: Record<string, Tab> = {
  requirements: 'prd',
  contract: 'plan',
  timeline: 'history',
  agents: 'history',
  chat: 'overview',
};

export function FeatureWorkspace() {
  const { featureId = '', tab } = useParams<{ featureId: string; tab?: string }>();
  const [params, setParams] = useSearchParams();
  const active: Tab = TABS.includes(tab as Tab) ? (tab as Tab) : (MOVED[tab ?? ''] ?? 'overview');

  const feature = useFeature(featureId);
  const featureWording = useWording('feature');
  // Watches the event table and refreshes whatever changed.
  const liveStatus = feature.data ? effectiveFeatureStatus(feature.data) : undefined;
  const { lastEventId, events, live } = useLiveFeature(featureId, liveStatus);
  const workstreams = useWorkstreams(featureId, lastEventId);
  const pullRequests = usePullRequests(featureId, lastEventId);

  // The assistant opens with the workspace when a link asked for it -- the repair card's "Ask
  // AI" does exactly that -- and otherwise stays closed. Whether it is open lives in the URL,
  // so "look at this feature with the assistant open" is a link.
  const wantsChat = tab === 'chat' || params.has('prompt') || params.has('chat');
  const [chatOpen, setChatOpen] = useState(() => wantsChat);
  useEffect(() => {
    if (wantsChat) setChatOpen(true);
  }, [wantsChat]);

  if (feature.isPending) {
    return (
      <div className="page">
        <TableSkeleton rows={8} label="Loading feature…" />
      </div>
    );
  }
  if (feature.isError) {
    return (
      <div className="page">
        <ErrorState error={feature.error} onRetry={feature.refetch} />
      </div>
    );
  }

  const data = feature.data;
  const effectiveStatus = effectiveFeatureStatus(data);
  const wording = featureWording(effectiveStatus);
  // Workstreams exist only once the planner has agreed the contract, so a queued feature has
  // none — and reading the count from them made a feature submitted against two repositories
  // announce "0 repositories" for as long as it sat in the queue. The feature's own count is
  // what it was submitted with; the workstreams are preferred once there are any, because a
  // repository can be added or dropped during planning.
  const repositoryCount = workstreams.data?.workstreams.length || data.repository_count;
  const pullRequestCount = pullRequests.data?.pull_requests.length ?? 0;
  const attention =
    wording.tone === 'attention' ||
    (workstreams.data?.workstreams ?? []).some((item) => item.status === 'waiting_for_contract_change');

  const base = `/features/${encodeURIComponent(featureId)}`;
  const tabs: TabDefinition[] = [
    { id: 'overview', label: 'Overview', to: base },
    { id: 'workflow', label: 'Workflow', to: `${base}/workflow` },
    { id: 'prd', label: 'PRD', to: `${base}/prd` },
    { id: 'plan', label: 'Plan', to: `${base}/plan` },
    { id: 'repositories', label: 'Repositories', to: `${base}/repositories`, count: repositoryCount },
    { id: 'artifacts', label: 'Artifacts', to: `${base}/artifacts` },
    // Beside History, and before it: the story is what most people want, and the record
    // in order is what somebody debugging wants next.
    { id: 'logbook', label: 'Logbook', to: `${base}/logbook` },
    { id: 'history', label: 'History', to: `${base}/history` },
    { id: 'pull-requests', label: 'Pull requests', to: `${base}/pull-requests`, count: pullRequestCount },
  ];

  return (
    <div className={chatOpen ? 'workspace-split workspace-split--chat' : 'workspace-split'}>
      <div className="page stack">
        <PageHeader
          title={data.title}
          // Above the title, small, and outside the heading: the reference is what somebody
          // arrived holding, and the title is what the page is about.
          eyebrow={
            <span className="page-header__reference mono">
              <CopyValue value={data.reference ?? data.feature_id} label="Copy feature ID" />
            </span>
          }
          badges={
            <>
              <StatusBadge wording={wording} />
              {attention ? <Badge tone="attention">Action required</Badge> : null}
              <Badge outline>{data.execution_mode}</Badge>
              {/* The performance tier, said once and here: the graph's edge labels write the
                  reasoning effort as "effort: high" precisely so a bare word there is never
                  read as this. Absent on a feature from a server that predates tiers. */}
              {performanceTierName(data.performance_tier) ? (
                <Badge outline>{performanceTierName(data.performance_tier)} tier</Badge>
              ) : null}
            </>
          }
          subtitle={
            <DetailList narrow>
              <DetailRow label="Repositories">
                {count(repositoryCount, 'repository', 'repositories')}
              </DetailRow>
              <DetailRow label="Pull requests">{pullRequestCount || '—'}</DetailRow>
              <DetailRow label="Phase">{wording.headline}</DetailRow>
              <DetailRow label="Agent">{data.current_agent ?? '—'}</DetailRow>
              <DetailRow label="Created">
                <span title={absoluteTime(data.created_at)}>{relativeTime(data.created_at)}</span>
              </DetailRow>
              <DetailRow label="Last activity">
                <span title={absoluteTime(data.updated_at)}>{relativeTime(data.updated_at)}</span>
              </DetailRow>
              <DetailRow label="Duration">{elapsed(data.created_at, data.updated_at) ?? '—'}</DetailRow>
            </DetailList>
          }
          // The internal execution id an engineer occasionally needs, kept available and
          // clearly secondary: it is not this feature's identity.
          footer={
            <details className="page-header__technical">
              <summary className="subtle">Technical details</summary>
              <DetailList narrow>
                <DetailRow label="Workflow ID">
                  <CopyValue value={data.workflow_id} label="Copy workflow ID" />
                </DetailRow>
                <DetailRow label="Internal feature ID">
                  <CopyValue value={data.feature_id} label="Copy internal feature ID" />
                </DetailRow>
              </DetailList>
            </details>
          }
          actions={
            <>
              <LiveIndicator live={live} status={effectiveStatus} updatedAt={data.updated_at} />
              <button
                type="button"
                className="button"
                aria-pressed={chatOpen}
                onClick={() => {
                  const next = new URLSearchParams(params);
                  if (chatOpen) {
                    next.delete('chat');
                    next.delete('prompt');
                  } else {
                    next.set('chat', 'open');
                  }
                  setParams(next, { replace: true });
                  setChatOpen((open) => !open);
                }}
              >
                <IconChat />
                Ask AI
              </button>
              {/* Before the generic controls: for a feature that did not land this is the
                  decision the page exists to offer, and it replaces the pull request such a
                  feature used to open on its own. It renders nothing when the server does
                  not advertise it. */}
              <PublishFeature feature={data} workstreams={workstreams.data?.workstreams ?? []} />
              {/* The one action a completed feature still has: asking for changes to the work
                  it published. Renders nothing unless the server advertises it. */}
              <ReviseFeature feature={data} workstreams={workstreams.data?.workstreams ?? []} />
              <FeatureControls
                feature={data}
                // A waiting feature has a specific human gate: clarification answers or an
                // immutable contract decision. Empty resume is not that decision and must not
                // be offered beside the authoritative control.
                genericResumeAllowed={data.status !== 'waiting_for_human'}
              />
            </>
          }
        />

        <ProgressStrip
          featureId={featureId}
          feature={data}
          workstreams={workstreams.data?.workstreams ?? []}
          at={lastEventId}
        />

        {/* Above the tabs so it is visible whichever section is open: a repository that
            stopped is not only the overview's business. */}
        <Notifications events={events} />

        <LinkTabs label="Feature sections" tabs={tabs} isActive={(item) => item.id === active} />

        {active === 'overview' ? <Overview featureId={featureId} at={lastEventId} /> : null}
        {active === 'workflow' ? <WorkflowTab featureId={featureId} at={lastEventId} /> : null}
        {active === 'prd' ? <PrdTab featureId={featureId} at={lastEventId} /> : null}
        {active === 'plan' ? <PlanTab featureId={featureId} at={lastEventId} /> : null}
        {active === 'repositories' ? <RepositoriesTab featureId={featureId} at={lastEventId} /> : null}
        {active === 'artifacts' ? <ArtifactsTab featureId={featureId} /> : null}
        {active === 'logbook' ? <LogbookTab featureId={featureId} at={lastEventId} /> : null}
        {active === 'history' ? <HistoryTab featureId={featureId} at={lastEventId} /> : null}
        {active === 'pull-requests' ? <PullRequestsTab featureId={featureId} at={lastEventId} /> : null}
      </div>

      {chatOpen ? (
        <>
        {/* Below the width where the assistant can sit beside the page it becomes an overlay.
            The scrim is what says so -- without it the panel reads as the layout breaking
            rather than as something covering it. It is not rendered at all on wide screens,
            where the panel is genuinely beside the content. */}
        <div className="chat-scrim" onClick={() => setChatOpen(false)} aria-hidden="true" />
        <ChatPanel
          featureId={featureId}
          feature={data}
          repositories={(workstreams.data?.workstreams ?? []).map((item) => item.repository_id)}
          onClose={() => {
            const next = new URLSearchParams(params);
            next.delete('chat');
            next.delete('prompt');
            setParams(next, { replace: true });
            setChatOpen(false);
          }}
        />
        </>
      ) : null}
    </div>
  );
}

/**
 * Whether what is on screen is current.
 *
 * It reports a condition rather than announcing news, so it is small and sits with the other
 * controls. It says "Live" while the event stream is connected, "Reconnecting" when it is not
 * and the feature can still change, and nothing implementation-specific either way -- a reader
 * does not need to know that a proxy closed a server-sent event stream.
 */
function LiveIndicator({
  live,
  status,
  updatedAt,
}: {
  live: boolean;
  status: string;
  updatedAt: string;
}) {
  const settled = ['completed', 'failed', 'cancelled', 'cancelled_with_external_side_effects'].includes(status);
  if (settled) {
    return (
      <span className="live" title={absoluteTime(updatedAt)}>
        <span className="live__dot" aria-hidden="true" />
        Last changed {relativeTime(updatedAt)}
      </span>
    );
  }
  return (
    <span className={live ? 'live live--on' : 'live live--off'} role="status">
      <span className="live__dot" aria-hidden="true" />
      {live ? 'Live' : 'Reconnecting…'}
    </span>
  );
}
