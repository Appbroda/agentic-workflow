import { Link } from 'react-router-dom';
import type { Feature } from '@/schemas/feature';
import { Async, EmptyState, PartialError, TableSkeleton } from '@/components/common/States';
import { Panel } from '@/components/ui/Layout';
import { Badge, RepositoryBadge } from '@/components/ui/Badge';
import { CodeBlock } from '@/components/ui/Code';
import { DetailList, DetailRow } from '@/components/ui/Value';
import { count } from '@/utils/count';
import { modelDisplayName, performanceTierName, providerDisplayName } from '@/utils/model';
import { absoluteTime, relativeTime } from '@/utils/time';
import {
  useArtifact,
  useArtifactList,
  useClarification,
  useFeature,
  useIntegrationReview,
  usePullRequests,
  useTimeline,
  useWording,
  useWorkstreams,
} from './hooks';
import { buildFeatureGraph, currentActivity, planView } from './graph';
import { RepositoryTable } from './RepositoryTable';
import { WorkflowMap } from './WorkflowMap';
import { buildStages } from './stages';
import { ClarificationPanel, InvestigatingPanel } from './HumanActions';
import { DesignConflictPanels } from './DesignConflicts';
import { ContractChangeActions } from './ContractChangeActions';
import { RepositoryRepairs } from './RepositoryRepair';
import { ActionHistory } from './ActionHistory';
import { TimelineList } from './TimelineList';
import { pullRequestView } from './workstream-view';
import { effectiveFeatureStatus } from './feature-status';

/**
 * What a person needs before deciding whether to look further.
 *
 * The order is the order of the questions: is anything waiting on me, where has this got to,
 * how is each repository doing, what happened recently. A product manager should be able to
 * stop after the first two sections; an engineer carries on into the repository table and out
 * to the evidence behind it.
 *
 * Each section loads independently. One of them failing shows a message in that section's
 * place and leaves the rest of the page alone.
 */
export function Overview({ featureId, at }: { featureId: string; at: number | null }) {
  const feature = useFeature(featureId);
  const workstreams = useWorkstreams(featureId, at);
  const timeline = useTimeline(featureId, at);
  const clarification = useClarification(featureId, at);
  const pullRequests = usePullRequests(featureId, at);
  const artifacts = useArtifactList(featureId, undefined, at);
  const integrationReview = useIntegrationReview(featureId, at);
  const workstreamWording = useWording('workstream');

  const list = workstreams.data?.workstreams ?? [];
  const prViews = (pullRequests.data?.pull_requests ?? []).map((artifact) =>
    pullRequestView(artifact, list),
  );

  // Serialized by the server; the fallback covers payloads from before the field existed.
  const clarificationState =
    clarification.data?.clarification_state ??
    (clarification.data?.awaiting_answers ? 'awaiting_answers' : 'idle');

  return (
    <div className="stack">
      {/* Anything that needs a person, first and unmissable -- and only what actually
          does. `investigating` renders the platform's own open items, explicitly not
          waiting on the user; the answer form appears only once the platform has decided
          it needs the human, with the grounding-failure fallback named when that is why. */}
      {clarificationState === 'investigating' && clarification.data ? (
        <InvestigatingPanel clarification={clarification.data} />
      ) : null}
      {clarification.data?.awaiting_answers && clarification.data.questions.length > 0 ? (
        <ClarificationPanel
          featureId={featureId}
          reference={feature.data?.reference}
          clarification={clarification.data}
          groundingFailed={clarificationState === 'asked_after_grounding_failure'}
        />
      ) : null}
      {/* A different question from the ones above: a workstream stopped because a design
          decision is being reversed, and only a person can settle which position holds.
          Rendered from the list rather than from `clarificationState`, so a design question
          is still shown when a requirement question happens to outrank it. */}
      <DesignConflictPanels
        featureId={featureId}
        reference={feature.data?.reference}
        conflicts={clarification.data?.design_conflicts ?? []}
      />
      <RepositoryRepairs featureId={featureId} workstreams={list} />
      {list.some((item) => item.status === 'waiting_for_contract_change') ? (
        <ContractChangeActions featureId={featureId} at={at} />
      ) : null}
      {feature.data ? <FailureEvidence feature={feature.data} /> : null}
      {feature.data ? <CleanupRequirements feature={feature.data} /> : null}

      <div className="grid-2">
        <Panel title="Current state">
          {feature.data ? <CurrentState feature={feature.data} /> : <TableSkeleton rows={3} />}
        </Panel>
        <Panel
          title="Progress"
          actions={
            <Link
              className="button button--small"
              to={`/features/${encodeURIComponent(featureId)}/workflow`}
            >
              View workflow
            </Link>
          }
        >
          {feature.data && workstreams.data && artifacts.data ? (
            <>
              <WorkflowMap
                stages={buildStages(
                  feature.data,
                  workstreams.data.workstreams,
                  new Set(artifacts.data.artifacts.map((item) => item.artifact_type)),
                  integrationReview,
                )}
              />
              {/* One line saying what is happening right now, read from the same graph the
                  Workflow tab draws so the two cannot say different things. */}
              <CurrentStep featureId={featureId} at={at} />
            </>
          ) : (
            <TableSkeleton rows={4} />
          )}
        </Panel>
      </div>

      <Panel
        title="Repositories"
        // Silent until the answer is known: "0 repositories" beside a loading skeleton is a
        // statement, and a wrong one. So is "0 repositories" on a queued feature that names
        // two: workstreams do not exist until the planner has agreed the contract, which is
        // what the empty state below explains.
        meta={
          workstreams.data
            ? count(list.length || feature.data?.repository_count || 0, 'repository', 'repositories')
            : undefined
        }
        actions={
          <Link className="button button--small" to={`/features/${encodeURIComponent(featureId)}/repositories`}>
            Open all
          </Link>
        }
        flush
      >
        {workstreams.isError ? (
          <PartialError what="the repository workstreams" error={workstreams.error} onRetry={workstreams.refetch} />
        ) : (
          <Async query={workstreams} skeleton={<TableSkeleton rows={3} />}>
            {(data) => (
              <RepositoryTable
                featureId={featureId}
                workstreams={data.workstreams}
                wordingFor={workstreamWording}
                pullRequests={prViews}
                onRetried={() => void workstreams.refetch()}
              />
            )}
          </Async>
        )}
      </Panel>

      <Panel
        title="Recent activity"
        actions={
          <Link className="button button--small" to={`/features/${encodeURIComponent(featureId)}/history`}>
            Full history
          </Link>
        }
        flush
      >
        {timeline.isError ? (
          <PartialError what="recent activity" error={timeline.error} onRetry={timeline.refetch} />
        ) : (
          <Async query={timeline} skeleton={<TableSkeleton rows={4} />}>
            {(data) =>
              data.events.length === 0 ? (
                <EmptyState title="Nothing has happened yet" detail="Events appear here as the platform works." />
              ) : (
                <TimelineList featureId={featureId} events={[...data.events].reverse().slice(0, 8)} />
              )
            }
          </Async>
        )}
      </Panel>

      <ActionHistory featureId={featureId} at={at} />
    </div>
  );
}

/**
 * The step the feature is on, in one line.
 *
 * "Still running" and "running its fourth attempt at the same repository" are different
 * situations, and only one of them is worth interrupting somebody about. This is the second,
 * said plainly, with the count measured against the budget the platform published.
 */
function CurrentStep({ featureId, at }: { featureId: string; at: number | null }) {
  const feature = useFeature(featureId);
  const workstreams = useWorkstreams(featureId, at);
  const artifacts = useArtifactList(featureId, undefined, at);
  const plans = useArtifactList(featureId, 'repository_execution_plan', at);
  const planArtifact = useArtifact(featureId, plans.data?.artifacts.at(-1)?.artifact_id ?? null);
  const integrationReview = useIntegrationReview(featureId, at);
  const wordingFor = useWording('workstream');

  if (!feature.data || !workstreams.data || !artifacts.data) return null;
  const graph = buildFeatureGraph({
    feature: feature.data,
    workstreams: workstreams.data.workstreams,
    artifactTypes: new Set(artifacts.data.artifacts.map((item) => item.artifact_type)),
    plan: planView(planArtifact.data?.payload),
    featureId,
    describeWorkstream: (status: string) => wordingFor(status).headline,
    integrationReview,
  });
  const now = currentActivity(graph);
  if (!now) return null;

  return (
    <p className="row" style={{ gap: 'var(--space-2)' }}>
      <span className="details__label">Currently</span>
      <span>{now.label}</span>
      {now.repositoryId ? <RepositoryBadge repositoryId={now.repositoryId} /> : null}
      {now.retrying ? <Badge tone="attention">Retrying</Badge> : null}
      {now.counter ? (
        <span className="subtle">
          {now.counter.label} {now.counter.current}/{now.counter.limit}
        </span>
      ) : null}
    </p>
  );
}

/**
 * What is happening right now, and what comes next.
 *
 * Both sentences are the server's: the status vocabulary owns the plain-language reading of a
 * status and its next step. `failed_requires_human` is exact and tells a product manager
 * nothing, and inventing better copy here would be this client asserting something it cannot
 * know.
 */
function CurrentState({ feature }: { feature: Feature }) {
  const wording = useWording('feature')(effectiveFeatureStatus(feature));
  return (
    <div className="stack stack--tight">
      <p className="prose">{wording.detail || wording.headline}</p>
      {wording.next_step ? (
        <p className="muted">
          <strong>Next:</strong> {wording.next_step}
        </p>
      ) : null}
      <DetailList narrow>
        <DetailRow label="Agent">{feature.current_agent ?? 'None running'}</DetailRow>
        <DetailRow label="Mode">{feature.execution_mode}</DetailRow>
        {/* Written with the same helper the execution records use, so one feature's provider
            is spelled the same everywhere it appears. */}
        {providerDisplayName(feature.agent_platform) ? (
          <DetailRow label="Agent platform">
            {providerDisplayName(feature.agent_platform)}
          </DetailRow>
        ) : null}
        {/* The cost basis this feature was submitted at, beside the provider it runs on.
            Absent on a feature read back from a server that predates tiers, which is shown
            as nothing rather than as a tier it might not have run at. */}
        {performanceTierName(feature.performance_tier) ? (
          <DetailRow label="Performance tier">
            {performanceTierName(feature.performance_tier)}
          </DetailRow>
        ) : null}
        <DetailRow label="Clarification rounds">{feature.clarification_rounds}</DetailRow>
        <DetailRow label="Review cycles">{feature.integration_review_cycles}</DetailRow>
        <DetailRow label="Last activity">
          <span title={absoluteTime(feature.updated_at)}>{relativeTime(feature.updated_at)}</span>
        </DetailRow>
        {feature.cancellation_status !== 'not_requested' ? (
          <DetailRow label="Cancellation">{feature.cancellation_status}</DetailRow>
        ) : null}
      </DetailList>
      <PinnedSetup feature={feature} />
    </div>
  );
}

/**
 * The setup a custom feature was pinned to at submission — four roles, their platforms,
 * models and efforts — exactly as snapshotted, whatever became of the setup since.
 *
 * The "edited since" sentence is G3 made visible rather than merely true: the feature keeps
 * running on its snapshot, and this says so where somebody comparing the setup's current
 * values would otherwise conclude the run had moved.
 */
function PinnedSetup({ feature }: { feature: Feature }) {
  const pinned = feature.model_setup;
  if (!pinned) return null;
  return (
    <div className="stack stack--tight">
      <p className="muted">
        <strong>Model setup:</strong> {pinned.name} — pinned at submission.
        {pinned.setup_state === 'edited'
          ? ' The setup has been edited since; this feature keeps running on the values below.'
          : ''}
        {pinned.setup_state === 'deleted'
          ? ' The setup has been deleted since; this feature keeps running on the values below.'
          : ''}
      </p>
      <DetailList narrow>
        {pinned.roles.map((role) => (
          <DetailRow key={role.role} label={role.role}>
            <span>
              {providerDisplayName(role.platform) ?? role.platform} ·{' '}
              {modelDisplayName(role.model) ?? role.model}
              {role.reasoning_effort ? ` · effort: ${role.reasoning_effort}` : ''}
              {role.max_tokens != null ? ` · max ${role.max_tokens.toLocaleString()}` : ''}
            </span>
          </DetailRow>
        ))}
      </DetailList>
    </div>
  );
}

/** The platform's own account of why a feature stopped, with the evidence it kept. */
function FailureEvidence({ feature }: { feature: Feature }) {
  const failure = feature.failure_summary;
  if (!failure) return null;
  return (
    <section className="action-card" aria-labelledby="failure-evidence-heading">
      <div className="action-card__header">
        <h2 className="action-card__title" id="failure-evidence-heading">
          Why the feature stopped
        </h2>
        <Badge tone="stopped">{failure.root_classification}</Badge>
        {failure.repository_id ? <RepositoryBadge repositoryId={failure.repository_id} /> : null}
      </div>
      <DetailList narrow>
        <DetailRow label="Stage">{failure.stage}</DetailRow>
        <DetailRow label="Agent">{failure.agent}</DetailRow>
        {failure.attempt !== null ? <DetailRow label="Attempt">{failure.attempt}</DetailRow> : null}
        {failure.exit_code !== null ? <DetailRow label="Exit code">{failure.exit_code}</DetailRow> : null}
        <DetailRow label="Recorded">
          <span title={absoluteTime(failure.recorded_at)}>{relativeTime(failure.recorded_at)}</span>
        </DetailRow>
        <DetailRow label="Automatic recovery">{failure.retryable ? 'Possible' : 'Not indicated'}</DetailRow>
      </DetailList>
      {failure.command.length > 0 ? (
        <CodeBlock title="Command" copyValue={failure.command.join(' ')}>
          {failure.command.join(' ')}
        </CodeBlock>
      ) : null}
      {failure.diagnostics.length > 0 ? (
        <ul className="bullets">
          {failure.diagnostics.map((item, index) => (
            <li key={index} className="prose">
              {item}
            </li>
          ))}
        </ul>
      ) : null}
      <p className="prose">
        <strong>What to do:</strong> {failure.next_action}
      </p>
    </section>
  );
}

/** External things a cancellation left behind, which nobody else will notice. */
function CleanupRequirements({ feature }: { feature: Feature }) {
  const items = feature.cleanup_requirements.filter(
    (item): item is Record<string, unknown> => typeof item === 'object' && item !== null,
  );
  if (items.length === 0) return null;
  return (
    <section className="action-card" aria-labelledby="cleanup-heading">
      <h2 className="action-card__title" id="cleanup-heading">
        External resources need checking
      </h2>
      <ul className="cards">
        {items.map((item, index) => (
          <li className="card" key={String(item.external_reference ?? index)}>
            <div className="card__header">
              <strong>{String(item.resource_type ?? 'External operation')}</strong>
              {item.repository_id ? <RepositoryBadge repositoryId={String(item.repository_id)} /> : null}
            </div>
            <p className="prose">{String(item.reason ?? '')}</p>
            <p className="muted">{String(item.recommended_action ?? '')}</p>
            {item.external_reference ? <code className="mono">{String(item.external_reference)}</code> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}
