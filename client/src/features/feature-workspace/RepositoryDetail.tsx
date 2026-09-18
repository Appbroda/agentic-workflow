import { useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { EmptyState, ErrorState, TableSkeleton } from '@/components/common/States';
import { Badge, ChangeBadge, RepositoryBadge, SeverityBadge, StatusBadge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { LinkTabs, PageHeader, Panel, type TabDefinition } from '@/components/ui/Layout';
import { Drawer } from '@/components/ui/Drawer';
import { CodeBlock, LogViewer, RawJson, UnifiedDiff } from '@/components/ui/Code';
import { CopyValue, DetailList, DetailRow, ExternalLink, Revision } from '@/components/ui/Value';
import { count } from '@/utils/count';
import { Markdown } from '@/components/common/Markdown';
import { absoluteTime, formatDuration, relativeTime } from '@/utils/time';
import { humanise } from '@/utils/text';
import type { Artifact, Workstream } from '@/schemas/feature';
import {
  useArtifact,
  useArtifactList,
  useFeature,
  usePullRequests,
  useTimeline,
  useWording,
  useWorkstreams,
} from './hooks';
import { useLiveFeature } from './useLiveFeature';
import { RetryWorkstream } from './RetryWorkstream';
import { RepositoryRepairs } from './RepositoryRepair';
import { TimelineList } from './TimelineList';
import {
  attempts,
  faultEvidence,
  pullRequestView,
  runtimeSentence,
  validationResults,
  type ValidationResult,
} from './workstream-view';
import { repositoryOf } from './requirements';

/**
 * One repository's workstream, in full.
 *
 * This is where an engineer ends up: the exact branch and revision, the files that changed,
 * the exact command each validation check ran and what it printed, the review's findings, and
 * the pull request. Everything on this page is what the platform recorded for this repository
 * -- nothing is reconstructed, and where the record is silent the page says so.
 */

const VIEWS = ['overview', 'changes', 'validation', 'review', 'history', 'pr'] as const;
type View = (typeof VIEWS)[number];

export function RepositoryDetail() {
  const { featureId = '', repositoryId = '' } = useParams<{
    featureId: string;
    repositoryId: string;
  }>();
  const [params] = useSearchParams();
  // The section lives in the URL, so this page is linkable at the depth somebody reached --
  // "the failing validation on backend-api" is a link, not a set of instructions.
  const view: View = VIEWS.includes(params.get('view') as View) ? (params.get('view') as View) : 'overview';

  const feature = useFeature(featureId);
  const { lastEventId } = useLiveFeature(featureId, feature.data?.status);
  const workstreams = useWorkstreams(featureId, lastEventId);
  const wordingFor = useWording('workstream');

  const workstream = workstreams.data?.workstreams.find((item) => item.repository_id === repositoryId);

  if (workstreams.isPending) {
    return (
      <div className="page">
        <TableSkeleton rows={8} label="Loading repository…" />
      </div>
    );
  }
  if (workstreams.isError) {
    return (
      <div className="page">
        <ErrorState error={workstreams.error} onRetry={workstreams.refetch} />
      </div>
    );
  }
  if (!workstream) {
    return (
      <div className="page">
        <EmptyState
          title="No such repository in this feature"
          detail={`This feature has no workstream for “${repositoryId}”.`}
          action={
            <Link className="button" to={`/features/${encodeURIComponent(featureId)}/repositories`}>
              Back to repositories
            </Link>
          }
        />
      </div>
    );
  }

  const base = `/features/${encodeURIComponent(featureId)}/repositories/${encodeURIComponent(repositoryId)}`;
  const validation = validationResults(workstream);

  const tabs: TabDefinition[] = [
    { id: 'overview', label: 'Overview', to: base },
    // No count on Changes: the workstream's path lists and the implementation artifact's file
    // list do not always agree, and the tab counting one while the table shows the other reads
    // as a defect. The table's own header carries the number it is actually showing.
    { id: 'changes', label: 'Changes', to: `${base}?view=changes` },
    { id: 'validation', label: 'Validation', to: `${base}?view=validation`, count: validation.length },
    { id: 'review', label: 'Review', to: `${base}?view=review` },
    { id: 'history', label: 'History', to: `${base}?view=history` },
    { id: 'pr', label: 'Pull request', to: `${base}?view=pr` },
  ];

  const profile = workstream.technology_profile ?? {};
  const stack = [profile.primary_language, ...(Array.isArray(profile.frameworks) ? profile.frameworks : [])]
    .filter((item): item is string => typeof item === 'string')
    .join(' · ');

  return (
    <div className="page stack">
      <PageHeader
        title={workstream.repository_name ?? workstream.repository_id}
        badges={
          <>
            <StatusBadge wording={wordingFor(workstream.status)} />
            {workstream.repository_role ? <Badge outline>{workstream.repository_role}</Badge> : null}
            <RepositoryBadge repositoryId={workstream.repository_id} />
          </>
        }
        subtitle={
          <DetailList narrow>
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
            <DetailRow label="Attempts">{attempts(workstream)}</DetailRow>
            {workstream.preflight_status ? (
              <DetailRow label="Preflight">{workstream.preflight_status}</DetailRow>
            ) : null}
            {workstream.planned_blind ? (
              <DetailRow label="Reconnaissance">Planned without checkout evidence</DetailRow>
            ) : null}
          </DetailList>
        }
        actions={
          <>
            {workstream.repository_url ? (
              <a
                className="button"
                href={workstream.repository_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open repository
              </a>
            ) : null}
            {workstream.available_actions?.includes('RETRY_WORKSTREAM') ? (
              <RetryWorkstream
                featureId={featureId}
                workstream={workstream}
                onDone={() => void workstreams.refetch()}
              />
            ) : null}
          </>
        }
      />

      <RepositoryRepairs featureId={featureId} workstreams={[workstream]} />

      {workstream.blocking_issues.length > 0 ? (
        <section className="action-card">
          <h2 className="action-card__title">Why this repository stopped</h2>
          <ul className="bullets">
            {/* The last entry is usually the operator question the platform ended on; it is
                shown in full rather than truncated, because it is the actionable part. */}
            {workstream.blocking_issues.map((issue, index) => (
              <li key={index} className="prose">
                {issue}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <LinkTabs
        label="Repository sections"
        tabs={tabs}
        isActive={(tab) => tab.id === view}
      />

      {view === 'overview' ? (
        <RepositoryOverview workstream={workstream} featureId={featureId} at={lastEventId} />
      ) : null}
      {view === 'changes' ? (
        <ChangesView featureId={featureId} workstream={workstream} at={lastEventId} />
      ) : null}
      {view === 'validation' ? <ValidationView results={validation} workstream={workstream} /> : null}
      {view === 'review' ? (
        <ReviewView featureId={featureId} workstream={workstream} at={lastEventId} />
      ) : null}
      {view === 'history' ? (
        <RepositoryHistory featureId={featureId} repositoryId={repositoryId} at={lastEventId} />
      ) : null}
      {view === 'pr' ? (
        <PullRequestView featureId={featureId} workstream={workstream} at={lastEventId} />
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ overview */

function RepositoryOverview({
  workstream,
  featureId,
  at,
}: {
  workstream: Workstream;
  featureId: string;
  at: number | null;
}) {
  const profile = workstream.technology_profile ?? {};
  // The attempt's absorbed-fault evidence lives on its result artifact, not on the
  // workstream: artifact metadata rather than child-state columns, so it is read from the
  // newest result this repository recorded.
  const results = useArtifactList(featureId, 'child_workflow_result', at);
  const mine = (results.data?.artifacts ?? []).filter(
    (artifact) => repositoryOf(artifact) === workstream.repository_id,
  );
  const evidence = faultEvidence(mine.at(-1)?.metadata);
  const runtime = evidence ? runtimeSentence(evidence) : null;
  const expectations = new Map(
    workstream.implementation_expectations
      .filter((item) => typeof item.requirement_id === 'string')
      .map((item) => [item.requirement_id as string, item]),
  );

  return (
    <div className="stack">
      <div className="grid-2">
        <Panel title="Execution">
          <DetailList narrow>
            <DetailRow label="Status">{workstream.status}</DetailRow>
            <DetailRow label="Implementation retries">{workstream.implementation_retry_count}</DetailRow>
            <DetailRow label="Validation retries">{workstream.validation_retry_count}</DetailRow>
            <DetailRow label="Setup retries">{workstream.repository_setup_retry_count}</DetailRow>
            <DetailRow label="Granted attempts">{workstream.granted_extra_attempts}</DetailRow>
            {runtime ? <DetailRow label="Runtime">{runtime}</DetailRow> : null}
            {evidence && evidence.faultClasses.length > 0 ? (
              <DetailRow label="Provider faults">{evidence.faultClasses.join(', ')}</DetailRow>
            ) : null}
            {evidence?.truncatedDegradedRetry ? (
              <DetailRow label="Truncation">
                A truncated response was re-asked once at reduced effort
              </DetailRow>
            ) : null}
            {workstream.failure_classification ? (
              <DetailRow label="Failure class">{workstream.failure_classification}</DetailRow>
            ) : null}
            {workstream.test_availability ? (
              <DetailRow label="Tests">{workstream.test_availability}</DetailRow>
            ) : null}
            {typeof workstream.meaningful_change === 'boolean' ? (
              <DetailRow label="Changed source">
                {workstream.meaningful_change ? 'Yes' : 'No'}
              </DetailRow>
            ) : null}
          </DetailList>
          {workstream.meaningful_change_reason ? (
            <p className="muted">{workstream.meaningful_change_reason}</p>
          ) : null}
          <DetailList narrow>
            <DetailRow label="Workspace">
              <CopyValue value={workstream.workspace_path} label="Copy path" />
            </DetailRow>
            <DetailRow label="Child workflow">
              <CopyValue value={workstream.child_workflow_id} label="Copy workflow ID" />
            </DetailRow>
          </DetailList>
        </Panel>

        <Panel title="Detected technology" meta="From the checkout, not declared">
          <DetailList narrow>
            <TechnologyRow label="Language" value={profile.primary_language} />
            <TechnologyRow label="Frameworks" value={profile.frameworks} />
            <TechnologyRow label="Test frameworks" value={profile.test_frameworks} />
            <TechnologyRow label="Linters" value={profile.linters} />
            <TechnologyRow label="Formatters" value={profile.formatters} />
            <TechnologyRow label="Build tools" value={profile.build_tools} />
            <TechnologyRow label="Package managers" value={profile.package_managers} />
          </DetailList>
          <p className="subtle">
            The platform chooses this repository&rsquo;s validation commands from exactly this,
            which is also the explanation for why those commands and not others.
          </p>
        </Panel>
      </div>

      {workstream.scoped_requirements.length > 0 || workstream.out_of_scope_requirements.length > 0 ? (
        <Panel title="Assigned work" meta="What this repository was asked to do">
          {workstream.scoped_requirements.length > 0 ? (
            <ul className="cards">
              {workstream.scoped_requirements.map((item, index) => {
                const id = typeof item.requirement_id === 'string' ? item.requirement_id : `#${index}`;
                const expectation = expectations.get(id);
                const areas = Array.isArray(expectation?.expected_source_areas)
                  ? (expectation.expected_source_areas as unknown[]).filter(
                      (area): area is string => typeof area === 'string',
                    )
                  : [];
                const criteria = Array.isArray(item.acceptance_criterion_ids)
                  ? (item.acceptance_criterion_ids as unknown[]).filter(
                      (criterion): criterion is string => typeof criterion === 'string',
                    )
                  : [];
                const done = workstream.requirements_implemented.includes(id);
                return (
                  <li key={id} className="card">
                    <div className="card__header">
                      <strong className="requirement__id">{id}</strong>
                      {typeof item.responsibility === 'string' ? (
                        <Badge outline>{item.responsibility}</Badge>
                      ) : null}
                      <Badge tone={done ? 'done' : 'neutral'}>
                        {done ? 'Reported implemented' : 'No result reported'}
                      </Badge>
                    </div>
                    {criteria.length > 0 ? (
                      <p className="muted">Acceptance criteria: {criteria.join(', ')}</p>
                    ) : null}
                    {areas.length > 0 ? <p className="muted">Expected to touch: {areas.join(', ')}</p> : null}
                    {expectation?.tests_required === true ? <p className="muted">Tests required</p> : null}
                  </li>
                );
              })}
            </ul>
          ) : null}
          {workstream.out_of_scope_requirements.length > 0 ? (
            <>
              {/* Stated explicitly, because "not implemented here" and "not implemented at
                  all" are different findings and only one of them is a problem. */}
              <p className="muted">Deliberately not this repository&rsquo;s:</p>
              <ul className="bullets">
                {workstream.out_of_scope_requirements.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          ) : null}
        </Panel>
      ) : null}

      {workstream.model_routing ? (
        <Panel title="Model routing" meta="Which model is doing this repository's work, and why">
          {/* Secondary on purpose. A product manager reading this page needs the outcome; an
              engineer asking why a correction was cheap, or why it stopped being cheap, needs
              the tier. Both are here, and neither is in the way of the other. */}
          <dl className="facts">
            <div>
              <dt>Doing</dt>
              <dd>
                {workstream.model_routing.execution_mode === 'REVIEW_REMEDIATION'
                  ? 'Correcting review findings'
                  : 'Implementing the workstream'}
              </dd>
            </div>
            <div>
              <dt>Model role</dt>
              <dd>{formatRoutingRole(workstream.model_routing.role)}</dd>
            </div>
            {workstream.model_routing.classification ? (
              <div>
                <dt>Fix complexity</dt>
                <dd>{workstream.model_routing.classification}</dd>
              </div>
            ) : null}
            <div>
              <dt>Attempt</dt>
              <dd>
                {(workstream.model_routing.attempt ?? workstream.retry_count) + 1} of{' '}
                {workstream.retry_count + 1 + Math.max(0, workstream.granted_extra_attempts)}
              </dd>
            </div>
          </dl>
          {/* The reason, not the model's reasoning: the routing decision is the platform's
              own and is safe to show. Nothing here exposes a model's chain of thought. */}
          {workstream.model_routing.routing_reason ? (
            <p className="prose">{workstream.model_routing.routing_reason}</p>
          ) : null}
        </Panel>
      ) : null}

      {workstream.retry_strategy || workstream.retry_refusal_reason || workstream.retry_grants.length > 0 ? (
        <Panel title="Retry history">
          {/* What the next attempt was told to do differently. Without it, a repository that
              tried nine times is nine identical-looking failures. */}
          {workstream.retry_strategy ? (
            <div className="callout callout--info">
              <p className="callout__title">Strategy for the next attempt</p>
              {typeof workstream.retry_strategy.root_cause === 'string' ? (
                <p className="prose">{workstream.retry_strategy.root_cause}</p>
              ) : null}
              {typeof workstream.retry_strategy.required_strategy_change === 'string' ? (
                <p className="prose">{workstream.retry_strategy.required_strategy_change}</p>
              ) : null}
            </div>
          ) : null}
          {workstream.retry_refusal_reason ? (
            <div className="callout callout--warn">
              <p className="callout__title">Why the platform stopped retrying</p>
              <p className="prose">{workstream.retry_refusal_reason}</p>
            </div>
          ) : null}
          {/* Whether pressing Publish on the feature would open a pull request for this
              repository, or the server's own sentence saying why it would not. */}
          {workstream.pull_request_artifact_id === null && workstream.publication_class ? (
            <div
              className={
                workstream.publication_class === 'unreviewed'
                  ? 'callout callout--attention'
                  : 'callout'
              }
            >
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
              {/* An override of the platform's own stop, shown as an override: without this
                  the only trace is an attempt count past the configured limit. */}
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
        </Panel>
      ) : null}

      <Panel title="Evidence" meta="The artifacts this repository produced">
        <ul className="bullets">
          {workstream.code_completion_artifact_id ? (
            <ArtifactLink
              featureId={featureId}
              artifactId={workstream.code_completion_artifact_id}
              label="Implementation result"
            />
          ) : null}
          {workstream.review_artifact_id ? (
            <ArtifactLink
              featureId={featureId}
              artifactId={workstream.review_artifact_id}
              label="Repository review"
            />
          ) : null}
          {workstream.pull_request_artifact_id ? (
            <ArtifactLink
              featureId={featureId}
              artifactId={workstream.pull_request_artifact_id}
              label="Pull request record"
            />
          ) : null}
          {!workstream.code_completion_artifact_id &&
          !workstream.review_artifact_id &&
          !workstream.pull_request_artifact_id ? (
            <li className="muted">This repository has produced no result artifacts yet.</li>
          ) : null}
        </ul>
      </Panel>
    </div>
  );
}

/**
 * Name a configured model role the way a person reads it. The role is the platform's own
 * vocabulary -- `scoped_fix` -- and the model behind it is deployment configuration that is
 * deliberately not shown here: what a reader needs is which role is doing the work.
 */
function formatRoutingRole(role: string | null | undefined): string {
  switch (role) {
    case 'coding':
      return 'Coding';
    case 'review':
      return 'Review';
    case 'scoped_fix':
      return 'Scoped fix';
    // Historical values remain readable after the active router moved to four roles.
    case 'fix':
      return 'Standard fix';
    case 'complex_fix':
      return 'Complex fix';
    case 'escalation':
      return 'Escalation';
    default:
      return role ?? 'unknown';
  }
}

function TechnologyRow({ label, value }: { label: string; value: unknown }) {
  const text = Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string').join(', ')
    : typeof value === 'string'
      ? value
      : '';
  // Omitted entirely when empty: an absent linter is not "none", it is a repository the
  // platform did not find one in, and an empty row implies it looked and found nothing.
  if (!text) return null;
  return <DetailRow label={label}>{text}</DetailRow>;
}

function ArtifactLink({
  featureId,
  artifactId,
  label,
}: {
  featureId: string;
  artifactId: string;
  label: string;
}) {
  return (
    <li>
      <Link
        to={`/features/${encodeURIComponent(featureId)}/artifacts?artifact=${encodeURIComponent(artifactId)}`}
      >
        {label}
      </Link>{' '}
      <span className="subtle mono">{artifactId}</span>
    </li>
  );
}

/* ------------------------------------------------------------------- changes */

interface FileChange {
  path: string;
  changeType: string;
  category: 'production' | 'test' | 'configuration';
  description: string;
}

/**
 * The files this repository changed.
 *
 * The change type and description come from the implementation artifact where there is one;
 * the workstream's own path lists are the fallback, and a path known only from those is shown
 * as "changed" rather than being assigned a type nobody recorded.
 */
function ChangesView({
  featureId,
  workstream,
  at,
}: {
  featureId: string;
  workstream: Workstream;
  at: number | null;
}) {
  const completions = useArtifactList(featureId, 'code_completion', at);
  const mine = (completions.data?.artifacts ?? []).filter(
    (artifact) => repositoryOf(artifact) === workstream.repository_id,
  );
  const latest = mine.at(-1) ?? null;
  const artifact = useArtifact(featureId, latest?.artifact_id ?? null);
  const [selected, setSelected] = useState<FileChange | null>(null);

  const rows = useMemo(
    () => fileChanges(workstream, artifact.data),
    [workstream, artifact.data],
  );

  const columns: Column<FileChange>[] = [
    {
      key: 'change',
      header: 'Change',
      shrink: true,
      sortValue: (row) => row.changeType,
      render: (row) => <ChangeBadge changeType={row.changeType} />,
    },
    {
      key: 'path',
      header: 'File',
      sortValue: (row) => row.path,
      render: (row) => (
        <button
          type="button"
          className="link-button"
          style={{ padding: 0, background: 'none' }}
          onClick={() => setSelected(row)}
        >
          <span className="mono truncate" title={row.path}>
            {row.path}
          </span>
        </button>
      ),
    },
    {
      key: 'category',
      header: 'Kind',
      shrink: true,
      sortValue: (row) => row.category,
      render: (row) => <Badge outline>{row.category}</Badge>,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (row) => <span className="muted truncate" title={row.description}>{row.description}</span>,
    },
  ];

  const commitSha = typeof artifact.data?.payload.commit_sha === 'string' ? artifact.data.payload.commit_sha : null;
  const summary = typeof artifact.data?.payload.summary === 'string' ? artifact.data.payload.summary : null;

  return (
    <div className="stack">
      {summary ? (
        <Panel title="What this attempt did">
          <div className="document__body">
            <Markdown>{summary}</Markdown>
          </div>
          {commitSha ? (
            <DetailList narrow>
              <DetailRow label="Commit">
                <Revision value={commitSha} />
              </DetailRow>
            </DetailList>
          ) : null}
        </Panel>
      ) : null}

      <Panel title="Changed files" meta={count(rows.length, 'file')} flush>
        <DataTable
          label="Changed files"
          columns={columns}
          rows={rows}
          rowKey={(row) => row.path}
          compact
          empty={
            <EmptyState
              title="No file changes recorded"
              detail="This repository has not reported an implementation result yet."
            />
          }
        />
      </Panel>

      {selected ? (
        <Drawer
          title={selected.path}
          subtitle={`${selected.category} · ${selected.changeType}`}
          onDismiss={() => setSelected(null)}
        >
          <FileDetail change={selected} artifact={artifact.data} />
        </Drawer>
      ) : null}
    </div>
  );
}

function fileChanges(workstream: Workstream, artifact: Artifact | undefined): FileChange[] {
  const categoryOf = (path: string): FileChange['category'] =>
    workstream.test_files_changed.includes(path)
      ? 'test'
      : workstream.configuration_files_changed.includes(path)
        ? 'configuration'
        : 'production';

  const byPath = new Map<string, FileChange>();
  for (const path of [
    ...workstream.production_files_changed,
    ...workstream.test_files_changed,
    ...workstream.configuration_files_changed,
  ]) {
    byPath.set(path, { path, changeType: 'changed', category: categoryOf(path), description: '' });
  }

  const recorded = Array.isArray(artifact?.payload.file_changes) ? artifact.payload.file_changes : [];
  for (const item of recorded) {
    if (typeof item !== 'object' || item === null) continue;
    const record = item as Record<string, unknown>;
    if (typeof record.path !== 'string') continue;
    byPath.set(record.path, {
      path: record.path,
      changeType: typeof record.change_type === 'string' ? record.change_type : 'changed',
      category: categoryOf(record.path),
      description: typeof record.description === 'string' ? record.description : '',
    });
  }

  return [...byPath.values()].sort((left, right) => left.path.localeCompare(right.path));
}

/**
 * What is known about one changed file.
 *
 * The platform records which files changed and how, not their contents, so there is usually no
 * diff to show. Where a payload does carry one, it is rendered as a diff rather than as a wall
 * of text; where it does not, this says so instead of leaving an empty panel that reads like a
 * loading failure.
 */
function FileDetail({ change, artifact }: { change: FileChange; artifact: Artifact | undefined }) {
  const patch = patchFor(change.path, artifact);
  const evidence = Array.isArray(artifact?.payload.requirement_implementation_evidence)
    ? artifact.payload.requirement_implementation_evidence
    : [];
  const mentions = evidence.filter(
    (item): item is Record<string, unknown> =>
      typeof item === 'object' &&
      item !== null &&
      Array.isArray((item as Record<string, unknown>).files) &&
      ((item as Record<string, unknown>).files as unknown[]).includes(change.path),
  );

  return (
    <div className="stack">
      <DetailList narrow>
        <DetailRow label="Path">
          <CopyValue value={change.path} label="Copy path" />
        </DetailRow>
        <DetailRow label="Change">{change.changeType}</DetailRow>
        <DetailRow label="Kind">{change.category}</DetailRow>
      </DetailList>
      {change.description ? <p className="prose">{change.description}</p> : null}

      {mentions.length > 0 ? (
        <div className="stack stack--tight">
          <span className="details__label">Cited as evidence for</span>
          <ul className="bullets">
            {mentions.map((item, index) => (
              <li key={index}>
                <span className="requirement__id">{String(item.requirement_id ?? '')}</span>
                {Array.isArray(item.symbols) && item.symbols.length > 0 ? (
                  <span className="muted"> — {(item.symbols as string[]).join(', ')}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {patch ? (
        <UnifiedDiff patch={patch} />
      ) : (
        <p className="muted">
          The platform records which files an attempt changed and how, not their contents, so
          there is no diff to show here. The pull request has the code.
        </p>
      )}
    </div>
  );
}

/** A diff for one path, if the implementation artifact happens to carry one. */
function patchFor(path: string, artifact: Artifact | undefined): string | null {
  const changes = Array.isArray(artifact?.payload.file_changes) ? artifact.payload.file_changes : [];
  for (const item of changes) {
    if (typeof item !== 'object' || item === null) continue;
    const record = item as Record<string, unknown>;
    if (record.path !== path) continue;
    for (const key of ['diff', 'patch', 'unified_diff']) {
      if (typeof record[key] === 'string' && record[key].trim().length > 0) return record[key] as string;
    }
  }
  return null;
}

/* ---------------------------------------------------------------- validation */

function ValidationView({
  results,
  workstream,
}: {
  results: ValidationResult[];
  workstream: Workstream;
}) {
  const [selected, setSelected] = useState<ValidationResult | null>(null);

  const columns: Column<ValidationResult>[] = [
    {
      key: 'check',
      header: 'Check',
      shrink: true,
      sortValue: (row) => row.validationType ?? row.name,
      render: (row) => (
        <span className="table__primary">{humanise(row.validationType ?? row.name)}</span>
      ),
    },
    {
      key: 'command',
      header: 'Command',
      sortValue: (row) => row.command,
      render: (row) => <CopyValue value={row.command} label="Copy command" />,
    },
    {
      key: 'status',
      header: 'Status',
      shrink: true,
      sortValue: (row) => (row.passed ? 1 : 0),
      render: (row) => <Badge tone={row.passed ? 'done' : 'stopped'}>{row.status}</Badge>,
    },
    {
      key: 'duration',
      header: 'Duration',
      shrink: true,
      align: 'right',
      sortValue: (row) => row.durationSeconds,
      render: (row) =>
        row.durationSeconds === null ? (
          <span className="subtle">—</span>
        ) : (
          <span>{row.durationSeconds.toFixed(1)}s</span>
        ),
    },
    {
      key: 'exit',
      header: 'Exit code',
      shrink: true,
      align: 'right',
      sortValue: (row) => row.exitCode,
      render: (row) => (row.exitCode === null ? <span className="subtle">—</span> : row.exitCode),
    },
    {
      key: 'revision',
      header: 'Revision',
      shrink: true,
      render: (row) => (row.revision ? <Revision value={row.revision} /> : <span className="subtle">—</span>),
    },
  ];

  const configured = workstream.configured_validation_commands;

  return (
    <div className="stack">
      <Panel
        title="Validation"
        meta={
          results.length > 0
            ? `${results.filter((item) => item.passed).length} of ${results.length} passed`
            : undefined
        }
        flush
      >
        <DataTable
          label="Validation checks"
          columns={columns}
          rows={results}
          rowKey={(row) => `${row.validationType ?? row.name}-${row.command}`}
          onRowClick={(row) => setSelected(row)}
          compact
          empty={
            <EmptyState
              title="No validation results"
              detail="This repository has not run its checks on the current revision."
            />
          }
        />
      </Panel>

      {configured.length > 0 ? (
        <Panel title="Configured checks" meta="Chosen from the detected technology">
          <ul className="bullets">
            {configured.map((item, index) => (
              <li key={index}>
                <code className="mono">
                  {Array.isArray(item.command)
                    ? (item.command as string[]).join(' ')
                    : String(item.command ?? '')}
                </code>
                <span className="muted">
                  {' '}
                  — {String(item.validation_type ?? 'check')}
                  {item.required === false ? ' (optional)' : ''}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      ) : null}

      {selected ? (
        <Drawer
          title={humanise(selected.validationType ?? selected.name)}
          subtitle={selected.command}
          onDismiss={() => setSelected(null)}
        >
          <DetailList narrow>
            <DetailRow label="Status">
              <Badge tone={selected.passed ? 'done' : 'stopped'}>{selected.status}</Badge>
            </DetailRow>
            <DetailRow label="Exit code">{selected.exitCode ?? '—'}</DetailRow>
            <DetailRow label="Duration">
              {selected.durationSeconds === null
                ? '—'
                : (formatDuration(selected.durationSeconds * 1000) ?? `${selected.durationSeconds.toFixed(1)}s`)}
            </DetailRow>
            <DetailRow label="Required">{selected.required ? 'Yes' : 'No'}</DetailRow>
            <DetailRow label="Current revision">{selected.isCurrent ? 'Yes' : 'Superseded'}</DetailRow>
            {selected.workingDirectory ? (
              <DetailRow label="Working directory">
                <CopyValue value={selected.workingDirectory} label="Copy directory" />
              </DetailRow>
            ) : null}
            {selected.revision ? (
              <DetailRow label="Revision">
                <Revision value={selected.revision} />
              </DetailRow>
            ) : null}
            {selected.resultCode ? <DetailRow label="Result code">{selected.resultCode}</DetailRow> : null}
            {selected.failureClassification ? (
              <DetailRow label="Failure class">{selected.failureClassification}</DetailRow>
            ) : null}
          </DetailList>

          <CodeBlock title="Command" copyValue={selected.command}>
            {selected.command}
          </CodeBlock>

          {/* Summaries, not full output: the platform stores bounded, redacted summaries so
              tool output never lands in durable state. */}
          <LogViewer label="Standard output" text={selected.stdout} />
          <LogViewer label="Standard error" text={selected.stderr} />

          <details>
            <summary className="muted">Everything the platform recorded</summary>
            <div style={{ marginTop: 'var(--space-3)' }}>
              <RawJson value={selected.raw} />
            </div>
          </details>
        </Drawer>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------- review */

interface Finding {
  id: string;
  severity: string;
  title: string;
  description: string;
  recommendation: string;
}

function ReviewView({
  featureId,
  workstream,
  at,
}: {
  featureId: string;
  workstream: Workstream;
  at: number | null;
}) {
  const reviews = useArtifactList(featureId, 'review', at);
  const mine = (reviews.data?.artifacts ?? []).filter(
    (artifact) => repositoryOf(artifact) === workstream.repository_id,
  );
  const latest = mine.at(-1) ?? null;
  const artifact = useArtifact(featureId, latest?.artifact_id ?? null);
  const [selected, setSelected] = useState<Finding | null>(null);
  const [openOnly, setOpenOnly] = useState(false);

  if (reviews.isPending || artifact.isPending) return <TableSkeleton rows={5} />;
  if (!latest || !artifact.data) {
    return (
      <Panel title="Review">
        <EmptyState
          title="No review yet"
          detail="A review is written once the repository reports an implementation it could validate."
        />
      </Panel>
    );
  }

  const payload = artifact.data.payload;
  const verdict = typeof payload.verdict === 'string' ? payload.verdict : 'unknown';
  const findings: Finding[] = Array.isArray(payload.findings)
    ? payload.findings
        .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
        .map((item, index) => ({
          id: typeof item.finding_id === 'string' ? item.finding_id : `finding-${index}`,
          severity: typeof item.severity === 'string' ? item.severity : 'unspecified',
          title:
            typeof item.title === 'string'
              ? item.title
              : typeof item.finding_id === 'string'
                ? item.finding_id
                : `Finding ${index + 1}`,
          description: typeof item.description === 'string' ? item.description : '',
          recommendation: typeof item.recommendation === 'string' ? item.recommendation : '',
        }))
    : [];

  const checks = Array.isArray(payload.requirement_checks)
    ? payload.requirement_checks.filter(
        (item): item is Record<string, unknown> => typeof item === 'object' && item !== null,
      )
    : [];
  const shownFindings = openOnly ? findings.filter((item) => severityRank(item.severity) >= 2) : findings;

  const findingColumns: Column<Finding>[] = [
    {
      key: 'severity',
      header: 'Severity',
      shrink: true,
      sortValue: (row) => -severityRank(row.severity),
      render: (row) => <SeverityBadge severity={row.severity} />,
    },
    {
      key: 'title',
      header: 'Finding',
      sortValue: (row) => row.title,
      render: (row) => (
        <button
          type="button"
          className="link-button"
          style={{ padding: 0, background: 'none' }}
          onClick={() => setSelected(row)}
        >
          <span className="table__primary truncate">{row.title}</span>
        </button>
      ),
    },
    {
      key: 'description',
      header: 'Detail',
      render: (row) => (
        <span className="muted truncate" title={row.description}>
          {row.description}
        </span>
      ),
    },
    {
      key: 'repository',
      header: 'Repository',
      shrink: true,
      render: () => <RepositoryBadge repositoryId={workstream.repository_id} />,
    },
  ];

  const checkColumns: Column<Record<string, unknown>>[] = [
    {
      key: 'requirement',
      header: 'Requirement',
      shrink: true,
      sortValue: (row) => String(row.requirement_id ?? ''),
      render: (row) => <span className="requirement__id">{String(row.requirement_id ?? '')}</span>,
    },
    {
      key: 'status',
      header: 'Status',
      shrink: true,
      sortValue: (row) => (row.passed === true ? 1 : 0),
      render: (row) => (
        <Badge tone={row.passed === true ? 'done' : 'stopped'}>
          {row.passed === true ? 'Satisfied' : 'Not met'}
        </Badge>
      ),
    },
    {
      key: 'evidence',
      header: 'Evidence',
      render: (row) => (
        // Reviewer evidence runs to a paragraph. Held to a width and truncated so the table
        // stays a table; the expander below has the whole thing.
        <div style={{ maxWidth: '46rem' }}>
          <span className="muted truncate" title={String(row.evidence ?? '')}>
            {String(row.evidence ?? '')}
          </span>
        </div>
      ),
    },
  ];

  const satisfied = checks.filter((item) => item.passed === true).length;

  return (
    <div className="stack">
      <Panel
        title="Review"
        actions={
          <Link
            className="button button--small"
            to={`/features/${encodeURIComponent(featureId)}/artifacts?artifact=${encodeURIComponent(latest.artifact_id)}`}
          >
            Open artifact
          </Link>
        }
      >
        <div className="row">
          <Badge tone={verdict === 'approved' ? 'done' : 'stopped'}>{verdict}</Badge>
          <span className="muted">
            Reviewed {relativeTime(artifact.data.timestamp)} · {mine.length}{' '}
            {mine.length === 1 ? 'review' : 'reviews'} written for this repository
          </span>
        </div>
        {typeof payload.summary === 'string' ? (
          <div className="document__body">
            <Markdown>{payload.summary}</Markdown>
          </div>
        ) : null}
      </Panel>

      {checks.length > 0 ? (
        <Panel title="Acceptance criteria" meta={`${satisfied} of ${checks.length} satisfied`} flush>
          <DataTable
            label="Requirement checks"
            columns={checkColumns}
            rows={checks}
            rowKey={(row) => String(row.requirement_id ?? Math.random())}
            compact
            expandLabel="Show evidence"
            expand={(row) => <p className="prose">{String(row.evidence ?? 'No evidence recorded.')}</p>}
          />
        </Panel>
      ) : null}

      <Panel
        title="Findings"
        meta={count(findings.length, 'finding')}
        actions={
          findings.length > 0 ? (
            <button
              type="button"
              className="button button--small"
              aria-pressed={openOnly}
              onClick={() => setOpenOnly((value) => !value)}
            >
              {openOnly ? 'All severities' : 'High and above'}
            </button>
          ) : undefined
        }
        flush
      >
        <DataTable
          label="Review findings"
          columns={findingColumns}
          rows={shownFindings}
          rowKey={(row) => row.id}
          compact
          empty={
            <EmptyState
              title={findings.length === 0 ? 'No findings' : 'Nothing at that severity'}
              detail={
                findings.length === 0
                  ? 'The reviewer raised nothing against this repository.'
                  : 'Show all severities to see the rest.'
              }
            />
          }
        />
      </Panel>

      {['architecture_assessment', 'security_assessment', 'test_coverage_assessment'].some(
        (key) => typeof payload[key] === 'string',
      ) ? (
        <Panel title="Assessments">
          {(
            [
              ['architecture_assessment', 'Architecture'],
              ['security_assessment', 'Security'],
              ['test_coverage_assessment', 'Test coverage'],
            ] as const
          ).map(([key, label]) =>
            typeof payload[key] === 'string' && payload[key] ? (
              <section className="document__section" key={key}>
                <h3>{label}</h3>
                <div className="document__body">
                  <Markdown>{payload[key] as string}</Markdown>
                </div>
              </section>
            ) : null,
          )}
        </Panel>
      ) : null}

      {selected ? (
        <Drawer title={selected.title} subtitle={selected.severity.toUpperCase()} onDismiss={() => setSelected(null)}>
          <DetailList narrow>
            <DetailRow label="Finding">{selected.id}</DetailRow>
            <DetailRow label="Severity">
              <SeverityBadge severity={selected.severity} />
            </DetailRow>
            <DetailRow label="Repository">
              <RepositoryBadge repositoryId={workstream.repository_id} />
            </DetailRow>
          </DetailList>
          <p className="prose">{selected.description}</p>
          {selected.recommendation ? (
            <div className="callout callout--info">
              <p className="callout__title">Recommended fix</p>
              <p className="prose">{selected.recommendation}</p>
            </div>
          ) : null}
        </Drawer>
      ) : null}
    </div>
  );
}

function severityRank(severity: string): number {
  const value = severity.trim().toLowerCase();
  if (value === 'critical' || value === 'blocker') return 3;
  if (value === 'high') return 2;
  if (value === 'medium' || value === 'moderate') return 1;
  return 0;
}

/* ------------------------------------------------------------------- history */

function RepositoryHistory({
  featureId,
  repositoryId,
  at,
}: {
  featureId: string;
  repositoryId: string;
  at: number | null;
}) {
  const timeline = useTimeline(featureId, at);
  const events = (timeline.data?.events ?? []).filter(
    (event) => event.details.repository_id === repositoryId,
  );

  return (
    <Panel title="History" meta={`Events for ${repositoryId}`} flush>
      {timeline.isPending ? (
        <TableSkeleton rows={5} />
      ) : events.length === 0 ? (
        <EmptyState
          title="No events for this repository"
          detail="Feature-wide events are on the feature's own History tab."
        />
      ) : (
        <TimelineList featureId={featureId} events={[...events].reverse()} />
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------------ pr */

function PullRequestView({
  featureId,
  workstream,
  at,
}: {
  featureId: string;
  workstream: Workstream;
  at: number | null;
}) {
  const pullRequests = usePullRequests(featureId, at);
  const artifact = (pullRequests.data?.pull_requests ?? []).find(
    (item) =>
      item.artifact_id === workstream.pull_request_artifact_id ||
      repositoryOf(item) === workstream.repository_id,
  );

  if (pullRequests.isPending) return <TableSkeleton rows={4} />;
  if (!artifact) {
    return (
      <Panel title="Pull request">
        <EmptyState
          title="No pull request yet"
          detail="One is opened once this repository passes its own review."
        />
      </Panel>
    );
  }

  const view = pullRequestView(artifact, [workstream]);
  const body = typeof artifact.payload.body === 'string' ? artifact.payload.body : null;

  return (
    <Panel
      title="Pull request"
      actions={
        view.url ? (
          <a className="button button--small" href={view.url} target="_blank" rel="noopener noreferrer">
            Open on GitHub
          </a>
        ) : undefined
      }
    >
      <div className="row">
        {view.number ? <Badge tone="working">#{view.number}</Badge> : null}
        {view.state ? <Badge outline>{view.state}</Badge> : null}
        {view.draft ? <Badge outline>draft</Badge> : null}
      </div>
      {view.title ? <p className="prose">{view.title}</p> : null}
      <DetailList narrow>
        <DetailRow label="Repository">{view.repository ?? workstream.repository_id}</DetailRow>
        {view.sourceBranch ? (
          <DetailRow label="Branch">
            <CopyValue value={view.sourceBranch} label="Copy branch" />
          </DetailRow>
        ) : null}
        {view.targetBranch ? <DetailRow label="Target">{view.targetBranch}</DetailRow> : null}
        {view.commitSha ? (
          <DetailRow label="Commit">
            <Revision value={view.commitSha} />
          </DetailRow>
        ) : null}
        {view.reviewers.length > 0 ? (
          <DetailRow label="Reviewers">{view.reviewers.join(', ')}</DetailRow>
        ) : null}
        {view.labels.length > 0 ? <DetailRow label="Labels">{view.labels.join(', ')}</DetailRow> : null}
        <DetailRow label="Recorded">
          <span title={absoluteTime(artifact.timestamp)}>{relativeTime(artifact.timestamp)}</span>
        </DetailRow>
      </DetailList>
      {view.url ? (
        <p className="muted">
          <ExternalLink href={view.url}>{view.url}</ExternalLink>
        </p>
      ) : null}
      {body ? (
        <section className="document__section">
          <h3>Description</h3>
          <div className="document__body">
            <Markdown>{body}</Markdown>
          </div>
        </section>
      ) : null}
    </Panel>
  );
}
