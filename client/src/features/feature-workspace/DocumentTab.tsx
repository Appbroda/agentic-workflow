import { useState, type ReactNode } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Async, EmptyState, TableSkeleton } from '@/components/common/States';
import { Panel, Segmented } from '@/components/ui/Layout';
import { Badge, RepositoryBadge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { RawJson } from '@/components/ui/Code';
import { DetailList, DetailRow } from '@/components/ui/Value';
import { count } from '@/utils/count';
import { Markdown } from '@/components/common/Markdown';
import { ArtifactViewer } from '@/components/artifacts/ArtifactViewer';
import { ARTIFACT_RENDERERS } from '@/components/artifacts/renderers';
import { recordedAttachments } from '@/components/artifacts/attachment-records';
import { AttachmentStrip, ProseWithImageChips } from '@/components/artifacts/attachments';
import { absoluteTime } from '@/utils/time';
import type { Artifact, Workstream } from '@/schemas/feature';
import { useArtifact, useArtifactList, useClarification, useWorkstreams } from './hooks';
import { ClarificationHistory } from './ClarificationHistory';
import { requirementProgress, type RequirementState } from './requirements';

/**
 * The documents a feature is built from, read as documents.
 *
 * A PRD is prose and a list of requirements; it was being rendered as an artifact envelope
 * with a JSON payload underneath. Here it reads as what it is, with the raw record one control
 * away for whoever needs to check exactly what was stored.
 */

/**
 * What each of the three readings says when there is nothing to show.
 *
 * The design's wording is the load-bearing one. "No design was cited" is a statement about the
 * request; anything vaguer would leave a reader unable to tell it apart from a design the
 * platform failed to resolve, which is the ambiguity this whole item exists to remove.
 */
const ABSENT: Record<'original' | 'technical' | 'design', { title: string; detail: string }> = {
  original: {
    title: 'No PRD recorded',
    detail: 'The submitted requirements are stored when a feature starts.',
  },
  technical: {
    title: 'No technical PRD yet',
    detail: 'The product manager agent writes this before planning begins.',
  },
  design: {
    title: 'No design was cited',
    detail:
      'Nobody attached a Figma frame to this request, so every role that built or judged it '
      + 'worked from the prose above. A design is attached when the feature is submitted.',
  },
};

/** The newest artifact of one type, with its payload. */
function useLatest(featureId: string, artifactType: string, at?: number | null) {
  const list = useArtifactList(featureId, artifactType, at);
  const latest = list.data?.artifacts.at(-1) ?? null;
  const artifact = useArtifact(featureId, latest?.artifact_id ?? null);
  return { list, artifact, revisions: list.data?.artifacts.length ?? 0 };
}

/**
 * The feature's requirements: what was asked for, what the platform understood, and what it
 * was meant to look like.
 *
 * Three readings of one request, not one. The submitted PRD is what a person wrote; the
 * technical PRD is the planner's reading of it; the design is what they were looking at while
 * they wrote it. The differences between them are where a feature goes wrong -- showing only
 * the interpretation hides the misreading, showing only the original hides what was acted on,
 * and showing neither design hides that a screen was built against prose.
 *
 * The design option renders empty with "no design was cited" rather than disappearing, so its
 * absence is legible: a feature nobody attached a mock to and a feature whose mock the platform
 * ignored would otherwise look identical.
 */
export function PrdTab({ featureId, at }: { featureId: string; at?: number | null }) {
  const [params, setParams] = useSearchParams();
  const requested = params.get('doc');
  const view =
    requested === 'technical' ? 'technical' : requested === 'design' ? 'design' : 'original';
  const [raw, setRaw] = useState(false);
  const clarification = useClarification(featureId, at ?? null);
  const workstreams = useWorkstreams(featureId, at ?? null);

  const original = useLatest(featureId, 'prd', at);
  const technical = useLatest(featureId, 'technical_prd', at);
  const design = useLatest(featureId, 'design_snapshot', at);
  const current = view === 'technical' ? technical : view === 'design' ? design : original;

  return (
    <div className="stack">
      <Panel
        title="Product requirements"
        actions={
          <>
            <Segmented
              label="Document"
              value={view}
              options={[
                { value: 'original', label: 'Original PRD' },
                { value: 'technical', label: 'Technical PRD' },
                { value: 'design', label: 'Design' },
              ]}
              onChange={(value) => {
                const next = new URLSearchParams(params);
                if (value === 'original') next.delete('doc');
                else next.set('doc', value);
                setParams(next, { replace: true });
              }}
            />
            <button
              type="button"
              className="button button--small"
              aria-pressed={raw}
              onClick={() => setRaw((value) => !value)}
            >
              {raw ? 'Formatted' : 'Raw'}
            </button>
          </>
        }
      >
        <Async query={current.list} skeleton={<TableSkeleton rows={5} />}>
          {(list) =>
            list.artifacts.length === 0 ? (
              <EmptyState
                title={ABSENT[view].title}
                detail={ABSENT[view].detail}
              />
            ) : (
              <Async query={current.artifact} skeleton={<TableSkeleton rows={5} />}>
                {(artifact) => (
                  <div className="stack">
                    <DocumentMeta artifact={artifact} revisions={current.revisions} />
                    {raw ? (
                      <RawJson value={artifact.payload} />
                    ) : view === 'design' ? (
                      // Through the ordinary viewer, so the design reads the way every other
                      // artifact does and the raw envelope stays one control away.
                      <ArtifactViewer
                        artifact={artifact}
                        renderers={ARTIFACT_RENDERERS}
                        featureId={featureId}
                      />
                    ) : view === 'technical' ? (
                      <TechnicalPrdBody
                        payload={artifact.payload}
                        workstreams={workstreams.data?.workstreams ?? []}
                      />
                    ) : (
                      <PrdBody
                        payload={artifact.payload}
                        workstreams={workstreams.data?.workstreams ?? []}
                      />
                    )}
                  </div>
                )}
              </Async>
            )
          }
        </Async>
      </Panel>

      {/* Kept beside the requirements rather than only on the answering form: once a feature
          has moved on, these answers are still what it was planned from, and the first thing
          to re-read when the result is not what was wanted. */}
      {clarification.data && Object.keys(clarification.data.previous_answers).length > 0 ? (
        <Panel title="Clarifications">
          <ClarificationHistory clarification={clarification.data} />
        </Panel>
      ) : null}
    </div>
  );
}

/**
 * The plan, as its four documents.
 *
 * The platform writes planning as separate immutable artifacts -- architecture, task plan,
 * repository execution plan, and the contract the repositories agree on. They are presented
 * side by side rather than concatenated, because each answers a different question and a
 * reader is usually after one of them.
 */
export function PlanTab({ featureId, at }: { featureId: string; at?: number | null }) {
  const [params, setParams] = useSearchParams();
  const section = params.get('section') ?? 'architecture';

  const DOCUMENTS: { value: string; label: string; type: string; absent: string }[] = [
    {
      value: 'architecture',
      label: 'Architecture',
      type: 'architecture',
      absent: 'The architect writes this once the technical requirements are ready.',
    },
    {
      value: 'contract',
      label: 'Integration contract',
      type: 'integration_contract',
      absent: 'Written before any repository starts, so they agree on the seam between them.',
    },
    {
      value: 'execution',
      label: 'Execution plan',
      type: 'repository_execution_plan',
      absent: 'The planner writes repository ownership, dependencies, order and rollout here.',
    },
    {
      value: 'tasks',
      label: 'Task plan',
      type: 'task_plan',
      absent: 'The planner breaks the architecture into validated tasks before execution.',
    },
  ];

  const active = DOCUMENTS.find((item) => item.value === section) ?? DOCUMENTS[0]!;

  return (
    <Panel
      title="Plan"
      actions={
        <Segmented
          label="Plan section"
          value={active.value}
          options={DOCUMENTS.map((item) => ({ value: item.value, label: item.label }))}
          onChange={(value) => {
            const next = new URLSearchParams(params);
            if (value === 'architecture') next.delete('section');
            else next.set('section', value);
            setParams(next, { replace: true });
          }}
        />
      }
    >
      <DocumentTab
        key={active.type}
        featureId={featureId}
        artifactType={active.type}
        title={`No ${active.label.toLowerCase()} yet`}
        absent={active.absent}
        at={at}
      />
    </Panel>
  );
}

/**
 * One artifact type, opened directly.
 *
 * Each is the latest artifact of its type rendered by the ordinary viewer, so the formatted and
 * raw views, the metadata and the unknown-type fallback are the same ones as everywhere else
 * rather than a second implementation that drifts.
 */
export function DocumentTab({
  featureId,
  artifactType,
  title,
  absent,
  at,
}: {
  featureId: string;
  artifactType: string;
  title: string;
  absent: string;
  at?: number | null;
}) {
  const { list, artifact, revisions } = useLatest(featureId, artifactType, at);

  return (
    <Async query={list} empty={<EmptyState title={title} detail={absent} />} skeleton={<TableSkeleton rows={4} />}>
      {(data) =>
        data.artifacts.length === 0 ? (
          <EmptyState title={title} detail={absent} />
        ) : (
          <Async query={artifact} skeleton={<TableSkeleton rows={4} />}>
            {(one) => (
              <>
                {revisions > 1 ? (
                  <p className="muted">
                    Revision {revisions} of {revisions}. Earlier revisions are in the Artifacts tab.
                  </p>
                ) : null}
                <ArtifactViewer artifact={one} renderers={ARTIFACT_RENDERERS} featureId={featureId} />
              </>
            )}
          </Async>
        )
      }
    </Async>
  );
}

function DocumentMeta({ artifact, revisions }: { artifact: Artifact; revisions: number }) {
  return (
    <DetailList narrow>
      <DetailRow label="Written by">{artifact.producer.replace(/_/g, ' ')}</DetailRow>
      <DetailRow label="Recorded">{absoluteTime(artifact.timestamp)}</DetailRow>
      <DetailRow label="Schema">{artifact.schema_version}</DetailRow>
      {revisions > 1 ? (
        <DetailRow label="Revision">
          {revisions} of {revisions}
        </DetailRow>
      ) : null}
    </DetailList>
  );
}

/* ------------------------------------------------------------------ bodies */

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function list(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function objects(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    : [];
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="document__section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Prose({
  title,
  value,
  imageMarkers,
}: {
  title: string;
  value: unknown;
  /** When given, `[image:marker]` in this prose becomes a chip that finds its thumbnail. */
  imageMarkers?: Set<string>;
}) {
  const content = text(value);
  if (!content) return null;
  return (
    <Section title={title}>
      {/* Agent and operator prose is markdown in practice. `Markdown` parses to React elements
          and never produces HTML, so this is structured without becoming an injection surface. */}
      <div className="document__body">
        {imageMarkers && imageMarkers.size > 0 ? (
          <ProseWithImageChips value={content} markers={imageMarkers} />
        ) : (
          <Markdown>{content}</Markdown>
        )}
      </div>
    </Section>
  );
}

function Bullets({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <Section title={title}>
      <ul className="bullets">
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </Section>
  );
}

function PrdBody({
  payload,
  workstreams,
}: {
  payload: Record<string, unknown>;
  workstreams: Workstream[];
}) {
  const stories = objects(payload.user_stories);
  const attachments = recordedAttachments(payload);
  const markers = new Set(attachments.map((item) => item.marker));
  return (
    <div className="document">
      <Prose title="Problem" value={payload.problem_statement} imageMarkers={markers} />
      {/* Before the derived sections and after the problem it illustrates: the strip is
          evidence for the prose above it, and the chips in that prose scroll down to here. */}
      {attachments.length > 0 ? (
        <Section title="Screens and mock-ups">
          <AttachmentStrip attachments={attachments} />
        </Section>
      ) : null}
      <Bullets title="Goals" items={list(payload.goals)} />
      {stories.length > 0 ? (
        <Section title="User stories">
          <ul className="cards">
            {stories.map((story, index) => (
              <li key={text(story.story_id) ?? index} className="card">
                <div className="card__header">
                  <strong>{text(story.story_id) ?? `Story ${index + 1}`}</strong>
                  {text(story.persona) ? <Badge outline>{text(story.persona)}</Badge> : null}
                </div>
                <p className="prose">
                  {text(story.need) ? `Needs to ${text(story.need)}` : ''}
                  {text(story.benefit) ? ` so that ${text(story.benefit)}` : ''}
                </p>
                {list(story.acceptance_criteria).length > 0 ? (
                  <ul className="bullets">
                    {list(story.acceptance_criteria).map((item, position) => (
                      <li key={position}>{item}</li>
                    ))}
                  </ul>
                ) : null}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
      <RequirementTable
        title="Requirements"
        items={objects(payload.requirements)}
        workstreams={workstreams}
      />
      <Bullets title="Constraints" items={list(payload.constraints)} />
      <Bullets title="Out of scope" items={list(payload.out_of_scope)} />
      <Bullets title="Stakeholders" items={list(payload.stakeholders)} />
    </div>
  );
}

function TechnicalPrdBody({
  payload,
  workstreams,
}: {
  payload: Record<string, unknown>;
  workstreams: Workstream[];
}) {
  const questions = objects(payload.unresolved_questions);
  return (
    <div className="document">
      <Prose title="Solution summary" value={payload.solution_summary} />
      <RequirementTable
        title="Functional requirements"
        items={objects(payload.functional_requirements)}
        workstreams={workstreams}
      />
      <RequirementTable
        title="Non-functional requirements"
        items={objects(payload.non_functional_requirements)}
        workstreams={workstreams}
      />
      <Bullets title="Data requirements" items={list(payload.data_requirements)} />
      <Bullets title="Integration requirements" items={list(payload.integration_requirements)} />
      <Bullets title="Security requirements" items={list(payload.security_requirements)} />
      <Bullets title="Assumptions" items={list(payload.assumptions)} />
      {questions.length > 0 ? (
        <Section title="Open questions">
          <ul className="cards">
            {questions.map((question, index) => (
              <li key={text(question.question_id) ?? index} className="card">
                <strong>{text(question.question_id) ?? `Question ${index + 1}`}</strong>
                <p className="prose">{text(question.question) ?? ''}</p>
                <p className="muted">{text(question.rationale) ?? ''}</p>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

const STATE_LABELS: Record<RequirementState, { label: string; tone: 'done' | 'stopped' | 'working' | 'neutral' }> = {
  implemented: { label: 'Implemented', tone: 'done' },
  not_implemented: { label: 'Not implemented', tone: 'stopped' },
  assigned: { label: 'Assigned', tone: 'working' },
  unassigned: { label: 'Unassigned', tone: 'neutral' },
};

/**
 * Requirements, with which repository owns each and whether it has been done.
 *
 * The status column is joined from the workstreams' own records -- what the plan scoped and
 * what each attempt reported implementing. Where the platform has recorded nothing, the row
 * says "Assigned" or "Unassigned" rather than guessing at progress.
 */
function RequirementTable({
  title,
  items,
  workstreams,
}: {
  title: string;
  items: Record<string, unknown>[];
  workstreams: Workstream[];
}) {
  if (items.length === 0) return null;

  const rows = items.map((item, index) => {
    const id = text(item.requirement_id) ?? `#${index + 1}`;
    return { id, item, progress: requirementProgress(id, workstreams) };
  });
  const done = rows.filter((row) => row.progress.state === 'implemented').length;
  const known = rows.filter((row) => row.progress.state !== 'unassigned').length;

  const columns: Column<(typeof rows)[number]>[] = [
    {
      key: 'id',
      header: 'ID',
      shrink: true,
      sortValue: (row) => row.id,
      render: (row) => <span className="requirement__id">{row.id}</span>,
    },
    {
      key: 'requirement',
      header: 'Requirement',
      render: (row) => <span className="prose">{text(row.item.description) ?? ''}</span>,
    },
    {
      key: 'priority',
      header: 'Priority',
      shrink: true,
      sortValue: (row) => text(row.item.priority) ?? '',
      render: (row) =>
        text(row.item.priority) ? <Badge outline>{text(row.item.priority)}</Badge> : <span className="subtle">—</span>,
    },
    {
      key: 'status',
      header: 'Status',
      shrink: true,
      sortValue: (row) => row.progress.state,
      render: (row) => (
        <Badge tone={STATE_LABELS[row.progress.state].tone}>{STATE_LABELS[row.progress.state].label}</Badge>
      ),
    },
    {
      key: 'repository',
      header: 'Repository',
      shrink: true,
      sortValue: (row) => row.progress.repositories.join(','),
      render: (row) =>
        row.progress.repositories.length > 0 ? (
          <span className="row" style={{ gap: 'var(--space-1)' }}>
            {row.progress.repositories.map((id) => (
              <RepositoryBadge key={id} repositoryId={id} />
            ))}
          </span>
        ) : (
          <span className="subtle">—</span>
        ),
    },
  ];

  return (
    <Section title={title}>
      {known > 0 ? (
        <p className="muted">
          {done} of {known} assigned {known === 1 ? 'requirement' : 'requirements'} reported as
          implemented. {count(items.length, 'requirement')} in total.
        </p>
      ) : null}
      <DataTable
        label={title}
        columns={columns}
        rows={rows}
        rowKey={(row) => row.id}
        compact
        expandLabel="Show acceptance criteria"
        expand={(row) => (
          <div className="stack stack--tight">
            {list(row.item.acceptance_criteria).length > 0 ? (
              <>
                <span className="details__label">Acceptance criteria</span>
                <ul className="bullets">
                  {list(row.item.acceptance_criteria).map((criterion, index) => (
                    <li key={index}>{criterion}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="muted">No acceptance criteria were recorded for this requirement.</p>
            )}
            {/* Surfaced deliberately: the reviewer sets these aside because no repository
                change can demonstrate them, and silently dropping them is what this field
                exists to prevent. */}
            {list(row.item.acceptance_criteria_not_reviewable).length > 0 ? (
              <div className="callout callout--warn">
                <p className="callout__title">Not checked by review</p>
                <p className="muted">No repository change can demonstrate these:</p>
                <ul className="bullets">
                  {list(row.item.acceptance_criteria_not_reviewable).map((criterion, index) => (
                    <li key={index}>{criterion}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {row.progress.responsibility ? (
              <DetailList narrow>
                <DetailRow label="Responsibility">{row.progress.responsibility}</DetailRow>
                {row.progress.criteria.length > 0 ? (
                  <DetailRow label="Criterion ids">{row.progress.criteria.join(', ')}</DetailRow>
                ) : null}
              </DetailList>
            ) : null}
            {list(row.item.dependencies).length > 0 ? (
              <DetailList narrow>
                <DetailRow label="Depends on">{list(row.item.dependencies).join(', ')}</DetailRow>
              </DetailList>
            ) : null}
          </div>
        )}
      />
    </Section>
  );
}
