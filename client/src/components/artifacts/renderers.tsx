/* eslint-disable react-refresh/only-export-components --
   This module's export is a registry of renderers keyed by artifact type, not a component.
   The presentational helpers below exist only to build that map; fast refresh cannot track a
   value-keyed registry either way, so the rule has nothing to protect here. */
import { useEffect, useState } from 'react';
import { recordedAttachments } from './attachment-records';
import { AttachmentStrip } from './attachments';
import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import type { ArtifactRenderer } from './ArtifactViewer';
import { RawJson } from '@/components/ui/Code';
import { Badge, SeverityBadge } from '@/components/ui/Badge';
import { DetailList, DetailRow, ExternalLink } from '@/components/ui/Value';
import { Markdown } from '@/components/common/Markdown';
import { safeHttpUrl } from '@/utils/url';

/**
 * Formatted views for the artifact types the platform produces today.
 *
 * Each reads defensively: an artifact is a model-produced document validated against a schema
 * that grows, so a missing section is omitted rather than crashing the panel it appears in.
 */

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
    <section className="artifact__section">
      <h4>{title}</h4>
      {children}
    </section>
  );
}

function Bullets({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <Section title={title}>
      <ul>
        {items.map((item, index) => (
          <li key={`${index}-${item.slice(0, 24)}`}>{item}</li>
        ))}
      </ul>
    </Section>
  );
}

function Urls({ title, value }: { title: string; value: unknown }) {
  if (!Array.isArray(value)) return null;
  const urls = value.map(safeHttpUrl).filter((item): item is string => item !== null);
  if (urls.length === 0) return null;
  return (
    <Section title={title}>
      <ul>
        {urls.map((url) => (
          <li key={url}>
            <ExternalLink href={url}>{url}</ExternalLink>
          </li>
        ))}
      </ul>
    </Section>
  );
}

/**
 * A list whose entries may be strings or objects.
 *
 * The contract's error and authorization entries are objects, and the plan's parallel groups
 * are lists of lists. Rendering only the strings dropped them entirely, so an object is
 * summarised from whatever fields it has rather than being skipped.
 */
function Entries({ title, value }: { title: string; value: unknown }) {
  if (!Array.isArray(value) || value.length === 0) return null;
  return (
    <Section title={title}>
      <ul>
        {value.map((item, index) => (
          <li key={index}>{describe(item)}</li>
        ))}
      </ul>
    </Section>
  );
}

function describe(item: unknown): string {
  if (typeof item === 'string') return item;
  if (Array.isArray(item)) return item.map(describe).join(', ');
  if (item && typeof item === 'object') {
    return Object.entries(item)
      .filter(([, value]) => value !== null && value !== '' && value !== undefined)
      .map(([key, value]) => `${humaniseKey(key)}: ${describe(value)}`)
      .join(' · ');
  }
  return String(item);
}

function humaniseKey(key: string): string {
  return key.replace(/_/g, ' ');
}

/** A small object rendered as its own fields, for contract sections that are maps. */
function KeyValues({ title, value }: { title: string; value: unknown }) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const entries = Object.entries(value as Record<string, unknown>).filter(
    ([, item]) => item !== null && item !== '' && item !== undefined,
  );
  if (entries.length === 0) return null;
  return (
    <Section title={title}>
      <DetailList>
        {entries.map(([key, item]) => (
          <DetailRow key={key} label={humaniseKey(key)}>
            {describe(item)}
          </DetailRow>
        ))}
      </DetailList>
    </Section>
  );
}

function Prose({ title, value }: { title: string; value: unknown }) {
  // A schema that grows can turn a paragraph into a list. Rendering nothing in that case is
  // how the whole rollback plan vanished from the completion artifact, so a non-string that
  // holds something is passed on rather than dropped.
  if (value !== null && value !== undefined && typeof value !== 'string') {
    return <Entries title={title} value={Array.isArray(value) ? value : [value]} />;
  }
  const content = text(value);
  if (content === null) return null;
  return (
    <Section title={title}>
      {/* Agent prose is markdown in practice. `Markdown` parses to React elements and never
          produces HTML, so this is structured without becoming a script-injection surface. */}
      <Markdown>{content}</Markdown>
    </Section>
  );
}

function Requirements({ items }: { items: Record<string, unknown>[] }) {
  if (items.length === 0) return null;
  return (
    <Section title="Requirements">
      <ul className="cards">
        {items.map((item, index) => (
          <li key={text(item.requirement_id) ?? index} className="card">
            <strong>{text(item.requirement_id) ?? `Requirement ${index + 1}`}</strong>
            {text(item.priority) ? <Badge outline>{text(item.priority)}</Badge> : null}
            <p className="prose">{text(item.description) ?? ''}</p>
            {list(item.acceptance_criteria).length > 0 ? (
              <ul className="bullets muted">
                {list(item.acceptance_criteria).map((criterion, position) => (
                  <li key={position}>{criterion}</li>
                ))}
              </ul>
            ) : null}
            {/* Surfaced deliberately: the reviewer sets these aside because no repository
                change can demonstrate them, and silently dropping them is what this field
                exists to prevent. */}
            {list(item.acceptance_criteria_not_reviewable).length > 0 ? (
              <div className="callout callout--warn">
                <p className="muted">Not checked by review — no change can demonstrate these:</p>
                <ul>
                  {list(item.acceptance_criteria_not_reviewable).map((criterion, position) => (
                    <li key={position}>{criterion}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </li>
        ))}
      </ul>
    </Section>
  );
}

const prd: ArtifactRenderer = (payload) => (
  <>
    <Prose title="Title" value={payload.title} />
    <Prose title="Problem" value={payload.problem_statement} />
    <Bullets title="Goals" items={list(payload.goals)} />
    {/* The same strip the document view shows, rather than the JSON array of hashes the
        generic fallback would otherwise dump. A reader opening a PRD through this renderer
        is reading the same submission. */}
    <AttachmentSection payload={payload} />
    <Entries title="User stories" value={payload.user_stories} />
    <Requirements items={objects(payload.requirements)} />
    <Bullets title="Constraints" items={list(payload.constraints)} />
    <Bullets title="Out of scope" items={list(payload.out_of_scope)} />
    <Bullets title="Stakeholders" items={list(payload.stakeholders)} />
  </>
);

function AttachmentSection({ payload }: { payload: Record<string, unknown> }) {
  const attachments = recordedAttachments(payload);
  if (attachments.length === 0) return null;
  return (
    <section className="artifact__section">
      <h4>Screens and mock-ups</h4>
      <AttachmentStrip attachments={attachments} />
    </section>
  );
}

const technicalPrd: ArtifactRenderer = (payload) => (
  <>
    <Prose title="Title" value={payload.title} />
    <Prose title="Solution summary" value={payload.solution_summary} />
    <Requirements items={objects(payload.functional_requirements)} />
    <Requirements items={objects(payload.non_functional_requirements)} />
    <Bullets title="Data requirements" items={list(payload.data_requirements)} />
    <Bullets title="Integration requirements" items={list(payload.integration_requirements)} />
    <Bullets title="Security requirements" items={list(payload.security_requirements)} />
    <Bullets title="Assumptions" items={list(payload.assumptions)} />
    {objects(payload.unresolved_questions).length > 0 ? (
      <Section title="Open questions">
        <ul className="cards">
          {objects(payload.unresolved_questions).map((question, index) => (
            <li key={text(question.question_id) ?? index} className="card">
              <strong>{text(question.question_id)}</strong>
              <p className="prose">{text(question.question) ?? ''}</p>
              <p className="muted">{text(question.rationale) ?? ''}</p>
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
  </>
);

const reconnaissance: ArtifactRenderer = (payload) => (
  <>
    <Prose title="Summary" value={payload.summary} />
    <Bullets title="Source areas" items={list(payload.source_areas)} />
    <Bullets title="Test areas" items={list(payload.test_areas)} />
    <Bullets title="Shared utilities" items={list(payload.shared_utilities)} />
    {objects(payload.conventions).length > 0 ? (
      <Section title="Conventions">
        <ul className="cards">
          {objects(payload.conventions).map((convention, index) => (
            <li key={text(convention.convention_id) ?? index} className="card">
              <strong>{text(convention.kind) ?? 'convention'}</strong>
              <p className="prose">{text(convention.description) ?? ''}</p>
              {text(convention.wiring_path) ? (
                <p className="muted">
                  New members are registered in <code>{text(convention.wiring_path)}</code>
                </p>
              ) : null}
              <p className="muted">Evidence: {list(convention.evidence_paths).join(', ')}</p>
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    {/* The most valuable part of this artifact: what the requirements assumed and the
        repository does not have. */}
    {objects(payload.contradicted_premises).length > 0 ? (
      <Section title="Contradicted premises">
        <ul className="cards">
          {objects(payload.contradicted_premises).map((premise, index) => (
            <li key={index} className="card callout--warn">
              <p className="prose">
                <strong>Assumed:</strong> {text(premise.premise) ?? ''}
              </p>
              <p className="prose">
                <strong>Actually:</strong> {text(premise.contradicted_by) ?? ''}
              </p>
              <p className="muted">Evidence: {list(premise.evidence_paths).join(', ')}</p>
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
  </>
);

const integrationContract: ArtifactRenderer = (payload) => (
  <>
    <Section title="Version">
      <p className="prose">
        {text(payload.contract_version) ?? 'unknown'} · {text(payload.api_style) ?? 'unspecified'} ·{' '}
        {text(payload.status) ?? ''}
      </p>
    </Section>
    <Bullets title="Owning repositories" items={list(payload.owning_workstreams)} />
    <KeyValues title="Authentication" value={payload.authentication_contract} />
    <Entries title="Authorization rules" value={payload.authorization_rules} />
    <Entries title="Error contracts" value={payload.error_contracts} />
    <Entries title="Events" value={payload.event_contracts} />
    <Entries title="Environment variables" value={payload.environment_variables} />
    <KeyValues title="Compatibility policy" value={payload.compatibility_policy} />
    {payload.openapi_document && typeof payload.openapi_document === 'object' ? (
      <Section title="OpenAPI document">
        {/* A whole specification. Summarising it into prose would lose the thing that makes it
            useful, so it is offered as itself. */}
        <details>
          <summary className="muted">Show the specification</summary>
          <RawJson value={payload.openapi_document} />
        </details>
      </Section>
    ) : null}
    {objects(payload.endpoints).length > 0 ? (
      <Section title="Endpoints">
        <ul className="cards">
          {objects(payload.endpoints).map((endpoint, index) => (
            <li key={text(endpoint.operation_id) ?? index} className="card">
              <code>
                {text(endpoint.method) ?? ''} {text(endpoint.path) ?? ''}
              </code>
              <p className="muted">{text(endpoint.operation_id) ?? ''}</p>
              <p className="prose">{text(endpoint.description) ?? ''}</p>
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    {objects(payload.shared_schemas).length > 0 ? (
      <Section title="Shared schemas">
        <RawJson value={payload.shared_schemas} />
      </Section>
    ) : null}
    {objects(payload.environment_variables).length > 0 ? (
      <Section title="Environment variables">
        <RawJson value={payload.environment_variables} />
      </Section>
    ) : null}
  </>
);

const integrationReview: ArtifactRenderer = (payload) => (
  <>
    <Section title="Verdict">
      <p className="prose">{text(payload.review_status) ?? 'unknown'}</p>
    </Section>
    {objects(payload.repository_results).length > 0 ? (
      <Section title="Repositories reviewed">
        <ul className="cards">
          {objects(payload.repository_results).map((item, index) => (
            <li key={text(item.repository_id) ?? index} className="card">
              <div className="card__header">
                <strong>{text(item.repository_id) ?? ''}</strong>
                <Badge>{text(item.status) ?? ''}</Badge>
              </div>
              {text(item.child_result_artifact_id) ? (
                <p className="muted">
                  <code>{text(item.child_result_artifact_id)}</code>
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    <Prose title="Compatibility" value={payload.compatibility_assessment} />
    <Prose title="Security" value={payload.security_assessment} />
    <Prose title="Deployment" value={payload.deployment_assessment} />
    <Bullets title="Checks performed" items={list(payload.contract_checks)} />
    <Bullets title="Merge order" items={list(payload.merge_order)} />
    {objects(payload.cross_repository_findings).length > 0 ? (
      <Section title="Findings">
        <ul className="cards">
          {objects(payload.cross_repository_findings).map((finding, index) => (
            <li key={text(finding.finding_id) ?? index} className="card">
              <strong>{text(finding.finding_id) ?? ''}</strong>
              <SeverityBadge severity={text(finding.severity) ?? 'unspecified'} />
              <p className="prose">{text(finding.description) ?? ''}</p>
              <p className="muted">Responsible: {text(finding.responsible_repository_id) ?? ''}</p>
              <p className="muted">{text(finding.recommended_fix) ?? ''}</p>
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
  </>
);

const review: ArtifactRenderer = (payload) => (
  <>
    <Section title="Verdict">
      <p className="prose">{text(payload.verdict) ?? 'unknown'}</p>
    </Section>
    <Prose title="Summary" value={payload.summary} />
    {objects(payload.findings).length > 0 ? (
      <Section title="Findings">
        <ul className="cards">
          {objects(payload.findings).map((finding, index) => (
            <li key={text(finding.finding_id) ?? index} className="card">
              <strong>{text(finding.title) ?? text(finding.finding_id) ?? ''}</strong>
              <SeverityBadge severity={text(finding.severity) ?? 'unspecified'} />
              <p className="prose">{text(finding.description) ?? ''}</p>
              {text(finding.recommendation) ? (
                <p className="muted">{text(finding.recommendation)}</p>
              ) : null}
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    {objects(payload.requirement_checks).length > 0 ? (
      <Section title="Requirement checks">
        {/* The findings say what is wrong; these say which requirement each verdict is about,
            with the evidence the reviewer actually read. Without them a rejection is a list of
            complaints with no map back to what was asked for. */}
        <ul className="cards">
          {objects(payload.requirement_checks).map((check, index) => (
            <li key={text(check.requirement_id) ?? index} className="card">
              <div className="card__header">
                <strong>{text(check.requirement_id) ?? `Check ${index + 1}`}</strong>
                <Badge tone={check.passed === true ? 'done' : 'stopped'}>
                  {check.passed === true ? 'passed' : 'not met'}
                </Badge>
              </div>
              {text(check.evidence) ? <p className="prose">{text(check.evidence)}</p> : null}
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    <Prose title="Architecture" value={payload.architecture_assessment} />
    <Prose title="Security" value={payload.security_assessment} />
    <Prose title="Tests" value={payload.test_coverage_assessment} />
  </>
);

const codeCompletion: ArtifactRenderer = (payload) => (
  <>
    <Section title="Status">
      <p className="prose">{text(payload.completion_status) ?? 'unknown'}</p>
    </Section>
    <Prose title="Summary" value={payload.summary} />
    <KeyValues
      title="Revision evidence"
      value={{
        commit_sha: payload.commit_sha,
        test_coverage_percent: payload.test_coverage_percent,
        production_diff_fingerprint: payload.production_diff_fingerprint,
      }}
    />
    {objects(payload.file_changes).length > 0 ? (
      <Section title="Files">
        {/* The platform records a change type per path -- added, modified, deleted -- which is
            what a code-change view can honestly show. It does not record diff content, so none
            is reconstructed here. */}
        <ul className="cards">
          {objects(payload.file_changes).map((change, index) => (
            <li key={text(change.path) ?? index} className="card">
              <div className="card__header">
                <code className="mono">{text(change.path) ?? ''}</code>
                <Badge outline>{text(change.change_type) ?? 'changed'}</Badge>
              </div>
              {text(change.description) ? (
                <p className="muted">{text(change.description)}</p>
              ) : null}
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    <Entries
      title="Implementation expectations satisfied"
      value={payload.implementation_expectations_satisfied}
    />
    <Entries title="Validation results" value={payload.validation_results} />
    {objects(payload.requirement_implementation_evidence).length > 0 ? (
      <Section title="Evidence per requirement">
        <ul className="cards">
          {objects(payload.requirement_implementation_evidence).map((item, index) => (
            <li key={text(item.requirement_id) ?? index} className="card">
              <strong>{text(item.requirement_id) ?? ''}</strong>
              {list(item.files).length > 0 ? (
                <p className="muted">Files: {list(item.files).join(', ')}</p>
              ) : null}
              {list(item.symbols).length > 0 ? (
                <p className="muted">Symbols: {list(item.symbols).join(', ')}</p>
              ) : null}
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    <Bullets title="Production files" items={list(payload.production_files_changed)} />
    <Bullets title="Test files" items={list(payload.test_files_changed)} />
    <Bullets title="Configuration files" items={list(payload.configuration_files_changed)} />
    <Bullets title="Requirements implemented" items={list(payload.requirements_implemented)} />
    <Bullets title="Requirements not implemented" items={list(payload.requirements_not_implemented)} />
    <Bullets title="Remaining work" items={list(payload.remaining_work)} />
    {/* The backend records changed paths, not diff content, so no diff is shown rather than
        one being reconstructed inaccurately here. */}
  </>
);

const pullRequest: ArtifactRenderer = (payload) => (
  <>
    <Section title="Pull request">
      <p className="prose">
        {safeHttpUrl(payload.url) ? (
          <a href={safeHttpUrl(payload.url)!} target="_blank" rel="noopener noreferrer">
            {text(payload.repository) ?? ''} #{String(payload.pull_request_number ?? '')}
          </a>
        ) : (
          <span>
            {text(payload.repository) ?? ''} #{String(payload.pull_request_number ?? '')}
          </span>
        )}
      </p>
      <p className="muted">
        {text(payload.source_branch) ?? ''} → {text(payload.target_branch) ?? ''} ·{' '}
        {text(payload.state) ?? ''}
      </p>
      {/* The commit this branch was pushed at. It is the one identifier that ties what the
          platform reports back to what is actually on the remote, so a reviewer can check
          the two agree rather than taking the artifact's word for it. */}
      {text(payload.commit_sha) ? (
        <p className="muted">
          Commit <code title={text(payload.commit_sha)!}>{text(payload.commit_sha)}</code>
        </p>
      ) : null}
      {list(payload.reviewers).length > 0 ? (
        <p className="muted">Reviewers: {list(payload.reviewers).join(', ')}</p>
      ) : null}
      {list(payload.labels).length > 0 ? (
        <p className="muted">Labels: {list(payload.labels).join(', ')}</p>
      ) : null}
    </Section>
    <Prose title="Title" value={payload.title} />
    <Section title="Body">
      <Markdown>{text(payload.body) ?? ''}</Markdown>
    </Section>
  </>
);

const featureCompletion: ArtifactRenderer = (payload) => (
  <>
    <Section title="Status">
      <p className="prose">{text(payload.status) ?? ''}</p>
    </Section>
    <Bullets title="Merge order" items={list(payload.merge_order)} />
    <Section title="Strategy">
      <p className="prose">
        Merge: {text(payload.merge_strategy) ?? 'unspecified'} · Deployment:{' '}
        {text(payload.deployment_strategy) ?? 'unspecified'}
      </p>
    </Section>
    <Urls title="Pull requests" value={payload.pull_request_urls} />
    <Entries title="Feature flags" value={payload.feature_flags} />
    <Bullets title="Known limitations" items={list(payload.known_limitations)} />
    <Entries title="Rollback plan" value={payload.rollback_plan} />
  </>
);

const repositoryExecutionPlan: ArtifactRenderer = (payload) => (
  <>
    <Bullets title="Execution order" items={list(payload.execution_order)} />
    <Entries title="Parallel groups" value={payload.parallel_groups} />
    <Entries title="Recommended merge order" value={payload.recommended_merge_order} />
    <Entries title="Integration test plan" value={payload.integration_test_plan} />
    <Entries title="Rollback strategy" value={payload.rollback_strategy} />
    <Entries title="Feature flags" value={payload.feature_flag_strategy} />
    <Section title="Strategy">
      <p className="prose">
        Merge: {text(payload.merge_strategy) ?? 'unspecified'} · Deployment:{' '}
        {text(payload.deployment_strategy) ?? 'unspecified'}
      </p>
    </Section>
    {objects(payload.workstreams).length > 0 ? (
      <Section title="Workstreams">
        <ul className="cards">
          {objects(payload.workstreams).map((workstream, index) => (
            <li key={text(workstream.workstream_id) ?? index} className="card">
              <strong>{text(workstream.repository_id) ?? ''}</strong>
              <Badge outline>{text(workstream.role) ?? ''}</Badge>
              <Bullets title="Responsibilities" items={list(workstream.responsibilities)} />
              <Bullets
                title="Depends on workstreams"
                items={list(workstream.dependency_workstream_ids)}
              />
              <Bullets title="Tasks" items={list(workstream.task_ids)} />
              <Bullets title="Acceptance criteria" items={list(workstream.acceptance_criteria)} />
              <Bullets
                title="Contract sections implemented"
                items={list(workstream.contract_sections_implemented)}
              />
              <Bullets
                title="Contract sections consumed"
                items={list(workstream.contract_sections_consumed)}
              />
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
  </>
);

const architecture: ArtifactRenderer = (payload) => (
  <>
    <Prose title="Overview" value={payload.system_overview} />
    <KeyValues title="Technology stack" value={payload.technology_stack} />
    <Entries title="Repository structure" value={payload.repository_structure} />
    {objects(payload.api_contracts).length > 0 ? (
      <Section title="API contracts">
        <ul className="cards">
          {objects(payload.api_contracts).map((contract, index) => (
            <li key={index} className="card">
              <code>
                {text(contract.method) ?? ''} {text(contract.path) ?? ''}
              </code>
              <p className="prose">{text(contract.summary) ?? ''}</p>
              {text(contract.request_schema) ? (
                <p className="muted">Request: {text(contract.request_schema)}</p>
              ) : null}
              {text(contract.response_schema) ? (
                <p className="muted">Response: {text(contract.response_schema)}</p>
              ) : null}
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    <Prose title="Deployment strategy" value={payload.deployment_strategy} />
    {objects(payload.risks).length > 0 ? (
      <Section title="Risks">
        <ul className="cards">
          {objects(payload.risks).map((risk, index) => (
            <li key={text(risk.risk_id) ?? index} className="card">
              <div className="card__header">
                <strong>{text(risk.risk_id) ?? `Risk ${index + 1}`}</strong>
                <Badge outline>
                  {text(risk.likelihood) ?? '?'} / {text(risk.impact) ?? '?'}
                </Badge>
              </div>
              <p className="prose">{text(risk.description) ?? ''}</p>
              {text(risk.mitigation) ? <p className="muted">{text(risk.mitigation)}</p> : null}
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    {objects(payload.components).length > 0 ? (
      <Section title="Components">
        <ul className="cards">
          {objects(payload.components).map((component, index) => (
            <li key={text(component.component_id) ?? index} className="card">
              <strong>{text(component.name) ?? ''}</strong>
              <p className="prose">{text(component.responsibility) ?? ''}</p>
              <p className="muted">{text(component.technology) ?? ''}</p>
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
    {objects(payload.decisions).length > 0 ? (
      <Section title="Decisions">
        <ul className="cards">
          {objects(payload.decisions).map((decision, index) => (
            <li key={text(decision.decision_id) ?? index} className="card">
              <strong>{text(decision.title) ?? ''}</strong>
              <p className="prose">{text(decision.rationale) ?? ''}</p>
            </li>
          ))}
        </ul>
      </Section>
    ) : null}
  </>
);

/**
 * One attempt at one repository, which is the artifact the platform produces most of.
 *
 * A repository that tried ten times has ten of these, and they are what the agent history
 * links to. Falling back to raw JSON meant the record of what an attempt actually did was the
 * least readable thing in the workspace.
 */
const childWorkflowResult: ArtifactRenderer = (payload) => (
  <>
    <Section title="Outcome">
      <p className="prose">
        {text(payload.repository_id) ?? ''} — {text(payload.status) ?? 'unknown'}
        {payload.pull_request_readiness === true ? ' · ready for a pull request' : ''}
      </p>
      {text(payload.failure_classification) ? (
        <p className="muted">Classified as {text(payload.failure_classification)}</p>
      ) : null}
      {/* Whether this attempt changed anything the last one had not. An attempt that submits
          the same source is what the platform refuses to repeat, so its own judgement of that
          belongs beside the outcome. */}
      {typeof payload.meaningful_change === 'boolean' ? (
        <p className="muted">
          {payload.meaningful_change ? 'Changed production source' : 'No meaningful change'}
          {text(payload.meaningful_change_reason)
            ? ` — ${text(payload.meaningful_change_reason)}`
            : ''}
        </p>
      ) : null}
    </Section>
    <Bullets title="Blocking issues" items={list(payload.blocking_issues)} />
    <Bullets title="All files changed" items={list(payload.changed_files)} />
    <Bullets title="Production files" items={list(payload.production_files_changed)} />
    <Bullets title="Test files" items={list(payload.test_files_changed)} />
    <Bullets title="Requirements implemented" items={list(payload.requirements_implemented)} />
    <Entries title="Requirements scoped here" value={payload.scoped_requirements} />
    <Bullets
      title="Deliberately not this repository's"
      items={list(payload.out_of_scope_requirements)}
    />
    <Bullets title="Contract sections consumed" items={list(payload.contract_sections_consumed)} />
    <KeyValues title="Strategy given to the next attempt" value={payload.retry_strategy} />
    {/* Whether the repository could run its own checks on an untouched checkout. A repository
        that cannot is a different problem from one whose code was wrong, and the platform
        classifies it that way. */}
    <KeyValues title="Repository preflight" value={payload.preflight_result} />
    <KeyValues title="Layout evidence" value={payload.layout_evidence} />
    {typeof payload.superseded_validation_count === 'number' &&
    payload.superseded_validation_count > 0 ? (
      <Section title="Superseded validation">
        <p className="muted">
          {payload.superseded_validation_count} earlier validation result
          {payload.superseded_validation_count === 1 ? '' : 's'} were replaced by a later
          revision of this repository.
        </p>
      </Section>
    ) : null}
    <Section title="Where it ran">
      <DetailList>
        <DetailRow label="Branch">{text(payload.branch_name) ?? ''}</DetailRow>
        <DetailRow label="Workspace">{text(payload.workspace_path) ?? ''}</DetailRow>
      </DetailList>
    </Section>
  </>
);

const executionGraph: ArtifactRenderer = (payload) => (
  <>
    <Prose title="Entry node" value={payload.entry_node_id} />
    <Bullets title="Terminal nodes" items={list(payload.terminal_node_ids)} />
    <Entries title="Nodes" value={payload.nodes} />
    <Entries title="Transitions" value={payload.edges} />
  </>
);

const taskPlan: ArtifactRenderer = (payload) => (
  <>
    <Prose title="Summary" value={payload.summary} />
    <Bullets title="Implementation order" items={list(payload.implementation_order)} />
    <Entries title="Tasks" value={payload.tasks} />
    <Entries title="Milestones" value={payload.milestones} />
    <Bullets title="Test strategy" items={list(payload.test_strategy)} />
    <Bullets title="Risk management" items={list(payload.risk_management_plan)} />
  </>
);

const contractChangeRequest: ArtifactRenderer = (payload) => (
  <>
    <Section title="Request">
      <p className="prose">
        {text(payload.change_request_id) ?? ''} · {text(payload.status) ?? 'unknown'}
      </p>
      <p className="muted">
        Requested by {text(payload.requested_by_repository_id) ?? 'unknown'} against contract{' '}
        {text(payload.current_contract_version) ?? 'unknown'}
      </p>
    </Section>
    <Prose title="Reason" value={payload.reason} />
    <Bullets title="Requested changes" items={list(payload.requested_changes)} />
    <Bullets title="Affected workstreams" items={list(payload.affected_workstreams)} />
    <Prose title="Compatibility impact" value={payload.compatibility_impact} />
    <Bullets title="Migration requirements" items={list(payload.migration_requirements)} />
    <Prose title="Resolution" value={payload.resolution} />
  </>
);

/**
 * One cited frame's preview, rendered by the server on demand.
 *
 * Fetched as bytes through the API rather than pointed at with a URL, for two reasons that are
 * both about honesty rather than convenience. Figma's render URLs expire and the snapshot they
 * belong to is frozen, so no URL is stored anywhere; and the console authenticates with a
 * bearer token, which an `<img src>` cannot carry.
 *
 * A preview that will not load is a missing picture and never a missing design: the frame's
 * text -- its names, its layout, its characters -- is what every judge was given, and it is
 * rendered below regardless.
 */
function DesignPreview({
  featureId,
  nodeId,
  label,
}: {
  featureId: string;
  nodeId: string;
  label: string;
}) {
  const api = useApi();
  // The query caches the Blob and never an object URL. An object URL minted inside the query
  // function outlives every component that read it — react-query holds the string, nothing
  // revokes it, and each refetch leaks another blob allocation for the lifetime of the page.
  const preview = useQuery({
    queryKey: ['design-preview', featureId, nodeId],
    queryFn: ({ signal }) => api.getDesignPreview(featureId, nodeId, signal),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });
  const blob = preview.data;
  // The object URL belongs to this mounted component: created when the blob arrives, revoked
  // in the effect's cleanup, so an unmount or a changed blob releases the allocation.
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob) return undefined;
    const url = URL.createObjectURL(blob);
    setObjectUrl(url);
    return () => {
      setObjectUrl(null);
      URL.revokeObjectURL(url);
    };
  }, [blob]);

  if (preview.isError) {
    return (
      <p className="muted">
        This frame could not be rendered just now. Its text is below, which is what every role
        that built or judged this work was given.
      </p>
    );
  }
  if (preview.isPending || !objectUrl) return <p className="muted">Rendering the frame…</p>;
  return <img className="design__preview" src={objectUrl} alt={`Preview of ${label}`} />;
}

/**
 * The designs a feature cited, resolved once.
 *
 * Frames as cards -- the preview, the label, the path, the names, the text -- with the three
 * omission lists as first-class content rather than a footnote. An unreported cap reads as
 * "the whole design was considered", so what the resolution could not show is shown first.
 */
const designSnapshot: ArtifactRenderer = (payload, context) => {
  const nodes = objects(payload.nodes);
  const omitted = objects(payload.design_nodes_omitted);
  const absent = objects(payload.design_nodes_absent);
  const unreachable = objects(payload.design_nodes_unreachable);
  const featureId = context?.featureId ?? '';
  return (
    <>
      {omitted.length + absent.length + unreachable.length > 0 ? (
        <Section title="Frames this snapshot does not contain">
          <div className="callout callout--warn">
            <p className="muted">
              These were cited and are not below, so nothing that built or judged this feature
              was shown them:
            </p>
            <ul>
              {[
                ...omitted.map((item) => ({ item, kind: 'did not fit' })),
                ...absent.map((item) => ({ item, kind: 'not in the file' })),
                ...unreachable.map((item) => ({ item, kind: 'could not be read' })),
              ].map(({ item, kind }, index) => (
                <li key={`${index}-${text(item.node_id) ?? ''}`}>
                  <strong>{text(item.label) || (text(item.node_id) ?? 'a frame')}</strong> —{' '}
                  {kind}
                  {text(item.reason) ? ` (${text(item.reason)})` : ''}
                  {typeof item.characters === 'number' && item.characters > 0
                    ? `, ${item.characters.toLocaleString()} characters`
                    : ''}
                </li>
              ))}
            </ul>
          </div>
        </Section>
      ) : null}
      {nodes.length === 0 ? (
        <Section title="Frames">
          <p className="muted">This snapshot quotes no frame.</p>
        </Section>
      ) : (
        <Section title={`Frames (${nodes.length})`}>
          <ul className="cards">
            {nodes.map((node, index) => {
              const nodeId = text(node.node_id) ?? String(index);
              const label = text(node.label) || nodeId;
              const design = (node.content ?? {}) as Record<string, unknown>;
              const url = safeHttpUrl(node.source_url);
              return (
                <li key={nodeId} className="card design__frame">
                  <div className="card__header">
                    <strong>{label}</strong>
                    <Badge outline>{text(design.type) ?? 'frame'}</Badge>
                  </div>
                  {featureId ? (
                    <DesignPreview featureId={featureId} nodeId={nodeId} label={label} />
                  ) : null}
                  <DetailList narrow>
                    <DetailRow label="Path">{text(design.path) ?? label}</DetailRow>
                    <DetailRow label="Frame">{nodeId}</DetailRow>
                    {url ? (
                      <DetailRow label="In Figma">
                        <ExternalLink href={url}>the link that was pasted</ExternalLink>
                      </DetailRow>
                    ) : null}
                    {list(node.applies_to).length > 0 ? (
                      <DetailRow label="Scoped to">{list(node.applies_to).join(', ')}</DetailRow>
                    ) : null}
                  </DetailList>
                  {/* Names first, because the names are what a change is judged against: a hex
                      code says what to hardcode and a style name says what to reuse. */}
                  <KeyValues title="Style names" value={designNames(design)} />
                  <Bullets title="Components" items={designComponents(design)} />
                  <Bullets title="Text" items={designText(design)} />
                  <RawJson value={design} />
                </li>
              );
            })}
          </ul>
        </Section>
      )}
      <Section title="How this was resolved">
        <DetailList narrow>
          <DetailRow label="Style names read from">
            {text(payload.style_name_source) === 'file_variables'
              ? 'the file’s variables'
              : 'the names on the nodes'}
          </DetailRow>
          {objects(payload.files).map((file, index) => (
            <DetailRow key={index} label={`File ${text(file.file_name) || index + 1}`}>
              {text(file.file_key) ?? ''} at version {text(file.file_version) || 'unknown'}
            </DetailRow>
          ))}
        </DetailList>
      </Section>
    </>
  );
};

/** Every style name a rendered frame references, anywhere in its tree. */
function designNames(design: Record<string, unknown>): Record<string, string> {
  const found: Record<string, string> = {};
  walkDesign(design, (node) => {
    const names = node.style_names;
    if (typeof names !== 'object' || names === null) return;
    for (const [slot, name] of Object.entries(names as Record<string, unknown>)) {
      if (typeof name === 'string') found[`${text(node.name) ?? slot}.${slot}`] = name;
    }
  });
  return found;
}

/** Every component and component set a rendered frame names. */
function designComponents(design: Record<string, unknown>): string[] {
  const found: string[] = [];
  walkDesign(design, (node) => {
    const component = node.component;
    if (typeof component !== 'object' || component === null) return;
    const entry = component as Record<string, unknown>;
    const name = text(entry.name);
    const set = text(entry.set);
    const rendered = set && name ? `${set} → ${name}` : (set ?? name);
    if (rendered && !found.includes(rendered)) found.push(rendered);
  });
  return found;
}

/** Every text node's actual characters, which is the one thing a diff can be checked against. */
function designText(design: Record<string, unknown>): string[] {
  const found: string[] = [];
  walkDesign(design, (node) => {
    const characters = text(node.text);
    if (characters) found.push(characters);
  });
  return found;
}

function walkDesign(
  node: Record<string, unknown>,
  visit: (node: Record<string, unknown>) => void,
): void {
  visit(node);
  for (const child of objects(node.children)) walkDesign(child, visit);
}

export const ARTIFACT_RENDERERS: Record<string, ArtifactRenderer> = {
  child_workflow_result: childWorkflowResult,
  prd,
  technical_prd: technicalPrd,
  repository_reconnaissance: reconnaissance,
  architecture,
  execution_graph: executionGraph,
  task_plan: taskPlan,
  repository_execution_plan: repositoryExecutionPlan,
  integration_contract: integrationContract,
  integration_review: integrationReview,
  review,
  code_completion: codeCompletion,
  pull_request: pullRequest,
  feature_completion: featureCompletion,
  contract_change_request: contractChangeRequest,
  design_snapshot: designSnapshot,
};
