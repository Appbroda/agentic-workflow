import type { Artifact, Workstream } from '@/schemas/feature';

/**
 * Connecting what was asked for to what was built.
 *
 * A product manager reading a PRD wants to know which repository owns each requirement and
 * whether it has been done. The platform records both, in two different places: the plan scopes
 * requirements to workstreams, and each attempt reports which of its scoped requirements it
 * implemented. This joins them.
 *
 * Nothing here invents completion. A requirement no repository claimed is `unassigned`; one
 * assigned but never reported on is `assigned`. "Implemented" is only ever said because a
 * repository said it.
 */

export type RequirementState = 'implemented' | 'not_implemented' | 'assigned' | 'unassigned';

export interface RequirementProgress {
  state: RequirementState;
  /** The repositories this requirement was scoped to, by id. */
  repositories: string[];
  responsibility: string | null;
  /** Acceptance criterion ids the plan attached to it, where it did. */
  criteria: string[];
}

export function requirementProgress(
  requirementId: string,
  workstreams: Workstream[],
): RequirementProgress {
  const repositories: string[] = [];
  let responsibility: string | null = null;
  const criteria = new Set<string>();
  let implemented = false;
  let notImplemented = false;

  for (const workstream of workstreams) {
    const scoped = workstream.scoped_requirements.find(
      (item) => item.requirement_id === requirementId,
    );
    if (scoped) {
      repositories.push(workstream.repository_id);
      if (typeof scoped.responsibility === 'string') responsibility = scoped.responsibility;
      if (Array.isArray(scoped.acceptance_criterion_ids)) {
        for (const id of scoped.acceptance_criterion_ids) {
          if (typeof id === 'string') criteria.add(id);
        }
      }
    }
    if (workstream.requirements_implemented.includes(requirementId)) implemented = true;
    if (workstream.requirements_not_implemented.includes(requirementId)) notImplemented = true;
  }

  const state: RequirementState = implemented
    ? 'implemented'
    : notImplemented
      ? 'not_implemented'
      : repositories.length > 0
        ? 'assigned'
        : 'unassigned';

  return { state, repositories, responsibility, criteria: [...criteria] };
}

export interface RequirementCheck {
  requirementId: string;
  passed: boolean;
  evidence: string;
  repositoryId: string | null;
  artifactId: string;
}

/**
 * What review actually verified, per requirement.
 *
 * Read from the review artifacts, where each check names the requirement, the verdict, and the
 * evidence the reviewer read. Only the newest review per repository counts: an earlier attempt
 * that failed the same requirement is history, not a second open finding.
 */
export function requirementChecks(reviews: Artifact[]): RequirementCheck[] {
  const newestPerRepository = new Map<string, Artifact>();
  for (const artifact of reviews) {
    const repositoryId = repositoryOf(artifact);
    const key = repositoryId ?? artifact.artifact_id;
    const seen = newestPerRepository.get(key);
    if (!seen || Date.parse(artifact.timestamp) >= Date.parse(seen.timestamp)) {
      newestPerRepository.set(key, artifact);
    }
  }

  const checks: RequirementCheck[] = [];
  for (const artifact of newestPerRepository.values()) {
    const raw = artifact.payload.requirement_checks;
    if (!Array.isArray(raw)) continue;
    for (const item of raw) {
      if (typeof item !== 'object' || item === null) continue;
      const record = item as Record<string, unknown>;
      if (typeof record.requirement_id !== 'string') continue;
      checks.push({
        requirementId: record.requirement_id,
        passed: record.passed === true,
        evidence: typeof record.evidence === 'string' ? record.evidence : '',
        repositoryId: repositoryOf(artifact),
        artifactId: artifact.artifact_id,
      });
    }
  }
  return checks;
}

/**
 * Which repository an artifact belongs to.
 *
 * The metadata carries it when the platform recorded it there; otherwise the identifier does,
 * because these are named `007_review.<repository>.attempt-N.json`.
 */
export function repositoryOf(artifact: Artifact): string | null {
  if (typeof artifact.metadata.repository_id === 'string') return artifact.metadata.repository_id;
  const match = /^\d+_[a-z_]+\.([^.]+)\./.exec(artifact.artifact_id);
  return match?.[1] ?? null;
}

export interface ReviewFinding {
  findingId: string;
  severity: string;
  title: string;
  description: string;
  recommendation: string;
  requirementId: string | null;
  repositoryId: string | null;
  artifactId: string;
}

/**
 * Findings from the newest review of each repository, plus the integration review.
 *
 * Superseded attempts are excluded on purpose: a repository that failed review four times and
 * passed on the fifth has no open findings, and listing all four would be a page of problems
 * that no longer exist.
 */
export function reviewFindings(reviews: Artifact[]): ReviewFinding[] {
  const newestPerRepository = new Map<string, Artifact>();
  for (const artifact of reviews) {
    const key = repositoryOf(artifact) ?? artifact.artifact_id;
    const seen = newestPerRepository.get(key);
    if (!seen || Date.parse(artifact.timestamp) >= Date.parse(seen.timestamp)) {
      newestPerRepository.set(key, artifact);
    }
  }

  const findings: ReviewFinding[] = [];
  for (const artifact of newestPerRepository.values()) {
    const repositoryId = repositoryOf(artifact);
    const lists = [artifact.payload.findings, artifact.payload.cross_repository_findings];
    for (const list of lists) {
      if (!Array.isArray(list)) continue;
      for (const item of list) {
        if (typeof item !== 'object' || item === null) continue;
        const record = item as Record<string, unknown>;
        findings.push({
          findingId: text(record.finding_id) ?? `${artifact.artifact_id}-${findings.length}`,
          severity: text(record.severity) ?? 'unspecified',
          title: text(record.title) ?? text(record.finding_id) ?? 'Finding',
          description: text(record.description) ?? '',
          recommendation: text(record.recommended_fix) ?? text(record.recommendation) ?? '',
          requirementId: text(record.requirement_id),
          repositoryId: text(record.responsible_repository_id) ?? repositoryId,
          artifactId: artifact.artifact_id,
        });
      }
    }
  }
  return findings;
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}
