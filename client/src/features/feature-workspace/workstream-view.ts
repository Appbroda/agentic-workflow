import type { Artifact, Workstream } from '@/schemas/feature';
import { count } from '@/utils/count';
import { formatDuration } from '@/utils/time';
import { safeHttpUrl } from '@/utils/url';

/**
 * Readings of a workstream that several views need, computed once.
 *
 * Everything here is derived from what the platform recorded and nothing is inferred beyond
 * it. Where the record cannot answer -- no review artifact, no validation run -- these return
 * null and the view shows an em dash, rather than a confident-looking "passed" nobody checked.
 */

export interface ValidationResult {
  name: string;
  command: string;
  validationType: string | null;
  passed: boolean;
  status: string;
  exitCode: number | null;
  durationSeconds: number | null;
  revision: string | null;
  workingDirectory: string | null;
  stdout: string;
  stderr: string;
  resultCode: string | null;
  failureClassification: string | null;
  required: boolean;
  isCurrent: boolean;
  raw: Record<string, unknown>;
}

function str(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The validation checks a repository ran, normalised.
 *
 * The platform stores bounded, redacted summaries of the output rather than whole logs, so
 * `stdout` and `stderr` here are those summaries and are frequently empty. That is a fact
 * about the record, and the viewer says so rather than showing an empty box.
 */
export function validationResults(workstream: Workstream): ValidationResult[] {
  return workstream.current_validation_results.map((result) => ({
    name: str(result.name) ?? str(result.validation_type) ?? 'check',
    command: Array.isArray(result.command)
      ? result.command.filter((item): item is string => typeof item === 'string').join(' ')
      : (str(result.command) ?? ''),
    validationType: str(result.validation_type),
    passed: result.passed === true,
    status: str(result.status) ?? (result.passed === true ? 'passed' : 'failed'),
    exitCode: num(result.exit_code),
    durationSeconds: num(result.duration_seconds),
    revision: str(result.repository_revision),
    workingDirectory: str(result.working_directory),
    stdout: str(result.stdout_summary) ?? '',
    stderr: str(result.stderr_summary) ?? '',
    resultCode: str(result.result_code),
    failureClassification: str(result.failure_classification),
    required: result.required !== false,
    isCurrent: result.is_current !== false,
    raw: result,
  }));
}

/** "3 of 4 passed", or nothing when this repository has not run its checks. */
export function validationSummary(workstream: Workstream): { passed: number; total: number } | null {
  const results = workstream.current_validation_results;
  if (results.length === 0) return null;
  return {
    passed: results.filter((result) => result.passed === true).length,
    total: results.length,
  };
}

/**
 * What review said, read from the workstream's own status.
 *
 * The findings live in the review artifact; this is only the verdict, and only where the
 * platform's status vocabulary states one. A repository still running has no verdict, and
 * saying "pending" would imply a review is under way when none has started.
 */
export function reviewOutcome(workstream: Workstream): 'approved' | 'rejected' | null {
  if (workstream.status === 'review_rejected') return 'rejected';
  if (workstream.status === 'approved' || workstream.status === 'completed') {
    return workstream.review_artifact_id ? 'approved' : null;
  }
  return null;
}

/**
 * How far this repository actually got, by which artifacts exist.
 *
 * Deliberately evidence rather than a stage name invented here: the workstream status says
 * what state it is in, and this says what it has produced. A repository that is `failed`
 * having already been implemented and validated reads very differently from one that never
 * cloned, and only the artifacts can tell those apart.
 */
export function reached(workstream: Workstream): string {
  if (workstream.review_artifact_id) return 'Reviewed';
  if (workstream.current_validation_results.length > 0) return 'Validated';
  if (workstream.code_completion_artifact_id) return 'Implemented';
  if (workstream.blocking_setup_issues.length > 0) return 'Repository setup';
  if (workstream.preflight_status) return 'Preflight';
  return 'Not started';
}

/**
 * Which attempt this repository is on.
 *
 * `retry_count` is the counter the platform bounds by `max_child_review_cycles`, and its own
 * word for `retry_count + 1` is "Attempt" -- `decide_child_retry` says "Attempt N may proceed
 * with a changed strategy". Using anything else here meant the repository table and the
 * workflow graph quoted two different numbers for the same repository.
 *
 * The per-classification counters -- implementation, validation, repository setup -- are
 * separate budgets and are shown against their own limits where they matter, never summed
 * into a single figure that matches nothing the platform enforces.
 */
export function attempts(workstream: Workstream): number {
  return workstream.retry_count + 1;
}

/**
 * The provider-fault history one attempt absorbed, read from its result artifact metadata.
 *
 * Recorded by the child loop at result-build time: the fault count, the seconds excluded
 * from the charged clock, the distinct fault labels, and whether a truncated response was
 * re-asked once at reduced effort. `null` means the artifact predates the recording — a
 * zero-fault attempt carries explicit zeros, never absent keys.
 */
export interface FaultEvidence {
  faultCount: number;
  faultSecondsExcluded: number;
  faultClasses: string[];
  truncatedDegradedRetry: boolean;
  wallSeconds: number | null;
  chargedSeconds: number | null;
}

export function faultEvidence(metadata: Record<string, unknown> | undefined): FaultEvidence | null {
  if (!metadata || typeof metadata.fault_count !== 'number') return null;
  return {
    faultCount: metadata.fault_count,
    faultSecondsExcluded: num(metadata.fault_seconds_excluded) ?? 0,
    faultClasses: Array.isArray(metadata.fault_classes)
      ? metadata.fault_classes.filter((item): item is string => typeof item === 'string')
      : [],
    truncatedDegradedRetry: metadata.truncated_degraded_retry === true,
    wallSeconds: num(metadata.runtime_wall_seconds),
    chargedSeconds: num(metadata.runtime_charged_seconds),
  };
}

/** A duration in the record, shown at the resolution a person reads: "51m 12s", "3s". */
function runtimeDuration(seconds: number): string {
  return formatDuration(seconds * 1000) ?? `${Math.round(seconds)}s`;
}

/**
 * The two clocks, labelled with what the provider cost this run.
 *
 * "wall 51m 0s, charged 37m 0s — 844s excluded across 1 provider fault" is the sentence the
 * 186 investigation had to assemble from journal timestamps by hand. Null where the record
 * carries neither a clock nor a fault, so the row is omitted rather than shown empty.
 */
export function runtimeSentence(evidence: FaultEvidence): string | null {
  const clocks =
    evidence.wallSeconds !== null && evidence.chargedSeconds !== null
      ? `wall ${runtimeDuration(evidence.wallSeconds)}, charged ${runtimeDuration(evidence.chargedSeconds)}`
      : null;
  const faults =
    evidence.faultCount > 0
      ? `${Math.round(evidence.faultSecondsExcluded)}s excluded across ${count(
          evidence.faultCount,
          'provider fault',
          'provider faults',
        )}`
      : null;
  if (clocks && faults) return `${clocks} — ${faults}`;
  return clocks ?? faults;
}

export interface PullRequestView {
  artifactId: string;
  repositoryId: string | null;
  repository: string | null;
  number: number | null;
  state: string | null;
  draft: boolean | null;
  sourceBranch: string | null;
  targetBranch: string | null;
  commitSha: string | null;
  title: string | null;
  url: string | null;
  reviewers: string[];
  labels: string[];
}

/**
 * A pull request artifact, read into the fields a table needs.
 *
 * The repository is matched through the workstream that recorded the artifact rather than by
 * parsing the artifact's filename; the filename is a fallback for records written before the
 * workstream carried the link.
 */
export function pullRequestView(artifact: Artifact, workstreams: Workstream[]): PullRequestView {
  const payload = artifact.payload;
  const owner = workstreams.find((item) => item.pull_request_artifact_id === artifact.artifact_id);
  const strings = (value: unknown): string[] =>
    Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];

  return {
    artifactId: artifact.artifact_id,
    repositoryId:
      owner?.repository_id ??
      str(artifact.metadata.repository_id) ??
      artifact.artifact_id.replace(/^\d+_pull_request\./, '').replace(/\.json$/, '') ??
      null,
    repository: str(payload.repository),
    number: num(payload.pull_request_number),
    state: str(payload.state),
    draft: typeof payload.draft === 'boolean' ? payload.draft : null,
    sourceBranch: str(payload.source_branch),
    targetBranch: str(payload.target_branch),
    commitSha: str(payload.commit_sha),
    title: str(payload.title),
    url: safeHttpUrl(payload.url),
    reviewers: strings(payload.reviewers),
    labels: strings(payload.labels),
  };
}
