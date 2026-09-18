import type { Feature, Workstream } from '@/schemas/feature';
import { humanise } from '@/utils/text';
import { effectiveFeatureStatus } from './feature-status';
import type { IntegrationReviewOutcome } from './integration-review';

/**
 * The shape of a feature's run, derived from what the backend reported.
 *
 * Everything here is read from state: the repositories are whatever the feature has, the
 * branch order is the plan's, and each stage's state comes from the artifacts and statuses
 * the platform published. Nothing is hardcoded -- there is no frontend/backend pair in this
 * model, only however many repositories a feature happens to name.
 */

export type StageState = 'done' | 'active' | 'pending' | 'stopped';

export interface Stage {
  id: string;
  label: string;
  state: StageState;
  /** Set when the reading needs a word the state alone does not carry, such as "Queued". */
  note?: string | null;
  /** Present only on the fan-out stage: one entry per repository, in plan order. */
  branches?: Branch[];
}

export interface Branch {
  repositoryId: string;
  label: string;
  state: StageState;
  status: string;
  /** Set when the reading differs from the raw status, so the record is not misrepresented. */
  note: string | null;
}

/**
 * Statuses the platform uses for a child that stopped and one that finished.
 *
 * Read as sets rather than compared one by one, because the server owns this vocabulary and
 * adds to it; anything unrecognised falls through to "still going", which is the safe reading.
 */
const CHILD_DONE = new Set(['approved', 'completed']);
const CHILD_STOPPED = new Set(['failed', 'review_rejected', 'blocked', 'cancelled']);
const CHILD_NOT_STARTED = new Set(['pending']);

/** Feature statuses that mean nothing further will happen without a person. */
const FEATURE_STOPPED = new Set([
  'failed',
  'failed_requires_human',
  'cancelled',
  'cancelled_with_external_side_effects',
]);

function branchState(status: string): StageState {
  if (CHILD_DONE.has(status)) return 'done';
  if (CHILD_STOPPED.has(status)) return 'stopped';
  if (CHILD_NOT_STARTED.has(status)) return 'pending';
  return 'active';
}

/**
 * Build the stages of one feature's run.
 *
 * `artifactTypes` is what the feature has actually produced. A stage is done when its output
 * exists, which is a fact rather than an inference from the status name -- a feature can be
 * `failed_requires_human` having already written a contract, and drawing that contract as
 * never reached would misdescribe where the work stopped.
 *
 * The integration review is the one stage where existence is not enough, because a review can
 * hand the work back: `integrationReview` carries the verdict, and a caller that has not read
 * it draws exactly what it drew before.
 */
export function buildStages(
  feature: Feature,
  workstreams: Workstream[],
  artifactTypes: Set<string>,
  integrationReview: IntegrationReviewOutcome | null = null,
): Stage[] {
  const effectiveStatus = effectiveFeatureStatus(feature);
  const stopped = FEATURE_STOPPED.has(effectiveStatus);
  // Accepted and durably queued, and no worker has picked it up. Nothing is in progress, so
  // the first stage must not draw as though the product manager were reading: a fresh
  // submission is visible now, and it used to claim work was under way the moment it existed.
  const queued = effectiveStatus === 'pending';
  const done = (produced: boolean, active: boolean): StageState => {
    if (produced) return 'done';
    if (active) return stopped ? 'stopped' : queued ? 'pending' : 'active';
    return 'pending';
  };

  const branches: Branch[] = workstreams.map((item) => {
    const state = branchState(item.status);
    // A child the platform never resolved is left at `running` in the record when a feature is
    // cancelled mid-flight. Nothing is running: feature -076 was cancelled in August and still
    // drew both repositories as in progress. The feature's own terminal state decides, because
    // it is the thing that stopped -- but the raw status is still shown, with a note, so the
    // record is read correctly rather than rewritten.
    const abandoned = stopped && state === 'active';
    return {
      repositoryId: item.repository_id,
      label: item.repository_name ?? item.repository_id,
      state: abandoned ? 'stopped' : state,
      status: item.status,
      note: abandoned ? 'left unresolved' : null,
    };
  });

  const anyBranchStarted = branches.some((item) => item.state !== 'pending');
  const allBranchesDone = branches.length > 0 && branches.every((item) => item.state === 'done');

  // A review that asked for changes is a gate that stayed shut. It sent the work back to a
  // repository, and another review has to follow it -- so this stage is not finished, and the
  // stage after it has not been reached. Run 185 spent nineteen minutes and three coding
  // attempts here while this line read "Integration review ✓ › Pull requests ●".
  const unapprovedReview = integrationReview && !integrationReview.approved ? integrationReview : null;
  const reviewApproved = artifactTypes.has('integration_review') && unapprovedReview === null;

  return [
    {
      id: 'technical_prd',
      label: 'Technical PRD',
      state: done(artifactTypes.has('technical_prd'), true),
      note: queued ? 'queued' : null,
    },
    {
      id: 'integration_contract',
      label: 'Shared contract',
      state: done(artifactTypes.has('integration_contract'), artifactTypes.has('technical_prd')),
    },
    {
      id: 'repository_execution_plan',
      label: 'Execution plan',
      state: done(
        artifactTypes.has('repository_execution_plan'),
        artifactTypes.has('integration_contract'),
      ),
    },
    {
      id: 'workstreams',
      // Named for what it is rather than for a fixed pair, because the count is the feature's.
      label: branches.length === 1 ? 'Repository workstream' : 'Repository workstreams',
      state: allBranchesDone
        ? 'done'
        : branches.some((item) => item.state === 'stopped')
          ? 'stopped'
          : anyBranchStarted
            ? 'active'
            : 'pending',
      branches,
    },
    {
      id: 'integration_review',
      label: 'Integration review',
      // Active, not done and not pending: the review happened, and the cycle it opened is
      // still running. A feature that stopped there stopped there, and the note says with
      // which verdict rather than leaving "✕" to be read as a review that never returned.
      state: unapprovedReview
        ? stopped
          ? 'stopped'
          : 'active'
        : done(reviewApproved, allBranchesDone),
      note: unapprovedReview ? humanise(unapprovedReview.status).toLowerCase() : null,
    },
    {
      id: 'pull_request',
      // Reached only by an approved review. The pull requests themselves are still what
      // finishes it: where they exist, they are the fact, whatever a later review says.
      label: 'Pull requests',
      state: done(artifactTypes.has('pull_request'), reviewApproved),
    },
  ];
}
