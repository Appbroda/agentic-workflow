import type { Artifact } from '@/schemas/feature';

/**
 * What the newest integration review decided.
 *
 * The integration review is the gate on publication, and it is a gate that can be closed
 * twice: a `changes_requested` verdict sends the feature back to a repository and another
 * review follows. Every surface that draws the review or the pull requests after it needs the
 * verdict, not merely the existence of a review artifact -- run 185 looped back for three more
 * coding attempts while the progress strip said "Integration review: done" and "Pull requests:
 * in progress", so a cold reader concluded the feature was publishing while a repository was
 * failing.
 */
export interface IntegrationReviewOutcome {
  /** The `review_status` the artifact recorded, unchanged: the record's own word. */
  status: string;
  /** The only verdict that opens the gate. Everything else leaves it shut. */
  approved: boolean;
}

/**
 * The verdict of the newest integration review among these artifacts, where it can be read.
 *
 * "Newest" is the last one in the list, which is the order the platform appended them in and
 * the same authority the workflow itself uses to decide what a repository is being asked to
 * fix (`_outstanding_integration_fixes`). Reading a timestamp instead would be a second
 * definition of the same word.
 *
 * Null has one meaning: *not known from these artifacts*. That covers a feature whose review
 * has not happened yet and, just as importantly, an artifact list fetched without payloads --
 * the list endpoint blanks `payload` unless asked, and `review_status` lives in the payload.
 * A caller holding envelopes alone gets null and draws what it drew before, rather than a
 * verdict invented from a field that was never sent.
 */
export function latestIntegrationReview(artifacts: Artifact[]): IntegrationReviewOutcome | null {
  let status: string | null = null;
  for (const artifact of artifacts) {
    if (artifact.artifact_type !== 'integration_review') continue;
    // Reassigned rather than accumulated: an older review's verdict must not stand in for a
    // newer one whose payload was not fetched.
    const value = artifact.payload.review_status;
    status = typeof value === 'string' && value !== '' ? value : null;
  }
  return status === null ? null : { status, approved: status === 'approved' };
}
