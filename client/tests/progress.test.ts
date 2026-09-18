import { describe, expect, it } from 'vitest';
import { buildStages } from '@/features/feature-workspace/stages';
import { latestIntegrationReview } from '@/features/feature-workspace/integration-review';
import { noticesFrom } from '@/features/feature-workspace/notices';
import { artifactsSchema, type Feature, type Workstream } from '@/schemas/feature';
import live185 from './fixtures/integration-reviews.bulk-apps-live-185.json';
import live185WithoutPayloads from './fixtures/integration-reviews.bulk-apps-live-185.no-payload.json';
import live186 from './fixtures/integration-reviews.bulk-apps-live-186.json';

function feature(status: string): Feature {
  return { status } as Feature;
}

function workstream(repositoryId: string, status: string): Workstream {
  return { repository_id: repositoryId, repository_name: null, status } as unknown as Workstream;
}

function event(id: number, name: string, repositoryId?: string) {
  return {
    id,
    timestamp: '2026-08-25T09:00:00Z',
    event_type: 'lifecycle',
    source: 'api',
    event: name,
    details: repositoryId ? { repository_id: repositoryId } : {},
  };
}

describe('workflow progress', () => {
  it('branches once per repository the feature actually has', () => {
    const one = buildStages(
      feature('running_child_workflows'),
      [workstream('only-repo', 'running')],
      new Set(['technical_prd', 'integration_contract', 'repository_execution_plan']),
    );
    const five = buildStages(
      feature('running_child_workflows'),
      ['a', 'b', 'c', 'd', 'e'].map((id) => workstream(id, 'running')),
      new Set(['technical_prd', 'integration_contract', 'repository_execution_plan']),
    );

    // Nothing here assumes a frontend/backend pair: the count is the feature's.
    const fanOut = (stages: ReturnType<typeof buildStages>) =>
      stages.find((item) => item.id === 'workstreams')!;
    expect(fanOut(one).branches).toHaveLength(1);
    expect(fanOut(one).label).toBe('Repository workstream');
    expect(fanOut(five).branches?.map((item) => item.repositoryId)).toEqual([
      'a',
      'b',
      'c',
      'd',
      'e',
    ]);
  });

  it('draws a queued feature as not started, and says it is queued', () => {
    // A submission is durably accepted and answered before anything runs, so this state is
    // now visible for real. Drawing the first stage as in progress claimed the product
    // manager was reading a PRD that no worker had picked up.
    const stages = buildStages(feature('pending'), [], new Set());
    const first = stages[0]!;

    expect(first.state).toBe('pending');
    expect(first.note).toBe('queued');
    expect(stages.every((stage) => stage.state === 'pending')).toBe(true);
  });

  it('reads a stage as reached when its output exists, not from the status name', () => {
    // A feature can stop needing a person having already written a contract. Drawing that
    // contract as never reached would misdescribe where the work actually stopped.
    const stages = buildStages(
      feature('failed_requires_human'),
      [workstream('backend', 'failed')],
      new Set(['technical_prd', 'integration_contract', 'repository_execution_plan']),
    );

    const state = (id: string) => stages.find((item) => item.id === id)!.state;
    expect(state('technical_prd')).toBe('done');
    expect(state('integration_contract')).toBe('done');
    expect(state('workstreams')).toBe('stopped');
    // Never reached. Shown as not started rather than as failed: the integration review did
    // not go wrong, it never happened, and saying it failed would send somebody to read a
    // review that does not exist.
    expect(state('integration_review')).toBe('pending');
    expect(state('pull_request')).toBe('pending');
  });

  it('does not draw an abandoned repository as still running', () => {
    // Cancelling a feature mid-flight leaves its children at `running` in the record. Nothing
    // is running: feature -076 was cancelled in August and still drew both repositories as in
    // progress. The feature's own terminal state decides, because it is what stopped.
    const stages = buildStages(
      feature('cancelled_with_external_side_effects'),
      [workstream('api', 'running'), workstream('web', 'running')],
      new Set(['technical_prd', 'integration_contract', 'repository_execution_plan']),
    );

    const fanOut = stages.find((item) => item.id === 'workstreams')!;
    expect(fanOut.branches?.map((item) => item.state)).toEqual(['stopped', 'stopped']);
    expect(fanOut.state).toBe('stopped');
    // The raw status is still shown, with a note, so the record is read rather than rewritten.
    expect(fanOut.branches?.[0]?.status).toBe('running');
    expect(fanOut.branches?.[0]?.note).toBe('left unresolved');
  });

  it('treats a status it has never seen as still going', () => {
    // The server owns this vocabulary and adds to it. A new status must not read as failure.
    const stages = buildStages(
      feature('some_status_added_next_month'),
      [workstream('backend', 'a_new_child_status')],
      new Set(['technical_prd']),
    );

    const fanOut = stages.find((item) => item.id === 'workstreams')!;
    expect(fanOut.branches?.[0]?.state).toBe('active');
    expect(fanOut.state).toBe('active');
  });
});

/**
 * The integration review loop-back, against the runs that exposed it.
 *
 * Run 185 (2026-09-01) is the whole shape in one feature: the review returned
 * `changes_requested` at 09:20:59, one repository went back for three more coding attempts,
 * and a second review approved at 09:39:36. Run 186 is the same verdict on a feature that
 * then stopped. Both fixtures are the API's own responses, parsed by the client's schema, so
 * a field the server stops sending fails here rather than being quietly read as undefined.
 */
const REVIEWS_185 = artifactsSchema.parse(live185).artifacts;
const REVIEWS_186 = artifactsSchema.parse(live186).artifacts;
/** The same two reviews as the progress surfaces list them: envelopes, `payload` blanked. */
const ENVELOPES_185 = artifactsSchema.parse(live185WithoutPayloads).artifacts;

/** Everything 185 had produced by 09:22, mid loop-back: one review, no pull request. */
const DURING_LOOP_BACK = new Set([
  'technical_prd',
  'integration_contract',
  'repository_execution_plan',
  'code_completion',
  'review',
  'child_workflow_result',
  'integration_review',
]);
const AT_PUBLICATION = new Set([...DURING_LOOP_BACK, 'pull_request']);

/** 185's two repositories at 09:22: the frontend was done, the backend was being fixed. */
const LOOPED_BACK_REPOSITORIES = [
  workstream('AB-console-admin-2.0', 'completed'),
  workstream('admanager_console-2.0', 'running'),
];

function stagesDuringLoopBack() {
  const stages = buildStages(
    feature('running_child_workflows'),
    LOOPED_BACK_REPOSITORIES,
    DURING_LOOP_BACK,
    latestIntegrationReview(REVIEWS_185.slice(0, 1)),
  );
  return (id: string) => stages.find((item) => item.id === id)!;
}

describe('the integration review loop-back', () => {
  it('does not call the review done while it is still asking for changes', () => {
    // What 185 drew instead: "Integration review ✓ › Pull requests ●", for nineteen minutes,
    // while a repository was on its third failed attempt. A cold reader concluded the feature
    // was publishing while a repository was failing.
    const stage = stagesDuringLoopBack();

    expect(stage('integration_review').state).toBe('active');
    expect(stage('integration_review').note).toBe('changes requested');
  });

  it('keeps the pull requests pending until a review approves', () => {
    // The stage after an unopened gate has not been reached. It was drawn as in progress
    // because an integration review artifact existed at all, whatever it decided.
    expect(stagesDuringLoopBack()('pull_request').state).toBe('pending');
  });

  it('says the verdict rather than a bare cross when the feature stopped there', () => {
    // Run 186 asked for changes and was then cancelled. "✕" alone reads as a review that
    // never returned; it returned, and what it said is why the feature is where it is.
    const stages = buildStages(
      feature('cancelled'),
      [workstream('AB-console-admin-2.0', 'completed'), workstream('admanager_console-2.0', 'running')],
      DURING_LOOP_BACK,
      latestIntegrationReview(REVIEWS_186),
    );
    const stage = (id: string) => stages.find((item) => item.id === id)!;

    expect(stage('integration_review').state).toBe('stopped');
    expect(stage('integration_review').note).toBe('changes requested');
    expect(stage('pull_request').state).toBe('pending');
  });

  it('finishes both stages once a later review approves and the pull requests exist', () => {
    // 185's second cycle. The loop-back reading must not outlive the loop.
    const stages = buildStages(
      feature('completed'),
      [workstream('AB-console-admin-2.0', 'completed'), workstream('admanager_console-2.0', 'completed')],
      AT_PUBLICATION,
      latestIntegrationReview(REVIEWS_185),
    );
    const stage = (id: string) => stages.find((item) => item.id === id)!;

    expect(stage('integration_review').state).toBe('done');
    expect(stage('integration_review').note).toBeNull();
    expect(stage('pull_request').state).toBe('done');
  });

  it('reads the newest review as the platform does: the last one appended', () => {
    // The workflow decides what a repository is being asked to fix from the last integration
    // review in state order. A second definition of "newest" here would let the strip and the
    // engineer disagree about which cycle the feature is in.
    expect(latestIntegrationReview(REVIEWS_185)).toEqual({ status: 'approved', approved: true });
    expect(latestIntegrationReview([...REVIEWS_185].reverse())).toEqual({
      status: 'changes_requested',
      approved: false,
    });
  });

  it('claims no verdict from envelopes fetched without their payloads', () => {
    // `review_status` lives in the payload, and the artifact list is fetched without payloads.
    // Reading an absent field as "not approved" would draw every finished feature as looping.
    expect(ENVELOPES_185.map((item) => item.payload)).toEqual([{}, {}]);
    expect(latestIntegrationReview(ENVELOPES_185)).toBeNull();

    const stages = buildStages(feature('completed'), LOOPED_BACK_REPOSITORIES, AT_PUBLICATION, null);
    expect(stages.find((item) => item.id === 'integration_review')!.state).toBe('done');
    expect(stages.find((item) => item.id === 'pull_request')!.state).toBe('done');
  });
});

describe('notifications', () => {
  it('reports what needs a person and ignores routine internal steps', () => {
    const notices = noticesFrom(
      [
        event(1, 'feature_started'),
        event(2, 'feature_checkpoint_after_validation'),
        event(3, 'child_workflow_started', 'backend'),
        event(4, 'child_workflow_failed', 'backend'),
        event(5, 'pull_request_created', 'frontend'),
      ],
      new Set(),
    );

    // Checkpoints and child-started are the platform working, not news.
    expect(notices.map((item) => item.message)).toEqual([
      'A pull request was opened.',
      'A repository stopped.',
    ]);
    expect(notices[1]?.repositoryId).toBe('backend');
  });

  it('never restates the feature\u2019s condition, which the header owns', () => {
    // Feature -086 completed, and a later request moved it to failed_requires_human. The page
    // showed "Stopped -- needs an engineer" above a notice saying "This feature completed."
    // Both were true of different moments, which is why only the header states the condition.
    const notices = noticesFrom(
      [
        event(1, 'feature_completed'),
        event(2, 'feature_failed'),
        event(3, 'feature_cancelled'),
        event(4, 'feature_waiting_for_human'),
      ],
      new Set(),
    );

    expect(notices).toEqual([]);
  });

  it('says one thing per repository, however many times it is polled', () => {
    const notices = noticesFrom(
      [
        event(1, 'child_workflow_failed', 'backend'),
        event(2, 'child_workflow_failed', 'backend'),
        event(3, 'child_workflow_failed', 'frontend'),
      ],
      new Set(),
    );

    // One repository failing twice is one thing to look at; two repositories failing is two.
    expect(notices).toHaveLength(2);
    expect(notices.map((item) => item.repositoryId)).toEqual(['frontend', 'backend']);
  });

  it('lets later news supersede earlier news about the same subject', () => {
    const notices = noticesFrom(
      [
        event(1, 'child_workflow_failed', 'backend'),
        event(2, 'child_workflow_completed', 'backend'),
        event(3, 'child_workflow_failed', 'frontend'),
      ],
      new Set(),
    );

    // A repository that failed on one attempt and passed on the next is not both. Found
    // against real data: feature -086 completed and was reporting both its repositories as
    // stopped, from their first attempts.
    expect(notices.map((item) => `${item.repositoryId}:${item.message}`)).toEqual([
      'frontend:A repository stopped.',
      'backend:A repository finished.',
    ]);
  });

  it('ignores an event it does not recognise', () => {
    // A new internal step must not start shouting at people because nobody updated a list.
    expect(noticesFrom([event(1, 'some_new_internal_step')], new Set())).toEqual([]);
  });

  it('drops a notice somebody dismissed', () => {
    const events = [event(7, 'child_workflow_failed', 'backend')];
    expect(noticesFrom(events, new Set())).toHaveLength(1);
    expect(noticesFrom(events, new Set([7]))).toEqual([]);
  });
});
