// @vitest-environment node
//
// Node, not jsdom: jsdom enforces the browser's same-origin rules, and the server has no
// CORS middleware on purpose -- in a browser the client reaches it through a same-origin
// proxy. Running this in jsdom tests the proxy's absence rather than the schemas.
import { describe, expect, it } from 'vitest';
import { ApiClient } from '@/api/client';
import { createFeatureApi } from '@/api/features';

/**
 * Reads every client-facing endpoint on a running backend and validates the response with the
 * schemas the application actually uses.
 *
 * The unit tests prove the components behave; they cannot prove the schemas match the server,
 * because their fixtures were written from the same reading of it. Schema drift is the one
 * defect that unit tests structurally cannot catch, and it shows up as a blank page.
 *
 * Skipped unless a backend is configured:
 *
 *   LIVE_API_URL=http://localhost:8000 LIVE_API_KEY=... npx vitest run tests/live-backend.test.ts
 */
const baseUrl = process.env.LIVE_API_URL;
const apiKey = process.env.LIVE_API_KEY;
const live = baseUrl && apiKey ? describe : describe.skip;

live('the client schemas against a running backend', () => {
  const api = createFeatureApi(
    new ApiClient({ baseUrl: baseUrl!, token: apiKey!, defaultTimeoutMs: 60_000 }),
  );

  it('reads the dashboard, then every panel of one real feature', async () => {
    const list = await api.listFeatures({ limit: 10 });
    expect(list.features.length).toBeGreaterThan(0);

    // Whichever feature the backend happens to hold. Pinning an id would make this pass or
    // fail on the state of one database rather than on whether the schemas fit.
    const featureId = list.features[0]!.feature_id;

    const [feature, workstreams, timeline, clarification, artifacts, pullRequests, events] =
      await Promise.all([
        api.getFeature(featureId),
        api.getWorkstreams(featureId),
        api.getTimeline(featureId),
        api.getClarification(featureId),
        api.listArtifacts(featureId, { includePayload: false }),
        api.getPullRequests(featureId),
        api.listEvents(featureId, null),
      ]);

    expect(feature.feature_id).toBe(featureId);
    expect(timeline.feature_id).toBe(featureId);
    expect(clarification.feature_id).toBe(featureId);
    expect(pullRequests.feature_id).toBe(featureId);
    expect(events.feature_id).toBe(featureId);
    // Every workstream parsed, including the retry-grant fields added for the retry control.
    for (const item of workstreams.workstreams) {
      expect(typeof item.granted_extra_attempts).toBe('number');
      expect(Array.isArray(item.retry_grants)).toBe(true);
    }

    // The dashboard renders these three; they were added to the list endpoint for it.
    expect(typeof list.features[0]!.repository_count).toBe('number');
    expect(typeof list.features[0]!.pull_request_count).toBe('number');
    expect(typeof list.features[0]!.human_action_required).toBe('boolean');
    expect(list.features[0]!.workflow_id).toBeTruthy();
    expect(['queued', 'running', 'waiting', 'failed', 'completed', 'cancelled']).toContain(
      list.features[0]!.dashboard_group,
    );
    // The identity the whole console leads with. Every feature has one after the backfill,
    // so a null here means the migration did not run against this database.
    expect(list.features[0]!.reference).toMatch(/^AB-Feature-\d+$/);

    // An artifact fetched singly must parse too: the list omits payloads, so the renderers
    // only ever see this shape.
    if (artifacts.artifacts.length > 0) {
      const one = await api.getArtifact(featureId, artifacts.artifacts[0]!.artifact_id);
      expect(one.artifact_id).toBe(artifacts.artifacts[0]!.artifact_id);
    }

    // Status wording and the lifecycle-aware dashboard category are both server-owned.
    const vocabulary = await api.getStatusVocabulary();
    expect(Object.keys(vocabulary)).toContain('feature');

    // The three reads the New Feature gate and Settings depend on. They are newer than
    // everything above, so schema drift here is the most likely kind.
    const setup = await api.getSetupState();
    expect(typeof setup.credentials_ready).toBe('boolean');
    expect(setup.providers.map((item) => item.provider).sort()).toEqual(['github', 'openai']);
    const saved = await api.listSavedRepositories();
    expect(Array.isArray(saved.repositories)).toBe(true);
    expect(saved.suggested_types.length).toBeGreaterThan(0);
  }, 120_000);

  it('is refused, with a reason, when it asks for something impossible', async () => {
    const list = await api.listFeatures({ limit: 10 });
    const featureId = list.features[0]!.feature_id;

    // A repository that is not part of the feature. This must be an answer -- 409 with a
    // reason -- and must not disturb the feature it names.
    const before = await api.getFeature(featureId);
    await expect(
      api.retryWorkstream(featureId, 'definitely-not-a-repository', {
        additional_attempts: 1,
        requested_by: 'integration-test',
        reason: 'Checking that the platform refuses this.',
      }),
    ).rejects.toMatchObject({ kind: 'conflict', status: 409 });

    const after = await api.getFeature(featureId);
    expect(after.status).toBe(before.status);
  }, 120_000);
});
