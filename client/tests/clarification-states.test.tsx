import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import { clarificationSchema } from '@/schemas/feature';
import { stubApi } from './fixtures';
import investigatingJson from './fixtures/clarification.investigating.json';
import awaitingJson from './fixtures/clarification.awaiting-answers.json';
import groundingFailedJson from './fixtures/clarification.asked-after-grounding-failure.json';

/**
 * The clarification surface's three states, each rendered from a payload the real
 * `/features/{id}/clarification` route produced.
 *
 * AB-Feature-173's panel polled that route fifty times over thirty-five minutes and rendered
 * "waiting for you" while the platform was investigating the questions itself; when it did
 * finally need the human, the reason -- grounding had failed -- was invisible. The saved
 * payloads were captured from the route serving real orchestrator runs (a premise question a
 * checkout contradicted; a grounding call that failed the way 173's and 174's did), and each
 * is parsed through the real zod schema so a dropped field fails here rather than in a
 * browser.
 *
 * Regenerate against a running stack with:
 *   curl -H "Authorization: Bearer $PLATFORM_API_KEY" \
 *     localhost:8000/features/<id>/clarification > tests/fixtures/clarification.<state>.json
 */

const investigating = clarificationSchema.parse(investigatingJson);
const awaiting = clarificationSchema.parse(awaitingJson);
const groundingFailed = clarificationSchema.parse(groundingFailedJson);

const QUESTION = 'Rely on the global middleware, or introduce per-route auth?';

function feature(status: string) {
  return {
    feature_id: investigating.feature_id,
    workflow_id: investigating.feature_id,
    status,
    title: 'Login audit trail',
    current_agent: null,
    repository_count: 2,
    required_repository_count: 2,
    repositories: [],
    clarification_rounds: 0,
    integration_review_cycles: 0,
    merge_strategy: null,
    deployment_strategy: null,
    execution_mode: 'mock',
    cancellation_status: 'not_requested',
    cancellation_requested_at: null,
    cancellation_reason: null,
    cleanup_requirements: [],
    available_actions: [],
    created_at: '2026-08-31T06:03:22Z',
    updated_at: '2026-08-31T06:10:19Z',
  };
}

function renderWorkspace(status: string, clarification: typeof investigating) {
  const api: Partial<FeatureApi> = {
    getFeature: async () => feature(status),
    getWorkstreams: async () => ({ feature_id: clarification.feature_id, workstreams: [] }),
    getTimeline: async () => ({ feature_id: clarification.feature_id, events: [] }),
    listEvents: async () => ({
      feature_id: clarification.feature_id,
      events: [],
      last_event_id: null,
    }),
    getClarification: async () => clarification,
  };
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[`/features/${clarification.feature_id}`]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('the clarification surface tells the truth about whose move it is', () => {
  it('renders investigating as the platform’s own open items, not an ask', async () => {
    expect(investigating.clarification_state).toBe('investigating');
    expect(investigating.awaiting_answers).toBe(false);
    renderWorkspace('analyzing_prd', investigating);

    expect(
      await screen.findByRole('heading', { name: 'Open questions the platform is investigating' }),
    ).toBeInTheDocument();
    expect(screen.getByText('Not waiting on you')).toBeInTheDocument();
    // The questions are visible -- 173's sat unrendered for 35 minutes -- but nothing is
    // actionable: no answer form, no submit.
    expect(screen.getByText(QUESTION)).toBeInTheDocument();
    expect(screen.queryByText('This feature is waiting on you')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Submit answers/i })).not.toBeInTheDocument();
  });

  it('renders awaiting_answers as today’s answer form, with no fallback notice', async () => {
    expect(awaiting.clarification_state).toBe('awaiting_answers');
    renderWorkspace('waiting_for_human', awaiting);

    expect(await screen.findByText('This feature is waiting on you')).toBeInTheDocument();
    expect(screen.getByText(QUESTION)).toBeInTheDocument();
    expect(
      screen.queryByText('The platform tried to answer these itself'),
    ).not.toBeInTheDocument();
  });

  it('says when the human is the fallback for a failed grounding attempt', async () => {
    expect(groundingFailed.clarification_state).toBe('asked_after_grounding_failure');
    expect(groundingFailed.awaiting_answers).toBe(true);
    renderWorkspace('waiting_for_human', groundingFailed);

    expect(await screen.findByText('This feature is waiting on you')).toBeInTheDocument();
    expect(screen.getByText('The platform tried to answer these itself')).toBeInTheDocument();
    expect(screen.getByText(QUESTION)).toBeInTheDocument();
  });
});
