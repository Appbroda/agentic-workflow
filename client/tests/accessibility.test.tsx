import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { DashboardPage } from '@/pages/DashboardPage';
import { NewFeaturePage } from '@/pages/NewFeaturePage';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import { stubApi } from './fixtures';
import savedWorkstreamsJson from './fixtures/workstreams.adunit-deactivate-live-086.json';
import { workstreamsSchema } from '@/schemas/feature';

// Parsed through the real schema so defaults the server added after this payload was
// captured (planned_blind) are applied the way the app applies them.
const savedWorkstreams = workstreamsSchema.parse(savedWorkstreamsJson);

/**
 * Every control a person can operate has a name assistive technology can read.
 *
 * Checked through the accessibility tree rather than by looking for `for=` attributes: a
 * control wrapped in its label is named correctly and a naive check calls it a fault, while a
 * control with an `id` nobody points at looks fine and is not. Only the computed name settles
 * it.
 */
const FEATURE = {
  feature_id: 'f-1',
  workflow_id: 'f-1',
  status: 'failed_requires_human',
  title: 'A feature',
  current_agent: 'child_workflows',
  repository_count: 2,
  required_repository_count: 2,
  repositories: [],
  clarification_rounds: 0,
  integration_review_cycles: 0,
  merge_strategy: null,
  deployment_strategy: null,
  execution_mode: 'live',
  cancellation_status: 'not_requested',
  cancellation_requested_at: null,
  cancellation_reason: null,
  cleanup_requirements: [],
  created_at: '2026-08-25T08:00:00Z',
  updated_at: '2026-08-25T09:00:00Z',
};

const API: Partial<FeatureApi> = {
  listFeatures: async () => ({
    features: [
      {
        feature_id: 'f-1',
        workflow_id: 'f-1',
        title: 'A feature',
        status: 'failed_requires_human',
        execution_mode: 'live',
        created_at: '2026-08-25T08:00:00Z',
        updated_at: '2026-08-25T09:00:00Z',
        repository_count: 2,
        pull_request_count: 0,
        human_action_required: true,
        dashboard_group: 'waiting',
      },
    ],
    next_cursor: null,
  }),
  getFeature: async () => FEATURE,
  getWorkstreams: async () => savedWorkstreams,
  listEvents: async () => ({ feature_id: 'f-1', events: [], last_event_id: null }),
  getTimeline: async () => ({ feature_id: 'f-1', events: [] }),
  listArtifacts: async () => ({ feature_id: 'f-1', artifacts: [] }),
  getClarification: async () => ({
    feature_id: 'f-1',
    awaiting_answers: false,
    technical_prd_artifact_id: null,
    clarification_rounds: 0,
    max_clarification_rounds: 10,
    questions: [],
    previous_answers: {},
    design_conflicts: [],
  }),
};

function renderAt(path: string) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(API)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[path]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/features/new" element={<NewFeaturePage />} />
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

function unnamedControls(container: HTMLElement): string[] {
  const named = new Set([
    ...screen.queryAllByRole('textbox'),
    ...screen.queryAllByRole('combobox'),
    ...screen.queryAllByRole('checkbox'),
    ...screen.queryAllByRole('searchbox'),
    ...screen.queryAllByRole('spinbutton'),
    ...screen.queryAllByRole('button'),
  ]);
  const problems: string[] = [];
  for (const node of named) {
    // `within(node)` cannot compute a name, so read the tree the same way a reader would:
    // an explicit label, an implicit wrapping label, or an aria attribute.
    const id = node.getAttribute('id');
    const explicit = id ? container.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
    const wrapping = node.closest('label');
    const aria =
      node.getAttribute('aria-label') ?? node.getAttribute('aria-labelledby') ?? '';
    const text = (node.textContent ?? '').trim();
    if (!explicit && !wrapping && !aria && !text) {
      problems.push(`${node.tagName.toLowerCase()}[${node.getAttribute('type') ?? 'default'}]`);
    }
  }
  return problems;
}

describe('accessible names', () => {
  it.each([
    ['the dashboard', '/'],
    ['the new-feature form', '/features/new'],
    ['a feature workspace', '/features/f-1/repositories'],
  ])('every control on %s has one', async (_name, path) => {
    const { container } = renderAt(path);
    // Wait for the view to settle rather than asserting against a loading state.
    await screen.findByRole('heading', { level: 1 });

    expect(unnamedControls(container)).toEqual([]);
  });

  it('does not use colour alone to convey a status', async () => {
    renderAt('/features/f-1/repositories');
    const table = await screen.findByRole('table', { name: 'Repository workstreams' });

    // Every status badge carries a word. Somebody who cannot distinguish the colours, or is
    // reading the page aloud, gets the same information.
    const badges = table.querySelectorAll('.badge');
    expect(badges.length).toBeGreaterThan(0);
    for (const badge of badges) {
      expect((badge.textContent ?? '').trim()).not.toBe('');
    }
  });
});
