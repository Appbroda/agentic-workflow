// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { RepositoryPage } from '@/pages/RepositoryPage';
import { workstreamsSchema, type Workstream } from '@/schemas/feature';
import type { FeatureApi } from '@/api/features';
import { stubApi } from './fixtures';
import saved from './fixtures/workstreams.adunit-deactivate-live-086.json';
import blocked from './fixtures/workstreams.admin-server-status-live-038.json';

/**
 * The repository drill-down, rendered from a payload the running platform actually produced.
 *
 * A fixture written by hand agrees with whatever its author believed the server sends; these
 * are saved responses from features -086 and -038 on the live stack, checked for credentials
 * before being committed. A field the server fills and this page drops shows up here as a
 * missing assertion rather than as a passing test.
 *
 * Refresh them with:
 *   curl -H "Authorization: Bearer $PLATFORM_API_KEY" \
 *     localhost:8000/features/<id>/workstreams > tests/fixtures/workstreams.<id>.json
 */

function feature(featureId: string) {
  return {
    feature_id: featureId,
    workflow_id: featureId,
    status: 'failed_requires_human',
    title: 'A feature with real repositories',
    current_agent: null,
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
    available_actions: [],
    created_at: '2026-08-24T22:40:00Z',
    updated_at: '2026-08-24T23:20:00Z',
  };
}

function renderRepository(featureId: string, workstreams: Workstream[], view?: string) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const api: Partial<FeatureApi> = {
    getFeature: async () => feature(featureId),
    getWorkstreams: async () => ({ feature_id: featureId, workstreams }),
    listEvents: async () => ({ feature_id: featureId, events: [], last_event_id: null }),
    getTimeline: async () => ({ feature_id: featureId, events: [] }),
    listArtifacts: async () => ({ feature_id: featureId, artifacts: [] }),
    getPullRequests: async () => ({ feature_id: featureId, pull_requests: [] }),
    listRepairs: async () => ({ feature_id: featureId, repairs: [] }),
  };
  const repositoryId = workstreams[0]!.repository_id;
  const path = `/features/${featureId}/repositories/${repositoryId}${view ? `?view=${view}` : ''}`;
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[path]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route
            path="/features/:featureId/repositories/:repositoryId"
            element={<RepositoryPage />}
          />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('a real repository workstream', () => {
  it('shows what it was asked to do, and what it is built from', async () => {
    const data = workstreamsSchema.parse(saved);
    renderRepository(data.feature_id, data.workstreams);

    // Detected from the checkout, and the reason the platform chose the commands it did.
    expect(await screen.findByText('Detected technology')).toBeInTheDocument();
    expect(screen.getByText('Language')).toBeInTheDocument();
    expect(screen.getAllByText(/JavaScript|TypeScript/).length).toBeGreaterThan(0);
    expect(screen.getByText('Frameworks')).toBeInTheDocument();

    // What this repository was asked to do, as distinct from what it did. A review rejection
    // reads differently once you can see which requirements were this repository's.
    expect(screen.getByText('Assigned work')).toBeInTheDocument();
    expect(screen.getAllByText(/Acceptance criteria:/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Expected to touch:/).length).toBeGreaterThan(0);
  });

  it('reaches the exact command each validation check ran', async () => {
    const data = workstreamsSchema.parse(saved);
    renderRepository(data.feature_id, data.workstreams, 'validation');

    // An engineer's first question about a repository is which check ran and how it ended.
    const table = await screen.findByRole('table', { name: 'Validation checks' });
    expect(screen.getAllByText(/npm run/).length).toBeGreaterThan(0);
    const results = data.workstreams[0]!.current_validation_results;
    expect(within(table).getAllByRole('row')).toHaveLength(results.length + 1);
  });

  it('opens one check to its output, working directory and exit code', async () => {
    const data = workstreamsSchema.parse(saved);
    renderRepository(data.feature_id, data.workstreams, 'validation');
    const user = userEvent.setup();

    const table = await screen.findByRole('table', { name: 'Validation checks' });
    await user.click(within(table).getAllByRole('row')[1]!);

    // The drawer is where the raw evidence lives: summary first, then the exact command, then
    // whatever the platform captured of the output.
    const drawer = await screen.findByRole('dialog');
    expect(within(drawer).getByText('Exit code')).toBeInTheDocument();
    expect(within(drawer).getByText('Command')).toBeInTheDocument();
    expect(within(drawer).getByText('Standard output')).toBeInTheDocument();
  });
});

describe('a repository the platform could not work in', () => {
  it('states the diagnosis, the suggested repair, and who has to make it', async () => {
    const data = workstreamsSchema.parse(blocked);
    const stopped = data.workstreams.filter((item) => item.blocking_setup_issues.length > 0);
    renderRepository(
      data.feature_id,
      // The saved payload predates backend-authoritative action availability.
      stopped.map((item) => ({ ...item, available_actions: ['RETRY_WORKSTREAM'] })),
    );

    // A checked-in setup that cannot run its own checks is a different failure from code that
    // was wrong: nothing written here could have been validated, so saying only "it failed"
    // sends somebody to read an implementation that was never the problem.
    expect(await screen.findByText(/could not run its own checks/)).toBeInTheDocument();
    // Asserted from the fixture rather than against a hardcoded id, so refreshing it from a
    // different blocked feature does not turn a passing check into a false failure.
    const issue = stopped[0]!.blocking_setup_issues[0]!;
    expect(screen.getByText(String(issue.issue_id))).toBeInTheDocument();
    expect(
      screen.getAllByText(new RegExp(escapeForRegExp(String(issue.description)))).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText(/Suggested repair:/).length).toBeGreaterThan(0);
    // This fixture's findings produced no repair proposal, so the platform says why rather
    // than leaving somebody to wonder how to unstick it. The approvable case is covered in
    // `repair.test.tsx`, against a real proposal.
    expect(screen.getAllByText(/proposed no repair for this/).length).toBeGreaterThan(0);
    // And the action that does exist is offered beside it.
    expect(screen.getAllByRole('button', { name: 'Grant another attempt' }).length).toBeGreaterThan(0);
  });
});

describe('a repository whose correction was routed to the scoped-fix model', () => {
  it('says which role is doing the work and why it was eligible', async () => {
    const data = workstreamsSchema.parse(saved);
    // The saved payload predates model routing, so the decision is added the way the server
    // now sends it. Everything else is the real response.
    const routed = data.workstreams.map((item, index) =>
      index === 0
        ? {
            ...item,
            model_routing: {
              execution_mode: 'REVIEW_REMEDIATION',
              role: 'scoped_fix',
              model: 'a-configured-model',
              reasoning: 'high',
              classification: 'STANDARD',
              attempt: 2,
              escalation_level: 0,
              escalated: false,
              previous_role: null,
              finding_ids: ['REV-003'],
              routing_reason:
                'Every blocking finding is a localized, independently verifiable correction.',
            },
          }
        : item,
    );
    renderRepository(data.feature_id, routed);

    expect(await screen.findByText('Model routing')).toBeInTheDocument();
    expect(screen.getByText('Correcting review findings')).toBeInTheDocument();
    expect(screen.getByText('Scoped fix')).toBeInTheDocument();
    expect(screen.getByText('STANDARD')).toBeInTheDocument();
    expect(screen.getByText(/localized, independently verifiable/)).toBeInTheDocument();
  });

  it('shows nothing at all for a workstream the server sent no decision for', async () => {
    const data = workstreamsSchema.parse(saved);
    renderRepository(data.feature_id, data.workstreams);

    // A panel with an empty routing table would imply the platform routed nothing, which is
    // not the same as a payload written before routing existed.
    expect(await screen.findByText('Detected technology')).toBeInTheDocument();
    expect(screen.queryByText('Model routing')).not.toBeInTheDocument();
  });
});

/** Escape a fixture string so it can be matched literally. */
function escapeForRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

describe('a repository whose plan was written blind', () => {
  it('says so beside the workstream, with the sanitized reason', async () => {
    // The saved payload predates the flag; the parse applies the schema default (false),
    // and the blind case is the same real workstream with the two new fields the server
    // now writes when the reconnaissance fail-soft fires.
    const data = workstreamsSchema.parse(saved);
    const blindWorkstream: Workstream = {
      ...data.workstreams[0]!,
      planned_blind: true,
      planned_blind_reason: 'the model provider call failed (ReadTimeout)',
    };
    renderRepository(data.feature_id, [blindWorkstream]);

    expect(await screen.findByText('Reconnaissance')).toBeInTheDocument();
    expect(screen.getByText('Planned without checkout evidence')).toBeInTheDocument();
  });

  it('is not mentioned at all for a workstream planned with evidence', async () => {
    const data = workstreamsSchema.parse(saved);
    expect(data.workstreams[0]!.planned_blind).toBe(false);
    renderRepository(data.feature_id, data.workstreams);

    await screen.findByText('Detected technology');
    expect(screen.queryByText('Planned without checkout evidence')).not.toBeInTheDocument();
  });
});
