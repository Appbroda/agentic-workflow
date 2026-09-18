// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import type { LogbookEntry } from '@/schemas/feature';
import { groupByDay, recordHref } from '@/features/feature-workspace/logbook';
import { stubApi } from './fixtures';

/**
 * The feature tells its story (57): a Logbook tab beside History that renders one run as a
 * conversation, every bubble attributed and every bubble anchored.
 *
 * The tests that decide correctness here are the anchoring one -- a bubble whose record the
 * reader cannot open is the failure mode this tab exists to avoid -- and the one that pins
 * where the sentences come from: the server composes all of them, and nothing in this client
 * may compose one, because a browser-side template is a narrator by another name.
 */

const FEATURE_ID = 'adunit-deactivate-live-184';

const FEATURE = {
  feature_id: FEATURE_ID,
  workflow_id: 'workflow-184',
  status: 'failed_requires_human',
  title: 'Bulk add apps',
  current_agent: 'publisher',
  repository_count: 2,
  required_repository_count: 2,
  repositories: [],
  clarification_rounds: 0,
  integration_review_cycles: 1,
  max_clarification_rounds: 10,
  max_integration_review_cycles: 5,
  max_child_review_cycles: 8,
  max_implementation_retries: 4,
  max_validation_retries: 2,
  max_repository_setup_retries: 1,
  merge_strategy: null,
  deployment_strategy: null,
  execution_mode: 'live',
  cancellation_status: 'not_requested',
  cancellation_requested_at: null,
  cancellation_reason: null,
  cleanup_requirements: [],
  available_actions: [],
  created_at: '2026-09-01T04:10:00Z',
  updated_at: '2026-09-01T04:55:00Z',
};

let sequence = -1;

/** One bubble as the endpoint serves it. Nothing here is derived; the server composed it. */
function entry(overrides: Partial<LogbookEntry> = {}): LogbookEntry {
  sequence += 1;
  return {
    sequence,
    emission: 0,
    timestamp: '2026-09-01T04:20:00Z',
    agent: 'Reviewer',
    tone: 'done',
    template: 'artifact.review.approved',
    text: 'The reviewer approved admanager-server.',
    detail: null,
    quote: null,
    quote_source: null,
    record: { kind: 'artifact', id: '007_review.backend.json', repository_id: 'backend' },
    repository_id: 'backend',
    ...overrides,
  };
}

/** The 184 thread, in the shape the server composes it from that run's records. */
function story(): LogbookEntry[] {
  sequence = -1;
  return [
    entry({
      agent: 'Person',
      tone: 'working',
      template: 'artifact.prd',
      text: 'Somebody asked for this feature: Bulk add apps.',
      quote: 'Operators add apps one at a time and it takes all afternoon.',
      quote_source: 'problem_statement',
      record: { kind: 'artifact', id: '001_prd.json', repository_id: null },
      repository_id: null,
      timestamp: '2026-09-01T04:10:00Z',
    }),
    entry(),
    entry({
      agent: 'Integration reviewer',
      tone: 'attention',
      template: 'artifact.integration_review.required_fix',
      text: 'The fix it required of admanager-server was this.',
      quote:
        'Make POST /apps/bulk atomic: either every app in the batch is created or none is, ' +
        'and the response says which.',
      quote_source: 'required_fixes[]',
      record: { kind: 'artifact', id: '010_integration_review.json', repository_id: 'backend' },
      timestamp: '2026-09-01T04:40:00Z',
    }),
    entry({
      agent: 'Platform',
      tone: 'stopped',
      template: 'operation.failed',
      text: 'Failed while opening a pull request against admanager-server.',
      detail: 'The platform recorded this as pull_request_unverified. Attempt 2 of 3.',
      record: { kind: 'operation', id: 'op-312183e4', repository_id: 'backend' },
      timestamp: '2026-09-01T04:44:00Z',
    }),
    entry({
      agent: 'Orchestrator',
      tone: 'stopped',
      template: 'event.feature_failed',
      text:
        'The feature stopped: the platform itself failed while running it, rather than ' +
        'anything being wrong with the repository or the request.',
      quote:
        'Publication was refused because the workspace content no longer matches the ' +
        'source evidence approved by the reviewer.',
      quote_source: 'details.diagnostics[0]',
      record: { kind: 'event', id: '412', repository_id: null },
      repository_id: null,
      // The next calendar day, so the thread has to break for a date header.
      timestamp: '2026-09-02T04:45:00Z',
    }),
  ];
}

function renderLogbook(entries: LogbookEntry[], overrides: Partial<FeatureApi> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const api: Partial<FeatureApi> = {
    listEvents: async () => ({ feature_id: FEATURE_ID, events: [], last_event_id: null }),
    getFeature: async () => FEATURE,
    getWorkstreams: async () => ({ feature_id: FEATURE_ID, workstreams: [] }),
    getPullRequests: async () => ({ feature_id: FEATURE_ID, pull_requests: [] }),
    getLogbook: async (featureId: string) => ({
      feature_id: featureId,
      entries,
      next_cursor: null,
      agents: ['Orchestrator', 'Reviewer', 'Platform', 'Person', 'Integration reviewer'],
    }),
    ...overrides,
  } as Partial<FeatureApi>;
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[`/features/${FEATURE_ID}/logbook`]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

async function thread(): Promise<HTMLElement[]> {
  await screen.findByText('The reviewer approved admanager-server.');
  return screen
    .getAllByRole('list')
    .filter((item) => (item.getAttribute('aria-label') ?? '').startsWith('Logbook,'))
    .flatMap((list) => within(list).getAllByRole('listitem'));
}

describe('the Logbook tab', () => {
  it('is a section of its own, beside History', async () => {
    renderLogbook(story());
    const tabs = await screen.findByRole('navigation', { name: 'Feature sections' });
    const names = within(tabs)
      .getAllByRole('link')
      .map((item) => item.textContent);
    expect(names).toContain('Logbook');
    expect(names).toContain('History');
    expect(names.indexOf('Logbook')).toBeLessThan(names.indexOf('History'));
  });

  it('reads as a run somebody who does not work here could follow', async () => {
    renderLogbook(story());
    const bubbles = await thread();

    // What was asked for, who approved what, what was required, and where it died -- in the
    // order the server placed them, which is the order it happened.
    expect(bubbles.map((item) => item.textContent?.split('\n')[0])).not.toContain(undefined);
    expect(await screen.findByText(/Somebody asked for this feature/)).toBeTruthy();
    expect(screen.getByText('The reviewer approved admanager-server.')).toBeTruthy();
    expect(screen.getByText(/The fix it required of admanager-server/)).toBeTruthy();
    expect(screen.getByText(/the platform itself failed while running it/)).toBeTruthy();
    // Every bubble says who is speaking.
    for (const bubble of bubbles) {
      const agents = ['Person', 'Reviewer', 'Integration reviewer', 'Platform', 'Orchestrator'];
      expect(agents.some((agent) => bubble.textContent?.includes(agent))).toBe(true);
    }
  });

  it('anchors every bubble to a record a reader can open', async () => {
    renderLogbook(story());
    const bubbles = await thread();
    expect(bubbles).toHaveLength(5);
    // A4, on this side of the wire: the record reference the server sent becomes a real link
    // to the surface that shows that record, for every single bubble.
    const destinations = bubbles.map((bubble) => {
      const link = within(bubble)
        .getAllByRole('link')
        .find((item) => /^View /.test(item.textContent ?? ''));
      expect(link, `bubble "${bubble.textContent?.slice(0, 40)}" has no record link`).toBeDefined();
      return link!.getAttribute('href');
    });
    expect(destinations).toEqual([
      `/features/${FEATURE_ID}/artifacts?artifact=001_prd.json`,
      `/features/${FEATURE_ID}/artifacts?artifact=007_review.backend.json`,
      `/features/${FEATURE_ID}/artifacts?artifact=010_integration_review.json`,
      `/features/${FEATURE_ID}/repositories/backend`,
      `/features/${FEATURE_ID}/history`,
    ]);
  });

  it('sets the agents’ own words apart from the platform’s sentence about them', async () => {
    renderLogbook(story());
    await thread();
    const quote = screen.getByText(/Make POST \/apps\/bulk atomic/);
    // A quotation, not a paragraph: the reader has to be able to tell what the platform said
    // from what a model said, and the field it was read from is stated beside it.
    expect(quote.tagName).toBe('BLOCKQUOTE');
    expect(within(quote).getByText('required_fixes[]')).toBeTruthy();
  });

  it('breaks the thread where the calendar does', async () => {
    renderLogbook(story());
    await thread();
    const dates = screen.getAllByRole('heading', { level: 3 });
    expect(dates).toHaveLength(2);
    expect(dates[0]!.textContent).not.toEqual(dates[1]!.textContent);
  });

  it('says so when a run is too long for one page', async () => {
    renderLogbook(story(), {
      getLogbook: async (featureId: string) => ({
        feature_id: featureId,
        entries: story(),
        next_cursor: 4,
        agents: [],
      }),
    });
    await thread();
    expect(screen.getByText(/longer than one page/)).toBeTruthy();
  });

  it('has nothing to say about a feature that has not started', async () => {
    renderLogbook([]);
    expect(await screen.findByText('Nothing has happened yet')).toBeTruthy();
  });

  it('renders a bubble whose record kind this client has never heard of', async () => {
    renderLogbook([
      entry({
        template: 'record.kind.added.later',
        text: 'The platform recorded something new.',
        record: { kind: 'inbox_message', id: 'msg-1', repository_id: null },
        repository_id: null,
      }),
    ]);
    const bubbles = await screen.findByText('The platform recorded something new.');
    expect(bubbles).toBeTruthy();
    // Still anchored, still shown -- it simply has no page here yet. Never dropped.
    expect(screen.getByText('msg-1')).toBeTruthy();
  });
});

describe('reading the thread', () => {
  it('sends each record kind to the surface that already shows it', () => {
    const href = (kind: string, repository: string | null = 'backend') =>
      recordHref('f 1', { kind, id: 'a/b', repository_id: repository });
    expect(href('artifact')).toBe('/features/f%201/artifacts?artifact=a%2Fb');
    expect(href('operation')).toBe('/features/f%201/repositories/backend');
    expect(href('workstream')).toBe('/features/f%201/repositories/backend');
    expect(href('event')).toBe('/features/f%201/history');
    // A journal row with no repository is a feature-level operation: the workflow view owns
    // those, and guessing a repository for one would badge it with somebody else's work.
    expect(href('operation', null)).toBe('/features/f%201/workflow');
    expect(href('something_new')).toBeNull();
  });

  it('groups consecutive days and never reorders inside one', () => {
    const days = groupByDay([
      entry({ timestamp: '2026-09-01T23:00:00Z' }),
      entry({ timestamp: '2026-09-01T23:30:00Z' }),
      entry({ timestamp: '2026-09-03T01:00:00Z' }),
    ]);
    expect(days).toHaveLength(2);
    expect(days[0]!.entries).toHaveLength(2);
    expect(days[1]!.entries).toHaveLength(1);
    expect(days.flatMap((day) => day.entries).map((item) => item.timestamp)).toEqual([
      '2026-09-01T23:00:00Z',
      '2026-09-01T23:30:00Z',
      '2026-09-03T01:00:00Z',
    ]);
  });
});
