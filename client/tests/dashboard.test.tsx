import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { DashboardPage } from '@/pages/DashboardPage';
import type { FeatureApi } from '@/api/features';
import { ApiError } from '@/api/errors';
import { FEATURE_PAGE, STATUS_VOCABULARY, featureSummary, stubApi } from './fixtures';

function renderDashboard(api: Partial<FeatureApi>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <DashboardPage />
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('DashboardPage', () => {
  it('renders features using the wording the server owns, not its own copy', async () => {
    renderDashboard({ listFeatures: async () => FEATURE_PAGE });

    const table = await screen.findByRole('table', { name: 'Features' });
    expect(within(table).getByText('Deactivate ad units from the console')).toBeInTheDocument();
    // `failed_requires_human` is exact and means nothing to a person.
    expect(within(table).getByText('Partly done — needs an engineer')).toBeInTheDocument();
    expect(within(table).queryByText('failed_requires_human')).not.toBeInTheDocument();
  });

  it('uses the dashboard category the server publishes', async () => {
    renderDashboard({ listFeatures: async () => FEATURE_PAGE });

    const tabs = await screen.findByRole('tablist', { name: 'Feature groups' });
    // One `attention`, one `done`, one `working` in the fixture page.
    expect(within(tabs).getByRole('tab', { name: 'Needs attention (1)' })).toBeInTheDocument();
    expect(within(tabs).getByRole('tab', { name: 'Running (1)' })).toBeInTheDocument();
    expect(within(tabs).getByRole('tab', { name: 'Completed (1)' })).toBeInTheDocument();
  });

  it('keeps failed and cancelled in separate PRD categories', async () => {
    renderDashboard({
      listFeatures: async () => ({
        features: [
          featureSummary({ status: 'failed', dashboard_group: 'failed' }),
          featureSummary({
            feature_id: 'cancelled-feature',
            workflow_id: 'cancelled-workflow',
            status: 'cancelled',
            dashboard_group: 'cancelled',
          }),
        ],
        next_cursor: null,
      }),
    });

    const tabs = await screen.findByRole('tablist', { name: 'Feature groups' });
    expect(within(tabs).getByRole('tab', { name: 'Failed (1)' })).toBeInTheDocument();
    expect(within(tabs).getByRole('tab', { name: 'Cancelled (1)' })).toBeInTheDocument();
  });

  it('filters to the features that need a person when that group is chosen', async () => {
    renderDashboard({ listFeatures: async () => FEATURE_PAGE });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('tab', { name: 'Needs attention (1)' }));

    expect(screen.getByText('Server uptime history on the admin console')).toBeInTheDocument();
    expect(screen.queryByText('Deactivate ad units from the console')).not.toBeInTheDocument();
  });

  it('searches by title, feature id, and workflow id', async () => {
    renderDashboard({ listFeatures: async () => FEATURE_PAGE });
    const user = userEvent.setup();
    const search = await screen.findByLabelText('Search');

    await user.type(search, 'uptime');
    expect(screen.getByText('Server uptime history on the admin console')).toBeInTheDocument();
    expect(screen.queryByText('Deactivate ad units from the console')).not.toBeInTheDocument();

    await user.clear(search);
    await user.type(search, 'live-086');
    expect(screen.getByText('Deactivate ad units from the console')).toBeInTheDocument();

    await user.clear(search);
    await user.type(search, 'adunit-deactivate-live-086');
    expect(screen.getByText('Deactivate ad units from the console')).toBeInTheDocument();
  });

  it('shows repository and pull-request counts without opening each feature', async () => {
    renderDashboard({ listFeatures: async () => FEATURE_PAGE });

    const table = await screen.findByRole('table', { name: 'Features' });
    // These come from the listing itself; fetching them per row would mean a request per feature.
    expect(within(table).getAllByTitle('2 repositories')).toHaveLength(3);
    // The one feature that has pull requests links to them by count.
    expect(within(table).getByRole('link', { name: '2' })).toBeInTheDocument();
  });

  it('keeps a status the vocabulary does not know rather than dropping the feature', async () => {
    renderDashboard({
      listFeatures: async () => ({
        features: [featureSummary({ status: 'a_status_added_later' })],
        next_cursor: null,
      }),
      getStatusVocabulary: async () => ({ feature: {}, workstream: {} }),
    });

    // The lifecycle grows on the server; an unknown status must not blank the dashboard.
    const table = await screen.findByRole('table', { name: 'Features' });
    expect(within(table).getByText('a_status_added_later')).toBeInTheDocument();
  });

  it('explains an API failure and offers a retry when retrying could help', async () => {
    renderDashboard({
      listFeatures: async () => {
        throw new ApiError('network', 'boom');
      },
    });

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not reach the platform API.');
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
  });

  it('does not offer a retry for a refusal that retrying cannot change', async () => {
    renderDashboard({
      listFeatures: async () => {
        throw new ApiError('conflict', 'nope', 409, 'feature is already running');
      },
    });

    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Try again' })).not.toBeInTheDocument();
  });

  it('renders an empty state rather than blank space', async () => {
    renderDashboard({ listFeatures: async () => ({ features: [], next_cursor: null }) });

    expect(await screen.findByText('No features yet')).toBeInTheDocument();
  });

  it('says how many rows the search actually covers', async () => {
    renderDashboard({ listFeatures: async () => FEATURE_PAGE });

    // The list endpoint is cursor-paged with no search, so the UI must not imply it searched
    // everything that exists.
    expect(await screen.findByText('Searching 3 loaded features.')).toBeInTheDocument();
  });

  it('exposes the vocabulary headline in the status filter', async () => {
    renderDashboard({ listFeatures: async () => FEATURE_PAGE });

    // The options are the statuses actually present in what loaded, so wait for the rows.
    await screen.findByRole('table', { name: 'Features' });
    const select = screen.getByLabelText('Status');
    expect(within(select).getByRole('option', { name: 'Finished' })).toBeInTheDocument();
    expect(STATUS_VOCABULARY.feature.completed.headline).toBe('Finished');
  });
});
