import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { AppLayout } from '@/app/AppLayout';
import { DashboardPage } from '@/pages/DashboardPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { clearToken, readToken, writeToken } from '@/api/token';
import type { FeatureApi } from '@/api/features';
import { FEATURE_PAGE, featureSummary, savedRepository, stubApi } from './fixtures';

/**
 * The refinements this pass made, tested where they are visible.
 *
 * Four things: the feature reference leading the table and being searchable, a queued feature
 * being counted apart from a running one, Settings holding only what somebody can configure,
 * and signing out clearing the session without touching what the account owns.
 */

beforeEach(() => writeToken('test-token'));

function renderApp(path: string, api: Partial<FeatureApi> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={[path]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<DashboardPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('the feature reference', () => {
  it('leads every row and is what search matches', async () => {
    renderApp('/', { listFeatures: async () => FEATURE_PAGE });
    const user = userEvent.setup();

    const table = await screen.findByRole('table', { name: 'Features' });
    expect(within(table).getByText('AB-Feature-86')).toBeInTheDocument();
    // The internal id is not what leads: the pilot features are named things like
    // `adunit-deactivate-live-086`, which is unique and meaningless.
    expect(within(table).queryByText('adunit-deactivate-live-086')).not.toBeInTheDocument();

    await user.type(screen.getByLabelText('Search'), 'AB-Feature-82');

    await waitFor(() =>
      expect(
        screen.getByText('Server uptime history on the admin console'),
      ).toBeInTheDocument(),
    );
    expect(
      screen.queryByText('Deactivate ad units from the console'),
    ).not.toBeInTheDocument();
  });

  it('counts a queued feature apart from a running one', async () => {
    renderApp('/', {
      listFeatures: async () => ({
        features: [
          featureSummary({
            feature_id: 'just-submitted',
            reference: 'AB-Feature-96',
            title: 'Submitted a moment ago',
            status: 'pending',
            dashboard_group: 'queued',
            pull_request_count: 0,
          }),
          ...FEATURE_PAGE.features,
        ],
        next_cursor: null,
      }),
    });
    const user = userEvent.setup();

    const groups = await screen.findByRole('tablist', { name: 'Feature groups' });
    // The distinction somebody watching a fresh submission cares about: accepted, but not
    // yet picked up.
    expect(within(groups).getByRole('tab', { name: 'Queued (1)' })).toBeInTheDocument();
    expect(within(groups).getByRole('tab', { name: 'Running (1)' })).toBeInTheDocument();

    await user.click(within(groups).getByRole('tab', { name: 'Queued (1)' }));

    const table = screen.getByRole('table', { name: 'Features' });
    expect(within(table).getByText('Submitted a moment ago')).toBeInTheDocument();
    expect(within(table).queryByText('Something still running')).not.toBeInTheDocument();
  });
});

describe('the account menu', () => {
  it('shows the signed-in identity, and offers Settings and signing out', async () => {
    renderApp('/', { listFeatures: async () => FEATURE_PAGE });
    const user = userEvent.setup();

    const trigger = await screen.findByRole('button', { name: /Platform operator/ });
    // No badge announcing the authentication mechanism beside the name. How the request
    // authenticated is a line in Settings, not a permanent label on the person.
    expect(screen.queryByText('shared key')).not.toBeInTheDocument();

    await user.click(trigger);
    const menu = screen.getByRole('menu');
    expect(within(menu).getByRole('menuitem', { name: 'Settings' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: 'Sign out' })).toBeInTheDocument();
  });

  it('signing out clears the session and nothing the account owns', async () => {
    const removeCredential = vi.fn();
    const deleteSavedRepository = vi.fn();
    const logout = vi.fn(async () => ({}));
    renderApp('/', {
      listFeatures: async () => FEATURE_PAGE,
      logout,
      removeCredential,
      deleteSavedRepository,
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: /Platform operator/ }));
    await user.click(screen.getByRole('menuitem', { name: 'Sign out' }));

    // The token stops being this browser's credential, and the platform is told so: a
    // session token is revocable, unlike the shared key this used to hold, so signing out
    // means something on the server as well as here.
    //
    // No page reload any more. It used to be how the previous identity's cached queries were
    // cleared; `signOut` clears the query cache directly, which is the same guarantee
    // without throwing the application away — and without depending on
    // `window.location.reload`, which jsdom does not implement.
    await waitFor(() => expect(readToken()).toBeUndefined());
    expect(logout).toHaveBeenCalled();
    // Saved credentials and repositories live on the server against the account. Signing
    // out is not a way to lose them.
    expect(removeCredential).not.toHaveBeenCalled();
    expect(deleteSavedRepository).not.toHaveBeenCalled();
    clearToken();
  });
});

const CREDENTIALS = {
  listCredentials: async () => ({
    credentials: [
      { provider: 'openai', configured: true, hint: 'abcd' },
      { provider: 'github', configured: false, hint: '' },
    ],
  }),
};

describe('Settings', () => {
  it('holds only what somebody can configure', async () => {
    renderApp('/settings', {
      ...CREDENTIALS,
      getReadiness: async () => ({
        status: 'ok',
        build_revision: 'b1cbcfba219a1234',
        workflow_schema_version: '1.0',
        runtime_compatible: true,
      }),
    });

    expect(await screen.findByRole('heading', { name: 'Profile', level: 2 })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Provider credentials', level: 2 })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Repositories', level: 2 })).toBeInTheDocument();
    // Gone: platform recovery data and a shared-key explanation are not preferences.
    expect(screen.queryByText('Unresolved operations')).not.toBeInTheDocument();
    expect(screen.queryByText('Platform access')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Forget platform access key/ })).not.toBeInTheDocument();
    // The build is still reachable — an undeployed fix is indistinguishable from a broken
    // one — but collapsed, last, and named for what it is.
    const developer = screen.getByText('Developer information');
    expect(developer.closest('details')).not.toHaveAttribute('open');
  });

  it('saves a repository without asking for a name, an identifier, or a URL', async () => {
    const saveRepository = vi.fn().mockResolvedValue(savedRepository());
    renderApp('/settings', {
      ...CREDENTIALS,
      saveRepository,
      listSavedRepositories: async () => ({
        repositories: [],
        suggested_types: ['Frontend', 'Backend'],
      }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: /Add repository/ }));
    // Chosen from what the account's GitHub token reaches, rather than typed. A URL field
    // asked somebody to guess both that the repository exists at that spelling and that
    // their token could reach it, and answered neither until a run failed.
    await user.selectOptions(
      await screen.findByLabelText('Repository'),
      'https://github.com/Appbroda/admanager_console-2.0',
    );
    // Derived and shown back, rather than asked for.
    expect(screen.getByText('admanager_console-2.0')).toBeInTheDocument();
    expect(screen.queryByLabelText(/Repository ID/)).not.toBeInTheDocument();
    // GitHub's own default branch for that repository, not this form's guess at one.
    expect(screen.getByLabelText('Default branch')).toHaveValue('master');

    await user.clear(screen.getByLabelText('Type'));
    await user.type(screen.getByLabelText('Type'), 'Backend');
    await user.click(screen.getByRole('button', { name: 'Save repository' }));

    await waitFor(() => expect(saveRepository).toHaveBeenCalledTimes(1));
    expect(saveRepository).toHaveBeenCalledWith({
      repository_url: 'https://github.com/Appbroda/admanager_console-2.0',
      default_branch: 'master',
      repository_type: 'Backend',
    });
  });

  it('edits a saved repository in place and removes it', async () => {
    const updateSavedRepository = vi.fn().mockResolvedValue(savedRepository());
    const deleteSavedRepository = vi.fn().mockResolvedValue(undefined);
    renderApp('/settings', {
      ...CREDENTIALS,
      updateSavedRepository,
      deleteSavedRepository,
      listSavedRepositories: async () => ({
        repositories: [savedRepository()],
        suggested_types: ['Backend'],
      }),
    });
    const user = userEvent.setup();

    const saved = await screen.findByRole('list', { name: 'Saved repositories' });
    expect(within(saved).getByText('admanager_console-2.0')).toBeInTheDocument();
    expect(within(saved).getByText('Backend')).toBeInTheDocument();
    expect(within(saved).getByText('master')).toBeInTheDocument();

    await user.click(within(saved).getByRole('button', { name: 'Edit' }));
    await user.clear(screen.getByLabelText('Default branch'));
    await user.type(screen.getByLabelText('Default branch'), 'develop');
    await user.click(screen.getByRole('button', { name: 'Save changes' }));

    await waitFor(() => expect(updateSavedRepository).toHaveBeenCalledTimes(1));
    expect(updateSavedRepository).toHaveBeenCalledWith('repo-1', {
      repository_url: 'https://github.com/Appbroda/admanager_console-2.0',
      default_branch: 'develop',
      repository_type: 'Backend',
    });

    await user.click(within(saved).getByRole('button', { name: 'Remove' }));
    await waitFor(() => expect(deleteSavedRepository).toHaveBeenCalledWith('repo-1'));
  });

  it('says what to do when nothing is saved yet', async () => {
    renderApp('/settings', {
      ...CREDENTIALS,
      listSavedRepositories: async () => ({ repositories: [], suggested_types: [] }),
    });

    expect(await screen.findByText('No repositories saved')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Add repository/ })).toBeInTheDocument();
  });
});
