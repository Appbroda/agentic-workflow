import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiContext } from '@/app/api-context';
import { SessionProvider } from '@/app/session';
import { SettingsView } from '@/features/settings/SettingsView';
import { ApiError } from '@/api/errors';
import { githubRepositories, savedRepository, stubApi } from './fixtures';

/**
 * Adding a repository, and what the platform knows before it lets somebody add one.
 *
 * This form used to be a URL field, which asked somebody to guess two things — that the
 * repository exists at that spelling, and that their token can reach it — and answered both by
 * a run failing hours later. What is being defended here is that the menu comes from GitHub,
 * that a repository that cannot be built in is visibly not selectable rather than quietly
 * missing, and that having no menu always comes with the server's sentence saying why.
 */

const READY = {
  status: 'ok',
  build_revision: 'abcdef0123456789',
  workflow_schema_version: '1.0',
  runtime_compatible: true,
};

function renderSettings(api: Record<string, unknown>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ApiContext.Provider value={stubApi(api)}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <SessionProvider>
            <SettingsView />
          </SessionProvider>
        </MemoryRouter>
      </ApiContext.Provider>
    </QueryClientProvider>,
  );
}

const BASE = {
  getReadiness: async () => READY,
  listUnresolvedOperations: async () => ({ operations: [] }),
  listCredentials: async () => ({
    credentials: [{ provider: 'github', configured: true, hint: 'abcd' }],
  }),
  getMe: async () => ({
    actor_id: 'user-1',
    display_name: 'Alex',
    authentication: 'user_token',
    roles: ['operator'],
    permissions: ['credential:manage'],
  }),
};

async function openTheForm() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole('button', { name: /Add repository/ }));
  return user;
}

describe('choosing a repository', () => {
  it('offers what the token reaches and refuses to offer what it cannot push to', async () => {
    renderSettings({ ...BASE, listSavedRepositories: async () => ({ repositories: [], suggested_types: ['Backend'] }) });
    await openTheForm();

    const picker = await screen.findByLabelText('Repository');
    const options = within(picker).getAllByRole('option');
    // The placeholder, then the two the token reaches.
    expect(options.map((option) => option.textContent)).toEqual([
      'Choose a repository…',
      'Appbroda/admanager_console-2.0 (private)',
      'Appbroda/read-only-service — read only',
    ]);
    // Present and disabled rather than absent: a repository that is silently not in the menu
    // is indistinguishable from one that does not exist, and sends somebody looking for the
    // wrong problem.
    expect(options[2]).toBeDisabled();
    expect(options[1]).not.toBeDisabled();
  });

  it('prefills the branch from the repository rather than from a guess', async () => {
    const saveRepository = vi.fn().mockResolvedValue(savedRepository());
    renderSettings({
      ...BASE,
      listSavedRepositories: async () => ({ repositories: [], suggested_types: ['Backend'] }),
      listGitHubRepositories: async () =>
        githubRepositories({
          repositories: [
            {
              full_name: 'Appbroda/service',
              repository_url: 'https://github.com/Appbroda/service',
              default_branch: 'develop',
              private: true,
              archived: false,
              can_push: true,
              already_saved: false,
            },
          ],
        }),
      saveRepository,
    });
    const user = await openTheForm();

    await user.selectOptions(
      await screen.findByLabelText('Repository'),
      'https://github.com/Appbroda/service',
    );

    // GitHub's answer, not `master` and not `main`. Both were wrong for somebody.
    expect(screen.getByLabelText('Default branch')).toHaveValue('develop');

    await user.click(screen.getByRole('button', { name: 'Save repository' }));
    expect(saveRepository).toHaveBeenCalledWith({
      repository_url: 'https://github.com/Appbroda/service',
      default_branch: 'develop',
      repository_type: 'Backend',
    });
  });

  it('leaves out a repository that is already saved', async () => {
    renderSettings({
      ...BASE,
      listGitHubRepositories: async () =>
        githubRepositories({
          repositories: [
            {
              full_name: 'Appbroda/admanager_console-2.0',
              repository_url: 'https://github.com/Appbroda/admanager_console-2.0',
              default_branch: 'master',
              private: true,
              archived: false,
              can_push: true,
              already_saved: true,
            },
            {
              full_name: 'Appbroda/other',
              repository_url: 'https://github.com/Appbroda/other',
              default_branch: 'main',
              private: false,
              archived: false,
              can_push: true,
              already_saved: false,
            },
          ],
        }),
    });
    await openTheForm();

    const picker = await screen.findByLabelText('Repository');
    const labels = within(picker)
      .getAllByRole('option')
      .map((option) => option.textContent);
    expect(labels).not.toContain('Appbroda/admanager_console-2.0 (private)');
    expect(labels).toContain('Appbroda/other');
  });

  it('keeps a repository editable even when the token no longer lists it', async () => {
    // A repository saved before the picker existed, or one on another host. Its branch and
    // label must still be changeable, so it is pinned into the menu as itself.
    renderSettings({
      ...BASE,
      listSavedRepositories: async () => ({
        repositories: [
          savedRepository({ repository_url: 'https://gitlab.example.com/acme/api' }),
        ],
        suggested_types: ['Backend'],
      }),
      listGitHubRepositories: async () => githubRepositories({ repositories: [] }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Edit' }));

    const picker = await screen.findByLabelText('Repository');
    expect(picker).toHaveValue('https://gitlab.example.com/acme/api');
    expect(
      within(picker).getByRole('option', { name: /gitlab.example.com\/acme\/api/ }),
    ).toBeInTheDocument();
  });

  it('says why there is no menu rather than showing an empty one', async () => {
    // An empty menu is indistinguishable from an answer of "your token reaches nothing", and
    // the two have completely different remedies.
    renderSettings({
      ...BASE,
      listSavedRepositories: async () => ({ repositories: [], suggested_types: ['Backend'] }),
      listGitHubRepositories: async () =>
        githubRepositories({
          available: false,
          detail:
            'No GitHub token is stored for this account. Add one in Provider credentials and ' +
            'the repositories it can reach will be listed here.',
          repositories: [],
        }),
    });
    await openTheForm();

    expect(await screen.findByText(/No GitHub token is stored/)).toBeInTheDocument();
  });

  it('passes on the deployment’s own explanation when the endpoint refuses', async () => {
    // A deployment that keeps no credentials explains itself better than any wording in this
    // client could, so its sentence is shown rather than replaced.
    renderSettings({
      ...BASE,
      listSavedRepositories: async () => ({ repositories: [], suggested_types: ['Backend'] }),
      listGitHubRepositories: async () => {
        throw new ApiError(
          'server',
          'unavailable',
          503,
          'This deployment does not store provider credentials. Supply them as request ' +
            'headers instead.',
        );
      },
    });
    await openTheForm();

    expect(await screen.findByText(/does not store provider credentials/)).toBeInTheDocument();
  });

  it('says so when the failure carries no explanation at all', async () => {
    renderSettings({
      ...BASE,
      listSavedRepositories: async () => ({ repositories: [], suggested_types: ['Backend'] }),
      listGitHubRepositories: async () => {
        throw new ApiError('network', 'Network request failed');
      },
    });
    await openTheForm();

    expect(await screen.findByText(/could not be asked/)).toBeInTheDocument();
  });

  it('says so when everything the token reaches is already saved', async () => {
    // The one empty menu the server has no sentence for: it answered, and every option it
    // gave is already in the list above.
    renderSettings({
      ...BASE,
      listGitHubRepositories: async () => {
        const listed = githubRepositories();
        return {
          ...listed,
          repositories: listed.repositories.map((option) => ({ ...option, already_saved: true })),
        };
      },
    });
    await openTheForm();

    expect(await screen.findByText(/already saved/)).toBeInTheDocument();
  });

  it('admits when the listing was cut short', async () => {
    // A truncated menu that does not say it is truncated reads as the whole answer.
    renderSettings({
      ...BASE,
      listSavedRepositories: async () => ({ repositories: [], suggested_types: ['Backend'] }),
      listGitHubRepositories: async () => {
        const listed = githubRepositories();
        return { ...listed, access: { ...listed.access, truncated: true } };
      },
    });
    await openTheForm();

    expect(await screen.findByText(/most recently pushed/)).toBeInTheDocument();
  });

  it('shows the server’s own sentence when a repository is refused', async () => {
    // The endpoint enforces the same rule the picker offers by, and its refusal names the
    // remedy. "The request was not valid" would throw away the only useful part.
    renderSettings({
      ...BASE,
      listSavedRepositories: async () => ({ repositories: [], suggested_types: ['Backend'] }),
      saveRepository: async () => {
        throw new ApiError(
          'validation',
          'refused',
          422,
          'Your GitHub token can see this repository but cannot push to it, so the platform ' +
            'could not open a pull request. Grant write access, then add it.',
        );
      },
    });
    const user = await openTheForm();

    await user.selectOptions(
      await screen.findByLabelText('Repository'),
      'https://github.com/Appbroda/admanager_console-2.0',
    );
    await user.click(screen.getByRole('button', { name: 'Save repository' }));

    expect(await screen.findByText(/Grant write access/)).toBeInTheDocument();
  });
});

describe('adding the token the picker depends on', () => {
  it('reports what GitHub said about a token that was just saved', async () => {
    const storeCredential = vi.fn().mockResolvedValue({
      provider: 'github',
      configured: true,
      hint: 'wxyz',
      access: {
        verified: 'accepted',
        token_kind: 'fine_grained',
        scopes: [],
        repositories_listed: true,
        repository_count: 4,
        writable_count: 2,
        truncated: false,
        advisories: ['A fine-grained token does not publish its own permissions.'],
      },
    });
    renderSettings({
      ...BASE,
      listCredentials: async () => ({
        credentials: [{ provider: 'github', configured: false, hint: '' }],
      }),
      storeCredential,
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Add' }));
    await user.type(screen.getByLabelText('GitHub key'), 'github_pat_typed');
    await user.click(screen.getByRole('button', { name: 'Save' }));

    // The number that decides whether anything can be built leads the sentence.
    expect(await screen.findByText(/2 writable of 4 repositories/)).toBeInTheDocument();
    expect(screen.getByText(/does not publish its own permissions/)).toBeInTheDocument();
  });

  it('shows a refused token’s reason instead of a generic validation message', async () => {
    // The whole point of asking at save time: the remedy arrives at the paste, not at a push
    // hours later. A generic "the request was not valid" would keep it invisible.
    renderSettings({
      ...BASE,
      listCredentials: async () => ({
        credentials: [{ provider: 'github', configured: false, hint: '' }],
      }),
      storeCredential: async () => {
        throw new ApiError(
          'validation',
          'refused',
          422,
          "This token carries no repository scope, so GitHub will refuse every clone and push " +
            "the platform makes. Re-issue it with the 'repo' scope.",
        );
      },
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Add' }));
    await user.type(screen.getByLabelText('GitHub key'), 'ghp_scopeless');
    await user.click(screen.getByRole('button', { name: 'Save' }));

    expect(await screen.findByText(/'repo' scope/)).toBeInTheDocument();
  });
});
