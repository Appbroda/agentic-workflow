import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiContext } from '@/app/api-context';
import { SessionProvider } from '@/app/session';
import { SettingsView } from '@/features/settings/SettingsView';
import { ApiError } from '@/api/errors';
import { stubApi } from './fixtures';

/**
 * Who this browser is acting as, and what the platform will admit about the keys it holds.
 *
 * The second half matters more than it looks: a credential the platform keeps must be
 * describable without being readable, and a page is the most likely place for that
 * distinction to quietly stop holding.
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
    credentials: [
      { provider: 'openai', configured: false, hint: '' },
      { provider: 'github', configured: false, hint: '' },
    ],
  }),
};

describe('who this browser is', () => {
  it('names the identity and says when it is the shared key', async () => {
    renderSettings({
      ...BASE,
      getMe: async () => ({
        actor_id: 'platform-admin',
        display_name: 'Platform operator',
        authentication: 'platform_key',
        roles: ['admin'],
        permissions: ['repair:approve'],
      }),
    });

    expect(await screen.findByText('Platform operator')).toBeInTheDocument();
    // Worth saying plainly: everything done with this key is recorded against one identity,
    // so an audit trail cannot say which person acted.
    expect(await screen.findByText(/recorded against/)).toBeInTheDocument();
  });

  it('says nothing about shared keys when signed in as a person', async () => {
    renderSettings({
      ...BASE,
      getMe: async () => ({
        actor_id: 'user-1',
        display_name: 'Alex',
        authentication: 'user_token',
        roles: ['operator'],
        permissions: ['feature:retry'],
      }),
    });

    expect(await screen.findByText('Alex')).toBeInTheDocument();
    // The shared-key explanation is the only thing that mentions how the request
    // authenticated, and a named person does not get it.
    expect(screen.queryByText(/shared administrative key/)).not.toBeInTheDocument();
    expect(screen.queryByText(/recorded against/)).not.toBeInTheDocument();
  });
});

describe('provider credentials', () => {
  const ME = {
    getMe: async () => ({
      actor_id: 'user-1',
      display_name: 'Alex',
      authentication: 'user_token',
      roles: ['operator'],
      permissions: ['credential:manage'],
    }),
  };

  it('shows only whether a key is configured, and its last four characters', async () => {
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => ({
        credentials: [
          {
            provider: 'openai',
            configured: true,
            hint: 'a9f2',
            updated_at: '2026-08-25T10:00:00Z',
          },
          { provider: 'github', configured: false, hint: '' },
        ],
      }),
    });

    const list = await screen.findByRole('list', { name: 'Provider credentials' });
    const openai = within(list).getAllByRole('listitem')[0]!;
    expect(within(openai).getByText('Configured')).toBeInTheDocument();
    expect(within(openai).getByText('…a9f2')).toBeInTheDocument();
    // There is nowhere on this page for the key itself, because the platform has no endpoint
    // that returns one.
    expect(list.textContent).not.toMatch(/sk-/);

    const github = within(list).getAllByRole('listitem')[1]!;
    expect(within(github).getByText('Not configured')).toBeInTheDocument();
    expect(within(github).getByRole('button', { name: 'Add' })).toBeInTheDocument();
  });

  it('stores a key without ever putting it back on the page', async () => {
    const storeCredential = vi.fn().mockResolvedValue({
      provider: 'openai',
      configured: true,
      hint: 'rson',
    });
    renderSettings({ ...BASE, ...ME, storeCredential });
    const user = userEvent.setup();

    await user.click((await screen.findAllByRole('button', { name: 'Add' }))[0]!);
    await user.type(screen.getByLabelText('OpenAI key'), 'sk-typed-by-a-person');
    await user.click(screen.getByRole('button', { name: 'Save' }));

    expect(storeCredential).toHaveBeenCalledWith('openai', 'sk-typed-by-a-person');
    // The input is cleared and the form closed, so the value is not left sitting in the DOM.
    expect(screen.queryByLabelText('OpenAI key')).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain('sk-typed-by-a-person');
  });

  it('removes a key when asked', async () => {
    const removeCredential = vi.fn().mockResolvedValue({
      provider: 'github',
      configured: false,
      hint: '',
    });
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => ({
        credentials: [{ provider: 'github', configured: true, hint: 'abcd' }],
      }),
      removeCredential,
    });
    const user = userEvent.setup();

    // Scoped to the credential list. Saved repositories have a Remove of their own, and an
    // unscoped query matched whichever rendered first -- which was the credential's only
    // because the Slack and design-source panels were slowing the other query down. Those
    // panels are administrator-only now, so the race resolved the other way.
    const credentials = await screen.findByLabelText('Provider credentials');
    await user.click(within(credentials).getByRole('button', { name: 'Remove' }));

    expect(removeCredential).toHaveBeenCalledWith('github');
  });

  it('reports a stored key this deployment can no longer open', async () => {
    // A deployment whose encryption key changed cannot read what it sealed. Saying so is the
    // only way somebody knows to enter it again.
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => ({
        credentials: [{ provider: 'openai', configured: true, hint: 'a9f2' }],
      }),
      checkCredential: async () => ({
        provider: 'openai',
        configured: true,
        usable: false,
        detail: 'a stored credential could not be opened and must be re-entered',
      }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Check' }));

    expect(await screen.findByText(/must be re-entered/)).toBeInTheDocument();
  });

  it('asks the provider only when Verify is pressed, and says which question it asked', async () => {
    // Two questions, and the endpoint only reaches the provider for the second one. A page
    // that sent `verify` on every Check would put somebody's key on the wire on page-load
    // reflex, which is the cost that made verification opt-in in the first place.
    const checkCredential = vi.fn().mockResolvedValue({
      provider: 'github',
      configured: true,
      usable: true,
      verified: 'accepted',
      detail: 'The stored credential is readable, and the provider accepted it just now.',
    });
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => ({
        credentials: [{ provider: 'github', configured: true, hint: 'abcd' }],
      }),
      checkCredential,
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Check' }));
    expect(checkCredential).toHaveBeenLastCalledWith('github', { verify: false });

    await user.click(screen.getByRole('button', { name: 'Verify' }));
    expect(checkCredential).toHaveBeenLastCalledWith('github', { verify: true });
    expect(await screen.findByText('Provider accepted it')).toBeInTheDocument();
  });

  it('shows a refusal as a refusal even though the key is still readable', async () => {
    // The run-190 failure exactly: the PAT decrypted fine all day and every clone using it
    // was refused. `usable` and `verified` disagree, and the page must not settle that
    // disagreement in favour of the reassuring half.
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => ({
        credentials: [{ provider: 'github', configured: true, hint: 'abcd' }],
      }),
      checkCredential: async () => ({
        provider: 'github',
        configured: true,
        usable: true,
        verified: 'refused',
        detail:
          'The stored credential is readable, but the provider refused it. It has expired or ' +
          'been revoked, and every operation using it will fail until it is replaced.',
      }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Verify' }));

    expect(await screen.findByText('Provider refused it')).toBeInTheDocument();
    const detail = screen.getByText(/expired or been revoked/);
    expect(detail).toHaveClass('field__error');
  });

  it('reports a provider that could not be asked as no answer, not as reassurance', async () => {
    // `unknown` is the absence of an answer. Rendering it like an acceptance is how a dead
    // credential stayed invisible for a day; rendering it like a refusal would cry wolf every
    // time a provider was slow.
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => ({
        credentials: [{ provider: 'openai', configured: true, hint: 'a9f2' }],
      }),
      checkCredential: async () => ({
        provider: 'openai',
        configured: true,
        usable: true,
        verified: 'unknown',
        detail:
          'The stored credential is readable. The provider could not be asked whether it ' +
          'still accepts it, so this says nothing about whether it works.',
      }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Verify' }));

    expect(await screen.findByText('Provider did not answer')).toBeInTheDocument();
    expect(screen.getByText(/says nothing about whether it works/)).toBeInTheDocument();
    expect(screen.queryByText('Provider accepted it')).not.toBeInTheDocument();
  });

  it('offers no provider verdict for a key that is not stored', async () => {
    // Nothing to ask about. Both buttons belong to a credential that exists.
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => ({
        credentials: [{ provider: 'github', configured: false, hint: '' }],
      }),
    });

    expect(await screen.findByRole('button', { name: 'Add' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Verify' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Check' })).not.toBeInTheDocument();
  });

  it('says plainly when the deployment keeps no credentials at all', async () => {
    // Not storing them is the platform's original behaviour and a legitimate configuration,
    // so it reads as a fact rather than as an error.
    renderSettings({
      ...BASE,
      ...ME,
      listCredentials: async () => {
        throw new ApiError('server', 'unavailable', 503, 'no store');
      },
    });

    expect(await screen.findByText(/does not store provider credentials/)).toBeInTheDocument();
  });
});
