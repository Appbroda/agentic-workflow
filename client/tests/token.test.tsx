import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import type { ReactNode } from 'react';
import { ApiContext } from '@/app/api-context';
import { SessionProvider } from '@/app/session';
import { useSession } from '@/app/session-context';
import { LoginPage } from '@/app/LoginPage';
import { ApiError } from '@/api/errors';
import { clearToken, readToken, writeToken } from '@/api/token';
import { stubApi } from './fixtures';

/**
 * Signing in, signing out, and what the browser is left holding.
 *
 * The credential used to be the deployment's shared API key, typed into a gate. It is a
 * session token minted for one account now, and two properties matter more than the form
 * around them: the token never reaches a build artefact, and ending a session takes the
 * previous identity's cached data with it.
 */

afterEach(() => {
  clearToken();
});

const ACTOR = {
  actor_id: 'user-1',
  display_name: 'A person',
  authentication: 'user_token',
  roles: ['operator'],
  permissions: ['feature:read'],
  subject: 'person@example.com',
  must_change_password: false,
};

function renderWithSession(api: Record<string, unknown>, children: ReactNode) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return {
    queryClient,
    ...render(
      <QueryClientProvider client={queryClient}>
        <ApiContext.Provider value={stubApi(api)}>
          <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
            <SessionProvider>{children}</SessionProvider>
          </MemoryRouter>
        </ApiContext.Provider>
      </QueryClientProvider>,
    ),
  };
}

/** Shows what the session currently is, so a test can assert on it without a whole shell. */
function SessionProbe() {
  const { actor, hasToken, signOut } = useSession();
  return (
    <div>
      <p>token: {hasToken ? 'held' : 'none'}</p>
      <p>actor: {actor ? actor.display_name : 'nobody'}</p>
      <button type="button" onClick={() => void signOut()}>
        Sign out
      </button>
    </div>
  );
}

describe('the session credential', () => {
  it('is not baked into the bundle', () => {
    // `VITE_PLATFORM_API_KEY` is retired: Vite substituted it at build time, which put the
    // credential in the bundle and therefore in the image -- pushed to a registry, pulled
    // onto machines. It also won over the runtime value and could not be cleared, so signing
    // out silently did nothing wherever it was set. A fresh browser holds nothing.
    expect(readToken()).toBeUndefined();
  });

  it('lives in the tab and nowhere else', async () => {
    const login = vi.fn(async () => ({
      token: 'session-token-123',
      expires_at: null,
      actor: ACTOR,
      must_change_password: false,
    }));
    renderWithSession({ login, getMe: async () => ACTOR }, <LoginPage />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Email'), 'person@example.com');
    await user.type(screen.getByLabelText('Password'), 'a-long-enough-password');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));

    await waitFor(() => {
      expect(window.sessionStorage.getItem('platform-api-token')).toBe('session-token-123');
    });
    expect(login).toHaveBeenCalledWith('person@example.com', 'a-long-enough-password');
    // sessionStorage, not localStorage: it dies with the tab rather than persisting on the
    // machine for whoever opens the browser next.
    expect(window.localStorage.getItem('platform-api-token')).toBeNull();
  });
});

describe('signing in', () => {
  it('asks for an email and a password before anything makes a request', () => {
    renderWithSession({ getMe: async () => ACTOR }, <LoginPage />);

    expect(screen.getByLabelText('Email')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
  });

  it('says one thing for every kind of refusal', async () => {
    // The server answers an unknown email, a wrong password and a disabled account with one
    // 401 and one body, deliberately. This page must not invent a distinction it refused to
    // make -- a specific message here is an account enumeration oracle just the same.
    const login = vi.fn(async () => {
      throw new ApiError('unauthorized', 'Invalid email or password.', 401);
    });
    renderWithSession({ login, getMe: async () => ACTOR }, <LoginPage />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Email'), 'nobody@example.com');
    await user.type(screen.getByLabelText('Password'), 'not-the-password');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'That email and password do not match an account.',
    );
    expect(readToken()).toBeUndefined();
  });

  it('says when the rate limit refused the attempt', async () => {
    // Distinct from a wrong password on purpose: this one is a fact about the platform, and
    // somebody who is locked out for fifteen minutes needs to know that rather than doubting
    // their own password.
    const login = vi.fn(async () => {
      throw new ApiError('unknown', 'Too many login attempts.', 429);
    });
    renderWithSession({ login, getMe: async () => ACTOR }, <LoginPage />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Email'), 'person@example.com');
    await user.type(screen.getByLabelText('Password'), 'a-long-enough-password');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Too many sign-in attempts');
  });

  it('clears the password field whether or not it worked', async () => {
    const login = vi.fn(async () => {
      throw new ApiError('unauthorized', 'Invalid email or password.', 401);
    });
    renderWithSession({ login, getMe: async () => ACTOR }, <LoginPage />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Email'), 'person@example.com');
    await user.type(screen.getByLabelText('Password'), 'a-long-enough-password');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));

    await screen.findByRole('alert');
    // A password left in component state after a failed attempt is a password in a React
    // devtools tree, which is exactly where a shared machine's next user would find it.
    expect(screen.getByLabelText('Password')).toHaveValue('');
  });
});

describe('signing out', () => {
  it('revokes the token, forgets it, and clears what was cached as that identity', async () => {
    writeToken('already-here');
    const logout = vi.fn(async () => ({}));
    const { queryClient } = renderWithSession(
      { logout, getMe: async () => ACTOR },
      <SessionProbe />,
    );
    // Something fetched as the signed-in identity. After signing out it must not be readable
    // from the cache: the next person on this tab would otherwise see it until each query
    // happened to refetch, which is a cross-user leak in the browser.
    queryClient.setQueryData(['features'], { features: [{ feature_id: 'theirs' }] });
    const user = userEvent.setup();

    expect(await screen.findByText('actor: A person')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Sign out' }));

    await waitFor(() => expect(screen.getByText('token: none')).toBeInTheDocument());
    expect(logout).toHaveBeenCalled();
    expect(readToken()).toBeUndefined();
    expect(queryClient.getQueryData(['features'])).toBeUndefined();
  });

  it('signs the browser out even when the platform cannot be reached', async () => {
    // Somebody who pressed sign out must end up signed out of this machine. The token stops
    // working here either way; an unrevoked session expires on its own.
    writeToken('already-here');
    const logout = vi.fn(async () => {
      throw new ApiError('network', 'Network request failed');
    });
    renderWithSession({ logout, getMe: async () => ACTOR }, <SessionProbe />);
    const user = userEvent.setup();

    await screen.findByText('actor: A person');
    await user.click(screen.getByRole('button', { name: 'Sign out' }));

    await waitFor(() => expect(screen.getByText('token: none')).toBeInTheDocument());
    expect(readToken()).toBeUndefined();
  });
});

describe('a session the platform no longer accepts', () => {
  it('is not a session', async () => {
    // The API client clears the token on any 401 -- see `onUnauthorized`. This asserts the
    // consequence: the provider reports no actor, which is what puts the login screen back.
    writeToken('revoked');
    renderWithSession(
      {
        getMe: async () => {
          throw new ApiError('unauthorized', 'Invalid authentication credentials.', 401);
        },
      },
      <SessionProbe />,
    );

    expect(await screen.findByText('actor: nobody')).toBeInTheDocument();
  });
});
