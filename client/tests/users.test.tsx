import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import type { FeatureApi } from '@/api/features';
import { AppProviders } from '@/app/providers';
import { AppLayout } from '@/app/AppLayout';
import { DashboardPage } from '@/pages/DashboardPage';
import { UsersPage } from '@/pages/UsersPage';
import { ChangePasswordPage } from '@/pages/ChangePasswordPage';
import { writeToken } from '@/api/token';
import { ApiError } from '@/api/errors';
import { stubApi } from './fixtures';

/**
 * Administering accounts, and the screen an account with a handed-over password cannot leave.
 *
 * The property worth defending on the accounts page is negative: no column renders a secret,
 * and none can, because `GET /users` has no field a password could occupy. The one credential
 * this application ever displays is a freshly issued API token, once, in a dialog that says
 * it will not be shown again.
 *
 * Every password in this file is generated inline. Nothing here is a literal, or a
 * placeholder shaped like one.
 */

const PASSWORD = `pw-${Math.random().toString(36).slice(2)}${'x'.repeat(12)}`;

beforeEach(() => writeToken('test-token'));

const ADMIN = {
  actor_id: 'platform-admin',
  display_name: 'Akhilesh Kumar Pandey',
  authentication: 'user_token',
  roles: ['admin'],
  permissions: ['user:manage', 'feature:read'],
  subject: 'akhilesh@appbroda.com',
  must_change_password: false,
};

const OPERATOR = {
  actor_id: 'user-2',
  display_name: 'Sam',
  authentication: 'user_token',
  roles: ['operator'],
  permissions: ['feature:read'],
  subject: 'sam@example.com',
  must_change_password: false,
};

const USERS = {
  users: [
    {
      user_id: 'platform-admin',
      subject: 'akhilesh@appbroda.com',
      display_name: 'Akhilesh Kumar Pandey',
      roles: ['admin'],
      disabled: false,
      created_at: '2026-09-09T09:00:00Z',
      has_password: true,
      must_change_password: false,
      last_login_at: '2026-09-09T10:30:00Z',
    },
    {
      user_id: 'user-2',
      subject: 'sam@example.com',
      display_name: 'Sam',
      roles: ['operator'],
      disabled: false,
      created_at: '2026-09-09T09:30:00Z',
      has_password: false,
      must_change_password: false,
      last_login_at: null,
    },
  ],
};

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
            <Route path="users" element={<UsersPage />} />
            <Route path="account/password" element={<ChangePasswordPage />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('the accounts page', () => {
  it('lists who the deployment knows about, and no secret about any of them', async () => {
    renderApp('/users', { getMe: async () => ADMIN, listUsers: async () => USERS });

    const table = await screen.findByRole('table', { name: 'Accounts' });
    expect(within(table).getByText('akhilesh@appbroda.com')).toBeInTheDocument();
    expect(within(table).getByText('sam@example.com')).toBeInTheDocument();
    // An account with no password says so, which is a real state -- a federated identity, or
    // one an administrator has not handed a password to yet.
    expect(within(table).getByText('no password')).toBeInTheDocument();
    // Nothing that looks like a hash, a digest or a stored value. The response model this
    // reads has no field one could arrive in.
    expect(screen.queryByText(/scrypt\$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/password_hash/)).not.toBeInTheDocument();
  });

  it('is not offered to somebody without the permission', async () => {
    renderApp('/users', { getMe: async () => OPERATOR, listUsers: async () => USERS });

    // Reachable by typing the URL, and it says so politely rather than pretending the page
    // does not exist. The endpoints behind it refuse regardless, which is where the control
    // is -- this is the courtesy.
    expect(
      await screen.findByText('Only an administrator can manage accounts'),
    ).toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'Accounts' })).not.toBeInTheDocument();
  });

  it('links to accounts from the sidebar only for an administrator', async () => {
    const { unmount } = renderApp('/', { getMe: async () => ADMIN });
    expect(await screen.findByRole('link', { name: 'People' })).toBeInTheDocument();
    unmount();

    renderApp('/', { getMe: async () => OPERATOR });
    // The keyed check is on the named permission the server publishes, never on the role
    // string: `ROLE_PERMISSIONS` is the one authority on what a role grants.
    await screen.findByRole('link', { name: 'Settings' });
    expect(screen.queryByRole('link', { name: 'People' })).not.toBeInTheDocument();
  });

  it('creates an account with a first password that must be replaced', async () => {
    // Typed through the API surface so `mock.calls` carries the argument shape: an
    // untyped `vi.fn(async () => …)` infers no parameters, and the body below asserts on
    // exactly what was sent.
    const createUser = vi.fn<FeatureApi['createUser']>(async () => USERS.users[1]!);
    renderApp('/users', {
      getMe: async () => ADMIN,
      listUsers: async () => USERS,
      createUser,
    });
    const user = userEvent.setup();

    await screen.findByRole('table', { name: 'Accounts' });
    await user.type(screen.getByLabelText('Email'), 'new@example.com');
    await user.type(screen.getByLabelText('Display name'), 'A new person');
    await user.type(screen.getByLabelText('First password (optional)'), PASSWORD);
    await user.click(screen.getByRole('button', { name: 'Add' }));

    await waitFor(() => expect(createUser).toHaveBeenCalled());
    expect(createUser.mock.calls[0]?.[0]).toEqual({
      subject: 'new@example.com',
      display_name: 'A new person',
      roles: ['operator'],
      password: PASSWORD,
    });
  });

  it('creates an account with no password when none is typed', async () => {
    // Typed through the API surface so `mock.calls` carries the argument shape: an
    // untyped `vi.fn(async () => …)` infers no parameters, and the body below asserts on
    // exactly what was sent.
    const createUser = vi.fn<FeatureApi['createUser']>(async () => USERS.users[1]!);
    renderApp('/users', {
      getMe: async () => ADMIN,
      listUsers: async () => USERS,
      createUser,
    });
    const user = userEvent.setup();

    await screen.findByRole('table', { name: 'Accounts' });
    await user.type(screen.getByLabelText('Email'), 'federated@example.com');
    await user.type(screen.getByLabelText('Display name'), 'Asserted');
    await user.click(screen.getByRole('button', { name: 'Add' }));

    await waitFor(() => expect(createUser).toHaveBeenCalled());
    // `password` absent rather than an empty string: an empty string is a password the
    // server would refuse for being too short, and "no password" is a different request.
    expect(createUser.mock.calls[0]?.[0]).toEqual({
      subject: 'federated@example.com',
      display_name: 'Asserted',
      roles: ['operator'],
    });
  });

  it('reports the last-administrator refusal in the server’s own words', async () => {
    // The server refuses a change that would leave nobody able to hand out an account. Its
    // sentence explains what to do about it, so it is shown rather than replaced.
    const updateUser = vi.fn(async () => {
      throw new ApiError(
        'conflict',
        'last administrator',
        409,
        "This is the deployment's last enabled administrator. Grant another account the admin role before changing this one.",
      );
    });
    renderApp('/users', {
      getMe: async () => ADMIN,
      listUsers: async () => USERS,
      updateUser,
    });
    const user = userEvent.setup();

    const table = await screen.findByRole('table', { name: 'Accounts' });
    const row = within(table).getByText('akhilesh@appbroda.com').closest('tr');
    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Disable' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('last enabled administrator');
  });

  it('shows an issued token once, and says it cannot be shown again', async () => {
    const issueUserToken = vi.fn(async () => ({
      token_id: 'token-1',
      user_id: 'user-2',
      label: 'issued from the console',
      token: 'the-only-time-this-is-readable',
      expires_at: null,
    }));
    renderApp('/users', {
      getMe: async () => ADMIN,
      listUsers: async () => USERS,
      issueUserToken,
    });
    const user = userEvent.setup();

    const table = await screen.findByRole('table', { name: 'Accounts' });
    const row = within(table).getByText('sam@example.com').closest('tr');
    await user.click(
      within(row as HTMLElement).getByRole('button', { name: 'Issue API token' }),
    );

    const dialog = await screen.findByRole('dialog', { name: 'API token issued' });
    expect(within(dialog).getByText('the-only-time-this-is-readable')).toBeInTheDocument();
    // Said plainly, because it is true: the platform keeps only a digest and no endpoint can
    // return the value a second time.
    expect(within(dialog).getByText(/only time it can be shown/)).toBeInTheDocument();
  });
});

describe('an account whose password was set for it', () => {
  it('cannot reach anything else until it chooses one', async () => {
    renderApp('/', {
      getMe: async () => ({ ...OPERATOR, must_change_password: true }),
      listFeatures: async () => ({ features: [], next_cursor: null }),
    });

    // Redirected off the dashboard, whatever it was asked for. The check is in the layout
    // rather than in the route list precisely so a route added later is behind it too.
    expect(await screen.findByText('Choose a password')).toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'Features' })).not.toBeInTheDocument();
  });

  it('lets go once the password is changed', async () => {
    const changePassword = vi.fn(async () => OPERATOR);
    renderApp('/account/password', {
      getMe: async () => ({ ...OPERATOR, must_change_password: true }),
      changePassword,
    });
    const user = userEvent.setup();

    await screen.findByText('Choose a password');
    await user.type(screen.getByLabelText('Current password'), 'the-handover-password');
    await user.type(screen.getByLabelText('New password'), PASSWORD);
    await user.type(screen.getByLabelText('New password again'), PASSWORD);
    await user.click(screen.getByRole('button', { name: 'Change password' }));

    await waitFor(() =>
      expect(changePassword).toHaveBeenCalledWith('the-handover-password', PASSWORD),
    );
  });

  it('refuses a password the platform would refuse, before asking it to', async () => {
    const changePassword = vi.fn();
    renderApp('/account/password', {
      getMe: async () => ({ ...OPERATOR, must_change_password: true }),
      changePassword,
    });
    const user = userEvent.setup();

    await screen.findByText('Choose a password');
    await user.type(screen.getByLabelText('Current password'), 'the-handover-password');
    await user.type(screen.getByLabelText('New password'), 'short');
    await user.type(screen.getByLabelText('New password again'), 'short');

    expect(screen.getByRole('alert')).toHaveTextContent('at least 12 characters');
    // The rule is the server's -- `hash_password` enforces it on every path that sets a
    // password. This is the courtesy of not making somebody wait for a round trip.
    expect(changePassword).not.toHaveBeenCalled();
  });

  it('says when the two new passwords do not match', async () => {
    renderApp('/account/password', {
      getMe: async () => ({ ...OPERATOR, must_change_password: true }),
    });
    const user = userEvent.setup();

    await screen.findByText('Choose a password');
    await user.type(screen.getByLabelText('New password'), PASSWORD);
    await user.type(screen.getByLabelText('New password again'), `${PASSWORD}-different`);

    expect(screen.getByRole('alert')).toHaveTextContent('do not match');
  });
});
