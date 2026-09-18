import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { ApiError, userMessage } from '@/api/errors';
import { useApi } from '@/app/api-context';
import { useSession } from '@/app/session-context';
import { EmptyState, ErrorState, TableSkeleton } from '@/components/common/States';
import { Badge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { Dialog } from '@/components/ui/Dialog';
import { PageHeader, Panel } from '@/components/ui/Layout';
import type { IssuedToken, PlatformUser } from '@/schemas/feature';

/** The roles this deployment hands out. The server refuses anything not in `Role`. */
const ROLES = [
  { value: 'operator', label: 'Operator — submits and runs features' },
  { value: 'viewer', label: 'Viewer — reads their own workspace' },
  { value: 'admin', label: 'Administrator — also manages accounts and the deployment' },
];

/**
 * Who this deployment knows about, and the four things an administrator does to an account.
 *
 * **No column here renders a secret, and none can.** The list comes from `GET /users`, whose
 * response model has no field a password could occupy; `has_password` is the whole of what it
 * says about one. The single exception is the dialog a freshly issued API token appears in,
 * which shows the value once and says plainly that it will not be shown again — because the
 * platform stores only a digest and there is no endpoint that could show it twice.
 *
 * Every control on this page is hidden for somebody without `user:manage`, and every one of
 * them is refused again by the server if called directly. The hiding is the courtesy.
 */
export function UsersView() {
  const api = useApi();
  const { may, actor } = useSession();
  const queryClient = useQueryClient();
  const [issued, setIssued] = useState<IssuedToken | undefined>();
  const [failure, setFailure] = useState<string | undefined>();

  const users = useQuery({
    queryKey: ['users'],
    queryFn: ({ signal }) => api.listUsers(signal),
    retry: false,
  });

  const refresh = () => queryClient.invalidateQueries({ queryKey: ['users'] });
  const report = (cause: unknown) => {
    setFailure(
      cause instanceof ApiError ? (cause.detail ?? userMessage(cause)) : 'That did not work.',
    );
  };

  const update = useMutation({
    mutationFn: ({
      userId,
      changes,
    }: {
      userId: string;
      changes: { roles?: string[]; disabled?: boolean };
    }) => api.updateUser(userId, changes),
    onSuccess: () => {
      setFailure(undefined);
      void refresh();
    },
    onError: report,
  });

  const issueToken = useMutation({
    mutationFn: (userId: string) => api.issueUserToken(userId, 'issued from the console'),
    onSuccess: (token) => {
      setFailure(undefined);
      setIssued(token);
    },
    onError: report,
  });

  if (!may('user:manage')) {
    return (
      <>
        <PageHeader title="People" />
        <EmptyState
          title="Only an administrator can manage accounts"
          detail="Somebody who can retry a repository should not thereby be able to grant themselves more."
        />
      </>
    );
  }

  const columns: Column<PlatformUser>[] = [
    {
      key: 'display_name',
      header: 'Name',
      sortValue: (row) => row.display_name,
      render: (row) => (
        <span>
          {row.display_name}
          {row.user_id === actor?.actor_id ? <span className="subtle"> (you)</span> : null}
        </span>
      ),
    },
    { key: 'subject', header: 'Email', sortValue: (row) => row.subject, render: (row) => row.subject },
    {
      key: 'roles',
      header: 'Role',
      shrink: true,
      sortValue: (row) => row.roles.join(','),
      render: (row) => (
        <select
          aria-label={`Role for ${row.display_name}`}
          value={row.roles[0] ?? 'operator'}
          disabled={update.isPending}
          onChange={(event) =>
            update.mutate({ userId: row.user_id, changes: { roles: [event.target.value] } })
          }
        >
          {ROLES.map((role) => (
            <option key={role.value} value={role.value}>
              {role.value}
            </option>
          ))}
        </select>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      shrink: true,
      sortValue: (row) => (row.disabled ? 1 : 0),
      render: (row) =>
        row.disabled ? (
          <Badge tone="stopped">disabled</Badge>
        ) : row.must_change_password ? (
          <Badge tone="attention">must change password</Badge>
        ) : row.has_password ? (
          <Badge tone="done">active</Badge>
        ) : (
          <Badge tone="neutral">no password</Badge>
        ),
    },
    {
      key: 'last_login_at',
      header: 'Last sign-in',
      sortValue: (row) => row.last_login_at ?? '',
      render: (row) => (row.last_login_at ? row.last_login_at.slice(0, 16).replace('T', ' ') : '—'),
    },
    {
      key: 'created_at',
      header: 'Created',
      sortValue: (row) => row.created_at,
      render: (row) => row.created_at.slice(0, 10),
    },
    {
      key: 'actions',
      header: 'Actions',
      shrink: true,
      render: (row) => (
        <div className="form__actions">
          <button
            type="button"
            className="button button--small"
            disabled={update.isPending}
            onClick={() =>
              update.mutate({ userId: row.user_id, changes: { disabled: !row.disabled } })
            }
          >
            {row.disabled ? 'Enable' : 'Disable'}
          </button>
          <button
            type="button"
            className="button button--small"
            disabled={issueToken.isPending || row.disabled}
            onClick={() => issueToken.mutate(row.user_id)}
          >
            Issue API token
          </button>
        </div>
      ),
    },
  ];

  return (
    <>
      <PageHeader
        title="People"
        subtitle="Each account has its own workspace: its own features, provider keys, saved repositories and model setups. Nobody sees anybody else's."
      />
      {failure ? (
        <p className="callout callout--warn" role="alert">
          {failure}
        </p>
      ) : null}
      <Panel title="Accounts" flush>
        {users.isLoading ? (
          <TableSkeleton label="Loading accounts…" />
        ) : users.error ? (
          <ErrorState error={users.error} onRetry={() => void users.refetch()} />
        ) : (
          <DataTable
            label="Accounts"
            columns={columns}
            rows={users.data?.users ?? []}
            rowKey={(row) => row.user_id}
            initialSort={{ key: 'created_at', direction: 'asc' }}
            empty={<EmptyState title="No accounts yet" />}
          />
        )}
      </Panel>
      <CreateUserPanel onCreated={() => void refresh()} onFailure={report} />
      <SetPasswordPanel users={users.data?.users ?? []} onFailure={report} onDone={refresh} />
      {issued ? <IssuedTokenDialog token={issued} onDismiss={() => setIssued(undefined)} /> : null}
    </>
  );
}

/**
 * Register an account.
 *
 * The initial password is optional. Given one, the account is created with
 * `must_change_password` — a password one person chose for another is a handover credential,
 * not that person's password. Left blank, the account cannot password-login at all until an
 * administrator sets one, which is the right state for an identity a provider will assert.
 */
function CreateUserPanel({
  onCreated,
  onFailure,
}: {
  onCreated: () => void;
  onFailure: (cause: unknown) => void;
}) {
  const api = useApi();
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState('operator');
  const [password, setPassword] = useState('');

  const create = useMutation({
    mutationFn: () =>
      api.createUser({
        subject: email.trim(),
        display_name: name.trim(),
        roles: [role],
        password: password || undefined,
      }),
    onSuccess: () => {
      setEmail('');
      setName('');
      setPassword('');
      onCreated();
    },
    onError: onFailure,
  });

  return (
    <Panel title="Add someone">
      <form
        className="form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!email.trim() || !name.trim() || create.isPending) return;
          create.mutate();
        }}
      >
        <label className="field__label" htmlFor="new-user-email">
          Email
        </label>
        <input
          id="new-user-email"
          type="email"
          autoComplete="off"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <label className="field__label" htmlFor="new-user-name">
          Display name
        </label>
        <input
          id="new-user-name"
          type="text"
          autoComplete="off"
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
        <label className="field__label" htmlFor="new-user-role">
          Role
        </label>
        <select
          id="new-user-role"
          value={role}
          onChange={(event) => setRole(event.target.value)}
        >
          {ROLES.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
        <label className="field__label" htmlFor="new-user-password">
          First password (optional)
        </label>
        <input
          id="new-user-password"
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        <p className="muted">
          They will have to replace it on first sign-in. Leave it blank to create the account
          without a password and set one later.
        </p>
        <div className="form__actions">
          <button
            type="submit"
            className="button button--primary"
            disabled={create.isPending || !email.trim() || !name.trim()}
          >
            {create.isPending ? 'Adding…' : 'Add'}
          </button>
        </div>
      </form>
    </Panel>
  );
}

/**
 * Set somebody else's password.
 *
 * Its own panel rather than a row action, because it ends every session that account had —
 * including one they may be using right now. That is the point of an administrator resetting
 * a password, and it is worth being deliberate about rather than one click in a table.
 */
function SetPasswordPanel({
  users,
  onFailure,
  onDone,
}: {
  users: PlatformUser[];
  onFailure: (cause: unknown) => void;
  onDone: () => Promise<unknown>;
}) {
  const api = useApi();
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');

  const set = useMutation({
    mutationFn: () => api.setUserPassword(userId, password),
    onSuccess: () => {
      setPassword('');
      void onDone();
    },
    onError: onFailure,
  });

  return (
    <Panel title="Reset a password">
      <form
        className="form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!userId || !password || set.isPending) return;
          set.mutate();
        }}
      >
        <label className="field__label" htmlFor="reset-user">
          Account
        </label>
        <select
          id="reset-user"
          value={userId}
          onChange={(event) => setUserId(event.target.value)}
        >
          <option value="">Choose an account…</option>
          {users.map((item) => (
            <option key={item.user_id} value={item.user_id}>
              {item.display_name} — {item.subject}
            </option>
          ))}
        </select>
        <label className="field__label" htmlFor="reset-password">
          New password
        </label>
        <input
          id="reset-password"
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        <p className="muted">
          Every browser session this account had stops working, and they must choose their own
          password on the next sign-in. Their API tokens are untouched.
        </p>
        <div className="form__actions">
          <button
            type="submit"
            className="button button--primary"
            disabled={set.isPending || !userId || !password}
          >
            {set.isPending ? 'Setting…' : 'Set password'}
          </button>
        </div>
      </form>
    </Panel>
  );
}

/**
 * The one place in this application that renders a credential.
 *
 * It says so, because the platform stores only a digest and there is no endpoint that can
 * show this again. Nothing here writes it anywhere: not to storage, not to a query cache,
 * not to a log. It is in one component's props until the dialog closes.
 */
function IssuedTokenDialog({
  token,
  onDismiss,
}: {
  token: IssuedToken;
  onDismiss: () => void;
}) {
  return (
    <Dialog title="API token issued" onDismiss={onDismiss}>
      <p className="prose">
        Copy this now. The platform keeps only a digest of it, so this is the only time it can
        be shown — there is no endpoint that could show it again.
      </p>
      <pre className="code code--wrap">{token.token}</pre>
      <div className="form__actions">
        <button
          type="button"
          className="button"
          onClick={() => void navigator.clipboard?.writeText(token.token)}
        >
          Copy
        </button>
        <button type="button" className="button button--primary" onClick={onDismiss}>
          Done
        </button>
      </div>
    </Dialog>
  );
}
