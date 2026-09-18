import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { useSession } from '@/app/session-context';
import { ApiError, refusalMessage, userMessage } from '@/api/errors';
import { Async, EmptyState } from '@/components/common/States';
import { PageHeader, Panel } from '@/components/ui/Layout';
import { Badge } from '@/components/ui/Badge';
import { DetailList, DetailRow } from '@/components/ui/Value';
import { IconPlus } from '@/components/ui/icons';
import { absoluteTime } from '@/utils/time';
import { ModelConfigurationPanel } from './ModelConfigurationPanel';
import { CredentialCheckResult } from './CredentialCheckResult';
import { GitHubAccessSummary } from './GitHubAccessSummary';
import { DesignSourcePanel } from './DesignSourcePanel';
import { SlackLinkSettings, SlackNotificationsPanel } from './SlackNotificationsPanel';
import type { Credential, SavedRepository } from '@/schemas/feature';

/**
 * Things the person using this can actually configure.
 *
 * Three sections are the three prerequisites for doing any work here: who you are, the
 * provider keys the platform will use on your behalf, and the repositories you work with. The
 * fourth is not a preference and is here anyway — which model each stage of a feature runs on
 * was, until this panel, readable only in the API's startup log. It is shown as what it is:
 * the deployment's own configuration, not editable from a page.
 *
 * Deliberately gone from this page: the unresolved-operation table, which is platform recovery
 * data and not a preference; the API build, schema version and readiness fields, which are
 * engineering diagnostics and now sit collapsed at the bottom under their own heading; and the
 * "platform access" panel, which explained a shared-key mechanism a normal user cannot act on.
 * Signing out lives in the user menu, where somebody would look for it.
 */
export function SettingsView() {
  const { may } = useSession();
  return (
    <div className="page stack">
      <PageHeader
        title="Settings"
        subtitle="Your account, the provider keys the platform uses on your behalf, and the repositories you build in."
      />
      <Panel title="Profile">
        <Profile />
      </Panel>
      <Panel title="Provider credentials">
        <CredentialSettings />
      </Panel>
      <ModelConfigurationPanel />
      {/* Slack delivery and the design source are single rows for the whole deployment:
          re-pointing either changes where *everybody's* feature threads go and which Figma
          account *everybody's* citations resolve against. That is an administrator's decision
          now, so these panels are not drawn for anybody else -- and the endpoints behind them
          refuse regardless, which is where the control is. */}
      {may('slack_configuration:manage') ? <SlackNotificationsPanel /> : null}
      {may('design_source:manage') ? <DesignSourcePanel /> : null}
      <Panel title="Repositories" meta="Saved so they are not retyped for every feature" flush>
        <SavedRepositories />
      </Panel>
      <DeveloperInformation />
    </div>
  );
}

/** The account this browser is acting as, as the authentication model describes it. */
function Profile() {
  const api = useApi();
  const me = useQuery({ queryKey: ['me'], queryFn: ({ signal }) => api.getMe(signal) });

  return (
    <Async query={me}>
      {(actor) => (
        <>
          <DetailList narrow>
            <DetailRow label="Name">{actor.display_name}</DetailRow>
            {/* The email first, because it is what somebody recognises as theirs. The account
                id is below it and is a machine string -- for the administrator it is
                permanently the literal `platform-admin`, which is a cosmetic cost of never
                rewriting an `owner_id`. */}
            {actor.subject ? <DetailRow label="Email">{actor.subject}</DetailRow> : null}
            <DetailRow label="Account">{actor.actor_id}</DetailRow>
            <DetailRow label="Roles">{actor.roles.join(', ') || 'none'}</DetailRow>
          </DetailList>
          <SlackLinkSettings />
          {actor.authentication === 'platform_key' ? (
            // Kept, but here rather than in the header. It changes what an audit record can
            // say, so somebody looking at their own account should be able to find it — and
            // nobody needs it announced beside their name on every screen.
            <p className="muted">
              This browser is signed in with the deployment’s shared administrative key, so
              actions are recorded against <code>{actor.actor_id}</code> rather than against a
              named person. Signing in with your own email and password is recorded against
              you, and is what this deployment expects.
            </p>
          ) : (
            <div className="form__actions">
              <Link className="button" to="/account/password">
                Change password
              </Link>
            </div>
          )}
          <p className="muted">Sign out from the account menu in the top right.</p>
        </>
      )}
    </Async>
  );
}

/**
 * Provider keys the platform holds for this account.
 *
 * It shows whether one is configured and its last four characters. It cannot show more,
 * because the platform does not have more to show: the value is sealed at rest and there is
 * no endpoint that returns it.
 *
 * These are no longer only a convenience. A feature is queued and executed after the request
 * that asked for it has been answered, so the work has no header to read — it resolves the
 * keys stored here, which is why creating a feature requires them.
 */
function CredentialSettings() {
  const api = useApi();
  const queryClient = useQueryClient();
  const credentials = useQuery({
    queryKey: ['credentials'],
    queryFn: ({ signal }) => api.listCredentials(signal),
    // A deployment that stores no credentials answers 503. That is a fact about the
    // deployment, not a failure, so it is not retried into an error box.
    retry: false,
  });

  if (credentials.isError) {
    const error = credentials.error;
    if (error instanceof ApiError && error.status === 503) {
      return (
        <p className="muted">
          This deployment does not store provider credentials. Supply them when you start a
          feature or grant a retry; they are sent as request headers and never kept.
        </p>
      );
    }
    return <p className="field__error">{userMessage(error as ApiError)}</p>;
  }

  return (
    <>
      <p className="muted">
        Kept encrypted, used by the work the platform does for you, and never shown back. A key
        typed into a form for one request still wins over the one stored here.
      </p>
      <p className="subtle">
        {/* Two buttons, two different questions, and the distinction is the whole reason the
            second exists: run 190's GitHub token had expired at midnight, "Check" reported it
            readable all day, and every clone the platform attempted was refused. */}
        <strong>Check</strong> reads the stored key here, which is what a changed encryption
        key breaks. <strong>Verify</strong> asks the provider whether it still accepts the key —
        the only question an expired or revoked one answers differently.
      </p>
      <Async query={credentials}>
        {(data) => (
          <ul className="cards" aria-label="Provider credentials">
            {data.credentials.map((credential) => (
              <CredentialRow
                key={credential.provider}
                credential={credential}
                onChanged={() => {
                  void queryClient.invalidateQueries({ queryKey: ['credentials'] });
                  // The New Feature gate reads the same answer from `/setup`, so it has to be
                  // refetched or the form stays blocked after the key it wanted was saved.
                  void queryClient.invalidateQueries({ queryKey: ['setup'] });
                }}
              />
            ))}
          </ul>
        )}
      </Async>
    </>
  );
}

function CredentialRow({
  credential,
  onChanged,
}: {
  credential: Credential;
  onChanged: () => void;
}) {
  const api = useApi();
  const [secret, setSecret] = useState('');
  const [editing, setEditing] = useState(false);

  const store = useMutation({
    mutationFn: () => api.storeCredential(credential.provider, secret),
    onSuccess: () => {
      setSecret('');
      setEditing(false);
      onChanged();
    },
  });
  const remove = useMutation({
    mutationFn: () => api.removeCredential(credential.provider),
    onSuccess: onChanged,
  });
  // One mutation for both questions, because they answer into the same place. Which one was
  // asked is carried in the response's `verified` field rather than in local state, so the
  // panel cannot label a local-only answer as a provider verdict.
  const check = useMutation({
    mutationFn: (options: { verify: boolean }) =>
      api.checkCredential(credential.provider, options),
  });

  const label = PROVIDER_LABELS[credential.provider] ?? credential.provider;

  return (
    <li className="card">
      <div className="card__header">
        <strong>{label}</strong>
        <Badge tone={credential.configured ? 'done' : 'neutral'}>
          {credential.configured ? 'Configured' : 'Not configured'}
        </Badge>
      </div>

      {credential.configured ? (
        <DetailList narrow>
          {/* The last four characters. Enough to recognise your own key; useless to anybody
              else, which is the whole reason it is the only part shown. */}
          <DetailRow label="Ends with">{`…${credential.hint}`}</DetailRow>
          {credential.updated_at ? (
            <DetailRow label="Updated">{absoluteTime(credential.updated_at)}</DetailRow>
          ) : null}
          {credential.last_used_at ? (
            <DetailRow label="Last used">{absoluteTime(credential.last_used_at)}</DetailRow>
          ) : null}
        </DetailList>
      ) : null}

      {check.data ? <CredentialCheckResult result={check.data} /> : null}

      {/* What GitHub said about the token that was just saved. Only a save carries it, and
          only GitHub answers it, so it appears once and is replaced by the next save. */}
      {store.data?.access ? <GitHubAccessSummary access={store.data.access} /> : null}

      {/* The server's own sentence when it wrote one. A GitHub token refused for a missing
          scope is refused with the scope named, and "the request was not valid" would throw
          away the only part of that answer somebody can act on. */}
      {[store.error, remove.error, check.error].map((error, index) =>
        error ? (
          <p className="field__error" role="alert" key={index}>
            {error instanceof ApiError ? refusalMessage(error) : 'That did not work.'}
          </p>
        ) : null,
      )}

      {editing ? (
        <form
          className="form"
          onSubmit={(event) => {
            event.preventDefault();
            if (secret.trim()) store.mutate();
          }}
        >
          <label className="field__label" htmlFor={`secret-${credential.provider}`}>
            {label} key
          </label>
          <input
            id={`secret-${credential.provider}`}
            type="password"
            autoComplete="off"
            value={secret}
            onChange={(event) => setSecret(event.target.value)}
          />
          <div className="form__actions">
            <button
              type="submit"
              className="button button--primary"
              disabled={store.isPending || !secret.trim()}
            >
              {store.isPending ? 'Saving…' : 'Save'}
            </button>
            <button
              type="button"
              className="button button--quiet"
              onClick={() => {
                setSecret('');
                setEditing(false);
              }}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <div className="form__actions">
          <button type="button" className="button" onClick={() => setEditing(true)}>
            {credential.configured ? 'Replace' : 'Add'}
          </button>
          {credential.configured ? (
            <>
              <button
                type="button"
                className="button button--quiet"
                disabled={check.isPending}
                onClick={() => check.mutate({ verify: false })}
              >
                {/* Which button is busy comes from the mutation's own variables, so pressing
                    one does not make the other announce work it is not doing. */}
                {check.isPending && !check.variables.verify ? 'Checking…' : 'Check'}
              </button>
              <button
                type="button"
                className="button button--quiet"
                disabled={check.isPending}
                // Named for what it asks rather than for what it costs, with the cost stated
                // where it can be read before clicking: this sends the stored key to the
                // provider, which is why it is a separate press and not part of Check.
                title={`Ask ${label} whether it still accepts this key`}
                onClick={() => check.mutate({ verify: true })}
              >
                {check.isPending && check.variables.verify ? 'Verifying…' : 'Verify'}
              </button>
              <button
                type="button"
                className="button button--quiet"
                disabled={remove.isPending}
                onClick={() => remove.mutate()}
              >
                {remove.isPending ? 'Removing…' : 'Remove'}
              </button>
            </>
          ) : null}
        </div>
      )}
    </li>
  );
}

const PROVIDER_LABELS: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  github: 'GitHub',
  slack: 'Slack',
  figma: 'Figma',
};

/**
 * The repositories this account works with.
 *
 * The point is that a feature does not ask for a URL and a branch again every time. The name
 * is derived from the URL and the identifier is the server's, so the form asks for the three
 * things only a person knows: which repository, which branch, and what to call the kind of
 * thing it is.
 */
function SavedRepositories() {
  const api = useApi();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState<SavedRepository | 'new' | null>(null);
  const saved = useQuery({
    queryKey: ['saved-repositories'],
    queryFn: ({ signal }) => api.listSavedRepositories(signal),
    retry: false,
  });
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['saved-repositories'] });
    void queryClient.invalidateQueries({ queryKey: ['setup'] });
  };
  const remove = useMutation({
    mutationFn: (configurationId: string) => api.deleteSavedRepository(configurationId),
    onSuccess: refresh,
  });

  if (saved.isError && saved.error instanceof ApiError && saved.error.status === 503) {
    return (
      <p className="muted" style={{ padding: 'var(--space-5)' }}>
        This deployment does not save repository configurations. Name the repositories on the
        feature itself instead.
      </p>
    );
  }

  return (
    <Async query={saved}>
      {(data) => (
        <div className="stack" style={{ padding: 'var(--space-5)', gap: 'var(--space-4)' }}>
          {data.repositories.length === 0 && editing === null ? (
            <EmptyState
              title="No repositories saved"
              detail="Save the repositories your team works with so they can be reused across features."
              action={
                <button
                  type="button"
                  className="button button--primary"
                  onClick={() => setEditing('new')}
                >
                  <IconPlus />
                  Add repository
                </button>
              }
            />
          ) : null}

          {data.repositories.length > 0 ? (
            <ul className="cards" aria-label="Saved repositories">
              {data.repositories.map((repository) => (
                <li className="card" key={repository.configuration_id}>
                  <div className="card__header">
                    <strong>{repository.name}</strong>
                    <Badge outline>{repository.repository_type}</Badge>
                  </div>
                  <DetailList narrow>
                    <DetailRow label="URL">
                      <a href={repository.repository_url} target="_blank" rel="noreferrer">
                        {displayUrl(repository.repository_url)}
                      </a>
                    </DetailRow>
                    <DetailRow label="Default branch">
                      <code className="mono">{repository.default_branch}</code>
                    </DetailRow>
                  </DetailList>
                  <div className="form__actions">
                    <button
                      type="button"
                      className="button"
                      onClick={() => setEditing(repository)}
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      className="button button--quiet"
                      disabled={remove.isPending}
                      onClick={() => remove.mutate(repository.configuration_id)}
                    >
                      Remove
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : null}

          {remove.error ? (
            <p className="field__error" role="alert">
              {remove.error instanceof ApiError
                ? userMessage(remove.error)
                : 'That repository could not be removed.'}
            </p>
          ) : null}

          {editing !== null ? (
            <RepositoryForm
              repository={editing === 'new' ? null : editing}
              suggestedTypes={data.suggested_types}
              onDone={() => {
                setEditing(null);
                refresh();
              }}
              onCancel={() => setEditing(null)}
            />
          ) : data.repositories.length > 0 ? (
            <div className="form__actions">
              <button type="button" className="button" onClick={() => setEditing('new')}>
                <IconPlus />
                Add repository
              </button>
            </div>
          ) : null}
        </div>
      )}
    </Async>
  );
}

/**
 * Add or edit one saved repository, choosing from what the account's GitHub token can reach.
 *
 * This used to be a URL field, and a URL field is a place to make two guesses at once: that
 * the repository exists at that spelling, and that your token can reach it. Both were answered
 * by a run failing hours later. GitHub knows both, so the form asks GitHub and offers the
 * answer — and the branch comes prefilled from the repository rather than from a guess about
 * what this deployment's repositories tend to call theirs.
 *
 * Repositories already saved are left out; ones the account cannot push to are shown and
 * disabled rather than hidden, because a repository that silently is not in the menu is
 * indistinguishable from one that does not exist.
 */
function RepositoryForm({
  repository,
  suggestedTypes,
  onDone,
  onCancel,
}: {
  repository: SavedRepository | null;
  suggestedTypes: string[];
  onDone: () => void;
  onCancel: () => void;
}) {
  const api = useApi();
  const [url, setUrl] = useState(repository?.repository_url ?? '');
  const [branch, setBranch] = useState(repository?.default_branch ?? '');
  const [type, setType] = useState(repository?.repository_type ?? suggestedTypes[0] ?? 'Other');

  const available = useQuery({
    queryKey: ['github-repositories'],
    queryFn: ({ signal }) => api.listGitHubRepositories(signal),
    // Asked once per form rather than kept fresh: the answer only changes when somebody
    // edits a grant on GitHub, and re-fetching under an open menu would move the options.
    staleTime: 60_000,
    retry: false,
  });

  const save = useMutation({
    mutationFn: () => {
      const body = {
        repository_url: url.trim(),
        default_branch: branch.trim(),
        repository_type: type,
      };
      return repository === null
        ? api.saveRepository(body)
        : api.updateSavedRepository(repository.configuration_id, body);
    },
    onSuccess: onDone,
  });

  const derived = derivedName(url);
  const offered = (available.data?.repositories ?? []).filter(
    // The one being edited stays selectable even though it is saved; every other saved one
    // would only be refused as a duplicate.
    (option) => !option.already_saved || option.repository_url === repository?.repository_url,
  );
  // A repository saved before this form asked GitHub — or one on another host — is still
  // editable: it is pinned into the menu as itself so its branch and label can be changed.
  const unlisted =
    repository !== null &&
    !offered.some((option) => option.repository_url === repository.repository_url);

  return (
    <form
      className="form"
      onSubmit={(event) => {
        event.preventDefault();
        if (url.trim() && branch.trim()) save.mutate();
      }}
    >
      <div className="field">
        <label className="field__label" htmlFor="repository-url">
          Repository
        </label>
        {available.isPending ? (
          <p className="field__hint">Asking GitHub which repositories your token can reach…</p>
        ) : null}
        {/* Every way of having no menu — no token stored, a deployment that asks GitHub
            nothing, a GitHub that would not answer — is a sentence the server wrote. Shown
            rather than guessed at, and shown instead of an empty menu, which is
            indistinguishable from an answer of "you can reach nothing". */}
        {available.isError ? (
          <p className="field__hint" role="status">
            {/* The server's sentence whenever it wrote one — a deployment that keeps no
                credentials explains itself far better than any wording here could. */}
            {available.error instanceof ApiError && available.error.detail
              ? available.error.detail
              : 'GitHub could not be asked which repositories your token can reach.'}
          </p>
        ) : null}
        {available.data?.detail ? (
          <p className="field__hint" role="status">
            {available.data.detail}
          </p>
        ) : null}
        {/* The one empty menu the server has no sentence for: it answered, and everything it
            offered is already here. Said out loud for the reason every other empty state is. */}
        {available.data?.available && offered.length === 0 && !unlisted ? (
          <p className="field__hint" role="status">
            Every repository your GitHub token reaches is already saved.
          </p>
        ) : null}
        {available.data?.access.truncated ? (
          <p className="field__hint">
            Only your most recently pushed repositories are listed. One you expected may be
            below that line.
          </p>
        ) : null}
        <select
          id="repository-url"
          value={url}
          disabled={available.isPending}
          onChange={(event) => {
            const chosen = event.target.value;
            setUrl(chosen);
            // The repository's own default branch, which is the answer somebody would
            // otherwise be typing from memory.
            const option = offered.find((item) => item.repository_url === chosen);
            if (option) setBranch(option.default_branch);
          }}
        >
          <option value="">Choose a repository…</option>
          {unlisted ? (
            <option value={repository.repository_url}>
              {displayUrl(repository.repository_url)} — currently saved
            </option>
          ) : null}
          {offered.map((option) => (
            <option
              key={option.repository_url}
              value={option.repository_url}
              // Disabled rather than absent, so "you cannot push to this" is legible where
              // somebody is looking for the repository they expected to find.
              disabled={!option.can_push || option.archived}
            >
              {option.full_name}
              {option.archived ? ' — archived' : ''}
              {!option.archived && !option.can_push ? ' — read only' : ''}
              {option.private ? ' (private)' : ''}
            </option>
          ))}
        </select>
        {/* Shown once chosen so the derivation is visible rather than a surprise on save. */}
        {derived ? (
          <p className="field__hint">
            Saved as <code className="mono">{derived}</code>
          </p>
        ) : null}
      </div>
      <div className="field">
        <label className="field__label" htmlFor="repository-branch">
          Default branch
        </label>
        <input
          id="repository-branch"
          value={branch}
          onChange={(event) => setBranch(event.target.value)}
        />
      </div>
      <div className="field">
        <label className="field__label" htmlFor="repository-type">
          Type
        </label>
        {/* A list of suggestions rather than a closed set: the label is the person's, and a
            fixed menu of somebody else's repository kinds is the assumption this platform
            avoids everywhere else. */}
        <input
          id="repository-type"
          list="repository-type-suggestions"
          value={type}
          onChange={(event) => setType(event.target.value)}
        />
        <datalist id="repository-type-suggestions">
          {suggestedTypes.map((item) => (
            <option key={item} value={item} />
          ))}
        </datalist>
      </div>
      {save.error ? (
        <p className="field__error" role="alert">
          {save.error instanceof ApiError ? refusalMessage(save.error) : 'That did not save.'}
        </p>
      ) : null}
      <div className="form__actions">
        <button
          type="submit"
          className="button button--primary"
          disabled={save.isPending || !url.trim() || !branch.trim()}
        >
          {save.isPending ? 'Saving…' : repository === null ? 'Save repository' : 'Save changes'}
        </button>
        <button type="button" className="button button--quiet" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

/**
 * The build and schema the API is running, and whether it is ready.
 *
 * Real and occasionally essential — a fix that was never deployed is indistinguishable from a
 * fix that did not work — but it is engineering information, not a setting. Collapsed, last,
 * and named for what it is.
 */
function DeveloperInformation() {
  const api = useApi();
  const readiness = useQuery({
    queryKey: ['readiness'],
    queryFn: ({ signal }) => api.getReadiness(signal),
    refetchInterval: 60_000,
  });

  return (
    <details className="advanced">
      <summary className="advanced__summary">
        <span className="advanced__title">Developer information</span>
        <span className="subtle">Which build of the API this browser is talking to.</span>
      </summary>
      <div className="advanced__body">
        <Async query={readiness}>
          {(data) => (
            <DetailList narrow>
              <DetailRow label="Status">{data.status}</DetailRow>
              <DetailRow label="Build">
                <code className="mono">{data.build_revision.slice(0, 12)}</code>
              </DetailRow>
              <DetailRow label="Workflow schema">{data.workflow_schema_version}</DetailRow>
              <DetailRow label="Runtime compatible">
                {data.runtime_compatible ? 'yes' : 'no'}
              </DetailRow>
            </DetailList>
          )}
        </Async>
      </div>
    </details>
  );
}

/** The repository name the server will derive, previewed while typing. Never sent. */
function derivedName(url: string): string | null {
  try {
    const segments = new URL(url.trim()).pathname
      .replace(/\.git$/, '')
      .split('/')
      .filter(Boolean);
    return segments.at(-1) ?? null;
  } catch {
    return null;
  }
}

/** A repository URL without its scheme, which is never the interesting part. */
function displayUrl(url: string): string {
  return url.replace(/^https?:\/\//, '');
}
