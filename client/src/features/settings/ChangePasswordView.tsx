import { useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ApiError, userMessage } from '@/api/errors';
import { useApi } from '@/app/api-context';
import { PageHeader, Panel } from '@/components/ui/Layout';

/** The shortest password the platform stores. Mirrors `MIN_PASSWORD_LENGTH` on the server. */
const MINIMUM_LENGTH = 12;

/**
 * Replace your own password.
 *
 * A route rather than a modal, deliberately: it is what a forced first-time change is gated
 * behind, and a refresh must not bypass it. A modal is dismissed by reloading the page.
 *
 * Length is checked here so somebody is not told to try again after a round trip, and the
 * server checks it too -- in `hash_password`, which every path that sets a password goes
 * through. This is the courtesy; that is the rule.
 */
export function ChangePasswordView({ forced = false }: { forced?: boolean }) {
  const api = useApi();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | undefined>();
  const [submitting, setSubmitting] = useState(false);

  const localProblem =
    next && next.length < MINIMUM_LENGTH
      ? `A password must be at least ${MINIMUM_LENGTH} characters.`
      : next && confirm && next !== confirm
        ? 'The two new passwords do not match.'
        : undefined;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (submitting || localProblem || !current || !next || next !== confirm) return;
    setSubmitting(true);
    setError(undefined);
    try {
      await api.changePassword(current, next);
      // The `/me` answer carries `must_change_password`, and it has just changed. Invalidate
      // rather than clear: this is the same person, so their other cached data is still
      // theirs -- unlike signing in, where a clear is what stops the previous identity's
      // features being on screen.
      await queryClient.invalidateQueries({ queryKey: ['me'] });
      navigate(forced ? '/' : '/settings', { replace: true });
    } catch (cause) {
      setError(
        cause instanceof ApiError && cause.kind === 'forbidden'
          ? 'That current password is not correct.'
          : cause instanceof ApiError
            ? (cause.detail ?? userMessage(cause))
            : 'Could not change the password.',
      );
    } finally {
      setSubmitting(false);
      // Never left in component state, whether it worked or not.
      setCurrent('');
      setNext('');
      setConfirm('');
    }
  };

  return (
    <>
      <PageHeader
        title={forced ? 'Choose a password' : 'Change password'}
        subtitle={
          forced
            ? 'Your password was set for you, so it is known to somebody else. Pick your own to continue.'
            : 'Changing your password signs out your other browser sessions. API tokens you were issued keep working.'
        }
      />
      <Panel title="Password">
        <form className="form" onSubmit={submit}>
          <label className="field__label" htmlFor="password-current">
            Current password
          </label>
          <input
            id="password-current"
            type="password"
            autoComplete="current-password"
            value={current}
            onChange={(event) => setCurrent(event.target.value)}
          />
          <label className="field__label" htmlFor="password-new">
            New password
          </label>
          <input
            id="password-new"
            type="password"
            autoComplete="new-password"
            value={next}
            onChange={(event) => setNext(event.target.value)}
          />
          <label className="field__label" htmlFor="password-confirm">
            New password again
          </label>
          <input
            id="password-confirm"
            type="password"
            autoComplete="new-password"
            value={confirm}
            onChange={(event) => setConfirm(event.target.value)}
          />
          <p className="muted">
            At least {MINIMUM_LENGTH} characters. There are no composition rules: length is
            what makes a password hard to guess, and demanding punctuation mostly produces
            passwords people cannot remember.
          </p>
          {localProblem ?? error ? (
            <p className="callout callout--warn" role="alert">
              {localProblem ?? error}
            </p>
          ) : null}
          <div className="form__actions">
            <button
              type="submit"
              className="button button--primary"
              disabled={submitting || Boolean(localProblem) || !current || !next || !confirm}
            >
              {submitting ? 'Changing…' : 'Change password'}
            </button>
          </div>
        </form>
      </Panel>
    </>
  );
}
