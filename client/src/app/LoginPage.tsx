import { useState } from 'react';
import { ApiError, userMessage } from '@/api/errors';
import { BRAND_NAME, BrandLockup, CircuitField } from '@/components/ui/Brand';
import { IconAgent, IconEye, IconEyeOff, IconShield, IconSettings } from '@/components/ui/icons';
import { useApi } from './api-context';
import { useSession } from './session-context';

/**
 * Sign in with an email and a password.
 *
 * This replaces the gate that asked for the deployment's shared API key. That key was one
 * credential for everybody, readable by anyone who could open devtools, and it made every
 * audit answer the same answer: somebody with the key. What arrives here instead is a session
 * token minted for one account, which expires, can be revoked, and names a person.
 *
 * The mechanism underneath is unchanged: the token lives in `sessionStorage` and dies with
 * the tab. Only its provenance changed.
 *
 * Every failure is one message, because the server sends one: an unknown email, a wrong
 * password and a disabled account are all the same `401` with the same body. Rendering
 * anything more specific would invent a distinction the server deliberately refuses to make,
 * and would turn this form into a way to find out which addresses have accounts.
 *
 * It is also the first thing anybody sees of CRYN3T Systems, so it is the one screen in the
 * application with a display-size lockup and a drawing on it. What it is not is a marketing
 * page: the three lines on the brand panel are facts about how this deployment handles a
 * session, and there is no "remember me" or "forgot password" control, because this platform
 * has neither -- a password is reset by an administrator, which the footnote says.
 */
export function LoginPage() {
  const api = useApi();
  const { signIn } = useSession();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [revealed, setRevealed] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const [submitting, setSubmitting] = useState(false);

  const incomplete = !email.trim() || !password;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (incomplete || submitting) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const result = await api.login(email.trim(), password);
      // Cleared before the token is stored, and stored before anything renders as the new
      // identity -- `signIn` does both, in that order.
      setPassword('');
      setRevealed(false);
      signIn(result.token);
    } catch (cause) {
      setError(loginMessage(cause));
      // Kept out of state on failure too. A password sitting in a component's state after a
      // failed attempt is a password in a React devtools tree.
      setPassword('');
      // And it is not left legible on screen for the next attempt either.
      setRevealed(false);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth">
      <aside className="auth__brand">
        <CircuitField />
        <BrandLockup stacked />
        <div className="auth__pitch">
          <h2 className="auth__headline">
            Describe the change. The platform plans it, builds it, and opens the pull requests.
          </h2>
          <p className="auth__lede">
            {BRAND_NAME} takes a written requirement across your repositories — planning,
            implementation, review and publication — and records every step of it so the work
            can be read back long after the run has finished.
          </p>
        </div>
        {/* Three facts about this deployment, not three claims about the category. */}
        <ul className="auth__points">
          <li className="auth__point">
            <IconShield />
            <span>
              <strong>Your own account.</strong> A session is minted for one identity, expires,
              and can be revoked.
            </span>
          </li>
          <li className="auth__point">
            <IconSettings />
            <span>
              <strong>Your own keys.</strong> Provider credentials and repositories are held
              against your account and are not shared with other people on this deployment.
            </span>
          </li>
          <li className="auth__point">
            <IconAgent />
            <span>
              <strong>A full record.</strong> Every run keeps who asked, which model ran, and
              what it changed.
            </span>
          </li>
        </ul>
      </aside>

      <div className="auth__panel">
        <section className="auth__card">
          {/* Shown only below the small breakpoint, where the brand panel is not on screen. */}
          <BrandLockup className="auth__card-brand" />
          <div className="auth__heading">
            <h1>Sign in</h1>
            <p className="muted">Secure access to the {BRAND_NAME} control plane.</p>
          </div>

          {/* `aria-busy` rather than disabled fields: disabling the input somebody is typing
              in takes the focus away from them, and after a refused attempt they have to find
              their way back to it. The submit button is the control that goes unavailable. */}
          <form className="auth__form" onSubmit={submit} aria-busy={submitting}>
            <div className="field">
              <label className="field__label" htmlFor="login-email">
                Email
              </label>
              <input
                id="login-email"
                type="email"
                autoComplete="username"
                autoFocus
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </div>

            <div className="field">
              <label className="field__label" htmlFor="login-password">
                Password
              </label>
              {/* The reveal is a real button with its own name, so it is reachable from the
                  keyboard and announces which state it will put the field into. `aria-pressed`
                  carries the current state; the icon carries it again for everybody else. */}
              <span className="input-affix">
                <input
                  id="login-password"
                  type={revealed ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
                <button
                  type="button"
                  className="input-affix__button"
                  aria-label={revealed ? 'Hide password' : 'Show password'}
                  aria-pressed={revealed}
                  aria-controls="login-password"
                  onClick={() => setRevealed((current) => !current)}
                >
                  {revealed ? <IconEyeOff /> : <IconEye />}
                </button>
              </span>
            </div>

            {error ? (
              <p className="callout callout--stopped" role="alert">
                {error}
              </p>
            ) : null}

            <button
              type="submit"
              className={
                submitting
                  ? 'button button--primary button--large button--block button--loading'
                  : 'button button--primary button--large button--block'
              }
              disabled={submitting || incomplete}
            >
              {submitting ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Signing in…
                </>
              ) : (
                'Sign in'
              )}
            </button>
          </form>

          <div className="auth__footnote">
            <p>
              The session is kept for this browser tab only. Closing the tab signs you out.
            </p>
            <p>Ask an administrator if you need an account or a password reset.</p>
          </div>
        </section>
      </div>
    </main>
  );
}

/**
 * The sentence to show for a failed sign-in.
 *
 * `401` gets a fixed message rather than the server's, so that this client cannot become the
 * place a distinction leaks back in. Everything else — the deployment having no user
 * directory, the rate limit, the network — is a fact about the platform rather than about
 * the guess, and says so.
 */
function loginMessage(cause: unknown): string {
  if (!(cause instanceof ApiError)) return 'Could not sign in.';
  if (cause.kind === 'unauthorized') return 'That email and password do not match an account.';
  if (cause.status === 429) {
    return 'Too many sign-in attempts. Wait a few minutes and try again.';
  }
  if (cause.status === 503) {
    return 'This deployment cannot check passwords. Its user directory is not configured.';
  }
  return userMessage(cause);
}
