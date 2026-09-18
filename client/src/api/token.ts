/**
 * Where the credential this browser authenticates with comes from.
 *
 * It is a session token now, minted by `POST /auth/login` in exchange for an email and a
 * password. The mechanism below is unchanged: `sessionStorage`, so the credential belongs to
 * one tab, dies when that tab closes, and never reaches a build artefact. Only its
 * provenance changed -- it used to be the deployment's shared administrative key, typed in.
 *
 * `VITE_PLATFORM_API_KEY` is retired. It was substituted at build time, which put the
 * credential inside the bundle and therefore inside the image -- pushed to a registry,
 * pulled onto machines, kept in layer caches; it happened here, from a `.env` written for a
 * local browser check. It also *won* over the runtime value and could not be cleared, so in
 * any environment that set it, logging out silently did nothing and the next person to use
 * the tab was still signed in as whoever the key belonged to. Both reasons are why the
 * branch is gone rather than deprecated.
 */
const STORAGE_KEY = 'platform-api-token';

function storage(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    // Storage can be unavailable -- a locked-down browser, a sandboxed frame. The application
    // must still load and ask for a password; it just will not remember the session.
    return null;
  }
}

export function readToken(): string | undefined {
  return storage()?.getItem(STORAGE_KEY) ?? undefined;
}

export function writeToken(token: string): void {
  storage()?.setItem(STORAGE_KEY, token);
}

export function clearToken(): void {
  storage()?.removeItem(STORAGE_KEY);
}
