import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { clearToken, readToken, writeToken } from '@/api/token';
import { useApi } from './api-context';
import { SessionContext, type Session } from './session-context';

/**
 * Who is signed in, and the two operations that change the answer.
 *
 * It holds the `/me` result rather than re-fetching it per component: several places want the
 * display name, and two want the permission list to decide what to draw. One query means one
 * answer on screen at a time -- the sidebar and the page cannot disagree about whether this
 * person may manage users.
 *
 * `signOut` clears the *query cache* as well as the token, and that is not tidiness. Every
 * cached query was fetched as the identity that is leaving; without the clear, the next
 * person to sign in on the same tab sees the previous one's features until each query happens
 * to refetch. It is a real cross-user leak in the browser and an easy one to miss, so it is
 * here, once, in the function that ends a session -- not at the call sites of a button.
 */

export function SessionProvider({ children }: { children: ReactNode }) {
  const api = useApi();
  const queryClient = useQueryClient();
  // Mirrored into state so signing in or out re-renders. `sessionStorage` is not reactive,
  // and a component that only read it would keep showing the previous answer.
  const [token, setToken] = useState(() => readToken());

  const me = useQuery({
    queryKey: ['me'],
    queryFn: ({ signal }) => api.getMe(signal),
    enabled: Boolean(token),
    // A 401 has already cleared the token by the time this settles -- see the client's
    // `onUnauthorized` -- so retrying would spend three requests proving the same thing.
    retry: false,
    staleTime: 60_000,
  });

  const signIn = useCallback(
    (value: string) => {
      writeToken(value);
      // Before the state change, so nothing can render with the new identity and the
      // previous identity's cached data. `clear` rather than `invalidateQueries`: an
      // invalidated query keeps serving its stale data while it refetches, which is exactly
      // the previous person's features on screen.
      queryClient.clear();
      setToken(value);
    },
    [queryClient],
  );

  const signOut = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // The server being unreachable must not leave somebody signed in on this machine
      // after they pressed sign out. The token stops working here either way; if the
      // revocation did not land, the session expires on its own.
    }
    clearToken();
    queryClient.clear();
    setToken(undefined);
  }, [api, queryClient]);

  const value = useMemo<Session>(() => {
    const permissions = new Set(me.data?.permissions ?? []);
    return {
      actor: token ? me.data : undefined,
      loading: Boolean(token) && me.isLoading,
      hasToken: Boolean(token),
      // Named permissions, never a role string: the server's `ROLE_PERMISSIONS` is the one
      // authority on what a role grants, and a client that checked `roles.includes('admin')`
      // would be a second one that silently disagrees the moment a grant moves.
      may: (permission: string) => permissions.has(permission),
      signIn,
      signOut,
    };
  }, [me.data, me.isLoading, signIn, signOut, token]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}
