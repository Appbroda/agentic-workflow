import { createContext, useContext } from 'react';
import type { Actor } from '@/schemas/feature';

/**
 * Kept out of the provider module so that file exports only components, which is what lets
 * fast refresh work and is the rule `react-refresh/only-export-components` enforces. Same
 * split, for the same reason, as `api-context.ts`.
 */

export interface Session {
  /** The signed-in identity, or nothing while there is no usable credential. */
  actor: Actor | undefined;
  /** Whether the credential is still being resolved. Distinct from "no credential". */
  loading: boolean;
  /** Whether this browser holds a token at all, whatever the platform thinks of it. */
  hasToken: boolean;
  /**
   * Whether this session holds one named permission.
   *
   * Named permissions, never a role string: the server's `ROLE_PERMISSIONS` is the one
   * authority on what a role grants, and a client that tested `roles.includes('admin')`
   * would be a second one that silently disagrees the moment a grant moves -- which is
   * exactly what happened to Slack and design-source management.
   */
  may: (permission: string) => boolean;
  signIn: (token: string) => void;
  signOut: () => Promise<void>;
}

export const SessionContext = createContext<Session | undefined>(undefined);

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error('useSession used outside a SessionProvider');
  return session;
}
