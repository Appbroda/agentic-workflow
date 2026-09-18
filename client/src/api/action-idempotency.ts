/** Browser-refresh-safe identity for one unresolved workflow mutation. */

const PREFIX = 'platform.pending-action.';
const fallback = new Map<string, string>();

function storage(): Storage | undefined {
  try {
    return globalThis.sessionStorage;
  } catch {
    return undefined;
  }
}

function createKey(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function pendingActionKey(scope: string): string {
  const name = `${PREFIX}${scope}`;
  const persisted = storage()?.getItem(name) ?? fallback.get(name);
  if (persisted) return persisted;
  const created = createKey();
  storage()?.setItem(name, created);
  fallback.set(name, created);
  return created;
}

export function clearPendingActionKey(scope: string): void {
  const name = `${PREFIX}${scope}`;
  storage()?.removeItem(name);
  fallback.delete(name);
}
