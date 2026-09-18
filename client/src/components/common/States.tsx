import type { ReactNode } from 'react';
import { ApiError, userMessage } from '@/api/errors';
import { IconAttention, IconEmpty } from '@/components/ui/icons';

/**
 * Every async surface gets all three of these, and none of them is blank space.
 *
 * The rules they encode: a load shows the shape of what is coming rather than a spinner; an
 * empty result says what would put something there; a failure says what failed, whether
 * retrying could help, and leaves the rest of the page alone. One secondary request failing
 * must never blank a page whose other data arrived.
 *
 * The glyphs are decoration and are hidden from assistive technology: in each of the three the
 * title is the message, and a screen reader that announced "tray" before it would be reading
 * the illustration out loud.
 */

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="state state--loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span className="muted">{label}</span>
    </div>
  );
}

/** The shape of a table that has not arrived yet, so the page does not jump when it does. */
export function TableSkeleton({ rows = 5, label = 'Loading…' }: { rows?: number; label?: string }) {
  return (
    <div className="skeleton-rows" role="status" aria-live="polite" aria-label={label}>
      {Array.from({ length: rows }, (_, index) => (
        <span
          key={index}
          className="skeleton"
          style={{ width: `${[92, 76, 84, 64, 88, 72][index % 6]}%` }}
        />
      ))}
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
  icon,
}: {
  title: string;
  detail?: string;
  action?: ReactNode;
  /** Overrides the default tray glyph where a more specific one says more. Decorative. */
  icon?: ReactNode;
}) {
  return (
    <div className="state state--empty">
      <span className="state__icon" aria-hidden="true">
        {icon ?? <IconEmpty size={20} />}
      </span>
      <p className="state__title">{title}</p>
      {detail ? <p className="state__detail">{detail}</p> : null}
      {action}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const apiError = error instanceof ApiError ? error : undefined;
  const message = apiError ? userMessage(apiError) : 'Something went wrong.';
  return (
    <div className="state state--error" role="alert">
      <span className="state__icon" aria-hidden="true">
        <IconAttention size={20} />
      </span>
      <p className="state__title">{message}</p>
      {apiError?.detail ? (
        <details className="state__detail">
          <summary>Technical detail</summary>
          {/* Rendered as text, never as markup: this string comes from the backend. */}
          <pre>{apiError.detail}</pre>
        </details>
      ) : null}
      <div className="form__actions">
        {onRetry && (!apiError || apiError.isRetryable) ? (
          <button type="button" className="button" onClick={onRetry}>
            Try again
          </button>
        ) : null}
        {/* No "use a different key" control any more. A `401` has already cleared the
            session by the time this renders -- the API client does it, once, for every
            request -- so `SessionProvider` is about to show the login screen and a button
            offering to do the same thing would be a second way to end a session. */}
      </div>
    </div>
  );
}

/**
 * A section whose own data failed while the rest of the page is fine.
 *
 * It reports what could not be refreshed and offers to try again, in the space that section
 * occupies -- rather than replacing the page, which is what makes one failed secondary request
 * look like an outage.
 */
export function PartialError({
  what,
  error,
  onRetry,
}: {
  what: string;
  error: unknown;
  onRetry?: () => void;
}) {
  const apiError = error instanceof ApiError ? error : undefined;
  return (
    <div className="callout callout--warn" role="alert">
      <p className="callout__title">Could not load {what}.</p>
      {apiError ? <p className="muted">{userMessage(apiError)}</p> : null}
      {onRetry ? (
        <div className="form__actions">
          <button type="button" className="button button--small" onClick={onRetry}>
            Try again
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function Async<T>({
  query,
  empty,
  skeleton,
  children,
}: {
  query: { isPending: boolean; isError: boolean; error: unknown; data: T | undefined; refetch: () => void };
  empty?: ReactNode;
  skeleton?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) return <>{skeleton ?? <TableSkeleton />}</>;
  if (query.isError) return <ErrorState error={query.error} onRetry={query.refetch} />;
  if (query.data === undefined) return <>{empty ?? <EmptyState title="Nothing to show" />}</>;
  return <>{children(query.data)}</>;
}
