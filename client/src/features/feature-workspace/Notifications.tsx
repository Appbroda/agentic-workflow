import { useMemo, useState } from 'react';
import type { FeatureEvent } from '@/schemas/feature';
import { RepositoryBadge } from '@/components/ui/Badge';
import { relativeTime, absoluteTime } from '@/utils/time';
import { noticesFrom } from './notices';

/**
 * What happened that somebody may need to act on.
 *
 * Rendered inline at the top of the workspace rather than as a toast or a modal. A feature can
 * sit waiting for an answer for hours, and a notice that disappears after four seconds is no
 * use to whoever opens the page afterwards; one that blocks the page is worse, because the
 * thing it is interrupting is usually the evidence needed to respond to it.
 *
 * `aria-live="polite"` so a new notice is announced without cutting off whatever is being read.
 */
export function Notifications({ events }: { events: FeatureEvent[] }) {
  const [dismissed, setDismissed] = useState<Set<number>>(() => new Set());
  const [expanded, setExpanded] = useState(false);
  const notices = useMemo(() => noticesFrom(events, dismissed), [events, dismissed]);

  if (notices.length === 0) return null;

  // This sits above every tab, so a feature with six repositories would otherwise push the
  // content a person came for below the fold. The most recent few are the news; the rest is
  // what the timeline is for.
  const shown = expanded ? notices : notices.slice(0, VISIBLE);
  const hidden = notices.length - shown.length;

  return (
    <ul className="alerts" aria-label="Notifications" aria-live="polite">
      {shown.map((notice) => (
        <li key={notice.id} className={`alerts__item alerts__item--${notice.tone}`}>
          <span>{notice.message}</span>
          {notice.repositoryId ? <RepositoryBadge repositoryId={notice.repositoryId} /> : null}
          <span className="subtle" title={absoluteTime(notice.timestamp)}>
            {relativeTime(notice.timestamp)}
          </span>
          <button
            type="button"
            className="button button--quiet button--small alerts__dismiss"
            // Dismissing hides the notice; it does not resolve anything. The feature's status
            // and the timeline remain the record of what is actually outstanding.
            onClick={() =>
              setDismissed((current) => new Set(current).add(notice.id))
            }
          >
            Dismiss
          </button>
        </li>
      ))}
      {hidden > 0 ? (
        <li className="alerts__item">
          <button type="button" className="button button--quiet" onClick={() => setExpanded(true)}>
            Show {hidden} more
          </button>
        </li>
      ) : null}
    </ul>
  );
}

const VISIBLE = 3;
