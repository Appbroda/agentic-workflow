import { useState } from 'react';
import { Link } from 'react-router-dom';
import type { TimelineEvent } from '@/schemas/feature';
import { AgentBadge, RepositoryBadge } from '@/components/ui/Badge';
import { RawJson } from '@/components/ui/Code';
import { IconAttention, IconDone, IconRunning, IconStopped } from '@/components/ui/icons';
import { absoluteTime, eventTime } from '@/utils/time';
import { humanise } from '@/utils/text';
import { classify, type EventKind } from './event-kinds';

/**
 * What happened, in order.
 *
 * Each row says when, what, which repository, and which agent -- the four things somebody
 * debugging asks in that order. Technical detail is collapsed: a run produces sixty of these
 * and inlining every payload would make the column unreadable for the one row that matters.
 */
export function TimelineList({
  featureId,
  events,
}: {
  featureId: string;
  events: TimelineEvent[];
}) {
  return (
    <ol className="timeline" aria-label="Timeline">
      {events.map((event, index) => (
        <TimelineRow key={`${event.timestamp}-${index}`} featureId={featureId} event={event} />
      ))}
    </ol>
  );
}

const ICONS: Record<EventKind['tone'], typeof IconDone> = {
  done: IconDone,
  stopped: IconStopped,
  attention: IconAttention,
  working: IconRunning,
};

function TimelineRow({ featureId, event }: { featureId: string; event: TimelineEvent }) {
  const [open, setOpen] = useState(false);
  const kind = classify(event);
  const Icon = ICONS[kind.tone];
  const repository = typeof event.details.repository_id === 'string' ? event.details.repository_id : null;
  const artifactId = typeof event.details.artifact_id === 'string' ? event.details.artifact_id : null;
  // The bare `artifact_id` and `repository_id` are already columns of their own; anything else
  // the platform attached is what the expander is for.
  const extra = Object.entries(event.details).filter(
    ([key]) => key !== 'artifact_id' && key !== 'repository_id',
  );

  return (
    <li className="timeline__item">
      <span className="timeline__time" title={absoluteTime(event.timestamp)}>
        {eventTime(event.timestamp)}
      </span>
      <span className={`timeline__icon timeline__icon--${kind.tone}`} aria-hidden="true">
        <Icon size={14} />
      </span>
      <div className="timeline__body">
        <div className="timeline__headline">
          <span className="timeline__event">{kind.label ?? humanise(event.event)}</span>
          {repository ? <RepositoryBadge repositoryId={repository} /> : null}
          {event.source && event.source !== 'api' ? <AgentBadge agent={event.source} /> : null}
          {artifactId ? (
            <Link
              className="subtle"
              to={`/features/${encodeURIComponent(featureId)}/artifacts?artifact=${encodeURIComponent(artifactId)}`}
            >
              View artifact
            </Link>
          ) : null}
          {extra.length > 0 ? (
            <button
              type="button"
              className="button button--quiet button--small"
              aria-expanded={open}
              onClick={() => setOpen((value) => !value)}
            >
              {open ? 'Hide detail' : 'Detail'}
            </button>
          ) : null}
        </div>
        {open && extra.length > 0 ? <RawJson value={Object.fromEntries(extra)} /> : null}
      </div>
    </li>
  );
}
