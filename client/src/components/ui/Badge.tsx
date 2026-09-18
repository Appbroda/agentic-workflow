import type { ReactNode } from 'react';
import type { StatusWording } from '@/hooks/useStatusVocabulary';
import { severityTone, toneOf, type Tone } from './tone';

/**
 * Every coloured label in the application.
 *
 * There is one tone vocabulary -- `working`, `attention`, `done`, `stopped`, `neutral` -- and
 * it is the server's, published with the status vocabulary. Nothing here invents a tone for a
 * status, and nothing outside this module writes a status colour.
 *
 * Status is never colour alone: each badge carries its word, and the lifecycle badges carry a
 * dot as well, so a greyscale screenshot and a colour-blind reader both still read it.
 */

export function Badge({
  tone = 'neutral',
  outline,
  mono,
  title,
  children,
}: {
  tone?: Tone;
  outline?: boolean;
  mono?: boolean;
  title?: string;
  children: ReactNode;
}) {
  const classes = ['badge', outline ? 'badge--outline' : `badge--${tone}`, mono ? 'badge--mono' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <span className={classes} title={title}>
      <span className="badge__text">{children}</span>
    </span>
  );
}

/**
 * A lifecycle status, in the server's words.
 *
 * `detail` goes on the title attribute rather than on the screen: it is the explanation, and
 * the badge's job in a dense table is to be scannable.
 */
export function StatusBadge({ wording }: { wording: StatusWording }) {
  return (
    <span className={`badge badge--${toneOf(wording.tone)}`} title={wording.detail || undefined}>
      <span className="badge__dot" aria-hidden="true" />
      <span className="badge__text">{wording.headline}</span>
    </span>
  );
}

/** Review and preflight severities, which the agents write in their own case. */
export function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span className={`badge badge--${severityTone(severity)}`}>
      <span className="badge__dot" aria-hidden="true" />
      <span className="badge__text">{severity.toUpperCase()}</span>
    </span>
  );
}

/**
 * A repository, by its id.
 *
 * Identity is the id and never the role: a feature can have five repositories and three of
 * them can share a role, so the role is shown separately where it is useful and is never used
 * to tell two repositories apart.
 */
export function RepositoryBadge({ repositoryId }: { repositoryId: string }) {
  return (
    <span className="badge badge--outline badge--mono" title={repositoryId}>
      <span className="badge__text">{repositoryId}</span>
    </span>
  );
}

/** Which agent did something. The name is the platform's own producer string. */
export function AgentBadge({ agent }: { agent: string }) {
  return (
    <span className="badge badge--neutral" title={agent}>
      <span className="badge__text">{agent.replace(/_/g, ' ')}</span>
    </span>
  );
}

const CHANGE_LETTERS: Record<string, { letter: string; label: string; className: string }> = {
  added: { letter: 'A', label: 'Added', className: 'change--added' },
  created: { letter: 'A', label: 'Added', className: 'change--added' },
  modified: { letter: 'M', label: 'Modified', className: 'change--modified' },
  changed: { letter: 'M', label: 'Modified', className: 'change--modified' },
  updated: { letter: 'M', label: 'Modified', className: 'change--modified' },
  deleted: { letter: 'D', label: 'Deleted', className: 'change--deleted' },
  removed: { letter: 'D', label: 'Deleted', className: 'change--deleted' },
  renamed: { letter: 'R', label: 'Renamed', className: 'change--renamed' },
  moved: { letter: 'R', label: 'Renamed', className: 'change--renamed' },
};

/**
 * The A/M/D/R letter for one changed file.
 *
 * The letter is the conventional shorthand and the accessible name is the whole word, so a
 * screen reader says "Modified" where a sighted reader scans a column of single letters. An
 * unrecognised change type keeps its own word rather than being forced into one of the four.
 */
export function ChangeBadge({ changeType }: { changeType: string }) {
  const known = CHANGE_LETTERS[changeType.trim().toLowerCase()];
  return (
    <span className={known ? `change ${known.className}` : 'change'} title={known?.label ?? changeType}>
      <span aria-hidden="true">{known?.letter ?? '?'}</span>
      <span className="sr-only">{known?.label ?? changeType}</span>
    </span>
  );
}
