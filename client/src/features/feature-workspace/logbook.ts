import type { LogbookEntry } from '@/schemas/feature';

/**
 * Reading a feature's run as a conversation.
 *
 * Deliberately thin. Every sentence, every attribution and every quote in this tab is
 * composed server-side by `services/logbook.py` from records the platform already wrote —
 * there is no template here, and no code in this client that could compose one. That is the
 * point of item 9: a bubble that can say "I fixed it" when nothing was fixed is worse than
 * no logbook, and the way to be certain is to have nothing on the browser's side of the wire
 * that is capable of writing a sentence about the run.
 *
 * What is left for this file is presentation: where a "view record" link goes, and where the
 * thread breaks for a date header.
 */

/**
 * Where the record behind a bubble is read in full.
 *
 * A table rather than a derivation, and a client concern rather than a server one: which page
 * shows an artifact is this application's decision, so the server names the record and this
 * decides where to look at it. Every destination is a surface that already existed.
 */
export function recordHref(featureId: string, record: LogbookEntry['record']): string | null {
  const base = `/features/${encodeURIComponent(featureId)}`;
  switch (record.kind) {
    case 'artifact':
      return `${base}/artifacts?artifact=${encodeURIComponent(record.id)}`;
    // The journal row and the workstream row are both read on the repository's own page,
    // which is where the operations drill-in and the retry history live.
    case 'operation':
    case 'workstream':
      return record.repository_id
        ? `${base}/repositories/${encodeURIComponent(record.repository_id)}`
        : `${base}/workflow`;
    // Lifecycle events are the history tab's whole subject.
    case 'event':
      return `${base}/history`;
    default:
      // A record kind added server-side after this shipped. The bubble still renders and
      // still names its record; it simply has no page of its own here yet, which is a
      // missing link rather than a missing bubble.
      return null;
  }
}

/** What the "view record" link says, by the kind of thing it opens. */
export const RECORD_LABELS: Record<string, string> = {
  artifact: 'View record',
  operation: 'View journal',
  workstream: 'View repository',
  event: 'View history',
};

export interface LogbookDay {
  /** The ISO date the group covers, as the browser's own locale renders it. */
  label: string;
  entries: LogbookEntry[];
}

/**
 * Break the thread into day groups, in order.
 *
 * A run that took twenty minutes is one group and reads as one conversation; a run that was
 * resumed the next morning is two, and the gap is the thing a reader most needs to see. The
 * boundary is the viewer's own calendar day, because "did this happen today" is a question
 * about their day and not about UTC.
 */
export function groupByDay(entries: LogbookEntry[]): LogbookDay[] {
  const groups: LogbookDay[] = [];
  for (const entry of entries) {
    const label = dayLabel(entry.timestamp);
    const current = groups[groups.length - 1];
    if (current && current.label === label) current.entries.push(entry);
    else groups.push({ label, entries: [entry] });
  }
  return groups;
}

function dayLabel(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(undefined, {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
}

/** The four tones the server sends, mapped to the ones the rest of the UI already uses. */
const KNOWN_TONES = new Set(['done', 'stopped', 'attention', 'working']);

export function toneClass(tone: string): string {
  return KNOWN_TONES.has(tone) ? tone : 'working';
}
