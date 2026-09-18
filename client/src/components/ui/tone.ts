/**
 * The tone vocabulary, kept apart from the components that paint it.
 *
 * The server publishes a tone with every status in `/console/status-vocabulary`. This is the
 * only place that decides what to do with a tone it does not recognise: show it as neutral,
 * never drop the badge. The server's lifecycle grows, and a client that rendered nothing for a
 * status added last week would hide the newest thing in the feature.
 */

export type Tone = 'working' | 'attention' | 'done' | 'stopped' | 'neutral';

const TONES = new Set<Tone>(['working', 'attention', 'done', 'stopped', 'neutral']);

export function toneOf(value: string | undefined | null): Tone {
  return value && TONES.has(value as Tone) ? (value as Tone) : 'neutral';
}

/** Severity words the review and preflight agents write, mapped onto the same five tones. */
export function severityTone(severity: string): Tone {
  const value = severity.trim().toLowerCase();
  if (value === 'critical' || value === 'blocker' || value === 'high') return 'stopped';
  if (value === 'medium' || value === 'moderate' || value === 'warning') return 'attention';
  if (value === 'low' || value === 'info' || value === 'informational') return 'working';
  return 'neutral';
}
