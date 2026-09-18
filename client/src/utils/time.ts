const MINUTE = 60;
const HOUR = 3_600;
const DAY = 86_400;
const WEEK = 604_800;

/** Short relative time for dense list rows. The absolute value stays available as a tooltip. */
export function relativeTime(iso: string, now: Date = new Date()): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return iso;

  const seconds = Math.round((now.getTime() - then.getTime()) / 1000);
  if (seconds < MINUTE) return 'just now';

  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  if (seconds < HOUR) return formatter.format(-Math.round(seconds / MINUTE), 'minute');
  if (seconds < DAY) return formatter.format(-Math.round(seconds / HOUR), 'hour');
  if (seconds < WEEK) return formatter.format(-Math.round(seconds / DAY), 'day');
  return then.toLocaleDateString();
}

export function absoluteTime(iso: string): string {
  const value = new Date(iso);
  return Number.isNaN(value.getTime()) ? iso : value.toLocaleString();
}

/**
 * The stamp on a timeline row.
 *
 * Clock time for today, date and clock time for anything older. A run can span days, and a
 * column of bare clock times reads as if everything happened this morning.
 */
export function eventTime(iso: string, now: Date = new Date()): string {
  const value = new Date(iso);
  if (Number.isNaN(value.getTime())) return iso;
  const time = value.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return value.toDateString() === now.toDateString()
    ? time
    : `${value.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${time}`;
}

/**
 * How long something took, or nothing when the two instants cannot say.
 *
 * A difference under a second is reported as unknown rather than as `0s`. Not every pair of
 * timestamps brackets real work -- some are written together after the fact -- and printing a
 * measurement the platform never took is worse than leaving the cell empty, because a reader
 * would believe it.
 */
export function elapsed(from: string, to: string | null | undefined): string | null {
  const start = Date.parse(from);
  const end = to ? Date.parse(to) : Date.now();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return formatDuration(end - start);
}

export function formatDuration(milliseconds: number): string | null {
  if (!Number.isFinite(milliseconds) || milliseconds < 1000) return null;
  const seconds = Math.round(milliseconds / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours}h ${minutes % 60}m` : `${Math.floor(hours / 24)}d ${hours % 24}h`;
}
