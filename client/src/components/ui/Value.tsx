import { useEffect, useRef, useState, type ReactNode } from 'react';
import { IconCheck, IconCopy, IconExternal } from './icons';

/**
 * Values that are too long for the space they are in.
 *
 * Repository names, branch names, workflow ids, commit hashes, commands and error text all
 * arrive at whatever length the platform gave them. A branch here is routinely 90 characters.
 * These primitives are how such a value gets into a table cell without widening the table:
 * truncate, keep the whole thing in the title, and offer a copy button so the truncation costs
 * nothing.
 */

export function CopyButton({ value, label = 'Copy' }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  return (
    <button
      type="button"
      className={copied ? 'copy copy--copied' : 'copy'}
      // The label says what will be copied, because a row can hold several of these.
      aria-label={copied ? 'Copied' : `${label}: ${value}`}
      title={copied ? 'Copied' : label}
      onClick={() => {
        // Not every browser context grants clipboard access -- an insecure origin, a denied
        // permission. Failing here must not break the row the button sits in.
        void navigator.clipboard
          ?.writeText(value)
          .then(() => {
            setCopied(true);
            if (timer.current) clearTimeout(timer.current);
            timer.current = setTimeout(() => setCopied(false), 1600);
          })
          .catch(() => undefined);
      }}
    >
      {copied ? <IconCheck size={13} /> : <IconCopy size={13} />}
    </button>
  );
}

/** A monospace value, truncated to its container, with the full text one click from the clipboard. */
export function CopyValue({
  value,
  display,
  label,
  mono = true,
}: {
  value: string;
  display?: string;
  label?: string;
  mono?: boolean;
}) {
  return (
    <span className="value-row">
      <span className={mono ? 'truncate mono' : 'truncate'} title={value}>
        {display ?? value}
      </span>
      <CopyButton value={value} label={label} />
    </span>
  );
}

/** A commit or revision: the short form on screen, the whole hash on the clipboard. */
export function Revision({ value }: { value: string }) {
  return <CopyValue value={value} display={value.slice(0, 8)} label="Copy revision" />;
}

export function Truncated({ text, title }: { text: string; title?: string }) {
  return (
    <span className="truncate" title={title ?? text}>
      {text}
    </span>
  );
}

/** An external link, marked as one and never handing the opener a window reference. */
export function ExternalLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="value-row">
      <span className="truncate">{children}</span>
      <IconExternal size={12} />
    </a>
  );
}

/** A label above a value. The unit the detail grids are built from. */
export function DetailRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="details__item">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

export function DetailList({ children, narrow }: { children: ReactNode; narrow?: boolean }) {
  return <dl className={narrow ? 'details details--narrow' : 'details'}>{children}</dl>;
}
