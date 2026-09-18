import type { ReactNode } from 'react';
import { CopyButton } from './Value';

/**
 * Technical evidence, shown as itself.
 *
 * Commands, captured output and diffs are read character by character, so they are monospace,
 * scrollable in their own box, and never reflowed. What they must not do is stretch the page:
 * a 400-character stack trace inside a table cell used to widen every column on the screen.
 *
 * All of this is rendered as text. Output here comes from repositories and from agents, which
 * the platform treats as untrusted; nothing in it is ever interpreted as markup.
 */

export function CodeBlock({
  children,
  title,
  copyValue,
  wrap,
  tall,
}: {
  children: string;
  title?: ReactNode;
  copyValue?: string;
  wrap?: boolean;
  tall?: boolean;
}) {
  const classes = ['code', wrap ? 'code--wrap' : '', tall ? 'code--tall' : ''].filter(Boolean).join(' ');
  return (
    <div>
      {title || copyValue ? (
        <div className="code-header">
          <span className="truncate">{title}</span>
          {copyValue ? (
            <span style={{ marginLeft: 'auto' }}>
              <CopyButton value={copyValue} label="Copy" />
            </span>
          ) : null}
        </div>
      ) : null}
      <pre className={classes}>{children}</pre>
    </div>
  );
}

/**
 * Captured stdout or stderr.
 *
 * The platform stores bounded, redacted summaries rather than whole logs -- tool output never
 * lands in durable state -- so this renders what there is and says when there is nothing,
 * rather than leaving an empty box that looks like a loading failure.
 */
export function LogViewer({ label, text }: { label: string; text: string }) {
  if (!text.trim()) {
    return (
      <div className="stack stack--tight">
        <span className="details__label">{label}</span>
        <p className="muted">Nothing was captured.</p>
      </div>
    );
  }
  return (
    <div className="stack stack--tight">
      <span className="details__label">{label}</span>
      <pre className="code code--wrap code--short">{text}</pre>
    </div>
  );
}

/**
 * A unified diff, when one exists.
 *
 * The platform records which files an attempt changed and how -- added, modified, deleted --
 * but it does not record diff content, so most repositories have no hunks to show here and
 * the changed-files table is the honest view. This renders a diff only where a payload really
 * carries one, and reconstructs nothing.
 */
export function UnifiedDiff({ patch }: { patch: string }) {
  const lines = patch.replace(/\n$/, '').split('\n');
  return (
    <div className="diff" role="group" aria-label="Unified diff">
      {lines.map((line, index) => {
        const kind = line.startsWith('+') && !line.startsWith('+++')
          ? 'add'
          : line.startsWith('-') && !line.startsWith('---')
            ? 'del'
            : line.startsWith('@@')
              ? 'hunk'
              : null;
        return (
          <div key={index} className={kind ? `diff__line diff__line--${kind}` : 'diff__line'}>
            <span className="diff__sign" aria-hidden="true">
              {kind === 'add' ? '+' : kind === 'del' ? '-' : ''}
            </span>
            <span className="diff__text">{kind === 'add' || kind === 'del' ? line.slice(1) : line}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Raw JSON, always available and never the default view. */
export function RawJson({ value }: { value: unknown }) {
  return <pre className="raw-json">{JSON.stringify(value, null, 2)}</pre>;
}
