import { useEffect, useRef, type ReactNode } from 'react';
import { IconClose } from './icons';

/**
 * Deep context, without leaving the page that referenced it.
 *
 * The evidence behind a row -- a validation check's output, a review finding, one agent run --
 * is read while looking at the table it came from. Navigating away and back loses the reader's
 * place in a table of forty rows; a modal that has to be scrolled is worse. So: a panel beside
 * the page, dismissable, with the page still there underneath.
 */
export function Drawer({
  title,
  subtitle,
  onDismiss,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  onDismiss: () => void;
  children: ReactNode;
}) {
  const container = useRef<HTMLDivElement>(null);
  const opener = useRef<Element | null>(null);

  useEffect(() => {
    opener.current = document.activeElement;
    container.current?.focus();
    return () => {
      if (opener.current instanceof HTMLElement) opener.current.focus();
    };
  }, []);

  return (
    <>
      <div className="drawer-scrim" onMouseDown={onDismiss} />
      <aside
        className="drawer"
        role="dialog"
        aria-modal="false"
        aria-label={typeof title === 'string' ? title : 'Details'}
        tabIndex={-1}
        ref={container}
        onKeyDown={(event) => {
          if (event.key === 'Escape') onDismiss();
        }}
      >
        <header className="drawer__header">
          <div className="stack stack--tight" style={{ minWidth: 0 }}>
            <p className="drawer__title">{title}</p>
            {subtitle ? <div className="muted">{subtitle}</div> : null}
          </div>
          <button
            type="button"
            className="button button--quiet button--icon"
            style={{ marginLeft: 'auto' }}
            onClick={onDismiss}
            aria-label="Close"
          >
            <IconClose />
          </button>
        </header>
        <div className="drawer__body">{children}</div>
      </aside>
    </>
  );
}
