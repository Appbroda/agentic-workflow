import { useEffect, useRef, type ReactNode } from 'react';
import { IconClose } from './icons';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * A modal for one short decision.
 *
 * Focus moves in when it opens, is held inside while it is open, and returns to whatever
 * opened it when it closes. Without the trap, somebody navigating by keyboard tabs from the
 * button they pressed straight past the confirmation into the rest of the page -- which for
 * "cancel this feature" is exactly the wrong thing to be able to do by accident.
 *
 * Deeper context belongs in a `Drawer` instead: a modal that has to be scrolled is a page
 * that should not have been a modal.
 */
export function Dialog({
  title,
  onDismiss,
  children,
}: {
  title: string;
  onDismiss: () => void;
  children: ReactNode;
}) {
  const container = useRef<HTMLDivElement>(null);
  const opener = useRef<Element | null>(null);

  useEffect(() => {
    opener.current = document.activeElement;
    const first = container.current?.querySelector<HTMLElement>(FOCUSABLE);
    (first ?? container.current)?.focus();
    return () => {
      if (opener.current instanceof HTMLElement) opener.current.focus();
    };
  }, []);

  return (
    <div className="scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) onDismiss(); }}>
      <div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        ref={container}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.stopPropagation();
            onDismiss();
            return;
          }
          if (event.key !== 'Tab') return;
          const focusable = [...(container.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? [])];
          const first = focusable.at(0);
          const last = focusable.at(-1);
          if (!first || !last) return;
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
          }
        }}
      >
        <div className="row row--between">
          <p className="dialog__title">{title}</p>
          <button type="button" className="button button--quiet button--icon" onClick={onDismiss} aria-label="Close">
            <IconClose />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
