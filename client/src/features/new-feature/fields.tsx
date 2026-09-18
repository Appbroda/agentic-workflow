import { useId, type ReactNode } from 'react';

/** Small labelled inputs that keep every control associated with its label and its error. */

export function Field({
  label,
  error,
  hint,
  children,
}: {
  label: string;
  error?: string | undefined;
  hint?: string;
  children: (id: string, describedBy: string | undefined) => ReactNode;
}) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const describedBy = [hintId, errorId].filter(Boolean).join(' ') || undefined;
  return (
    <div className="field">
      <label className="field__label" htmlFor={id}>
        {label}
      </label>
      {children(id, describedBy)}
      {hint ? (
        <p className="field__hint" id={hintId}>
          {hint}
        </p>
      ) : null}
      {error ? (
        <p className="field__error" id={errorId} role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function Fieldset({
  legend,
  error,
  children,
}: {
  legend: string;
  error?: string | undefined;
  children: ReactNode;
}) {
  return (
    <fieldset className="fieldset">
      <legend>{legend}</legend>
      {error ? (
        <p className="field__error" role="alert">
          {error}
        </p>
      ) : null}
      {children}
    </fieldset>
  );
}

export function RemoveButton({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button type="button" className="button button--quiet" onClick={onClick} aria-label={label}>
      Remove
    </button>
  );
}
