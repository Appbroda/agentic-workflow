import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

/**
 * The page furniture: headers, panels, tabs and the filter chips that double as a summary.
 *
 * These exist so that "a titled box with a table in it" is one thing rather than fifteen
 * slightly different ones. Every page in the application is built from them.
 */

export function PageHeader({
  title,
  eyebrow,
  subtitle,
  badges,
  actions,
  footer,
}: {
  title: ReactNode;
  /**
   * A small line above the title. Outside the heading on purpose: a feature's reference is
   * what somebody arrived holding, but the heading is what the page is *about*, and folding
   * the identifier into it would make every screen reader announce both as one name.
   */
  eyebrow?: ReactNode;
  subtitle?: ReactNode;
  badges?: ReactNode;
  actions?: ReactNode;
  /** Secondary metadata below the description — the engineering identifiers, typically. */
  footer?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div className="page-header__text">
        {eyebrow ? <div className="page-header__eyebrow">{eyebrow}</div> : null}
        <div className="page-header__title">
          <h1>{title}</h1>
          {badges}
        </div>
        {subtitle ? <div className="page-header__description">{subtitle}</div> : null}
        {footer}
      </div>
      {actions ? <div className="page-header__actions">{actions}</div> : null}
    </header>
  );
}

export function Panel({
  title,
  meta,
  actions,
  flush,
  children,
}: {
  title?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  /** Set when the panel's content is a table, which draws its own edges. */
  flush?: boolean;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      {title || actions ? (
        <header className="panel__header">
          <h2 className="panel__title">{title}</h2>
          {meta ? <span className="panel__meta">{meta}</span> : null}
          {actions ? <div className="panel__actions">{actions}</div> : null}
        </header>
      ) : null}
      <div className={flush ? 'panel__body panel__body--flush' : 'panel__body'}>{children}</div>
    </section>
  );
}

export interface TabDefinition {
  id: string;
  label: string;
  to: string;
  count?: number;
  urgent?: boolean;
}

/**
 * Section navigation, as links rather than local state.
 *
 * A view reached by clicking a tab is a place: it must survive a refresh, be linkable to a
 * colleague, and respond to the back button. State would give none of those.
 */
export function LinkTabs({ label, tabs, isActive }: {
  label: string;
  tabs: TabDefinition[];
  isActive: (tab: TabDefinition) => boolean;
}) {
  return (
    <nav className="tabs" aria-label={label}>
      {tabs.map((tab) => {
        const active = isActive(tab);
        return (
          <NavLink
            key={tab.id}
            to={tab.to}
            aria-current={active ? 'page' : undefined}
            className={[
              'tab',
              active ? 'tab--active' : '',
              tab.urgent && (tab.count ?? 0) > 0 ? 'tab--urgent' : '',
            ]
              .filter(Boolean)
              .join(' ')}
          >
            {tab.label}
            {tab.count !== undefined && tab.count > 0 ? (
              <span className="tab__count">{tab.count}</span>
            ) : null}
          </NavLink>
        );
      })}
    </nav>
  );
}

/** Two or three mutually exclusive views of the same data, switched in place. */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="tab"
          aria-selected={value === option.value}
          className={value === option.value ? 'segmented__option segmented__option--active' : 'segmented__option'}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export interface FilterChip<T extends string> {
  value: T;
  label: string;
  count: number;
  tone?: 'attention' | 'stopped' | 'done' | 'working';
}

/**
 * A count that is also the filter for what it counted.
 *
 * The dashboard summary is these. A number nobody can act on is decoration; the same number
 * that narrows the table below it is a control.
 *
 * Two sizes, same control. `metric` gives each count a tile with the number set large -- for
 * the summary row above a list, where the number is what somebody came to read. Without it
 * they are inline chips, which is what a row of event-kind filters inside a toolbar needs;
 * tiles there would push the toolbar to four times its height for information nobody is
 * scanning.
 */
export function FilterChips<T extends string>({
  label,
  value,
  chips,
  metric = false,
  onChange,
}: {
  label: string;
  value: T;
  chips: FilterChip<T>[];
  metric?: boolean;
  onChange: (value: T) => void;
}) {
  return (
    <div className={metric ? 'chips chips--metric' : 'chips'} role="tablist" aria-label={label}>
      {chips.map((chip) => (
        <button
          key={chip.value}
          type="button"
          role="tab"
          aria-selected={value === chip.value}
          // The name reads as a sentence -- "Waiting for you (2)" -- rather than as the
          // number-then-word order the eye scans.
          aria-label={`${chip.label} (${chip.count})`}
          className={[
            'chip',
            value === chip.value ? 'chip--active' : '',
            chip.tone ? `chip--${chip.tone}` : '',
          ]
            .filter(Boolean)
            .join(' ')}
          onClick={() => onChange(chip.value)}
        >
          <span className="chip__count">{chip.count}</span>
          <span>{chip.label}</span>
        </button>
      ))}
    </div>
  );
}
