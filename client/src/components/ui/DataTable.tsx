import { Fragment, useMemo, useState, type ReactNode } from 'react';
import { IconChevronDown, IconChevronRight, IconChevronUp } from './icons';

/**
 * The one table in this application.
 *
 * Every dense list here -- features, repositories, validation checks, review findings, agent
 * runs, pull requests -- is the same object: a header, sortable columns, rows that may open
 * something, and a state for having nothing to show. Writing that once means a column added to
 * one table cannot quietly get different padding, a different empty state, or no sorting.
 *
 * Rows are clickable only through a real link inside the row. A `<tr>` with an onClick is
 * unreachable by keyboard and invisible to a screen reader; the row-level click here is a
 * convenience layered on top of that link, never the only way in.
 */

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  /** Supplying this makes the column sortable. Return null for "no value to sort by". */
  sortValue?: (row: T) => string | number | null;
  align?: 'right';
  /** Hold the column to its content, for badges, counts and single-icon cells. */
  shrink?: boolean;
}

type SortState = { key: string; direction: 'asc' | 'desc' } | null;

export function DataTable<T>({
  label,
  columns,
  rows,
  rowKey,
  onRowClick,
  expand,
  expandLabel = 'Show detail',
  empty,
  compact,
  initialSort,
}: {
  label: string;
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  /** Called when the row itself is clicked. The row must still contain a real link. */
  onRowClick?: (row: T) => void;
  /** Return the detail panel for a row to make it expandable. */
  expand?: (row: T) => ReactNode;
  expandLabel?: string;
  empty?: ReactNode;
  compact?: boolean;
  initialSort?: { key: string; direction: 'asc' | 'desc' };
}) {
  const [sort, setSort] = useState<SortState>(initialSort ?? null);
  const [open, setOpen] = useState<Set<string>>(() => new Set());

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const column = columns.find((item) => item.key === sort.key);
    if (!column?.sortValue) return rows;
    const direction = sort.direction === 'asc' ? 1 : -1;
    // A stable sort with absent values last in either direction: a repository with no pull
    // request should not float to the top merely because its cell is empty.
    return [...rows].sort((left, right) => {
      const a = column.sortValue!(left);
      const b = column.sortValue!(right);
      if (a === null && b === null) return 0;
      if (a === null) return 1;
      if (b === null) return -1;
      if (typeof a === 'number' && typeof b === 'number') return (a - b) * direction;
      return String(a).localeCompare(String(b)) * direction;
    });
  }, [rows, sort, columns]);

  if (rows.length === 0 && empty) return <>{empty}</>;

  const toggle = (key: string) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  return (
    <div className="table-wrap">
      <table className={compact ? 'table table--compact' : 'table'} aria-label={label}>
        <thead>
          <tr>
            {expand ? <th className="table__cell--shrink"><span className="sr-only">Detail</span></th> : null}
            {columns.map((column) => (
              <th
                key={column.key}
                className={
                  [column.shrink ? 'table__cell--shrink' : '', column.align === 'right' ? 'table__cell--number' : '']
                    .filter(Boolean)
                    .join(' ') || undefined
                }
                aria-sort={
                  sort?.key === column.key
                    ? sort.direction === 'asc'
                      ? 'ascending'
                      : 'descending'
                    : column.sortValue
                      ? 'none'
                      : undefined
                }
              >
                {column.sortValue ? (
                  <button
                    type="button"
                    onClick={() =>
                      setSort((current) =>
                        current?.key === column.key
                          ? { key: column.key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
                          : { key: column.key, direction: 'asc' },
                      )
                    }
                  >
                    {column.header}
                    {sort?.key === column.key ? (
                      sort.direction === 'asc' ? (
                        <IconChevronUp size={12} />
                      ) : (
                        <IconChevronDown size={12} />
                      )
                    ) : null}
                  </button>
                ) : (
                  column.header
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const key = rowKey(row);
            const expanded = open.has(key);
            return (
              <Fragment key={key}>
                <tr
                  className={[
                    onRowClick ? 'table__row--link' : '',
                    expanded ? 'table__row--expanded' : '',
                  ]
                    .filter(Boolean)
                    .join(' ') || undefined}
                  onClick={
                    onRowClick
                      ? (event) => {
                          // A click that landed on a control inside the row belongs to that
                          // control. Without this, opening a pull request also navigates.
                          const target = event.target as HTMLElement;
                          if (target.closest('a, button, input, select, textarea, details')) return;
                          onRowClick(row);
                        }
                      : undefined
                  }
                >
                  {expand ? (
                    <td className="table__cell--shrink">
                      <button
                        type="button"
                        className="button button--quiet button--icon button--small"
                        aria-expanded={expanded}
                        aria-label={expandLabel}
                        onClick={() => toggle(key)}
                      >
                        {expanded ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
                      </button>
                    </td>
                  ) : null}
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className={
                        [
                          column.shrink ? 'table__cell--shrink' : '',
                          column.align === 'right' ? 'table__cell--number' : '',
                        ]
                          .filter(Boolean)
                          .join(' ') || undefined
                      }
                    >
                      {column.render(row)}
                    </td>
                  ))}
                </tr>
                {expand && expanded ? (
                  <tr className="table__detail">
                    <td colSpan={columns.length + 1}>
                      <div className="table__detail-body">{expand(row)}</div>
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
