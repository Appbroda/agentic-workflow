import { useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Async, EmptyState, TableSkeleton } from '@/components/common/States';
import { FilterChips, Panel, Segmented, type FilterChip } from '@/components/ui/Layout';
import { useTimeline } from './hooks';
import { TimelineList } from './TimelineList';
import { AgentHistory } from './AgentHistory';
import { CATEGORY_LABELS, TIMELINE_FILTERS, matchesFilter, type TimelineFilter } from './event-kinds';

/**
 * Everything that happened, two ways.
 *
 * The timeline is the record in order; the agent view is the same events read as "who ran, on
 * what, and how did it end". Both come from the same endpoint rather than a second source,
 * because a parallel history would drift from this one.
 *
 * The view and the filter live in the URL, so a colleague can be sent "the validation errors
 * on this feature" rather than told which two controls to press.
 */
export function HistoryTab({ featureId, at }: { featureId: string; at: number | null }) {
  const [params, setParams] = useSearchParams();
  const view = params.get('view') === 'agents' ? 'agents' : 'timeline';
  const filter = (params.get('filter') ?? 'all') as TimelineFilter;
  const timeline = useTimeline(featureId, at);

  const events = useMemo(() => [...(timeline.data?.events ?? [])].reverse(), [timeline.data]);
  const shown = useMemo(() => events.filter((event) => matchesFilter(event, filter)), [events, filter]);

  const chips: FilterChip<TimelineFilter>[] = TIMELINE_FILTERS.map(
    (value): FilterChip<TimelineFilter> => ({
      value,
      label: CATEGORY_LABELS[value],
      count: events.filter((event) => matchesFilter(event, value)).length,
      tone: value === 'errors' ? 'stopped' : undefined,
    }),
  ).filter((chip) => chip.count > 0 || chip.value === 'all');

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value === 'all' || value === 'timeline') next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true });
  };

  return (
    <Panel
      title="History"
      actions={
        <Segmented
          label="History view"
          value={view}
          options={[
            { value: 'timeline', label: 'Timeline' },
            { value: 'agents', label: 'Agent runs' },
          ]}
          onChange={(value) => set('view', value)}
        />
      }
      flush
    >
      {view === 'timeline' ? (
        <>
          <div className="toolbar">
            <FilterChips
              label="Event kinds"
              value={filter}
              chips={chips}
              onChange={(value) => set('filter', value)}
            />
          </div>
          <Async query={timeline} skeleton={<TableSkeleton rows={8} />}>
            {() =>
              shown.length === 0 ? (
                <EmptyState
                  title={events.length === 0 ? 'No events yet' : 'Nothing of that kind'}
                  detail={
                    events.length === 0
                      ? 'The platform records each step as it takes it.'
                      : 'Choose another category to see the rest of the run.'
                  }
                />
              ) : (
                <TimelineList featureId={featureId} events={shown} />
              )
            }
          </Async>
        </>
      ) : (
        <Async query={timeline} skeleton={<TableSkeleton rows={8} />}>
          {(data) => <AgentHistory events={data.events} featureId={featureId} />}
        </Async>
      )}
    </Panel>
  );
}
