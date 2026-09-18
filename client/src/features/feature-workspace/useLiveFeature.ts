import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { FEATURE_EVENT_PAGE_SIZE } from '@/api/features';
import type { FeatureEvent } from '@/schemas/feature';
import { pollInterval } from './hooks';

/**
 * Keeps an open feature current.
 *
 * The server streams lifecycle events over an authenticated request, so this watches that
 * stream and falls back to polling the same events when it cannot. Both read the same indexed
 * event table, so nothing is true of one and not the other -- the stream only says *when* to
 * look, and the REST endpoints stay what the application believes.
 *
 * `EventSource` is not used, and the reason is not incidental: it cannot send an
 * `Authorization` header, and the usual way round that puts the platform token in the query
 * string, where proxies log it and browsers keep it in history.
 *
 * The event id is returned rather than used to invalidate caches. Invalidation raced: when an
 * event arrived while a dependent query was still fetching, React Query coalesced the two and
 * the refresh was silently dropped. Making the id part of those queries' keys states the
 * dependency instead -- "workstreams as of event 7" -- so a new event always produces a fresh
 * read, and a duplicated or missed event cannot desynchronise anything.
 */
export function useLiveFeature(featureId: string, status: string | undefined) {
  const api = useApi();
  const queryClient = useQueryClient();
  const cursor = useRef<number | null>(null);
  const activeFeature = useRef(featureId);
  // Each read returns only what followed the cursor, so anything that wants the run as a whole
  // -- the notices, for one -- has to accumulate. Bounded, because a long feature's event log
  // is unbounded and this is held to decide what to tell somebody, not to be the record.
  const seen = useRef<FeatureEvent[]>([]);
  const [live, setLive] = useState(false);
  // Bumped whenever the stream delivers something, purely to make React re-render. The event
  // data itself lives in the query cache below, which stays the single source.
  const [streamTick, setStreamTick] = useState(0);

  // React Router reuses this component when navigating directly from one feature to another.
  // Refs survive that reuse; carrying feature A's cursor and notices into feature B would skip
  // B's early events and briefly report A's failures on the wrong workspace.
  if (activeFeature.current !== featureId) {
    activeFeature.current = featureId;
    cursor.current = null;
    seen.current = [];
  }

  const events = useQuery({
    queryKey: ['feature-events', featureId],
    queryFn: async ({ signal }) => {
      const events: FeatureEvent[] = [];
      let after = cursor.current;
      let feature = featureId;
      // A settled feature is deliberately not polled again. Drain every full page now so a
      // reconnect after a long run reaches its terminal events instead of stopping at the
      // first 200 records forever.
      for (;;) {
        const page = await api.listEvents(featureId, after, signal);
        feature = page.feature_id;
        events.push(...page.events);
        const next = page.last_event_id ?? after;
        const advanced = next !== after;
        after = next;
        if (page.events.length < FEATURE_EVENT_PAGE_SIZE || !advanced) break;
      }
      cursor.current = after;
      retain(seen, events);
      return { feature_id: feature, events, last_event_id: after };
    },
    // Polling is the fallback, not the mechanism. While the stream is connected this stops,
    // and it resumes the moment the stream drops -- so a proxy that kills long connections
    // degrades to what the client did before rather than to nothing.
    refetchInterval: live ? false : pollInterval(status),
  });

  // Whether this feature can still produce an event, rather than the status itself. The
  // effect below depends on this so it tears the stream down and reopens it only when that
  // answer changes -- depending on the raw status reconnected on every status transition,
  // including the first one from `undefined` while the feature query was still loading.
  const watchable = pollInterval(status) !== false;

  useEffect(() => {
    // A settled feature will not produce another event. Holding a connection open for one
    // would be a connection per viewer for nothing.
    if (!watchable) {
      setLive(false);
      return undefined;
    }

    const controller = new AbortController();
    let cancelled = false;

    const run = async () => {
      // Reconnect with a growing delay, so a server that is down is not hammered by every
      // open tab. Reset on a successful connection, because a stream that ran for an hour
      // and dropped is not the same situation as one that never opened.
      let backoff = 1_000;
      while (!cancelled) {
        try {
          for await (const frame of api.streamFeatureEvents(featureId, cursor.current, {
            signal: controller.signal,
          })) {
            if (cancelled) return;
            if (frame.event === 'open') {
              backoff = 1_000;
              setLive(true);
              continue;
            }
            if (frame.event !== 'event') continue;
            const event = frame.data as FeatureEvent;
            // Ahead of the cursor only. A duplicate delivery -- a reconnect that overlaps,
            // a proxy that replays -- is dropped here rather than applied twice.
            if (typeof event.id !== 'number' || event.id <= (cursor.current ?? 0)) continue;
            cursor.current = event.id;
            retain(seen, [event]);
            setStreamTick((tick) => tick + 1);
            // The stream says when; the REST reads say what. Refetching the event query
            // keeps one cache and one cursor rather than two that can disagree.
            void queryClient.invalidateQueries({ queryKey: ['feature', featureId] });
          }
        } catch {
          // Any failure to read -- dropped connection, a server restart, a proxy timeout --
          // is the same situation: fall back to polling and try again shortly.
        }
        if (cancelled) return;
        setLive(false);
        await new Promise((resolve) => setTimeout(resolve, backoff));
        backoff = Math.min(backoff * 2, 30_000);
      }
    };

    void run();
    return () => {
      cancelled = true;
      controller.abort();
      setLive(false);
    };
  }, [api, featureId, queryClient, watchable]);

  return {
    lastEventId: cursor.current ?? events.data?.last_event_id ?? null,
    // Read through the query's own data or the stream tick so a re-render follows either;
    // the ref alone would not tell React anything changed.
    events: events.data || streamTick > 0 ? seen.current : [],
    /** Whether the event stream is currently connected, rather than being polled. */
    live,
    query: events,
  };
}

/** Keep the most recent events, de-duplicated by id and in order. */
function retain(store: { current: FeatureEvent[] }, incoming: FeatureEvent[]): void {
  const byId = new Map(store.current.map((event) => [event.id, event]));
  for (const event of incoming) byId.set(event.id, event);
  store.current = [...byId.values()]
    .sort((left, right) => left.id - right.id)
    .slice(-MAX_RETAINED_EVENTS);
}

const MAX_RETAINED_EVENTS = 500;
