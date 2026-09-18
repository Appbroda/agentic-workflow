import { useMemo } from 'react';
import { keepPreviousData, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import type { WorkstreamOperations } from '@/schemas/feature';
import { statusWording, useStatusVocabulary } from '@/hooks/useStatusVocabulary';
import { effectiveFeatureStatus } from './feature-status';
import { latestIntegrationReview } from './integration-review';

/**
 * Data for one feature.
 *
 * There is no event stream on the server, so these poll. The interval is decided by the
 * feature's own status rather than by a timer the client invents: a finished feature is not
 * polled at all, and a running one is refreshed often enough to feel live without hammering an
 * API whose reads hydrate parent state.
 */

const ACTIVE_INTERVAL_MS = 4_000;
const IDLE_INTERVAL_MS = 20_000;

/** Statuses in which nothing will change without a person, so frequent polling buys nothing. */
const SETTLED = new Set([
  'completed',
  'failed',
  'cancelled',
  'cancelled_with_external_side_effects',
]);
const AWAITING_HUMAN = new Set(['waiting_for_human', 'failed_requires_human']);

export function pollInterval(status: string | undefined): number | false {
  if (status === undefined) return ACTIVE_INTERVAL_MS;
  if (SETTLED.has(status)) return false;
  if (AWAITING_HUMAN.has(status)) return IDLE_INTERVAL_MS;
  return ACTIVE_INTERVAL_MS;
}

export function useFeature(featureId: string) {
  const api = useApi();
  return useQuery({
    queryKey: ['feature', featureId],
    queryFn: ({ signal }) => api.getFeature(featureId, signal),
    refetchInterval: (query) =>
      pollInterval(query.state.data ? effectiveFeatureStatus(query.state.data) : undefined),
  });
}

/**
 * `at` is the id of the newest lifecycle event seen. It is part of the key on purpose: a new
 * event means a fresh read, and nothing here polls on a timer.
 */
export function useWorkstreams(featureId: string, at: number | null) {
  const api = useApi();
  return useQuery({
    queryKey: ['workstreams', featureId, at],
    queryFn: ({ signal }) => api.getWorkstreams(featureId, signal),
    placeholderData: (previous) => previous,
  });
}

/**
 * Each repository's external-operation journal, keyed by repository id.
 *
 * Polled on the feature's own cadence rather than keyed by the event cursor, deliberately:
 * a heartbeat renewal writes no lifecycle event, so a cursor-keyed read would never see the
 * one thing this exists to show — whether the running operation is still alive.
 *
 * The whole response rather than its rows, because the same response now also carries where
 * each finished attempt ended. Both surfaces that need it — the drill-in and the graph's
 * sub-stage strip — read this one poll, so opening either still costs no request of its own.
 */
export function useWorkstreamOperations(
  featureId: string,
  repositoryIds: string[],
  status: string | undefined,
): ReadonlyMap<string, WorkstreamOperations> {
  const api = useApi();
  return useQueries({
    queries: repositoryIds.map((repositoryId) => ({
      queryKey: ['operations', featureId, repositoryId],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api.listWorkstreamOperations(featureId, repositoryId, signal),
      refetchInterval: pollInterval(status),
      placeholderData: keepPreviousData,
    })),
    combine: (results) => {
      const byRepository = new Map<string, WorkstreamOperations>();
      results.forEach((result, index) => {
        const repositoryId = repositoryIds[index];
        if (repositoryId && result.data) byRepository.set(repositoryId, result.data);
      });
      return byRepository;
    },
  });
}

/**
 * The feature's run as a thread of attributed bubbles, composed server-side.
 *
 * Keyed by the event cursor and never polled on a timer, like the timeline: the endpoint
 * hydrates parent state to read the artifacts each agent wrote, so it is read when the view
 * opens and again after something actually happened.
 */
export function useLogbook(featureId: string, at: number | null) {
  const api = useApi();
  return useQuery({
    queryKey: ['logbook', featureId, at],
    queryFn: ({ signal }) => api.getLogbook(featureId, null, signal),
    placeholderData: (previous) => previous,
  });
}

export function useTimeline(featureId: string, at: number | null) {
  const api = useApi();
  // This endpoint merges lifecycle events with artifacts and therefore loads parent state, so
  // it is read when the view opens and after a real event, never on a timer.
  return useQuery({
    queryKey: ['timeline', featureId, at],
    queryFn: ({ signal }) => api.getTimeline(featureId, signal),
    placeholderData: (previous) => previous,
  });
}

/**
 * Who performed each transition, keyed by the event cursor like every other live read.
 *
 * There is no separate poll for model information: a new lifecycle event advances the cursor
 * and this refetches with everything else, which is how an edge moves from queued to running
 * to completed without a second update mechanism.
 */
export function useExecutions(featureId: string, at: number | null) {
  const api = useApi();
  return useQuery({
    queryKey: ['executions', featureId, at],
    queryFn: ({ signal }) => api.getExecutions(featureId, signal),
    placeholderData: (previous) => previous,
  });
}

export function usePullRequests(featureId: string, at: number | null) {
  const api = useApi();
  return useQuery({
    queryKey: ['pull-requests', featureId, at],
    queryFn: ({ signal }) => api.getPullRequests(featureId, signal),
    placeholderData: (previous) => previous,
  });
}

export function useClarification(featureId: string, at: number | null) {
  const api = useApi();
  return useQuery({
    queryKey: ['clarification', featureId, at],
    queryFn: ({ signal }) => api.getClarification(featureId, signal),
    placeholderData: (previous) => previous,
  });
}

export function useFeatureActions(featureId: string, at: number | null) {
  const api = useApi();
  return useQuery({
    queryKey: ['actions', featureId, at],
    queryFn: ({ signal }) => api.listActions(featureId, signal),
    placeholderData: (previous) => previous,
    refetchInterval: (query) =>
      query.state.data?.actions.some((action) => action.in_progress) ? 2_000 : false,
  });
}

/** Envelopes only; a payload is fetched when something actually opens it. */
/**
 * The feature's artifacts, without their payloads.
 *
 * `at` is the newest event id. Passing it makes a new event produce a fresh read, which is
 * what keeps the workflow map current while a feature runs; omitting it is right for the
 * artifacts tab, where a person is reading one document and does not want the list moving
 * underneath them.
 */
export function useArtifactList(
  featureId: string,
  artifactType?: string,
  at?: number | null,
  includePayload = false,
) {
  const api = useApi();
  return useQuery({
    queryKey: ['artifacts', featureId, artifactType ?? 'all', at ?? 'static', includePayload],
    queryFn: ({ signal }) =>
      api.listArtifacts(featureId, { artifactType, includePayload }, signal),
  });
}

/**
 * The verdict of the newest integration review.
 *
 * Its own read, and deliberately a narrow one: the artifact list every progress surface
 * already holds is fetched without payloads, and `review_status` lives in the payload. Asking
 * for the integration reviews alone bounds this to at most one artifact per review cycle
 * rather than pulling every payload the feature has produced -- the list with payloads is the
 * most expensive read this API serves.
 *
 * Returns null while the read is in flight, which is the same "not known" the builders treat
 * as "draw what you drew before": the map appears immediately and gains the verdict when it
 * arrives, instead of the whole panel waiting on a second request.
 */
export function useIntegrationReview(featureId: string, at?: number | null) {
  const reviews = useArtifactList(featureId, 'integration_review', at, true);
  const artifacts = reviews.data?.artifacts;
  // Memoised on the query's own data, which react-query keeps referentially stable between
  // renders: a fresh object each render would be a changing dependency and would rebuild the
  // memoised workflow graph on every poll of every other query.
  return useMemo(() => latestIntegrationReview(artifacts ?? []), [artifacts]);
}

export function useArtifact(featureId: string, artifactId: string | null) {
  const api = useApi();
  return useQuery({
    queryKey: ['artifact', featureId, artifactId],
    queryFn: ({ signal }) => api.getArtifact(featureId, artifactId!, signal),
    enabled: artifactId !== null,
  });
}

/**
 * The server's wording for a status, as a function stable across renders.
 *
 * Memoised on the vocabulary rather than rebuilt each time, so it can be a real dependency of
 * anything that derives from it -- a fresh closure every render would make a memoised graph
 * rebuild on every render for a value that had not changed.
 */
export function useWording(scope: 'feature' | 'workstream') {
  const vocabulary = useStatusVocabulary();
  return useMemo(
    () => (status: string) => statusWording(vocabulary.data, scope, status),
    [vocabulary.data, scope],
  );
}

/**
 * Refresh everything about one feature after an action changed it.
 *
 * Invalidating the feature alone is not enough, and quietly was not: the workstreams, timeline,
 * clarification and artifact reads are all keyed by the newest event id, so they only refresh
 * when the event cursor advances. A feature waiting for a human is polled slowly, and one that
 * has settled is not polled at all -- so answering a clarification left "This feature is
 * waiting on you" on the screen after the platform had accepted the answers and run the feature
 * to completion. It never recovered without a reload.
 *
 * So the event read is refetched too, which advances the cursor, which is what every dependent
 * view is actually watching.
 */
export function useRefreshFeature(featureId: string) {
  const queryClient = useQueryClient();
  return () => {
    void queryClient.invalidateQueries({ queryKey: ['feature', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['workstreams', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['timeline', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['logbook', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['clarification', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['artifacts', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['executions', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['pull-requests', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['repairs', featureId] });
    void queryClient.invalidateQueries({ queryKey: ['actions', featureId] });
    void queryClient.refetchQueries({ queryKey: ['feature-events', featureId] });
  };
}
