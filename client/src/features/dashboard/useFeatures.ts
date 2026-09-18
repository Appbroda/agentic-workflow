import { useMemo } from 'react';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import type { FeatureSummary } from '@/schemas/feature';
import { countByGroup } from './grouping';

/**
 * The feature list, read once for the whole application.
 *
 * The sidebar's attention count and the dashboard table are the same data, so they are the
 * same query: one cache entry, one request, and no possibility of the badge saying two while
 * the table shows three. The list endpoint is cursor-paged, so "loaded" is what has been
 * fetched -- which is why the dashboard says so in as many words.
 */

export const FEATURE_PAGE_SIZE = 50;

export function useFeatureList() {
  const api = useApi();
  const query = useInfiniteQuery({
    queryKey: ['features', FEATURE_PAGE_SIZE],
    queryFn: ({ pageParam, signal }) =>
      api.listFeatures({ limit: FEATURE_PAGE_SIZE, cursor: pageParam }, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor ?? null,
  });

  const features: FeatureSummary[] = useMemo(
    () => query.data?.pages.flatMap((page) => page.features) ?? [],
    [query.data],
  );
  const counts = useMemo(() => countByGroup(features), [features]);

  return { query, features, counts };
}
