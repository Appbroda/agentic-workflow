import { useQuery } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';

export interface StatusWording {
  headline: string;
  detail: string;
  next_step: string;
  tone: string;
}

/**
 * Plain-language wording for a lifecycle status, from the server.
 *
 * The client never authors status copy. `failed_requires_human` is exact and tells a product
 * manager nothing; the server already owns that translation and tests that every status has
 * wording, so a status added on the server arrives here explained rather than raw.
 */
export function useStatusVocabulary() {
  const api = useApi();
  return useQuery({
    queryKey: ['status-vocabulary'],
    queryFn: ({ signal }) => api.getStatusVocabulary(signal),
    staleTime: 60 * 60 * 1000,
  });
}

export function statusWording(
  vocabulary: Record<string, Record<string, StatusWording>> | undefined,
  scope: 'feature' | 'workstream',
  status: string,
): StatusWording {
  const wording = vocabulary?.[scope]?.[status];
  if (wording) return wording;
  // An unknown status is shown as itself rather than hidden. The server may be ahead of this
  // client, and inventing wording here would be the client asserting something it cannot know.
  return { headline: status, detail: '', next_step: '', tone: 'working' };
}
