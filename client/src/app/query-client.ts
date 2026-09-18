import { QueryClient } from '@tanstack/react-query';
import { ApiError } from '@/api/errors';

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5_000,
        refetchOnWindowFocus: false,
        // Only retry what a retry could fix. Retrying a 409 hides the platform's refusal
        // behind three identical rejections.
        retry: (failureCount, error) =>
          error instanceof ApiError && error.isRetryable && failureCount < 2,
      },
      mutations: { retry: false },
    },
  });
}
