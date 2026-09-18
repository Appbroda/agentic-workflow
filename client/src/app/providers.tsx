import { QueryClientProvider, type QueryClient } from '@tanstack/react-query';
import { useMemo, type ReactNode } from 'react';
import { ApiClient, readConfigFromEnv, type ApiClientConfig } from '@/api/client';
import { createFeatureApi, type FeatureApi } from '@/api/features';
import { ApiContext } from './api-context';
import { createQueryClient } from './query-client';
import { SessionProvider } from './session';

/** `api` and `queryClient` are injectable so tests drive real components without a network. */
export function AppProviders({
  children,
  api,
  queryClient,
  config,
}: {
  children: ReactNode;
  api?: FeatureApi;
  queryClient?: QueryClient;
  config?: ApiClientConfig;
}) {
  const client = useMemo(() => queryClient ?? createQueryClient(), [queryClient]);
  const featureApi = useMemo(
    () => api ?? createFeatureApi(new ApiClient(config ?? readConfigFromEnv())),
    [api, config],
  );
  return (
    <QueryClientProvider client={client}>
      <ApiContext.Provider value={featureApi}>
        {/* Inside the query client and inside the API context, because it uses both: it
            holds the `/me` query, and signing out clears the cache. Outside the router,
            because the router's own layout asks who is signed in. */}
        <SessionProvider>{children}</SessionProvider>
      </ApiContext.Provider>
    </QueryClientProvider>
  );
}
