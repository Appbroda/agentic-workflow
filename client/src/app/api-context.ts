import { createContext, useContext } from 'react';
import type { FeatureApi } from '@/api/features';

/**
 * Kept out of the provider module so that file exports only components, which is what lets
 * fast refresh work and is the rule `react-refresh/only-export-components` enforces.
 */
export const ApiContext = createContext<FeatureApi | null>(null);

export function useApi(): FeatureApi {
  const api = useContext(ApiContext);
  if (!api) throw new Error('useApi must be used inside <AppProviders>');
  return api;
}
