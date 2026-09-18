import type { FeatureSummary } from '@/schemas/feature';
/**
 * Dashboard grouping comes directly from `FeatureSummary.dashboard_group`.
 *
 * Failed and cancelled are separate PRD categories, while both use the same visual tone. The
 * server therefore publishes the category explicitly instead of making React reconstruct a
 * lifecycle distinction from status strings or presentation colors.
 *
 * `queued` is the newest of them, and it is the one somebody watching a fresh submission cares
 * about: the platform accepts a feature and answers before anything runs, so "accepted but not
 * picked up" is a real state rather than a moment nobody sees.
 */
export type FeatureGroup = FeatureSummary['dashboard_group'];

export const GROUP_ORDER: readonly FeatureGroup[] = [
  'queued',
  'running',
  'waiting',
  'failed',
  'completed',
  'cancelled',
];

export const GROUP_LABELS: Record<FeatureGroup, string> = {
  queued: 'Queued',
  running: 'Running',
  waiting: 'Needs attention',
  failed: 'Failed',
  completed: 'Completed',
  cancelled: 'Cancelled',
};

/**
 * The tone each category is counted in, so a summary chip and the status badges inside it
 * agree. Cancelled is neutral rather than red: it is a decision somebody made, not a failure.
 */
export const GROUP_TONES: Record<FeatureGroup, 'attention' | 'working' | 'stopped' | 'done' | undefined> = {
  queued: 'working',
  running: 'working',
  waiting: 'attention',
  failed: 'stopped',
  completed: 'done',
  cancelled: undefined,
};

/** Whether a string names one of the server's categories. Used to read one out of a URL. */
export function isFeatureGroup(value: string | null): value is FeatureGroup {
  return value !== null && (GROUP_ORDER as readonly string[]).includes(value);
}

export interface DashboardFilters {
  search: string;
  status: string | 'all';
  group: FeatureGroup | 'all';
}

export const EMPTY_FILTERS: DashboardFilters = { search: '', status: 'all', group: 'all' };

/**
 * Filtering happens over the rows already loaded. The list endpoint is cursor-paged and has no
 * search, so this cannot claim to search everything; the UI says how many rows it searched.
 *
 * The search matches the feature reference, the title, and the internal ids. The reference is
 * what somebody actually has to hand — it is in the pull request title and in whatever message
 * sent them here — so typing `AB-Feature-42` has to find that feature.
 */
export function filterFeatures(
  features: readonly FeatureSummary[],
  filters: DashboardFilters,
): FeatureSummary[] {
  const needle = filters.search.trim().toLowerCase();
  return features.filter((feature) => {
    if (filters.status !== 'all' && feature.status !== filters.status) return false;
    if (filters.group !== 'all' && feature.dashboard_group !== filters.group) return false;
    if (!needle) return true;
    return (
      (feature.reference ?? '').toLowerCase().includes(needle) ||
      feature.title.toLowerCase().includes(needle) ||
      feature.feature_id.toLowerCase().includes(needle) ||
      feature.workflow_id.toLowerCase().includes(needle)
    );
  });
}

export function countByGroup(
  features: readonly FeatureSummary[],
): Record<FeatureGroup, number> {
  const counts: Record<FeatureGroup, number> = {
    queued: 0,
    running: 0,
    waiting: 0,
    failed: 0,
    completed: 0,
    cancelled: 0,
  };
  for (const feature of features) counts[feature.dashboard_group] += 1;
  return counts;
}
