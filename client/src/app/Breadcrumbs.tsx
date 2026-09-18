import { Fragment } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import type { Feature } from '@/schemas/feature';
import { useApi } from './api-context';
import { humanise } from '@/utils/text';

/**
 * Where you are, and how to get back one step.
 *
 * Derived from the URL rather than declared by each page, so a route added later cannot forget
 * to say where it is. The feature's title comes from the query cache when the workspace has
 * already loaded it -- the trail says "Server uptime history" rather than a bare identifier --
 * and falls back to the identifier when it has not, which is what a fresh deep link sees for a
 * moment.
 */

const SECTION_LABELS: Record<string, string> = {
  'needs-attention': 'Needs attention',
  running: 'Running',
  completed: 'Completed',
  settings: 'Settings',
  new: 'New feature',
  'pull-requests': 'Pull requests',
  prd: 'PRD',
};

export function Breadcrumbs() {
  const api = useApi();
  const { pathname } = useLocation();
  const segments = pathname.split('/').filter(Boolean);
  const featureId =
    segments[0] === 'features' && segments[1] && segments[1] !== 'new'
      ? decodeURIComponent(segments[1])
      : null;
  // Read from the cache the workspace already filled, and subscribe to it: `getQueryData` is
  // not reactive, so the trail would keep showing the identifier after the title arrived.
  // Disabled on purpose -- the page below owns fetching this, and a breadcrumb must not be
  // the reason a request happens. It is the same key and the same function, so when the page
  // does fetch, this sees it.
  const feature = useQuery<Feature>({
    queryKey: ['feature', featureId ?? ''],
    queryFn: ({ signal }) => api.getFeature(featureId ?? '', signal),
    enabled: false,
  }).data;

  const crumbs: { label: string; to?: string }[] = [];

  if (segments.length === 0) {
    crumbs.push({ label: 'Features' });
  } else if (featureId) {
    crumbs.push({ label: 'Features', to: '/' });
    crumbs.push({ label: feature?.title ?? featureId, to: `/features/${encodeURIComponent(featureId)}` });

    if (segments[2] === 'repositories' && segments[3]) {
      const repositoryId = decodeURIComponent(segments[3]);
      crumbs.push({
        label: 'Repositories',
        to: `/features/${encodeURIComponent(featureId)}/repositories`,
      });
      crumbs.push({ label: repositoryId });
    } else if (segments[2]) {
      crumbs.push({ label: label(segments[2]) });
    }
  } else if (segments[0] === 'features') {
    crumbs.push({ label: 'Features', to: '/' });
    crumbs.push({ label: 'New feature' });
  } else {
    crumbs.push({ label: label(segments[0] ?? '') });
  }

  return (
    <nav className="crumbs" aria-label="Breadcrumb">
      {crumbs.map((crumb, index) => {
        const last = index === crumbs.length - 1;
        return (
          <Fragment key={`${crumb.label}-${index}`}>
            {index > 0 ? (
              <span className="crumbs__sep" aria-hidden="true">
                /
              </span>
            ) : null}
            {crumb.to && !last ? (
              <Link className="crumbs__item" to={crumb.to} title={crumb.label}>
                {crumb.label}
              </Link>
            ) : (
              <span
                className={last ? 'crumbs__item crumbs__item--current' : 'crumbs__item'}
                aria-current={last ? 'page' : undefined}
                title={crumb.label}
              >
                {crumb.label}
              </span>
            )}
          </Fragment>
        );
      })}
    </nav>
  );
}

function label(segment: string): string {
  return SECTION_LABELS[segment] ?? humanise(segment);
}
