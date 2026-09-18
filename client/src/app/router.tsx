import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppLayout } from './AppLayout';
import { DashboardPage } from '@/pages/DashboardPage';
import { NewFeaturePage } from '@/pages/NewFeaturePage';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import { RepositoryPage } from '@/pages/RepositoryPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { UsersPage } from '@/pages/UsersPage';
import { ChangePasswordPage } from '@/pages/ChangePasswordPage';
import { NotFoundPage } from '@/pages/NotFoundPage';

/**
 * Feature workspace sections are routes rather than local state, so a view is linkable,
 * survives a refresh, and the back button does what a person expects.
 *
 * `/needs-attention`, `/running` and `/completed` used to be their own pages. Each was the
 * Features page with one of the server's categories pre-selected, which made four routes for
 * one screen and put a filter in the sidebar as though it were a place. They now redirect to
 * the Features page with the same filter in the query string, so links people already sent
 * each other keep landing on the same set of rows.
 */
const ROUTES = [
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'needs-attention', element: <Navigate to="/?group=waiting" replace /> },
      { path: 'running', element: <Navigate to="/?group=running" replace /> },
      { path: 'completed', element: <Navigate to="/?group=completed" replace /> },
      // Never a page in this client, but the concept existed in the product and somebody may
      // have written the URL down. It means "everything", which is the Features page.
      { path: 'activity', element: <Navigate to="/" replace /> },
      { path: 'features/new', element: <NewFeaturePage /> },
      { path: 'features/:featureId', element: <FeatureWorkspacePage /> },
      { path: 'features/:featureId/repositories/:repositoryId', element: <RepositoryPage /> },
      { path: 'features/:featureId/:tab', element: <FeatureWorkspacePage /> },
      { path: 'settings', element: <SettingsPage /> },
      // Accounts. Its link appears only for somebody holding `user:manage`, and the page
      // itself says so politely for anybody who types the URL -- the endpoints behind it
      // refuse regardless, which is where the control actually is.
      { path: 'users', element: <UsersPage /> },
      // A route rather than a modal, because a forced first-time change is gated on it and
      // a modal is dismissed by reloading the page.
      { path: 'account/password', element: <ChangePasswordPage /> },
      { path: 'index.html', element: <Navigate to="/" replace /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
];

/**
 * The application is served under `/ui`, so every link and every URL it writes carries that
 * prefix. It is not at the root because the API owns `/features/{id}` on the same origin and
 * that is also this application's page for a feature; the API's paths are the contract, so
 * this moved. `import.meta.env.BASE_URL` is Vite's `base`, which keeps the router, the asset
 * URLs and the server's mount from drifting apart.
 */
export const router = createBrowserRouter(ROUTES, {
  basename: import.meta.env.BASE_URL.replace(/\/$/, ''),
  future: { v7_relativeSplatPath: true },
});
