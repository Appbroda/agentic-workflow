import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import { stubApi } from './fixtures';

/**
 * Publishing a feature that did not land.
 *
 * The half of task 81- that keeps narrowing automatic publication from being a regression. A
 * partial feature no longer opens a pull request nobody asked for, so if the offer that
 * replaces it is quiet, the work is lost exactly as it was before -- just more politely.
 */

const FEATURE = {
  feature_id: 'f-1',
  workflow_id: 'w-1',
  status: 'failed_requires_human',
  title: 'A feature that did not land',
  current_agent: 'feature_workflow',
  repository_count: 3,
  required_repository_count: 3,
  repositories: [],
  clarification_rounds: 0,
  integration_review_cycles: 0,
  merge_strategy: null,
  deployment_strategy: null,
  execution_mode: 'mock',
  cancellation_status: 'not_requested',
  cancellation_requested_at: null,
  cancellation_reason: null,
  cleanup_requirements: [],
  available_actions: ['CANCEL_WORKFLOW', 'RETIRE_FEATURE', 'PUBLISH_FEATURE'],
  created_at: '2026-09-06T09:00:00Z',
  updated_at: '2026-09-06T09:40:00Z',
};

function workstream(overrides: Record<string, unknown> = {}) {
  return {
    repository_id: 'backend',
    repository_name: 'Backend',
    repository_role: 'backend',
    child_workflow_id: 'f-1:backend',
    workstream_id: 'ws-backend',
    status: 'failed',
    branch_name: 'feature/f-1',
    workspace_path: '/workspaces/f-1/backend',
    retry_count: 1,
    code_completion_artifact_id: null,
    review_artifact_id: null,
    blocking_issues: [],
    pull_request_artifact_id: null,
    current_validation_results: [],
    scoped_requirements: [],
    out_of_scope_requirements: [],
    planned_blind: false,
    production_files_changed: [],
    test_files_changed: [],
    configuration_files_changed: [],
    requirements_implemented: [],
    requirements_not_implemented: [],
    implementation_retry_count: 0,
    validation_retry_count: 0,
    repository_setup_retry_count: 0,
    integration_retry_count: 0,
    granted_extra_attempts: 0,
    retry_grants: [],
    implementation_expectations: [],
    configured_validation_commands: [],
    blocking_setup_issues: [],
    available_actions: [],
    publication_class: null,
    publication_refusal: null,
    ...overrides,
  };
}

const WORKSTREAMS = {
  feature_id: 'f-1',
  workstreams: [
    workstream({
      repository_id: 'frontend',
      repository_name: 'Frontend',
      status: 'approved',
      publication_class: 'reviewed',
    }),
    workstream({ publication_class: 'unreviewed' }),
    workstream({
      repository_id: 'worker',
      repository_name: 'Worker',
      publication_class: null,
      publication_refusal:
        "This repository's last attempt did not pass every check it requires (repository lint).",
    }),
  ],
};

const BASE: Partial<FeatureApi> = {
  getFeature: async () => FEATURE,
  getWorkstreams: async () => WORKSTREAMS,
  listEvents: async () => ({ feature_id: 'f-1', events: [], last_event_id: null }),
  getTimeline: async () => ({ feature_id: 'f-1', events: [] }),
  getClarification: async () => ({
    feature_id: 'f-1',
    awaiting_answers: false,
    technical_prd_artifact_id: null,
    clarification_rounds: 0,
    max_clarification_rounds: 10,
    questions: [],
    previous_answers: {},
    design_conflicts: [],
  }),
};

function renderWorkspace(api: Partial<FeatureApi> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi({ ...BASE, ...api })} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={['/features/f-1/repositories']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('publishing a feature that did not land', () => {
  it('offers the decision, counted, and says what each repository would get', async () => {
    renderWorkspace();
    const user = userEvent.setup();

    // Two of the three repositories would be published, and the control says so before it is
    // pressed: a bare "Publish" is exactly the quiet control this change cannot afford.
    const publish = await screen.findByRole('button', { name: 'Publish 2 repositories' });
    await user.click(publish);

    expect(screen.getByText('Passed review — 1')).toBeVisible();
    expect(screen.getByText('Rejected by review — 1')).toBeVisible();
    // And the one that is held back names the check that is holding it, in the server's own
    // sentence rather than a client-side guess.
    expect(screen.getByText('Not published — 1')).toBeVisible();
    expect(screen.getByText(/did not pass every check it requires/)).toBeVisible();
    // The rejected one is declared as an override, not slipped in beside the approved work.
    expect(screen.getByText(/overrules that judgement/)).toBeVisible();
  });

  it('will not send without a stated reason, then sends exactly that', async () => {
    const publishFeature = vi.fn();
    renderWorkspace({ publishFeature });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Publish 2 repositories' }));
    const submit = screen.getByRole('button', { name: 'Publish' });
    expect(submit).toBeDisabled();

    await user.type(
      screen.getByLabelText('Why are you publishing this?'),
      'The reviewer is wrong and the backend is not going to land.',
    );
    expect(submit).toBeEnabled();
    await user.click(submit);

    // No credentials travel with the decision: the publication is queued and the worker
    // resolves the feature owner's stored keys, so there is nothing for this dialog to collect.
    expect(publishFeature).toHaveBeenCalledWith('f-1', {
      requested_by: '',
      reason: 'The reviewer is wrong and the backend is not going to land.',
    });
  });

  it('is not offered when the platform does not advertise it', async () => {
    renderWorkspace({
      getFeature: async () => ({
        ...FEATURE,
        available_actions: ['CANCEL_WORKFLOW', 'RETIRE_FEATURE'],
      }),
    });

    await screen.findByRole('table', { name: 'Repository workstreams' });
    // A control whose only outcome is a 409 is not a control.
    expect(screen.queryByRole('button', { name: /^Publish/ })).toBeNull();
  });
});
