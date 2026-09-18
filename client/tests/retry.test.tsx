import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import { ApiError } from '@/api/errors';
import { stubApi } from './fixtures';

const FEATURE = {
  feature_id: 'f-1',
  workflow_id: 'w-1',
  status: 'failed_requires_human',
  title: 'A feature that stopped',
  current_agent: 'child_workflows',
  repository_count: 2,
  required_repository_count: 2,
  repositories: [],
  clarification_rounds: 0,
  integration_review_cycles: 0,
  merge_strategy: null,
  deployment_strategy: null,
  execution_mode: 'live',
  cancellation_status: 'not_requested',
  cancellation_requested_at: null,
  cancellation_reason: null,
  cleanup_requirements: [],
  created_at: '2026-08-24T22:40:00Z',
  updated_at: '2026-08-24T23:20:00Z',
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
    retry_count: 3,
    code_completion_artifact_id: null,
    review_artifact_id: null,
    blocking_issues: ['npm ci failed'],
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
    repository_setup_retry_count: 2,
    integration_retry_count: 0,
    granted_extra_attempts: 0,
    retry_grants: [],
    implementation_expectations: [],
    configured_validation_commands: [],
    blocking_setup_issues: [],
  retry_refusal_reason: 'The repository_setup_retry_count budget of 2 is exhausted.',
    available_actions: ['RETRY_WORKSTREAM'],
    ...overrides,
  };
}

const BASE: Partial<FeatureApi> = {
  getFeature: async () => FEATURE,
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

function renderWorkspace(api: Partial<FeatureApi>) {
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

/** Expand the repository's row, then open its grant dialog. */
async function openGrant(user: ReturnType<typeof userEvent.setup>) {
  await user.click((await screen.findAllByRole('button', { name: 'Show detail' }))[0]!);
  await user.click(screen.getByRole('button', { name: 'Grant another attempt' }));
}

describe('granting a stopped repository another attempt', () => {
  it('is offered only for a repository the server would accept', async () => {
    renderWorkspace({
      getWorkstreams: async () => ({
        feature_id: 'f-1',
        workstreams: [
          workstream(),
          workstream({
            repository_id: 'frontend',
            repository_name: 'Frontend',
            status: 'approved',
            available_actions: [],
            blocking_issues: [],
            retry_refusal_reason: null,
          }),
        ],
      }),
    });

    const user = userEvent.setup();
    const table = await screen.findByRole('table', { name: 'Repository workstreams' });
    // The control lives with the repository it acts on, one expansion from the row.
    for (const button of within(table).getAllByRole('button', { name: 'Show detail' })) {
      await user.click(button);
    }

    // The stopped repository can be granted an attempt.
    expect(screen.getByRole('button', { name: 'Grant another attempt' })).toBeVisible();
    // The approved one cannot: retrying it would discard work that passed review, and the
    // server refuses. A button whose only outcome is a 409 is worse than no button -- so
    // there is exactly one of these on a page showing one stopped repository and one
    // approved repository.
    expect(screen.getAllByRole('button', { name: 'Grant another attempt' })).toHaveLength(1);
  });

  it('requires an author and a reason before it will send', async () => {
    const retryWorkstream = vi.fn();
    renderWorkspace({
      getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [workstream()] }),
      retryWorkstream,
    });
    const user = userEvent.setup();

    await openGrant(user);
    const submit = screen.getByRole('button', { name: 'Grant and run' });
    // An override with no author and no stated reason is indistinguishable, later, from the
    // platform having changed its own mind.
    expect(submit).toBeDisabled();

    await user.type(screen.getByLabelText('Your name'), 'akhilesh');
    expect(submit).toBeDisabled();
    await user.type(
      screen.getByLabelText('What changed since the last attempt?'),
      'Installed the missing registry token.',
    );
    expect(submit).toBeEnabled();

    await user.click(submit);
    // No credentials travel with the grant: the attempt is queued and the worker resolves
    // the feature owner's stored keys, so there is nothing for this dialog to collect.
    expect(retryWorkstream).toHaveBeenCalledWith('f-1', 'backend', {
      additional_attempts: 1,
      requested_by: 'akhilesh',
      reason: 'Installed the missing registry token.',
    });
  });

  it('reports the platform refusing the grant', async () => {
    renderWorkspace({
      getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [workstream()] }),
      retryWorkstream: async () => {
        throw new ApiError('conflict', 'no', 409, 'answer the open clarification questions first');
      },
    });
    const user = userEvent.setup();

    await openGrant(user);
    await user.type(screen.getByLabelText('Your name'), 'akhilesh');
    await user.type(screen.getByLabelText('What changed since the last attempt?'), 'Fixed it.');
    await user.click(screen.getByRole('button', { name: 'Grant and run' }));

    // The server is the authority on whether the grant may run, and its reason is shown.
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'clarification questions',
    );
  });

  it('shows a past grant as somebody having overridden the platform', async () => {
    renderWorkspace({
      getWorkstreams: async () => ({
        feature_id: 'f-1',
        workstreams: [
          workstream({
            granted_extra_attempts: 1,
            retry_grants: [
              {
                granted_by: 'akhilesh',
                attempts: 1,
                reason: 'Installed the missing registry token.',
                granted_at: '2026-08-25T09:00:00Z',
              },
            ],
          }),
        ],
      }),
    });
    const user = userEvent.setup();

    await user.click((await screen.findAllByRole('button', { name: 'Show detail' }))[0]!);

    // Without this the only trace is an attempt count past the configured limit, which reads
    // as a platform defect rather than as a decision somebody made.
    expect(screen.getByText(/akhilesh granted/)).toBeInTheDocument();
    expect(screen.getByText(/registry token/)).toBeInTheDocument();
  });
});
