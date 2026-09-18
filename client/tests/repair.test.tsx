import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiContext } from '@/app/api-context';
import { RepositoryRepairs } from '@/features/feature-workspace/RepositoryRepair';
import type { RepositoryRepair } from '@/schemas/feature';
import { workstreamsSchema } from '@/schemas/feature';
import { stubApi } from './fixtures';
import blocked from './fixtures/workstreams.admin-server-status-live-038.json';

/**
 * Deciding whether to change somebody's repository.
 *
 * The platform stops a repository whose own setup will not let it run its checks, and refuses
 * to fix that by itself: it proposes, and waits. These tests are about the shape of that
 * decision on the page -- what is shown before somebody commits to it, and what is shown after.
 */

const data = workstreamsSchema.parse(blocked);
const WORKSTREAM = data.workstreams.find((item) => item.blocking_setup_issues.length > 0)!;

function repair(overrides: Partial<RepositoryRepair> = {}): RepositoryRepair {
  return {
    repair_id: 'repair-1',
    feature_id: data.feature_id,
    repository_id: WORKSTREAM.repository_id,
    originating_stage: 'repository_preflight',
    failure_classification: 'validation_configuration_failure',
    detected_problem: 'eslint-config-house is required by .eslintrc.json and is not declared',
    evidence: ["npx eslint . exited 2: Cannot find module 'eslint-config-house'"],
    proposed_repair: 'Declare eslint-config-house as a development dependency',
    affected_files: ['package.json'],
    affected_dependencies: ['eslint-config-house'],
    commands: [],
    expected_impact: 'The repository can run its own checks again.',
    risk: 'medium',
    changes_source_logic: false,
    proposed_at_revision: 'revision-one',
    status: 'proposed',
    approved_by: null,
    approved_at: null,
    rejected_by: null,
    rejection_reason: null,
    execution_result: null,
    resulting_revision: null,
    stale: false,
    created_at: null,
    ...overrides,
  };
}

function renderPanel(api: Record<string, unknown>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ApiContext.Provider value={stubApi(api)}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <RepositoryRepairs featureId={data.feature_id} workstreams={[WORKSTREAM]} />
        </MemoryRouter>
      </ApiContext.Provider>
    </QueryClientProvider>,
  );
}

describe('a repository repair', () => {
  it('shows what would change before anybody agrees to it', async () => {
    renderPanel({
      listRepairs: async () => ({ feature_id: data.feature_id, repairs: [repair()] }),
    });

    // The diagnosis, the fix, and what it touches -- because "approve" is meaningless
    // without knowing what is being approved.
    expect(await screen.findByText(/eslint-config-house is required/)).toBeInTheDocument();
    expect(screen.getByText(/Declare eslint-config-house/)).toBeInTheDocument();
    expect(screen.getByText(/Affected files: package.json/)).toBeInTheDocument();
    expect(screen.getByText(/Affected dependencies: eslint-config-house/)).toBeInTheDocument();
    expect(screen.getByText('risk: medium')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Ask AI' })).toHaveAttribute(
      'href',
      expect.stringContaining('/chat?prompt='),
    );
  });

  it('asks before approving a repair that changes the repository', async () => {
    const approveRepair = vi.fn().mockResolvedValue({});
    renderPanel({
      listRepairs: async () => ({ feature_id: data.feature_id, repairs: [repair()] }),
      approveRepair,
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Approve repair' }));

    // Nothing has been sent yet: the first press opens the confirmation, it does not act.
    expect(approveRepair).not.toHaveBeenCalled();
    expect(screen.getByText(/changes the repository’s checked-in files/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Yes, approve and retry' }));
    expect(approveRepair).toHaveBeenCalledWith(data.feature_id, 'repair-1', {
      credentials: {},
    });
  });

  it('will not reject a repair without a reason', async () => {
    const rejectRepair = vi.fn().mockResolvedValue({});
    renderPanel({
      listRepairs: async () => ({ feature_id: data.feature_id, repairs: [repair()] }),
      rejectRepair,
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Reject repair' }));
    const submit = screen.getByRole('button', { name: 'Reject repair' });

    // A stop left in place with no recorded reason explains nothing to the next person, and
    // the server requires one too.
    expect(submit).toBeDisabled();
    await user.type(
      screen.getByLabelText(/Why is this repair not the right change/),
      'We are removing that lint rule instead.',
    );
    expect(submit).toBeEnabled();
    await user.click(submit);
    expect(rejectRepair).toHaveBeenCalledWith(
      data.feature_id,
      'repair-1',
      'We are removing that lint rule instead.',
    );
  });

  it('says when the repository has moved since the diagnosis', async () => {
    renderPanel({
      listRepairs: async () => ({
        feature_id: data.feature_id,
        repairs: [repair({ stale: true })],
      }),
    });

    expect(await screen.findByText('out of date')).toBeInTheDocument();
    expect(screen.getByText(/changed since the diagnosis was written/)).toBeInTheDocument();
  });

  it('reports what happened after a repair was applied', async () => {
    renderPanel({
      listRepairs: async () => ({
        feature_id: data.feature_id,
        repairs: [
          repair({
            status: 'succeeded',
            approved_by: 'alex',
            execution_result: 'The repository ran its own checks after the repair was applied.',
            resulting_revision: 'revision-two-abcdef',
          }),
        ],
      }),
    });

    expect(await screen.findByText('Repair applied')).toBeInTheDocument();
    expect(screen.getByText('Approved by alex.')).toBeInTheDocument();
    expect(screen.getByText(/now at revision-two/)).toBeInTheDocument();
    // A decided repair offers no buttons: the decision has been made.
    expect(screen.queryByRole('button', { name: 'Approve repair' })).not.toBeInTheDocument();
  });

  it('records who declined a repair and why', async () => {
    renderPanel({
      listRepairs: async () => ({
        feature_id: data.feature_id,
        repairs: [
          repair({
            status: 'rejected',
            rejected_by: 'alex',
            rejection_reason: 'We are removing that lint rule instead.',
          }),
        ],
      }),
    });

    expect(
      await screen.findByText(/Rejected by alex: We are removing that lint rule instead./),
    ).toBeInTheDocument();
  });
});
