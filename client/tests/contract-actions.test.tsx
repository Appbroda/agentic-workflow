import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import type { FeatureApi } from '@/api/features';
import { ContractChangeActions } from '@/features/feature-workspace/ContractChangeActions';
import { stubApi } from './fixtures';

const envelope = {
  workflow_id: 'f-1',
  schema_version: '1',
  producer: 'planner',
  timestamp: '2026-08-25T10:00:00Z',
  metadata: {},
  validation_status: 'valid',
};

const request = {
  ...envelope,
  artifact_id: '013_contract_change_request.api.1.json',
  artifact_type: 'contract_change_request',
  payload: {
    change_request_id: 'f-1:api:1',
    status: 'pending',
    reason: 'The implementation needs a new response field.',
    requested_changes: ['Add `active` to StatusResponse.'],
    affected_workstreams: ['api', 'admin'],
    compatibility_impact: 'Existing consumers remain compatible.',
  },
};

const contract = {
  ...envelope,
  artifact_id: '009_integration_contract.json',
  artifact_type: 'integration_contract',
  payload: {
    feature_id: 'f-1',
    status: 'approved',
    contract_version: '1.0.0',
    api_style: 'rest',
    endpoints: [],
    shared_schemas: [],
    authentication_contract: null,
    authorization_rules: [],
    error_contracts: [],
    event_contracts: [],
    environment_variables: [],
    compatibility_policy: { backward_compatible_changes: [], breaking_change_policy: 'major' },
    owning_workstreams: ['api', 'admin'],
  },
};

function renderActions(api: Partial<FeatureApi>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ContractChangeActions featureId="f-1" at={7} />
      </MemoryRouter>
    </AppProviders>,
  );
}

function artifactApi(overrides: Partial<FeatureApi>): Partial<FeatureApi> {
  return {
    listArtifacts: async (_featureId, params) => ({
      feature_id: 'f-1',
      artifacts: params?.artifactType === 'contract_change_request' ? [request] : [contract],
    }),
    listEvents: async () => ({ feature_id: 'f-1', events: [], last_event_id: 7 }),
    ...overrides,
  };
}

describe('contract change decisions', () => {
  it('shows the real request and confirms a rejection before sending it', async () => {
    const rejectContractChange = vi.fn().mockResolvedValue({});
    renderActions(artifactApi({ rejectContractChange }));
    const user = userEvent.setup();

    expect(await screen.findByText('A shared contract change needs a decision')).toBeInTheDocument();
    expect(screen.getByText('Add `active` to StatusResponse.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Review rejection' }));
    expect(rejectContractChange).not.toHaveBeenCalled();

    const dialog = screen.getByRole('dialog', { name: 'reject contract change' });
    await user.type(within(dialog).getByLabelText('Decision rationale'), 'Keep version 1 stable.');
    await user.click(within(dialog).getByRole('button', { name: 'Reject contract change' }));

    expect(rejectContractChange).toHaveBeenCalledWith(
      'f-1',
      'f-1:api:1',
      { resolution: 'Keep version 1 stable.' },
    );
  });

  it('prefills the current revision and sends credentials with an approval', async () => {
    const approveContractChange = vi.fn().mockResolvedValue({});
    renderActions(artifactApi({ approveContractChange }));
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Review approval' }));
    const dialog = screen.getByRole('dialog', { name: 'approve contract change' });
    const editor = within(dialog).getByLabelText('Complete replacement contract (JSON)');
    expect((editor as HTMLTextAreaElement).value).toContain('"contract_version": "1.0.0"');
    expect((editor as HTMLTextAreaElement).value).not.toContain('"feature_id"');
    await user.type(within(dialog).getByLabelText('Decision rationale'), 'Compatible additive field.');
    await user.type(within(dialog).getByLabelText('OpenAI API key'), 'sk-contract');
    await user.type(within(dialog).getByLabelText('GitHub token'), 'gh-contract');
    await user.click(
      within(dialog).getByRole('button', { name: 'Approve and rerun affected workstreams' }),
    );

    expect(approveContractChange).toHaveBeenCalledWith(
      'f-1',
      'f-1:api:1',
      expect.objectContaining({
        resolution: 'Compatible additive field.',
        updated_contract: expect.objectContaining({ contract_version: '1.0.0' }),
      }),
      { credentials: { openaiApiKey: 'sk-contract', githubToken: 'gh-contract' } },
    );
  });
});
