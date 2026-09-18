import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import type { ExecutionRecord } from '@/schemas/feature';
import { modelDisplayName } from '@/utils/model';
import { edgeLabel, executionsByEdge } from '@/features/feature-workspace/executions';
import { stubApi } from './fixtures';

/**
 * What the arrows say, and what they must never say.
 *
 * An arrow that names a model is a claim about what ran. These tests are mostly about the
 * cases where the honest answer is "not that": a deterministic validator, a transition nobody
 * has routed yet, a retry the reliability logic refused. A confident-looking wrong label is
 * worse than none, because it is indistinguishable from a right one.
 */

const FEATURE = {
  feature_id: 'adunit-deactivate-live-086',
  workflow_id: 'workflow-086',
  status: 'running_child_workflows',
  title: 'Deactivate ad units from the console',
  current_agent: 'engineer',
  repository_count: 1,
  required_repository_count: 1,
  repositories: [],
  clarification_rounds: 0,
  integration_review_cycles: 0,
  max_clarification_rounds: 10,
  max_integration_review_cycles: 5,
  max_child_review_cycles: 8,
  max_implementation_retries: 4,
  max_validation_retries: 2,
  max_repository_setup_retries: 1,
  merge_strategy: null,
  deployment_strategy: null,
  execution_mode: 'live',
  cancellation_status: 'not_requested',
  cancellation_requested_at: null,
  cancellation_reason: null,
  cleanup_requirements: [],
  available_actions: [],
  created_at: '2026-08-24T22:40:00Z',
  updated_at: '2026-08-24T23:20:00Z',
};

function workstream(overrides: Record<string, unknown> = {}) {
  return {
    repository_id: 'admanager-server',
    repository_name: 'Ad Manager Server',
    repository_role: 'backend',
    child_workflow_id: 'f:admanager-server',
    workstream_id: 'admanager-server',
    status: 'running',
    branch_name: 'ai/x',
    workspace_path: '/w',
    retry_count: 1,
    code_completion_artifact_id: '006_code_completion.admanager-server.attempt-1.json',
    review_artifact_id: null,
    blocking_issues: [],
    pull_request_artifact_id: null,
    current_validation_results: [],
    production_files_changed: [],
    test_files_changed: [],
    configuration_files_changed: [],
    requirements_implemented: [],
    requirements_not_implemented: [],
    implementation_retry_count: 1,
    validation_retry_count: 0,
    repository_setup_retry_count: 0,
    integration_retry_count: 0,
    granted_extra_attempts: 0,
    retry_grants: [],
    implementation_expectations: [],
    configured_validation_commands: [],
    blocking_setup_issues: [],
    scoped_requirements: [],
    out_of_scope_requirements: [],
    planned_blind: false,
    ...overrides,
  };
}

function execution(overrides: Partial<ExecutionRecord> & { execution_id: string }): ExecutionRecord {
  return {
    from_stage: 'repository',
    to_stage: 'implementation',
    repository_id: 'admanager-server',
    is_retry: false,
    handler_type: 'model',
    handler: 'Engineer',
    agent_type: 'Engineer',
    model_resolved: true,
    status: 'completed',
    escalated: false,
    command: [],
    execution_mode: 'live',
    ...overrides,
  } as ExecutionRecord;
}

function renderWorkflow(api: Partial<FeatureApi>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={['/features/adunit-deactivate-live-086/workflow']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
          <Route path="/features/:featureId" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

function apiWith(
  executions: ExecutionRecord[],
  overrides: Partial<FeatureApi> = {},
): Partial<FeatureApi> {
  return {
    listEvents: async () => ({ feature_id: FEATURE.feature_id, events: [], last_event_id: null }),
    getFeature: async () => FEATURE,
    getWorkstreams: async () => ({
      feature_id: FEATURE.feature_id,
      workstreams: [workstream()],
    }),
    getTimeline: async () => ({ feature_id: FEATURE.feature_id, events: [] }),
    listArtifacts: async () => ({ feature_id: FEATURE.feature_id, artifacts: [] }),
    getExecutions: async () => ({ feature_id: FEATURE.feature_id, executions }),
    ...overrides,
  } as Partial<FeatureApi>;
}

/** The arrow's own button, found by the sentence it reads out. */
function edge(match: RegExp) {
  return screen.findByRole('button', { name: match });
}

describe('model display', () => {
  it('writes an identifier once, in one place, without inventing a version', () => {
    expect(modelDisplayName('gpt-5.6-sol')).toBe('GPT-5.6 Sol');
    expect(modelDisplayName('gpt-5.3-codex')).toBe('GPT-5.3 Codex');
    expect(modelDisplayName('gpt-4o')).toBe('GPT-4o');
    expect(modelDisplayName('claude-opus-5')).toBe('Claude Opus 5');
    expect(modelDisplayName('gpt-5-mini')).toBe('GPT-5 mini');
    // A model nobody here has heard of is still readable, and nothing is added to it.
    expect(modelDisplayName('some-vendor-model-9')).toBe('Some Vendor Model 9');
    expect(modelDisplayName(null)).toBeNull();
    expect(modelDisplayName('   ')).toBeNull();
  });
});

describe('attaching executions to arrows', () => {
  it('gives one execution per attempt its own arrow entry, newest last', () => {
    const records = [
      execution({ execution_id: 'implementation:r:0', attempt: 1 }),
      execution({ execution_id: 'implementation:r:1', attempt: 2 }),
    ].map((item) => ({ ...item, repository_id: 'r' }));
    const attached = executionsByEdge(records, ['r']);
    expect(attached.get('repo:r->implement:r')?.map((item) => item.attempt)).toEqual([1, 2]);
    expect(edgeLabel(attached.get('repo:r->implement:r') ?? [])?.previous).toBe(1);
  });

  it('spreads a feature-wide convergence execution across every repository lane', () => {
    const records = [
      execution({
        execution_id: 'integration_review:1',
        from_stage: 'review',
        to_stage: 'integration_review',
        repository_id: null,
        handler: 'Integration reviewer',
        model: 'gpt-5.6-sol',
      }),
    ];
    const attached = executionsByEdge(records, ['api', 'web']);
    expect(attached.has('review:api->integration_review')).toBe(true);
    expect(attached.has('review:web->integration_review')).toBe(true);
  });
});

describe('the workflow graph arrows', () => {
  it('names the model on an AI-backed transition, with the effort it was asked for', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'implementation:admanager-server:0',
          attempt: 1,
          max_attempts: 8,
          model: 'gpt-5.6-sol',
          reasoning_effort: 'xhigh',
          provider: 'openai',
        }),
      ]),
    );
    const button = await edge(/handled by GPT-5\.6 Sol · effort: xhigh/i);
    // The name and the effort are two tokens, so a narrow column gap can put them on two
    // lines without ever breaking the identifier itself. The token says "effort:" because a
    // bare "xhigh" beside a model name reads as a performance tier, which it is not.
    expect(within(button).getByText('GPT-5.6 Sol')).toBeInTheDocument();
    expect(within(button).getByText('effort: xhigh')).toBeInTheDocument();
  });

  it('names the validator on a deterministic transition and claims no model for it', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'validation:admanager-server:0',
          from_stage: 'implementation',
          to_stage: 'validation',
          handler_type: 'deterministic',
          handler: 'Validator',
          agent_type: null,
          model: null,
          model_resolved: false,
          status: 'failed',
          validation_passed: 1,
          validation_total: 2,
          command: ['npm', 'run', 'lint'],
          exit_code: 1,
          failure_classification: 'lint',
        }),
      ]),
    );
    const button = await edge(/handled by Validator/i);
    expect(within(button).getByText('Validator')).toBeInTheDocument();
    expect(screen.queryByText(/GPT/)).not.toBeInTheDocument();
  });

  it('says which gate a failed validation failed on, in one word', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'validation:admanager-server:0',
          from_stage: 'implementation',
          to_stage: 'validation',
          handler_type: 'deterministic',
          handler: 'Validator',
          agent_type: null,
          model: null,
          model_resolved: false,
          status: 'failed',
          validation_passed: 1,
          validation_total: 2,
          command: ['npm', 'test'],
          exit_code: 1,
          failure_classification: 'test',
        }),
      ]),
    );
    const button = await edge(/failure class own tests/i);
    expect(within(button).getByText('Failed · own tests')).toBeInTheDocument();
  });

  it('names the external service that did not answer, rather than showing nothing', async () => {
    // A workstream that spent its fault allowance carries a feature-level classification
    // instead of a retry-policy one, and an operator retry afterwards makes it the previous
    // attempt this arrow reads. Both values were unmapped, so the arrow showed no word at
    // all for the two failures where "which service" is the entire question.
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'retry:admanager-server:0',
          from_stage: 'review',
          to_stage: 'implementation',
          is_retry: true,
          attempt: 2,
          handler_type: 'agent',
          handler: 'Engineer',
          status: 'failed',
          failure_classification: 'git_remote_unavailable',
        }),
      ]),
    );
    const button = await edge(/failure class git remote/i);
    expect(within(button).getByText('Failed · git remote')).toBeInTheDocument();
  });

  it('names the design source too, which is the third service and was the unmapped one', async () => {
    // AB-Feature-228 ended on a Figma rate limit. The server filed it as `provider_unavailable`
    // until 2026-09-12, and once it filed it honestly the word was missing here instead -- so
    // the arrow went from naming the wrong service to naming none.
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'retry:admanager-server:0',
          from_stage: 'review',
          to_stage: 'implementation',
          is_retry: true,
          attempt: 2,
          handler_type: 'agent',
          handler: 'Engineer',
          status: 'failed',
          failure_classification: 'design_source_unavailable',
        }),
      ]),
    );
    const button = await edge(/failure class design source/i);
    expect(within(button).getByText('Failed · design source')).toBeInTheDocument();
  });

  it('keeps an unmapped classification off the arrow rather than abbreviating a guess', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'validation:admanager-server:0',
          from_stage: 'implementation',
          to_stage: 'validation',
          handler_type: 'deterministic',
          handler: 'Validator',
          agent_type: null,
          model: null,
          model_resolved: false,
          status: 'failed',
          validation_passed: 0,
          validation_total: 1,
          failure_classification: 'something_new_the_server_writes',
        }),
      ]),
    );
    const button = await edge(/handled by Validator/i);
    expect(within(button).getByText('Failed')).toBeInTheDocument();
    expect(within(button).queryByText(/something/)).not.toBeInTheDocument();
  });

  it('says a retry is a retry, which attempt it is, and what is answering it', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'retry:admanager-server:1',
          from_stage: 'review',
          to_stage: 'implementation',
          is_retry: true,
          attempt: 2,
          max_attempts: 8,
          model: 'gpt-5.3-codex',
          status: 'running',
        }),
      ]),
    );
    const button = await edge(/Retry attempt 2 of 8/i);
    expect(within(button).getByText('Retry 2 / 8')).toBeInTheDocument();
    expect(within(button).getByText('GPT-5.3 Codex')).toBeInTheDocument();
    expect(within(button).getByText('Running')).toBeInTheDocument();
  });

  it('shows the model a queued retry will use only because the backend resolved it', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'retry:admanager-server:1',
          from_stage: 'review',
          to_stage: 'implementation',
          is_retry: true,
          attempt: 2,
          max_attempts: 8,
          model: 'gpt-5.3-codex',
          model_role: 'scoped_fix',
          status: 'queued',
        }),
      ]),
    );
    const button = await edge(/Retry attempt 2 of 8.*GPT-5\.3 Codex.*queued/i);
    expect(within(button).getByText('Queued')).toBeInTheDocument();
  });

  it('says the model is not chosen yet rather than predicting one', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'retry:admanager-server:1',
          from_stage: 'review',
          to_stage: 'implementation',
          is_retry: true,
          attempt: 2,
          max_attempts: 8,
          model: null,
          model_resolved: false,
          status: 'queued',
        }),
      ]),
    );
    const button = await edge(/no model resolved yet/i);
    expect(within(button).getByText('Model pending')).toBeInTheDocument();
  });

  it('does not promise a retry the reliability logic refused', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'retry-refused:admanager-server:3',
          from_stage: 'review',
          to_stage: 'implementation',
          is_retry: true,
          attempt: 3,
          max_attempts: 8,
          handler_type: 'human',
          handler: 'Human action',
          agent_type: null,
          model: 'gpt-5.3-codex',
          model_resolved: false,
          status: 'needs_human',
          human_action: 'retry_refused',
          human_requirement: 'No material progress: the same diagnostics came back unchanged.',
          attempt_meaning: '3 attempt(s) were spent of a maximum of 8. No further attempt is scheduled.',
          failure_classification: 'contract_mismatch',
        }),
      ]),
    );
    const button = await edge(/needs a person/i);
    expect(within(button).getByText('No further retry')).toBeInTheDocument();
    expect(within(button).queryByText(/Retry 4/)).not.toBeInTheDocument();
    // The class of the failure that exhausted the retries, in the platform's one word.
    expect(within(button).getByText(/· contract/)).toBeInTheDocument();
  });
});

describe('the execution drawer', () => {
  const RETRY = execution({
    execution_id: 'retry:admanager-server:1',
    from_stage: 'review',
    to_stage: 'implementation',
    is_retry: true,
    attempt: 2,
    max_attempts: 8,
    attempt_meaning: 'Remediation attempt 2 of a maximum of 8.',
    model: 'gpt-5.3-codex',
    reasoning_effort: 'high',
    model_role: 'scoped_fix',
    provider: 'openai',
    status: 'running',
    failure_classification: 'review_scope_failure',
    failure_summary: 'Authentication middleware is not applied to the new endpoint.',
    failure_severity: 'high',
    remediation_summary:
      'Apply the existing requireAdminAuth middleware to the new route and update the tests.',
    routing_reason: 'STANDARD review remediation.',
    review_verdict: 'changes_requested',
    review_finding_count: 3,
    revision_before: 'abc12345abc',
    revision_after: 'def67890def',
    review_artifact_id: '007_review.admanager-server.attempt-0.json',
    result_artifact_id: '011_child_workflow_result.admanager-server.attempt-0.json',
    counter_label: 'Implementation retries',
    counter_value: 1,
    counter_limit: 4,
    input_fingerprint: 'sha256:abcdef',
  });

  it('opens from the arrow and separates why it failed from what must be fixed', async () => {
    renderWorkflow(apiWith([RETRY]));
    await userEvent.click(await edge(/Retry attempt 2 of 8/i));

    const drawer = await screen.findByRole('dialog');
    expect(within(drawer).getByText('Retry execution')).toBeInTheDocument();
    expect(within(drawer).getByText('Why the previous attempt failed')).toBeInTheDocument();
    expect(
      within(drawer).getByText('Authentication middleware is not applied to the new endpoint.'),
    ).toBeInTheDocument();
    expect(within(drawer).getByText('What needs to be fixed')).toBeInTheDocument();
    expect(within(drawer).getByText(/requireAdminAuth middleware/)).toBeInTheDocument();
    // The agent role and the model are separate facts and are shown as separate rows.
    expect(within(drawer).getByText('Engineer')).toBeInTheDocument();
    expect(within(drawer).getByText('GPT-5.3 Codex')).toBeInTheDocument();
    expect(within(drawer).getByText('high')).toBeInTheDocument();
    // The ratio, and the backend's own sentence for what it counts.
    expect(within(drawer).getByText('2 / 8')).toBeInTheDocument();
    expect(
      within(drawer).getByText('Remediation attempt 2 of a maximum of 8.'),
    ).toBeInTheDocument();
    // Both revisions, so an engineer can diff exactly what changed.
    expect(within(drawer).getByTitle('abc12345abc')).toBeInTheDocument();
    expect(within(drawer).getByTitle('def67890def')).toBeInTheDocument();
  });

  it('routes to the existing evidence rather than rebuilding it', async () => {
    renderWorkflow(apiWith([RETRY]));
    await userEvent.click(await edge(/Retry attempt 2 of 8/i));
    const drawer = await screen.findByRole('dialog');

    const review = within(drawer).getByRole('link', { name: 'Open review' });
    expect(review).toHaveAttribute(
      'href',
      '/features/adunit-deactivate-live-086/repositories/admanager-server?view=review',
    );
    expect(within(drawer).getByRole('link', { name: 'Open changes' })).toHaveAttribute(
      'href',
      '/features/adunit-deactivate-live-086/repositories/admanager-server?view=changes',
    );
    expect(within(drawer).getByRole('link', { name: 'Open attempt record' })).toHaveAttribute(
      'href',
      '/features/adunit-deactivate-live-086/artifacts?artifact=011_child_workflow_result.admanager-server.attempt-0.json',
    );
  });

  it('links a validation transition to the validation view and shows the failing command', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'validation:admanager-server:0',
          from_stage: 'implementation',
          to_stage: 'validation',
          handler_type: 'deterministic',
          handler: 'Validator',
          agent_type: null,
          model: null,
          model_resolved: false,
          status: 'failed',
          validation_passed: 1,
          validation_total: 2,
          command: ['npm', 'run', 'lint'],
          exit_code: 1,
          duration_seconds: 4.5,
        }),
      ]),
    );
    await userEvent.click(await edge(/handled by Validator/i));
    const drawer = await screen.findByRole('dialog');

    expect(within(drawer).getByText('Validator')).toBeInTheDocument();
    expect(within(drawer).getByText('1 of 2 passed')).toBeInTheDocument();
    expect(within(drawer).getByText('1')).toBeInTheDocument();
    expect(within(drawer).getAllByTitle('npm run lint').length).toBeGreaterThan(0);
    expect(within(drawer).getByRole('link', { name: 'Open validation' })).toHaveAttribute(
      'href',
      '/features/adunit-deactivate-live-086/repositories/admanager-server?view=validation',
    );
  });

  /**
   * The retry drawer's selector (65 C).
   *
   * The mechanism was always here -- the selection already drove every block below it. What
   * shipped was a list of subtle buttons at the BOTTOM, under the detail it drove, with
   * nothing saying the list was a selector. The operator watching run 201 read it as a log.
   */
  const twoAttempts = () => [
    execution({
      execution_id: 'retry:admanager-server:1',
      from_stage: 'review',
      to_stage: 'implementation',
      is_retry: true,
      attempt: 2,
      max_attempts: 8,
      model: 'gpt-5.3-codex',
      failure_summary: 'Review still found the auth issue.',
      remediation_summary: 'Apply the middleware to the new route.',
      revision_before: 'aaaaaaaaaaaa',
      status: 'completed',
    }),
    execution({
      execution_id: 'retry:admanager-server:2',
      from_stage: 'review',
      to_stage: 'implementation',
      is_retry: true,
      attempt: 3,
      max_attempts: 8,
      model: 'gpt-5.6-sol',
      status: 'running',
    }),
  ];

  it('C1 — the dropdown drives the drawer, and nothing else is fetched to do it', async () => {
    let executionReads = 0;
    const attempts = twoAttempts();
    renderWorkflow(
      apiWith(attempts, {
        getExecutions: async () => {
          executionReads += 1;
          return { feature_id: FEATURE.feature_id, executions: attempts };
        },
      }),
    );

    const button = await edge(/Retry attempt 3 of 8/i);
    expect(within(button).getByText('Running · +1 earlier')).toBeInTheDocument();
    await userEvent.click(button);
    const drawer = await screen.findByRole('dialog');
    await waitFor(() => expect(executionReads).toBeGreaterThan(0));
    const reads = executionReads;

    // The control is at the top, labelled, and the newest attempt is what opened.
    const picker = within(drawer).getByLabelText('Attempt');
    expect(within(drawer).getByText('GPT-5.6 Sol', { selector: 'dd' })).toBeInTheDocument();

    // Choosing the earlier attempt details THAT attempt: its callouts, its model, its
    // revision -- which is already how the component works, so this is wiring.
    await userEvent.selectOptions(picker, 'retry:admanager-server:1');
    await waitFor(() =>
      expect(within(drawer).getByText('Why the previous attempt failed')).toBeInTheDocument(),
    );
    expect(within(drawer).getByText('Review still found the auth issue.')).toBeInTheDocument();
    expect(within(drawer).getByText('What needs to be fixed')).toBeInTheDocument();
    expect(within(drawer).getByText('GPT-5.3 Codex', { selector: 'dd' })).toBeInTheDocument();
    expect(within(drawer).getByText('retry:admanager-server:1')).toBeInTheDocument();

    // And back to the newest.
    await userEvent.selectOptions(picker, 'retry:admanager-server:2');
    await waitFor(() =>
      expect(within(drawer).getByText('GPT-5.6 Sol', { selector: 'dd' })).toBeInTheDocument(),
    );

    // No fetch of any kind happened to swap attempts: every record was already on the arrow.
    expect(executionReads).toBe(reads);
  });

  it('C1 — the bottom list is gone, so one choice has one control', async () => {
    renderWorkflow(apiWith(twoAttempts()));
    await userEvent.click(await edge(/Retry attempt 3 of 8/i));
    const drawer = await screen.findByRole('dialog');

    expect(within(drawer).queryByText('Attempts on this transition')).not.toBeInTheDocument();
    expect(within(drawer).getByLabelText('Attempt')).toBeInTheDocument();
  });

  it('C3 — nothing the old list showed is lost', async () => {
    renderWorkflow(apiWith(twoAttempts()));
    await userEvent.click(await edge(/Retry attempt 3 of 8/i));
    const drawer = await screen.findByRole('dialog');

    // The list carried a number over its maximum, a handler or model name, and a status word.
    // All three are in the option text -- with the glyph never travelling without the word.
    const options = within(drawer)
      .getAllByRole('option')
      .map((option) => option.textContent);
    expect(options).toEqual([
      '● 3 / 8 · GPT-5.6 Sol · Running',
      '✓ 2 / 8 · GPT-5.3 Codex · Completed',
    ]);
    // The fourth thing it carried was the failure summary, which is a sentence: it is rendered
    // in full in the Explanation block for whichever attempt is selected.
    await userEvent.selectOptions(
      within(drawer).getByLabelText('Attempt'),
      'retry:admanager-server:1',
    );
    await waitFor(() =>
      expect(within(drawer).getByText('Review still found the auth issue.')).toBeInTheDocument(),
    );
  });

  it('C3 — an attempt the platform never numbered keeps the list\'s own treatment', async () => {
    const attempts = twoAttempts();
    renderWorkflow(
      apiWith([
        { ...attempts[0]!, attempt: null, max_attempts: null } as ExecutionRecord,
        attempts[1]!,
      ]),
    );
    await userEvent.click(await edge(/Retry attempt 3 of 8/i));
    const drawer = await screen.findByRole('dialog');

    // '—', exactly as the list rendered it. Not '0', and not a number invented for the gap.
    expect(
      within(drawer).getByRole('option', { name: '✓ — · GPT-5.3 Codex · Completed' }),
    ).toBeInTheDocument();
  });

  it('C2 — a single execution renders no picker and no list', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'retry:admanager-server:1',
          from_stage: 'review',
          to_stage: 'implementation',
          is_retry: true,
          attempt: 2,
          max_attempts: 8,
          model: 'gpt-5.3-codex',
          status: 'completed',
        }),
      ]),
    );
    await userEvent.click(await edge(/Retry attempt 2 of 8/i));
    const drawer = await screen.findByRole('dialog');

    // Nothing to choose between, so no control at all -- byte-identical to today.
    expect(within(drawer).queryByLabelText('Attempt')).not.toBeInTheDocument();
    expect(within(drawer).queryByRole('combobox')).not.toBeInTheDocument();
    expect(within(drawer).queryByText('Attempts on this transition')).not.toBeInTheDocument();
  });

  it('says a human is required without assigning a model to the person', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'human:clarification:002_technical_prd.json',
          from_stage: 'technical_prd',
          to_stage: 'integration_contract',
          repository_id: null,
          handler_type: 'human',
          handler: 'Human action',
          agent_type: null,
          model: null,
          model_resolved: false,
          status: 'needs_human',
          human_action: 'clarification',
          human_requirement: '2 question(s) about this feature must be answered.',
          attempt: 1,
          max_attempts: 10,
        }),
      ]),
    );
    await userEvent.click(await edge(/handled by Human action/i));
    const drawer = await screen.findByRole('dialog');

    expect(within(drawer).getByText('What is required')).toBeInTheDocument();
    expect(
      within(drawer).getByText('2 question(s) about this feature must be answered.'),
    ).toBeInTheDocument();
    expect(within(drawer).queryByText('Reasoning')).not.toBeInTheDocument();
    expect(within(drawer).getByRole('link', { name: 'Answer the questions' })).toHaveAttribute(
      'href',
      '/features/adunit-deactivate-live-086/prd',
    );
  });
});

describe('several repositories', () => {
  it('labels every lane from its own executions and never shares one between them', async () => {
    const executions = [
      execution({
        execution_id: 'implementation:admanager-server:0',
        repository_id: 'admanager-server',
        attempt: 1,
        max_attempts: 8,
        model: 'gpt-5.6-sol',
      }),
      execution({
        execution_id: 'implementation:ab-console-admin:0',
        repository_id: 'ab-console-admin',
        attempt: 1,
        max_attempts: 8,
        model: 'gpt-5.3-codex',
      }),
    ];
    renderWorkflow(
      apiWith(executions, {
        getWorkstreams: async () => ({
          feature_id: FEATURE.feature_id,
          workstreams: [
            workstream(),
            workstream({ repository_id: 'ab-console-admin', repository_name: 'Console' }),
          ],
        }),
      }),
    );

    expect(await edge(/in admanager-server.*GPT-5\.6 Sol/i)).toBeInTheDocument();
    expect(await edge(/in ab-console-admin.*GPT-5\.3 Codex/i)).toBeInTheDocument();
  });
});

describe('the currently block', () => {
  it('names the repository its retry counter belongs to', async () => {
    renderWorkflow(
      apiWith([], {
        getWorkstreams: async () => ({
          feature_id: FEATURE.feature_id,
          workstreams: [workstream({ validation_retry_count: 1 })],
        }),
      }),
    );
    // "Retry 1 of 2" alone is ambiguous the moment two lanes are retrying; the row says
    // whose counter it is.
    const count = await screen.findByText('1 of 2');
    expect(
      within(count.parentElement as HTMLElement).getByTitle('admanager-server'),
    ).toBeInTheDocument();
  });
});

describe('when the platform recorded no execution', () => {
  it("still draws the retry loop the workstream's own counters describe", async () => {
    renderWorkflow(apiWith([]));
    // The loop label is the graph's existing reading of `retry_count` against the feature's
    // published limit. It has nothing to open, because there is no execution behind it.
    const label = await screen.findByTitle('Retry 2 / 8');
    expect(label).toBeDisabled();
  });

  it('does not fail the view when the executions read fails', async () => {
    renderWorkflow(
      apiWith([], {
        getExecutions: vi.fn().mockRejectedValue(new Error('unavailable')),
      }),
    );
    expect(await screen.findByRole('group', { name: 'Feature execution graph' })).toBeInTheDocument();
  });
});

describe('the planning nodes list their journaled calls', () => {
  it('renders each pre-coding call as a line on the node its stage names', async () => {
    renderWorkflow(
      apiWith([
        execution({
          execution_id: 'planning_call:pm-draft',
          from_stage: 'request',
          to_stage: 'technical_prd',
          repository_id: null,
          handler: 'Product manager',
          agent_type: 'product_manager',
          started_at: '2026-08-24T22:41:00Z',
          completed_at: '2026-08-24T22:41:39Z',
          duration_seconds: 39,
        }),
        execution({
          execution_id: 'planning_call:recon-admanager',
          from_stage: 'technical_prd',
          to_stage: 'integration_contract',
          repository_id: 'admanager-server',
          handler: 'Repository reconnaissance',
          agent_type: 'repository_reconnaissance',
          started_at: '2026-08-24T22:42:00Z',
          completed_at: '2026-08-24T22:44:00Z',
          duration_seconds: 120,
        }),
        execution({
          execution_id: 'planning_call:planner',
          from_stage: 'technical_prd',
          to_stage: 'integration_contract',
          repository_id: null,
          handler: 'Technical planner',
          agent_type: 'feature_planner',
          status: 'running',
          started_at: '2026-08-24T22:45:00Z',
          completed_at: null,
        }),
      ]),
    );
    // The Product manager node speaks the journal's own step name, ticked once it completed.
    const productManager = await screen.findByRole('button', {
      name: /Stage, Product manager.*product_manager succeeded/,
    });
    expect(within(productManager).getByText('product_manager')).toBeInTheDocument();
    // The Shared contract node lists reconnaissance and the planner, each with its own state,
    // and names the repository the reconnaissance clone read.
    const contract = await screen.findByRole('button', {
      name: /Stage, Shared contract.*repository_reconnaissance succeeded, feature_planner running/,
    });
    expect(within(contract).getByText('repository_reconnaissance')).toBeInTheDocument();
    expect(within(contract).getByText(/admanager-server · 2m 0s/)).toBeInTheDocument();
    expect(within(contract).getByText('feature_planner')).toBeInTheDocument();
  });

  it('keeps the planning nodes as they were when no planning calls were journaled', async () => {
    renderWorkflow(apiWith([execution({ execution_id: 'implementation:admanager-server:0' })]));
    const productManager = await screen.findByRole('button', { name: /Stage, Product manager/ });
    expect(within(productManager).queryByText('product_manager')).not.toBeInTheDocument();
  });
});
