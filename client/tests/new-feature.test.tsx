import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { NewFeaturePage } from '@/pages/NewFeaturePage';
import type { FeatureApi } from '@/api/features';
import { ApiError } from '@/api/errors';
import { SETUP_READY, savedRepository, stubApi } from './fixtures';
import { modelSetupsSchema } from '@/schemas/feature';
import capturedSetups from './fixtures/model-setups.mixed.json';

/**
 * Asking for a feature.
 *
 * Two things are the subject of most of these: what the form asks for, and what it refuses to
 * show until the prerequisites are met. The identifiers this form used to ask for — a feature
 * id, and a repository id, name and role each — are the server's to derive, so a test that
 * filled one in would be encoding the thing this pass removed.
 */

function renderForm(api: Partial<FeatureApi>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi(api)} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={['/features/new']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/new" element={<NewFeaturePage />} />
          <Route path="/features/:featureId" element={<p>Workspace for feature</p>} />
          <Route path="/settings" element={<p>Settings page</p>} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

/** Everything the form requires, and nothing else. */
async function fillMinimum(user: ReturnType<typeof userEvent.setup>) {
  await user.type(await screen.findByLabelText('Title'), 'Deactivate ad units');
  await user.type(screen.getByLabelText('Problem statement'), 'Operators cannot deactivate.');
  await user.click(screen.getByRole('button', { name: /admanager_console-2\.0/ }));
}

describe('the prerequisites for creating a feature', () => {
  it('refuses the form and names each provider that is missing', async () => {
    const createFeature = vi.fn();
    renderForm({
      createFeature,
      getSetupState: async () => ({
        ...SETUP_READY,
        credentials_ready: false,
        providers: [
          { provider: 'openai', label: 'OpenAI', configured: true },
          { provider: 'github', label: 'GitHub', configured: false },
        ],
      }),
    });

    expect(
      await screen.findByRole('heading', { name: 'Provider setup required', level: 1 }),
    ).toBeInTheDocument();
    const providers = screen.getByRole('list', { name: 'Required providers' });
    // Partial setup is shown as partial: the one that is done says so.
    expect(within(providers).getByText('OpenAI').closest('li')).toHaveTextContent('Configured');
    expect(within(providers).getByText('GitHub').closest('li')).toHaveTextContent(
      'Not configured',
    );
    expect(screen.getByRole('link', { name: /Configure credentials/ })).toHaveAttribute(
      'href',
      '/settings',
    );
    // And the form is genuinely not there, rather than there and disabled.
    expect(screen.queryByLabelText('Problem statement')).not.toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('shows the form once every required credential is configured', async () => {
    renderForm({ createFeature: vi.fn() });

    expect(await screen.findByLabelText('Problem statement')).toBeInTheDocument();
    expect(screen.queryByText('Provider setup required')).not.toBeInTheDocument();
  });

  it('asks for a repository before a feature, and still allows a one-time one', async () => {
    renderForm({
      createFeature: vi.fn(),
      getSetupState: async () => ({
        ...SETUP_READY,
        repositories_ready: false,
        saved_repository_count: 0,
      }),
      listSavedRepositories: async () => ({ repositories: [], suggested_types: [] }),
    });
    const user = userEvent.setup();

    expect(
      await screen.findByRole('heading', { name: 'No repositories configured', level: 1 }),
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Configure repositories/ })).toHaveAttribute(
      'href',
      '/settings',
    );

    await user.click(screen.getByRole('button', { name: /one-time repository/ }));
    expect(await screen.findByLabelText('Problem statement')).toBeInTheDocument();
  });
});

describe('NewFeaturePage', () => {
  it('starts a feature from a title, a problem and a saved repository', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-created' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(await screen.findByText('Workspace for feature')).toBeInTheDocument();
    const [input, options] = createFeature.mock.calls[0] as [
      Record<string, unknown>,
      { idempotencyKey: string; credentials?: unknown },
    ];
    const prd = input.prd as Record<string, unknown>;
    expect(prd.title).toBe('Deactivate ad units');
    expect(prd.problem_statement).toBe('Operators cannot deactivate.');
    // Nothing is written on the author's behalf to satisfy validation: what was not asked
    // for is sent as absent, and the platform derives it.
    expect(prd.goals).toEqual([]);
    expect(prd.user_stories).toEqual([]);
    expect(prd.requirements).toEqual([]);
    // Live by default: that is what somebody opening this page came to do, and the mode is on
    // the form rather than behind the disclosure precisely because the default has
    // consequences.
    expect(input.execution_mode).toBe('live');
    // The platform the feature runs on travels with it, and is fixed from here on. OpenAI is
    // the first-visit default while Claude sits behind the not-ready gate.
    expect(input.agent_platform).toBe('openai');
    // Standard when the person touches nothing. The server's own default is `high` for API
    // compatibility, so the interface's recommendation has to be stated explicitly.
    expect(input.performance_tier).toBe('medium');
    expect(options.idempotencyKey).toBeTruthy();
    // Credentials come from the account now: the work happens after this request is answered,
    // so a header would be of no use to it.
    expect(options.credentials).toBeUndefined();
  });

  it('offers six labeled options, preselects Standard on OpenAI, and gates Claude as not ready', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-platform' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    // Beside the execution mode rather than inside the advanced disclosure: the disclosure is
    // for what the platform works out for itself, and this is neither derived nor safe to
    // default silently.
    const platform = await screen.findByLabelText('Platform and performance tier');
    expect(platform).toBeVisible();
    expect(platform).toHaveValue('openai:medium');
    // All six (platform, tier) pairings stay listed, honestly labeled, from the saved real
    // /setup payload. The OpenAI pairings are selectable; the Claude pairings are configured
    // on the server but withheld by the not-ready gate — rendered disabled with the reason
    // on the option itself, never hidden.
    for (const name of ['OpenAI — Economy', 'OpenAI — Standard', 'OpenAI — Max']) {
      expect(
        within(platform as HTMLSelectElement).getByRole('option', { name }),
      ).toBeEnabled();
    }
    for (const name of [
      'Claude — Economy (Not ready)',
      'Claude — Standard (Not ready)',
      'Claude — Max (Not ready)',
    ]) {
      expect(
        within(platform as HTMLSelectElement).getByRole('option', { name }),
      ).toBeDisabled();
    }

    // The one thing a submitter cannot see for themselves: each stage of the work, the model
    // this choice resolves it to, the effort it is sent at, and that none of it can be
    // changed afterwards. The roles, model names and efforts come from the server; only the
    // spelling is the client's.
    const stages = screen.getByRole('table', {
      name: 'Stages, the models they run on and the effort each is sent at',
    });
    const reasoningRow = within(stages).getByText('Reasoning').closest('tr')!;
    expect(within(reasoningRow).getByTitle('gpt-5.6-sol')).toHaveTextContent('GPT-5.6 Sol');
    expect(reasoningRow).toHaveTextContent('high');
    const reviewRow = within(stages).getByText('Review').closest('tr')!;
    expect(within(reviewRow).getByTitle('gpt-5.6-terra')).toHaveTextContent('GPT-5.6 Terra');
    const fixRow = within(stages).getByText('Scoped fix').closest('tr')!;
    expect(within(fixRow).getByTitle('gpt-5.3-codex')).toHaveTextContent('GPT-5.3 Codex');
    expect(screen.getByText(/Standard is the recommended default/)).toBeInTheDocument();
    expect(screen.getByText(/cannot be changed after the feature starts/)).toBeInTheDocument();

    // Switching the selection re-derives the table from the newly chosen pairing: Economy
    // resolves reasoning to a different model, at a different effort, than Standard does.
    await user.selectOptions(platform, 'openai:low');
    const economyReasoning = within(
      screen.getByRole('table', {
        name: 'Stages, the models they run on and the effort each is sent at',
      }),
    )
      .getByText('Reasoning')
      .closest('tr')!;
    expect(within(economyReasoning).getByTitle('gpt-5.6-terra')).toBeInTheDocument();
    expect(economyReasoning).toHaveTextContent('medium');

    // Max is the tier the effort column exists to distinguish: the same reasoning model as
    // Standard, sent at a different effort. A table of models alone shows the two as equal.
    await user.selectOptions(platform, 'openai:high');
    const maxReasoning = within(
      screen.getByRole('table', {
        name: 'Stages, the models they run on and the effort each is sent at',
      }),
    )
      .getByText('Reasoning')
      .closest('tr')!;
    expect(within(maxReasoning).getByTitle('gpt-5.6-sol')).toBeInTheDocument();
    expect(maxReasoning).toHaveTextContent('max');

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [
      { agent_platform: string; performance_tier: string },
    ];
    expect(input.agent_platform).toBe('openai');
    expect(input.performance_tier).toBe('high');
  });

  it('carries the Economy scope warning on the option itself, not only in documentation', async () => {
    renderForm({ createFeature: vi.fn() });
    const user = userEvent.setup();

    const platform = await screen.findByLabelText('Platform and performance tier');
    await user.selectOptions(platform, 'openai:low');

    expect(
      screen.getByText(
        /Economy is for small, well-bounded changes — larger work will cost more here, not less, through retries/,
      ),
    ).toBeInTheDocument();
  });

  it('falls back to OpenAI when the remembered provider is gated as not ready', async () => {
    window.localStorage.setItem('newFeature.lastAgentPlatform', 'anthropic');
    try {
      renderForm({ createFeature: vi.fn() });

      const platform = await screen.findByLabelText('Platform and performance tier');
      // A remembered Claude preference must not open the form on a selection it refuses.
      // The stored memory itself is kept, so re-opening the platform restores it.
      expect(platform).toHaveValue('openai:medium');
      expect(window.localStorage.getItem('newFeature.lastAgentPlatform')).toBe('anthropic');
    } finally {
      window.localStorage.removeItem('newFeature.lastAgentPlatform');
    }
  });

  it('preselects Standard on whichever provider the operator last used', async () => {
    window.localStorage.setItem('newFeature.lastAgentPlatform', 'openai');
    try {
      renderForm({ createFeature: vi.fn() });

      const platform = await screen.findByLabelText('Platform and performance tier');
      // The provider is remembered; the tier is always Standard. Remembering that somebody
      // once picked Max would quietly turn one expensive choice into a standing one.
      expect(platform).toHaveValue('openai:medium');
    } finally {
      window.localStorage.removeItem('newFeature.lastAgentPlatform');
    }
  });

  it('renders an unconfigured selection disabled with the reason and refuses to submit', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-one-platform' });
    renderForm({
      createFeature,
      getSetupState: async () => ({
        ...SETUP_READY,
        providers: [
          { provider: 'openai', label: 'OpenAI', configured: true },
          { provider: 'github', label: 'GitHub', configured: true },
        ],
        // What a partially configured deployment serves: OpenAI Standard is not offered at
        // all, OpenAI Economy stays visible and disabled, and only OpenAI Max can run.
        agent_platforms: SETUP_READY.agent_platforms
          .filter(
            (item) => !(item.platform === 'openai' && item.performance_tier === 'medium'),
          )
          .map((item) =>
            item.platform === 'openai' && item.performance_tier === 'low'
              ? { ...item, configured: false, models: {} as Record<string, string> }
              : item,
          ),
      }),
    });
    const user = userEvent.setup();

    const platform = await screen.findByLabelText('Platform and performance tier');
    // The default — Standard on OpenAI — cannot run here. The selection stays exactly where
    // it was, rendered as itself and disabled with the reason: the form quietly re-aiming an
    // unconfigured choice at load is what ran AB-Feature-173 on the wrong platform.
    expect(platform).toHaveValue('openai:medium');
    expect(
      within(platform as HTMLSelectElement).getByRole('option', {
        name: /openai — medium — not configured on this deployment/,
      }),
    ).toBeDisabled();
    expect(
      within(platform as HTMLSelectElement).getByRole('option', {
        name: /OpenAI — Economy — not configured on this deployment/,
      }),
    ).toBeDisabled();
    // No stage table for a selection that does not resolve: there are no models to promise,
    // and inventing rows here would state a fallback as fact.
    expect(
      screen.queryByRole('table', { name: 'Stages, the models they run on and the effort each is sent at' }),
    ).not.toBeInTheDocument();

    // And it refuses to submit rather than substituting a selection.
    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));
    expect(
      await screen.findByText(/the form will not substitute one for you/),
    ).toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();

    // The person chooses; only then does it submit — with what they chose.
    await user.selectOptions(platform, 'openai:high');
    await user.click(screen.getByRole('button', { name: 'Create feature' }));
    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [
      { agent_platform: string; performance_tier: string },
    ];
    expect(input.agent_platform).toBe('openai');
    expect(input.performance_tier).toBe('high');
  });

  it('never asks for a feature ID, and says the server assigns one', async () => {
    renderForm({ createFeature: vi.fn() });

    expect(await screen.findByText('Feature ID')).toBeInTheDocument();
    expect(screen.getByText('Assigned automatically on submission')).toBeInTheDocument();
    // Not a field. Nothing is reserved by opening this page, and nothing can be typed.
    expect(screen.queryByLabelText('Feature ID')).not.toBeInTheDocument();
  });

  it('sends only the repository fields the server cannot derive', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-derived' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [{ repositories: Record<string, unknown>[] }];
    // Three keys. The saved label is deliberately not among them: an organisational choice
    // must not reach the planner as a role.
    expect(input.repositories).toEqual([
      {
        repository_url: 'https://github.com/Appbroda/admanager_console-2.0',
        default_branch: 'master',
        required: true,
      },
    ]);
  });

  it('shows a selected repository with the label and branch it was saved under', async () => {
    renderForm({ createFeature: vi.fn() });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: /admanager_console-2\.0/ }));

    const selected = screen.getByRole('list', { name: 'Selected repositories' });
    expect(within(selected).getByText('admanager_console-2.0')).toBeInTheDocument();
    expect(within(selected).getByText('Backend')).toBeInTheDocument();
    expect(within(selected).getByText('master')).toBeInTheDocument();
  });

  it('selects and deselects a repository from the same control', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    const option = await screen.findByRole('button', { name: /admanager_console-2\.0/ });
    await user.click(option);
    expect(option).toHaveAttribute('aria-pressed', 'true');
    await user.click(option);
    expect(option).toHaveAttribute('aria-pressed', 'false');

    await user.click(screen.getByRole('button', { name: 'Create feature' }));
    expect(await screen.findByText('Select at least one repository')).toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('supports any number of repositories, including two of the same type', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-many' });
    renderForm({
      createFeature,
      listSavedRepositories: async () => ({
        repositories: [
          savedRepository({ configuration_id: 'repo-1', name: 'payments-api', repository_url: 'https://github.com/owner/payments-api' }),
          savedRepository({ configuration_id: 'repo-2', name: 'reporting-api', repository_url: 'https://github.com/owner/reporting-api' }),
          savedRepository({
            configuration_id: 'repo-3',
            name: 'ab-console-admin',
            repository_url: 'https://github.com/owner/ab-console-admin',
            repository_type: 'Frontend',
          }),
        ],
        suggested_types: [],
      }),
    });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Title'), 'Three repositories');
    await user.type(screen.getByLabelText('Problem statement'), 'It spans three.');
    for (const name of [/payments-api/, /reporting-api/, /ab-console-admin/]) {
      await user.click(screen.getByRole('button', { name }));
    }
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [{ repositories: Record<string, unknown>[] }];
    expect(input.repositories).toHaveLength(3);
    // Two of them are Backend. Nothing about a saved label is exclusive, and no role is sent.
    expect(input.repositories.every((item) => !('role' in item))).toBe(true);
  });

  it('lets a repository in the selection be made optional', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-optional' });
    renderForm({
      createFeature,
      listSavedRepositories: async () => ({
        repositories: [
          savedRepository({ configuration_id: 'repo-1', name: 'server' , repository_url: 'https://github.com/owner/server' }),
          savedRepository({ configuration_id: 'repo-2', name: 'docs', repository_url: 'https://github.com/owner/docs' }),
        ],
        suggested_types: [],
      }),
    });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Title'), 'Optional docs');
    await user.type(screen.getByLabelText('Problem statement'), 'Docs are a nice-to-have.');
    await user.click(screen.getByRole('button', { name: /server/ }));
    await user.click(screen.getByRole('button', { name: /docs/ }));
    // The control only exists once there is more than one repository: with a single one,
    // optional would leave the feature with no completion gate at all.
    const [, second] = screen.getAllByRole('checkbox', { name: 'Required' });
    await user.click(second!);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [{ repositories: { required: boolean }[] }];
    expect(input.repositories.map((item) => item.required)).toEqual([true, false]);
  });

  it('accepts a one-time repository without saving it', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-one-off' });
    const saveRepository = vi.fn();
    renderForm({ createFeature, saveRepository });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Title'), 'A spike');
    await user.type(screen.getByLabelText('Problem statement'), 'Try it somewhere else.');
    await user.click(screen.getByRole('button', { name: /Add one-time repository/ }));
    await user.type(screen.getByLabelText('URL'), 'https://github.com/owner/spike');
    await user.click(screen.getByRole('button', { name: 'Add to this feature' }));
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [{ repositories: Record<string, unknown>[] }];
    expect(input.repositories).toEqual([
      {
        repository_url: 'https://github.com/owner/spike',
        default_branch: 'master',
        required: true,
      },
    ]);
    // Used, and not kept. Saving it is a separate act, in Settings.
    expect(saveRepository).not.toHaveBeenCalled();
  });

  it('rejects a one-time repository URL that carries credentials', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Title'), 'Bad URL');
    await user.type(screen.getByLabelText('Problem statement'), 'It has a token in it.');
    await user.click(screen.getByRole('button', { name: /Add one-time repository/ }));
    await user.type(
      screen.getByLabelText('URL'),
      'https://token@github.com/example/private.git?ref=main',
    );
    await user.click(screen.getByRole('button', { name: 'Add to this feature' }));
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(
      await screen.findByText(
        'Do not put credentials, query parameters, or fragments in a repository URL',
      ),
    ).toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('rejects the same repository twice', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    // Once from the saved list, once typed by hand: the same repository either way.
    await user.click(screen.getByRole('button', { name: /Add one-time repository/ }));
    await user.type(
      screen.getByLabelText('URL'),
      'https://github.com/Appbroda/admanager_console-2.0.git',
    );
    await user.click(screen.getByRole('button', { name: 'Add to this feature' }));
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(await screen.findByText('This repository is already listed')).toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('keeps the derived detail out of the way until somebody asks for it', async () => {
    renderForm({ createFeature: vi.fn() });
    const user = userEvent.setup();

    await screen.findByLabelText('Title');
    expect(screen.getByRole('button', { name: 'Add story', hidden: true })).not.toBeVisible();

    await user.click(screen.getByText('Advanced details'));
    expect(screen.getByRole('button', { name: 'Add story' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Add requirement' })).toBeVisible();
  });

  it('shows the execution mode on the form, and says what the chosen one does', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-mock' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    // Visible without opening anything: the default writes to real repositories, and a
    // default that consequential must not be two clicks away.
    const mode = await screen.findByLabelText('Mode');
    expect(mode).toBeVisible();
    expect(mode).toHaveValue('live');
    expect(screen.getByText(/opens a draft pull request/)).toBeInTheDocument();

    await user.selectOptions(mode, 'mock');
    expect(screen.getByText(/without reaching a provider or touching a repository/)).toBeInTheDocument();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [{ execution_mode: string }];
    expect(input.execution_mode).toBe('mock');
  });

  it('sends the structured detail somebody chose to write', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-detailed' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByText('Advanced details'));
    await user.click(screen.getByRole('button', { name: 'Add requirement' }));
    await user.type(screen.getByLabelText('Description'), 'Extend the status endpoint');
    await user.type(screen.getByLabelText('Acceptance criterion 1'), 'Status accepts INACTIVE');
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [{ prd: Record<string, unknown> }];
    const requirements = input.prd.requirements as Record<string, unknown>[];
    expect(requirements).toHaveLength(1);
    expect(requirements[0]!.description).toBe('Extend the status endpoint');
    expect(requirements[0]!.acceptance_criteria).toEqual(['Status accepts INACTIVE']);
    expect(requirements[0]!.dependencies).toEqual([]);
  });

  it('opens the advanced section when something inside it failed validation', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByText('Advanced details'));
    // A story that was started and left incomplete. Submitting must say so rather than
    // failing silently behind a section that closed again.
    await user.click(screen.getByRole('button', { name: 'Add story' }));
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(await screen.findAllByText('Required')).not.toHaveLength(0);
    expect(screen.getByLabelText('Persona')).toBeVisible();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('reports which field is wrong instead of letting the server answer with a 422', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Create feature' }));

    expect(await screen.findAllByText('Required')).not.toHaveLength(0);
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('explains a rejected submission without losing what was typed', async () => {
    const createFeature = vi
      .fn()
      .mockRejectedValue(new ApiError('conflict', 'exists', 409, 'feature already exists'));
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The platform refused this action in the feature’s current state.',
    );
    expect(screen.getByLabelText('Title')).toHaveValue('Deactivate ad units');
  });

  it('reuses the idempotency key when an identical timed-out submission is retried', async () => {
    const createFeature = vi.fn().mockRejectedValue(new ApiError('timeout', 'timed out'));
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));
    await waitFor(() => expect(createFeature).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: 'Create feature' }));
    await waitFor(() => expect(createFeature).toHaveBeenCalledTimes(2));

    const first = createFeature.mock.calls[0]![1] as { idempotencyKey: string };
    const second = createFeature.mock.calls[1]![1] as { idempotencyKey: string };
    expect(second.idempotencyKey).toBe(first.idempotencyKey);
  });

  it('loads an uploaded PRD into the problem statement, as written', async () => {
    renderForm({ createFeature: vi.fn() });
    const user = userEvent.setup();
    const file = new File(['# Outage history\n\nOperators cannot see it.'], 'prd.md', {
      type: 'text/markdown',
    });

    await user.upload(await screen.findByLabelText('Or upload a PRD'), file);

    // Reading the file is asynchronous, so the element exists before its value arrives.
    await waitFor(() =>
      expect(screen.getByLabelText('Problem statement')).toHaveValue(
        '# Outage history\n\nOperators cannot see it.',
      ),
    );
    // Nothing is parsed out of the document; saying which one was read is the whole feedback.
    expect(screen.getByText('Loaded prd.md')).toBeInTheDocument();
  });

  it('starts a feature from a title and an uploaded PRD alone', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-uploaded' });
    renderForm({ createFeature });
    const user = userEvent.setup();
    const file = new File(['Operators cannot deactivate an ad unit.'], 'prd.md', {
      type: 'text/markdown',
    });

    await user.type(await screen.findByLabelText('Title'), 'Deactivate ad units');
    await user.upload(screen.getByLabelText('Or upload a PRD'), file);
    await waitFor(() => expect(screen.getByLabelText('Problem statement')).not.toHaveValue(''));
    await user.click(screen.getByRole('button', { name: /admanager_console-2\.0/ }));
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [{ prd: Record<string, unknown> }];
    expect(input.prd.problem_statement).toBe('Operators cannot deactivate an ad unit.');
  });
});

/**
 * Choosing a custom model setup instead of a (platform, tier) pairing.
 *
 * The setups come from a captured real `/model-setups` payload parsed through the production
 * schema (T15), and the properties defended are the spec's T16: exactly one selection travels
 * — never both a setup id and a pairing — and an unusable option renders disabled with its
 * reason rather than being silently substituted.
 */
describe('custom model setups on the form', () => {
  const ownSetups = modelSetupsSchema.parse(capturedSetups);

  it('offers the caller’s setups and pins the submission to the setup alone', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-pinned' });
    renderForm({ createFeature, listModelSetups: async () => ownSetups });
    const user = userEvent.setup();

    const platform = await screen.findByLabelText('Platform and performance tier');
    const setup = ownSetups.setups[0]!;
    const group = within(platform as HTMLSelectElement).getByRole('group', {
      name: 'Your setups',
    });
    expect(within(group).getByRole('option', { name: setup.name })).toBeEnabled();

    await user.selectOptions(platform, `custom:${setup.setup_id}`);
    // The honest scope note, stated where the choice is made — a setup is neither cheaper
    // nor better by construction.
    expect(screen.getByText(/neither cheaper nor better by construction/)).toBeInTheDocument();

    // The stage table shows each role's pinned model, the effort its author pinned — and,
    // uniquely to a mixed setup, the provider each role is pinned to. Rows come in execution
    // order, not payload order: the captured payload lists coding first, the table still
    // leads with reasoning.
    const stages = screen.getByRole('table', {
      name: 'Stages, the models they run on and the effort each is sent at',
    });
    const reasoningRow = within(stages).getByText('Reasoning').closest('tr')!;
    expect(within(stages).getAllByRole('row')[1]).toBe(reasoningRow);
    expect(within(reasoningRow).getByTitle('claude-opus-5')).toHaveTextContent('Claude Opus 5');
    expect(within(reasoningRow).getByText('Anthropic')).toBeInTheDocument();
    expect(within(reasoningRow).getByText('max')).toBeInTheDocument();
    const codingRow = within(stages).getByText('Coding').closest('tr')!;
    expect(within(codingRow).getByTitle('gpt-5.6-sol')).toHaveTextContent('GPT-5.6 Sol');
    expect(within(codingRow).getByText('OpenAI')).toBeInTheDocument();
    // A role its author left blank runs at the provider's default, and the table says that
    // rather than leaving the cell empty — the captured setup pins Haiku, which takes no
    // effort at all.
    const fixRow = within(stages).getByText('Scoped fix').closest('tr')!;
    expect(within(fixRow).getByText('Provider default')).toBeInTheDocument();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [
      { model_setup_id?: string; agent_platform?: string; performance_tier?: string },
    ];
    // Exactly one selection: the server refuses two answers to one question, and the client
    // never asks it.
    expect(input.model_setup_id).toBe(setup.setup_id);
    expect(input.agent_platform).toBeUndefined();
    expect(input.performance_tier).toBeUndefined();
  });

  it('renders an unusable setup disabled, with its reason, and substitutes nothing', async () => {
    window.localStorage.removeItem('newFeature.lastAgentPlatform');
    const unusable = {
      ...ownSetups.setups[0]!,
      setup_id: 'setup-broken',
      name: 'Broken setup',
      usable: false,
      missing_credentials: ['OpenAI'],
    };
    renderForm({
      createFeature: vi.fn(),
      listModelSetups: async () => ({ setups: [unusable] }),
    });

    const platform = await screen.findByLabelText('Platform and performance tier');
    const option = within(platform as HTMLSelectElement).getByRole('option', {
      name: /Broken setup — missing OpenAI credential/,
    });
    expect(option).toBeDisabled();
    // The selection itself is untouched: the form never re-aims a choice.
    expect(platform).toHaveValue('openai:medium');
  });
});

/**
 * The design half of the request.
 *
 * A person asking for a screen has two documents — what it must do and what it must look
 * like — and this form only ever accepted the first. These check the second one travels, and
 * that a submission which attaches nothing sends exactly what it always sent.
 */
describe('attaching a design to a feature request', () => {
  it('sends the pasted link and its label, and derives nothing on the server’s behalf', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-designed' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Attach a design' }));
    await user.type(
      screen.getByLabelText('Figma link'),
      'https://www.figma.com/design/28gd2JrZO28FCN9PCKM4qK/OpenCRM?node-id=2-303',
    );
    await user.type(screen.getByLabelText('What this is'), 'Empty state');
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(await screen.findByText('Workspace for feature')).toBeInTheDocument();
    const [input] = createFeature.mock.calls[0] as [Record<string, unknown>];
    const prd = input.prd as Record<string, unknown>;
    // Only the URL and the label. The file key and the frame ids are the server's derivation
    // from the same URL — sending this client's copy would make its guess authoritative the
    // moment the two disagreed, which is `derivedRepositoryName`'s rule.
    expect(prd.design_references).toEqual([
      {
        url: 'https://www.figma.com/design/28gd2JrZO28FCN9PCKM4qK/OpenCRM?node-id=2-303',
        label: 'Empty state',
      },
    ]);
  });

  it('omits the key entirely when nothing was attached', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-plain' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [Record<string, unknown>];
    const prd = input.prd as Record<string, unknown>;
    // Absent, not `[]`. This function used to emit every PRD key unconditionally, and with
    // the key omitted the request a citation-free submission sends is byte-identical to the
    // one this form has always sent — which is what makes "a feature that cites no design
    // behaves exactly as it does today" a fact about the wire rather than a hope.
    expect('design_references' in prd).toBe(false);
    // And the other optional sections are unchanged: they still travel as empty arrays,
    // because the server has always received them that way.
    expect(prd.goals).toEqual([]);
    expect(prd.stakeholders).toEqual([]);
  });

  it('refuses a link that is not a Figma design file, before the server has to', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Attach a design' }));
    // A prototype link: same file key, different kind of document, and the extraction reads
    // design nodes.
    await user.type(
      screen.getByLabelText('Figma link'),
      'https://www.figma.com/proto/28gd2JrZO28FCN9PCKM4qK/OpenCRM?node-id=2-303',
    );
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(await screen.findByText(/Paste a Figma design link/)).toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('refuses a branch link by name, because it would cite the parent file', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Attach a design' }));
    // `/design/<parentKey>/branch/<branchKey>/<Name>`: segment 1 is the *parent* file's key,
    // so accepting this would silently snapshot a design the person was not looking at.
    await user.type(
      screen.getByLabelText('Figma link'),
      'https://www.figma.com/design/28gd2JrZO28FCN9PCKM4qK/branch/9XyZAbC123456/OpenCRM?node-id=2-303',
    );
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    // The branch-specific sentence, not the generic "paste a design link" advice — the person
    // did paste one, of a branch. It mirrors the server's refusal word for word.
    expect(await screen.findByText(/branches are not supported yet/)).toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('refuses the same frame cited twice, however the two links were spelled', async () => {
    const createFeature = vi.fn();
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Attach a design' }));
    await user.click(screen.getByRole('button', { name: 'Add another design' }));
    const links = screen.getAllByLabelText('Figma link');
    await user.type(
      links[0]!,
      'https://www.figma.com/design/28gd2JrZO28FCN9PCKM4qK/OpenCRM?node-id=2-303',
    );
    // The API's own colon spelling of the same frame, on the pre-rename URL form.
    await user.type(
      links[1]!,
      'https://www.figma.com/file/28gd2JrZO28FCN9PCKM4qK/OpenCRM?node-id=2:303',
    );
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    expect(await screen.findByText('This design is already cited')).toBeInTheDocument();
    expect(createFeature).not.toHaveBeenCalled();
  });

  it('accepts a link with no frame, which is a whole-file citation', async () => {
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-whole-file' });
    renderForm({ createFeature });
    const user = userEvent.setup();

    await fillMinimum(user);
    await user.click(screen.getByRole('button', { name: 'Attach a design' }));
    await user.type(
      screen.getByLabelText('Figma link'),
      'https://www.figma.com/design/28gd2JrZO28FCN9PCKM4qK/OpenCRM',
    );
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await screen.findByText('Workspace for feature');
    const [input] = createFeature.mock.calls[0] as [Record<string, unknown>];
    const prd = input.prd as Record<string, unknown>;
    expect(prd.design_references).toEqual([
      { url: 'https://www.figma.com/design/28gd2JrZO28FCN9PCKM4qK/OpenCRM', label: '' },
    ]);
  });
});
