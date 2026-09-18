import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiContext } from '@/app/api-context';
import { ApiError } from '@/api/errors';
import { ModelConfigurationPanel } from '@/features/settings/ModelConfigurationPanel';
import type {
  ModelConfiguration,
  ModelRoleConfiguration,
  ModelSetup,
  ModelSetups,
} from '@/schemas/feature';
import { modelConfigurationSchema, modelSetupsSchema } from '@/schemas/feature';
import { stubApi } from './fixtures';
import capturedConfiguration from './fixtures/model-configuration.custom-setup.json';
import capturedSetups from './fixtures/model-setups.mixed.json';

/**
 * The resolved model-roles table, on a page instead of in a startup log.
 *
 * Two properties are defended here. The first is that the panel reports the server's answer
 * whole — every configured pairing, every role, and the values a person came to compare
 * against the deployment's environment. The second is that it never pretends to be a control:
 * the page says the defaults are not editable, and there is nothing on it that could change
 * one. A read-only view that grows an input is how somebody comes to believe they changed a
 * model they did not change.
 */

function role(overrides: Partial<ModelRoleConfiguration> = {}): ModelRoleConfiguration {
  return {
    role: 'coding',
    platform: null,
    model: 'claude-opus-5',
    reasoning_effort: 'xhigh',
    requested_reasoning_effort: null,
    max_tokens: 128000,
    routing_reason: 'Feature implementation and substantive fixes use the coding role.',
    model_variable: 'ANTHROPIC_CODING_MODEL',
    reasoning_variable: 'ANTHROPIC_CODING_REASONING_EFFORT',
    resolved_from_legacy_variable: false,
    ...overrides,
  };
}

function setup(overrides: Partial<ModelSetup> = {}): ModelSetup {
  return {
    platform: 'anthropic',
    platform_label: 'Claude',
    performance_tier: 'high',
    tier_label: 'Max',
    label: 'Claude — Max',
    roles: [
      role({ role: 'reasoning', model: 'claude-opus-5', reasoning_effort: 'max' }),
      role(),
      role({ role: 'review', reasoning_effort: 'high', max_tokens: 48000 }),
      role({ role: 'scoped_fix', model: 'claude-sonnet-5', reasoning_effort: 'high' }),
    ],
    origin: 'deployment',
    setup_id: null,
    editable: false,
    ...overrides,
  };
}

function renderPanel(configuration: ModelConfiguration, ownSetups?: ModelSetups) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ApiContext.Provider
        value={stubApi({
          getModelConfiguration: async () => configuration,
          listModelSetups: async () => ownSetups ?? { setups: [] },
        })}
      >
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <ModelConfigurationPanel />
        </MemoryRouter>
      </ApiContext.Provider>
    </QueryClientProvider>,
  );
}

describe('the resolved model table', () => {
  it('names the model, effort and bound each role runs on', async () => {
    renderPanel({ editable: false, setups: [setup()] });

    const table = await screen.findByRole('table', { name: /Claude — Max/ });
    const rows = within(table).getAllByRole('row');
    // Header plus the four roles, in the order the server published them.
    expect(rows).toHaveLength(5);
    const coding = within(table).getByRole('row', { name: /Coding/ });
    // Written the way every other surface writes a model, with the identifier itself kept on
    // the element for anybody comparing this against their environment.
    expect(within(coding).getByTitle('claude-opus-5')).toHaveTextContent('Claude Opus 5');
    expect(coding.textContent).toContain('xhigh');
    expect(coding.textContent).toContain('128,000');
    expect(within(table).getByRole('row', { name: /Scoped fix/ })).toBeInTheDocument();
  });

  it('says the defaults are not editable, and offers nothing that would edit one', async () => {
    renderPanel({ editable: false, setups: [setup()] });

    expect(await screen.findByText(/not editable here/)).toBeInTheDocument();
    // The strong form of the claim: no control on this panel can carry a value anywhere.
    expect(screen.queryAllByRole('textbox')).toHaveLength(0);
    expect(screen.queryAllByRole('combobox')).toHaveLength(0);
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
    expect(screen.queryAllByRole('button', { name: /save|apply|change|edit/i })).toHaveLength(0);
  });

  it('shows one tier at a time and switches between the configured ones', async () => {
    renderPanel({
      editable: false,
      setups: [
        setup({
          performance_tier: 'low',
          tier_label: 'Economy',
          label: 'Claude — Economy',
          roles: [
            role({ role: 'reasoning', model: 'claude-sonnet-5', reasoning_effort: 'medium' }),
            role({ role: 'coding', model: 'claude-sonnet-5', reasoning_effort: 'medium' }),
            role({ role: 'review', model: 'claude-sonnet-5', reasoning_effort: 'medium' }),
            role({ role: 'scoped_fix', model: 'claude-haiku-4-5', reasoning_effort: 'medium' }),
          ],
        }),
        setup(),
      ],
    });
    const user = userEvent.setup();

    // The first published pairing is what is shown before anybody chooses.
    const economy = await screen.findByRole('table', { name: /Claude — Economy/ });
    // Written by the shared model-name rule, which spells versioned segments as words --
    // asserted as it actually renders rather than as the raw identifier.
    expect(within(economy).getByRole('row', { name: /Scoped fix/ }).textContent).toContain(
      'Claude Haiku 4 5',
    );

    await user.click(screen.getByRole('tab', { name: 'Claude — Max' }));

    const max = await screen.findByRole('table', { name: /Claude — Max/ });
    expect(within(max).getByRole('row', { name: /Coding/ }).textContent).toContain('128,000');
    // One table, not two: a tier is a whole preset and reading two at once is how the pinning
    // rules get misread.
    expect(screen.queryByRole('table', { name: /Claude — Economy/ })).not.toBeInTheDocument();
  });

  it('opens a role to say why it runs and which variable set it', async () => {
    renderPanel({ editable: false, setups: [setup()] });
    const user = userEvent.setup();

    const table = await screen.findByRole('table', { name: /Claude — Max/ });
    const coding = within(table).getByRole('row', { name: /Coding/ });
    await user.click(
      within(coding).getByRole('button', { name: /Why this role runs/ }),
    );

    expect(
      await screen.findByText(/Feature implementation and substantive fixes/),
    ).toBeInTheDocument();
    // The panel cannot edit the value, so naming where it comes from is the difference
    // between a report and a dead end.
    expect(screen.getByText('ANTHROPIC_CODING_MODEL')).toBeInTheDocument();
    expect(screen.getByText('ANTHROPIC_CODING_REASONING_EFFORT')).toBeInTheDocument();
  });

  it('shows both efforts when the configured one was dropped as unsupported', async () => {
    // The normalization behind feature 181: an effort configured past what the model accepts.
    // Showing only the effective value would present the provider's default as a choice.
    renderPanel({
      editable: false,
      setups: [
        setup({
          roles: [
            role({
              role: 'coding',
              reasoning_effort: null,
              requested_reasoning_effort: 'xhigh',
            }),
          ],
        }),
      ],
    });

    const coding = await screen.findByRole('row', { name: /Coding/ });
    expect(coding.textContent).toContain('provider default');
    expect(coding.textContent).toContain('xhigh');
    expect(coding.textContent).toMatch(/does not accept it/);
  });

  it('marks a role that has no variable of its own', async () => {
    renderPanel({
      editable: false,
      setups: [
        setup({
          roles: [
            role({
              role: 'review',
              model_variable: 'OPENAI_REASONING_MODEL',
              reasoning_variable: null,
              resolved_from_legacy_variable: true,
            }),
          ],
        }),
      ],
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: /Why this role runs/ }));

    expect(await screen.findByText('Older variable')).toBeInTheDocument();
    expect(screen.getByText(/no variable of its own/)).toBeInTheDocument();
  });

  it('says so plainly when the deployment resolved no models at all', async () => {
    // An empty table would read as "no models are needed". A deployment in this state cannot
    // run a feature, and the panel is the place that can say why.
    renderPanel({ editable: false, setups: [] });

    expect(await screen.findByText('No model configuration resolved')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });
});

/**
 * The custom half, tested against captured real payloads rather than the factories above.
 *
 * The payloads in `fixtures/model-configuration.custom-setup.json` and
 * `fixtures/model-setups.mixed.json` are genuine responses from the real application —
 * refreshed by `server/tests/capture_model_setup_fixtures.py` — parsed through the production
 * schemas. Hand-written fixtures dropped a dozen fields from this client before, and here a
 * missing `origin` would silently render a custom row as a deployment row.
 */
describe('the caller’s own setups in the resolved table', () => {
  const configuration = modelConfigurationSchema.parse(capturedConfiguration);
  const ownSetups = modelSetupsSchema.parse(capturedSetups);

  it('parses the captured payloads through the production schemas', () => {
    // The capture carries both kinds of row, which is exactly what the panel must tell apart.
    expect(configuration.editable).toBe(true);
    expect(configuration.setups.map((item) => item.origin)).toContain('custom');
    expect(configuration.setups.map((item) => item.origin)).toContain('deployment');
    expect(ownSetups.setups.length).toBeGreaterThan(0);
    // A mixed setup names each role's own platform on the custom row.
    const custom = configuration.setups.find((item) => item.origin === 'custom')!;
    const platforms = new Set(custom.roles.map((item) => item.platform));
    expect(platforms.has('openai')).toBe(true);
    expect(platforms.has('anthropic')).toBe(true);
  });

  it('offers authoring and marks the custom row as the caller’s own', async () => {
    renderPanel(configuration, ownSetups);
    const user = userEvent.setup();

    // The deployment rows keep their read-only sentence; the affordance is additive.
    expect(await screen.findByText(/You can author your own setup/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'New setup' })).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: /Mixed pilot setup — yours/ }));

    expect(await screen.findByText(/Your setup — authored here/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
    const table = screen.getByRole('table', { name: /Mixed pilot setup/ });
    const coding = within(table).getByRole('row', { name: /Coding/ });
    // The mixed setup's coding role runs on the other provider, and the row says so.
    expect(coding.textContent).toContain('openai');
  });

  it('offers no environment variable to copy on a custom row', async () => {
    // T17: a custom row has no variable behind it, and a copyable name would send somebody
    // grepping their deployment for a string that is not there.
    renderPanel(configuration, ownSetups);
    const user = userEvent.setup();

    await user.click(await screen.findByRole('tab', { name: /Mixed pilot setup — yours/ }));
    const table = await screen.findByRole('table', { name: /Mixed pilot setup/ });
    const coding = within(table).getByRole('row', { name: /Coding/ });
    await user.click(within(coding).getByRole('button', { name: /Why this role runs/ }));

    expect(await screen.findByText('authored in this setup')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Copy variable name' })).not.toBeInTheDocument();
  });

  it('opens the authoring form with four role rows and the server’s vocabulary', async () => {
    renderPanel(configuration, ownSetups);
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'New setup' }));

    expect(await screen.findByRole('dialog', { name: 'New model setup' })).toBeInTheDocument();
    const dialog = screen.getByRole('dialog', { name: 'New model setup' });
    // Four role groups, each with a platform, a free-text model, and an effort control.
    for (const legend of ['Reasoning', 'Coding', 'Review', 'Scoped fix']) {
      expect(within(dialog).getByRole('group', { name: legend })).toBeInTheDocument();
    }
    // The model is a free-text identifier with suggestions, never a closed list: a new model
    // must be usable the day the provider ships it.
    const reasoning = within(dialog).getByRole('group', { name: 'Reasoning' });
    // An input bound to a datalist reads as a combobox: typing stays free, suggestions ride
    // along. A closed `<select>` of models is exactly what this must not be.
    expect(within(reasoning).getByLabelText('Model identifier')).toHaveAttribute('list');
    // "provider default" is a distinct, chosen option in the effort vocabulary.
    expect(
      within(within(dialog).getByRole('group', { name: 'Coding' }).closest('form')!).getAllByRole(
        'option',
        { name: 'provider default' },
      ).length,
    ).toBeGreaterThan(0);
  });

  it('renders a save refusal verbatim, as the server sentenced it', async () => {
    // The 181 refusal is the whole point of the screen; a paraphrase would soften it.
    const refusal =
      "roles.coding: reasoning effort 'xhigh' on claude-sonnet-5 needs at least 48000 output " +
      "tokens and the model's declared ceiling is 32000; choose a lower effort or a " +
      'different model for this role';
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={queryClient}>
        <ApiContext.Provider
          value={stubApi({
            getModelConfiguration: async () => configuration,
            listModelSetups: async () => ownSetups,
            createModelSetup: async () => {
              throw new ApiError('validation', 'The request was not valid.', 422, refusal);
            },
          })}
        >
          <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
            <ModelConfigurationPanel />
          </MemoryRouter>
        </ApiContext.Provider>
      </QueryClientProvider>,
    );
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'New setup' }));
    const dialog = await screen.findByRole('dialog', { name: 'New model setup' });
    await user.click(within(dialog).getByRole('button', { name: 'Save setup' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(refusal);
  });
});
