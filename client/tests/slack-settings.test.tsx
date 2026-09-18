import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiContext } from '@/app/api-context';
import { SessionProvider } from '@/app/session';
import { SettingsView } from '@/features/settings/SettingsView';
import { ApiError } from '@/api/errors';
import type { SlackConfiguration } from '@/schemas/feature';
import { stubApi } from './fixtures';

/**
 * The Slack half of Settings: where feature threads go, and how this person is mentioned.
 *
 * Two properties are defended here. The panel never holds the bot token — it points at the
 * credentials panel and shows only a hint — and every verdict sentence on it is the server's,
 * rendered verbatim: the connection test, a save refusal, and the degraded banner's reason
 * all originate on the other side, where they are owned and tested.
 */

const READY = {
  status: 'ok',
  build_revision: 'abcdef0123456789',
  workflow_schema_version: '1.0',
  runtime_compatible: true,
};

const BASE = {
  getReadiness: async () => READY,
  listCredentials: async () => ({
    credentials: [{ provider: 'slack', configured: true, hint: 'x9k2' }],
  }),
};

/** An operator who may save the configuration; the server publishes the permission. */
const OPERATOR = {
  getMe: async () => ({
    actor_id: 'user-1',
    display_name: 'Alex',
    authentication: 'user_token',
    roles: ['operator'],
    permissions: ['slack_configuration:manage'],
  }),
};

function configured(overrides: Partial<SlackConfiguration> = {}): SlackConfiguration {
  return {
    configured: true,
    enabled: true,
    workspace_id: 'T0AB12CD3',
    workspace_name: 'AppBroda',
    channel_id: 'C0AB12CD34E',
    channel_name: 'feature-updates',
    token_owner_id: 'platform-admin',
    verbosity: 'milestones',
    status: 'active',
    status_reason: null,
    console_base_url: 'https://console.example.com',
    credential_configured: true,
    credential_hint: 'x9k2',
    updated_by: 'platform-admin',
    updated_at: '2026-09-01T10:00:00Z',
    ...overrides,
  };
}

function renderSettings(api: Record<string, unknown>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ApiContext.Provider value={stubApi(api)}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <SessionProvider>
          <SettingsView />
          </SessionProvider>
        </MemoryRouter>
      </ApiContext.Provider>
    </QueryClientProvider>,
  );
}

describe('the Slack notifications panel', () => {
  it('renders the configuration and saves the typed input', async () => {
    const getSlackConfiguration = vi.fn().mockResolvedValue(configured());
    const saveSlackConfiguration = vi.fn().mockResolvedValue(configured());
    renderSettings({ ...BASE, ...OPERATOR, getSlackConfiguration, saveSlackConfiguration });
    const user = userEvent.setup();

    // The stored values populate the form, and the token is present only as its hint.
    const channel = await screen.findByLabelText('Channel ID');
    expect(channel).toHaveValue('C0AB12CD34E');
    expect(screen.getByText(/Bot token: configured/)).toBeInTheDocument();
    expect(screen.getByText('platform-admin')).toBeInTheDocument();

    await user.clear(channel);
    await user.type(channel, 'C0999NEW');
    await user.click(screen.getByRole('button', { name: 'Save Slack configuration' }));

    expect(saveSlackConfiguration).toHaveBeenCalledWith({
      enabled: true,
      channel_id: 'C0999NEW',
      channel_name: 'feature-updates',
      verbosity: 'milestones',
      console_base_url: 'https://console.example.com',
      // Kept, not re-defaulted: re-saving must not re-point delivery at the saver's own key.
      token_owner_id: 'platform-admin',
    });
    // The save invalidates the query, so what is on screen is re-read rather than assumed.
    await waitFor(() => expect(getSlackConfiguration).toHaveBeenCalledTimes(2));
  });

  it('offers no verbosity control, and states the delivery latency plainly', async () => {
    renderSettings({ ...BASE, ...OPERATOR, getSlackConfiguration: async () => configured() });

    expect(
      await screen.findByText(/an update can arrive up to 30 seconds after the event/),
    ).toBeInTheDocument();
    // Milestones-only in v1: a toggle offering more would be a control that lies.
    expect(screen.queryByLabelText(/verbosity/i)).not.toBeInTheDocument();
  });

  it('shows the degraded banner with the server’s own reason', async () => {
    const reason = 'the bot token was rejected — re-save it';
    renderSettings({
      ...BASE,
      ...OPERATOR,
      getSlackConfiguration: async () =>
        configured({ status: 'degraded', status_reason: reason }),
    });

    const banner = await screen.findByRole('alert');
    expect(banner).toHaveTextContent('Slack delivery is paused');
    expect(banner).toHaveTextContent(reason);
  });

  it('renders the connection test’s verdict sentence verbatim', async () => {
    const detail = 'Slack accepted the token just now, and the configured channel is reachable.';
    const checkSlackConfiguration = vi.fn().mockResolvedValue({
      provider: 'slack',
      configured: true,
      usable: true,
      verified: 'accepted',
      detail,
    });
    renderSettings({
      ...BASE,
      ...OPERATOR,
      getSlackConfiguration: async () => configured(),
      checkSlackConfiguration,
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Test connection' }));

    expect(checkSlackConfiguration).toHaveBeenCalled();
    expect(await screen.findByText('Provider accepted it')).toBeInTheDocument();
    expect(screen.getByText(detail)).toBeInTheDocument();
  });

  it('says plainly when the deployment does not deliver Slack notifications', async () => {
    renderSettings({
      ...BASE,
      ...OPERATOR,
      getSlackConfiguration: async () => {
        throw new ApiError('server', 'unavailable', 503, 'no directory');
      },
      getSlackLink: async () => {
        throw new ApiError('server', 'unavailable', 503, 'no directory');
      },
    });

    expect(
      await screen.findByText('This deployment does not deliver Slack notifications.'),
    ).toBeInTheDocument();
    // The profile half says nothing at all: there is nothing to link to.
    expect(screen.queryByLabelText('Slack member ID')).not.toBeInTheDocument();
  });

  it('is not shown at all to somebody without the permission', async () => {
    // `slack_configuration:manage` moved to administrators when workspaces became isolated:
    // there is one enabled Slack configuration for the whole deployment, so an operator
    // holding this was an ordinary user re-pointing everybody's feature threads at their own
    // channel. The panel used to render read-only for them; now it does not render, because
    // where somebody else's threads go is not their business to read either.
    //
    // The default stub identity has no permissions. Hiding is a courtesy: `PUT
    // /slack-configuration` refuses them whether or not this page draws a form.
    renderSettings({ ...BASE, getSlackConfiguration: async () => configured() });

    // Something from the page rendered, so this is not asserting against a blank screen.
    expect(await screen.findByText('Settings')).toBeInTheDocument();
    expect(screen.queryByText('feature-updates')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Save Slack configuration' }),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Test connection' })).not.toBeInTheDocument();
  });
});

describe('the profile’s Slack link', () => {
  it('saves the member ID and chosen scope', async () => {
    const saveSlackLink = vi.fn().mockResolvedValue({
      user_id: 'user-1',
      slack_user_id: 'U0123ABCDEF',
      notify_scope: 'human_interaction',
    });
    renderSettings({ ...BASE, ...OPERATOR, saveSlackLink });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Slack member ID'), 'U0123ABCDEF');
    // The three scopes are offered in the person's words, not the vocabulary's.
    const scopes = screen.getByLabelText('Slack mentions');
    await user.selectOptions(
      scopes,
      screen.getByRole('option', { name: 'Mention me when a feature needs a person' }),
    );
    await user.click(screen.getByRole('button', { name: 'Save Slack link' }));

    expect(saveSlackLink).toHaveBeenCalledWith({
      slack_user_id: 'U0123ABCDEF',
      notify_scope: 'human_interaction',
    });
  });

  it('renders the server’s refusal of a malformed member ID verbatim', async () => {
    const refusal =
      'That does not look like a Slack member ID (it looks like U0123ABCDEF, from your Slack ' +
      'profile). A wrong id would silently mention nobody.';
    renderSettings({
      ...BASE,
      ...OPERATOR,
      saveSlackLink: async () => {
        throw new ApiError('validation', 'The request was not valid.', 422, refusal);
      },
    });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Slack member ID'), 'not-a-member-id');
    await user.click(screen.getByRole('button', { name: 'Save Slack link' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(refusal);
  });
});
