import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import type { SaveSlackConfigurationInput, SaveSlackLinkInput } from '@/api/features';
import { Async } from '@/components/common/States';
import { Badge } from '@/components/ui/Badge';
import { Panel } from '@/components/ui/Layout';
import { DetailList, DetailRow } from '@/components/ui/Value';
import { Field } from '@/features/new-feature/fields';
import { CredentialCheckResult } from './CredentialCheckResult';
import type { SlackConfiguration, SlackLink } from '@/schemas/feature';

/**
 * Where feature threads go: one Slack channel, one thread per feature.
 *
 * The bot token deliberately has no field here — it is a `slack` row in the Provider
 * credentials panel above, one credential UI and one storage path — so this panel shows only
 * whether one is stored, by hint. Verbosity has no control either: milestones is the only
 * feed the platform delivers today, and a toggle offering more would be a control that lies.
 */
export function SlackNotificationsPanel() {
  return (
    <Panel title="Slack notifications" meta="One thread per feature, in one channel">
      <SlackConfigurationSettings />
    </Panel>
  );
}

function SlackConfigurationSettings() {
  const api = useApi();
  const me = useQuery({ queryKey: ['me'], queryFn: ({ signal }) => api.getMe(signal) });
  const configuration = useQuery({
    queryKey: ['slack-configuration'],
    queryFn: ({ signal }) => api.getSlackConfiguration(signal),
    // A deployment without the directory answers 503. That is a fact about the deployment,
    // not a failure, so it is not retried into an error box.
    retry: false,
  });

  if (configuration.isError) {
    const error = configuration.error;
    if (error instanceof ApiError && error.status === 503) {
      return <p className="muted">This deployment does not deliver Slack notifications.</p>;
    }
    return <p className="field__error">{userMessage(error as ApiError)}</p>;
  }

  // Hiding is a courtesy; the server checks the permission again on the write.
  const mayManage = me.data?.permissions.includes('slack_configuration:manage') ?? false;

  return (
    <Async query={configuration}>
      {(data) => <SlackConfigurationView configuration={data} mayManage={mayManage} />}
    </Async>
  );
}

function SlackConfigurationView({
  configuration,
  mayManage,
}: {
  configuration: SlackConfiguration;
  mayManage: boolean;
}) {
  return (
    <div className="stack">
      {configuration.status === 'degraded' ? (
        <div className="callout callout--attention" role="alert">
          <p className="callout__title">Slack delivery is paused</p>
          {/* The platform's own words about why it degraded, never a raw provider message. */}
          {configuration.status_reason ? <p>{configuration.status_reason}</p> : null}
        </div>
      ) : null}
      <p className="muted">
        Messages are delivered by a background sweep — an update can arrive up to 30 seconds
        after the event. Milestones only: started, waiting on a person, finished.
      </p>
      {configuration.credential_configured ? (
        <p className="muted">
          Bot token: configured (…{configuration.credential_hint}). It is entered in the
          Provider credentials panel above, where it appears as the “Slack” row.
        </p>
      ) : (
        <p className="muted">
          <Badge tone="attention">No bot token stored</Badge> Enter it in the Provider
          credentials panel above — it appears there as the “Slack” row. Nothing is delivered
          without it.
        </p>
      )}
      {configuration.workspace_name || configuration.token_owner_id ? (
        <DetailList narrow>
          {configuration.workspace_name ? (
            <DetailRow label="Workspace">{configuration.workspace_name}</DetailRow>
          ) : null}
          {configuration.token_owner_id ? (
            // Read-only: whose stored `slack` credential delivery resolves. Re-saving keeps
            // it; a fresh configuration records the saver.
            <DetailRow label="Token owner">{configuration.token_owner_id}</DetailRow>
          ) : null}
        </DetailList>
      ) : null}
      {mayManage ? (
        <SlackConfigurationForm configuration={configuration} />
      ) : configuration.configured ? (
        <>
          <DetailList narrow>
            <DetailRow label="Delivery">{configuration.enabled ? 'on' : 'off'}</DetailRow>
            <DetailRow label="Channel">
              {configuration.channel_name ?? configuration.channel_id ?? 'not set'}
            </DetailRow>
          </DetailList>
          <p className="muted">An operator saves this configuration.</p>
        </>
      ) : (
        <p className="muted">Not configured yet. An operator saves this configuration.</p>
      )}
    </div>
  );
}

function SlackConfigurationForm({ configuration }: { configuration: SlackConfiguration }) {
  const api = useApi();
  const queryClient = useQueryClient();
  const [enabled, setEnabled] = useState(configuration.enabled);
  const [channelId, setChannelId] = useState(configuration.channel_id ?? '');
  const [channelName, setChannelName] = useState(configuration.channel_name ?? '');
  const [consoleBaseUrl, setConsoleBaseUrl] = useState(configuration.console_base_url ?? '');

  const save = useMutation({
    mutationFn: (input: SaveSlackConfigurationInput) => api.saveSlackConfiguration(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['slack-configuration'] });
    },
  });
  const check = useMutation({ mutationFn: () => api.checkSlackConfiguration() });

  return (
    <form
      className="form"
      onSubmit={(event) => {
        event.preventDefault();
        if (!channelId.trim()) return;
        save.mutate({
          enabled,
          channel_id: channelId.trim(),
          channel_name: channelName.trim() || null,
          verbosity: 'milestones',
          console_base_url: consoleBaseUrl.trim() || null,
          // Kept rather than re-defaulted: a different operator re-saving must not silently
          // re-point delivery at their own stored credential.
          token_owner_id: configuration.token_owner_id ?? null,
        });
      }}
    >
      <label className="field__label">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(event) => setEnabled(event.target.checked)}
        />{' '}
        Deliver feature threads to Slack
      </label>
      <Field
        label="Channel ID"
        hint="The channel messages are keyed on — from the channel’s “About” tab, it looks like C0123456789."
      >
        {(id, describedBy) => (
          <input
            id={id}
            aria-describedby={describedBy}
            value={channelId}
            onChange={(event) => setChannelId(event.target.value)}
          />
        )}
      </Field>
      <Field label="Channel name" hint="Display only — the ID is what delivery uses, so this may go stale.">
        {(id, describedBy) => (
          <input
            id={id}
            aria-describedby={describedBy}
            value={channelName}
            onChange={(event) => setChannelName(event.target.value)}
          />
        )}
      </Field>
      <Field
        label="Console base URL"
        hint="Used for the “Open in the console” link in each thread’s root message."
      >
        {(id, describedBy) => (
          <input
            id={id}
            aria-describedby={describedBy}
            value={consoleBaseUrl}
            placeholder="https://console.example.com"
            onChange={(event) => setConsoleBaseUrl(event.target.value)}
          />
        )}
      </Field>

      {check.data ? <CredentialCheckResult result={check.data} /> : null}

      {[save.error, check.error].map((error, index) =>
        error ? (
          <p className="field__error" role="alert" key={index}>
            {error instanceof ApiError ? (error.detail ?? userMessage(error)) : 'That did not work.'}
          </p>
        ) : null,
      )}

      <div className="form__actions">
        <button
          type="submit"
          className="button button--primary"
          disabled={save.isPending || !channelId.trim()}
        >
          {save.isPending ? 'Saving…' : 'Save Slack configuration'}
        </button>
        <button
          type="button"
          className="button button--quiet"
          disabled={check.isPending}
          onClick={() => check.mutate()}
        >
          {check.isPending ? 'Testing…' : 'Test connection'}
        </button>
      </div>
    </form>
  );
}

/**
 * The per-user half, rendered inside the Profile panel: this person's own Slack member ID and
 * how often the thread mentions them. Self-service — opting into being mentioned is nobody's
 * decision but the person's, so no permission gates it.
 */
export function SlackLinkSettings() {
  const api = useApi();
  const link = useQuery({
    queryKey: ['slack-link'],
    queryFn: ({ signal }) => api.getSlackLink(signal),
    retry: false,
  });

  if (link.isError) {
    // A 503 means the deployment does not do Slack, so the profile says nothing about it.
    return link.error instanceof ApiError && link.error.status === 503 ? null : (
      <p className="field__error">{userMessage(link.error as ApiError)}</p>
    );
  }
  if (!link.data) return null;
  return <SlackLinkForm link={link.data} />;
}

function SlackLinkForm({ link }: { link: SlackLink }) {
  const api = useApi();
  const queryClient = useQueryClient();
  const [memberId, setMemberId] = useState(link.slack_user_id ?? '');
  const [scope, setScope] = useState<SaveSlackLinkInput['notify_scope']>(link.notify_scope);

  const save = useMutation({
    mutationFn: (input: SaveSlackLinkInput) => api.saveSlackLink(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['slack-link'] });
    },
  });

  return (
    <form
      className="form"
      onSubmit={(event) => {
        event.preventDefault();
        save.mutate({ slack_user_id: memberId.trim() || null, notify_scope: scope });
      }}
    >
      <Field
        label="Slack member ID"
        hint="Your Slack profile → Copy member ID (looks like U0123ABCDEF)"
      >
        {(id, describedBy) => (
          <input
            id={id}
            aria-describedby={describedBy}
            value={memberId}
            onChange={(event) => setMemberId(event.target.value)}
          />
        )}
      </Field>
      <Field label="Slack mentions">
        {(id, describedBy) => (
          <select
            id={id}
            aria-describedby={describedBy}
            value={scope}
            onChange={(event) =>
              setScope(event.target.value as SaveSlackLinkInput['notify_scope'])
            }
          >
            <option value="none">No Slack mentions</option>
            <option value="human_interaction">Mention me when a feature needs a person</option>
            <option value="all">Mention me on every milestone</option>
          </select>
        )}
      </Field>
      {save.error ? (
        <p className="field__error" role="alert">
          {save.error instanceof ApiError
            ? (save.error.detail ?? userMessage(save.error))
            : 'That did not work.'}
        </p>
      ) : null}
      <div className="form__actions">
        <button type="submit" className="button" disabled={save.isPending}>
          {save.isPending ? 'Saving…' : 'Save Slack link'}
        </button>
      </div>
    </form>
  );
}
