import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import type { SaveDesignSourceInput } from '@/api/features';
import { Async } from '@/components/common/States';
import { Badge } from '@/components/ui/Badge';
import { Panel } from '@/components/ui/Layout';
import { DetailList, DetailRow } from '@/components/ui/Value';
import { Field } from '@/features/new-feature/fields';
import { CredentialCheckResult } from './CredentialCheckResult';
import type { DesignSource } from '@/schemas/feature';

/**
 * Where this deployment's designs come from: one Figma account, and which files may be cited.
 *
 * The personal access token deliberately has no field here — it is a `figma` row in the
 * Provider credentials panel above, one credential UI and one storage path — so this panel
 * shows only whether one is stored, by hint.
 *
 * The allowlist's emptiness is stated rather than implied. Empty means *any file that token
 * can read*, which is the right default for a single-team deployment and the wrong one for a
 * shared token, so the panel says which of the two this deployment is in.
 */
export function DesignSourcePanel() {
  return (
    <Panel title="Design source" meta="Where a feature's design links are resolved from">
      <DesignSourceSettings />
    </Panel>
  );
}

function DesignSourceSettings() {
  const api = useApi();
  const me = useQuery({ queryKey: ['me'], queryFn: ({ signal }) => api.getMe(signal) });
  const configuration = useQuery({
    queryKey: ['design-source'],
    queryFn: ({ signal }) => api.getDesignSource(signal),
    // A deployment without the directory answers 503. That is a fact about the deployment,
    // not a failure, so it is not retried into an error box.
    retry: false,
  });

  if (configuration.isError) {
    const error = configuration.error;
    if (error instanceof ApiError && error.status === 503) {
      return <p className="muted">This deployment does not resolve design references.</p>;
    }
    return <p className="field__error">{userMessage(error as ApiError)}</p>;
  }

  // Hiding is a courtesy; the server checks the permission again on the write.
  const mayManage = me.data?.permissions.includes('design_source:manage') ?? false;

  return (
    <Async query={configuration}>
      {(data) => <DesignSourceView configuration={data} mayManage={mayManage} />}
    </Async>
  );
}

function DesignSourceView({
  configuration,
  mayManage,
}: {
  configuration: DesignSource;
  mayManage: boolean;
}) {
  return (
    <div className="stack">
      {configuration.status === 'degraded' ? (
        <div className="callout callout--attention" role="alert">
          <p className="callout__title">Design resolution is paused</p>
          {/* The platform's own words about why it degraded, never a raw provider message. */}
          {configuration.status_reason ? <p>{configuration.status_reason}</p> : null}
          <p className="muted">Re-save this configuration once the token has been replaced.</p>
        </div>
      ) : null}
      <p className="muted">
        A feature that pastes a Figma frame link has it resolved once, before anything is
        planned, and every role that builds or judges the work is shown it. A feature that
        cites no design is unaffected by anything on this panel.
      </p>
      {configuration.credential_configured ? (
        <p className="muted">
          Figma token: configured (…{configuration.credential_hint}). It is entered in the
          Provider credentials panel above, where it appears as the “Figma” row.
        </p>
      ) : (
        <p className="muted">
          <Badge tone="attention">No Figma token stored</Badge> Enter it in the Provider
          credentials panel above — it appears there as the “Figma” row. Nothing is resolved
          without it, and a feature that cites a design is refused at submission until it is.
        </p>
      )}
      {configuration.configured ? (
        <DetailList narrow>
          <DetailRow label="Resolution">{configuration.enabled ? 'on' : 'off'}</DetailRow>
          {configuration.token_owner_id ? (
            // Read-only: whose stored `figma` credential the resolver opens. Re-saving keeps
            // it; a fresh configuration records the saver.
            <DetailRow label="Token owner">{configuration.token_owner_id}</DetailRow>
          ) : null}
          <DetailRow label="Files that may be cited">
            {configuration.file_allowlist_permits_any_file
              ? 'Any file this token can read'
              : configuration.file_allowlist.join(', ')}
          </DetailRow>
        </DetailList>
      ) : null}
      {mayManage ? (
        <DesignSourceForm configuration={configuration} />
      ) : (
        <p className="muted">
          {configuration.configured
            ? 'An operator saves this configuration.'
            : 'Not configured yet. An operator saves this configuration.'}
        </p>
      )}
    </div>
  );
}

function DesignSourceForm({ configuration }: { configuration: DesignSource }) {
  const api = useApi();
  const queryClient = useQueryClient();
  const [enabled, setEnabled] = useState(configuration.enabled);
  const [allowlist, setAllowlist] = useState(configuration.file_allowlist.join('\n'));

  const save = useMutation({
    mutationFn: (input: SaveDesignSourceInput) => api.saveDesignSource(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['design-source'] });
    },
  });
  const check = useMutation({
    mutationFn: () => api.checkDesignSource(),
    onSuccess: () => {
      // A refused check degrades the configuration on the server, so the banner above has to
      // be re-read rather than left showing the state before the button was pressed.
      void queryClient.invalidateQueries({ queryKey: ['design-source'] });
    },
  });

  return (
    <form
      className="form"
      onSubmit={(event) => {
        event.preventDefault();
        save.mutate({
          enabled,
          file_allowlist: allowlist
            .split('\n')
            .map((item) => item.trim())
            .filter((item) => item.length > 0),
          // Kept rather than re-defaulted: a different operator re-saving must not silently
          // re-point resolution at their own stored credential.
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
        Resolve design links on feature submissions
      </label>
      <Field
        label="Permitted file keys"
        hint="One per line. The key is the segment after /design/ or /file/ in a Figma URL, not the whole URL. Leave empty to permit any file this token can read."
      >
        {(id, describedBy) => (
          <textarea
            id={id}
            aria-describedby={describedBy}
            rows={4}
            value={allowlist}
            placeholder="28gd2JrZO28FCN9PCKM4qK"
            onChange={(event) => setAllowlist(event.target.value)}
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
        <button type="submit" className="button button--primary" disabled={save.isPending}>
          {save.isPending ? 'Saving…' : 'Save design source'}
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
