import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import { CredentialFields, type Credentials } from '@/components/common/CredentialFields';
import { modelProviderFor } from '@/utils/model';
import { useArtifactList, useFeature, useRefreshFeature } from './hooks';

/** Fields accepted by the server's `IntegrationContractRevision` request model. */
const REVISION_FIELDS = [
  'contract_version',
  'api_style',
  'endpoints',
  'shared_schemas',
  'authentication_contract',
  'authorization_rules',
  'error_contracts',
  'event_contracts',
  'environment_variables',
  'compatibility_policy',
  'owning_workstreams',
] as const;

/**
 * Resolve a real pending immutable-contract request through the backend's existing actions.
 *
 * Approval requires the complete replacement contract; the API intentionally does not accept a
 * patch because every affected repository must consume one immutable revision. The editor is
 * therefore prefilled from the current contract and the backend validates the final shape.
 */
export function ContractChangeActions({
  featureId,
  at,
}: {
  featureId: string;
  at: number | null;
}) {
  const api = useApi();
  const refresh = useRefreshFeature(featureId);
  const feature = useFeature(featureId);
  const requests = useArtifactList(featureId, 'contract_change_request', at, true);
  const contracts = useArtifactList(featureId, 'integration_contract', at, true);
  const pending = useMemo(
    () =>
      [...(requests.data?.artifacts ?? [])]
        .reverse()
        .find((item) => item.payload.status === 'pending') ?? null,
    [requests.data],
  );
  const current = contracts.data?.artifacts.at(-1) ?? null;
  const requestId = typeof pending?.payload.change_request_id === 'string'
    && pending.payload.change_request_id.trim()
    ? pending.payload.change_request_id
    : null;
  const currentIsComplete = current !== null
    && REVISION_FIELDS.every((field) => Object.hasOwn(current.payload, field));
  const [decision, setDecision] = useState<'approve' | 'reject' | null>(null);
  const [resolution, setResolution] = useState('');
  const [revision, setRevision] = useState('');
  const [parseError, setParseError] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<Credentials>({});

  const initialRevision = current
    ? JSON.stringify(
        Object.fromEntries(REVISION_FIELDS.map((field) => [field, current.payload[field]])),
        null,
        2,
      )
    : '';

  const approve = useMutation({
    mutationFn: async () => {
      setParseError(null);
      let updatedContract: unknown;
      try {
        updatedContract = JSON.parse(revision);
      } catch {
        setParseError('The replacement contract must be valid JSON.');
        throw new LocalValidationError();
      }
      if (!updatedContract || typeof updatedContract !== 'object' || Array.isArray(updatedContract)) {
        setParseError('The replacement contract must be a JSON object.');
        throw new LocalValidationError();
      }
      if (!requestId) {
        setParseError('The pending request has no valid request identifier.');
        throw new LocalValidationError();
      }
      return api.approveContractChange(
        featureId,
        requestId,
        { resolution: resolution.trim(), updated_contract: updatedContract },
        { credentials },
      );
    },
    onSuccess: () => {
      setDecision(null);
      setCredentials({});
      refresh();
    },
  });
  const reject = useMutation({
    mutationFn: () => {
      if (!requestId) throw new LocalValidationError();
      return api.rejectContractChange(
        featureId,
        requestId,
        { resolution: resolution.trim() },
      );
    },
    onSuccess: () => {
      setDecision(null);
      refresh();
    },
  });

  if (requests.isError) {
    return (
      <section className="callout callout--warn" aria-labelledby="contract-change-error-heading">
        <h2 id="contract-change-error-heading">Contract decisions are unavailable</h2>
        <ActionError error={requests.error} />
      </section>
    );
  }
  if (!pending) return null;
  const requestedChanges = strings(pending.payload.requested_changes);
  const affected = strings(pending.payload.affected_workstreams);
  const busy = approve.isPending || reject.isPending;
  const error = approve.error instanceof LocalValidationError ? null : approve.error ?? reject.error;

  return (
    <section className="callout callout--attention" aria-labelledby="contract-change-heading">
      <h2 id="contract-change-heading">A shared contract change needs a decision</h2>
      <p className="prose">{String(pending.payload.reason ?? '')}</p>
      {requestedChanges.length > 0 ? (
        <ul>{requestedChanges.map((item) => <li key={item}>{item}</li>)}</ul>
      ) : null}
      {affected.length > 0 ? <p className="muted">Affected: {affected.join(', ')}</p> : null}
      {typeof pending.payload.compatibility_impact === 'string' ? (
        <p className="muted">Compatibility: {pending.payload.compatibility_impact}</p>
      ) : null}
      <p className="muted">
        Approval creates a new immutable contract revision and reruns the affected workstreams.
      </p>
      {!requestId ? (
        <p className="field__error" role="alert">
          This request has no valid identifier. It cannot be acted on safely.
        </p>
      ) : null}
      {contracts.isError ? <ActionError error={contracts.error} /> : null}
      {!contracts.isPending && !contracts.isError && !currentIsComplete ? (
        <p className="field__error" role="alert">
          The current complete contract revision is unavailable, so approval is disabled.
          Rejection remains available.
        </p>
      ) : null}

      {decision === null ? (
        <div className="form__actions">
          <button
            type="button"
            className="button button--primary"
            disabled={!requestId || !currentIsComplete || contracts.isError}
            onClick={() => {
              setRevision(initialRevision);
              setDecision('approve');
            }}
          >
            Review approval
          </button>
          <button type="button" className="button" disabled={!requestId} onClick={() => setDecision('reject')}>
            Review rejection
          </button>
          <Link
            className="button button--quiet"
            to={`/features/${encodeURIComponent(featureId)}/chat?prompt=${encodeURIComponent(
              `Explain contract change ${requestId ?? ''}, its compatibility impact, and what I should verify before deciding.`,
            )}`}
          >
            Ask AI
          </Link>
        </div>
      ) : (
        <div role="dialog" aria-modal="true" aria-label={`${decision} contract change`}>
          <label className="field__label" htmlFor="contract-resolution">
            Decision rationale
          </label>
          <textarea
            id="contract-resolution"
            rows={3}
            value={resolution}
            disabled={busy}
            onChange={(event) => setResolution(event.target.value)}
          />
          {decision === 'approve' ? (
            <>
              <label className="field__label" htmlFor="contract-revision">
                Complete replacement contract (JSON)
              </label>
              <p className="field__hint">
                Prefilled from the current approved revision. The platform validates every field
                and refuses incompatible or incomplete contracts. Set contract_version to a
                higher semantic version before approval; choose major, minor, or patch according
                to the compatibility impact.
              </p>
              <textarea
                id="contract-revision"
                rows={16}
                value={revision}
                disabled={busy}
                onChange={(event) => setRevision(event.target.value)}
              />
              <details>
                <summary className="muted">Use different keys for this rerun (optional)</summary>
                <CredentialFields
                  value={credentials}
                  onChange={setCredentials}
                  need={[modelProviderFor(feature.data?.agent_platform), 'github']}
                  note="Left blank, the rerun uses the credentials stored in Settings. Keys typed here override them for this request only and are never stored."
                />
              </details>
            </>
          ) : null}
          {parseError ? <p className="field__error" role="alert">{parseError}</p> : null}
          {error ? <ActionError error={error} /> : null}
          <div className="form__actions">
            <button
              type="button"
              className="button button--primary"
              disabled={busy || !resolution.trim() || (decision === 'approve' && !revision.trim())}
              onClick={() => (decision === 'approve' ? approve.mutate() : reject.mutate())}
            >
              {decision === 'approve' ? 'Approve and rerun affected workstreams' : 'Reject contract change'}
            </button>
            <button type="button" className="button button--quiet" disabled={busy} onClick={() => setDecision(null)}>
              Keep pending
            </button>
          </div>
        </div>
      )}
    </section>
  );
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function ActionError({ error }: { error: unknown }) {
  const apiError = error instanceof ApiError ? error : null;
  return (
    <p className="field__error" role="alert">
      {apiError ? userMessage(apiError) : 'That decision could not be recorded.'}
      {apiError?.detail ? ` — ${apiError.detail}` : ''}
    </p>
  );
}

class LocalValidationError extends Error {}
