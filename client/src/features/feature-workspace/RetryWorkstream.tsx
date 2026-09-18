import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import type { Workstream } from '@/schemas/feature';
import { StoredCredentialsNote } from '@/components/common/CredentialFields';
import { Dialog } from '@/components/ui/Dialog';
import { useFeature } from './hooks';

/**
 * Buy a stopped repository more attempts.
 *
 * The platform stops a repository when its budget runs out and will not reset that by itself,
 * because repeating an attempt on unchanged inputs costs money to reach the same place. This
 * is the override, so it asks for the two things that make it reviewable later: who is asking,
 * and what they know that the last attempt did not. Both are required by the server too --
 * client-side validation here is a courtesy, not the rule.
 */
export function RetryWorkstream({
  featureId,
  workstream,
  onDone,
}: {
  featureId: string;
  workstream: Workstream;
  onDone: () => void;
}) {
  const api = useApi();
  // One fact about the feature: which model key an attempt on it uses. Read from the cache
  // the workspace already holds rather than threaded through the repository table.
  const feature = useFeature(featureId);
  const [open, setOpen] = useState(false);
  const [requestedBy, setRequestedBy] = useState('');
  const [reason, setReason] = useState('');
  const [attempts, setAttempts] = useState(1);

  const retry = useMutation({
    // No credentials travel with the grant: the attempt is queued, and the worker resolves
    // the feature owner's stored keys -- a header would be discarded.
    mutationFn: () =>
      api.retryWorkstream(featureId, workstream.repository_id, {
        additional_attempts: attempts,
        requested_by: requestedBy.trim(),
        reason: reason.trim(),
      }),
    onSuccess: () => {
      setOpen(false);
      onDone();
    },
  });

  const ready = requestedBy.trim().length > 0 && reason.trim().length > 0;

  if (!open) {
    return (
      <button type="button" className="button button--small" onClick={() => setOpen(true)}>
        Grant another attempt
      </button>
    );
  }

  return (
    <Dialog
      title={`Grant ${workstream.repository_id} another attempt`}
      onDismiss={() => setOpen(false)}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (ready) retry.mutate();
        }}
        className="stack stack--tight"
      >
      <p className="muted">
        This runs a real attempt: it clones the repository, writes code, runs its checks and can
        push a branch. It may take several minutes.
      </p>

      <div className="field">
        <label className="field__label" htmlFor={`retry-by-${workstream.repository_id}`}>
          Your name
        </label>
        <input
          id={`retry-by-${workstream.repository_id}`}
          value={requestedBy}
          onChange={(event) => setRequestedBy(event.target.value)}
          disabled={retry.isPending}
        />
      </div>

      <div className="field">
        <label className="field__label" htmlFor={`retry-why-${workstream.repository_id}`}>
          What changed since the last attempt?
        </label>
        <textarea
          id={`retry-why-${workstream.repository_id}`}
          rows={3}
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          disabled={retry.isPending}
        />
      </div>

      <div className="field">
        <label className="field__label" htmlFor={`retry-n-${workstream.repository_id}`}>
          Attempts to grant
        </label>
        <input
          id={`retry-n-${workstream.repository_id}`}
          type="number"
          min={1}
          max={5}
          value={attempts}
          onChange={(event) => setAttempts(Number(event.target.value) || 1)}
          disabled={retry.isPending}
        />
      </div>

      {feature.data?.execution_mode === 'live' ? <StoredCredentialsNote /> : null}

      {retry.error ? (
        <p className="field__error" role="alert">
          {retry.error instanceof ApiError ? userMessage(retry.error) : 'That did not work.'}
          {retry.error instanceof ApiError && retry.error.detail
            ? ` — ${retry.error.detail}`
            : ''}
        </p>
      ) : null}

      <div className="form__actions">
        <button
          type="submit"
          className="button button--primary"
          disabled={!ready || retry.isPending}
        >
          {retry.isPending ? 'Running…' : 'Grant and run'}
        </button>
        <button
          type="button"
          className="button button--quiet"
          disabled={retry.isPending}
          onClick={() => setOpen(false)}
        >
          Cancel
        </button>
      </div>
      </form>
    </Dialog>
  );
}
