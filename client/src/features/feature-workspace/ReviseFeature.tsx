import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import { Dialog } from '@/components/ui/Dialog';
import { StoredCredentialsNote } from '@/components/common/CredentialFields';
import type { Feature, Workstream } from '@/schemas/feature';
import { useRefreshFeature } from './hooks';

/**
 * Ask for changes to a completed feature's published work.
 *
 * A completed feature's pull requests are open for a person to review, and this is what that
 * person presses when some of it does not look right. The request text becomes the revision
 * run's requirement set verbatim: the work is revised on a new `-V{n}` branch created from the
 * branch that published it, a replacement pull request opens, and the superseded one is closed
 * with a comment pointing at its replacement.
 *
 * Whether it is offered at all is the server's answer, read from `available_actions` — it
 * appears only on a completed feature that actually shipped a pull request.
 */
export function ReviseFeature({
  feature,
  workstreams,
}: {
  feature: Feature;
  workstreams: Workstream[];
}) {
  const api = useApi();
  const invalidate = useRefreshFeature(feature.feature_id);
  const [open, setOpen] = useState(false);
  const [request, setRequest] = useState('');

  const revise = useMutation({
    // No credentials travel with the request: the revision run is queued, and the worker
    // resolves the feature owner's stored keys -- a header would be discarded.
    mutationFn: () =>
      api.reviseFeature(feature.feature_id, { request: request.trim(), requested_by: '' }),
    onSuccess: () => {
      setOpen(false);
      setRequest('');
      invalidate();
    },
  });

  if (!(feature.available_actions ?? []).includes('REVISE_FEATURE')) return null;

  const published = workstreams.filter((item) => item.pull_request_artifact_id !== null);
  const nextVersion = (feature.revision ?? 0) + 2;
  const ready = request.trim().length > 0;

  if (!open) {
    return (
      <button type="button" className="button button--primary" onClick={() => setOpen(true)}>
        Request changes
      </button>
    );
  }

  return (
    <Dialog title="Request changes to this feature?" onDismiss={() => setOpen(false)}>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (ready) revise.mutate();
        }}
        className="stack stack--tight"
      >
        <p className="muted">
          The published work is kept and revised on a new branch (V{nextVersion}). A replacement
          pull request opens for each repository below, and the one it supersedes is closed with
          a comment pointing at it. Nothing is merged.
        </p>

        {published.length ? (
          <div className="callout">
            <p className="callout__title">Published work being revised — {published.length}</p>
            <ul className="bullets">
              {published.map((item) => (
                <li key={item.repository_id} className="prose">
                  {item.repository_name ?? item.repository_id}
                  {item.branch_name ? ` — ${item.branch_name}` : ''}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="field">
          <label className="field__label" htmlFor="revise-request">
            What should change?
          </label>
          <textarea
            id="revise-request"
            rows={5}
            value={request}
            onChange={(event) => setRequest(event.target.value)}
            disabled={revise.isPending}
            placeholder="Describe the changes you want, in your own words. This becomes the revision's requirement."
          />
        </div>

        {feature.execution_mode === 'live' ? <StoredCredentialsNote /> : null}

        {revise.error ? (
          <p className="field__error" role="alert">
            {revise.error instanceof ApiError ? userMessage(revise.error) : 'That did not work.'}
            {revise.error instanceof ApiError && revise.error.detail
              ? ` — ${revise.error.detail}`
              : ''}
          </p>
        ) : null}

        <div className="form__actions">
          <button
            type="submit"
            className="button button--primary"
            disabled={!ready || revise.isPending}
          >
            {revise.isPending ? 'Submitting…' : `Start V${nextVersion}`}
          </button>
          <button
            type="button"
            className="button button--quiet"
            disabled={revise.isPending}
            onClick={() => setOpen(false)}
          >
            Cancel
          </button>
        </div>
      </form>
    </Dialog>
  );
}
