import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import { Dialog } from '@/components/ui/Dialog';
import { StoredCredentialsNote } from '@/components/common/CredentialFields';
import type { Feature, Workstream } from '@/schemas/feature';
import { useRefreshFeature } from './hooks';

/**
 * Open the pull requests a feature that did not land is holding.
 *
 * A feature that fully landed publishes itself. One that did not publishes nothing until a
 * person says so, because a pull request whose sibling repository does not exist is not
 * reviewable work -- it is a trap for whoever opens it. This is the decision that replaces
 * that automatic pull request, so it has to be impossible to miss and it has to say, before
 * it is pressed, exactly what it would open: which repositories passed review, which ones
 * are clean but were rejected, and which are being held back and why.
 *
 * Whether it is offered at all is the server's answer, read from `available_actions`. A
 * control whose only outcome is a 409 is not a control.
 */
export function PublishFeature({
  feature,
  workstreams,
}: {
  feature: Feature;
  workstreams: Workstream[];
}) {
  const api = useApi();
  const invalidate = useRefreshFeature(feature.feature_id);
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState('');

  const publish = useMutation({
    // No credentials travel with the decision: the publication is queued, and the worker
    // resolves the feature owner's stored keys -- a header would be discarded.
    mutationFn: () =>
      api.publishFeature(feature.feature_id, { requested_by: '', reason: reason.trim() }),
    onSuccess: () => {
      setOpen(false);
      setReason('');
      invalidate();
    },
  });

  if (!(feature.available_actions ?? []).includes('PUBLISH_FEATURE')) return null;

  const reviewed = workstreams.filter((item) => item.publication_class === 'reviewed');
  const unreviewed = workstreams.filter((item) => item.publication_class === 'unreviewed');
  const held = workstreams.filter(
    (item) => !item.publication_class && item.pull_request_artifact_id === null,
  );
  const ready = reason.trim().length > 0;

  if (!open) {
    return (
      <button type="button" className="button button--primary" onClick={() => setOpen(true)}>
        Publish {reviewed.length + unreviewed.length} repositor
        {reviewed.length + unreviewed.length === 1 ? 'y' : 'ies'}
      </button>
    );
  }

  return (
    <Dialog title="Publish this feature's work?" onDismiss={() => setOpen(false)}>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (ready) publish.mutate();
        }}
        className="stack stack--tight"
      >
        <p className="muted">
          This feature did not land, so nothing was published automatically. Publishing it now
          opens a draft pull request for each repository below. Nothing is merged.
        </p>

        {reviewed.length ? (
          <div className="callout">
            <p className="callout__title">Passed review — {reviewed.length}</p>
            <ul className="bullets">
              {reviewed.map((item) => (
                <li key={item.repository_id} className="prose">
                  {item.repository_name ?? item.repository_id}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {unreviewed.length ? (
          <div className="callout callout--attention">
            <p className="callout__title">Rejected by review — {unreviewed.length}</p>
            <p className="muted">
              Every check these repositories require passed, and the review still said no.
              Publishing them overrules that judgement: their pull requests are titled
              <strong> [REVIEW REJECTED]</strong> and their descriptions list what the reviewer
              asked for.
            </p>
            <ul className="bullets">
              {unreviewed.map((item) => (
                <li key={item.repository_id} className="prose">
                  {item.repository_name ?? item.repository_id}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {held.length ? (
          <div className="callout">
            <p className="callout__title">Not published — {held.length}</p>
            <ul className="bullets">
              {held.map((item) => (
                <li key={item.repository_id} className="prose">
                  {item.repository_name ?? item.repository_id}
                  {item.publication_refusal ? ` — ${item.publication_refusal}` : ''}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="field">
          <label className="field__label" htmlFor="publish-why">
            Why are you publishing this?
          </label>
          <textarea
            id="publish-why"
            rows={3}
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            disabled={publish.isPending}
          />
        </div>

        {feature.execution_mode === 'live' ? <StoredCredentialsNote /> : null}

        {publish.error ? (
          <p className="field__error" role="alert">
            {publish.error instanceof ApiError ? userMessage(publish.error) : 'That did not work.'}
            {publish.error instanceof ApiError && publish.error.detail
              ? ` — ${publish.error.detail}`
              : ''}
          </p>
        ) : null}

        <div className="form__actions">
          <button
            type="submit"
            className="button button--primary"
            disabled={!ready || publish.isPending}
          >
            {publish.isPending ? 'Publishing…' : 'Publish'}
          </button>
          <button
            type="button"
            className="button button--quiet"
            disabled={publish.isPending}
            onClick={() => setOpen(false)}
          >
            Cancel
          </button>
        </div>
      </form>
    </Dialog>
  );
}
