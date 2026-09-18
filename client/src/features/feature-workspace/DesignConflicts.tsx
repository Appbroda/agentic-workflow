import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { StoredCredentialsNote } from '@/components/common/CredentialFields';
import { RepositoryBadge } from '@/components/ui/Badge';
import type { DesignConflict } from '@/schemas/feature';
import { useFeature, useRefreshFeature } from './hooks';
import { ActionError } from './HumanActions';

/**
 * A design decision one repository keeps reversing, put to the person who owns the feature.
 *
 * This is a different kind of question from a clarification: it arrives after coding, it is
 * about one repository rather than the requirement, and the workstream it belongs to has
 * already stopped. The platform stopped it on purpose — a review demanded a change, a later
 * cycle stopped demanding it, and it came back, so another attempt would re-argue the question
 * instead of answering it.
 *
 * Nothing here ranks the two positions or suggests an answer. Both are shown as they were
 * actually worded, and the decision is the reader's. What the answer buys is a resumed attempt
 * bound by it: the server carries the decision into the next coding call as an invariant, so
 * whichever way it goes, it is not argued again.
 */

const AUTHORITY_LABELS: Record<DesignConflict['demanded']['authority'], string> = {
  repository_review: 'This repository’s review',
  integration_review: 'The integration review',
};

export function DesignConflictPanels({
  featureId,
  reference,
  conflicts,
}: {
  featureId: string;
  /** The feature's own identity, so the card says what it is about when read on its own. */
  reference?: string | null;
  conflicts: DesignConflict[];
}) {
  if (conflicts.length === 0) return null;
  return (
    <>
      {conflicts.map((conflict) => (
        <DesignConflictCard
          key={conflict.conflict_id}
          featureId={featureId}
          reference={reference}
          conflict={conflict}
        />
      ))}
    </>
  );
}

export function DesignConflictCard({
  featureId,
  reference,
  conflict,
}: {
  featureId: string;
  reference?: string | null;
  conflict: DesignConflict;
}) {
  const api = useApi();
  const refresh = useRefreshFeature(featureId);
  // Read from the cache the workspace already holds: this card needs one fact about the
  // feature, which model key its next attempt will run on.
  const feature = useFeature(featureId);
  const [choice, setChoice] = useState<'requirement_holds' | 'removal_holds' | null>(null);
  const [decision, setDecision] = useState('');
  // Only ever above zero when the workstream has nothing left to spend. The server refuses a
  // verdict it could not act on and says so; this is the field that answers it.
  const [grant, setGrant] = useState(1);

  const submit = useMutation({
    // No credentials travel with the verdict: the attempt it authorises is queued, and the
    // worker resolves the feature owner's stored keys -- a header would be discarded.
    mutationFn: () =>
      api.answerDesignConflict(featureId, conflict.conflict_id, {
        verdict: choice ?? 'requirement_holds',
        decision,
        additional_attempts: conflict.attempts_remaining === 0 ? grant : 0,
      }),
    onSuccess: refresh,
  });

  // The decision travels into the next coding call verbatim, so an empty one is not a
  // decision. The server rejects it too; this stops the round trip.
  const ready = choice !== null && decision.trim().length > 0;

  return (
    <section className="action-card" aria-labelledby={`conflict-${conflict.conflict_id}`}>
      <div className="action-card__header">
        <h2 className="action-card__title" id={`conflict-${conflict.conflict_id}`}>
          {conflict.kind === 'recurring_demand'
            ? 'A review keeps demanding the same change — this one is yours to settle'
            : conflict.cross_authority
              ? 'Two reviews disagree — this one is yours to settle'
              : 'A design decision is being reversed — this one is yours to settle'}
        </h2>
        <span className="muted">
          {reference ? <span className="mono">{reference}</span> : null}
          {reference ? ' · ' : ''}
          <RepositoryBadge repositoryId={conflict.repository_id} />
        </span>
      </div>

      <p className="prose">{conflict.question}</p>

      <div className={conflict.satisfied ? 'grid-2' : undefined}>
        <div className="callout callout--warn">
          <p className="callout__title">
            {AUTHORITY_LABELS[conflict.demanded.authority]} requires this now
          </p>
          <p className="prose">{conflict.demanded.statement}</p>
        </div>
        {conflict.satisfied ? (
          <div className="callout">
            <p className="callout__title">
              {AUTHORITY_LABELS[conflict.satisfied.authority]} required it, and stopped
            </p>
            <p className="prose">{conflict.satisfied.statement}</p>
            {conflict.satisfied.grounds ? (
              <p className="muted">
                The attempt that removed it reported: {conflict.satisfied.grounds}
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      <details>
        <summary className="muted">
          What the platform read to conclude this ({conflict.attempts_spent} attempt
          {conflict.attempts_spent === 1 ? '' : 's'} spent
          {conflict.kind === 'recurring_demand'
            ? ', never satisfied'
            : `, removed ${conflict.removals} time${conflict.removals === 1 ? '' : 's'}`}
          )
        </summary>
        <ul className="bullets">
          {conflict.evidence.map((item, index) => (
            <li key={index} className="prose">
              {item}
            </li>
          ))}
        </ul>
      </details>

      {conflict.answerable ? null : (
        <p className="field__hint" role="status">
          This repository cannot act on a decision any more — it has used every review cycle
          this feature allows. The question is recorded; acting on it means starting a fresh
          feature.
        </p>
      )}

      {conflict.answerable ? (
        <>
          <fieldset className="field">
            <legend className="field__label">Which position holds?</legend>
            <label className="field__hint">
              <input
                type="radio"
                name={`verdict-${conflict.conflict_id}`}
                checked={choice === 'requirement_holds'}
                onChange={() => setChoice('requirement_holds')}
              />{' '}
              The requirement stands. The next attempt implements it and may not remove it
              again.
            </label>
            <label className="field__hint">
              <input
                type="radio"
                name={`verdict-${conflict.conflict_id}`}
                checked={choice === 'removal_holds'}
                onChange={() => setChoice('removal_holds')}
              />{' '}
              The requirement does not stand. The review is overruled and the next attempt must
              not implement it.
            </label>
          </fieldset>

          <div className="field">
            <label className="field__label" htmlFor={`decision-${conflict.conflict_id}`}>
              Why — and how the next attempt should handle it
            </label>
            <p className="field__hint">
              This is handed to the next coding attempt word for word as the decision it may not
              argue. An answer that says how to implement it is what has preceded a delivery;
              “yes” on its own is not one.
            </p>
            <textarea
              id={`decision-${conflict.conflict_id}`}
              rows={4}
              value={decision}
              onChange={(event) => setDecision(event.target.value)}
            />
          </div>

          {conflict.attempts_remaining === 0 ? (
            <div className="field">
              <label className="field__label" htmlFor={`grant-${conflict.conflict_id}`}>
                Attempts to grant
              </label>
              <p className="field__hint">
                This repository has no attempts left, so acting on the decision has to buy at
                least one. Deciding the question does not refund the attempts the stop declined
                to spend.
              </p>
              <input
                id={`grant-${conflict.conflict_id}`}
                type="number"
                min={1}
                max={5}
                value={grant}
                onChange={(event) => setGrant(Number(event.target.value))}
              />
            </div>
          ) : (
            <p className="field__hint">
              This repository still has {conflict.attempts_remaining} attempt
              {conflict.attempts_remaining === 1 ? '' : 's'} of its budget, unspent — the stop
              did not use them. Nothing needs to be bought.
            </p>
          )}

          {feature.data?.execution_mode === 'live' ? <StoredCredentialsNote /> : null}

          {submit.isError ? <ActionError error={submit.error} /> : null}

          <div className="form__actions">
            <button
              type="button"
              className="button button--primary"
              disabled={!ready || submit.isPending}
              onClick={() => submit.mutate()}
            >
              {submit.isPending ? 'Recording…' : 'Record decision and continue'}
            </button>
          </div>
        </>
      ) : null}
    </section>
  );
}
