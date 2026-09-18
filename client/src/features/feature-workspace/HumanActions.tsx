import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import { Dialog } from '@/components/ui/Dialog';
import { StoredCredentialsNote } from '@/components/common/CredentialFields';
import type { Clarification, Feature } from '@/schemas/feature';
import { useRefreshFeature } from './hooks';
import { ClarificationHistory } from './ClarificationHistory';

/**
 * The actions a person can take on a feature.
 *
 * Which of them is legal is the platform's decision, not this component's: every button asks
 * the server, and a refusal comes back as a 409 that is shown as an explanation. Nothing here
 * re-implements a transition rule, and an action the feature does not currently offer is not
 * rendered at all -- a control whose only outcome is a refusal is not a control.
 */

export function FeatureControls({
  feature,
  genericResumeAllowed,
}: {
  feature: Feature;
  genericResumeAllowed: boolean;
}) {
  const api = useApi();
  const [confirming, setConfirming] = useState<'cancel' | 'resume' | null>(null);
  const [reason, setReason] = useState('');
  const invalidate = useRefreshFeature(feature.feature_id);

  const cancel = useMutation({
    mutationFn: () => api.cancelFeature(feature.feature_id, reason.trim() || undefined),
    onSuccess: () => {
      setConfirming(null);
      setReason('');
      invalidate();
    },
  });
  const resume = useMutation({
    // An empty answer list is the server's own "recover this feature" call, distinct from
    // answering a clarification. No credentials travel with it: a resume is queued, and the
    // worker resolves the feature owner's stored keys -- a header would be discarded.
    mutationFn: () => api.resumeFeature(feature.feature_id, []),
    onSuccess: () => {
      setConfirming(null);
      invalidate();
    },
  });

  const mayResume = feature.available_actions?.includes('RESUME_WORKFLOW') ?? false;
  const mayCancel = feature.available_actions?.includes('CANCEL_WORKFLOW') ?? false;
  if (!mayResume && !mayCancel) return null;

  return (
    <>
      {genericResumeAllowed && mayResume ? (
        <button type="button" className="button" onClick={() => setConfirming('resume')}>
          Resume
        </button>
      ) : null}
      {mayCancel ? (
        <button type="button" className="button button--danger" onClick={() => setConfirming('cancel')}>
          Cancel feature
        </button>
      ) : null}

      {confirming === 'cancel' ? (
        <Dialog title="Cancel this feature?" onDismiss={() => setConfirming(null)}>
          <p className="muted">
            Work already pushed and any pull requests already opened are kept. The platform stops
            scheduling new work.
          </p>
          <div className="field">
            <label className="field__label" htmlFor="cancel-reason">
              Reason (optional)
            </label>
            <input id="cancel-reason" value={reason} onChange={(event) => setReason(event.target.value)} />
          </div>
          {cancel.error ? <ActionError error={cancel.error} /> : null}
          <div className="form__actions">
            <button
              type="button"
              className="button button--danger"
              disabled={cancel.isPending}
              onClick={() => cancel.mutate()}
            >
              {cancel.isPending ? 'Cancelling…' : 'Cancel feature'}
            </button>
            <button type="button" className="button button--quiet" onClick={() => setConfirming(null)}>
              Keep as is
            </button>
          </div>
        </Dialog>
      ) : null}

      {confirming === 'resume' ? (
        <Dialog title="Resume this feature?" onDismiss={() => setConfirming(null)}>
          <p className="muted">
            The platform continues the incomplete part from its last safe checkpoint. It refuses
            if the feature is not resumable.
          </p>
          {feature.execution_mode === 'live' ? <StoredCredentialsNote /> : null}
          {resume.error ? <ActionError error={resume.error} /> : null}
          <div className="form__actions">
            <button
              type="button"
              className="button button--primary"
              disabled={resume.isPending}
              onClick={() => resume.mutate()}
            >
              {resume.isPending ? 'Resuming…' : 'Resume'}
            </button>
            <button type="button" className="button button--quiet" onClick={() => setConfirming(null)}>
              Keep as is
            </button>
          </div>
        </Dialog>
      ) : null}
    </>
  );
}

/**
 * The questions the platform is answering itself, shown as its open items.
 *
 * AB-Feature-173's operator watched an empty panel for 35 minutes while the platform read
 * checkouts and grounded answers to exactly these questions -- the old rendering collapsed
 * "we are answering these ourselves" into "waiting for you". No form: nothing here is
 * actionable, and the copy says whose items they are.
 */
export function InvestigatingPanel({ clarification }: { clarification: Clarification }) {
  return (
    <section className="panel" aria-labelledby="investigating-heading">
      <div className="panel__header">
        <h2 className="panel__title" id="investigating-heading">
          Open questions the platform is investigating
        </h2>
        <span className="muted">Not waiting on you</span>
      </div>
      <p className="muted">
        These questions came up while analysing the requirement. The platform is reading your
        repositories to answer them itself; whatever it cannot settle will be asked here, with
        its suggested answers filled in.
      </p>
      <ul className="bullets">
        {clarification.questions.map((question) => (
          <li key={question.question_id} className="prose">
            {question.question}
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * The questions the platform stopped to ask.
 *
 * This is the single most important thing on the page when it is present, so it is an action
 * card at the top of the Overview rather than a section somebody has to find. Each question
 * carries the evidence behind it -- which repository, what the requirements assumed, what the
 * checkout has instead -- so the answer can be checked rather than guessed.
 */
export function ClarificationPanel({
  featureId,
  reference,
  clarification,
  groundingFailed = false,
}: {
  featureId: string;
  /** The feature's own identity, so this card says what it is about when read on its own. */
  reference?: string | null;
  clarification: Clarification;
  /**
   * The platform tried to answer these from the repositories, failed, and fell back to
   * asking -- a person who knows that reads the questions differently.
   */
  groundingFailed?: boolean;
}) {
  const api = useApi();
  const refresh = useRefreshFeature(featureId);
  // Prefilled from the platform's own suggestions, so accepting one is doing nothing and
  // changing one is typing over it. A question with no suggestion starts empty.
  const [answers, setAnswers] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      clarification.questions.map((question) => [
        question.question_id,
        question.suggested_answer,
      ]),
    ),
  );

  const submit = useMutation({
    // No credentials travel with the answers: the continuation is queued, and the worker
    // resolves the feature owner's stored keys -- a header would be discarded.
    mutationFn: () =>
      api.resumeFeature(
        featureId,
        clarification.questions.map((question) => ({
          question_id: question.question_id,
          answer: answers[question.question_id] ?? '',
        })),
      ),
    onSuccess: refresh,
  });

  // The server requires an answer for every open question and rejects a partial set, so the
  // button waits until all of them have something rather than letting the request fail.
  const complete = clarification.questions.every(
    (question) => (answers[question.question_id] ?? '').trim().length > 0,
  );
  const suggested = clarification.questions.filter(
    (question) => question.suggested_answer.trim().length > 0,
  );
  const edited = suggested.some(
    (question) => answers[question.question_id] !== question.suggested_answer,
  );

  return (
    <section className="action-card" aria-labelledby="clarification-heading">
      <div className="action-card__header">
        <h2 className="action-card__title" id="clarification-heading">
          This feature is waiting on you
        </h2>
        <span className="muted">
          {reference ? <span className="mono">{reference}</span> : null}
          {reference ? ' · ' : ''}
          Round {clarification.clarification_rounds + 1} of {clarification.max_clarification_rounds}
        </span>
      </div>
      <p className="muted">
        Every question needs an answer before the feature can continue. Where the platform read
        the answer in one of your repositories it has filled it in — check it, and change it if
        it is wrong.
      </p>

      {groundingFailed ? (
        <div className="callout callout--warn">
          <p className="callout__title">The platform tried to answer these itself</p>
          <p className="prose">
            Reading your repositories for answers failed, so these questions come to you
            without the platform’s suggestions checked against the checkouts. Answer them from
            your own knowledge of the repositories.
          </p>
        </div>
      ) : null}

      <ClarificationHistory clarification={clarification} />

      {clarification.questions.map((question) => (
        <div key={question.question_id} className="field">
          <label className="field__label" htmlFor={`q-${question.question_id}`}>
            {question.question}
          </label>
          <p className="field__hint">{question.rationale}</p>
          {question.suggested_answer ? (
            // The suggestion is shown as well as prefilled, so that after an edit it is still
            // possible to see what was proposed and where it came from.
            <div className="suggestion">
              <p className="suggestion__source">
                {question.suggestion_source || 'Suggested answer'}
                {question.suggestion_confidence ? (
                  <span className="subtle"> · {question.suggestion_confidence} confidence</span>
                ) : null}
              </p>
              <p className="suggestion__answer">{question.suggested_answer}</p>
            </div>
          ) : null}
          <textarea
            id={`q-${question.question_id}`}
            rows={3}
            value={answers[question.question_id] ?? ''}
            onChange={(event) =>
              setAnswers((current) => ({ ...current, [question.question_id]: event.target.value }))
            }
          />
        </div>
      ))}

      {submit.isError ? <ActionError error={submit.error} /> : null}

      <div className="form__actions">
        <button
          type="button"
          className="button button--primary"
          disabled={!complete || submit.isPending}
          onClick={() => submit.mutate()}
        >
          {submit.isPending ? 'Submitting…' : 'Submit answers'}
        </button>
        {/* Offered only once something has been changed away from the suggestions: before
            that it would do nothing, since the fields already hold them. */}
        {suggested.length > 0 && edited ? (
          <button
            type="button"
            className="button button--quiet"
            onClick={() =>
              setAnswers((current) => ({
                ...current,
                ...Object.fromEntries(
                  suggested.map((question) => [question.question_id, question.suggested_answer]),
                ),
              }))
            }
          >
            Restore suggested answers
          </button>
        ) : null}
      </div>
    </section>
  );
}

export function ActionError({ error }: { error: unknown }) {
  const apiError = error instanceof ApiError ? error : undefined;
  return (
    <p className="field__error" role="alert">
      {apiError ? userMessage(apiError) : 'The action failed.'}
      {apiError?.detail ? ` — ${apiError.detail}` : ''}
    </p>
  );
}
