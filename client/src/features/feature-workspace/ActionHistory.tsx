import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import type { FeatureAction } from '@/schemas/feature';
import { Badge, RepositoryBadge } from '@/components/ui/Badge';
import { Panel } from '@/components/ui/Layout';
import { DetailList, DetailRow } from '@/components/ui/Value';
import type { Tone } from '@/components/ui/tone';
import { absoluteTime, relativeTime } from '@/utils/time';
import { useFeatureActions } from './hooks';

/**
 * How a durable action's own status reads. `requires_reconciliation` is the one that needs a
 * person, so it is the one that is coloured for attention.
 */
function statusTone(status: string): Tone {
  if (status === 'executed' || status === 'succeeded') return 'done';
  if (status === 'failed' || status === 'cancelled' || status === 'rejected') return 'stopped';
  if (status === 'requires_reconciliation' || status === 'needs_attention') return 'attention';
  if (status === 'executing' || status === 'claimed' || status === 'confirmed') return 'working';
  return 'neutral';
}

/** Durable user-requested mutations, including outcomes recovered after a process crash. */
export function ActionHistory({ featureId, at }: { featureId: string; at: number | null }) {
  const actions = useFeatureActions(featureId, at);
  const api = useApi();
  const me = useQuery({ queryKey: ['me'], queryFn: ({ signal }) => api.getMe(signal) });

  if (actions.isError) {
    return (
      <Panel title="Requested actions">
        <p className="field__error" role="alert">
          {actions.error instanceof ApiError
            ? userMessage(actions.error)
            : 'The durable action history could not be loaded.'}
        </p>
      </Panel>
    );
  }
  if (!actions.data || actions.data.actions.length === 0) return null;
  const mayReconcile = me.data?.permissions.includes('action:reconcile') ?? false;

  return (
    <Panel
      title="Requested actions"
      meta="Durable requests from controls and chat"
    >
      <p className="muted">
        An interrupted request remains here after a refresh or restart rather than disappearing
        with the page that submitted it.
      </p>
      <ul className="cards" aria-label="Requested actions">
        {actions.data.actions.map((action) => (
          <ActionCard
            key={action.action_id}
            featureId={featureId}
            action={action}
            mayReconcile={mayReconcile}
          />
        ))}
      </ul>
    </Panel>
  );
}

function ActionCard({
  featureId,
  action,
  mayReconcile,
}: {
  featureId: string;
  action: FeatureAction;
  mayReconcile: boolean;
}) {
  const api = useApi();
  const queryClient = useQueryClient();
  const [reason, setReason] = useState('');
  const reconcile = useMutation({
    mutationFn: (outcome: 'succeeded' | 'failed') =>
      api.reconcileAction(featureId, action.action_id, outcome, reason.trim()),
    onSuccess: () => {
      setReason('');
      void queryClient.invalidateQueries({ queryKey: ['actions', featureId] });
      void queryClient.invalidateQueries({ queryKey: ['action', featureId, action.action_id] });
      void queryClient.invalidateQueries({ queryKey: ['chat', featureId] });
    },
  });
  const detail = action.result_summary ?? action.error_message;

  return (
    <li className="card">
      <div className="card__header">
        <strong>{action.action_type}</strong>
        <Badge tone={statusTone(action.status)}>{action.status}</Badge>
        {action.repository_id ? <RepositoryBadge repositoryId={action.repository_id} /> : null}
      </div>
      <DetailList narrow>
        <DetailRow label="Requested by">{action.actor_display_name ?? action.actor_id}</DetailRow>
        <DetailRow label="Attempt">
          {action.attempt} of {action.max_attempts}
        </DetailRow>
        <DetailRow label="Requested">
          <span title={absoluteTime(action.created_at)}>{relativeTime(action.created_at)}</span>
        </DetailRow>
        {action.reconciled_by ? (
          <DetailRow label="Reconciled by">{action.reconciled_by}</DetailRow>
        ) : null}
      </DetailList>
      {detail ? <p className="muted">{detail}</p> : null}
      {action.status === 'confirmed' && !action.in_progress ? (
        <p className="field__hint">
          Recovery proved this request is safe to ask for again. Return to its control or chat
          proposal to confirm the new attempt.
        </p>
      ) : null}
      {action.reconciliation_reason && action.reconciliation_reason !== detail ? (
        <p className="muted">Reconciliation: {action.reconciliation_reason}</p>
      ) : null}

      {action.status === 'requires_reconciliation' ? (
        <div className="callout callout--attention">
          <p className="state__title">The platform cannot prove this action’s outcome</p>
          <p className="muted">
            Check the feature, repository, and linked provider operations. Reconciliation records
            that finding; it never reruns this action or mutates the workflow.
          </p>
          {mayReconcile ? (
            <>
              <label className="field__label" htmlFor={`reconcile-${action.action_id}`}>
                Evidence checked and conclusion
              </label>
              <textarea
                id={`reconcile-${action.action_id}`}
                rows={3}
                value={reason}
                disabled={reconcile.isPending}
                onChange={(event) => setReason(event.target.value)}
              />
              <div className="form__actions">
                <button
                  type="button"
                  className="button button--primary"
                  disabled={reconcile.isPending || !reason.trim()}
                  onClick={() => reconcile.mutate('succeeded')}
                >
                  Mark verified complete
                </button>
                <button
                  type="button"
                  className="button"
                  disabled={reconcile.isPending || !reason.trim()}
                  onClick={() => reconcile.mutate('failed')}
                >
                  Mark verified not completed
                </button>
              </div>
            </>
          ) : (
            <p className="muted">An administrator must record the verified outcome.</p>
          )}
          {reconcile.error ? (
            <p className="field__error" role="alert">
              {reconcile.error instanceof ApiError
                ? userMessage(reconcile.error)
                : 'The reconciliation decision could not be recorded.'}
            </p>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}
