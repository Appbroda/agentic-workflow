import { useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { useRefreshFeature } from './hooks';
import { ApiError, userMessage } from '@/api/errors';
import { StreamError } from '@/api/stream';
import { ErrorState, TableSkeleton } from '@/components/common/States';
import { CredentialFields, type Credentials } from '@/components/common/CredentialFields';
import { modelProviderFor } from '@/utils/model';
import { Markdown } from '@/components/common/Markdown';
import { Badge, RepositoryBadge } from '@/components/ui/Badge';
import { IconClose } from '@/components/ui/icons';
import type { ChatMessage, Feature, FeatureAction } from '@/schemas/feature';

/**
 * The feature assistant, docked beside the workspace.
 *
 * It is part of this page rather than a separate chat product: it knows which feature is open,
 * says so, and suggests the questions somebody actually opens it to ask. It answers questions
 * and may propose an action, which does nothing until a person confirms it. Confirmation goes
 * to the server, which turns it into a durable action and decides for itself whether it is
 * legal -- so a refusal appears here as a refusal, not as a failed request, and an action that
 * was already carried out is not carried out twice.
 *
 * The answer arrives as it is written. That is an authenticated `fetch` reading server-sent
 * events, not `EventSource`, which cannot send an `Authorization` header.
 */

/**
 * The questions this panel exists to answer.
 *
 * Shown only before a conversation has started: once somebody is talking, a row of prompts is
 * clutter over the thing they are reading.
 */
const SUGGESTIONS = [
  'Why is this blocked?',
  'Summarise progress.',
  'Explain the latest failure.',
  'What needs my attention?',
  'What work is left?',
];

export function ChatPanel({
  featureId,
  feature,
  repositories,
  onClose,
}: {
  featureId: string;
  feature?: Feature;
  repositories?: string[];
  onClose?: () => void;
}) {
  const api = useApi();
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const [draft, setDraft] = useState(() => searchParams.get('prompt') ?? '');
  // Held for this page visit only, never stored, and sent as a request header. A key stored
  // for this identity is used when this is empty; supplying one here is a deliberate choice
  // about which key does this work, and the server prefers it for exactly that reason.
  const [credentials, setCredentials] = useState<Credentials>({});
  // The answer currently being written. Not in the transcript yet: the server persists it
  // when the stream ends, and this is what is on the screen until then.
  const [streaming, setStreaming] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [lastAsked, setLastAsked] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const history = useQuery({
    queryKey: ['chat', featureId],
    queryFn: ({ signal }) => api.getChatHistory(featureId, signal),
  });

  const refreshFeature = useRefreshFeature(featureId);
  const refresh = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['chat', featureId] });
    // A confirmed action changes the feature, so the rest of the workspace is stale too --
    // including the views keyed by the event cursor, which is why this goes through the
    // shared refresh rather than invalidating the feature alone.
    refreshFeature();
  }, [featureId, queryClient, refreshFeature]);

  const send = useCallback(
    async (message: string) => {
      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;
      setStreaming('');
      setStreamError(null);
      setLastAsked(message);
      let received = '';
      try {
        for await (const event of api.streamChatMessage(featureId, message, {
          credentials,
          signal: controller.signal,
        })) {
          if (event.event === 'delta') {
            received += (event.data as { text?: string }).text ?? '';
            setStreaming(received);
          } else if (event.event === 'user') {
            // The question is already in the transcript. Refreshing now means it appears
            // even if the answer never arrives.
            void queryClient.invalidateQueries({ queryKey: ['chat', featureId] });
          } else if (event.event === 'error') {
            setStreamError(String((event.data as { detail?: string }).detail ?? 'Unknown error'));
          }
        }
      } catch (error) {
        if (controller.signal.aborted) {
          // Cancelled deliberately. Whatever arrived is already persisted by the server, so
          // the transcript is the record and this partial view can go.
          setStreaming(null);
          refresh();
          return;
        }
        setStreamError(
          error instanceof StreamError || error instanceof Error
            ? error.message
            : 'The assistant could not be reached.',
        );
      } finally {
        if (abort.current === controller) abort.current = null;
      }
      setStreaming(null);
      refresh();
    },
    [api, credentials, featureId, queryClient, refresh],
  );

  // A stream left running after this panel is gone would keep a connection open and set
  // state on a component nobody is looking at.
  useEffect(() => () => abort.current?.abort(), []);

  const confirm = useMutation({
    // A proposal may resume, retry, or approve a repair. Confirmation therefore needs the
    // same request-scoped provider credentials as the ordinary action control.
    mutationFn: (messageId: number) => api.confirmChatAction(featureId, messageId, { credentials }),
    onSettled: () => refresh(),
  });
  const reject = useMutation({
    mutationFn: (messageId: number) => api.rejectChatAction(featureId, messageId),
    onSettled: () => refresh(),
  });

  const messages = history.data?.messages ?? [];
  useEffect(() => {
    // Scrolling to the newest turn is a convenience, not a requirement, and not every
    // environment implements it. It must never be the reason the conversation fails to render.
    endRef.current?.scrollIntoView?.({ block: 'end' });
  }, [messages.length, streaming]);

  const busy = confirm.isPending || reject.isPending;
  const generating = streaming !== null;

  return (
    <aside className="chat" aria-label="Feature assistant">
      <header className="chat__header">
        <span className="chat__title">Assistant</span>
        {onClose ? (
          <button
            type="button"
            className="button button--quiet button--icon"
            style={{ marginLeft: 'auto' }}
            onClick={onClose}
            aria-label="Close assistant"
          >
            <IconClose />
          </button>
        ) : null}
      </header>

      {/* A small statement of what it is looking at, not a dump of the context it was given. */}
      {feature ? (
        <div className="chat__context">
          <span className="details__label">Context</span>
          <span className="truncate" title={feature.title}>
            {feature.title}
          </span>
          {repositories && repositories.length > 0 ? (
            <span className="row" style={{ gap: 'var(--space-1)' }}>
              {repositories.slice(0, 4).map((id) => (
                <RepositoryBadge key={id} repositoryId={id} />
              ))}
              {repositories.length > 4 ? <span className="subtle">+{repositories.length - 4}</span> : null}
            </span>
          ) : null}
        </div>
      ) : null}

      {history.isPending ? (
        <TableSkeleton rows={3} label="Loading conversation…" />
      ) : history.isError ? (
        <div className="chat__log">
          <ChatLoadError error={history.error} onRetry={history.refetch} />
        </div>
      ) : (
        <ol className="chat__log" aria-label="Conversation">
          {messages.length === 0 && !generating ? (
            <li className="muted">
              Ask about this feature — what happened, what is blocked, why a repository stopped.
            </li>
          ) : null}
          {messages.map((message) => (
            <ChatTurn
              key={message.id}
              featureId={featureId}
              message={message}
              busy={busy}
              onConfirm={() => confirm.mutate(message.id)}
              onReject={() => reject.mutate(message.id)}
            />
          ))}
          {generating ? (
            <li className="chat__turn chat__turn--assistant">
              <span className="chat__role">Assistant</span>
              <div className="chat__bubble">
                {/* Markdown, rendered to React elements rather than HTML, so a heading or a
                    list in a half-written answer reads as one. */}
                <Markdown>{streaming}</Markdown>
                <p className="subtle" role="status">
                  Writing…
                </p>
              </div>
            </li>
          ) : null}
          <div ref={endRef} />
        </ol>
      )}

      {streamError ? (
        <div className="callout callout--warn" role="alert" style={{ margin: '0 var(--space-4)' }}>
          <p className="prose">{streamError}</p>
          {lastAsked ? (
            <div className="form__actions">
              <button
                type="button"
                className="button button--small"
                onClick={() => {
                  void send(lastAsked);
                }}
              >
                Ask again
              </button>
            </div>
          ) : null}
        </div>
      ) : null}

      {[confirm.error, reject.error].map((error, index) =>
        error ? (
          <p className="field__error" role="alert" key={index} style={{ padding: '0 var(--space-4)' }}>
            {error instanceof ApiError ? userMessage(error) : 'That did not work.'}
            {error instanceof ApiError && error.detail ? ` — ${error.detail}` : ''}
          </p>
        ) : null,
      )}

      {messages.length === 0 && !generating && !history.isError ? (
        <div className="chat__suggestions">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="chat__suggestion"
              onClick={() => void send(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      ) : null}

      <form
        className="chat__composer"
        onSubmit={(event) => {
          event.preventDefault();
          const message = draft.trim();
          if (!message || generating) return;
          setDraft('');
          void send(message);
        }}
      >
        <label className="field__label" htmlFor="chat-message">
          Ask about this feature
        </label>
        <textarea
          id="chat-message"
          rows={3}
          value={draft}
          disabled={generating}
          onChange={(event) => setDraft(event.target.value)}
        />
        <details className="chat__credentials">
          <summary className="muted">Provider key</summary>
          <CredentialFields
            value={credentials}
            onChange={setCredentials}
            need={[modelProviderFor(feature?.agent_platform), 'github']}
            note="The assistant runs on the same platform as the feature, so it uses that provider's key. Confirmed resume, retry or repair actions also use the GitHub token. A key stored in Settings is used when these are empty; anything typed here wins, is held for this page only, and is never stored."
          />
        </details>
        <div className="form__actions">
          <button type="submit" className="button button--primary" disabled={generating || !draft.trim()}>
            {generating ? 'Answering…' : 'Send'}
          </button>
          {generating ? (
            <button type="button" className="button button--quiet" onClick={() => abort.current?.abort()}>
              Stop
            </button>
          ) : null}
        </div>
      </form>
    </aside>
  );
}

/**
 * A deployment without a model answers 503. That is a fact about the deployment, not a failure
 * to render, and saying so beats an error box that implies something broke.
 */
function ChatLoadError({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  if (error instanceof ApiError && error.status === 503) {
    return (
      <li className="muted">
        The assistant is not configured for this deployment. Everything else on this page works
        without it.
      </li>
    );
  }
  return (
    <li>
      <ErrorState error={error} onRetry={onRetry} />
    </li>
  );
}

function ChatTurn({
  featureId,
  message,
  busy,
  onConfirm,
  onReject,
}: {
  featureId: string;
  message: ChatMessage;
  busy: boolean;
  onConfirm: () => void;
  onReject: () => void;
}) {
  return (
    <li className={`chat__turn chat__turn--${message.role}`}>
      <span className="chat__role">{message.role === 'user' ? 'You' : 'Assistant'}</span>
      <div className="chat__bubble">
        {/* Assistant output is untrusted text. `Markdown` parses to React elements and never
            produces HTML, so a `<script>` or an `onerror=` in it arrives as the characters it
            is -- which is what makes formatting it safe rather than a risk taken for looks. */}
        {message.role === 'user' ? (
          <p className="prose">{message.content}</p>
        ) : (
          <Markdown>{message.content}</Markdown>
        )}
      </div>

      {message.proposed_action ? (
        <ProposedAction
          featureId={featureId}
          message={message}
          busy={busy}
          onConfirm={onConfirm}
          onReject={onReject}
        />
      ) : null}
    </li>
  );
}

/**
 * Something the assistant is offering to do, rendered as a decision rather than as prose.
 *
 * A consequential action must never be a sentence in a paragraph somebody skims. It names the
 * action, the repository it would touch, and the reason given, and it does nothing until
 * somebody presses Confirm.
 */
function ProposedAction({
  featureId,
  message,
  busy,
  onConfirm,
  onReject,
}: {
  featureId: string;
  message: ChatMessage;
  busy: boolean;
  onConfirm: () => void;
  onReject: () => void;
}) {
  const action = message.proposed_action!;
  const repositoryId =
    typeof action.arguments.repository_id === 'string' ? action.arguments.repository_id : null;
  const reason = typeof action.arguments.reason === 'string' ? action.arguments.reason : null;

  return (
    <div className="action-card">
      <div className="action-card__header">
        <span className="details__label">Proposed action</span>
        <Badge tone="attention">{action.type.replace(/_/g, ' ').toLowerCase()}</Badge>
        {repositoryId ? <RepositoryBadge repositoryId={repositoryId} /> : null}
      </div>
      <p className="action-card__title">{action.summary}</p>
      {reason ? <p className="muted">{reason}</p> : null}

      {message.action_status === 'pending' ? (
        <div className="form__actions">
          {/* Nothing has happened yet. The platform still decides whether it may. */}
          <button type="button" className="button button--primary" disabled={busy} onClick={onConfirm}>
            Confirm
          </button>
          <button type="button" className="button button--quiet" disabled={busy} onClick={onReject}>
            Reject
          </button>
        </div>
      ) : (
        <ActionOutcome featureId={featureId} message={message} />
      )}
    </div>
  );
}

/**
 * What became of a confirmed proposal.
 *
 * Read from the durable action rather than from the message when there is one, because the
 * message is a view and the action is the record. That is what makes this survive a reload
 * mid-execution: the page follows the action, so refreshing shows "still running" rather than
 * a Confirm button for something already underway.
 */
function ActionOutcome({ featureId, message }: { featureId: string; message: ChatMessage }) {
  const api = useApi();
  const action = useQuery({
    queryKey: ['action', featureId, message.action_id],
    queryFn: ({ signal }) => api.getAction(featureId, message.action_id!, signal),
    enabled: Boolean(message.action_id),
    // Polled only while something is actually happening. A settled action is not asked about
    // again, and one that never started is not polled at all.
    refetchInterval: (query) => (query.state.data?.in_progress ? 2_000 : false),
  });

  const record: FeatureAction | undefined = action.data;
  const label = actionStatusLabel(record?.status ?? message.action_status);
  const detail = record?.result_summary ?? record?.error_message ?? message.action_result;

  return (
    <>
      <p className="muted" role="status">
        {label}
        {detail ? ` — ${detail}` : ''}
      </p>
      {record?.in_progress ? (
        <p className="muted">
          Still running. This is safe to leave: the platform is doing it, not this page.
        </p>
      ) : null}
      {record?.status === 'requires_reconciliation' ? (
        <p className="field__hint">
          The platform was interrupted while doing this and cannot confirm what reached the
          repository. Check the requested-action record on Overview together with any unresolved
          operations in Settings before an administrator records the outcome.
        </p>
      ) : null}
      {record && record.actor_display_name ? (
        <p className="muted">Asked for by {record.actor_display_name}.</p>
      ) : null}
    </>
  );
}

function actionStatusLabel(status: string | null | undefined): string {
  if (status === 'executed' || status === 'succeeded') return 'Confirmed';
  if (status === 'rejected') return 'Rejected';
  if (status === 'executing' || status === 'claimed' || status === 'confirmed') return 'Executing';
  if (status === 'needs_attention' || status === 'requires_reconciliation') {
    return 'Needs attention';
  }
  if (status === 'failed' || status === 'cancelled') return 'Not completed';
  return 'Status unavailable';
}
