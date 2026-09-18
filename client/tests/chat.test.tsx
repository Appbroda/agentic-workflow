import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import { ApiError } from '@/api/errors';
import type { ChatMessage } from '@/schemas/feature';
import { stubApi } from './fixtures';

const FEATURE = {
  feature_id: 'f-1',
  workflow_id: 'w-1',
  status: 'completed',
  title: 'A feature',
  current_agent: null,
  repository_count: 1,
  required_repository_count: 1,
  repositories: [],
  clarification_rounds: 0,
  integration_review_cycles: 0,
  merge_strategy: null,
  deployment_strategy: null,
  execution_mode: 'live',
  cancellation_status: 'not_requested',
  cancellation_requested_at: null,
  cancellation_reason: null,
  cleanup_requirements: [],
  created_at: '2026-08-24T22:40:00Z',
  updated_at: '2026-08-24T23:20:00Z',
};

const BASE: Partial<FeatureApi> = {
  getFeature: async () => FEATURE,
  listEvents: async () => ({ feature_id: 'f-1', events: [], last_event_id: null }),
  getWorkstreams: async () => ({ feature_id: 'f-1', workstreams: [] }),
  getTimeline: async () => ({ feature_id: 'f-1', events: [] }),
  getClarification: async () => ({
    feature_id: 'f-1',
    awaiting_answers: false,
    technical_prd_artifact_id: null,
    clarification_rounds: 0,
    max_clarification_rounds: 10,
    questions: [],
    previous_answers: {},
    design_conflicts: [],
  }),
};

function renderChat(api: Partial<FeatureApi>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi({ ...BASE, ...api })} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={['/features/f-1/chat']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId/:tab" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

const ANSWER = {
  id: 2,
  role: 'assistant',
  content: 'Both repositories finished and opened pull requests.',
  proposed_action: null,
  action_status: null,
  action_result: null,
  created_at: null,
};

const PROPOSAL = {
  id: 4,
  role: 'assistant',
  content: 'I can cancel this feature.',
  proposed_action: {
    type: 'CANCEL_WORKFLOW',
    arguments: { reason: 'no longer needed' },
    summary: 'Cancel this feature.',
  },
  action_status: 'pending',
  action_result: null,
  created_at: null,
};

describe('ChatPanel', () => {
  it('shows the persisted conversation when the feature is reopened', async () => {
    renderChat({
      getChatHistory: async () => ({
        feature_id: 'f-1',
        messages: [
          { ...ANSWER, id: 1, role: 'user', content: 'What happened?' },
          ANSWER,
        ],
      }),
    });

    const log = await screen.findByRole('list', { name: 'Conversation' });
    expect(within(log).getByText('What happened?')).toBeInTheDocument();
    expect(
      within(log).getByText('Both repositories finished and opened pull requests.'),
    ).toBeInTheDocument();
  });

  it('shows the answer arriving in pieces, then the persisted turn', async () => {
    const streamChatMessage = vi.fn();
    let messages: ChatMessage[] = [];
    renderChat({
      streamChatMessage: (featureId: string, message: string, options: unknown) => {
        streamChatMessage(featureId, message, options);
        return (async function* () {
          yield { event: 'user', data: {}, id: null };
          yield { event: 'delta', data: { text: 'Both repositories ' }, id: null };
          yield { event: 'delta', data: { text: 'finished and opened pull requests.' }, id: null };
          // The server persists the turn when the stream ends, which is why the transcript
          // read afterwards is what carries it rather than anything held on this page.
          messages = [{ ...ANSWER, id: 1, role: 'user', content: 'What happened?' }, ANSWER];
          yield { event: 'message', data: ANSWER, id: null };
          yield { event: 'done', data: {}, id: null };
        })();
      },
      getChatHistory: async () => ({ feature_id: 'f-1', messages }),
    });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Ask about this feature'), 'What happened?');
    await user.click(screen.getByRole('button', { name: 'Send' }));

    expect(streamChatMessage).toHaveBeenCalledWith(
      'f-1',
      'What happened?',
      expect.objectContaining({ credentials: {} }),
    );
    expect(
      await screen.findByText('Both repositories finished and opened pull requests.'),
    ).toBeInTheDocument();
  });

  it('keeps what was read when the answer is cut off partway', async () => {
    // The person read those words. A page that discards them on failure disagrees with what
    // they saw, and the server has already written them into the transcript.
    renderChat({
      streamChatMessage: () =>
        (async function* () {
          yield { event: 'delta', data: { text: 'The backend stopped because' }, id: null };
          yield { event: 'error', data: { detail: 'The assistant did not finish answering.' }, id: null };
        })(),
      getChatHistory: async () => ({ feature_id: 'f-1', messages: [] }),
    });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Ask about this feature'), 'Why?');
    await user.click(screen.getByRole('button', { name: 'Send' }));

    expect(await screen.findByText(/did not finish answering/)).toBeInTheDocument();
    // And a way to try again, rather than making somebody retype the question.
    expect(await screen.findByRole('button', { name: 'Ask again' })).toBeInTheDocument();
  });

  it('sends the provider key with the question, and holds it only in memory', async () => {
    const streamChatMessage = vi.fn();
    renderChat({
      streamChatMessage: (featureId: string, message: string, options: unknown) => {
        streamChatMessage(featureId, message, options);
        return (async function* () {
          yield { event: 'done', data: {}, id: null };
        })();
      },
      getChatHistory: async () => ({ feature_id: 'f-1', messages: [] }),
    });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('Ask about this feature'), 'What happened?');
    await user.type(screen.getByLabelText('OpenAI API key'), 'sk-typed-by-a-person');
    await user.click(screen.getByRole('button', { name: 'Send' }));

    // A key typed here wins over anything stored, is sent as a header, and is never written
    // to storage -- which is asserted below against the real storage objects.
    expect(streamChatMessage).toHaveBeenCalledWith('f-1', 'What happened?', {
      credentials: { openaiApiKey: 'sk-typed-by-a-person' },
      signal: expect.anything(),
    });
    expect(window.localStorage.getItem('openaiApiKey')).toBeNull();
    expect(document.body.innerHTML).not.toContain('sk-typed-by-a-person');
  });

  it('shows a proposal without doing anything until it is confirmed', async () => {
    const confirmChatAction = vi.fn().mockResolvedValue({
      ...PROPOSAL,
      action_status: 'executed',
      action_result: 'Feature is now cancelled.',
    });
    renderChat({
      confirmChatAction,
      getChatHistory: async () => ({ feature_id: 'f-1', messages: [PROPOSAL] }),
    });
    const user = userEvent.setup();

    expect(await screen.findByText('Cancel this feature.')).toBeInTheDocument();
    // The assistant proposed; nothing has been asked of the platform yet.
    expect(confirmChatAction).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(confirmChatAction).toHaveBeenCalledWith('f-1', 4, { credentials: {} });
  });

  it('keeps provider credentials on a confirmed live action', async () => {
    const confirmChatAction = vi.fn().mockResolvedValue({
      ...PROPOSAL,
      action_status: 'executed',
      action_result: 'Feature resumed.',
    });
    renderChat({
      confirmChatAction,
      getChatHistory: async () => ({ feature_id: 'f-1', messages: [PROPOSAL] }),
    });
    const user = userEvent.setup();

    await user.type(await screen.findByLabelText('OpenAI API key'), 'sk-live');
    await user.type(screen.getByLabelText('GitHub token'), 'gh-live');
    await user.click(screen.getByRole('button', { name: 'Confirm' }));

    expect(confirmChatAction).toHaveBeenCalledWith('f-1', 4, {
      credentials: { openaiApiKey: 'sk-live', githubToken: 'gh-live' },
    });
  });

  it('reports the platform refusing a confirmed action', async () => {
    let messages: ChatMessage[] = [PROPOSAL];
    renderChat({
      getChatHistory: async () => ({ feature_id: 'f-1', messages }),
      confirmChatAction: async () => {
        messages = [{
          ...PROPOSAL,
          action_status: 'failed',
          action_result: 'The platform did not complete this action.',
        }];
        throw new ApiError('conflict', 'no', 409, 'completed or cancelled features cannot be resumed');
      },
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Confirm' }));

    // The platform decides legality; the assistant's proposal does not overrule it.
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The platform refused this action in the feature’s current state.',
    );
    expect(screen.getByRole('alert')).toHaveTextContent('cannot be resumed');
    expect(await screen.findByText(/Not completed/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Confirm' })).not.toBeInTheDocument();
  });

  it('records a rejected proposal rather than hiding it', async () => {
    const rejectChatAction = vi
      .fn()
      .mockResolvedValue({ ...PROPOSAL, action_status: 'rejected', action_result: 'Rejected by the operator.' });
    let messages: ChatMessage[] = [PROPOSAL];
    renderChat({
      rejectChatAction: async (featureId: string, messageId: number) => {
        const result = await rejectChatAction(featureId, messageId);
        messages = [result];
        return result;
      },
      getChatHistory: async () => ({ feature_id: 'f-1', messages }),
    });
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: 'Reject' }));

    await waitFor(() => expect(screen.getByText(/Rejected/)).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: 'Confirm' })).not.toBeInTheDocument();
  });

  it('says the assistant is unconfigured rather than showing an error', async () => {
    renderChat({
      getChatHistory: async () => {
        throw new ApiError('unknown', 'unavailable', 503, 'not configured');
      },
    });

    // A deployment without a model is a fact about the deployment, not a failure to render.
    expect(
      await screen.findByText(/The assistant is not configured for this deployment/),
    ).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('renders assistant output as text, never as markup', async () => {
    renderChat({
      getChatHistory: async () => ({
        feature_id: 'f-1',
        messages: [{ ...ANSWER, content: 'Try <img src=x onerror="alert(1)"> in the console.' }],
      }),
    });

    // Agent output is untrusted; it must reach the page as characters.
    expect(
      await screen.findByText('Try <img src=x onerror="alert(1)"> in the console.'),
    ).toBeInTheDocument();
    expect(document.querySelector('img')).toBeNull();
  });
});

describe('a confirmed action, after the page reloads', () => {
  /**
   * The window this covers is the one durable actions exist for: somebody confirmed
   * something, it is still running, and they reloaded. The page has to follow the action --
   * which the platform is executing -- rather than the request, which is gone.
   */
  const CONFIRMED = {
    ...PROPOSAL,
    action_status: 'executing',
    action_id: 'action-1',
  };

  it('shows an action still running rather than a button to start it again', async () => {
    renderChat({
      getChatHistory: async () => ({ feature_id: 'f-1', messages: [CONFIRMED] }),
      getAction: async () => ({
        action_id: 'action-1',
        feature_id: 'f-1',
        action_type: 'CANCEL_WORKFLOW',
        actor_id: 'user-1',
        actor_display_name: 'Alex',
        origin: 'chat',
        status: 'executing',
        attempt: 1,
        max_attempts: 1,
        created_at: '2026-08-25T10:00:00Z',
        in_progress: true,
        external_operation_ids: [],
      }),
    });

    expect(await screen.findByText(/Executing/)).toBeInTheDocument();
    expect(await screen.findByText(/safe to leave/)).toBeInTheDocument();
    // Confirming again is not offered: the platform is already doing it.
    expect(screen.queryByRole('button', { name: 'Confirm' })).not.toBeInTheDocument();
  });

  it('reports an action whose outcome the platform could not confirm', async () => {
    // The one case that must never quietly become "try again": the platform was interrupted
    // and does not know what reached the repository.
    renderChat({
      getChatHistory: async () => ({
        feature_id: 'f-1',
        messages: [{ ...CONFIRMED, action_status: 'needs_attention' }],
      }),
      getAction: async () => ({
        action_id: 'action-1',
        feature_id: 'f-1',
        action_type: 'RETRY_WORKSTREAM',
        actor_id: 'user-1',
        actor_display_name: 'Alex',
        origin: 'chat',
        status: 'requires_reconciliation',
        attempt: 1,
        max_attempts: 1,
        created_at: '2026-08-25T10:00:00Z',
        in_progress: false,
        error_message: 'One operation may have taken effect and could not be confirmed.',
        external_operation_ids: ['operation-1'],
      }),
    });

    expect(await screen.findByText(/Needs attention/)).toBeInTheDocument();
    expect(await screen.findByText(/cannot confirm what reached the repository/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Confirm' })).not.toBeInTheDocument();
  });

  it('names who asked for a completed action', async () => {
    renderChat({
      getChatHistory: async () => ({
        feature_id: 'f-1',
        messages: [{ ...CONFIRMED, action_status: 'executed' }],
      }),
      getAction: async () => ({
        action_id: 'action-1',
        feature_id: 'f-1',
        action_type: 'CANCEL_WORKFLOW',
        actor_id: 'user-1',
        actor_display_name: 'Alex',
        origin: 'chat',
        status: 'succeeded',
        attempt: 1,
        max_attempts: 1,
        created_at: '2026-08-25T10:00:00Z',
        in_progress: false,
        result_summary: 'Feature is now cancelled.',
        external_operation_ids: [],
      }),
    });

    expect(await screen.findByText(/Confirmed — Feature is now cancelled./)).toBeInTheDocument();
    expect(await screen.findByText('Asked for by Alex.')).toBeInTheDocument();
  });
});
