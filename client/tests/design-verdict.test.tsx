import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { FeatureWorkspacePage } from '@/pages/FeatureWorkspacePage';
import type { FeatureApi } from '@/api/features';
import { clarificationSchema } from '@/schemas/feature';
import { stubApi } from './fixtures';
import verdictJson from './fixtures/clarification.awaiting-design-verdict.json';

/**
 * The design-conflict surface: a workstream stopped because a decision is being reversed.
 *
 * A different kind of question from a clarification, and it has to read as one. It arrives
 * after coding, it is about one repository, and the feature it belongs to has already stopped
 * — a person shown "this feature is waiting on you" here would go looking for the answer form
 * that reopens planning, which is not what settles this.
 *
 * The payload is one the real `/features/{id}/clarification` route produced, saved by
 * `server/tests/test_design_conflict_verdict.py` and parsed here through the real zod schema,
 * so a field the server stops sending fails in this file rather than in a browser.
 *
 * Regenerate by running that server test, or against a running stack with:
 *   curl -H "Authorization: Bearer $PLATFORM_API_KEY" \
 *     localhost:8000/features/<id>/clarification > tests/fixtures/clarification.awaiting-design-verdict.json
 */

const clarification = clarificationSchema.parse(verdictJson);
const [conflict] = clarification.design_conflicts;
if (!conflict) throw new Error('the saved payload must carry the design question under test');

function feature() {
  return {
    feature_id: clarification.feature_id,
    workflow_id: clarification.feature_id,
    status: 'failed_requires_human',
    title: 'Bulk app creation',
    reference: 'AB-Feature-197',
    current_agent: null,
    repository_count: 2,
    required_repository_count: 2,
    repositories: [],
    clarification_rounds: 0,
    integration_review_cycles: 0,
    merge_strategy: null,
    deployment_strategy: null,
    execution_mode: 'mock',
    cancellation_status: 'not_requested',
    cancellation_requested_at: null,
    cancellation_reason: null,
    cleanup_requirements: [],
    available_actions: ['ANSWER_DESIGN_VERDICT', 'CANCEL_WORKFLOW'],
    created_at: '2026-09-02T06:03:22Z',
    updated_at: '2026-09-02T07:10:19Z',
  };
}

function renderWorkspace(api: Partial<FeatureApi> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders
      api={stubApi({
        getFeature: async () => feature(),
        getWorkstreams: async () => ({ feature_id: clarification.feature_id, workstreams: [] }),
        getTimeline: async () => ({ feature_id: clarification.feature_id, events: [] }),
        listEvents: async () => ({
          feature_id: clarification.feature_id,
          events: [],
          last_event_id: null,
        }),
        getClarification: async () => clarification,
        ...api,
      })}
      queryClient={queryClient}
    >
      <MemoryRouter
        initialEntries={[`/features/${clarification.feature_id}`]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/:featureId" element={<FeatureWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

describe('a design conflict is shown as a decision, not as a defect', () => {
  it('is its own clarification state, and not the requirement answer form', () => {
    // The server decides this, and the client renders it. `awaiting_answers` stays false
    // because that flag is the technical-PRD contract: its answers reopen planning.
    expect(clarification.clarification_state).toBe('awaiting_design_verdict');
    expect(clarification.awaiting_answers).toBe(false);
    expect(clarification.questions).toHaveLength(0);
  });

  it('shows both positions, so the reader can see one defect and two sentences', async () => {
    renderWorkspace();

    expect(
      await screen.findByRole('heading', {
        name: 'A design decision is being reversed — this one is yours to settle',
      }),
    ).toBeInTheDocument();
    // The platform's own question, and each side named with the review that holds it.
    expect(screen.getByText(conflict.question)).toBeInTheDocument();
    expect(screen.getByText(conflict.demanded.statement)).toBeInTheDocument();
    expect(screen.getByText(conflict.satisfied!.statement)).toBeInTheDocument();
    expect(screen.getByText(/This repository’s review requires this now/)).toBeInTheDocument();
    expect(
      screen.getByText(/This repository’s review required it, and stopped/),
    ).toBeInTheDocument();
    // And what the attempt that removed it said, which is the grounds a reader weighs. Scoped
    // to the position it belongs to: the same sentence is also in the evidence list, and an
    // unscoped match would pass on a card that had put it under the wrong side.
    const satisfiedSide = screen
      .getByText(/This repository’s review required it, and stopped/)
      .closest('.callout') as HTMLElement;
    expect(within(satisfiedSide).getByText(/The attempt that removed it reported/)).toHaveTextContent(
      conflict.satisfied!.grounds,
    );
    // Not the requirement form: nothing here reopens planning.
    expect(screen.queryByText('This feature is waiting on you')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Submit answers/i })).not.toBeInTheDocument();
  });

  it('will not submit a verdict without the reasoning the next attempt is bound by', async () => {
    const answer = vi.fn(async () => feature());
    renderWorkspace({ answerDesignConflict: answer as never });
    const user = userEvent.setup();

    const submit = await screen.findByRole('button', {
      name: 'Record decision and continue',
    });
    // Nothing chosen, nothing written: the server refuses both, and so does the button.
    expect(submit).toBeDisabled();

    await user.click(screen.getByRole('radio', { name: /The requirement stands/ }));
    expect(submit).toBeDisabled();

    await user.type(
      screen.getByLabelText('Why — and how the next attempt should handle it'),
      'It stands. Unwind in the same transaction.',
    );
    expect(submit).toBeEnabled();
    await user.click(submit);

    await waitFor(() => expect(answer).toHaveBeenCalledTimes(1));
    // No credentials travel with the verdict: the attempt it authorises is queued and the
    // worker resolves the feature owner's stored keys, so there is nothing here to collect.
    expect(answer).toHaveBeenCalledWith(clarification.feature_id, conflict.conflict_id, {
      verdict: 'requirement_holds',
      decision: 'It stands. Unwind in the same transaction.',
      // This fixture's repository has no attempts left, so the decision buys the one it
      // needs — stated, never topped up on the caller's behalf.
      additional_attempts: 1,
    });
  });

  it('says whether the decision has to buy an attempt, from what the server published', async () => {
    renderWorkspace();

    expect(conflict.attempts_remaining).toBe(0);
    expect(await screen.findByLabelText('Attempts to grant')).toBeInTheDocument();
    expect(
      screen.getByText(/has no attempts left, so acting on the decision has to buy at least one/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/of its budget, unspent/)).not.toBeInTheDocument();
  });

  it('offers no control for a question the server says it cannot act on', async () => {
    // Parsed rather than assembled, so this stays a payload the server could actually send.
    const unanswerable = clarificationSchema.parse({
      ...clarification,
      design_conflicts: [{ ...conflict, answerable: false }],
    });
    renderWorkspace({ getClarification: async () => unanswerable });

    // The question is still worth reading; the control whose only outcome is a refusal is not
    // a control.
    expect(await screen.findByText(conflict.question)).toBeInTheDocument();
    expect(
      screen.getByText(/it has used every review cycle this feature allows/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Record decision and continue' }),
    ).not.toBeInTheDocument();
  });
});
