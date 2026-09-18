// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { RepositoryPage } from '@/pages/RepositoryPage';
import { workstreamsSchema, type Artifact } from '@/schemas/feature';
import type { FeatureApi } from '@/api/features';
import {
  faultEvidence,
  runtimeSentence,
} from '@/features/feature-workspace/workstream-view';
import { stubApi } from './fixtures';
import saved from './fixtures/workstreams.adunit-deactivate-live-086.json';

/**
 * Absorbed provider faults reach the repository detail panel.
 *
 * 186's attempt-0 record carried only the two runtime clocks; the fault class, the count
 * and the degraded-retry marker existed nowhere the page could read. The child loop now
 * folds them into the result artifact's metadata, and the Execution panel labels the
 * wall-vs-charged difference with them instead of leaving it a bare pair of numbers.
 */

/** The metadata the child loop records for 186's shape: one ReadError, one degraded retry. */
const FAULTED_METADATA = {
  child_retry_count: 0,
  fault_count: 1,
  fault_seconds_excluded: 844,
  fault_classes: ['LLMAdapterError/ReadError'],
  truncated_degraded_retry: true,
  runtime_wall_seconds: 3060,
  runtime_charged_seconds: 2216,
};

function resultArtifact(repositoryId: string, metadata: Record<string, unknown>): Artifact {
  return {
    artifact_id: `011_child_workflow_result.${repositoryId}.attempt-0.json`,
    artifact_type: 'child_workflow_result',
    schema_version: '1.0',
    producer: 'child_workflow',
    timestamp: '2026-08-31T10:00:00Z',
    metadata,
    validation_status: 'valid',
    payload: {},
  };
}

describe('the fault evidence, read from a result artifact', () => {
  it('labels the wall-vs-charged difference with what the provider cost this run', () => {
    const evidence = faultEvidence(FAULTED_METADATA);
    expect(evidence).not.toBeNull();
    expect(runtimeSentence(evidence!)).toBe(
      'wall 51m 0s, charged 36m 56s — 844s excluded across 1 provider fault',
    );
  });

  it('reads explicit zeros as a fault-free attempt, not as an unrecorded one', () => {
    const evidence = faultEvidence({
      fault_count: 0,
      fault_seconds_excluded: 0,
      fault_classes: [],
      truncated_degraded_retry: false,
      runtime_wall_seconds: 180,
      runtime_charged_seconds: 180,
    });
    expect(evidence).not.toBeNull();
    expect(evidence!.faultCount).toBe(0);
    expect(evidence!.truncatedDegradedRetry).toBe(false);
    expect(runtimeSentence(evidence!)).toBe('wall 3m 0s, charged 3m 0s');
  });

  it('treats absent keys as "recorded before this shipped"', () => {
    expect(faultEvidence({ child_retry_count: 2 })).toBeNull();
    expect(faultEvidence(undefined)).toBeNull();
  });
});

describe('the repository detail panel', () => {
  it('renders the wall-vs-charged sentence beside the retry counters', async () => {
    const data = workstreamsSchema.parse(saved);
    const repositoryId = data.workstreams[0]!.repository_id;
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const api: Partial<FeatureApi> = {
      getFeature: async () => ({
        feature_id: data.feature_id,
        workflow_id: data.feature_id,
        status: 'failed_requires_human',
        title: 'A feature with recorded faults',
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
        available_actions: [],
        created_at: '2026-08-24T22:40:00Z',
        updated_at: '2026-08-24T23:20:00Z',
      }),
      getWorkstreams: async () => data,
      listEvents: async () => ({ feature_id: data.feature_id, events: [], last_event_id: null }),
      getTimeline: async () => ({ feature_id: data.feature_id, events: [] }),
      listArtifacts: async () => ({
        feature_id: data.feature_id,
        artifacts: [resultArtifact(repositoryId, FAULTED_METADATA)],
      }),
      getPullRequests: async () => ({ feature_id: data.feature_id, pull_requests: [] }),
      listRepairs: async () => ({ feature_id: data.feature_id, repairs: [] }),
    };

    render(
      <AppProviders api={stubApi(api)} queryClient={queryClient}>
        <MemoryRouter
          initialEntries={[`/features/${data.feature_id}/repositories/${repositoryId}`]}
          future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
        >
          <Routes>
            <Route
              path="/features/:featureId/repositories/:repositoryId"
              element={<RepositoryPage />}
            />
          </Routes>
        </MemoryRouter>
      </AppProviders>,
    );

    expect(
      await screen.findByText('wall 51m 0s, charged 36m 56s — 844s excluded across 1 provider fault'),
    ).toBeInTheDocument();
    expect(screen.getByText('Provider faults')).toBeInTheDocument();
    expect(screen.getByText('LLMAdapterError/ReadError')).toBeInTheDocument();
    expect(screen.getByText('Truncation')).toBeInTheDocument();
  });
});
