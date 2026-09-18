import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import { ArtifactViewer } from '@/components/artifacts/ArtifactViewer';
import { ARTIFACT_RENDERERS } from '@/components/artifacts/renderers';
import { artifactSchema } from '@/schemas/feature';
import saved from './fixtures/artifacts.adunit-deactivate-live-086.json';

/**
 * Every artifact type the platform produces, rendered from a real payload.
 *
 * The renderers read fields by name, so a field the server fills and a renderer never mentions
 * disappears silently — and that had happened to user stories, parallel groups, merge order,
 * the contract's authentication and error sections, the review's per-requirement checks, and
 * the created/modified/deleted file list. Nothing failed; the information was simply not on
 * the page. Only a comparison against a real payload finds that.
 *
 * Refresh the fixture from a running stack:
 *   for each artifact type, GET /features/<id>/artifacts/<artifact_id>
 * and save one of each. Check it for credentials first; this one was.
 */
const artifacts = saved.map((item) => artifactSchema.parse(item));

/**
 * Fields that are identifiers or digests rather than content.
 *
 * They are deliberately not rendered: an artifact id is followed by opening that artifact, and
 * a fingerprint is a comparison key with nothing to read in it.
 */
const NOT_CONTENT = new Set([
  'artifact_type',
  'schema_version',
  'feature_id',
  'repository_id',
  'parent_workflow_id',
  'child_workflow_id',
  'workstream_id',
  'contract_artifact_id',
  'code_completion_artifact_id',
  'review_artifact_id',
  'integration_review_artifact_id',
  'technical_prd_artifact_id',
  'integration_contract_artifact_id',
  'repository_execution_plan_artifact_id',
  'pull_request_artifact_ids',
  'child_workflow_results',
  'production_diff_fingerprint',
  'test_diff_fingerprint',
  'repository_revision',
  'approved_at',
  'completed_at',
]);

function populated(payload: Record<string, unknown>): string[] {
  return Object.entries(payload)
    .filter(([key, value]) => {
      if (NOT_CONTENT.has(key)) return false;
      if (value === null || value === undefined || value === '') return false;
      if (Array.isArray(value)) return value.length > 0;
      if (typeof value === 'object') return Object.keys(value as object).length > 0;
      return true;
    })
    .map(([key]) => key);
}

describe('the artifact renderers, against real payloads', () => {
  it('has one for every type this platform produces', () => {
    const types = [...new Set(artifacts.map((item) => item.artifact_type))];
    expect(types.length).toBeGreaterThan(6);
    for (const type of types) {
      // An unknown type falls back to raw JSON rather than a blank panel, which is right for a
      // type nobody has written a view for yet -- but not for one the platform produces on
      // every run. `child_workflow_result` sat in that fallback, and it is the artifact the
      // platform emits most.
      expect(ARTIFACT_RENDERERS[type], `no renderer for ${type}`).toBeDefined();
    }
  });

  it('does not make an unsafe pull-request URL clickable', () => {
    const artifact = artifactSchema.parse({
      artifact_id: '008_pull_request.evil.json',
      artifact_type: 'pull_request',
      workflow_id: 'feature-1:evil',
      schema_version: '1',
      producer: 'github_agent',
      timestamp: '2026-08-25T10:00:00Z',
      metadata: {},
      validation_status: 'valid',
      payload: {
        repository: 'evil',
        pull_request_number: 1,
        url: 'javascript:alert(1)',
        source_branch: 'feature',
        target_branch: 'main',
        state: 'open',
        body: 'Review this.',
      },
    });

    const { container } = render(
      <ArtifactViewer artifact={artifact} renderers={ARTIFACT_RENDERERS} />,
    );

    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(container).toHaveTextContent('evil #1');
  });

  it.each(artifacts.map((item) => [item.artifact_type, item] as const))(
    'renders every populated field of a real %s',
    (_type, artifact) => {
      const { container } = render(
        <ArtifactViewer artifact={artifact} renderers={ARTIFACT_RENDERERS} />,
      );
      // Attributes count as rendered: a link's URL lives in `href`, not in the text.
      const attributes = [...container.querySelectorAll('*')]
        .flatMap((node) => [...node.attributes].map((attribute) => attribute.value))
        .join(' ');
      const shown = `${container.textContent ?? ''} ${attributes}`;

      for (const field of populated(artifact.payload)) {
        // Look for the content, not the field name: a renderer may label it anything, and may
        // legitimately show one part of a structure rather than all of it. Any recognisable
        // token from the value proves the field reached the page; none of them proves it did
        // not.
        const candidates = tokens(artifact.payload[field]);
        if (candidates.length === 0) continue;
        expect(
          candidates.some((token) => shown.includes(token)),
          `${artifact.artifact_type}.${field} is not on the page (looked for ${candidates
            .slice(0, 3)
            .map((token) => JSON.stringify(token))
            .join(', ')})`,
        ).toBe(true);
      }
    },
  );
});

/** Distinctive strings from a payload value, bounded so a large document stays cheap. */
function tokens(value: unknown, depth = 0): string[] {
  if (depth > 3) return [];
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length >= 4 ? [trimmed.slice(0, 24)] : [];
  }
  if (Array.isArray(value)) return value.slice(0, 4).flatMap((item) => tokens(item, depth + 1));
  if (value && typeof value === 'object') {
    return Object.values(value)
      .slice(0, 8)
      .flatMap((item) => tokens(item, depth + 1));
  }
  return [];
}
