import { useState, type ReactNode } from 'react';
import type { Artifact } from '@/schemas/feature';
import { absoluteTime } from '@/utils/time';
import { humanise } from '@/utils/text';
import { Segmented } from '@/components/ui/Layout';
import { RawJson } from '@/components/ui/Code';
import { CopyValue, DetailList, DetailRow } from '@/components/ui/Value';

/**
 * One viewer for every artifact, with a formatted renderer chosen by `artifact_type` and the
 * raw record always one control away.
 *
 * Formatted is the default and raw is the escape hatch, never the other way round. The registry
 * falls back rather than throwing: the platform adds artifact types, and a viewer that only
 * understood a fixed list would show a blank panel for the newest and most interesting thing in
 * the feature.
 */

/**
 * What a renderer may need beyond the payload.
 *
 * One field, and it exists for one renderer: a design snapshot's previews are rendered by the
 * server on demand, and asking for one needs the feature the snapshot belongs to. Optional, so
 * every other renderer is untouched and a viewer that has no feature in hand still works.
 */
export interface ArtifactRenderContext {
  featureId?: string;
}

export type ArtifactRenderer = (
  payload: Record<string, unknown>,
  context?: ArtifactRenderContext,
) => ReactNode;

export function ArtifactViewer({
  artifact,
  renderers,
  featureId,
}: {
  artifact: Artifact;
  renderers: Record<string, ArtifactRenderer>;
  /** The feature this artifact belongs to, where the caller knows it. */
  featureId?: string;
}) {
  const [mode, setMode] = useState<'formatted' | 'raw'>('formatted');
  const renderer = renderers[artifact.artifact_type];

  return (
    <article className="artifact">
      <header className="artifact__header">
        <div className="stack stack--tight" style={{ minWidth: 0 }}>
          <h3>{humanise(artifact.artifact_type)}</h3>
          <span className="subtle">
            <CopyValue value={artifact.artifact_id} label="Copy artifact ID" />
          </span>
        </div>
        <Segmented
          label="Artifact view"
          value={mode}
          options={[
            { value: 'formatted', label: 'Formatted' },
            { value: 'raw', label: 'Raw' },
          ]}
          onChange={setMode}
        />
      </header>

      <DetailList narrow>
        {artifact.workflow_id ? <DetailRow label="Workflow">{artifact.workflow_id}</DetailRow> : null}
        <DetailRow label="Producer">{artifact.producer.replace(/_/g, ' ')}</DetailRow>
        <DetailRow label="Produced">{absoluteTime(artifact.timestamp)}</DetailRow>
        <DetailRow label="Schema">{artifact.schema_version}</DetailRow>
        <DetailRow label="Validation">{artifact.validation_status}</DetailRow>
        {typeof artifact.metadata.repository_id === 'string' ? (
          <DetailRow label="Repository">{artifact.metadata.repository_id}</DetailRow>
        ) : null}
      </DetailList>

      {mode === 'raw' ? (
        <RawJson value={artifact.payload} />
      ) : renderer ? (
        renderer(artifact.payload, { featureId })
      ) : (
        <UnknownArtifact payload={artifact.payload} />
      )}
    </article>
  );
}

function UnknownArtifact({ payload }: { payload: Record<string, unknown> }) {
  return (
    <div className="stack stack--tight">
      <p className="muted">
        No formatted view for this artifact type yet; showing its contents directly.
      </p>
      <RawJson value={payload} />
    </div>
  );
}
