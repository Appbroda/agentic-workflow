import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Async, EmptyState, ErrorState, TableSkeleton } from '@/components/common/States';
import { Panel } from '@/components/ui/Layout';
import { ArtifactViewer } from '@/components/artifacts/ArtifactViewer';
import { ARTIFACT_RENDERERS } from '@/components/artifacts/renderers';
import { humanise } from '@/utils/text';
import { relativeTime } from '@/utils/time';
import { useArtifact, useArtifactList } from './hooks';

/**
 * Every document the run produced, with one open beside the list.
 *
 * The selection is in the URL, which is what makes an artifact linkable -- how somebody sends
 * a colleague the exact review rather than "open the artifacts tab and scroll" -- and what
 * lets the agent history link straight to the result a run produced.
 *
 * Payloads are fetched only when opened. A completed two-repository feature has around
 * forty-six of these and roughly 400 KB of payload; the list carries envelopes only.
 */
export function ArtifactsTab({ featureId }: { featureId: string }) {
  const list = useArtifactList(featureId);
  const [params, setParams] = useSearchParams();
  const selected = params.get('artifact');
  const [type, setType] = useState('');
  const artifact = useArtifact(featureId, selected);

  const select = (artifactId: string) => {
    const next = new URLSearchParams(params);
    next.set('artifact', artifactId);
    setParams(next, { replace: true });
  };

  return (
    <div className="split">
      <Panel title="Artifacts" flush>
        <Async query={list} skeleton={<TableSkeleton rows={6} />}>
          {(data) => {
            const types = [...new Set(data.artifacts.map((item) => item.artifact_type))].sort();
            // Most of these are repeats of the same few types across attempts, so the filter
            // is the difference between a browsable list and a wall of identifiers.
            const shown = type ? data.artifacts.filter((item) => item.artifact_type === type) : data.artifacts;
            return data.artifacts.length === 0 ? (
              <EmptyState
                title="No artifacts yet"
                detail="Each agent writes its result here as it finishes."
              />
            ) : (
              <>
                <div className="toolbar">
                  <label className="filter-inline" htmlFor="artifact-type">
                    <span className="details__label">Type</span>
                    <select id="artifact-type" value={type} onChange={(event) => setType(event.target.value)}>
                      <option value="">All ({data.artifacts.length})</option>
                      {types.map((item) => (
                        <option key={item} value={item}>
                          {humanise(item)} ({data.artifacts.filter((entry) => entry.artifact_type === item).length})
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <ul className="artifact-list" aria-label="Artifacts">
                  {shown.map((item) => (
                    <li key={item.artifact_id}>
                      <button
                        type="button"
                        className={
                          selected === item.artifact_id ? 'link-button link-button--active' : 'link-button'
                        }
                        onClick={() => select(item.artifact_id)}
                      >
                        <span className="truncate">{humanise(item.artifact_type)}</span>
                        <span className="subtle truncate">{item.artifact_id}</span>
                        <span className="subtle">
                          {item.producer.replace(/_/g, ' ')} · {relativeTime(item.timestamp)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            );
          }}
        </Async>
      </Panel>

      <div>
        {selected === null ? (
          <Panel>
            <EmptyState
              title="Select an artifact"
              detail="Its contents load on demand, so opening one costs a single request."
            />
          </Panel>
        ) : artifact.isPending ? (
          <Panel>
            <TableSkeleton rows={6} />
          </Panel>
        ) : artifact.isError ? (
          <Panel>
            <ErrorState error={artifact.error} onRetry={artifact.refetch} />
          </Panel>
        ) : artifact.data ? (
          <Panel>
            <ArtifactViewer
              artifact={artifact.data}
              renderers={ARTIFACT_RENDERERS}
              featureId={featureId}
            />
          </Panel>
        ) : null}
      </div>
    </div>
  );
}
