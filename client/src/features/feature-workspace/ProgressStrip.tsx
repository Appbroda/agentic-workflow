import { Fragment } from 'react';
import type { Feature, Workstream } from '@/schemas/feature';
import { useArtifactList, useIntegrationReview } from './hooks';
import { buildStages } from './stages';

/**
 * Where the feature has got to, in one line.
 *
 * The full map with a branch per repository is on the Overview; this is the version that
 * belongs beside the title, because "which of the six stages are we in" is the question a
 * product manager asks first and it should not cost a scroll.
 *
 * The stages are derived from the feature's own artifacts and workstream statuses, so a
 * one-repository feature and a five-repository one both draw honestly.
 */
export function ProgressStrip({
  featureId,
  feature,
  workstreams,
  at,
}: {
  featureId: string;
  feature: Feature;
  workstreams: Workstream[];
  at: number | null;
}) {
  const artifacts = useArtifactList(featureId, undefined, at);
  const integrationReview = useIntegrationReview(featureId, at);
  if (!artifacts.data) return null;

  const stages = buildStages(
    feature,
    workstreams,
    new Set(artifacts.data.artifacts.map((item) => item.artifact_type)),
    integrationReview,
  );

  return (
    <div className="progress-strip" aria-label="Workflow progress">
      {stages.map((stage, index) => (
        <Fragment key={stage.id}>
          {index > 0 ? (
            <span className="progress-strip__sep" aria-hidden="true">
              ›
            </span>
          ) : null}
          <span
            className={`progress-strip__step progress-strip__step--${stage.state}`}
            // "Technical PRD ○" and "Technical PRD ○ queued" are different facts: the second
            // says the platform has the work and has not started it.
            title={stage.note ?? undefined}
          >
            <span aria-hidden="true">{MARK[stage.state]}</span>
            {stage.label}
            {stage.note ? <span className="subtle"> · {stage.note}</span> : null}
          </span>
        </Fragment>
      ))}
    </div>
  );
}

/* Shape as well as colour: this line has to survive a greyscale screenshot. */
const MARK: Record<string, string> = {
  done: '✓',
  active: '●',
  stopped: '✕',
  pending: '○',
};
