import type { Branch, Stage } from './stages';

/**
 * Where a feature has got to, drawn from its own state.
 *
 * The fan-out stage lists one branch per repository the feature actually has, in the order the
 * plan gave. There is no frontend/backend pair in this component -- a one-repository feature
 * draws one branch and a five-repository feature draws five.
 *
 * Deliberately a list and not a canvas: this is read to answer "where did it stop", which a
 * labelled sequence answers as well as a diagram would and a screen reader can follow.
 */
export function WorkflowMap({ stages }: { stages: Stage[] }) {
  return (
    <ol className="flow" aria-label="Workflow progress">
      {stages.map((stage) => (
        <li key={stage.id} className={`flow__stage flow__stage--${stage.state}`}>
          <div className="flow__row">
            <span className="flow__marker" aria-hidden="true" />
            <span className="flow__label">{stage.label}</span>
            {/* The note wins where there is one: "queued" says the platform has this work
                and has not begun it, which "not started" does not distinguish from a stage
                three steps ahead that nothing has reached. */}
            <span className="flow__state">{stage.note ?? STATE_WORDING[stage.state]}</span>
          </div>
          {stage.branches && stage.branches.length > 0 ? (
            <ul className="flow__branches" aria-label={`${stage.label} by repository`}>
              {stage.branches.map((branch) => (
                <BranchRow key={branch.repositoryId} branch={branch} />
              ))}
            </ul>
          ) : null}
        </li>
      ))}
    </ol>
  );
}

const STATE_WORDING: Record<Stage['state'], string> = {
  done: 'done',
  active: 'in progress',
  pending: 'not started',
  stopped: 'stopped',
};

function BranchRow({ branch }: { branch: Branch }) {
  return (
    <li className={`flow__branch flow__branch--${branch.state}`}>
      <span className="flow__label">{branch.label}</span>
      {/* The server's own status word, not a translation of it: the vocabulary is the
          platform's and it grows. The note appears only when the reading differs, so the
          record is never quietly rewritten. */}
      <span className="flow__state">
        {branch.status}
        {branch.note ? ` — ${branch.note}` : ''}
      </span>
    </li>
  );
}
