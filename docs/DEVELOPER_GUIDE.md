# Developer Guide

## Development workflow

Install the locked environment with `uv sync`, then run the quality gate:

```sh
uv run --directory server ruff check .
uv run --directory server ruff format --check .
uv run --directory server mypy .
uv run --directory server pytest
```

The client has its own four, run from `client/`:

```sh
npm run lint
npm run typecheck
npm run test
npm run build
```

Install the same checks as local Git hooks with `uv run --directory server pre-commit install`. The hook definition
uses the project's locked `uv` environment, so it does not download unreviewed hook runtimes.

## Adding an agent

Agents communicate exclusively through validated artifacts. Add an agent by:

1. Defining any new strict artifact schema in `artifacts/schemas.py`, including the inherited
   envelope fields through `BaseArtifact` and cross-reference validators.
2. Adding a stable artifact filename to `agents/shared/contracts.py` when the artifact becomes a
   workflow handoff.
3. Creating an agent module with an injected protocol boundary. Its `run(state)` method must
   consume required artifacts with `require_artifact`, render a versioned prompt, validate model
   output, and return `artifact_update`.
4. Giving the agent a place in `workflows/feature_workflow.py`, which is the one orchestrator.
   Keep every retry bounded by the authority that already owns it -- see
   [retry strategy](RETRY_STRATEGY.md) -- and add coverage before changing a transition.
5. Extending `configs/agent.yaml` for model selection, timeout, and retry policy. Every
   model-backed workflow agent names a logical `model_role` and takes its resolved model and
   effort from the central role configuration. Never read a model name in agent or workflow code. See
   [model routing](MODEL_ROUTING.md).
6. Writing independent mock-only tests under `tests/` plus orchestration-level coverage for the
   new transition.

Do not pass conversational strings between agents, accept provider tokens in artifacts, or let an
agent mutate paths outside `WorkspaceDescriptor.root_path`.

## Adding a tool or external service

Put reusable workspace-safe logic in `tools/`. Put network-capable or SDK-specific code behind a
Protocol in `adapters/`, alongside a deterministic mock implementation. Inject that protocol into
the agent or service constructor; never construct a live client inside workflow code.

Tools that spawn processes must use fixed argument vectors, a workspace-local `cwd`, explicit
timeouts, and `shell=False`. Git operations must preserve the default-branch and force-push
safety rules in `adapters/git_adapter.py`. Add tests that prove both the normal operation and its
security boundary.

For any new live side effect, use `ExternalOperationExecutor` rather than calling a provider or
subprocess directly. Supply only credential-free fingerprints/metadata, a deterministic logical
step, and a cancellation token. The executor commits intent before execution and stores a terminal
result before workflow state advances. Add a recovery test for a crash after the operation and a
cancellation test that proves no later side effect starts.

## Adding a prompt version

Prompts live under `prompts/<agent>/` and are rendered only through `PromptLoader`. To revise one:

1. Add a new immutable template such as `prompts/planner/v2.jinja2`; do not overwrite a version
   used by persisted artifact metadata.
2. Pass only explicit structured context to the template and rely on `StrictUndefined` to catch
   omissions.
3. Update the agent's configured template reference and record it in artifact metadata.
4. Add rendering tests for the version and agent tests for the expected structured output.

Prompt templates specify output contracts, but Pydantic validation remains the authority at the
agent boundary. Never trust a model response until it has been parsed and validated.

## Runtime and deployment boundaries

The Docker image is a non-root Python runtime with Git installed for repository actions. Docker
is deliberately not installed in the image; the Compose or EC2 host owns container orchestration.
Before a Compose build, export `BUILD_REVISION="$(git rev-parse --verify HEAD)"` and keep
`WORKFLOW_SCHEMA_VERSION` aligned with `workflow_schema.WORKFLOW_SCHEMA_VERSION`. Compose passes
the revision as both an image build argument and the controller's expected runtime identity.
`docker compose up --build` starts PostgreSQL, Redis, migrations, and the API. The `api-dev`
Compose profile bind-mounts source code and enables Uvicorn reload.

`main.py` starts `SqlAlchemyFeatureControlPlane` with a Redis lifecycle lock and
`ProductionFeatureRunner`. The `/workflow/*` routes are served from the same control plane through
`api/workflow_control_plane.py`; an isolated application runs that same durable plane against a
private SQLite database rather than a second implementation of the lifecycle. Provider headers must
never be persisted by any implementation. `WorkspaceProvisioner` accepts only token-free GitHub
HTTPS URLs and clones them beneath `WORKSPACE_ROOT`; `EngineerAgent` commits and pushes the
configured working branch before `GitHubAgent` creates the PR.

Changes to this path require tests for durable restart behavior, workspace containment, credential
non-persistence, and readiness failure. Follow [the deployment guide](DEPLOYMENT.md) for the
staging workflow required before release.

## Multi-repository features

`workflows/feature_workflow.py` is the additive parent orchestrator. It owns the one PM/planner
pass, contract gate, parallel child scheduling, integration review, and coordinated PR publication.
It reuses `EngineerAgent` and `ReviewerAgent` through repository-specific child state in
`services/feature_runtime.py`; do not create frontend/backend-specific coding personalities.

New feature artifacts are numbered `009` through `014`. The `SqlAlchemyFeatureControlPlane` stores
parent state plus independently queryable repository, child, contract, review, PR, event, and
idempotency records. Keep credentials request-scoped. Use `MockContractCodeGenerator` and mock
mode for unit tests; live staging is covered in [multi-repository workflows](MULTI_REPOSITORY_WORKFLOWS.md).

## Repository-aware validation

Use `WorkspaceValidationTools.run_validations*`, not direct Python tool assumptions, for a cloned
repository. It detects manifests/scripts, fingerprints the checkout before every command, and saves
only safe output summaries in the journal. See [repository validation](REPOSITORY_VALIDATION.md).

Live children must run `RepositoryPreflight` before invoking `EngineerAgent`. Do not add dependency
installation back into the validation plan: the preflight owns deterministic Node bootstrap and
records any unsafe setup defect. Extend the repository execution plan with explicit implementation
expectations, then run `validate_implementation_completeness` before repository review. Retry
changes must be built through the failure classifier and retain independent setup, validation,
implementation, and integration counters. See [retry strategy](RETRY_STRATEGY.md).

Use [the recovery guide](RECOVERY_AND_IDEMPOTENCY.md) before changing operation status or restart
logic. Never convert an `UNKNOWN_EXTERNAL_STATE` into a retry without provider reconciliation.
