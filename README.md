# AI Software Engineering Platform

An artifact-driven platform that turns product requirements into reviewed software
changes through a bounded, human-supervised multi-agent workflow.

## Local development

1. Copy `.env.example` to `.env` and provide the platform key and model configuration.
2. Start PostgreSQL and Redis with `docker compose up -d`.
3. Run the API with `uv run --directory server python main.py`.

Live workflows require `X-OpenAI-Api-Key` and `X-GitHub-Token` on each start or resume request.
They are request-scoped and must not be placed in `.env` or JSON payloads.
Live mode is currently limited to trusted disposable staging repositories: repository commands
still share the API container's UID, filesystem namespace, and network. See the
[production-readiness audit](docs/PRODUCTION_READINESS.md) before enabling it.

For a coordinated frontend/backend feature, use `/features/start` with one PRD and multiple
repository specifications. It creates a shared immutable contract, isolated child workstreams, an
integration review, and linked draft PRs. The existing `/workflow/*` routes remain available for a
single repository.

The foundational health endpoints are available at `/healthz` and `/readyz`.

A browser console for non-engineers is served at `/console`. It submits a feature and reads its
progress, pull requests, and per-repository blocking issues in plain language. It is a static
page holding no secrets; every request it makes carries the operator's platform key, so the data
behind it stays behind the same authentication as the API.

## Containers

From a committed checkout with an empty `git status --porcelain`, export
`BUILD_REVISION="$(git rev-parse --verify HEAD)"`, then build and start the complete stack,
including migrations, with `docker compose up --build`. Production refuses an unknown or stale
runtime identity. `/healthz` and `/readyz` report the active build and workflow schema versions.
The API listens on `http://localhost:8000` by default. For hot reload inside a container,
run `docker compose --profile dev up api-dev postgres redis` and use port `8001`.

The runtime image installs Git for repository operations. It does not install a Docker CLI:
container orchestration remains the responsibility of the host or deployment platform.

## Documentation

- [Product manager guide](docs/PRODUCT_MANAGER_GUIDE.md) — the `/console` page, writing a feature, and reading the result.
- [User guide](docs/USER_GUIDE.md) — authenticated API usage, token handling, and UI integration.
- [Developer guide](docs/DEVELOPER_GUIDE.md) — extension points for agents, tools, and prompts.
- [Deployment guide](docs/DEPLOYMENT.md) — EC2, secret injection, staging validation, and rollback.
- [Production readiness audit](docs/PRODUCTION_READINESS.md) — verified scope, stop-ship findings, and controlled-pilot checklist.
- [Multi-repository workflows](docs/MULTI_REPOSITORY_WORKFLOWS.md) — parent features and child workstreams.
- [Model routing](docs/MODEL_ROUTING.md) — four configured roles, conservative scoped fixes, and retry-safe metadata.
- [Web control plane](docs/WEB_CLIENT.md) — running the client, its API layer, real-time updates and chat.
- [Web control plane delivery report](docs/WEB_CLIENT_DELIVERY.md) — what was built, what was verified, and what was not built.
- [Feature API](docs/FEATURE_WORKFLOW_API.md) — `/features/*` request and read endpoints.

## Repository layout

```text
server/   the Python platform: agents, workflows, API, migrations and tests
client/   the React web control plane
docs/     operator and developer documentation
```

`server/` is the Python project root: `pyproject.toml`, `uv.lock` and `alembic.ini` live there, so
imports are unprefixed (`from agents...`) and every `uv` command takes `--directory server`. The
`Dockerfile` and `docker-compose.yml` stay at the repository root because the image build verifies
the checkout's Git identity against `BUILD_REVISION` and needs `.git` in its context.
