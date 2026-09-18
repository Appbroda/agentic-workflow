# User Guide

## Starting the platform

Copy `.env.example` to `.env`, set both model identifiers, and set
`BOOTSTRAP_ADMIN_PASSWORD` for the first boot. Then start the full local stack:

```sh
docker compose up --build
```

This starts PostgreSQL and Redis, applies Alembic migrations, and serves the API at
`http://localhost:8000`. It runs the durable PostgreSQL-backed workflow control plane. Use
`docker compose --profile dev up api-dev postgres redis` for an autoreloading API on port `8001`.
Health checks are public at `/healthz` and `/readyz`; readiness returns `503` if PostgreSQL or
Redis is unavailable, migrations are pending, or live-operation recovery remains unresolved. All
workflow endpoints require platform authentication.

## API usage

## Signing in

Every person has an account with a password. Open the console, sign in with your email, and
you will be asked to choose your own password if an administrator set the first one for you.

Your workspace is yours: your features, your provider keys, your saved repositories and your
model setups. Nobody else can see them, and asking for somebody else's feature by its
identifier is answered exactly as a feature that does not exist. An administrator can read
across workspaces -- the disaster-recovery runbook needs that -- and cannot retry, publish or
retire anybody else's work, because doing so would spend your money and push with your token.

For scripts, ask an administrator for an API token from the People page. It is shown once.
Send it as `Authorization: Bearer <token>`.

`/workflow/*` -- the original single-repository surface -- is deprecated and now requires an
administrator. Use `/features/*`, which is workspace-scoped.

## API usage against `/workflow/*` (deprecated, administrators only)

Send an administrator's token as `Authorization: Bearer <token>`. A workflow start requires a
stable workspace descriptor and a structured PRD. The workspace path must be a new absolute child
of the configured `WORKSPACE_ROOT` (the Compose default is `/workspaces`), and the working branch
must differ from the default branch. Repository URLs must not contain userinfo, query parameters,
or fragments; provider credentials belong only in request headers. The runner clones the repository
and creates that branch.

For a feature workstream, the status response includes detected repository technology, the selected
validation plan, current revision validation evidence, superseded validation count, and the scoped
and out-of-scope requirement IDs. A failed child retry affects only that repository.

```sh
curl --request POST http://localhost:8000/workflow/start \
  --header "Authorization: Bearer $ADMIN_TOKEN" \
  --header "X-OpenAI-Api-Key: $OPENAI_API_KEY" \
  --header "X-GitHub-Token: $GITHUB_TOKEN" \
  --header "Idempotency-Key: launch-20260801-001" \
  --header "Content-Type: application/json" \
  --data '{
    "workflow_id": "catalog-search-001",
    "workspace_descriptor": {
      "workspace_id": "catalog-search",
      "root_path": "/workspaces/catalog-search-001",
      "source_repo_url": "https://github.com/example/catalog.git",
      "default_branch": "main",
      "working_branch": "workflow/catalog-search-001"
    },
    "prd": {
      "title": "Catalog search",
      "problem_statement": "Customers need to find products quickly.",
      "goals": ["Provide ranked product search."],
      "user_stories": [{
        "story_id": "story-search",
        "persona": "Customer",
        "need": "Search the catalog",
        "benefit": "I can find a product",
        "acceptance_criteria": ["Results are returned for a query."]
      }],
      "requirements": [{
        "requirement_id": "requirement-search",
        "description": "Expose product search.",
        "priority": "must",
        "acceptance_criteria": ["A query produces ranked results."],
        "dependencies": []
      }],
      "constraints": ["Keep credentials out of workflow data."],
      "out_of_scope": ["Catalog migration."],
      "stakeholders": ["Commerce team"]
    }
  }'
```

`Idempotency-Key` is optional but recommended. Repeating a key with exactly the same request
returns the original workflow; reusing it with different data returns `409 Conflict`.

Use the returned `workflow_id` with these authenticated endpoints:

| Operation | Endpoint |
| --- | --- |
| Read current state | `GET /workflow/{workflow_id}` |
| Read versioned handoffs | `GET /workflow/{workflow_id}/artifacts` |
| Read structured execution events | `GET /workflow/{workflow_id}/logs` |
| Read merged chronological history | `GET /workflow/{workflow_id}/timeline` |
| Stop active work | `POST /workflow/cancel` with `{"workflow_id":"..."}` |

When the workflow reports `waiting_for_human`, submit exactly one non-empty answer for every
unresolved question:

```sh
curl --request POST http://localhost:8000/workflow/resume \
  --header "Authorization: Bearer $ADMIN_TOKEN" \
  --header "X-OpenAI-Api-Key: $OPENAI_API_KEY" \
  --header "X-GitHub-Token: $GITHUB_TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "workflow_id": "catalog-search-001",
    "answers": [{"question_id": "database", "answer": "Use PostgreSQL."}]
  }'
```

## Provider token options

A personal platform token identifies its user. The shared platform key remains an explicit
administrative compatibility credential. Live workflows need provider credentials supplied in
request headers or configured for that identity in Settings. They never belong in request JSON,
artifacts, checkpoints, logs, URLs, Git remote URLs, or browser storage:

| Header | Purpose |
| --- | --- |
| `X-OpenAI-Api-Key` | A request-scoped OpenAI credential for the submitted operation. |
| `X-GitHub-Token` | A request-scoped GitHub credential for the submitted operation. |

Request header values exist only while the request is being handled. Optional stored credentials
are sealed at rest and resolved server-side for their owning identity; only configured/not
configured descriptors are returned. Do not send provider tokens on `GET` requests.

### What a GitHub token has to be able to do

The platform clones, pushes a branch, and opens a pull request, so a token that only reads is
not enough. `PUT /credentials/github` asks GitHub before it stores anything and refuses a token
GitHub answered no about — an expired or revoked one, a classic PAT with neither the `repo` nor
the `public_repo` scope, or one that reaches no repositories at all. A GitHub that could not be
reached refuses nothing.

* **Classic PAT** — grant `repo` (or `public_repo` if you only build in public repositories).
  Add `workflow` if features may change files under `.github/workflows/`; without it GitHub
  rejects the whole push, whatever the `repo` scope says.
* **Fine-grained PAT** — name each repository the token may use, with **Contents** and **Pull
  requests** set to read and write. GitHub does not publish a fine-grained token's own
  permissions, so nothing can verify those two before the first push; what the platform lists is
  the account's role in each repository. A repository the token does not name simply does not
  appear.

`GET /credentials/github/repositories` lists what the stored token reaches, which is what the
console's repository picker offers, and `POST /repositories` refuses a github.com repository that
listing does not show as writable.

## UI integration

1. Give each user a personal platform token. Use the shared platform key only for documented
   administrative compatibility, and never bake either token into a browser bundle.
2. Submit the start payload from that server, generating an idempotency key per user action.
3. Render `status`, `current_agent`, `approval_state`, and `retry_count` from
   `GET /workflow/{workflow_id}`. Poll at a bounded interval; the API does not expose a
   websocket stream.
4. If the status is `waiting_for_human`, fetch the technical PRD from the artifacts endpoint and
   render its `unresolved_questions`. Submit the answers as one resume request.
5. Use the timeline endpoint for an audit view and the pull-request artifact for the final PR
   link. Treat artifact payloads as structured data, not HTML, and escape any text displayed in
   the UI.

The deployed application uses a PostgreSQL control plane. Workflow records, artifacts, lifecycle
events, and idempotency records survive API restarts; provider tokens do not. See the [deployment guide](DEPLOYMENT.md) before exposing the API to users.

## Cancelling live work

`POST /features/{feature_id}/cancel` accepts an optional `{"reason":"..."}` body. The response
may report cancellation requested or in progress before it reaches a terminal cancelled state.
Completed local Git work and remotely created PRs remain visible as cleanup requirements; the
platform never deletes them automatically. `POST /workflow/cancel` similarly stops future work and
interrupts local Git/validation subprocesses. See [the cancellation model](CANCELLATION_MODEL.md).

## Frontend and backend feature delivery

For one feature spanning multiple repositories, use `POST /features/start`, not two unrelated
`/workflow/start` calls. The parent runs clarification and planning once, approves a shared API
contract, then creates isolated child branches and workspaces. Required workstreams may run in
parallel only after that contract is approved. See [multi-repository workflows](MULTI_REPOSITORY_WORKFLOWS.md)
for the request format and [feature API](FEATURE_WORKFLOW_API.md) for all read/resume endpoints.
