# Deployment Guide

## What the production stack runs

Build only from a committed checkout whose `git status --porcelain` output is empty; otherwise a
Git SHA cannot identify the files copied into the image. Set the immutable identity from that exact
checkout, then build and start the release:

```sh
export BUILD_REVISION="$(git rev-parse --verify HEAD)"
export WORKFLOW_SCHEMA_VERSION="1.0"
docker compose up --detach --build --force-recreate
```

`docker compose up --build` starts PostgreSQL, Redis, a one-shot Alembic migration job, a
workspace-permission initializer, and the API. The API refuses to become ready until PostgreSQL
and Redis answer health probes. Production startup also refuses `unknown`, non-Git, or mismatched
build identity and a mismatched workflow schema. Compose bakes `BUILD_REVISION` into every platform
image and passes it independently as `EXPECTED_BUILD_REVISION`; it maps
`WORKFLOW_SCHEMA_VERSION` to `EXPECTED_WORKFLOW_SCHEMA_VERSION`. A direct host deployment must
export both `EXPECTED_*` variables explicitly. It stores control-plane state in PostgreSQL and uses Redis
to serialize idempotency and lifecycle operations.

Every live workflow must provide these headers to both `/workflow/start` and `/workflow/resume`:

| Header | Required capability |
| --- | --- |
| `X-OpenAI-Api-Key` | Access to every model this deployment configures for a role. |
| `X-GitHub-Token` | Read/write repository contents, create branches, push commits, and create pull requests. |

The runner never writes those values to the database, artifacts, logs, Git remote URL, or
configuration files. Git receives the token through a temporary mode-0700 askpass script that is
removed at the end of the request.

The same headers apply to `live` `/features/start`, `/features/{feature_id}/resume`, and contract
change approval. A feature token must access every listed repository. Test coordinated frontend and
backend PR creation only in a disposable staging GitHub organization first. The platform creates
draft PRs by default and never merges them automatically.

Live repository processes are not yet isolated into per-job containers or namespaces. Until the
stop-ship isolation controls in [the production-readiness audit](PRODUCTION_READINESS.md) are
implemented, accept only explicitly trusted disposable staging repositories; do not expose live
mode to arbitrary or multi-tenant repository input.

## EC2 prerequisites

Provision an EC2 instance or Auto Scaling group with:

- Docker Engine and Docker Compose v2;
- an encrypted EBS volume mounted for `/workspaces` and sufficient space for concurrent clones;
- private connectivity to PostgreSQL and Redis, or the supplied Compose services for a small
  single-instance installation;
- a TLS-terminating reverse proxy/load balancer with an explicit request timeout suitable for
  long-running coding workflows;
- security groups that expose only the proxy port and keep PostgreSQL/Redis private; and
- an instance role permitted to read `BOOTSTRAP_ADMIN_PASSWORD` from AWS Secrets Manager for
  the first boot of a new deployment. Do not put it in the image, repository, or AMI, and
  remove it from the environment afterwards -- see "First boot" below.
- the same, for `PLATFORM_API_KEY`, **only if** this deployment still sets one. It is
  optional now and unset is the target state; see
  [Authentication and workspaces](AUTHENTICATION_AND_WORKSPACES.md).

Set `OPENAI_REASONING_MODEL`, `OPENAI_REASONING_EFFORT`, `OPENAI_CODING_MODEL`,
`OPENAI_REASONING_TIMEOUT_SECONDS`,
`OPENAI_CODING_REASONING_EFFORT`, `OPENAI_REVIEW_MODEL`,
`OPENAI_REVIEW_REASONING_EFFORT`, `OPENAI_SCOPED_FIX_MODEL` and
`OPENAI_SCOPED_FIX_REASONING_EFFORT` through the encrypted deployment environment. The checked-in
`.env.example` contains the intended defaults. Also set database/Redis URLs and
`CONTAINER_WORKSPACE_ROOT` (normally `/workspaces`) there. Provider API tokens belong to user
requests, not deployment configuration. For direct host runs, use the separately configured
absolute `WORKSPACE_ROOT` value.

`SECRET_ENCRYPTION_KEY` is effectively required for a multi-user deployment. Without it there
is no credential store, so nobody can save a provider key and every feature falls back to
request headers -- which queued work cannot read, because it runs after the request was
answered. Generate one with `services.secrets.generate_encryption_key`.

`PLATFORM_API_KEY` is optional and **unset is the target state**. Anybody holding it is an
unnamed administrator with every permission, and every person has an account now. Leave it
unset and keep an admin API token, issued in advance, where the key used to be kept: it can be
revoked and it names somebody.

## First boot of a new deployment

1. Apply the migrations. `0031` creates the administrator's account with **no password**.
2. Set `BOOTSTRAP_ADMIN_PASSWORD` in the environment (and `BOOTSTRAP_ADMIN_EMAIL` if the
   administrator is not `akhilesh@appbroda.com`). Start the API. It sets the password once and
   logs `administrator_password_bootstrap outcome=set`.
3. Sign in to the console with that email and password. It will require a new password
   immediately, because a password the deployment's environment knows is a handover
   credential.
4. **Remove `BOOTSTRAP_ADMIN_PASSWORD` from the environment.** It stops working as soon as the
   administrator picks their own, and the bootstrap never overwrites an existing hash -- so
   forgetting is survivable rather than dangerous. Remove it anyway.
5. Create accounts from the console's People page. No database or environment edit is needed
   for any of them.

The bootstrap is idempotent and safe to leave in place: on every later boot it logs
`outcome=already_set` and changes nothing.

An environment with only the two original model variables still starts: review falls back to the
reasoning model and scoped fix falls back to coding. Set all four roles explicitly for independent
routing. Startup logs `model_roles_resolved`, naming every resolved model and effort and which
roles still use a fallback. See [model routing](MODEL_ROUTING.md).

Set `ALLOWED_HOSTS` to a JSON array of the externally served API hostnames, for example
`["api.staging.example.com"]`. The production Compose service fails settings validation without it;
the development profile remains restricted to local hosts by the checked-in policy.

The supplied Compose file intentionally binds PostgreSQL and Redis to `127.0.0.1` only; they are
available to Compose services over the internal network and to an operator on the EC2 host, but are
not publicly reachable. For a
multi-instance EC2 deployment, replace their connection values with RDS and ElastiCache endpoints,
retain the migration job, and mount the same durable workspace volume only where the worker that
owns the clone executes. Do not run more than one API worker against a local named workspace
volume.

## Staging validation before release

Use a dedicated staging GitHub organization and a disposable repository. Never use a production
default branch for this validation.

1. Inject a staging platform key through Secrets Manager and start the stack with `docker compose
   up --build`.
2. Wait for `GET /readyz` to return `200`, confirm its `build_revision` exactly equals
   `$BUILD_REVISION` and its `workflow_schema_version` equals `$WORKFLOW_SCHEMA_VERSION`, and
   confirm the migration job exited successfully. A `runtime_compatible: false` response is a
   stop-ship condition; mutating endpoints return `503` while read-only diagnostics remain usable.
3. Submit a representative PRD with a new `/workspaces/<workflow-id>` root, a staging OpenAI key,
   and a least-privileged staging GitHub token.
4. Confirm the workflow writes only inside that directory, creates the requested working branch,
   commits it, pushes it, and opens a PR against the configured non-production default branch.
5. Restart the API after a human-clarification pause, resubmit the request-scoped headers on
   `/workflow/resume`, and confirm the workflow completes from its persisted feature state.
6. Inspect PostgreSQL records, API logs, Git remote configuration, and the PR body to confirm that
   neither provider token appears. Revoke the staging token after the test.

## Rollback and incident response

Deploy immutable image tags. Roll back by selecting the prior verified image tag, then run Alembic
only in the forward direction unless a migration has an explicitly tested downgrade. Revoke a
suspected provider token immediately, cancel affected workflows through the API, and retain the
workflow timeline and PR artifacts for investigation.

To lock an account out: disable it from the console's People page, which stops its logins and
revokes its sessions on the next request. To rotate its password instead, reset it from the
same page -- every session that account held stops working and it must choose a new one. If
this deployment still sets `PLATFORM_API_KEY`, rotate it through Secrets Manager and restart
the API instances; the better answer is to unset it and use a revocable admin token.

**A disabled account's already-queued work still runs.** The dispatcher resolves stored
credentials by id and does not authenticate. Cancel or retire the features first if that
matters.
