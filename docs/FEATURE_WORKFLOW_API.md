# Feature Workflow API

All endpoints require `Authorization: Bearer <token>` — a session token from
`POST /auth/login`, or an API token an administrator issued. The deployment's shared
`PLATFORM_API_KEY` still authenticates if one is set, and unset is the target state; see
[Authentication and workspaces](AUTHENTICATION_AND_WORKSPACES.md).

**Every `/features/*` endpoint is owner-scoped.** `GET /features` returns only your own
features; a feature in somebody else's workspace answers `404` with the same body as a
feature that never existed, on every read and every mutation. An administrator reads across
workspaces and cannot act in one that is not theirs. Provider tokens are headers only
and are needed only by `live` state-changing operations.

| Operation | Endpoint |
| --- | --- |
| Start parent feature | `POST /features/start` |
| Resume PM clarification | `POST /features/{feature_id}/resume` |
| Cancel parent feature | `POST /features/{feature_id}/cancel` |
| Read parent state | `GET /features/{feature_id}` |
| Read all artifacts | `GET /features/{feature_id}/artifacts` |
| Read child workstreams | `GET /features/{feature_id}/workstreams` |
| Read execution transitions | `GET /features/{feature_id}/executions` |
| Read coordinated PRs | `GET /features/{feature_id}/pull-requests` |
| Read timeline | `GET /features/{feature_id}/timeline` |
| Approve contract change | `POST /features/{feature_id}/contract-change-requests/{request_id}/approve` |
| Reject contract change | `POST /features/{feature_id}/contract-change-requests/{request_id}/reject` |
| What this identity must configure | `GET /setup` |
| Resolved model roles, read-only | `GET /model-configuration` |
| Saved repositories | `GET`/`POST` `/repositories`, `PUT`/`DELETE` `/repositories/{configuration_id}` |

`POST /features/start` accepts `feature_id` (optional), `prd`, `repositories`, `execution_mode`,
and `idempotency_key` (optional). Prefer the `Idempotency-Key` request header. Reusing a key with
the same body returns the original feature; using it with different data returns `409`.

**It returns before anything runs.** The feature, its reference and one durable queue entry
commit together, and the response comes back with `status: "pending"` — accepted and queued.
A worker claims the entry afterwards under a lease, so a process that dies mid-run leaves work
another one picks up rather than a feature stuck where it stopped. Poll the feature, its events,
or the event stream to watch it progress.

Two consequences worth stating:

* A `live` start needs its provider credentials **stored** against the calling identity, not
  supplied as headers. Execution outlives the request, so there is no header left to read, and
  a secret must not be copied into a queue row. `GET /setup` reports what is missing; a live
  start without them is refused with `422` naming the providers.
* Every feature carries `reference`, the identity people use: `AB-Feature-42`. It is allocated
  by the database at creation, is immutable, and is what pull-request titles are prefixed with.
  The internal `feature_id` remains the API path segment and the durable key.

Each `RepositorySpec` requires a unique `repository_id`, display `name`, `role`, GitHub
`repository_url`, and `default_branch`. `role` may be `frontend`, `backend`, `service`, `shared`,
`infrastructure`, or `other`; omitted, it defaults to `other` and the planner decides. Saved
repository configurations carry a separate user-chosen `repository_type` label, which is
organisational metadata and is deliberately never sent as `role`. Set `required: false` for non-blocking workstreams and optionally
set `implementation_order` to form dependency phases. Repository URLs must be token-free: URL
userinfo, query parameters, and fragments are rejected before feature state is created.

`GET /features/{feature_id}/workstreams` reports the isolated child ID, branch, workspace,
retry count, review/code artifact references, blockers, and PR artifact reference. Poll the parent
or this endpoint at a bounded interval; no websocket is exposed.

## Execution transitions

`GET /features/{feature_id}/executions` answers a different question from every other read:
not *where* a feature is, but *what moved it there*. One record per transition between two
stages, for the whole feature in one request — a five-repository feature on its fourth attempt
has well over a hundred, and a request per transition would make watching a feature cost more
than running it.

Each record names its handler, and the distinction is load-bearing:

| `handler_type` | What it means | Example `handler` |
| --- | --- | --- |
| `model` | A configured model performed this. `model`, `provider`, `reasoning_effort` and `model_role` describe which. | `Engineer`, `Code reviewer` |
| `deterministic` | No model was involved. | `Validator`, `Orchestrator`, `GitHub`, `Contract validator` |
| `human` | A person has to act, and no model is named for their decision. | `Human action` |

`model_resolved` is `false` where the platform has not selected a model for the transition yet.
A client must say so rather than predict one: model routing is a server decision and there is
no second copy of that policy anywhere.

Three rules the derivation follows, all of them observable:

* **Historical accuracy.** A completed transition's model comes from the immutable artifact that
  execution wrote, never from current configuration. Changing `OPENAI_CODING_MODEL` does not
  change what last week's attempt ran on.
* **One record per attempt.** `execution_id` distinguishes attempts between the same two stages,
  so a workstream that succeeded on its fourth try still publishes the three before it, each
  with its own model, revisions, failure and remediation.
* **No retry is promised that will not run.** Where the reliability logic refused another
  attempt, the record is `status: "needs_human"` with the platform's own refusal reason —
  never `Retry 4 / 8` on the strength of remaining budget.

A retry record additionally carries `failure_summary` (why the previous attempt did not pass)
and `remediation_summary` (what the next one was asked to change). They are separate answers:
a repository whose checked-in lint configuration is broken *fails* as a lint error and is
*remediated* by repairing the repository. Both are read from records the workflow already
persists — the review finding, the retry plan, the failure classification — and neither is
model reasoning, which the platform does not record.

## Authentication and accounts

| Method | Path | Auth | What it does |
| --- | --- | --- | --- |
| `POST` | `/auth/login` | none | Body `{email, password}`. Returns `{token, expires_at, actor, must_change_password}`. The token is returned exactly once; the platform stores only a digest. |
| `POST` | `/auth/logout` | bearer | Revokes the presenting token. Idempotent, always `204`. |
| `POST` | `/auth/password` | bearer | Body `{current_password, new_password}`. Revokes this account's other **sessions** and leaves its `api` tokens working. |
| `GET` | `/me` | bearer | Adds `subject` (the login email) and `must_change_password` to the existing fields. |
| `GET` | `/users` | `user:manage` | Adds `subject`, `last_login_at`, `has_password`, `must_change_password`. Never a password, never a token. |
| `POST` | `/users` | `user:manage` | Accepts an optional `password`. Normalises `subject` to lowercase. `409` on a duplicate subject. |
| `PATCH` | `/users/{user_id}` | `user:manage` | Change `display_name`, `roles`, `disabled`. `409` if it would leave the deployment with no enabled administrator. |
| `POST` | `/users/{user_id}/password` | `user:manage` | An administrator sets a password. Revokes every session that account had, forces a change, returns no password. |

A failed login is one `401` with one body for an unknown email, a wrong password, an account
with no password and a disabled account. `POST /auth/login` is rate limited per source address
and per email and answers `429` with `Retry-After`.

## What changed for existing clients

* `GET /features` returns only the caller's own features. An administrator sees all.
* Every `/features/{id}/...` read and mutation answers `404` for a feature in another
  workspace, byte-identical to a feature that does not exist.
* `GET /features/operations/unresolved` is owner-scoped. An administrator gets everything,
  including rows whose `feature_id` is null; anybody else gets only their own features'
  operations.
* `Idempotency-Key` is namespaced per account. A key recorded before this change no longer
  matches, so a submission replayed across the deploy creates a new feature rather than
  returning the old one — the safe direction, since the alternative was two people sharing a
  key (or an identical PRD, which needs no key at all) and being handed each other's feature.
* `/workflow/*` requires an administrator and is deprecated.
* `PUT /slack-configuration` and `PUT /design-source` require an administrator. This is a
  narrowing: an operator held both before.
