# Production Readiness Audit — 2026-08-08

> ## Update — 2026-08-16
>
> The audit below is retained as written. It is accurate as history and stale as a status
> report: **45 commits** have landed since it was published, at `680ed32`.
>
> **What changed.** Almost all of those commits attack the audit's central finding — that
> low-level operations succeeded 98.4% of the time while live features failed 94.2%, because
> orchestration misread repository context, completeness, retry progress, or recovery. The
> largest group corrects how a repository's work is judged and retried. A second group closes
> a defect class the audit did not separate out: **reviewed work that never became a pull
> request.** That failed five ways — a sibling repository's fault aborting the publisher, an
> integration review that never converged, a blocking finding naming no owner, a rework that
> regressed an already-approved repository, and journaled operations whose *first* failure was
> recorded as terminal and so could never be retried, on that run or any resume. The rule is
> now asserted once, over every one of those routes, in `tests/test_feature_workflow.py` and
> again over the real executor and real Git in `tests/test_multi_repo_e2e.py`.
>
> A `/console` page was added so a product manager can submit a feature and read its state
> without composing JSON or interpreting lifecycle enums. It is static, holds no secrets, and
> every request it makes carries the operator's platform key.
>
> A `GET /features` listing was added, keyset-paged and bounded. There was previously no way
> to enumerate features, so a generated ID was the only route back to one.
>
> **Re-verified at this commit:** 399 tests passed, Ruff lint and format clean, strict mypy
> clean across 119 source files, working tree clean.
>
> **Deployment re-verified** in an isolated Compose project (`console-verify`, separate ports,
> network and volumes; the live stack was not touched and was left running):
>
> | Check | Result |
> | --- | --- |
> | Image build from the clean commit | Passed; `BUILD_REVISION` baked as `a42372c` |
> | Runtime image contents | `api/static/console.html` present; `.git` and `tests` absent; runs as non-root uid 999 |
> | Migration job | Exit 0 at head `20260808_0007` |
> | `/healthz` and `/readyz` | Both `ok`, exact build revision, schema `1.0`, `runtime_compatible: true` |
> | Console page and vocabulary over HTTP | 200; 16 feature and 9 workstream states served |
> | Feature data without a platform key | 401 |
> | End-to-end mock feature on PostgreSQL + Redis | Completed; both pull requests recorded; listing returned it |
>
> That last row matters: it exercised the durable SQLAlchemy control plane, not the in-memory
> one the unit tests use. Afterwards the verification project and its volumes were removed.
>
> ### Production evidence: what the last live runs actually did
>
> The live database holds 11 more live runs since the audit (`-055` through `-065`), the most
> recent finishing **2026-08-09 03:57** — before all 48 commits described above. Ten ended
> `failed_requires_human` and one is still `waiting_for_human`. Their operation funnel:
>
> | Operation | Succeeded |
> | --- | ---: |
> | clone / branch / dependency install | 18 each |
> | coding executor / file writes | 44 / 62 |
> | tests / lint / build | 34 each |
> | **commit** | **3** |
> | **push** | **3** |
> | **pull request** | **0 — never even attempted** |
>
> Two things follow, and they are the two halves of the current work.
>
> **First: three repositories passed review, committed and pushed, and got no pull request.**
> In runs `-061`, `-063` and `-065` exactly one repository reached `approved` with a pushed
> branch while its sibling failed, and no pull request was created in any of them — 0 rows in
> `feature_pull_requests`. That is not a provider fault; every push succeeded. It is the
> publication defect described above, observed three times in eleven runs, and it is the
> reason the completed work was invisible. Those three runs would each produce a pull request
> under the current code.
>
> **Second: only 3 commits came out of 18 repository attempts.** The rest never got past
> review or completeness. The recorded classifications are 6 validation-source, 4
> implementation-missing, 2 review-scope, 1 validation-configuration — the same distribution
> the audit found, and the exact set the Aug 10–13 commits target (lint budget accounting,
> wiring and reachability gates, completeness as a deterministic gate, retry-progress
> accounting). Whether those fixes move this number is the open question below, and only a
> live run answers it.
>
> **Explicitly NOT verified, and still open:**
>
> - **The live success rate has been re-measured.** See *Live pilot, 2026-08-17 to 2026-08-19*
>   below. The 1/52 figure further down predates every fix and should not be quoted as current.
> - **Blockers 2, 3 and 4 are closed.** Blocker 1 stands as written and gates untrusted
>   repositories only.
>
> **Scope decision.** This platform is being used against repositories the team owns and
> controls. Blocker 1 (per-job execution isolation) is explicitly scoped to *untrusted*
> repositories and does not gate that use; the trusted-repository rating below applies. It
> must be closed before any repository the team does not control is connected.

## Executive status

Overall status: **NOT_READY for unrestricted live execution of arbitrary repositories**.

The control plane, mock workflows, durability model, retry policy, publication gates, and runtime
identity checks are suitable for local acceptance and a controlled trusted-repository staging
pilot. Live repository commands still execute in the API container under one UID, with a shared
workspace volume and unrestricted container-network access. Environment and Git-hook credentials
are now isolated, but that is not a substitute for per-job PID/user/mount/network isolation.

| Capability | Rating | Current evidence |
| --- | --- | --- |
| Mock single/multi-repository orchestration | READY_WITH_LIMITATIONS | Deterministic workflow/API tests, durable SQLite control-plane tests, strict artifacts, and retry/resume regressions pass. |
| Trusted-repository staging pilot | READY_WITH_LIMITATIONS | Fail-closed preflight, post-review publication, operation journal, exact PR-head verification, and runtime identity are implemented. Real OpenAI/GitHub staging is still required. |
| Arbitrary or untrusted repository live mode | NOT_READY | Repository-controlled processes share the API container UID, filesystem namespace, `/proc`, cache/volume, and network. |
| Broad multi-tenant production | NOT_READY | Per-tenant authorization, pagination, rate limiting, isolated workers, and provider staging evidence remain outstanding. |

No production repository should be connected until the isolation and staging gates below are
closed. The current live Compose stack was intentionally left untouched: the working tree contains
the audit changes and its running API image/database still represent the previous release.

## Evidence investigated

This audit read the complete historical prompt set, current architecture, tests, migrations,
repository state, production PostgreSQL records, Docker images/containers/volumes, and available
logs. The durable database and workspace volume—not the host `logs/` directory—contained the
useful production evidence.

### Historical run inventory

| Record | Count |
| --- | ---: |
| Feature workflows | 54 |
| Repository selections | 107 |
| Child workflows | 89 |
| Feature artifacts | 421 |
| Feature timeline events | 752 |
| External operations / attempts / operation events | 1,286 / 1,287 / 5,151 |
| Integration contracts / reviews / contract changes | 45 / 3 / 1 |
| Pull-request records | 6 |
| Legacy single-workflow records | 0 |

Feature outcomes were 45 `failed_requires_human`, 4 `failed`, 3 `completed`, 1
`running_child_workflows`, and 1 `waiting_for_human`. Two completions were mock runs. Among the 52
numbered live runs (`-003` through `-054`), only `-054` completed: 1/52 (1.9%) succeeded and 49/52
(94.2%) explicitly failed. Child outcomes were 66 failed, 9 completed, 2 approved, 10 pending, 1
running, and 1 waiting for a contract change.

Of 55 classified child failures, 41 (74.5%) were validation-source failures, followed by 7
implementation-missing, 3 dependency-installation, 2 validation-configuration, 1 contract, and 1
test-infrastructure failure. Thirty-four older child records had no structured classification.

Low-level operations succeeded 1,265/1,286 times (98.4%), while live features explicitly failed
94.2% of the time. That contrast is the central historical finding: provider and subprocess calls
usually completed, but orchestration interpreted repository context, completeness, validation,
retry progress, or recovery incorrectly.

The Git funnel was also highly lossy: 76 clone/branch operations led to 66 commits, 55 pushes, and
only 6 PR records. Runs `-051` and `-052` created two PRs but were marked failed after later
coordination work; `-014` remained stale in child execution; `-025` remained at a human pause; and
nine live runs created no child despite selecting two repositories.

### Runtime evidence

- The production API, PostgreSQL, and Redis containers were healthy when inspected; migration was
  `20260803_0006` and the API source/image matched then-current commit `4e719153...`.
- Production mounts only the named `/workspaces` volume. It does not bind-mount source, so host
  fixes have no effect until an image is rebuilt and the API is recreated.
- The workspace volume held roughly 7.4 GiB across six retained workspaces. Repeated historical
  builds left many dangling images, making build identity ambiguous before this audit.
- Host `logs/` contained only `.gitkeep`; container logs were dominated by 10-second health-probe
  access lines and did not emit a trustworthy startup build identity.
- The production API had no restart policy. The development service declared `/workspaces` but did
  not mount the shared workspace volume.

## Root causes and implemented controls

### State, retries, and recovery

- Clarification answers now become authoritative planner input and a traceable technical-PRD
  revision. A valid answer at the final configured round is accepted.
- Child checkpoint boundary, layout evidence, and retry-refusal reason round-trip through durable
  storage. Retry counters and strategy are checkpointed before every next attempt.
- Empty resume refuses an already exhausted deterministic retry state. A crash between attempts no
  longer resets the retry budget or reruns a completed sibling.
- Every engineer, reviewer, child-result, and integration-review attempt has a distinct immutable
  identifier. Feature child IDs include the repository ID, preventing parallel sibling collisions;
  downstream artifacts reference the exact attempt they consumed.
- Feature state writes serialize locally, through the distributed lock, and on the database row.
  Repository-scoped checkpoints merge sibling progress and append-only artifacts instead of
  replacing a newer snapshot. Cancellation cannot erase a concurrent checkpoint or reopen a
  completed journal operation.
- Human clarification emits an immutable technical-PRD revision, and a resume continues from the
  feature's persisted state rather than replaying completed upstream agents. A failure before the
  first checkpoint re-enters only journaled workspace provisioning.
- Artifact revisions append with `supersedes` lineage; new duplicate artifact IDs are rejected and
  artifacts are frozen against field reassignment.
- External-operation claims are atomic. Terminal failures cannot be reopened, heartbeat/lease loss
  cancels the protected action, failed dependency installs remain retryable, and successful retry
  evidence is journaled rather than replaying a failed process result.
- Cancellation uses attempt-fenced journal transitions. An unknown remote effect remains
  `cancelled_with_external_side_effects` and retains its operation ID for reconciliation even when
  the provider never returned a URL or SHA.
- A bounded background recovery sweep revisits operations that were still fresh at startup. It
  atomically claims only expired heartbeats, survives transient sweep errors, and shuts down with
  the API lifespan. Repeating authenticated cancel after reconciliation promotes an owner to the
  explicit cleanup-required status and restores unrelated readiness.
- Redis workflow-lock renewal loss cancels the lock owner and cannot delete a replacement owner's
  key.

### Repository understanding and validation

- Repository technology, manifests, scripts, nested workspace boundaries, and nearest lockfiles
  determine validation. Python, Node, frontend, and backend checks are not applied globally.
- `uv run ... pytest` exit 5 is correctly classified as no tests, and nested pnpm/yarn workspaces
  select the ancestor manager instead of silently falling back to npm.
- Exact `packageManager` versions and supported `engines.node` declarations are checked against the
  worker before install/coding. Manager/lock conflicts, absent locks, unsupported versions, and
  unknown ranges fail early with typed evidence. Unsupported Yarn 4, for example, is refused on a
  Yarn 1 worker rather than run with the wrong binary.
- Dependency installs, workspace reset/clean/prepare, formatters, linters, tests, and builds all
  check their exit status. Setup failures use a separate bounded budget and do not consume coding
  attempts.
- The legacy live path now performs the same lockfile-selected frozen preflight before coding and
  injects real dependency synchronization, formatting, and lint-capability tools. A failed setup
  never calls Engineer or consumes a review retry.
- Implementation completeness fails when any declared production category or required test is
  missing. Production progress fingerprints include path and bytes, so changing the same file is
  meaningful while test-only edits cannot satisfy production work.
- Coding-operation replay stores content hashes. An earlier operation cannot claim bytes written by
  a later attempt at the same path.

### Review, Git, and pull requests

- The single-repository engineer no longer commits or pushes before review. An approved publisher
  commits exactly the reviewed files and emits a separate published artifact.
- The reviewer receives bounded current source or complete changed hunks for the cumulative dirty
  file set, not only completion metadata. High-confidence credential literals are withheld while
  dynamic authentication references, placeholders, and explicit environment templates remain
  reviewable. Redacted, sensitive, oversized, or unreadable evidence routes once to terminal manual
  review instead of burning coding retries. Review metadata binds exact paths and bytes to both
  legacy and feature publication, so a retry that repairs only file A cannot drop dirty file B.
- Feature and legacy commit journal keys use reviewed file bytes/paths/modes and survive a changed
  `HEAD`. A crash after commit or push reconciles the original operation without another coding or
  reviewer call.
- Cleanup, declared prepare scripts, staging, and Git hooks cannot silently alter reviewed bytes.
  The commit is verified after hooks; push is refused unless the branch head is the approved SHA.
- Remote push success requires the remote SHA to equal the approved local commit.
- PR completion is fetch-verified against repository, base/head branches, title, and exact head SHA.
  A publisher without a read-back interface fails closed.
- Partial PR resume skips already published children and creates only missing PRs. Optional failed
  repositories no longer block approved required work, and a failed convenience cross-link comment
  does not misreport already-created PRs as a failed feature.
- A shared contract revision invalidates every repository approval. Before any commit, every child
  is rerun and the integration gate rejects a stale contract artifact ID/version. Once a child has
  committed, in-place revision is refused and requires a fresh feature/branch set; old branch bytes
  cannot be relabelled as reviewed against a new contract by validating only a later delta.

### Runtime, storage, and deployment

- Health/readiness expose build SHA, workflow-schema version, and compatibility. Production startup
  refuses unknown, malformed, missing, or controller-mismatched identity.
- Mutating requests fail with 503 when identity, migration, PostgreSQL, Redis, startup recovery, or
  unresolved critical operation state is unsafe. Authenticated resume/reconciliation paths remain
  available so an unknown remote operation does not permanently deadlock recovery.
- Workflow snapshots persist creator and last-executor build identity. Migration-backed
  `legacy-unverified` snapshots remain readable for audit but every mutation fails closed.
- The image build includes Git evidence, verifies the exact full `BUILD_REVISION`, rejects dirty
  copied bytes, then removes `.git`, tests, prompts, and logs from the runtime layer. Compose passes
  the build expectation independently and adds the OCI revision label.
- Compose passes build arguments to every platform image, restarts the API unless stopped, mounts
  the workspace volume in development, and suppresses routine health access-log noise.
- Workspace names include a digest. Cleanup is serialized, rejects symlinks, consults durable
  terminal state, preserves cancelled-with-side-effects evidence, removes only auto-managed
  completed/cancelled workspaces, and refuses new work below the configured free-space floor.

### Credential and subprocess hardening

- Repository-controlled commands receive a strict allowlisted environment instead of the API
  process environment. User Git/pip/npm configuration is disabled.
- Local Git hooks receive no provider token. Credentialed remote operations disable credential
  helpers and repository pre-push hooks; legacy GitPython execution is similarly masked.
- Repository stdout/stderr is bounded and treated as ephemeral classifier input. Durable validation,
  preflight, formatter, install, and workspace diagnostics contain platform-owned status/result
  codes rather than attacker-controlled transcripts.
- Rejected schema/model values and arbitrary exception tracebacks are excluded from durable state
  and retained service logs; platform-owned error types/categories remain observable.
- The outer HTTP boundary consumes arbitrary exceptions into a generic 500 before Starlette or
  Uvicorn can retain their messages, and HTTP logs use static route templates rather than
  user-controlled path identifiers.
- Repository URLs containing userinfo, query parameters, or fragments are rejected by request/state
  validation before either legacy or feature workflow data can be persisted.
- Adversarial lifecycle, pre-commit, pre-push, validation-output, and environment tests prove that
  ordinary environment/hook paths do not expose or persist platform, database, Redis, OpenAI, or
  GitHub credentials.

## Verification record

The final audit must keep every item below green; no check is waived because a historical repository
is inconvenient.

| Check | Result |
| --- | --- |
| Full Python test suite | Passed |
| Ruff lint | Passed |
| Ruff format check | Passed |
| Strict mypy | Passed across 116 source files |
| `git diff --check` | Passed |
| Focused crash/retry/publication/security/runtime suites | Passed and included in the full suite |
| PostgreSQL migration upgrade/downgrade/upgrade | Passed on PostgreSQL 16; legacy rows received explicit `legacy-unverified` provenance |
| Clean-commit Docker build and isolated Compose health/readiness | Passed at temporary commit `ff3b18d182a93b90123da860a82a537692a8b6d5`; migration `0007`, exact identity, non-root runtime, clean source surface, and fail-closed mismatch verified |
| Live OpenAI/Git/GitHub staging | Not run; no staging credentials or repositories were supplied |

Commands executed against the final source tree included the following. They were run from the
repository root, which was the Python project root at the time; the project now lives under
`server/`, so the equivalent commands take `uv run --directory server`.

```sh
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
uv run pre-commit run --all-files
git diff --check
```

The isolated PostgreSQL check ran `alembic upgrade head`, downgrade to `20260803_0006`, and
upgrade to head again, then repeated `0006 -> 0007` with synthetic legacy rows and inspected the
backfilled columns/JSON. The isolated Compose check used a clean temporary Git commit, separate
project name, ports, network, and volumes; built with `--force-recreate`; inspected migration/image
metadata/runtime contents; queried `/healthz` and `/readyz`; and proved a deliberately mismatched
controller build exits with `RuntimeIdentityError`. All audit containers, volumes, and the temporary
checkout were removed afterward; the live stack was not changed.

SQLite is used only for isolated ORM tests. The Alembic chain contains PostgreSQL-specific sequence
operations, so PostgreSQL—not SQLite—is the authoritative migration check.

## Live pilot, 2026-08-17 to 2026-08-19

Runs `-066` through `-081` against `cryn3t/admanager_console-2.0` and
`cryn3t/AB-console-admin-2.0`, both private and owned by the team, on one PRD: a rolling
server-health history exposed by the backend and rendered by the console.

| | before | after |
| --- | --- | --- |
| last live run | `-065`, 2026-08-09 | `-081`, 2026-08-19 |
| pull requests opened | 0 | 10 across 7 runs |
| repositories reaching a commit | 3 of 18 | 2 of 2 in each of the last four runs |
| features completing both repositories | 0 | 4, including `-081` after a `SIGKILL` |

Every pull request is a draft on `master` whose head equals the approved commit, and the
publication invariant held in all ten: no repository that passed review went unpublished.

**Thirteen platform defects were found and fixed during the pilot**, each with a live
reproduction and a test asserting the effect rather than the call. The ones worth knowing:
the change fingerprint hashed file paths rather than bytes, so answering a review finding in
place read as a repeat; a generated `openapi.yaml` and a helper named after a module both made
unreachable code look wired in; provider faults, clone failures and interrupted coding calls
each ended a repository that was entitled to another attempt.

**The largest residual risk is not in the platform.** Runs `-066` through `-075` failed mostly
because the clarification answers asserted repository conventions that did not exist — a
health-evaluation source, per-route authentication, a shared time formatter. The reviewer
enforced each faithfully and no attempt could satisfy them, which is indistinguishable from a
platform defect in the artifacts. Verify every "use the existing X" against the checkout before
answering. Two latent defects in the target repositories were found the same way and belong
fixed there: `admanager_console-2.0` cannot import its own Express app
(`server/routes/auth.route.js` passes an undefined handler), and `AB-console-admin-2.0`'s Jest
cannot parse `nanoid`'s ESM, so any test importing the route registry runs zero tests.

Not yet demonstrated: any feature shape other than a read-only endpoint plus a page — no
migration, no schema change, no destructive operation — and no concurrent features. None of
the ten pull requests has been reviewed or merged by a human.

## Remaining production blockers

### 1. Per-job execution isolation — stop-ship for untrusted repositories

Environment filtering prevents the ordinary secret-leak paths, but repository code can still run
under the API UID and potentially inspect sibling workspaces/caches, same-UID `/proc` data, a
temporary askpass file during a concurrent remote operation, or infrastructure endpoints on the
container network. The production architecture needs isolated worker containers/pods with:

- a unique UID plus PID/user/mount namespaces per job;
- only that job's checkout mounted and no shared secret-bearing cache;
- seccomp/AppArmor (or equivalent), `no-new-privileges`, and a read-only base filesystem;
- deny-by-default egress, with narrow brokered access for model and Git operations;
- CPU, memory, process-count, disk, and wall-clock limits; and
- an authenticated queue/lease protocol separating the API from workers.

Until that exists, live mode is restricted to explicitly trusted disposable staging repositories.

### 2. Real-provider recovery drill — CLOSED 2026-08-19

Every item was exercised against the real providers, not doubles.

- **Token scope.** The pilot fine-grained PAT authenticates, pushes and opens pull requests, and
  is refused administrative writes: branch-protection reads and repository-settings `PATCH` both
  return 403. (`admin=true` in the repository permissions block is the *account's* role, not the
  token's grant; the 403s are the evidence.) The token is also refused on the Issues API:
  `GET /repos/{repo}/issues/{n}` returns 403, which is what failed every pull-request
  cross-link comment for the platform's first ten features. The adapter no longer makes that
  read — it comments through `GET /pulls/{n}` plus `POST /issues/{n}/comments`, which GitHub's
  permission table lists under `Pull requests: write` as well as `Issues: write`. If a
  publication still logs `pull_request_cross_link_failed`, its diagnostics now name the
  failing endpoint and status, and a 403 on the `POST` means the token needs a write grant on
  `Pull requests` (or `Issues`) for the target repositories.
- **Draft PR behaviour.** Ten pull requests opened across seven runs, every one a draft on
  `master` whose head SHA equals the approved commit. None merged; verified after the fact.
- **Cancellation with work in flight.** Four features cancelled mid-run. Each stopped
  cooperatively and reported its retained external side effects rather than deleting them.
- **Commit reconciliation after process death.** The API was `SIGKILL`ed during an in-flight
  `create_commit` (attempt 1, heartbeat three seconds old). Startup recovery correctly left it
  alone while its lease was still fresh; the periodic sweep reconciled it once the 300-second
  window expired, resolving it to `succeeded` and binding the journal to the real commit
  `e2d42820`. The checkout carried exactly one commit ahead of `master` — no duplicate, no lost
  work — and the resume that followed reused that commit instead of re-executing it. The run
  then finished: `-081` completed both repositories, and `e2d42820` is the head of the pull
  request it opened (`AB-console-admin-2.0` #18). The commit that survived a `SIGKILL` mid-write
  is the commit that shipped.
- **Recovery reconciles operations, not orchestration.** After the kill, `-081` held its running
  status for a day with its commit already reconciled and nothing left to move it, because a
  resume is operator-initiated. That is by design and is now stated in the runbook;
  `platform_live_features_in_flight` is how an operator notices it.
- **Stale-image protection, observed rather than asserted.** Restarting the API with a
  controller-supplied revision that did not match its image failed closed with
  `RuntimeIdentityError: build_revision_mismatch` before serving any request.
- **A client disconnecting does not stop the work.** Server-side execution continued after the
  HTTP client died, so a timed-out request must not be read as an abandoned run.

Provider availability is the residual risk rather than an open item: four runs were ended by
provider faults before infrastructure faults were made retryable, and one blip produced three
failures inside four minutes.

### 3. Current deployment has not received these changes — CLOSED

The deployment is rebuilt from the commit under test on every change, and `/readyz` and
`/healthz` are checked against `git rev-parse HEAD` each time. A mismatch fails closed, which
was observed during the recovery drill above.

### 4. Broad-service operational controls — CLOSED for single-tenant trusted use

- **Workspace archival policy.** Retention reclaimed only completed and cancelled workspaces, so
  a feature that ended needing a human kept its checkout forever; the volume filled and the
  capacity preflight refused a run. Those checkouts now expire after
  `workspace_failed_retention_hours` (default 72), and the database keeps the evidence either way.
- **Rate limits and quotas.** `max_concurrent_live_features` (default 3) bounds live work.
  `waiting_for_human` counts against it, because it holds a checkout and a slot until answered.
- **Operator reconciliation API.** `GET /features/operations/unresolved` lists external effects
  needing a decision. `POST /features/{id}/actions/{action_id}/reconcile` records an admin's
  verified outcome without rerunning the action, and `POST /features/{id}/retire` records an
  attributed close-out for an unrecoverable feature without discarding evidence.
- **Retained telemetry.** `GET /metrics` publishes features by status, live features in flight,
  the configured quota, unresolved operations, free workspace bytes, and runtime compatibility.
  Every compose service rotates logs at 50 MB across five files.
- **Backup, restore and failover.** `docs/DISASTER_RECOVERY.md`, written from a drill that was
  executed against this deployment. Failover today means restore: this topology runs one
  PostgreSQL container, and that is the honest ceiling on recovery time.
- **Stale workflows.** `-014`, `-025` and `-055` are resolved through the retire endpoint with
  their artifacts, children and journal rows intact. No feature remains in a non-terminal state.

Still open, and deliberately, because they are **multi-tenant** requirements rather than
single-tenant ones: user/repository authorization, pagination for the remaining long histories,
and distributed tracing. Add them before a second team or an untrusted caller is given access.

## Deployment and pilot procedure

1. Review and commit this patch. Confirm the release checkout is exact and clean:

   ```sh
   test -z "$(git status --porcelain)"
   export BUILD_REVISION="$(git rev-parse --verify HEAD)"
   export WORKFLOW_SCHEMA_VERSION="1.0"
   ```

2. Supply production settings from the secret manager, then build and recreate all platform
   services from that same checkout:

   ```sh
   docker compose up --detach --build --force-recreate
   docker compose ps
   ```

3. Require the migration job to exit 0 at Alembic head `20260808_0007`. Require `/healthz` and
   `/readyz` to report the exact `$BUILD_REVISION`, schema `1.0`, and
   `runtime_compatible: true`. Any mismatch or 503 is a stop-ship condition.
4. Run one mock request and inspect artifacts, attempt IDs, child checkpoints, and failure summary.
5. Use protected repositories your team owns and scoped staging credentials for one live
   request. Worker isolation (blocker 1) gates *untrusted* repositories, not this step; do not
   connect a repository the team does not control until it exists. Kill the worker after
   coding, commit, push, and first PR creation in separate drills; each resume must reuse
   completed effects and verify provider state.

   **Watch these two numbers, because they are what the last live runs failed on.** How many
   repositories reach a commit at all, and whether every repository that reaches one also gets
   a pull request. The second is now covered by tests over every failure route; the first is
   the open question. Both are one query each:

   ```sh
   docker compose exec postgres psql -U platform -d ai_platform -c \
     "SELECT operation_type, count(*) FROM external_operations
      WHERE status='succeeded' AND feature_id = 'YOUR-FEATURE-ID' GROUP BY 1;"
   docker compose exec postgres psql -U platform -d ai_platform -c \
     "SELECT repository_id, status, pull_request_artifact_id IS NOT NULL AS published
      FROM feature_child_workflows WHERE feature_id = 'YOUR-FEATURE-ID';"
   ```

   Any row with `status='approved'` and `published=false` is a regression of the publication
   rule and should be reported as one.
6. Confirm every PR is a draft at the approved SHA and that no merge/deploy occurred. Revoke pilot
   credentials and retain database, operation-event, PR, and runtime-identity evidence.

Do not run `docker compose down --volumes` on production state. Prefer forward migrations and a new
immutable image; if rollback is necessary, preserve the operation journal and workspace evidence
needed to reconcile external effects.
