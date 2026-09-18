# Disaster Recovery

The database is the system of record. Workspaces are re-clonable working copies, and Redis
holds locks and cancellation signals that are rebuilt on reconnect. Everything that must
survive a loss — features, artifacts, child workstreams, pull requests, the external
operation journal and its events — lives in PostgreSQL.

## Before anything else: how you get in

Every command here needs `$ADMIN_TOKEN` — an API token belonging to an account holding
`admin`, issued in advance from the console's People page and kept where a shared key used to
be kept. It is revocable and it names somebody, which a shared key is not and does not. Issue
one now if this deployment has none; doing it during an incident requires a working database,
which is exactly what may not be available.

If the *database* is what is broken, a per-user token cannot be resolved either: the directory
lives there. That is the one case `PLATFORM_API_KEY` still exists for. Set it in the
environment, restart the API, and it authenticates as `platform-admin` with every permission —
then unset it again once the incident is over. See
[Authentication and workspaces](AUTHENTICATION_AND_WORKSPACES.md).

An administrator reads every workspace, which is what the steps below depend on. An
administrator deliberately **cannot** retry, publish, resume, cancel or retire a feature they
do not own: that would spend that person's money on model calls and push to their repository
with their token. Where a step needs one of those, it is the feature's owner who performs it.

## Backup

Take a custom-format dump, which restores selectively and compresses:

```sh
docker compose exec -T postgres pg_dump -U platform -d ai_platform \
  --format=custom --file=/tmp/ai_platform.dump
docker compose exec -T postgres sh -c "cat /tmp/ai_platform.dump" > ai_platform.dump
```

The second command is not optional. A dump written inside the container is on the same
volume as the database it protects, so it is not a backup until it leaves that volume.

Schedule this at an interval you can afford to lose. A feature that is mid-flight when the
snapshot is taken restores as a feature that needs recovery, which the platform handles —
see *After a restore* below.

## Restore

Restore into a scratch database first and compare it to the original. A dump nobody has
restored is a hypothesis.

```sh
docker compose exec -T postgres psql -U platform -d postgres -c "CREATE DATABASE restore_drill;"
docker compose exec -T postgres pg_restore -U platform -d restore_drill --no-owner \
  /tmp/ai_platform.dump
```

Compare row counts and the migration head between the two:

```sh
for db in ai_platform restore_drill; do
  docker compose exec -T postgres psql -U platform -d "$db" -t -A -c \
    "SELECT 'features=' || (SELECT count(*) FROM feature_workflows)
         || ' artifacts=' || (SELECT count(*) FROM feature_artifacts)
         || ' operations=' || (SELECT count(*) FROM external_operations)
         || ' prs=' || (SELECT count(*) FROM feature_pull_requests);"
  docker compose exec -T postgres psql -U platform -d "$db" -t -A -c \
    "SELECT version_num FROM alembic_version;"
done
```

To restore in place instead, stop the API first so nothing writes during the restore, then
`pg_restore --clean --if-exists` into `ai_platform` and start the API again. The API refuses
to serve when its migration head does not match the packaged one, so a restore that lands on
an older schema fails closed at `/readyz` rather than serving corrupt state.

### Drill evidence — 2026-08-19

Executed against this deployment, not described:

| check | original | restored |
| --- | --- | --- |
| features | 81 | 81 |
| artifacts | 907 | 907 |
| external operations | 2229 | 2229 |
| pull requests | 19 | 19 |
| migration head | `20260817_0008` | `20260817_0008` |

Content spot-check: the three `feature_retired_by_operator` events survived with their
operator attribution intact. Dump size 4.4 MB. The scratch database was dropped afterwards.

## After a restore

Restarting the API runs bounded recovery over the operation journal and expired durable action
leases, and then over the features themselves.

A feature that was mid-flight when the process died no longer waits for a person. Its queue
claim lapses, the next worker claims the entry, and — having established that nothing else is
executing it — continues the run from its last checkpoint, recording a `feature_run_continued`
event. During the recovery drill one such feature sat in `running_child_workflows` for a day
because the claim that arrived was dropped as unwanted; that is what changed. The queue
entry's attempt budget still bounds it: a feature that crashes as many times as its entry
allows gets a terminal status saying it was continued and did not survive.

Resume by hand is still available, and is what you use for a feature that stopped for some
other reason:

```sh
curl -s -X POST "$API/features/$FEATURE_ID/resume" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "X-OpenAI-Api-Key: $OPENAI_KEY" -H "X-GitHub-Token: $GITHUB_TOKEN" \
  -H "Content-Type: application/json" -d '{"answers": []}'
```

Provider credentials are request-scoped. A deployment with `SECRET_ENCRYPTION_KEY` set may
also hold them per identity, in which case a resume can use the stored one; otherwise a resume
needs fresh ones. The encryption key lives in the environment and never in the database it
protects, so losing it loses no workflow data -- stored credentials simply have to be entered
again.
For a planned rotation, deploy the new version as `SECRET_ENCRYPTION_KEY` and retain old
versioned material temporarily in the JSON `SECRET_ENCRYPTION_PREVIOUS_KEYS` list. Reads
re-seal old rows with the current key; do not remove an old key until all credential owners
have been migrated or have re-entered their credentials.
A `409 workflow operation is already in progress` means exactly that and is not a stale lock —
wait and retry. `GET /metrics` publishes `platform_live_features_in_flight`, which is how you
notice a feature that stopped progressing without reaching a terminal status.

Workspace-local operations — clone, install, build, coding, formatter, linter, tests, typecheck, file writes —
are reconciled automatically because nothing outside the checkout depends on them. Operations
with external effects — push, pull-request create, labels, reviewers — are left
`awaiting_reconciliation` instead, because the sweep holds no provider credentials and the
next credentialed request through the same code path can prove what happened. That request
now arrives on its own for a feature whose executor died, because the continuation re-enters
the same publication path.

An external effect becomes `unknown_external_state` for a human in three cases: its replay
budget is spent, the local evidence a clone or commit needed is absent, or it is still
deferred a day after the feature that owns it ended and no request is ever going to come.
See `RECOVERY_AND_IDEMPOTENCY.md` for the full disposition table.

Read what needs attention. **This needs an administrator's token**, and it always did in
practice: the endpoint is owner-scoped since workspaces became isolated, and only an
administrator sees every workspace's operations -- including the rows whose `feature_id` is
null, which are single-workflow operations belonging to no workspace at all. An operator's
token returns only their own features' operations, which is not what this step is for.

```sh
curl -s "$API/features/operations/unresolved?limit=100" -H "Authorization: Bearer $ADMIN_TOKEN"
```

For each one, inspect the repository branch, commit and pull request **before** any retry.
Never force-push, delete a pull request, or recreate a branch as automated cleanup.

An action whose provider operations were settled can still require reconciliation when the
platform cannot prove the workflow-state transaction committed. Open the feature's Overview,
inspect its Requested actions, then have an administrator record the verified outcome. The
equivalent API is:

```sh
curl -s -X POST "$API/features/$FEATURE_ID/actions/$ACTION_ID/reconcile" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"outcome":"failed","reason":"Checked workflow, branch and PR; no action result committed."}'
```

This closes only the action record. It never replays the action or changes workflow state.

Workspaces are not backed up. After a restore they are gone, and a feature that needs one
re-clones. A feature whose checkout is gone and whose work was never pushed has lost that
work; its artifacts and journal remain, which is what tells you so.

## Failover

This deployment runs a single PostgreSQL container, so failover today means restore. That is
a deliberate limit of the compose topology, not an oversight, and it is the honest ceiling on
recovery time: however long a restore takes is the outage.

Before running this where that ceiling is unacceptable, move PostgreSQL to a managed instance
with a replica and automated failover, point `DATABASE_URL` at its endpoint, and re-run the
restore drill against it. Nothing in the platform assumes a local database.

Redis needs no failover plan. Losing it loses locks and in-flight cancellation signals; the
API rebuilds both on reconnect, and readiness refuses work until it is reachable.

## Retiring a feature that cannot recover

A feature whose state predates verified build provenance is audit-only: resume, cancel and
checkpoint all refuse it, correctly, because its recorded identity is not evidence of what
produced it. Close it out with an attributed decision instead:

```sh
curl -s -X POST "$API/features/$FEATURE_ID/retire" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"operator":"your-name","reason":"why this cannot progress"}'
```

Nothing executes and nothing is discarded. It writes a terminal status and an event naming
who decided and why. A feature with active external operations is refused — cancel it
instead, so those operations are given the chance to stop.
