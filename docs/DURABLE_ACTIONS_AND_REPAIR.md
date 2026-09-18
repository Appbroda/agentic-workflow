# Durable actions, repository repair, identity and stored credentials

Four capabilities that share one idea: the platform should be able to say what happened, who
asked for it, and whether it finished — after a restart, not only during one.

## Durable actions

A workflow-changing action a person asks for is recorded before anything runs.

```text
PROPOSED → CONFIRMED → CLAIMED → EXECUTING → SUCCEEDED
                                           ↘ FAILED
                                           ↘ REQUIRES_RECONCILIATION
```

Its identity is `(feature, repository, action type, normalized payload, context version)`,
hashed and made unique by the database. Two concurrent confirmations both insert; one wins and
the other reads the winner's row. That is what makes a double-clicked Confirm buy one retry.

`context_version` is what the action was decided against — the repository revision, the
clarification round, the repair being approved. Without it a grant made after a failed granted
attempt would replay the first grant's result instead of buying a second attempt.

Ownership is a lease (`DEFAULT_LEASE_SECONDS`, renewed at a third of it). The platform runs
actions inside the request that asked for them — there is no worker queue — so the executor is
a process that can die holding work. An executor that cannot prove it still owns the action
aborts rather than continuing unowned; that is the one path that could otherwise produce the
duplicate effect this whole mechanism exists to prevent.

### A result is only recorded for work that happened

The control plane marks a feature `failed_requires_human` when an operation on it fails
unexpectedly — a full workspace volume, an unconfigured runtime — so the diagnosis is durable
rather than an HTTP 500. It then raises `FeatureOperationFailedError` (HTTP 409) instead of
returning the feature. Returning it was indistinguishable from success: the action recorded
`"<TYPE> committed"` for an operation that had committed nothing, told the caller it had
worked, and — for a caller relying on the platform's own deterministic action identity rather
than sending its own `Idempotency-Key` — replayed that false success on every later attempt,
so the same repair could never be approved and nobody was told why.

The message is platform-owned and never quotes the underlying exception, which may carry a
provider response or a path inside a repository. The reason stays on the feature.

### Recovery does not rerun

`ActionRecoveryService` classifies each abandoned action from the external-operation journal,
which is written *before* each side effect and therefore survives the crash:

| Evidence | Verdict |
|---|---|
| committed domain-result checkpoint | `SUCCEEDED` — workflow state committed before final action status |
| claimed but never began | `CONFIRMED` — safe to ask again |
| executing with no domain checkpoint | `REQUIRES_RECONCILIATION` — an internal commit cannot be ruled out |
| every external effect confirmed succeeded, no domain checkpoint | `REQUIRES_RECONCILIATION` — provider success does not prove workflow state committed |
| any effect in `UNKNOWN_EXTERNAL_STATE` | `REQUIRES_RECONCILIATION` — never retried automatically |
| a remote effect (push, PR) unsettled | `REQUIRES_RECONCILIATION` |
| workspace-local effects only | `FAILED` — the next attempt rebuilds the workspace |
| the action changes nothing outside the platform | `CONFIRMED` — cancel and reject are idempotent |

A failed action is **refused** rather than replayed. A failure may have crossed an external
checkpoint before it was reported, so "done" is a claim the platform cannot support and "try
again" risks repeating whatever landed.

An administrator closes `REQUIRES_RECONCILIATION` only after checking workflow and provider
state, using `POST /features/{feature_id}/actions/{action_id}/reconcile` with an explicit
`succeeded` or `failed` conclusion and evidence in `reason`. The decision records the
reconciler and timestamp. It does not execute the stored action or mutate workflow state.

This is not a second journal. External effects are still journaled by
`ExternalOperationJournal`, which is what makes them idempotent; the action record answers only
what that journal cannot, which is whether the thing somebody asked for was carried out.

## Repository repair

A repository whose checked-in setup will not let it run its own checks is a different failure
from code that was wrong: nothing written there could have been validated. The platform stops,
and will not fix somebody else's repository on its own — so it writes down a proposal precise
enough to approve or refuse.

```text
PROPOSED → APPROVED → EXECUTING → SUCCEEDED
        ↘ REJECTED            ↘ FAILED
        ↘ SUPERSEDED
```

Proposals are produced only for failures that are about the repository
(`VALIDATION_CONFIGURATION_FAILURE`, `DEPENDENCY_INSTALLATION_FAILURE`) intersected with
repairable preflight categories. An ordinary failing test produces none: asking somebody to
approve a change to their repository for a problem that was never in it is worse than saying
nothing.

Approving one whose repository has moved since the diagnosis is refused and the proposal
superseded — the commands were chosen against a checkout that no longer exists. The refusal
carries the state change, so the stale proposal is not offered again.

A proposal is only made when the platform could actually carry it out. A dependency the
repository never declared is fixed by declaring it, so where the package manager has no
command that records a declaration — `pip` installs into an environment and writes no
manifest — no repair is offered, and the finding is reported as something a person must
change. An Approve button that changes nothing is worse than saying so.

Package names come from a package manager's own error output, which quotes strings the
repository controls. They are validated against npm and PyPI naming and dropped otherwise;
commands run through `create_subprocess_exec` with no shell, so an argument cannot become a
second command, but a name is still handed to a tool that will try to fetch it.

An approved repair's commands run in the child's checkout **before preflight** — the point of
a repair is that preflight should then pass — and only when somebody authorized it. Each goes
through the operation journal like every other side effect, so approving twice replays the
recorded result instead of installing twice. Where a repair needs no command (a declared
dependency that is merely absent, an installation that failed), the granted attempt clones
afresh and installs deterministically, and that is the repair.

## Identity and authorization

A bearer token resolves to a person -- a session token from a password login, or an API token
an administrator issued. The deployment's shared `PLATFORM_API_KEY`, if one is still set,
resolves to a
named `platform-admin` identity. The shared key is retained deliberately — the operator
console, the browser flows and existing deployments all authenticate with it — but it is now an
identity with a name rather than an absence of one.

Roles are `viewer`, `operator`, `admin`. An unrecognised role grants nothing. Authorization is
checked on the route, never inferred from which buttons a client drew.

On the route rather than in the function body, specifically, wherever a route resolves the
caller's stored provider credentials to build its arguments. FastAPI solves route dependencies
first and the endpoint's own last, so a permission checked in the body runs after the request
has been parsed and every collaborator resolved — which meant an identity that may not create
a feature was told which fields its body was missing, and had its own stored keys unsealed and
their last-used timestamp moved, on the way to being refused.

Chat is checked twice, and the second check is the one that matters. The route can only ask
"may this identity execute a chat action" — it has a message id and nothing else — so the
specific permission is checked once the action type is known: confirming a repair approval in
a sentence needs `repair:approve`, exactly as pressing the button does. An action type this
build does not recognise requires `user:manage`, so a newer proposal cannot be executed by
somebody a newer build would not have allowed.

Who granted a retry is taken from the authenticated identity, not from the request body. Where
the shared key was used, the typed name is kept and labelled `(via platform key)` — the
platform cannot verify who held it, and presenting an unverified name as an audit answer would
be worse than saying so.

## Stored provider credentials

Optional, and off unless `SECRET_ENCRYPTION_KEY` is configured. Without it the platform keeps
its original behaviour of persisting no provider secret at all.

What is stored is sealed with AES-GCM using a key from the environment — never from the
database it protects — and bound to `(owner, provider)` as associated data, so a row moved into
another name will not open. What is returned is never the secret: only whether one is
configured and its last four characters.

Keys are versioned. To rotate, set a new `SECRET_ENCRYPTION_KEY` version and list the old
versioned values in `SECRET_ENCRYPTION_PREVIOUS_KEYS` as JSON. A credential opened with an old
key is immediately re-sealed with the current key; retire the old key only after all stored
rows have been exercised or otherwise migrated. Reusing a version label is rejected.

A request header always wins. Supplying a key for one request is a deliberate choice about
which key does that work, and only what the request did not supply falls back to the store.
`resolve_credentials` returns the same `RequestScopedCredentials` object the platform always
used, so no runner, orchestrator, adapter or agent learns that a store exists.

## Streaming

Chat answers and lifecycle events are server-sent events over ordinary authenticated requests
read with `fetch`. `EventSource` is not used: it cannot send an `Authorization` header, and the
usual way round it puts the token in the query string, where proxies log it and browsers keep
it in history.

The assistant writes prose first and appends any proposal after `<<<PLATFORM_ACTION>>>`.
Everything before it is streamed; everything after is read as a command and never shown. A
delimiter split across two deltas is withheld until the next delta settles it.

A stream is never the record. The question is persisted before the model is asked and whatever
prose arrived is persisted when the stream ends, however it ended — so a reader who closed the
tab sees the same transcript as one who watched. The event stream reads the same indexed table
the polling endpoint does; it says *when* to look, and polling remains the fallback.
