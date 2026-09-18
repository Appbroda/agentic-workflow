# Authentication and workspaces

This deployment went from *one shared workspace with named identities* to *many isolated
workspaces*. Each person has an account with a password, and a feature — its plan, its chat,
its agent transcripts, its pull requests — belongs to whoever submitted it. Nobody else can
see it, including an administrator who can.

This document records the decisions behind that, because several of them exist only because
the obvious alternative destroys data or reopens a hole.

---

## The two identities that are not the same field

`feature_workflows.owner_id` — **whose workspace this feature is in.** Set from the submitting
actor at creation, never reassigned. It decides who may see and act on the feature.

`feature_execution_queue.requested_by` — **whose stored provider credentials the worker
resolves when it runs.** Nothing else. The queue row deliberately holds no secret; the
dispatcher looks the identity's keys up at claim time.

**For a feature owned by U, every enqueue carries `requested_by = U`, whoever pressed the
button.** Before that rule, an administrator granting a retry on somebody's feature put
themselves on the queue entry — so the worker resolved the *administrator's* GitHub token and
pushed to that person's repository with it. One person's credential doing another person's
work.

The audit answer to "who decided this" travels in `intent_payload["requested_by"]`, where it
always has. Crossing the two fields once killed AB-Feature-111's granted retry permanently: a
queue entry named `"akhilesh (via platform key)"` as an account that never existed, and the
poisoned value survived the code fix. The comments at the `requeue` call sites in
`server/storage/feature_store.py` say so; read them before touching either field.

---

## Nothing rewrote an `owner_id`, and nothing should

`platform-admin` used to be a constructed `Actor` — the identity the shared platform key
resolved to. It is a `platform_users` row now (migration 0031), keeping the same id, and
migration 0030 backfilled every pre-existing feature to it.

The obvious alternative was to mint a fresh user id for the administrator and `UPDATE` every
`owner_id`. That destroys data:

> `owner_id` and `provider` are the AES-GCM **additional authenticated data** of a stored
> provider credential (`_associated_data` in `server/services/secrets.py`). Rewriting
> `owner_id` on `provider_credentials` without decrypting and re-sealing every row makes
> every stored credential permanently undecryptable. Nothing warns you: the rows look
> perfectly correct in SQL and fail at the next `resolve`. It also cannot be a pure SQL
> migration, because only the application holds the key.

The price is that the administrator's `user_id` is the machine string `platform-admin` for
ever. It is never displayed — `display_name` is, and `subject` carries the email that login
matches on. A cosmetic cost against a data-destruction risk.

If a future change ever must move an `owner_id`, it decrypts and re-seals in application
code, never in SQL, and re-verifies each provider afterwards. `test_workspace_migration.py`
holds a test that opens a credential sealed before the migration; it is the only assertion
that can catch this, because no query can.

---

## Where the ownership check lives

One predicate, in the store, in the SQL `WHERE`: `_require_model` in
`server/storage/feature_store.py`. Every read and every mutation reaches a feature through
`get_record`, which reaches it through there.

A row somebody else owns is `WorkflowNotFoundError`, which the routes already answer **404** —
byte-identical to a feature that never existed. Not 403: `AB-Feature-N` is a dense global
sequence, so a 403 confirms existence and lets anybody count how much work the platform is
doing and for whom.

Routes cannot reach around it. `get_feature_control_plane` — the dependency every feature
route resolves — returns a `ScopedFeatureControlPlane`, which carries the request's workspace
and supplies it on every call whether the caller passed one or not. That is deliberately
stronger than a required parameter, which somebody in a hurry can satisfy with
`WorkspaceScope.unscoped()`. There are more than twenty read routes and a dozen mutating ones
keyed by `{feature_id}`, plus the chat action path and the SSE stream; a per-route check would
be thirty places to forget one.

Two layers, answering different risks: the `WHERE` means the answer cannot be wrong, and the
wrapper means the question cannot be omitted.

The chat service is rebound to the same scoped view per request, because a confirmed chat
proposal reaches the same control-plane methods the buttons do.

### `WORKSPACE_READ_ANY` is a read grant

`admin` holds it and nobody else does. An administrator may read any workspace — the
disaster-recovery runbook has them reconciling unresolved external operations across every
feature — and may **not** retry, publish, retire, resume or cancel somebody else's. Doing so
would spend that person's money on model calls and push with their GitHub token.

Mechanically: mutating calls ask with `scope.for_mutation()`, which drops `may_read_any`. An
administrator acting outside their own workspace gets the same 404 anybody else would. If a
real need appears it gets its own permission and its own audit record rather than being folded
into the read grant.

Named grants, never role strings. `Actor.roles` holds raw strings and an unrecognised one
grants nothing; a scattered `"admin" in actor.roles` would be a second authorization authority
beside `ROLE_PERMISSIONS`, and the two would drift. The console checks the published
permission list for the same reason.

---

## Passwords

`scrypt` from `cryptography`, which this platform already depends on — AES-GCM comes from it —
so a memory-hard KDF arrives with no new dependency and no C extension to screen. Parameters
`n=2**15, r=8, p=1`, a fresh 16-byte salt per password, encoded as
`scrypt$n$r$p$salt$hash` in one column so the work factor can be raised by rewriting rows
rather than by a second migration. `argon2id` is the marginally better primitive and is the
alternative if a reviewer prefers it.

`api/passwords.py` is its own module, and that matters: `hash_token` beside it is a plain
sha256, and its docstring explains why — a token is 256 bits of randomness with nothing
cheaper to guess than the token itself. A human-chosen password is the opposite case, and the
two must never look interchangeable to somebody reaching for one.

Minimum 12 characters. No composition rules: they push people towards `Password1!`, and this
platform's threat model is credential stuffing rather than a cracking rig. Maximum 1024, so a
request cannot make the KDF the denial of service.

### Every failed login says the same thing

An unknown email, a wrong password, an account with no password and a disabled account are one
`401` with one body. The KDF runs against a throwaway hash when the account does not exist, so
absence is not a fast path — without that, the response time tells an attacker which addresses
are real. Nothing logs an email beside an outcome; a log line saying "login failed for
alice@example.com" is an enumeration oracle for everybody who can read logs, which is more
people than can read the database.

### Sessions are ordinary tokens

A successful login mints a row in `platform_api_tokens` through the existing `issue_token` and
returns it once, exactly as `POST /users/{id}/tokens` already does, with `kind='session'` and
`expires_at = now + SESSION_TOKEN_TTL_HOURS` (12 by default).

No JWT, no session middleware, no cookie store, no second `ActorDirectory`. The path stays
`PlatformAuthenticator` → `ActorDirectory.resolve(token)` → `Actor`, which already handles
expiry, revocation, disabled accounts and `last_used_at`, and already answers one
indistinguishable way for every kind of failure. A stateless token would duplicate all of that
and lose immediate revocation — which matters more here than usual, because these credentials
authorise spending money on model calls and opening pull requests on other people's
repositories.

The honest cost is a database round trip per request to resolve the token. That was already
true for per-user tokens and has not been a problem. If it becomes one, the fix is a short-TTL
cache keyed on the digest with revocation punching through — not a token format change.

### `kind` earns its column

`'session'` or `'api'`, `NOT NULL`, `server_default='api'` so every token issued before it
existed keeps its meaning. Two operations need it and cannot be expressed without it: changing
a password revokes the account's **sessions** and must leave its long-lived automation tokens
working, and "log out everywhere" means the same thing. Encoding it in `label` by convention
would make a display string load-bearing.

A password change spares the session that made it — being logged out by securing your own
account is a surprise nobody benefits from. An administrator resetting somebody's password
spares nothing, which is the point: the usual reason is that the account may be compromised.

### Rate limiting, not lockout

Counted in Redis, per source address **and** per email, 10 attempts per 15 minutes by default.
Both counters, because an attacker with a botnet defeats a per-address limit and one spraying
a single password across a leaked address list defeats a per-email limit.

Lockout was considered and rejected: it stops an attacker and hands anybody who knows an email
address a denial of service against that person. A counter that cannot be reached allows the
attempt and logs that it could not count — a broken counter must not be an outage for
everybody.

The source address is `request.client.host` and nothing else. `X-Forwarded-For` is
client-supplied, and trusting it would let an attacker choose their own bucket per attempt,
which is not a limit. Behind a proxy that rewrites the peer address the per-address counter
becomes per-proxy and the per-email counter carries the defence.

---

## `PLATFORM_API_KEY`

**Optional now, and unset is the target state.** Anybody holding it is an unnamed
administrator with every permission, and with per-user passwords there is an account for every
person.

It is not removed. Break-glass access when the *database* is the thing that is broken is a
real requirement — a per-user token cannot be resolved without the directory. The recommended
break-glass credential is an admin API token issued in advance and kept where the key used to
be kept: it can be revoked, and it names somebody.

An empty value means **unset**, not a key nobody can guess. Compose passes an unexported
variable through as an empty string, and an empty *string* would be a live shared key that
`secrets.compare_digest` matches for a caller sending an empty bearer token.

### A rejected credential is 401, not 503

`PlatformAuthenticator` answers `503` only when it can check *nothing* — no user directory
**and** no shared key. If a directory is bound and it said no, the credential is unknown,
expired, revoked or belongs to a disabled account, and that is `401`.

This needed correcting when the key became optional. The `503` branch used to be reachable
only by a deployment with neither a key nor a directory; with the key normally absent it
became **every** rejected credential's answer — claiming authentication was not configured
when it was, and never firing the console's 401 handling, so an expired session showed error
boxes for ever instead of a login page.

---

## The three decisions the PRD left open

### Slack thread leakage across workspaces — safe default implemented

`slack_workspace_configurations` permits one enabled row for the whole deployment, so every
feature's thread posts into the same channel where other people can read titles, statuses and
whatever the thread summary quotes. That defeats workspace isolation through a side channel
the API never sees.

**Implemented: a feature owned by a non-admin account gets no Slack anchor and no replies**,
and the configuration surface is administrator-only. That is option (b) of the three the
specification listed.

Mechanically: `SlackNotificationDispatcher._deliverable` drops candidates whose
`feature_workflows.owner_id` is not in `administrator_ids()`, and logs once per transition how
many it withheld — a notification that never arrives is otherwise indistinguishable from one
that had nothing to say. With no user directory bound nothing is withheld, which is right for
an application assembled without identity: there is no second workspace in one of those for a
thread to leak into.

The product owner may still choose (a) — accept the leak and tell people their feature titles
are visible to the deployment's Slack channel — or (c), per-user Slack configuration, which
means dropping the partial unique index that makes the configuration a singleton and
re-deriving `token_owner_id` per feature: a follow-up of comparable size to this change.

### Account lockout versus rate limiting — rate limiting only

Recorded above. No existing pattern in this repository decided it; the reasoning is the
denial-of-service asymmetry.

### The legacy single-workflow router — administrator-only, deprecated

`server/api/routes.py` exposes `POST /workflow/start`, `/resume`, `/cancel` and four reads.
The specification expected it to sit on the `workflows` table with no owner. **It does not**:
it delegates to the feature control plane through `FeatureBackedWorkflowControlPlane`, and it
was doing so *unscoped* — a fully open read path over every workspace's features, reachable by
anybody the router let through.

Its request and response schemas have no owner anywhere in them, so it cannot be scoped to its
caller: it has no way to express whose work a workflow is. It is therefore **confined to
`platform-admin`'s workspace** by the control plane behind it, and gated on
`WORKSPACE_READ_ANY` at the router. Which is internally consistent — it operates on exactly
the features it created, plus the ones migration 0030 backfilled to the same owner, which is
every workflow that existed before workspaces did.

Deprecated rather than removed: `docs/USER_GUIDE.md`, `docs/DEPLOYMENT.md` and the canary all
address it, and break-glass access to a single-repository run is a real requirement. New work
goes through `/features/*`.

---

## Known limitations

**Slack and Figma are deployment singletons.** Both `slack_workspace_configurations` and
`design_source_configurations` permit one enabled row, enforced by a partial unique index over
a constant expression. User-scoping either means dropping that index and re-deriving
`token_owner_id` per feature, which is its own project. Both are administrator-only, and
`SLACK_CONFIGURATION_MANAGE` and `DESIGN_SOURCE_MANAGE` moved out of the operator role to make
that true — **a narrowing of an existing grant**, and the one behaviour change an upgrading
deployment will notice.

The consequence for Slack is handled (no thread for a non-admin's feature). The consequence
for Figma is not, and is smaller: design *resolution* runs against the deployment's single
Figma account, so a feature in anybody's workspace cites files from the same allowlist. That
is one deployment-wide capability rather than a leak between workspaces — nothing about one
person's work reaches another — but it does mean an ordinary user's citations are limited to
files the administrator permitted, and cannot use their own Figma account.

**`AB-Feature-N` stays a global sequence.** It is quoted in Slack threads, PR titles, logs,
the runbooks and roughly every comment in this codebase; per-workspace numbering would make
`AB-Feature-108` ambiguous. Accepted leak: somebody can infer the platform-wide submission
count from the gaps between their own reference numbers.

**A header credential still beats a stored one, for one request.** `resolve_credentials` lets
`X-OpenAI-Api-Key`, `X-Anthropic-Api-Key` and `X-GitHub-Token` win. That is a caller supplying
*their own* key, not a path to anybody else's, so it is not an isolation hole — and it is inert
for queued work, which reads only the store. **A header-supplied key never reaches background
execution.**

**A disabled account's queued work still runs.** `resolve` refuses their tokens immediately,
but the dispatcher does not authenticate — it resolves stored credentials by id. Disabling
somebody does not stop a feature of theirs that is already queued. The recommended fix is for
the dispatcher to skip entries whose owner is disabled, leaving them queued and visible on the
accounts page; silently running work for a disabled account is worse, and silently deleting it
is worse still. Not implemented — see the report.

**SSE re-checks ownership, not the token.** `GET /features/{id}/events/stream` checks the
workspace at subscribe *and* on every poll, because `events_after` is scoped and the stream
calls it every two seconds. It does not re-check the token, so a revoked session's open stream
keeps delivering that workspace's events until the connection drops. The data stays confined to
the workspace it belongs to; the credential's revocation is not immediate for an already-open
stream.

---

## Rolling back

Roll back the **application only**. Keep migrations 0029–0031 applied: new columns are ignored
by older code, and reverting the schema loses every password and every session.

If the application is rolled back past login, access is regained by setting `PLATFORM_API_KEY`
in the environment again — which is the reason the mechanism is retained rather than removed.
