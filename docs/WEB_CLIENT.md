# Web Control Plane

The client under `client/` is a React + TypeScript + Vite application that renders the feature
workflow the platform runs. It holds no workflow rules of its own: it displays what the API
reports and asks the API to act.

## Running it locally

```sh
cd client
npm install
npm run dev               # http://localhost:5173/ui/
```

Verification, all four of which must pass:

```sh
npm run lint
npm run typecheck
npm run test
npm run build
```

`tests/live-backend.test.ts` is skipped unless a backend is configured. It reads every
client-facing endpoint on a running server and validates the responses with the application's
own schemas — the one class of defect the unit tests structurally cannot catch, because their
fixtures were written from the same reading of the API:

```sh
LIVE_API_URL=http://localhost:8000/api LIVE_API_KEY=... npx vitest run tests/live-backend.test.ts
```

It runs in the `node` environment on purpose. jsdom enforces the browser's same-origin rules,
and the server has no CORS middleware — in a browser the client reaches it through a
same-origin proxy, so running this under jsdom would test the proxy's absence.

## Where things live on the origin

```text
/ui/…    the application   (dev: Vite at :5173/ui/,  production: served by the API)
/api/…   the API as the browser calls it
/…       the API's own paths, unchanged — what every other client uses
```

**The application is under `/ui`, not `/`.** Its page for a feature is `/features/{id}`, which
is also the API's URL for that feature. Sharing an origin, one of them had to move, and it was
not the API — those paths are the contract the existing clients, the tests and the health
checks all address. Before this, loading `/features/<id>` in a browser returned the API's JSON:
deep links, refreshes and new tabs were all broken, while clicking through from the dashboard
worked because that never leaves the page.

**The API is also served under `/api`** for the same reason. It is a second address for the
same routers, not a replacement and not a way around authentication.

In development Vite proxies only `/api` to `VITE_API_PROXY_TARGET` (default
`http://localhost:8000`). The browser and the API therefore share an origin, which is why
**no CORS middleware was added to the server**. In production the API serves the built assets
itself, from `WEB_CLIENT_ROOT` (default `/app/client/dist`, which the image populates); with no
build present the API runs alone.

**Signing in.** The application asks for an email and a password, exchanges them at
`POST /auth/login`, and keeps the returned session token in `sessionStorage` — so the
credential belongs to one browser tab and dies with it. The token is bounded
(`SESSION_TOKEN_TTL_HOURS`, 12 by default) and revocable: signing out revokes it on the
platform as well as forgetting it here.

The API client reads the token at request time. `AppProviders` is constructed before the login
page runs, so capturing it once during startup would leave every later request
unauthenticated.

An HTTP 401 clears the session **in the API client**, once, where `fetch` happens — not at
each call site. A session can expire during any request, including a poll or a background
refetch nobody is looking at, and handled per call site it would be handled in the loud places
and forgotten in the quiet ones. `SessionProvider` reacts to the cleared token by showing the
login screen.

**Signing out clears the query cache as well as the token,** and so does signing in. Every
cached query was fetched as the identity that is leaving; without the clear, the next person to
sign in on the same tab sees the previous one's features until each query happens to refetch.
That is a real cross-user leak in the browser and it is easy to miss, so it lives in
`signOut`/`signIn` rather than in a button's handler.

**`VITE_PLATFORM_API_KEY` is retired and no longer read.** Do not reintroduce it. Vite
substituted it at build time, which put a credential in the built bundle and therefore in any
image built from it — pushed to a registry, pulled onto machines, kept in layer caches; that
happened here, from a `client/.env` written for a browser check. And it *won* over the runtime
credential and could not be cleared, so wherever it was set, signing out silently did nothing
and the next person to use the tab was still signed in as whoever the key belonged to.
`.dockerignore` excludes every `.env` at any depth, and a test asserts the bundle carries no
credential.

**What the console hides is a courtesy.** The sidebar's People link, the accounts page's
controls and the Slack and design-source panels are drawn from the permission list
`GET /me` publishes — the named permissions, never a role string, because the server's
`ROLE_PERMISSIONS` is the one authority on what a role grants. Every one of them is checked
again on the route that performs the action.

Provider credentials (`X-OpenAI-Api-Key`, `X-GitHub-Token`) are typed into the form for a live
submission, held in component state for that submission, sent as request headers, and never
stored. The server treats them the same way.

Production `/ui` responses carry a self-only Content Security Policy, deny framing, disable MIME
sniffing, suppress referrers, and disable camera, microphone, and geolocation permissions. The
API docs keep their existing CDN behavior because the CSP is scoped to the application routes.

## Architecture

```text
component → hook (TanStack Query) → api/features.ts → api/client.ts → server
                                                                       → control plane
                                                                       → workflow services
                                                                       → agents
```

`api/client.ts` is the only place that calls `fetch`. It owns the base URL, bearer auth,
per-request provider credentials, `Idempotency-Key`, timeouts, cancellation, error
normalisation and Zod validation. Components never make arbitrary requests.

Errors normalise to a discriminated union. `conflict` (HTTP 409) is its own kind and is **not**
retryable: the platform refusing an illegal transition is an answer, and retrying it three times
would just hide the explanation.

Two conventions are worth knowing before changing anything:

**Statuses are strings, not enums.** The server owns the lifecycle and grows it —
`inspecting_repositories` was added recently — and a client that rejects an unknown status turns
a backend improvement into a broken page.

**Status wording comes from the server.** `GET /console/status-vocabulary` is owned and tested in
`server/api/console.py`; the client never writes its own status copy. It is also what the
dashboard groups by, through each status's `tone`.

## Navigation, and what it is not

Two places: **Features** and **New feature**, with Settings apart at the bottom of the sidebar
and in the account menu. Features is the operational home — every feature the platform has
accepted, from the moment it accepted it.

There is deliberately no Activity section. `/needs-attention`, `/running` and `/completed` were
once their own routes, and each was the Features page with one of the server's own categories
pre-selected: four routes for one screen, with a filter presented as a place. The filter, the
status and the search now live in the query string, so a link still says which rows the sender
was looking at, and the three old URLs redirect to it.

## Settings

Four panels. Three are prerequisites for doing any work — the account, the provider keys the
platform holds, and the saved repositories — and the fourth is not a preference at all.

**Provider credentials** offers two checks against one stored key, because they answer different
questions and can disagree. *Check* resolves the value locally, which is what a changed
encryption key breaks. *Verify* calls `POST /credentials/{provider}/check?verify=true`, which
asks the provider whether it still accepts the key. Verification is opt-in because it puts
somebody's credential on the wire on a button press. The verdict is rendered as the server's
three values: accepted, refused, or *did not answer* — never as reassurance for the third, and a
refusal is shown as a refusal even though `usable` is true beside it. That combination is the
run-190 failure exactly: a GitHub token that had expired at midnight decrypted fine all day
while every clone using it was refused.

Saving a **GitHub** token asks GitHub about it first, and a token GitHub answers no about is not
stored. Three answers refuse a save, and every one is something GitHub stated: it refused the
token; the classic token it described carries neither the `repo` nor the `public_repo` scope; or
it listed no repositories at all. The refusal is the server's sentence, naming the remedy, and
the panel shows it verbatim rather than "the request was not valid". Everything else stores —
including a GitHub that timed out, and a deployment that installs no probe — because refusing a
real token over a provider blip locks somebody out of their own setup. A token that is stored
comes back with what GitHub said: how many repositories it reaches, how many of those are
writable, and any advisories (a classic token with no `workflow` scope cannot push a change to
`.github/workflows/`; a fine-grained token does not publish its own permissions at all, so what
is listed is the *account's* role rather than the token's grant).

**Repositories** offers what that token reaches instead of a URL field, from
`GET /credentials/github/repositories`. A URL field asked for two guesses at once — that the
repository exists at that spelling, and that the token can reach it — and answered both by a run
failing hours later. Repositories already saved are left out; ones the account cannot push to,
and archived ones, are shown **disabled** rather than hidden, because a repository silently
missing from a menu is indistinguishable from one that does not exist. Choosing one prefills the
branch from GitHub's own default for that repository. Having no menu always comes with the
server's sentence saying why — no token stored, no probe in this deployment, GitHub silent —
since an empty menu is indistinguishable from an answer of "you can reach nothing". A repository
saved before this existed, or one on another host, stays pinned into its own menu so its branch
and label remain editable.

The same rule is enforced at `POST`/`PUT /repositories`, so the guarantee is the platform's
rather than one form's: a github.com URL this identity's token does not list as writable is
refused with the remedy that fits — "cannot reach it" and "can see it but cannot push to it"
have different fixes. Only github.com is judged. An enterprise host or any other forge is
outside what the probe can see, and the endpoint says nothing about those rather than refusing
them.

**Models** renders `GET /model-configuration`: for each configured (platform, tier) pairing, the
four roles with the model, effort, output bound and routing reason each resolved to. One pairing
at a time, chosen with a segmented control, because a tier is a whole preset. Expanding a role
gives the reason it exists and the configuration variables behind its values. It is explicitly
**not** editable and says so on the page rather than merely omitting controls — an absent button
is indistinguishable from one somebody has not found. The panel takes `editable` from the
response, so a user-authored setup will not require finding and changing that sentence.

## Feature identity

A feature is `AB-Feature-42`. The server allocates the number at creation from a table whose
primary key does the allocating, so two simultaneous submissions cannot be given the same one,
and it never changes afterwards. It leads the Features table, sits above the title in the
workspace, prefixes every pull request the feature opens, and is what the search matches.

The internal `feature_id` is still the URL segment and the durable key, and the workflow id is
still what an engineer occasionally needs. Both are under **Technical details** in the
workspace header — available, copyable, and clearly not the feature's identity.

## Creating a feature

Two prerequisites, checked through `GET /setup` before the form renders and shown as blocked
states rather than as validation:

1. **Provider credentials**, stored against the account. A feature is accepted and executed
   afterwards, so the work has no request header to read — it resolves what is stored. The
   blocked state names each provider and whether it is configured.
2. **At least one saved repository**, so a feature is chosen from a list rather than retyped.
   A one-time repository is still offered for something not worth saving.

The form itself asks for a title, a problem statement (or an uploaded PRD), and which saved
repositories it touches. The feature id is read-only and unreserved: nothing is allocated by
opening the page. Repository id, display name and role are all derived by the server from the
URL, and a saved repository's `repository_type` is an organisational label that is deliberately
**not** sent as a role — a label somebody typed must not change how the planner orders work.

Submitting navigates straight to the feature, which exists, is queued, and has its reference.
Nothing waits on analysis.

## Clarification answers

Where the platform read the answer in one of the repositories, the question arrives with a
`suggested_answer`, a `suggestion_source` naming what it was read from, and a confidence. The
client prefills the field with it and shows the suggestion separately, so an edited answer can
still be compared against what was proposed. Where there is no grounded answer all three fields
are empty and the field starts blank — **the client never invents one**, because a suggestion
the browser made is indistinguishable from one the platform justified. Nothing is submitted
until somebody presses the button.

## Adding UI for a new artifact

Artifacts render through a registry in `src/components/artifacts/renderers.tsx`:

```ts
export const ARTIFACT_RENDERERS: Record<string, ArtifactRenderer> = {
  my_new_artifact: (payload) => <>{/* read defensively */}</>,
};
```

An artifact type with no renderer falls back to metadata plus raw JSON rather than a blank
panel, so a new artifact type on the server is readable before anybody writes a view for it.
Read payload fields defensively: they are model-produced documents against a schema that grows.

## Adding a new workflow action

```text
control-plane method  →  route in server/api/feature_routes.py
                      →  operation in client/src/api/features.ts
                      →  UI control that calls it
                      →  confirmation for anything significant
```

The client must not decide whether an action is legal. Render the control, let the server
refuse, and show the refusal. If a capability does not exist on the server, do not add a
client-side operation for it — its absence should be a compile error at the call site rather
than a request that 404s.

Pending contract-change artifacts are surfaced above the feature overview. Rejecting one records
a rationale. Approving one pre-fills the complete current immutable contract, requires an
explicit rationale and confirmation, and submits the complete replacement revision the backend
schema requires; affected workstreams are rerun only by the control plane.

## Granting a stopped repository another attempt

The platform stops a repository when its retry budget is spent, and ordinary resume will not
reset that — repeating an attempt on unchanged inputs costs money to reach the same place.
`POST /features/{id}/workstreams/{repository_id}/retry` is the override, and it is the only
path that may raise a retry budget.

It requires an author and a reason, both recorded against the repository in `retry_grants`,
because the alternative trace is an attempt count that has quietly passed its configured limit
— which reads as a platform defect rather than as somebody's decision. The grant raises that
one repository's ceiling and no sibling's, and is not charged to the integration allowance the
contract loop draws on. The re-run then goes through the same child loop, review and
publication as any other attempt; `decide_child_retry` still decides afterwards whether another
may follow.

The control appears only on a repository whose status the server would accept (`failed`,
`review_rejected`). The server remains the authority — that check only avoids rendering a
button whose sole outcome is a 409.

## The documents

`requirements`, `plan` and `contract` are tabs, not entries to find in a list of forty-odd
artifacts. Each is the latest artifact of its type rendered by the ordinary viewer, so the
formatted and raw views, the metadata and the unknown-type fallback are the same everywhere
rather than a second implementation that drifts.

`requirements` shows **both** the submitted PRD and the planner's reading of it. The difference
between them is where a feature goes wrong: showing only the interpretation hides a misreading,
showing only the original hides what was acted on.

Prose in artifacts renders through `components/common/Markdown`, which parses to React elements
and **never produces HTML**. There is no sanitiser to misconfigure — a `<script>` in agent
output has nowhere to become markup. Links are restricted to http(s); a `javascript:` URL is
shown as text rather than linked or hidden.

## Agent history

`agents` derives each agent run from the same timeline events, pairing a start with its own
end keyed by repository so two repositories running at once do not close each other's run.

It shows a duration **only when the events can support one**. A child workstream's `started`
and `failed` events are both written when its result is persisted — the same millisecond, after
the attempt has already run for minutes — so subtracting them yields `0s`. Printing that would
invent a measurement the platform never took, and a reader would believe it.

## The workflow graph, and what its arrows claim

`graph.ts` builds the layered model — the stages before the fan-out, one lane per repository,
the convergence into integration review and pull requests — and `WorkflowGraph` draws it. That
part is unchanged: the geometry is arithmetic, so a feature with one repository and one with
five are the same code.

Nodes and arrows answer different questions, and both are interactive for that reason. A node
says *what state this stage is in* and opens the stage. An arrow says *who moved the feature
here* and opens the execution behind it.

Everything an arrow claims comes from `GET /features/{id}/executions`. Nothing in this client
can produce a model name any other way, which is the point: which model ran, whether one ran at
all, what a retry was asked to change, and what a counter counts are all server answers. The
client's share is three things.

- **`executions.ts`** joins the server's stage vocabulary to this graph's node identifiers, and
  decides what each arrow says. Three rows at most — the attempt count, the handler, then the
  status and the attempt history together — because a four-row loop label overlapped the lane
  below it. A retry carries `↺` and its own count, so the two kinds of arrow are distinguishable
  without colour.
- **`utils/model.ts`** is the only place a model identifier is rewritten for display:
  `gpt-5.6-sol` → `GPT-5.6 Sol`. It knows no model's name — the rules are about the shape of an
  identifier — and the raw value stays available in the drawer's technical details. A name and
  its effort are two tokens, so a narrow column gap can put them on two lines and neither is
  ever broken through the middle.
- **`ExecutionDrawer`** summarises and routes. It never rebuilds the validation output, the
  review findings or the diff: each already has a screen, and two places that could disagree
  about one artifact is one too many.

The labels are real buttons positioned over the SVG rather than SVG text, which is what makes an
arrow focusable, tab-ordered and readable by a screen reader; the drawn path carries a wide
transparent hit stroke so clicking the arrow itself works too. Overlapping labels are nudged
apart rather than dropped — a busy graph must not lose execution information — and
`tests/executions.test.tsx` covers the cases where the honest label is "no model": a
deterministic validator, a transition nobody has routed yet, a retry the reliability logic
refused.

## Testing against what the platform really sends

Three tests read saved responses from a running stack rather than hand-written fixtures, and
they are the ones that keep finding things:

- `tests/artifacts.test.tsx` renders one real artifact of each of the twelve types and fails
  when a populated field does not reach the page. The renderers read fields by name, so a field
  nobody named vanishes silently — which had happened to user stories, merge order, the
  review's per-requirement checks, the created/modified/deleted file list, and an entire
  rollback plan.
- `tests/detail.test.tsx` renders a real workstream through the repository detail.
- `tests/live-backend.test.ts` parses every endpoint's live response with the app's own schemas.

A fixture written by hand contains exactly the fields its author remembered, which is why these
use saved real payloads. Refresh them from a running stack and check for credentials first;
the committed ones were checked.

## Progress and notifications

`stages.ts` derives where a feature has got to from its own artifacts and workstream statuses,
and `WorkflowMap` draws it. The fan-out stage has one branch per repository the feature
actually has — one, or five — so nothing assumes a frontend/backend pair.

A stage counts as reached when its output artifact exists, not from the feature's status name.
A feature can stop needing a person having already written a contract, and drawing that
contract as never reached would misdescribe where the work stopped. A stage that was never
reached is `pending`, not `stopped`: the integration review did not go wrong, it never
happened.

`notices.ts` turns lifecycle events into the short list of things somebody may need to act on.
It is an allowlist of event names **taken from the server** (`api/feature_control_plane.py`,
`storage/feature_store.py`) rather than guessed from the status vocabulary — guessing produced
`child_workflow_approved`, which does not exist, and a completed feature went on reporting both
its repositories as stopped. Unknown events are ignored, so a new internal step does not start
shouting at people.

One notice per subject — each repository, plus the feature itself — newest wins, so later news
supersedes earlier news instead of sitting beside it. Notices render inline with
`aria-live="polite"`, never as a toast: a feature can wait hours for an answer, and a notice
that vanishes after four seconds is no use to whoever opens the page next.

## Real-time updates

There is no event stream. `GET /features/{id}/events?after=<id>` returns lifecycle events after
a cursor, read straight from the indexed event table, and the client polls that — nothing else
polls on a timer. The interval comes from the feature's own status: a finished feature is not
polled at all.

The newest event id is part of the query keys for workstreams, timeline and clarification, so a
new event produces a fresh read. This replaced cache invalidation, which raced: an event
arriving while a dependent query was still fetching was coalesced away and the refresh silently
lost. The REST endpoints stay authoritative, so a duplicated or missed event cannot
desynchronise the screen.
Replayed event ids are deduplicated. Navigating directly between two feature routes resets both
the cursor and retained notices, so one feature can never inherit another feature's history.
If a response fills the 200-event page, the client drains subsequent pages immediately. This
matters for a completed feature: its normal poll interval is disabled, so stopping after the
first page would otherwise leave its terminal state permanently unseen after reconnection.

**Why not SSE.** `EventSource` cannot send an `Authorization` header, and putting the platform
token in a URL is what the security requirements forbid. A fetch-based stream reader could carry
the header but would hold a database connection open per viewer to buy latency this screen does
not need. `useLiveFeature` is the single place to change if that trade ever stops holding.

## Chat

```text
message → server assembles selective context
        → assistant replies, optionally with a typed proposal
        → proposal is stored pending; nothing has happened
        → a person confirms
        → the server atomically claims the pending proposal
        → the same control-plane method the ordinary buttons use runs it
        → the platform's verdict is recorded and shown
```

The assistant never mutates workflow state. Its action list is the four the control plane
exposes (`ANSWER_CLARIFICATION`, `RESUME_WORKFLOW`, `RETRY_WORKSTREAM`, `CANCEL_WORKFLOW`) and it
refuses to propose anything else. Retry proposals require the repository, attempt count,
operator and reason before they become confirmable; confirmation calls the same audited retry
method as the repository control. Every action's arguments are validated both when proposed and
again when confirmed. The atomic claim occurs before the workflow mutation, so simultaneous
confirmations cannot grant the same retry or start the same recovery twice.

Context is assembled in `server/agents/assistant/context.py` rather than dumped: feature state,
workstreams with blocking and current validation evidence, open questions, recent events, and
the latest contract and reconnaissance inline. The latest relevant artifact type is selectively
included when a question asks about a plan, review, failure, pull request, or named artifact.
Everything else is named by id. A completed two-repository feature carries roughly 400 KB of
artifacts, and sending all of it is slower, dearer and worse at answering.

A deployment without a model configured answers 503, and the client says the assistant is not
configured rather than showing an error.
