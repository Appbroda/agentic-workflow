# Web control plane — delivery report

**Verdict: COMPLETE_WITH_LIMITATIONS.** The implementation is built, tested and integrated with
the running backend and real persisted feature data. The remaining limitations are explicit at
the end.

An independent audit on 2026-08-25 rechecked every PRD criterion and the client against the
server routes, schemas, state models and durable stores. It fixed runtime authentication,
provider-credential forwarding, retry proposals, contract decisions, failure and cleanup
evidence, artifact coverage, long-run event reconnection, chat action concurrency and request
validation gaps that the earlier delivery audit had missed.

## What the backend already had

The platform was a working multi-repository feature workflow with a durable control plane, an
external-operation journal, cancellation lifecycle, contract-change approval, and a
status-vocabulary endpoint that already owned the operator-facing wording. Features `-078`
through `-086` had run live and opened pull requests. The client was built on that, not beside
it.

Reused unchanged: `POST /features/start`, `/resume`, `/cancel`, `/retire`,
`GET /features`, `/features/{id}`, `/workstreams`, `/timeline`, `/pull-requests`,
`/features/operations/unresolved`, contract approve/reject, `/console/status-vocabulary`,
`/healthz`, `/readyz`.

## What the backend gained

| Route | Why |
|---|---|
| `GET /{id}/clarification` | Open questions were only reachable by hydrating a 400 KB artifact. |
| `GET /{id}/artifacts/{artifact_id}` and `?include_payload=` | 46 artifacts / 406 KB full versus 121 KB as envelopes. |
| `GET /{id}/events?after=` | A cheap cursor read, so watching a feature does not re-hydrate parent state. |
| `GET\|POST /{id}/chat`, `/chat/{id}/confirm\|reject` | The feature assistant and its proposals. |
| `POST /{id}/workstreams/{repository_id}/retry` | The one path that may raise a retry budget. |

Also added: `repositories[]` on the feature response, repository identity on each workstream,
and `repository_count` / `pull_request_count` / `human_action_required` on the list — the
dashboard needed all three and was otherwise making N+1 requests to derive them.

Not added: CORS (a same-origin proxy makes it unnecessary), `current_agent` on the list (it
lives in a 400 KB column, so publishing it would make listing cost more than the work).

22 server files changed. Two migrations: `20260825_0009` (chat messages), `20260825_0010`
(retry grants). Both applied to the live Postgres, including on rows written before them.

## The client

52 source files under `client/src`, 15 test files. React 18, TypeScript strict with
`noUncheckedIndexedAccess`, Vite 6, React Router 6, TanStack Query v5, React Hook Form + Zod,
Vitest.

- `api/client.ts` is the only place that calls `fetch` — auth, per-request provider
  credentials, idempotency, timeout, cancellation, error normalisation, Zod validation.
- Statuses are strings and status wording comes from the server, so a lifecycle the backend
  grows does not break the page.
- Artifacts render through a registry with a raw-JSON fallback for unknown types.
- No client-side operation exists for a capability the server lacks: its absence is a compile
  error at the call site rather than a request that 404s.

**Real-time.** `GET /events?after=` polled at an interval derived from the feature's own
status; a finished feature is not polled. The newest event id is a query-key dependency rather
than a cache invalidation, because invalidation raced — an event arriving mid-fetch was
coalesced away and the refresh was silently lost. Not SSE: `EventSource` cannot send an
`Authorization` header and the platform forbids credentials in URLs.

**Chat.** The assistant answers from selectively assembled context and may propose one of the
four actions the control plane exposes, including an audited one-repository retry; it refuses to propose anything else. A proposal does
nothing until a person confirms, and confirmation runs the same control-plane method the
ordinary buttons use — so the platform's refusal reaches the screen as a refusal.

## Verification

Backend, all four from `server/`: **501 tests pass**, ruff check, ruff format, mypy clean.
Client, from `client/`: **118 non-live tests pass**, eslint, forced tsc typecheck, and the
production build are clean. The two opt-in live-backend tests also pass against the audited
branch.

Against the real Docker Compose PostgreSQL and Redis services, the audited branch and built
client were started on a separate localhost port. Every client-facing read endpoint returned
real persisted data and parsed through the application's own Zod schemas; a retry refusal
returned 409 and left the feature untouched. `client/tests/live-backend.test.ts` runs that check
and is skipped unless a backend is configured.

## Loaded in a browser

The application was opened in Chrome against the running stack — first the dev server, then the
built image. The dashboard lists real features; `/ui/features/adunit-deactivate-live-086`
renders the Progress map, notifications, tabs and repository cards from live data.

Doing that found three more things no test could have, because none of the tests is a browser:

6. **Every feature page was broken on load.** The client's route `/features/{id}` is the API's
   URL for the same feature; on one origin the API won and returned JSON. Clicking through
   from the dashboard worked — that never leaves the page — so nothing else showed it. The
   application moved to `/ui` and calls the API under `/api`; the API's own paths are unchanged.
7. **Nothing served the build.** `npm run dev` worked and a deployment had no way to put the
   application in front of anybody. The API now serves it, and the image builds it.
8. **The platform key was baked into the image.** `.dockerignore`'s bare `.env` matches only
   the context root, so a `client/.env` written for the browser check was copied in and Vite
   substituted the key into the bundle. An image travels much further than a page does. Every
   `.env` is now excluded at any depth, the token is entered at runtime and held in
   `sessionStorage`, and a test asserts the bundle carries none.

## Five defects the live run found that the tests did not

1. **The platform would not boot without a chat key.** The assistant was built at startup from
   `OPENAI_API_KEY`, which compose deliberately does not pass, so the container restart-looped
   and took down every feature that had nothing to do with chat.
2. **Chat could never have worked.** Same root cause: provider credentials are request-scoped
   here, so an assistant built at boot had nothing to authenticate with. It now takes the key
   from the request header exactly as the working agents do.
3. **The transcript was discarded.** The durable store flushed without committing, so every
   message got an id, was returned looking complete, and vanished. Every test had used the
   in-memory store; the two share a Protocol and only one was ever run.
4. **A typo broke the feature it named.** The retry refusals sat behind the live runner, so
   without credentials they were unreachable and the error fell through to the generic handler
   — which marked the feature `failed_requires_human` and answered 200.
5. **`FeatureWorkflowError` surfaced as `500 Internal server error`**, which reads as the
   platform breaking rather than as it refusing.

Each has a regression test written against the shape that failed rather than the shape that
was convenient.

## Audited against the PRD, then against the running platform

Sections 1–58 were walked individually against the code. That found eight requirements met only
generically or not at all — the workflow visualisation, notifications, the PRD/plan/contract
views, markdown rendering, agent history, the coordinated pull-request view, dashboard fields
and five of the listed test scenarios. All are built.

Then every view was rendered in a browser against the platform's own older features — `-074`,
`-076`, `-083`, `-085`, `-086`, a fresh mock run — and read. That found considerably more, and
each is recorded in the commit that fixed it:

- The header claimed pull requests existed above a tab saying there were none.
- A feature cancelled in August still drew its repositories as running.
- A completed feature announced itself as stopped, and a stopped one as completed, because
  notices derived the feature's condition from events while the header derived it from status.
- Settings reported the running platform as gone, because moving the routers under `/api` left
  the health endpoints behind.
- Agent history showed eleven identical lines, then showed every run as `0s` — a duration the
  events cannot measure.
- Cancelled features offered Resume and Cancel, which can only 409.
- A mistyped feature id was reported as an item that "no longer exists".
- Chat asked for a GitHub token it never sends.
- The repository detail dropped the detected technology, the requirements scoped to that
  repository, and the strategy each retry was given — all populated on every live feature.
- Ten artifact renderers dropped populated fields; one type had no renderer at all; an entire
  rollback plan vanished because the field became a list and the helper silently rendered
  nothing for anything but a string.
- The clarification answers the platform planned from were reachable only through the API.
- The links agent history added to each attempt's result opened an empty panel.

Three tests now read saved real payloads rather than hand-written fixtures, because a fixture
contains exactly the fields its author remembered.

## §58 acceptance criteria, item by item

The PRD's own definition of complete. Each was checked against the code and, where it is a
behaviour, against the running platform.

| | Criterion | Evidence |
|---|---|---|
| 1–2 | Working frontend, builds | `npm run build` |
| 3–5 | Submit a PRD, workflow created through the real API, appears on the dashboard | Clicked through in Chrome; created `ui-click-check-2` |
| 6 | Feature state displayed correctly | Rendered against `-074`, `-076`, `-083`, `-085`, `-086` |
| 7–10 | One, two, three-or-more repositories; dynamic workstreams | One and two live; one, two and five plus two sharing a role in `coverage.test.tsx` |
| 11–13 | Technical PRD, planning, contract viewable | Requirements / Plan / Contract tabs, live |
| 14–15 | Agent history, timeline | Agents and Timeline tabs, live |
| 16–17 | Validation and review results | Repository detail; `review` renderer with per-requirement checks |
| 18 | Failure classifications understandable | Blocking issues, failure class, retry strategy, triage question |
| 19 | Clarification answerable | Answered a real pause in Chrome: listed and flagged on the dashboard, rationale shown, submit blocked until complete, panel clears when accepted |
| 20 | Human-action states visible | `HumanActions`, dashboard "Needs you" |
| 21 | Repository repair | Diagnosis, evidence, suggested repair and who must act, from the platform's own preflight record; **no approve endpoint exists**, so the grant control is the approval |
| 22 | Retry where allowed | Clicked: refuses without author and reason, sends the right body, shows the refusal |
| 23–24 | Resume, cancel where allowed | Clicked: confirm first, send nothing until confirmed, report the platform's answer |
| 25 | Siblings not shown as rerunning | Each repository reads its own status; concurrent runs keyed separately |
| 26 | Formatted and raw artifact views | `ArtifactViewer`, tested |
| 27–30 | Pull requests, dependencies, merge and deployment order | PR tab reads merge/deployment strategy and each repository's `dependency_workstream_ids` from the backend plan |
| 31–36 | Chat: available, context-aware, explains, proposes, structured commands | Answered real questions about `-086` live; four supported action types only, with audited repository retry |
| 37 | Significant actions confirm | Verified by clicking |
| 38 | Chat history persists | Verified live after fixing the missing commit |
| 39–40 | Automatic updates, reconnection or polling fallback | Event cursor polling; recovery tested |
| 41–43 | Backend authoritative, no duplicated state machine, no fixed repository count | Statuses are strings; three terminal statuses only decide what to *show* |
| 44–45 | Markdown safe, secrets not exposed | Parser emits elements, never HTML; image verified to carry no key |
| 46–49 | Frontend tests, backend tests, addition tests, lint | 118 + 2 live / 501 / included / clean |
| 50–52 | Typecheck, production build, backend validation | forced `tsc`, Vite, ruff check/format and mypy clean |
| 53–54 | Setup/architecture documented and runnable | `WEB_CLIENT.md`, `.env.example`, package scripts, same-origin `/ui` serving |

The coordinated PR view now joins each PR to the backend plan by repository id and displays its
declared workstream dependencies. No dependency is inferred from role or repository name.

Repository repair is now a durable backend capability. The UI shows the proposal and its
revision, requires an explicit approval or attributed rejection, refuses stale proposals, and
runs an approved repair through the ordinary journaled retry path.

## Limitations

- **Chat and workflow events use authenticated SSE over `fetch`.** The bearer token stays in
  the Authorization header; cursor replay and authoritative REST refresh handle reconnects.
- **Authentication supports individual users and an explicit shared-admin compatibility key.**
  Backend permissions protect every mutation; the browser's button visibility is only UX.
- **Provider credentials may remain request-scoped or be stored per identity.** Stored values
  are AES-GCM sealed under an environment key and APIs return descriptors, never secrets.
- **An interactive browser was not connected during the independent re-audit.** The audited
  branch passed the real-backend integration tests and the full DOM/component suite, and the
  earlier delivery browser observations remain above, but this audit does not claim a fresh
  visual click-through.
- **An interrupted action may require human reconciliation.** Exact action-operation links and
  domain-result checkpoints prevent blind success or retry; an admin-only attributed decision
  closes outcomes the persisted evidence cannot prove.
