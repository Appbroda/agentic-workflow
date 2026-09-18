# Recovery and Idempotency

## What recovers what

Three sweeps run at startup and every thirty seconds afterwards, in this order, because each
reads what the previous one settled:

1. **External operations.** `RecoveryService` decides every operation whose executor stopped
   without recording a result.
2. **Durable actions.** An action's verdict is read off its operations, so those settle first.
3. **Feature runs.** A feature is only abandoned if nothing holds a live lease on it, so the
   action sweep has to have released dead executors' leases before this one looks.

## Operations: settled locally, deferred, or put to a person

`RecoveryService` holds no credentials and never will — it takes a journal, a workspace root
and a process runner, and that is deliberate, so nothing has to persist a caller's provider
tokens for startup recovery. What it can conclude is therefore bounded, and every operation
type has an explicit disposition rather than a fallthrough:

| Disposition | Types | What the sweep does |
| --- | --- | --- |
| Local evidence | clone, commit | Verifies the checkout — a valid clone against its `origin`, a commit by its saved SHA or by its durable parent SHA and exact message — and records success. |
| Workspace-local | coding, review, validation, formatter, file writes, dependency install | Ends the operation without ever calling it externally unknown. There is nothing outside the checkout to reconcile. |
| Deferred | push, PR create, labels, reviewers | Left `awaiting_reconciliation`, because the next credentialed request through the same code path can prove what happened and this sweep cannot. |
| Terminal benign | the PR cross-link comment | Ended. The publisher does not gate on it, and an unconfirmed one is not worth a person's attention. |
| Manual review | anything whose replay budget is spent, or whose local evidence is absent | `UNKNOWN_EXTERNAL_STATE` with `MANUAL_REVIEW_REQUIRED`. |

**Deferral is the normal path, not a problem.** Readiness does not count deferred operations
and one interrupted push no longer refuses unrelated writes across the deployment. The
reconcile callbacks that settle them — a remote branch SHA matching the intended local SHA, an
existing matching head/base pull request — fire when a credentialed request re-enters the same
code path, which for a live feature is the ordinary course of events.

A deferred operation whose owning feature has reached a terminal status is the one case where
no such request is coming. After `DEFERRED_OPERATION_SETTLE_AFTER_SECONDS` (a day by default)
the sweep converts it to `UNKNOWN_EXTERNAL_STATE` with `MANUAL_REVIEW_REQUIRED`, so a person is
asked once rather than never. Nothing is guessed about the provider and nothing is settled
silently. `GET /features/operations/unresolved` is the queue those land in.

## Features: a crashed run is continued, not tombstoned

Every safe point is checkpointed — before the clone, before coding, after validation, before
and after the pull request — and those checkpoints are now consumed automatically.

When a worker dies, its queue claim lapses and another worker claims the entry. That claim
lands on a feature which still says it is executing, and the platform answers one of three
ways:

- the feature has moved past this claim → the claim is dropped;
- the feature is resting where a person owns it (`waiting_for_human`, `contract_ready`,
  `changes_requested`, `ready_for_pull_requests`) → the claim is dropped;
- the feature says it is executing and nothing is executing it → **the run is continued from
  its checkpoint**, and a `feature_run_continued` event records it.

"Nothing is executing it" is established, not assumed: no live action lease, and no queue
claim other than this one. It is the same query the abandoned-run sweep uses to decide what to
leave alone, so the two cannot disagree.

A continuation goes to checkpoint recovery, not to a fresh start. It walks Technical PRD →
contract → plan and re-enters only the incomplete part: the product manager, reconnaissance and
the planner are not re-run when their artifacts exist, an approved or completed repository is
not re-run, no counter is reset, and a persisted integration verdict is reused rather than
bought again. Because a continuation re-enters the publication path with credentials, it also
reconciles its own deferred push or pull request.

The queue entry's attempt budget bounds this. A feature that crashes as many times as its
entry allows is not continued again; `reconcile_abandoned_runs` gives it a terminal status
whose diagnostic says it was continued and did not survive.

## Resume

`POST /features/{id}/resume` with an empty `answers` list still works and still means "continue
from the checkpoint". It is no longer the only way a crashed run is picked up, and it is what
a person uses to restart a feature that stopped for a reason other than its process dying.

Resume eligibility is per repository. A repository that has exhausted its retry budget or
carries a refusal is not re-run and keeps its status, its refusal and its counters; its
siblings still run. A feature where every repository has reached a terminal retry decision is
left exactly where it is and records a `feature_resume_found_no_eligible_workstreams` event,
rather than returning unchanged while the queue entry closes `succeeded`.

## Idempotency

The journal is the exact durable boundary around clone, coding, validation, commit, push and
pull-request side effects: intent is written before the effect, with a deterministic key, and
successful operations are reused rather than replayed. That is what makes a continuation safe
— one clone per repository, one commit per approved attempt, one push per branch, one pull
request per repository, however many times a run is picked back up.
