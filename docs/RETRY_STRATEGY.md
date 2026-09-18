# Retry strategy

Child retries are classified before another Engineer invocation:

- `implementation_missing`
- `validation_source_failure`
- `validation_configuration_failure`
- `dependency_installation_failure`
- `test_infrastructure_missing`
- `review_scope_failure`
- `contract_mismatch`

The resulting retry plan records root cause, a required strategy change, source areas to inspect,
the prior approach to avoid, and commands to rerun. The Engineer receives that plan, previous
findings, unresolved expectations, current revision, and preflight data; a retry never receives an
unchanged generic prompt.

Budgets are separate: implementation retries default to 4, source-validation retries to 2, and
repository setup retries to 1. Setup failures do not consume implementation retries. An unsafe or
undeclared dependency/configuration repair stops the child as `FAILED_REQUIRES_HUMAN` rather than
guessing a package change.

Each attempt persists meaningful-change status, explanation, production-diff fingerprint, and its
previous fingerprint. New production implementation, material required-source changes, resolved
blocking configuration, executable tests for implemented behavior, or a relevant validation
improvement count as progress. Format/comment-only edits and repeated tests-only changes do not.
Two consecutive no-progress retries stop automatic retry and require human intervention. Successful
siblings remain untouched, and integration review never starts after a required child setup failure.

## Convergence: repeating is not attempting

Budgets bound how many attempts a workstream gets. They say nothing about whether it is using
them. Four rules answer that, and all four live inside `decide_child_retry` — the caller
supplies evidence about the attempt just finished and applies the verdict, and reaches no
conclusion of its own.

| Rule | Bound | Why that number |
| --- | --- | --- |
| Identical resubmission | 2 | Repeating once is a model that did not read its diagnostic; twice is one that cannot act on it. Stopping on the first repeat ended a workstream one documentation edit from approval. |
| Return to an earlier submitted state | 2, never reset | The repeat rule looks back one attempt, so `A -> B -> A -> C -> A` clears it every other time. This one counts arrivals at a state already submitted, however much variety there was on the way round. |
| Repeated diagnostic signature | 3 consecutive | The same defect — same exception, same line — however the sentence around it is reworded. Three rather than two because reducing a diagnostic to a signature is lossy, and stopping a converging workstream early costs more than one extra attempt. |
| Deterministic gate exemption | — | A check that emits the same sentence for as long as its condition holds is consistent, not a model ignoring feedback. Exempt from both repeated-diagnostic readings. |

Two flags modify what counts as a repeat, and neither may be simplified away:

- **`source_rejected_before_commit`.** The commit gate resets the workspace, so the fingerprint
  taken afterwards describes the state the attempt started from rather than the source it wrote.
  A rejected attempt is therefore not read as a return to an earlier state — and it does not
  *clear* a count of repeats either, because it says nothing in that direction.
- **The three exempt classifications.** `dependency_installation_failure` and
  `validation_capacity_failure` never have to show a production diff: the change they ask for is
  to the machine or the command, not to application source. `validation_configuration_failure`
  and `test_infrastructure_missing` are refused outright, ahead of every other rule, because
  coding again cannot fix a checked-in configuration.

Until task `30-`, three of these rules ran in `_run_one_child` *after* `decide_child_retry` had
returned `should_retry=True`. They stopped the workstream anyway and rewrote its reason, which is
why six production rows record their own stop as "Attempt N may proceed with a changed strategy".

## Every retry authority in this platform

Fourteen components can cause work to be attempted again. That is not a defect — they bound
different things, at different layers, over different units of work — but reconstructing the
list by grep is, so it is written down here. A change to any bound belongs in this table.

| Authority | Owner | Counter | Bound | Why it is separate |
| --- | --- | --- | --- | --- |
| Child workstream retry | `tools/retry_strategy.py::decide_child_retry` | `implementation_retry_count`, `validation_retry_count`, `repository_setup_retry_count`, `integration_retry_count`, and `retry_count` against the review-cycle ceiling | 4 / 2 / 1 per class; 12 review cycles | The only authority on "may this workstream attempt again". Everything below bounds something that is not that question. |
| Integration remediation | `workflows/feature_workflow.py::_integration_remediation_decision` | `integration_retry_count` | `max_integration_review_cycles` (5) | Not separate — it routes through `decide_child_retry`. Listed because it is the edge that used to bypass it, and a repository was coded twelve times against a ceiling of five. |
| Child-loop provider faults | `_run_one_child` | `infrastructure_faults`, per attempt | 4, with exponential backoff | A provider fault says nothing about the repository's code, so it earns the attempt again rather than the workstream's end. Consumes no coding budget by design. |
| Feature-stage provider faults | `_despite_provider_faults` | `faults`, per stage | 2 | The same rule for product manager, reconnaissance, planner and clarification grounding. Two rather than four because one maximum-effort planning call is long enough that four retries would put a feature hours from the person who has to decide about it. For the journaled calls the loop sits inside `_journaled_planning_call` — their per-call nonce means a journal budget can never re-drive them, so the retry is in-place and each try writes its own row. |
| Queue entry attempts | `services/feature_queue.py` | `FeatureQueueEntry.attempt` | 3 | Bounds how many times a *claim* may be redelivered after a transient provider fault, not how many times work may be attempted. Its unit is the entry. |
| Crash continuation | `storage/feature_store.py::_claim_disposition` | queue claim attempts vs `max_attempts` | the same 3 | Bounds how many times a feature whose executor *died* is picked back up. A feature that has crashed as many times as its entry allows is systematically wrong; the sweep records the terminal state. |
| Step ceiling | `services/feature_queue.py` | steps taken by one run | `MAX_STEPS` (200) | A claim advances a feature one step, so the loop between claims is durable and needs something able to end it. Bounds repetition of a *step*, not of an attempt. |
| External operation attempts | `services/external_operations.py::ExternalOperationExecutor` | `ExternalOperation.attempt` | per call site: 6 for pull-request creation, 3 for label / reviewer / comment mutations, 1 by default | Bounds replay of one *effect* against a provider, gated on reconciliation proving the effect absent. Nothing here decides whether the work was right. |
| Cross-link comment driver | `services/journaled_github.py::add_comment` | the loop's own try count | the same 3 as the comment's operation budget | The journal's budget only advances when something calls `run()` again, and nothing re-drives a publication's comment — the publisher calls once and absorbs. This loop is that re-entry: admitted per fault by `is_transient_provider_fault`, safe only because the action and reconcile prove presence or absence on the provider first. |
| Provider SDK retries | `adapters/llm_adapter.py`, via `configs/agent.yaml` | the OpenAI client's own `max_retries` | 3 per agent, 2 for the reviewer | Transport-level. It retries an HTTP request; every authority above retries a decision. |
| Malformed-response repair | `agents/planner`, `agents/reviewer`, `agents/engineer` | one repair per response | 1, one-shot | A response rejected by deterministic validation is re-asked once with exactly what was wrong. It is not an attempt: nothing was executed and no budget is spent. |
| Operator retry grants | `workflows/feature_workflow.py::grant_additional_attempts` | `granted_extra_attempts`, per repository | whatever a person granted | The one path that may raise a budget, and it is an override rather than a policy. Scoped to one repository so a sibling that spent nothing inherits nothing. |
| Repository repair grants | `services/repository_repair.py` + `approve_repair` | the granted attempt | 1 per approved repair | An approved repair changes the repository's checked-in setup, which is new information the refused attempt did not have. Granted by a person, never by the platform. |
| Feature action attempts | `services/feature_actions.py::FeatureActionService` | `FeatureAction.attempt` | `max_attempts`, 1 by default | Bounds replay of a *durable action a person asked for*. An action whose executor died is reconciled rather than replayed, which is the whole reason it counts separately. |
| Clarification rounds | `FeatureWorkflowSnapshot.max_clarification_rounds` | `clarification_rounds` | 10 | Bounds how many times a feature may go back to a person with questions. Nothing is retried; the requirement is being changed. |

## When a workstream stops

`decide_child_retry` is the only place the retry verdict is produced, and it also owns what is
asked when the verdict is no: every refusal carries an `operator_question`.

`triage_stopped_workstream` then attributes the stop to a cause, using evidence the attempts
already produced. It decides nothing about retrying — by the time it runs, that answer is already
no — and exists because "the requirement, the repository, or the budget: you decide" hands a
person three possibilities and the job of working out which:

| `terminal_cause` | Evidence | What the question asks |
| --- | --- | --- |
| `repository_setup` | The checkout was blocked before any code was written | Repair the repository, or run without the validation it lacks? |
| `never_implemented` | No attempt produced a production source change | Does what the requirement asks for exist in this repository at all? |
| `blocker_unresponsive` | Source changed; the blocking diagnostics did not | What has to be fixed here before this work can land? |
| `budget_exhausted` | Source changed and the blocker moved | Falls back to the open question |

The last row is deliberate. Where the evidence distinguishes nothing, triage asks openly rather
than guessing a cause: a confidently wrong attribution sends someone to repair a repository that
was never broken.

The question is appended to the child's `blocking_issues` as well as recorded in the result's
metadata alongside `terminal_cause` and `terminal_evidence`, because that list is what already
reaches the parent, the `/features` API and the console. A question recorded only in metadata is
a question nobody is shown. A workstream that may still continue, and one that was approved, are
asked nothing.
