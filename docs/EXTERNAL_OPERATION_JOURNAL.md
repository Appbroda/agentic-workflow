# External Operation Journal

Every live clone, branch, coding/file update, validation command, commit, push, and pull-request
mutation has an `external_operations` row before the adapter is called. The row has a deterministic
idempotency key:

```text
{feature-or-workflow}:{child}:{repository}:{operation-type}:{logical-step}:{input-fingerprint}
```

The database enforces uniqueness. Each operation records `PENDING → STARTING → RUNNING → terminal`
transitions in `external_operation_events`; attempts are kept separately in
`external_operation_attempts`. The journal contains only identifiers, hashes, paths, safe Git/PR
references, and bounded error summaries. It rejects credential-shaped metadata keys.

Validation journal entries additionally store the repository revision, command fingerprint, and
current/superseded relationship. Their idempotency input includes the HEAD/index/worktree/untracked
fingerprint and validation environment/configuration fingerprint. Earlier validation entries remain
append-only but are never reusable for a changed revision.

`SUCCEEDED` operations are reused on restart. `STARTING`, `RUNNING`, and cancellation-in-progress
records are never blindly repeated. A stale or uncertain remote operation becomes
`UNKNOWN_EXTERNAL_STATE`, which blocks readiness until an operator reconciles it. This deliberate
failure mode is safer than creating a second commit, push, or PR.

See [Recovery and idempotency](RECOVERY_AND_IDEMPOTENCY.md) for reconciliation behavior.
