# Architecture Details

The authoritative overview is [ARCHITECTURE.md](../ARCHITECTURE.md). The one orchestration path is
implemented in `workflows/feature_workflow.py`; its production composition is
`services/feature_runtime.py`. The older single-repository `/workflow/*` routes keep their request
and response schemas and are served from the same path by `api/workflow_control_plane.py`, which
translates between the two status vocabularies at the API boundary and nowhere else. That surface
is deprecated and administrator-only: it has no owner in its schemas, so it cannot be scoped to
its caller and is confined to `platform-admin`'s workspace instead.

## Identity and workspaces

Each person has an account with a password, and a feature belongs to whoever submitted it.
`feature_workflows.owner_id` is the root of that: thirteen tables cascade from `feature_id`, so
one column at the parent makes all thirteen filterable. The check is one predicate in the SQL
`WHERE` of `_require_model` in `storage/feature_store.py`, and routes reach the store only
through a `ScopedFeatureControlPlane` that supplies the request's workspace on every call.

The decisions behind it — including why no `owner_id` was ever rewritten, and why the queue's
`requested_by` is the feature's owner rather than the acting actor — are in
[AUTHENTICATION_AND_WORKSPACES.md](AUTHENTICATION_AND_WORKSPACES.md).

## Live execution reliability

`storage/external_operation_store.py` is the durable boundary around every live side effect. Its
operation, attempt, and event tables are shared by every workstream. `services/recovery_service.py` runs at startup; `services/cancellation.py` combines
durable PostgreSQL state with a short-lived Redis signal; and `services/process_runner.py` owns
interruptible local subprocess groups.

Feature snapshots persist orchestration safe points, while every child has an independent
operation-journal scope around clone, coding, validation, commit, push, and pull-request effects.
The journal is the recovery source of truth when an external effect succeeds before the parent
snapshot can be advanced. Validation is repository-aware: manifest inspection builds fixed commands
and revision fingerprints make journal reuse safe only for the exact checkout state. Child reviews
receive scoped workstream requirements; cross-repository requirements remain the integration
reviewer's responsibility.
