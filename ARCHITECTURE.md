# Architecture

The platform has one orchestration path, and two doors into it:

```text
/features/*  → one feature PRD → repository reconnaissance → shared contract/plan → child repositories → integration review → coordinated PRs
/workflow/*  → the same, with one repository
```

`/workflow/*` keeps its own request and response schemas, including its older and smaller status
vocabulary, and `api/workflow_control_plane.py` is the only place that vocabulary exists: a
single-repository request becomes a feature carrying one `RepositorySpec`, and what answers it is
the feature path. There is no second engine, no second store, and no second set of convergence
guards to keep in step -- which is what having two of each cost, because every reliability fix of
the preceding two months applied to only one of them.

Reconnaissance reads every target checkout before the plan is written, so requirement scope,
expected source areas, and the contract are decided against what the repositories contain rather
than against what the PRD assumed. Its artifacts record which conventions each repository
actually has and which requirement premises the checkout contradicts; a plan carries
`planned_from_checkout_evidence` so a grounded plan can be told from a blind one. See
[repository reconnaissance](docs/REPOSITORY_RECONNAISSANCE.md).

The parent feature workflow is persisted in PostgreSQL. Repository specifications, independently
tracked child states, contracts, contract change requests, integration reviews, PR records,
artifacts, events, and idempotency records are stored separately. Redis serializes feature lifecycle
operations. OpenAI and GitHub credentials are accepted as request headers. A deployment
configured with `SECRET_ENCRYPTION_KEY` may also keep them per identity, sealed at rest and
bound to their owner; a request header always wins over anything stored, and without that key
the platform persists no provider secret at all. Requests resolve to an actor -- a person
holding a token, or the shared platform key as a named administrative identity -- and every
workflow-changing action is recorded against one as a durable, leased, reconciled record.
See [durable actions and repository repair](docs/DURABLE_ACTIONS_AND_REPAIR.md).

Read [Multi-Repository Workflows](docs/MULTI_REPOSITORY_WORKFLOWS.md) for the execution model and
[Integration Contracts](docs/INTEGRATION_CONTRACTS.md) for the contract boundary.

Repository validation is selected from each checkout's manifests and scripts. Validation operation
keys include the committed and uncommitted repository revision, so a reviewer only consumes results
for the current code. Repository child reviews receive scoped requirements and contract
responsibilities; integration review is the only cross-repository gate, checking contract
conformance from child result metadata and — where a diff source and model are configured —
reading the changed source of every repository together to review the seam between them. See
[repository validation](docs/REPOSITORY_VALIDATION.md) and
[scoped repository review](docs/SCOPED_REPOSITORY_REVIEW.md).

Which model does a piece of work is configuration, not code. The platform names roles --
`REASONING`, `CODING`, `REVIEW`, `SCOPED_FIX` -- and a deployment decides what each one is. An
initial implementation uses coding; independent review uses review; planning uses reasoning. A
remediation the existing failure classifier attributes to an Engineer uses scoped fix only when
every finding is a deterministic localized mechanical correction. Substantive, mixed and
ambiguous work stays on coding. Routing answers only which model executes an attempt the retry
policy has already granted -- it cannot grant one, raise a limit, or cycle models to create
progress.
See [model routing](docs/MODEL_ROUTING.md).
