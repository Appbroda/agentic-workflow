# Integration Contracts

`009_integration_contract.json` is the feature's single source of truth. It contains stable
operation IDs, paths, request/response JSON Schemas, authentication and authorization behavior,
error contracts, events, environment compatibility, and rollback policy. REST and mixed contracts
also contain an OpenAPI 3.1 projection; live child workspaces receive it as `openapi.yaml`.

The feature planner approves contract version `1.0.0` before any child starts. Each child receives
a read-only serialized copy and its assigned consumed/implemented sections from
`010_repository_execution_plan.json`. A child cannot mutate the approved contract. If a generated
OpenAPI projection is changed, the child emits `013_contract_change_request.json` and the parent
pauses for a human contract owner.

To approve a change request, provide a complete replacement contract with a new version:

```text
POST /features/{feature_id}/contract-change-requests/{request_id}/approve
```

The request body includes `resolution` and `updated_contract`. Before any child commit exists, the
replacement is validated, stored as a new immutable contract artifact, and every consumer is rerun;
no approval against the superseded contract survives. After any child commit exists, an in-place
revision is refused with `failed_requires_human`: the operator must start a new feature revision on
fresh branches, because reviewing only a later delta cannot prove that the complete branch conforms
to the replacement contract. Rejecting a request also records the decision and changes the feature
to `failed_requires_human`; the platform never guesses or silently relabels contract changes.

The contract generator exposes backend schema definitions and frontend operation IDs without
requiring Node.js during normal tests. Repository-specific tooling may use the OpenAPI projection
to generate validation models or a typed frontend client. Integration review verifies that all
required contract owners are review-ready before PR creation.
