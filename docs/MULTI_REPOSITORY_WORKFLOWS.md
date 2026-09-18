# Multi-Repository Workflows

`POST /features/start` creates a parent feature workflow. It accepts one structured PRD and any
number of repository specifications. The existing `/workflow/*` routes remain the single-repository
API, with their schemas and status vocabulary unchanged; what answers them is this same path,
running one repository.

The call returns as soon as the feature, its `AB-Feature-N` reference and one durable queue entry
have committed — `status: "pending"`, meaning accepted and queued. Execution happens on a worker
that claims the entry under a lease, so nothing below this line is something a caller waits for.
See [the API reference](FEATURE_WORKFLOW_API.md) for what that means for credentials.

The parent Product Manager runs once, then every target repository is cloned and read before
planning — see [repository reconnaissance](REPOSITORY_RECONNAISSANCE.md). Clarification happens
after that read, so the human is asked in one round about both what the product document left
open and what the checkouts contradict, rather than being asked to guess at conventions nobody
had looked for yet. The Technical Planner then runs once against that evidence and emits an
approved immutable integration contract and a repository execution plan. Only then are child repository workstreams
allowed to fan out. A child is identified as `{feature_id}:{repository_id}`, uses a dedicated
workspace beneath `WORKSPACE_ROOT`, and uses a deterministic safe branch:

```text
/workspaces/{feature-id}/{repository-id}
ai/{feature-id}/{repository-id}/{feature-title}
```

Repositories with the same `implementation_order` may run in parallel. A later order depends on
earlier workstreams. A repository review failure retries only that child; a successful sibling is
not rerun. The parent waits only for required workstreams. Optional workstreams do not block the
parent unless they own a required contract section.

## Start a mock feature workflow

Mock mode is the default. It makes no OpenAI or GitHub calls and is the right first integration
test for an API client.

```sh
curl --request POST http://localhost:8000/features/start \
  --header "Authorization: Bearer $TOKEN" \
  --header "Idempotency-Key: login-audit-001" \
  --header "Content-Type: application/json" \
  --data @feature.json
```

`feature.json` contains a normal structured `prd` plus `repositories`:

```json
{
  "feature_id": "login-audit",
  "execution_mode": "mock",
  "repositories": [
    {"repository_id": "backend", "name": "API", "role": "backend", "repository_url": "https://github.com/acme/api.git", "default_branch": "main"},
    {"repository_id": "frontend", "name": "Web", "role": "frontend", "repository_url": "https://github.com/acme/web.git", "default_branch": "main"}
  ]
}
```

Add the complete structured `prd` object from the normal workflow start request; the abbreviated
object above shows only the new repository fields.

## Enable live repositories

Set `execution_mode` to `live`. Because a feature is executed after its request has been
answered, a live **start** requires the calling identity's provider credentials to be stored
(`PUT /credentials/{provider}`); it is refused with `422` naming what is missing otherwise, and
`GET /setup` reports the same thing. Every other state-changing request still accepts headers,
which take precedence over anything stored:

```http
X-OpenAI-Api-Key: <request-scoped OpenAI key>
X-GitHub-Token: <request-scoped GitHub token>
```

Repository URLs must be token-free GitHub HTTPS URLs. The live runner normalizes a missing `.git`
suffix, clones only beneath `WORKSPACE_ROOT`, creates the generated non-default branch, and never
force-pushes. The GitHub token must access every selected repository. See
[Deployment](DEPLOYMENT.md) for staging requirements.

The first live run should always use a disposable GitHub organization and branches. Live mode opens
draft PRs by default. It does not merge or deploy them.

## Recovery and cancellation

Each repository child has an independent operation-journal scope. A completed clone, commit, push,
or PR operation is reused after a parent process restart; an uncertain remote operation blocks
readiness for reconciliation rather than being recreated. Resume an interrupted, non-clarification
feature with `POST /features/{feature_id}/resume`, `{"answers": []}`, and fresh provider headers.
If the feature is cancelled while one child is active, completed siblings remain recorded and the
coordinator will not start integration review or new PR mutations. Review
[recovery semantics](RECOVERY_AND_IDEMPOTENCY.md) and the [live runbook](LIVE_EXECUTION_RUNBOOK.md)
before retrying a feature.

Each plan also records repository-scoped requirements, explicitly shared responsibilities, and
out-of-scope requirement IDs. Child retry runs only the failed repository and obtains fresh
validation for its new checkout revision; completed siblings are not revalidated. See
[scoped repository review](SCOPED_REPOSITORY_REVIEW.md).

Each child also has a repository preflight, explicit implementation expectations, completion
evidence, separate retry counters, and meaningful-progress indicators. A required child blocked by
repository setup transitions the parent to `FAILED_REQUIRES_HUMAN`; successful siblings are
preserved, later workstreams are not started, integration review is skipped, and no PRs are created.
See [repository preflight](REPOSITORY_PREFLIGHT.md) and [retry strategy](RETRY_STRATEGY.md).
