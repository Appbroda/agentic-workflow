# Cross-Repository Review

After required child repository reviews pass, the Integration Reviewer emits
`012_integration_review.json`. It is a **contract-conformance gate, not a review of the
coordinated change**, and it decides from `011_child_workflow_result.json` metadata alone. It
checks exactly three things:

- every child result records the active contract's artifact ID and version (`critical` if not);
- every repository named in the contract's `owning_workstreams` produced a result (`critical`);
- every owner's result is `approved` and PR-ready (`high`).

The review status is derived only from the severity of those findings.

## Reviewing the seam

Given a diff source and a model, the same gate also reads the changed production source of every
repository together and compares them. This is the only point in the platform that sees more than
one repository at once: child reviews are deliberately scoped to their own repository's
requirements (`agents/reviewer/agent.py`), so a consumer calling a provider with the wrong field
name, method, or auth header passes every other gate.

It looks for one thing — places the changes do not fit together — and is bound by three rules.
A finding must name a repository actually in the feature, because a blocking finding sends that
repository back through its whole engineer/reviewer loop and one routed to nobody is an
integration failure the parent cannot act on. Evidence must quote both sides of the mismatch,
and only the changed files are shown, so a call site in unchanged code is not something it can
see or assume about. `critical` and `high` block; anything less certain is recorded for a human
and blocks nothing.

**When it does not run.** Without a diff source or a model, and whenever fewer than two
repositories produced a readable change, the gate reports that the seam was not reviewed — never
that it was reviewed and found clean. Those are different claims. It performs no security and no
deployment analysis in either mode; `merge_order` is copied from the approved execution plan and
is not verified. `metadata.assessment_coverage` records which dimensions were actually checked,
and the feature completion summary and every PR body carry whatever was not.

An approved integration review is required before any PR is created. If a finding names a
responsible repository, only that repository is sent back through its Engineer and Reviewer loop
and the parent retains other child results. A contract-change request is raised by a *child*
workstream that finds the contract itself wrong, not by this gate. The configured
integration-review limit prevents unbounded loops; exceeding it publishes the pull requests for
repositories that passed their own review and ends at `failed_requires_human`.

On approval, the platform creates one draft-by-default PR per workstream. Each PR records the
parent feature ID, child workflow ID, contract version, validation summary, merge/deployment order,
feature flag advice, rollback advice, and known limitations. After all PRs exist, the platform adds
a comment with the complete related PR set. A partial PR failure is retained in durable records and
requires human cleanup. The platform never auto-merges a PR.
