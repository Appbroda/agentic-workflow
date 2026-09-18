# Scoped repository review

The feature planner assigns each child explicit `ScopedRequirementReference` entries, including a
requirement ID, acceptance-criterion IDs, and one responsibility: `implements`, `consumes`,
`validates`, or `documents`. A workstream also records explicit out-of-scope and shared
requirements plus contract sections it consumes or implements.

The repository reviewer receives only its workstream scope, scoped PRD projection, shared contract,
current code completion report, current-revision validation results, and previous findings from the
same child. The full feature PRD is not a review checklist. A backend reviewer therefore cannot
reject a backend child for a frontend tile, and the inverse is also true.

Findings identify their repository, requirement or contract reference, responsibility, validated
revision, severity, evidence, and recommended fix. Security, code-quality, repository-health, and
validation findings may be repository-scoped exceptions when no product requirement reference is
appropriate.

Cross-repository compatibility belongs to the integration reviewer, which reads the changed
source of every repository together — the only point in the platform that sees more than one at
once. Where it has no diff source, no model, or fewer than two readable changes, it reports that
the seam was not reviewed rather than approving it silently, and the seam is then a human's
responsibility at PR review. See [cross-repository review](CROSS_REPOSITORY_REVIEW.md).

## Criteria no change can demonstrate

An acceptance criterion settled only by measuring a running deployment — a latency percentile, an
uptime figure, a load or penetration test, an observation in staging — cannot be met by any
attempt, because a reviewer judges the diff and the output of the repository's own commands.
Feature -047 spent five attempts rejected for one.

These are detected in `tools/acceptance_criteria.py` and handled twice. The clarification gate
asks the author to restate the criterion before anything is planned. If one reaches review anyway,
the reviewer withholds it from its judgement — in both the scoped and unscoped projections, so the
same criterion cannot be unmeetable in one review and enforced in another — and records an
`ACCEPTANCE_CRITERIA_NOT_REVIEWED-<requirement_id>` finding naming exactly what it set aside. That
finding is `low` and never changes the verdict: blocking on a criterion no attempt can satisfy is
the loop the withholding exists to end. The review artifact also carries
`metadata.acceptance_criteria_not_reviewed`.

Before model review, the deterministic completion gate verifies the workstream's implementation
expectations and requirement-to-file evidence. See
[implementation completeness](IMPLEMENTATION_COMPLETENESS.md). Reviewers report source omissions
such as `BACKEND_ROUTE_NOT_IMPLEMENTED` and `TESTS_ONLY_CHANGE`, and report
`TEST_COMMAND_NOT_CONFIGURED` or `LINT_CONFIGURATION_INVALID` without falsely calling them passing
tests or source lint violations.
