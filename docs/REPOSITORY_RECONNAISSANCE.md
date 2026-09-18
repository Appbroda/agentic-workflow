# Repository Reconnaissance

Every target repository is cloned and read before the feature is planned. The plan freezes
requirement scope, acceptance criteria, `expected_source_areas`, `implementation_expectations`,
the integration contract, and merge order; until reconnaissance existed, all of that was decided
from a PRD and a repository URL, and the first component to see any code was the Engineer — by
which point nothing upstream could be corrected.

## Why

A requirement resting on a convention the repository does not have becomes a workstream no
attempt can finish. The reviewer enforces it faithfully, the retry loop reclassifies the failure
as `implementation_missing` or `review_scope_failure`, and every remaining attempt is spent being
told to try harder at something impossible. Four premises of exactly this kind — a status source
behind what is a one-line literal, per-route authentication in a repository that authenticates
globally, a console section that does not exist, and a shared time formatter that was never
shared — dominated the failures in live runs 066–075.

## What it produces

One `015_repository_reconnaissance.<repository_id>.json` per repository:

- `source_areas` / `test_areas` — directories that exist, for the plan to bind to.
- `conventions` — how the repository already does a kind of work, each with `evidence_paths` and
  a `wiring_path`: the file a new member must be registered in to be reachable at all.
- `shared_utilities` — helpers that call sites genuinely share. A module most call sites bypass
  is not one; where the real convention is to call a library directly, that is a convention.
- `contradicted_premises` — requirement premises the checkout does not support, each with the
  question a human should be asked instead of the assumption.

## Contradicted premises become the clarification round

Reconnaissance runs before the clarification gate, and every `contradicted_premise` is added to
the technical PRD's `unresolved_questions` as `recon-<repository_id>-<n>`. The feature then pauses
once, asking both the product manager's open questions and the repository's, and the answers are
folded into the PRD the planner receives.

Previously clarification ran before anything had read a repository, so the answers were themselves
assumptions — "use the existing X" where no X existed — and became requirements the reviewer
enforced faithfully and no attempt could satisfy. The question now carries the evidence: which
repository, what the requirements assumed, what the checkout has instead, and the paths that show
it, so the answer can be checked rather than guessed.

The question list is deliberately uncapped. Each entry is a premise that would otherwise be
written into a workstream and spend its whole retry budget failing, so dropping any to keep the
round short trades a short round for a dead repository. A feature raising a great many of these
is reporting something true: it is not the feature these repositories can accept.

## How it is kept honest

The evidence is gathered by `tools/repository_reconnaissance.py`, which only reads files and
never runs repository code. Paths in the model's response are checked against that scan, and a
response naming a path the scan did not read is rejected, repaired once, then refused. Without
that check the stage could restate the same guesses it exists to replace, in an artifact the
planner is now required to trust — worse than no reconnaissance, because the guess would arrive
carrying evidence's authority.

Registries are found structurally, by how many of the repository's own modules a file pulls in,
not by file name. The pilot backend's registry is `server/config/express.js`, which no list of
conventional names would nominate, while plenty of repositories have an `index.js` that assembles
nothing.

Credential-shaped paths never enter the scan; the filter is shared with the Engineer's context
builder in `tools/file_tools.py` so the two cannot drift apart.

## Failure and reuse

A repository that cannot be cloned or read produces no artifact and is logged. The feature is not
failed: that repository is planned the way every repository was planned before this stage
existed. Reconnaissance is performed once per feature — a clarification answer re-enters planning,
and the checkouts have not moved, so the existing artifacts are reused rather than re-read.

The clone is disposable and is removed once read. It is deliberately not the child's workspace:
that clone is journal-reused under its own operation scope and carries the recovery path's
assumptions about what exists when, which is not worth coupling to for one avoided fetch.
