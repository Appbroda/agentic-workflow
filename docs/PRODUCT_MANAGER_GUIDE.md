# Product Manager Guide

This is the guide for describing a feature and reading the result. It assumes no knowledge of
the workflow internals. If you are wiring up the platform itself, read the
[user guide](USER_GUIDE.md) instead.

## What this does, in one paragraph

You describe a feature once. The platform works out what each repository has to build, agrees a
shared contract between them so the pieces fit, then implements, tests and reviews each
repository on its own. What passes review is pushed to a branch and opened as a **draft pull
request** for an engineer to read. Nothing is ever merged for you, and nothing reaches your
default branch.

## Getting in

Open `/console` on the platform URL your engineering team gives you — for a local stack that is
<http://localhost:8000/console>.

You need a personal **platform token** from an administrator (or the documented shared admin key
in a local/trusted deployment). For a live run you also need an **OpenAI API key** and a
**GitHub token**. You may provide provider keys for one action, where they remain in page memory,
or store them for your identity in Settings. Stored provider keys are encrypted server-side and
the browser can only see whether each is configured.

## Writing a feature that works

The form asks for the things the planner actually uses. The two that most affect the result:

- **Acceptance criteria.** Every user story and requirement needs at least one way to tell it
  works. These become the checks the reviewer holds the code to. Vague criteria produce vague
  work. "The dashboard shows a status tile that turns red when the server is unreachable" is
  useful; "the dashboard is better" is not.

  Write criteria a reviewer can check by reading the change and running the repository's own
  tests. A criterion that can only be settled by measuring something running — a p95 latency, an
  uptime figure, a load or penetration test, an observation in staging — cannot be met by any
  amount of code, so the platform will stop and ask you to restate it before it plans anything.
  Say what you actually want instead: that the endpoint does only cheap read-only work, that the
  request never blocks.
- **Repositories.** Add every repository the feature touches, and mark a repository **required**
  if the feature is not done without it. A feature must have at least one required repository —
  that is what completion is measured against.

Start with a **rehearsal** run. It plans the work and shows you how the feature is broken up
without writing code or touching GitHub. When the plan reads correctly, run it live.

## Reading the result

The status view refreshes itself every five seconds until the feature settles. It tells you
three things: what state the feature is in, what that means, and what to do about it.

| What you see | What it means |
| --- | --- |
| **Building** / **Checking the pieces fit** | Working. Nothing for you to do. |
| **Waiting on you** | Planning has a question it cannot answer. Answer it and resume. |
| **Finished** | Every required repository passed review and has a draft pull request. |
| **Partly done — needs an engineer** | Some repositories finished, some did not. See below. |
| **Stopped** | Nothing could be completed. Send the per-repository reasons to an engineer. |
| **Cancelled — leftovers to check** | Stopped after something had already reached GitHub. |

### "Partly done" is the one worth understanding

This is the most common outcome and the easiest to misread. It does **not** mean nothing
happened. Every repository is reviewed on its own, so a repository that passed its own review
gets its pull request opened even when another repository failed. Those pull requests are real
and worth reviewing.

The per-repository list below the status shows, for each one, whether it finished and — if it
did not — **what stopped it**, in the words of the check that stopped it. That list is what an
engineer needs; you can copy it straight into a ticket.

## What you can do to a running feature

- **Resume** — after answering questions, or to continue a feature that was interrupted.
  Resuming never redoes work that already succeeded and never opens a second pull request for
  a branch that already has one.
- **Cancel** — stops scheduling new work. Anything already pushed to GitHub stays there and is
  listed for cleanup; cancelling is not an undo.

## Things worth knowing before you rely on it

- **Draft pull requests only.** Nothing merges automatically and nothing deploys. A person
  reviews and merges, exactly as with any other pull request.
- **Point it at repositories your team owns.** Code from the target repository runs during a
  build, so this is currently for your own trusted repositories, not arbitrary third-party
  ones. See the [production readiness audit](PRODUCTION_READINESS.md) for the boundary.
- **It does not always succeed.** A repository can fail its own tests or review and stop. That
  is the system working as designed — it refuses to open a pull request for work that did not
  pass review, rather than handing an engineer something broken.
- **Rehearsal runs cost nothing and touch nothing.** Use them freely.
