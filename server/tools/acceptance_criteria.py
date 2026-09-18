"""Acceptance criteria a repository change cannot demonstrate, and who gets told about them.

A criterion naming a measurement against a running deployment can never be met by any attempt
a workstream makes: the reviewer reads a diff and the output of the repository's own commands,
and neither contains a latency percentile. Feature -047 spent five of its attempts being
rejected for one.

The reviewer therefore withholds such a criterion from its own judgement -- but withholding
alone is a decision made on the author's behalf and never reported, so a requirement someone
wrote was quietly dropped and the run either failed for a reason nobody could see or passed
without the criterion being checked. Detection lives here so the clarification gate can ask
about it before anything is planned, and the reviewer can say what it set aside, from one list.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Ways of proving a requirement that a change to a repository cannot carry out. These name
# measurement performed against a deployed system, not behaviour a reviewer can read in the
# diff. Deliberately about verification method only: a feature that genuinely builds monitoring
# or an alerting dashboard is ordinary scoped work and must keep its criteria.
UNVERIFIABLE_ACCEPTANCE_PHRASES = (
    "95th percentile",
    "99th percentile",
    "benchmark",
    "canary",
    "load test",
    "load-test",
    "p95",
    "p99",
    "penetration test",
    "production-like",
    "profiling",
    "soak test",
    "staging environment",
    "stress test",
    "uptime",
    # The purest form of the same thing, and the one the list originally missed. Live feature
    # -083 spent ten frontend attempts against a scoped criterion that "requires manual
    # verification of authenticated API loading, healthy and unhealthy rendering ...", which
    # no attempt can satisfy and no retry can approach. A deployed measurement is unreachable
    # because the workflow has no deployment; this is unreachable because it explicitly asks
    # for a person.
    "manual verification",
    "manually verif",
    "manual test",
    "manually test",
    "visual inspection",
    "visually inspect",
    "verified by a human",
    "verify by hand",
)


@dataclass(frozen=True, slots=True)
class UnverifiableCriterion:
    """One acceptance criterion no repository change can demonstrate, and where it came from."""

    requirement_id: str
    criterion: str


def requires_deployed_measurement(criterion: object) -> bool:
    """Return whether a criterion can only be proved by measuring a running deployment."""
    if not isinstance(criterion, str):
        return False
    lowered = criterion.lower()
    return any(phrase in lowered for phrase in UNVERIFIABLE_ACCEPTANCE_PHRASES)


def unverifiable_criteria(requirements: Iterable[object]) -> list[UnverifiableCriterion]:
    """List every acceptance criterion across these requirements that no attempt can satisfy."""
    found: list[UnverifiableCriterion] = []
    for requirement in requirements:
        requirement_id = getattr(requirement, "requirement_id", None)
        criteria = getattr(requirement, "acceptance_criteria", None)
        if not isinstance(requirement_id, str) or not isinstance(criteria, Sequence):
            continue
        found.extend(
            UnverifiableCriterion(requirement_id=requirement_id, criterion=item)
            for item in criteria
            if requires_deployed_measurement(item)
        )
    return found


__all__ = [
    "UNVERIFIABLE_ACCEPTANCE_PHRASES",
    "UnverifiableCriterion",
    "requires_deployed_measurement",
    "unverifiable_criteria",
]
