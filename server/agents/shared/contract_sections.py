"""The approved contract's own text, selected for the judges that rule on compliance.

Three roles are asked whether a change complies with the integration contract -- the Engineer,
its self-review pass, and the independent Reviewer -- and until this module existed all three
were asked while holding contract section *names* and nothing else. Feature -197's frontend
attempt-2 is what that costs: the Reviewer blocked, at high severity, because the client gated
import success on a `code` field and a non-zero `created`, calling both "undocumented"; the
approved contract's `BulkImportSuccess` makes `code` a `const`, `created` an integer with
`minimum: 1`, and both `required`. The attempt was the contract restated in code. The
remediation removed the gate and shipped a client that accepts a response its own contract
rules out.

Nothing about the contract was unavailable -- it is a persisted artifact carried in every
child's state, and the names to select by are already in the review scope. It was simply never
quoted. So this module quotes it, once, in one selection every judge is given, because three
judges sharing one blindness must not become three judges with three slightly different
sights.

Two properties make the quoting safe to put in front of a model:

**Bounded, and honest when the bound bites.** A section whose text does not fit is left out
whole and named in `sections_omitted` with the reason and its size. It is never trimmed: a
judge shown two thirds of a schema reads a `required` list that is missing entries and finds
exactly the violation -197 found, with extra steps. A judge shown nothing knows it was shown
nothing.

**Absent is a fact, not a gap.** A review scope naming a section this contract version does not
define is itself something the judges should see -- the plan is wrong, or the contract moved --
so those names are listed in `sections_absent` rather than quietly dropped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from artifacts.schemas import IntegrationContractArtifact, TaskPlanArtifact
from state.models import AgentState

# One section's serialized definition may spend this much; the whole selection this much.
# Calibrated against -197's own frontend workstream, the largest real selection on record: 16
# sections spanning two endpoints, five shared schemas and nine error contracts, 16,455
# characters in total with its biggest single section (`bulkImportApps`) at 3,270. Both bounds
# are therefore headroom rather than a trim on today's contracts -- which is the point, because
# a bound that bites routinely is a bound that routinely hides the contract.
_SECTION_MAX_CHARACTERS = 6_000
_SECTIONS_MAX_TOTAL_CHARACTERS = 40_000

# The five kinds a `contract_sections_*` entry may name, each paired with the artifact field
# that holds them and the attribute that names one. Shared with the planner's own validation of
# those lists, so "does this name exist in the contract" has exactly one answer in this
# codebase: a name the planner accepts is a name this selection can quote, and a name it
# rejects is one that would be reported absent here.
_SECTION_KINDS: tuple[tuple[str, str, str], ...] = (
    ("endpoint", "endpoints", "operation_id"),
    ("shared_schema", "shared_schemas", "name"),
    ("event_contract", "event_contracts", "event_name"),
    ("error_contract", "error_contracts", "code"),
    ("environment_variable", "environment_variables", "name"),
)

_RELATIONSHIP_IMPLEMENTS = "implements"
_RELATIONSHIP_CONSUMES = "consumes"
_RELATIONSHIP_BOTH = "implements_and_consumes"


def contract_section_index(
    contract: IntegrationContractArtifact,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Map every section name this contract defines to its kind and its own definition.

    Insertion order follows ``_SECTION_KINDS`` and, within a kind, the contract's own order, so
    two callers reading this index agree on more than membership. Duplicate names cannot occur:
    the artifact's own validator rejects repeated operation, schema, rule and error identifiers.
    """
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    for kind, field, name_attribute in _SECTION_KINDS:
        for entry in getattr(contract, field):
            index[getattr(entry, name_attribute)] = (
                kind,
                entry.model_dump(mode="json"),
            )
    return index


def contract_section_names(contract: IntegrationContractArtifact) -> frozenset[str]:
    """Return every section name a workstream may legitimately implement or consume."""
    return frozenset(contract_section_index(contract))


def scoped_section_relationships(task_plan: TaskPlanArtifact) -> dict[str, str]:
    """Return the section names this task plan's review scope claims, and how it claims them.

    Read from the review scope rather than from the workstream plan because the review scope is
    what both judges are bounded by, and reading the bound from anywhere else would let the
    quoted text and the assignment disagree. Implemented sections come first: a workstream that
    owns a section is judged harder on it than one that merely calls it.
    """
    scope = task_plan.metadata.get("review_scope")
    if not isinstance(scope, Mapping):
        return {}
    relationships: dict[str, str] = {}
    for field, relationship in (
        ("contract_sections_implemented", _RELATIONSHIP_IMPLEMENTS),
        ("contract_sections_consumed", _RELATIONSHIP_CONSUMES),
    ):
        for name in _clean_names(scope.get(field)):
            existing = relationships.get(name)
            relationships[name] = (
                _RELATIONSHIP_BOTH if existing and existing != relationship else relationship
            )
    return relationships


def contract_section_context(
    *,
    contract: IntegrationContractArtifact | None,
    task_plan: TaskPlanArtifact,
    section_max_characters: int = _SECTION_MAX_CHARACTERS,
    total_max_characters: int = _SECTIONS_MAX_TOTAL_CHARACTERS,
) -> dict[str, Any] | None:
    """Select the contract text for one workstream's own sections, or ``None`` if there is none.

    ``None`` rather than an empty block for the two cases that are not a bounded selection at
    all: a workflow with no integration contract in its state, which is every single-repository
    run, and a workstream whose review scope claims no sections. Callers render nothing for it,
    so those runs' prompts are unchanged byte for byte.

    The bound is spent in scope order, and a section that does not fit costs only itself: a
    later, smaller section is still quoted when it fits in what remains. Dropping everything
    after the first oversized section would hide small sections for no reason other than the
    order the plan happened to list them in.
    """
    if contract is None:
        return None
    relationships = scoped_section_relationships(task_plan)
    if not relationships:
        return None
    index = contract_section_index(contract)
    sections: list[dict[str, Any]] = []
    absent: list[dict[str, str]] = []
    omitted: list[dict[str, Any]] = []
    spent = 0
    for name, relationship in relationships.items():
        resolved = index.get(name)
        if resolved is None:
            absent.append({"name": name, "relationship": relationship})
            continue
        kind, definition = resolved
        entry = {
            "name": name,
            "kind": kind,
            "relationship": relationship,
            "definition": definition,
        }
        characters = len(json.dumps(entry, indent=2, sort_keys=True))
        reason = _omission_reason(
            characters,
            spent=spent,
            section_max_characters=section_max_characters,
            total_max_characters=total_max_characters,
        )
        if reason is not None:
            omitted.append(
                {
                    "name": name,
                    "kind": kind,
                    "relationship": relationship,
                    "characters": characters,
                    "reason": reason,
                }
            )
            continue
        spent += characters
        sections.append(entry)
    return {
        "contract_version": contract.contract_version,
        "contract_status": contract.status,
        "sections": sections,
        "sections_absent": absent,
        "sections_omitted": omitted,
        "character_bounds": {
            "per_section": section_max_characters,
            "total": total_max_characters,
        },
        "characters_selected": spent,
    }


def contract_section_context_from_state(
    state: AgentState,
    task_plan: TaskPlanArtifact,
) -> dict[str, Any] | None:
    """Select from whichever integration contract this workflow's state carries, if any.

    The single entry point both judges call, taking the same two inputs, so the Engineer and the
    Reviewer cannot be given different selections of the same contract for the same attempt. The
    lookup does not raise on a missing contract: the Reviewer and the Engineer both run in
    single-repository workflows that have no contract at all, and those must keep working.
    """
    return contract_section_context(contract=_contract_in_state(state), task_plan=task_plan)


def contract_section_names_from_state(state: AgentState) -> frozenset[str]:
    """Return every section name this workflow's contract defines, or an empty set.

    The reference space a `contract_reference` on a finding is checked against -- the whole
    contract, deliberately, not this workstream's selection. A finding citing a real section
    outside its own scope is asking a scope question and its citation is a fact; a finding
    citing a name the contract does not contain anywhere has invented its reference.
    """
    contract = _contract_in_state(state)
    if contract is None:
        return frozenset()
    return contract_section_names(contract)


def contract_reference_resolves(reference: str | None, names: AbstractSet[str]) -> bool:
    """Say whether a finding's `contract_reference` names a section this contract defines.

    Substring containment rather than equality, because the field is free text and a reviewer
    citing `BulkImportSuccess.created` or `bulkImportApps 201 response` has cited a real
    section perfectly well. Only a reference that names no defined section anywhere in its text
    is unresolved -- the reading that cannot mistake a precise citation for an invented one.

    An empty ``names`` -- no contract in this workflow at all -- resolves everything, because
    there is no reference space to be wrong about.
    """
    if not names:
        return True
    if reference is None or not reference.strip():
        return False
    return any(name in reference for name in names)


def contract_section_context_json(context: dict[str, Any] | None) -> str:
    """Render a selection for a prompt, or the empty string when there is nothing to render."""
    if context is None:
        return ""
    return json.dumps(context, indent=2, sort_keys=True)


def _omission_reason(
    characters: int,
    *,
    spent: int,
    section_max_characters: int,
    total_max_characters: int,
) -> str | None:
    """Say why this section cannot be quoted whole, or ``None`` when it can."""
    if characters > section_max_characters:
        return "over_per_section_character_bound"
    if spent + characters > total_max_characters:
        return "over_total_character_bound"
    return None


def _contract_in_state(state: AgentState) -> IntegrationContractArtifact | None:
    """Return the latest same-workflow integration contract in this state, or nothing."""
    for artifact in reversed(state["artifacts"]):
        if (
            isinstance(artifact, IntegrationContractArtifact)
            and artifact.workflow_id == state["workflow_id"]
        ):
            return artifact
    return None


def _clean_names(value: Any) -> Iterator[str]:
    """Yield the well-formed section names in a persisted list, in order, without repeats."""
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return
    yield from _unique(item.strip() for item in value if isinstance(item, str) and item.strip())


def _unique(values: Iterable[str]) -> Iterator[str]:
    """Yield each value the first time it appears."""
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            yield value


__all__ = [
    "contract_reference_resolves",
    "contract_section_context",
    "contract_section_context_from_state",
    "contract_section_context_json",
    "contract_section_index",
    "contract_section_names",
    "contract_section_names_from_state",
    "scoped_section_relationships",
]
