"""Regression coverage for checkout-derived layout handling."""

from __future__ import annotations

from pathlib import Path

from artifacts.schemas import RepositoryWorkstreamPlan
from tools.repository_layout import reconcile_workstream_layout


def test_python_layout_rebinds_a_stale_source_prefix_from_checkout_evidence(tmp_path: Path) -> None:
    """A Python project need not use any assumed source-root directory name."""
    _write(tmp_path, "application/api/status.py", "def status() -> str:\n    return 'ok'\n")
    _write(tmp_path, "checks/test_status.py", "def test_status():\n    assert True\n")

    workstream, evidence = reconcile_workstream_layout(_workstream(["src/api"]), tmp_path)

    expectation = workstream.implementation_expectations[0]
    assert expectation.expected_source_areas == ["application/api"]
    assert evidence.remapped_areas == (("src/api", "application/api"),)
    assert evidence.source_directories == ("application/api",)
    assert evidence.test_directories == ("checks",)


def test_node_layout_rebinds_a_root_relative_area_without_a_node_rule(tmp_path: Path) -> None:
    """The same reconciliation works for a Node checkout with another layout."""
    _write(tmp_path, "engine/http/status.js", "export const status = () => 'ok';\n")

    workstream, evidence = reconcile_workstream_layout(_workstream(["http"]), tmp_path)

    assert workstream.implementation_expectations[0].expected_source_areas == ["engine/http"]
    assert evidence.remapped_areas == (("http", "engine/http"),)


def test_mixed_layout_does_not_guess_or_enforce_an_ambiguous_plan_area(tmp_path: Path) -> None:
    """Mixed repositories retain ambiguity as evidence without rejecting either valid layout."""
    _write(tmp_path, "api/routes/status.py", "def status() -> str:\n    return 'ok'\n")
    _write(tmp_path, "web/routes/status.js", "export const status = () => 'ok';\n")

    workstream, evidence = reconcile_workstream_layout(_workstream(["routes"]), tmp_path)

    assert workstream.implementation_expectations[0].expected_source_areas == []
    assert evidence.remapped_areas == ()
    assert evidence.ambiguous_areas == ("routes",)


def test_unresolved_plan_area_is_not_enforced_against_checkout_evidence(tmp_path: Path) -> None:
    """A pre-clone source-path guess cannot make a real repository impossible to complete."""
    _write(tmp_path, "service/status.py", "def status() -> str:\n    return 'ok'\n")

    workstream, evidence = reconcile_workstream_layout(_workstream(["guessed/api"]), tmp_path)

    assert workstream.implementation_expectations[0].expected_source_areas == []
    assert evidence.unresolved_areas == ("guessed/api",)


def _workstream(areas: list[str]) -> RepositoryWorkstreamPlan:
    return RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": "repository",
            "repository_id": "repository",
            "role": "other",
            "requirement_ids": ["status"],
            "scoped_requirements": [
                {
                    "requirement_id": "status",
                    "acceptance_criterion_ids": ["status:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Implement the status behavior."],
            "task_ids": ["status-task"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["Status works."],
            "test_requirements": ["Run native tests."],
            "documentation_requirements": [],
            "expected_files_or_areas": [],
            "implementation_expectations": [
                {
                    "requirement_id": "status",
                    "expected_change_categories": ["production", "test"],
                    "expected_source_areas": areas,
                    "tests_required": True,
                }
            ],
            "required": True,
        }
    )


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
