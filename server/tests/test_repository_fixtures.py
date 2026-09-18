"""Technology-aware validation planning verified against real repository checkouts.

Every assertion here runs against an actual git working tree rather than a synthesized
profile, because the production defect was precisely that a JavaScript repository was
validated with Python tooling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures import (
    broken_eslint_repository,
    mixed_monorepo_repository,
    node_javascript_repository,
    python_fastapi_repository,
    react_vite_typescript_repository,
)
from tools.repository_preflight import RepositoryHealthDisposition, classify_lint_failure
from tools.technology_detection import (
    calculate_repository_revision_sync,
    inspect_repository_technology,
)
from tools.validation_tools import DefaultValidationPlanBuilder

_PYTHON_ONLY_TOOLS = ("ruff", "pytest", "mypy")


def test_a_javascript_service_is_never_validated_with_python_tooling(tmp_path: Path) -> None:
    """Scenario A: the exact cross-stack mistake that failed the live backend."""
    root = node_javascript_repository(tmp_path / "node-service")

    profile = inspect_repository_technology(root, repository_id="backend")
    plan = DefaultValidationPlanBuilder().build_plan_sync(root)
    executables = {command.command[0] for command in plan.commands}

    assert "JavaScript" in profile.languages
    assert plan.source == "repository_scripts"
    assert executables == {"npm"}
    assert not executables & set(_PYTHON_ONLY_TOOLS)
    assert {command.validation_type for command in plan.commands} >= {"lint", "test"}


def test_a_react_typescript_application_runs_its_configured_scripts(tmp_path: Path) -> None:
    """A Vite/TypeScript frontend is planned from its own package scripts."""
    root = react_vite_typescript_repository(tmp_path / "react-admin")

    profile = inspect_repository_technology(root, repository_id="frontend")
    plan = DefaultValidationPlanBuilder().build_plan_sync(root)
    scripts = {
        command.command[2] for command in plan.commands if command.command[:2] == ["npm", "run"]
    }

    assert "TypeScript" in profile.languages
    assert {"lint", "typecheck", "build"} <= scripts
    assert all(command.command[0] == "npm" for command in plan.commands)


def test_a_python_service_uses_its_configured_python_tooling(tmp_path: Path) -> None:
    """Python repositories must keep working exactly as before."""
    root = python_fastapi_repository(tmp_path / "python-service")

    profile = inspect_repository_technology(root, repository_id="service")
    plan = DefaultValidationPlanBuilder().build_plan_sync(root)
    rendered = {" ".join(command.command) for command in plan.commands}

    assert profile.primary_language == "Python"
    assert any("ruff" in command for command in rendered)
    assert any("pytest" in command for command in rendered)
    assert not any(command.startswith("npm") for command in rendered)


def test_a_mixed_monorepo_scopes_each_toolchain_to_its_own_directory(tmp_path: Path) -> None:
    """Neither project in a mixed repository may be validated with the other's tools."""
    root = mixed_monorepo_repository(tmp_path / "monorepo")

    plan = DefaultValidationPlanBuilder().build_plan_sync(root)
    node_directories = {
        command.working_directory for command in plan.commands if command.command[0] == "npm"
    }
    python_directories = {
        command.working_directory
        for command in plan.commands
        if command.command[0] in {"ruff", "pytest", "mypy", "uv"}
    }

    assert node_directories == {"web"}
    assert python_directories <= {"service"}
    assert python_directories


def test_an_undeclared_shared_lint_config_is_detected_from_a_real_checkout(tmp_path: Path) -> None:
    """The broken-eslint fixture reproduces feature -007's repository defect on disk."""
    root = broken_eslint_repository(tmp_path / "broken-eslint")

    classification = classify_lint_failure(
        result_stdout="",
        result_stderr=(
            'Error: Failed to load config "airbnb-base" to extend from.\n'
            "Cannot find module 'eslint-config-airbnb-base'"
        ),
        repository_root=root,
    )

    assert classification.kind == "missing_dependency"
    assert classification.disposition is RepositoryHealthDisposition.REQUIRES_HUMAN
    assert "airbnb-base" in classification.undeclared_references


@pytest.mark.parametrize(
    "build",
    [node_javascript_repository, react_vite_typescript_repository, python_fastapi_repository],
)
def test_editing_a_repository_changes_its_revision_fingerprint(
    tmp_path: Path, build: object
) -> None:
    """Scenario B: an engineer edit must invalidate any earlier validation evidence."""
    root = build(tmp_path / "repository")  # type: ignore[operator]

    before = calculate_repository_revision_sync(root)
    (root / "NEW_SOURCE_FILE.txt").write_text("changed\n", encoding="utf-8")
    after = calculate_repository_revision_sync(root)

    assert before.head_sha == after.head_sha
    assert before.combined_fingerprint != after.combined_fingerprint
