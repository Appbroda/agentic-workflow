"""Realistic on-disk repository fixtures for technology-aware validation tests.

These build actual git checkouts with real manifests, scripts, source files and tests, so
technology detection, validation planning and preflight run against genuine evidence
rather than a hand-written profile. Mock-only fixtures are what allowed a JavaScript
repository to be validated with Python tooling in production.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

__all__ = [
    "broken_eslint_repository",
    "commit_all",
    "init_git_repository",
    "mixed_monorepo_repository",
    "node_javascript_repository",
    "python_fastapi_repository",
    "react_vite_typescript_repository",
    "start_working_branch",
]


def init_git_repository(root: Path) -> Path:
    """Create a real git checkout with a deterministic, non-signing local identity."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "config", "user.name", "Fixture Author")
    _git(root, "config", "commit.gpgsign", "false")
    return root


def commit_all(root: Path, message: str = "Fixture baseline") -> str:
    """Commit the whole working tree and return the resulting commit SHA."""
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD").strip()


def start_working_branch(root: Path, name: str) -> str:
    """Branch off the current revision the way a workspace does, and return that revision.

    The returned SHA is what the attempts on this branch are measured against: a workflow
    branches from the default branch and every attempt commits on top, so the branch point is
    the baseline for the whole lineage rather than whatever `HEAD` has reached.
    """
    revision = _git(root, "rev-parse", "HEAD").strip()
    _git(root, "checkout", "-b", name)
    return revision


def node_javascript_repository(root: Path, *, with_tests: bool = True) -> Path:
    """Build an Express-style JavaScript service mirroring the live backend repository."""
    init_git_repository(root)
    scripts = {"lint": "eslint .", "start": "node src/index.js"}
    if with_tests:
        scripts["test"] = "node --test"
    _write_json(
        root / "package.json",
        {
            "name": "node-javascript-service",
            "version": "1.0.0",
            "private": True,
            "scripts": scripts,
            "devDependencies": {"eslint": "8.57.0"},
        },
    )
    _write_json(
        root / "package-lock.json", {"name": "node-javascript-service", "lockfileVersion": 3}
    )
    _write(root / "eslint.config.js", "module.exports = [];\n")
    _write(
        root / "src" / "index.js",
        "const routes = require('./routes/status');\n\nmodule.exports = { routes };\n",
    )
    _write(
        root / "src" / "routes" / "status.js",
        "function statusRoute(req, res) {\n  res.json({ status: 'ok' });\n}\n\n"
        "module.exports = { statusRoute };\n",
    )
    if with_tests:
        _write(
            root / "test" / "status.test.js",
            "const { test } = require('node:test');\n"
            "const assert = require('node:assert');\n"
            "const { statusRoute } = require('../src/routes/status');\n\n"
            "test('status route responds', () => {\n"
            "  let body = null;\n"
            "  statusRoute({}, { json: (value) => { body = value; } });\n"
            "  assert.deepStrictEqual(body, { status: 'ok' });\n"
            "});\n",
        )
    commit_all(root)
    return root


def react_vite_typescript_repository(root: Path) -> Path:
    """Build a Vite + TypeScript React application with the usual configured scripts."""
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "react-vite-admin",
            "version": "1.0.0",
            "private": True,
            "type": "module",
            "scripts": {
                "lint": "eslint .",
                "typecheck": "tsc --noEmit",
                "test": "vitest run",
                "build": "vite build",
            },
            "dependencies": {"react": "18.3.1", "react-dom": "18.3.1"},
            "devDependencies": {"typescript": "5.5.4", "vite": "5.4.0", "vitest": "2.0.5"},
        },
    )
    _write_json(root / "package-lock.json", {"name": "react-vite-admin", "lockfileVersion": 3})
    _write_json(
        root / "tsconfig.json",
        {"compilerOptions": {"strict": True, "jsx": "react-jsx", "noEmit": True}},
    )
    _write(root / "vite.config.ts", "export default { plugins: [] };\n")
    _write(
        root / "src" / "components" / "ServerStatusTile.tsx",
        "export function ServerStatusTile({ status }: { status: string }) {\n"
        "  return <span>{status}</span>;\n"
        "}\n",
    )
    _write(
        root / "src" / "api" / "client.ts",
        "export async function fetchServerStatus(): Promise<{ status: string }> {\n"
        "  const response = await fetch('/api/status');\n"
        "  return response.json();\n"
        "}\n",
    )
    commit_all(root)
    return root


def python_fastapi_repository(root: Path) -> Path:
    """Build a Python service configured for ruff and pytest through pyproject."""
    init_git_repository(root)
    _write(
        root / "pyproject.toml",
        '[project]\nname = "python-fastapi-service"\nversion = "1.0.0"\n'
        'requires-python = ">=3.12"\ndependencies = ["fastapi"]\n\n'
        "[tool.ruff]\nline-length = 100\n\n"
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
    )
    _write(
        root / "app" / "__init__.py",
        '"""Python service package."""\n',
    )
    _write(
        root / "app" / "status.py",
        '"""Server status endpoint."""\n\n\n'
        "def server_status() -> dict[str, str]:\n"
        '    """Return the current service status."""\n'
        '    return {"status": "ok"}\n',
    )
    _write(
        root / "tests" / "test_status.py",
        '"""Status endpoint tests."""\n\n'
        "from app.status import server_status\n\n\n"
        "def test_server_status_reports_ok() -> None:\n"
        '    """The status endpoint reports a healthy service."""\n'
        '    assert server_status() == {"status": "ok"}\n',
    )
    commit_all(root)
    return root


def broken_eslint_repository(root: Path) -> Path:
    """Build the exact production defect: a shared ESLint config that is never declared."""
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "broken-eslint-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "eslint ."},
            # airbnb-base is extended but deliberately absent from devDependencies.
            "eslintConfig": {"extends": "airbnb-base"},
            "devDependencies": {"eslint": "8.57.0"},
        },
    )
    _write_json(root / "package-lock.json", {"name": "broken-eslint-service", "lockfileVersion": 3})
    _write(root / "src" / "index.js", "module.exports = {};\n")
    commit_all(root)
    return root


def mixed_monorepo_repository(root: Path) -> Path:
    """Build a repository holding independent Node and Python projects in subdirectories."""
    init_git_repository(root)
    _write_json(
        root / "web" / "package.json",
        {
            "name": "monorepo-web",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "eslint .", "test": "vitest run"},
        },
    )
    _write_json(root / "web" / "package-lock.json", {"name": "monorepo-web", "lockfileVersion": 3})
    _write(root / "web" / "src" / "main.js", "export const main = () => 'web';\n")
    _write(
        root / "service" / "pyproject.toml",
        '[project]\nname = "monorepo-service"\nversion = "1.0.0"\n'
        'requires-python = ">=3.12"\n\n[tool.ruff]\nline-length = 100\n',
    )
    _write(
        root / "service" / "app.py",
        '"""Monorepo service entry point."""\n\n\n'
        "def handler() -> str:\n"
        '    """Return the service name."""\n'
        '    return "service"\n',
    )
    commit_all(root)
    return root


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, f"{json.dumps(payload, indent=2)}\n")


def _git(root: Path, *arguments: str) -> str:
    """Run one git command with no shell and an explicit working directory."""
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return completed.stdout
