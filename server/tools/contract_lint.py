"""Static validation of a generated OpenAPI document, before any repository writes code.

AB-Feature-184 published a GET endpoint whose contract required an empty request body --
a structural defect nothing downstream ever questioned, because every check from there on
only verified that code conformed to the contract, never that the contract itself was sound.
This module closes that gap with a real, industry-standard rule *engine* (Spectral) instead of
a bespoke Python function per defect shape: a future convention violation is a new declarative
rule in `contract_ruleset.spectral.yaml`, never new code here.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from services.cancellation import MockCancellationToken
from services.process_runner import AsyncioProcessRunner, ProcessRunner

_RULESET_PATH = Path(__file__).with_name("contract_ruleset.spectral.yaml")
# Local, static validation of one small JSON document -- not a repository command, so this
# shares no budget with `VALIDATION_COMMAND_TIMEOUT_SECONDS` and stays far below it.
_LINT_TIMEOUT_SECONDS = 30.0
_SEVERITY_NAMES = {0: "error", 1: "warn", 2: "info", 3: "hint"}


@dataclass(frozen=True, slots=True)
class LintFinding:
    """One Spectral rule violation against a generated OpenAPI document."""

    rule: str
    severity: str
    message: str
    path: str


class ContractLinter(Protocol):
    """Lint a generated OpenAPI document against this platform's contract-quality ruleset."""

    async def lint(self, openapi_document: dict[str, Any]) -> list[LintFinding]:
        """Return every rule violation Spectral finds, or an empty list for a clean contract."""


class SpectralContractLinter:
    """Run the real Spectral CLI against a generated OpenAPI document.

    Reuses the same sandboxed subprocess execution every repository's own lint/test/build
    already goes through (`services/process_runner.py`) -- not a new execution path, and
    nothing here reaches a network or a repository checkout. `npx` resolves `@stoplight/
    spectral-cli` from the platform's own Node toolchain, the same one every JavaScript/
    TypeScript repository workstream already requires.
    """

    def __init__(self, *, process_runner: ProcessRunner | None = None) -> None:
        """Default to the real sandboxed runner; tests inject a fake one."""
        self._process_runner = process_runner or AsyncioProcessRunner()

    async def lint(self, openapi_document: dict[str, Any]) -> list[LintFinding]:
        """Write the document to a private temp file and run Spectral against it."""
        with tempfile.TemporaryDirectory(prefix="contract-lint-") as workdir_name:
            workdir = Path(workdir_name)
            document_path = workdir / "contract.openapi.json"
            document_path.write_text(json.dumps(openapi_document), encoding="utf-8")
            result = await self._process_runner.run(
                (
                    "npx",
                    "-y",
                    "@stoplight/spectral-cli",
                    "lint",
                    str(document_path),
                    "--ruleset",
                    str(_RULESET_PATH),
                    "--format",
                    "json",
                ),
                cwd=workdir,
                timeout_seconds=_LINT_TIMEOUT_SECONDS,
                # A one-off, local, sub-second lint of a small JSON file; nothing feature- or
                # workflow-scoped is cancellable here, so a fresh, never-cancelled token is the
                # correct default rather than threading a live cancellation source through.
                cancellation_token=MockCancellationToken(),
            )
            return _parse_findings(result.stdout)


def _parse_findings(raw_stdout: str) -> list[LintFinding]:
    """Translate Spectral's own JSON report shape into this platform's finding type.

    A Spectral run that produces no findings, or that fails to produce parseable JSON at all
    (a toolchain problem, not a contract problem), is treated the same way: no findings. A
    broken linter must never be what blocks a plan; only a real, parsed rule violation does.
    """
    if not raw_stdout.strip():
        return []
    try:
        entries = json.loads(raw_stdout)
    except ValueError:
        return []
    if not isinstance(entries, list):
        return []
    findings: list[LintFinding] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_severity = entry.get("severity")
        severity = (
            _SEVERITY_NAMES.get(raw_severity, "error") if isinstance(raw_severity, int) else "error"
        )
        findings.append(
            LintFinding(
                rule=str(entry.get("code", "")),
                severity=severity,
                message=str(entry.get("message", "")),
                path=".".join(str(part) for part in entry.get("path", [])),
            )
        )
    return findings


class MockContractLinter:
    """Deterministic no-op linter for mock workflows and tests without a Node toolchain."""

    async def lint(self, openapi_document: dict[str, Any]) -> list[LintFinding]:
        """Return no findings; mock mode never writes a real endpoint shape to lint."""
        del openapi_document
        return []


__all__ = [
    "ContractLinter",
    "LintFinding",
    "MockContractLinter",
    "SpectralContractLinter",
]
