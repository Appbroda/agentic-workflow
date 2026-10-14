"""`lint_contract` pins Spectral-shaped findings becoming blocking contract defects.

AB-Feature-182's backend reproduced AB-Feature-184's GET-with-body defect: the generated
contract required an empty request body on a GET endpoint, and nothing downstream ever
questioned the contract's own quality. These tests pin `SpectralContractLinter`'s JSON-report
parsing (via a fake process runner, so no real Node/npx toolchain is required to run them) and
`MockContractLinter`'s no-op behavior for mock workflows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from services.cancellation import CancellationToken
from services.process_runner import ProcessResult
from tools.contract_lint import LintFinding, MockContractLinter, SpectralContractLinter

_SAMPLE_OPENAPI_DOCUMENT = {
    "openapi": "3.1.0",
    "info": {"title": "Feature test", "version": "1.0.0"},
    "paths": {
        "/service-requests/ad-unit-export/statuses": {
            "get": {
                "operationId": "getSupportedAdUnitStatuses",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"maxProperties": 0}}},
                },
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}


class _FakeProcessRunner:
    """Returns a canned `ProcessResult` instead of ever invoking a real subprocess."""

    def __init__(self, stdout: str, *, return_code: int = 0) -> None:
        self._stdout = stdout
        self._return_code = return_code
        self.calls: list[tuple[str, ...]] = []

    async def run(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
        cancellation_token: CancellationToken,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        del cwd, timeout_seconds, cancellation_token, environment
        self.calls.append(tuple(command))
        return ProcessResult(
            command=tuple(command),
            return_code=self._return_code,
            stdout=self._stdout,
            stderr="",
            duration_seconds=0.01,
        )


def _spectral_entry(
    *, code: str, severity: int, message: str, path: list[str]
) -> dict[str, object]:
    return {"code": code, "severity": severity, "message": message, "path": path}


@pytest.mark.asyncio
async def test_a_get_with_body_finding_is_reported_as_an_error() -> None:
    """AB-Feature-184/182's exact reproduction: a GET endpoint requiring an empty body."""
    stdout = json.dumps(
        [
            _spectral_entry(
                code="luxass/oas3/get-request-no-body",
                severity=0,
                message="GET requests must not have a requestBody.",
                path=["paths", "/service-requests/ad-unit-export/statuses", "get", "requestBody"],
            )
        ]
    )
    runner = _FakeProcessRunner(stdout)
    linter = SpectralContractLinter(process_runner=runner)

    findings = await linter.lint(_SAMPLE_OPENAPI_DOCUMENT)

    assert findings == [
        LintFinding(
            rule="luxass/oas3/get-request-no-body",
            severity="error",
            message="GET requests must not have a requestBody.",
            path="paths./service-requests/ad-unit-export/statuses.get.requestBody",
        )
    ]
    # The ruleset this platform owns is passed explicitly, not left to Spectral's own default.
    assert any("contract_ruleset.spectral.yaml" in part for part in runner.calls[0])


@pytest.mark.asyncio
async def test_a_clean_contract_reports_no_findings() -> None:
    """Spectral's own empty-array report for a contract with nothing wrong."""
    runner = _FakeProcessRunner(json.dumps([]))
    linter = SpectralContractLinter(process_runner=runner)

    findings = await linter.lint(_SAMPLE_OPENAPI_DOCUMENT)

    assert findings == []


@pytest.mark.asyncio
async def test_a_warn_level_finding_is_not_silently_promoted_to_error() -> None:
    """Severity is read from Spectral's own report, not assumed -- a style nit stays advisory."""
    stdout = json.dumps(
        [
            _spectral_entry(
                code="info-contact",
                severity=1,
                message="Info object should contain a contact object.",
                path=["info"],
            )
        ]
    )
    runner = _FakeProcessRunner(stdout)
    linter = SpectralContractLinter(process_runner=runner)

    (finding,) = await linter.lint(_SAMPLE_OPENAPI_DOCUMENT)

    assert finding.severity == "warn"


@pytest.mark.asyncio
async def test_an_unparseable_report_is_treated_as_no_findings() -> None:
    """A broken linter must never be what blocks a plan -- only a parsed violation does."""
    runner = _FakeProcessRunner("not actually json")
    linter = SpectralContractLinter(process_runner=runner)

    findings = await linter.lint(_SAMPLE_OPENAPI_DOCUMENT)

    assert findings == []


@pytest.mark.asyncio
async def test_the_mock_linter_never_reaches_a_process_runner() -> None:
    """Mock workflows and most tests need zero Node/npx toolchain dependency."""
    findings = await MockContractLinter().lint(_SAMPLE_OPENAPI_DOCUMENT)

    assert findings == []
