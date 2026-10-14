"""The reviewer's execution-grounded route evidence: requiring real files, never guessing."""

from __future__ import annotations

from typing import Any

from agents.reviewer.agent import _execution_grounded_routes
from agents.shared.contracts import create_artifact
from artifacts.schemas import RepositoryReconnaissanceArtifact
from tools.route_dump import RouteDumpResult
from tools.validation_tools import ValidationResult


def _reconnaissance(
    *, repository_id: str = "admanager_console-2.0", registered_routes: list[dict[str, str]]
) -> RepositoryReconnaissanceArtifact:
    return create_artifact(
        RepositoryReconnaissanceArtifact,
        workflow_id="workflow-1",
        artifact_id="003_repository_reconnaissance.json",
        producer="recon",
        payload={
            "feature_id": "feature-1",
            "repository_id": repository_id,
            "repository_revision": "deadbeef",
            "summary": "a checkout",
            "source_areas": [],
            "test_areas": [],
            "conventions": [],
            "shared_utilities": [],
            "contradicted_premises": [],
            "registered_routes": registered_routes,
        },
        metadata={},
    )


def _state(artifacts: list[Any]) -> Any:
    return {"artifacts": artifacts}


def _unused_validation_result() -> ValidationResult:
    return ValidationResult(
        command=(), return_code=0, stdout="", stderr="", timed_out=False, duration_seconds=0.0
    )


class _FakeValidationTool:
    """A double recording which files it was asked to dump, and answering each in turn."""

    def __init__(self, answers: dict[str, RouteDumpResult]) -> None:
        self._answers = answers
        self.requested_paths: list[str] = []

    def run_ruff(self, *, timeout_seconds: float) -> ValidationResult:
        return _unused_validation_result()

    def run_pytest(self, *, timeout_seconds: float) -> ValidationResult:
        return _unused_validation_result()

    async def run_route_dump_async(self, relative_path: str) -> RouteDumpResult:
        self.requested_paths.append(relative_path)
        return self._answers[relative_path]


class _LegacyValidationTool:
    """A double with no `run_route_dump_async` at all, the shape an old test still passes."""

    def run_ruff(self, *, timeout_seconds: float) -> ValidationResult:
        return _unused_validation_result()

    def run_pytest(self, *, timeout_seconds: float) -> ValidationResult:
        return _unused_validation_result()


async def test_each_distinct_express_file_is_dumped_once() -> None:
    reconnaissance = _reconnaissance(
        registered_routes=[
            {
                "method": "POST",
                "path": "/service-requests/ad-units/export",
                "source_path": "server/routes/serviceRequest.route.js",
            },
            {
                "method": "POST",
                "path": "/ad-units/export",
                "source_path": "server/routes/serviceRequest.route.js",
            },
            {
                "method": "GET",
                "path": "/auth/login",
                "source_path": "server/routes/auth.route.js",
            },
        ]
    )
    tool = _FakeValidationTool(
        {
            "server/routes/serviceRequest.route.js": RouteDumpResult(
                source_path="server/routes/serviceRequest.route.js",
                routes=(),
                error=None,
            ),
            "server/routes/auth.route.js": RouteDumpResult(
                source_path="server/routes/auth.route.js", routes=(), error="could not require it"
            ),
        }
    )

    result = await _execution_grounded_routes(
        _state([reconnaissance]),
        repository_id="admanager_console-2.0",
        validation_tool=tool,
    )

    # The same file named twice in reconnaissance (bare and mounted) is dumped once, not twice.
    assert sorted(tool.requested_paths) == [
        "server/routes/auth.route.js",
        "server/routes/serviceRequest.route.js",
    ]
    assert {item["source_path"]: item["error"] for item in result} == {
        "server/routes/serviceRequest.route.js": None,
        "server/routes/auth.route.js": "could not require it",
    }


async def test_a_file_no_dump_script_claims_is_still_attempted_and_its_error_surfaces() -> None:
    """Which suffixes are supported is `tools/route_dump.py`'s call, not this function's.

    A `.py` route module has no dump script yet, but this still names it to `dump`, the same
    as any other file, and reports whatever error comes back -- it never re-derives "is this
    framework supported" itself, so a new dump script extends coverage without this function
    changing at all.
    """
    reconnaissance = _reconnaissance(
        registered_routes=[
            {"method": "GET", "path": "/export", "source_path": "server/routes.py"},
        ]
    )
    tool = _FakeValidationTool(
        {
            "server/routes.py": RouteDumpResult(
                source_path="server/routes.py",
                routes=(),
                error="no route dump is implemented for this file's framework (.py)",
            )
        }
    )

    result = await _execution_grounded_routes(
        _state([reconnaissance]), repository_id="admanager_console-2.0", validation_tool=tool
    )

    assert tool.requested_paths == ["server/routes.py"]
    assert result == [
        {
            "source_path": "server/routes.py",
            "routes": [],
            "error": "no route dump is implemented for this file's framework (.py)",
        }
    ]


async def test_no_reconnaissance_artifact_yields_no_evidence_not_an_error() -> None:
    result = await _execution_grounded_routes(
        _state([]), repository_id="admanager_console-2.0", validation_tool=_FakeValidationTool({})
    )

    assert result == []


async def test_a_validation_tool_with_no_dump_method_yields_no_evidence() -> None:
    reconnaissance = _reconnaissance(
        registered_routes=[
            {"method": "GET", "path": "/export", "source_path": "server/routes/a.route.js"},
        ]
    )

    result = await _execution_grounded_routes(
        _state([reconnaissance]),
        repository_id="admanager_console-2.0",
        validation_tool=_LegacyValidationTool(),
    )

    assert result == []


async def test_reconnaissance_from_a_different_repository_is_not_used() -> None:
    reconnaissance = _reconnaissance(
        repository_id="a-different-repository",
        registered_routes=[
            {"method": "GET", "path": "/export", "source_path": "server/routes/a.route.js"},
        ],
    )
    tool = _FakeValidationTool({})

    result = await _execution_grounded_routes(
        _state([reconnaissance]), repository_id="admanager_console-2.0", validation_tool=tool
    )

    assert result == []
    assert tool.requested_paths == []
