"""Cross-repository endpoint manifests, mechanically diffed against the contract.

AB-Feature-184's frontend hardcoded `/v1/service-requests/export-active-ad-units` as a real,
callable constant; its backend registered `/service-requests/export-active-ad-units` -- no
`/v1`. Both repositories individually matched the (wrong) contract text, so nothing caught the
drift between them. These tests reproduce that exact shape against real files on disk (the
only way these extractors operate), and pin that a genuinely matching pair raises nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from artifacts.schemas import CompatibilityPolicy, EndpointContract, IntegrationContractArtifact
from tools.api_surface import (
    diff_endpoint_manifests,
    extract_referenced_endpoints,
    extract_registered_routes,
)

_FRONTEND_API_UTILS = """
const AD_UNIT_EXPORT = '__PATH__';

export async function exportActiveAdUnits(payload) {
  return client.post(AD_UNIT_EXPORT, payload);
}
"""

_BACKEND_INDEX_ROUTE = """
import serviceRequestRoutes from './serviceRequest.route';
router.use('{prefix}', serviceRequestRoutes);
"""

_BACKEND_SERVICE_REQUEST_ROUTE = """
router
  .route('{route_path}')
  .post(validate(schema), asyncHandler(controller.exportActiveAdUnits));
"""


def _endpoint(
    *,
    operation_id: str,
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    path: str,
) -> EndpointContract:
    return EndpointContract(
        operation_id=operation_id,
        method=method,
        path=path,
        summary="Export active ad units.",
        request_schema={"type": "object"},
        response_schema={"type": "object"},
        query_parameters=[],
        path_parameters=[],
        headers=[],
        authentication_required=True,
        authorization_requirements=[],
        success_status_codes=[200],
        error_codes=[],
        version="1.0.0",
    )


def _contract(endpoint: EndpointContract) -> IntegrationContractArtifact:
    return IntegrationContractArtifact(
        schema_version="1.0",
        workflow_id="feature-184",
        artifact_id="009_integration_contract.json",
        producer="feature_planner",
        timestamp=datetime(2026, 10, 7, tzinfo=UTC),
        metadata={"source": "test"},
        validation_status="valid",
        feature_id="feature-184",
        contract_version="1.0.0",
        status="approved",
        api_style="rest",
        endpoints=[endpoint],
        shared_schemas=[],
        authentication_contract=None,
        authorization_rules=[],
        error_contracts=[],
        event_contracts=[],
        environment_variables=[],
        compatibility_policy=CompatibilityPolicy(
            policy="Additive changes only.",
            breaking_change_allowed=False,
            migration_requirements=[],
            rollback_requirements=["Revert coordinated changes."],
        ),
        owning_workstreams=["admanager_console-2.0", "AB-console-admin-2.0"],
        approved_at=datetime(2026, 10, 7, tzinfo=UTC),
    )


def _write(tmp_path: Path, repository: str, relative_path: str, content: str) -> Path:
    file_path = tmp_path / repository / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return tmp_path / repository


def test_a_frontend_v1_prefix_the_backend_never_registers_is_caught(tmp_path: Path) -> None:
    """AB-Feature-184's exact reproduction: consumer and provider silently disagree."""
    frontend_workspace = _write(
        tmp_path,
        "AB-console-admin-2.0",
        "src/apiUtils/serviceRequests.apiUtils.js",
        _FRONTEND_API_UTILS.replace("__PATH__", "/v1/service-requests/export-active-ad-units"),
    )
    backend_workspace = _write(
        tmp_path,
        "admanager_console-2.0",
        "server/routes/index.route.js",
        _BACKEND_INDEX_ROUTE.format(prefix="/service-requests"),
    )
    _write(
        tmp_path,
        "admanager_console-2.0",
        "server/routes/serviceRequest.route.js",
        _BACKEND_SERVICE_REQUEST_ROUTE.format(route_path="/export-active-ad-units"),
    )
    contract = _contract(
        _endpoint(
            operation_id="exportActiveAdUnits",
            method="POST",
            path="/v1/service-requests/export-active-ad-units",
        )
    )

    consumer_refs = {
        "AB-console-admin-2.0": extract_referenced_endpoints(
            frontend_workspace, ["src/apiUtils/serviceRequests.apiUtils.js"]
        ),
        "admanager_console-2.0": [],
    }
    provider_routes = {
        "admanager_console-2.0": extract_registered_routes(
            backend_workspace,
            ["server/routes/index.route.js", "server/routes/serviceRequest.route.js"],
        ),
        "AB-console-admin-2.0": [],
    }

    mismatches = diff_endpoint_manifests(contract, consumer_refs, provider_routes)

    assert len(mismatches) == 1
    (mismatch,) = mismatches
    assert mismatch.operation_id == "exportActiveAdUnits"
    assert mismatch.consumer_confirms is True
    assert mismatch.provider_confirms is False
    assert "admanager_console-2.0" in mismatch.non_confirming_repository_ids


def test_a_genuinely_matching_pair_raises_nothing(tmp_path: Path) -> None:
    """The same shape, corrected: no `/v1` anywhere, and nothing is reported."""
    correct_path = "/service-requests/export-active-ad-units"
    frontend_workspace = _write(
        tmp_path,
        "AB-console-admin-2.0",
        "src/apiUtils/serviceRequests.apiUtils.js",
        _FRONTEND_API_UTILS.replace("__PATH__", correct_path),
    )
    backend_workspace = _write(
        tmp_path,
        "admanager_console-2.0",
        "server/routes/index.route.js",
        _BACKEND_INDEX_ROUTE.format(prefix="/service-requests"),
    )
    _write(
        tmp_path,
        "admanager_console-2.0",
        "server/routes/serviceRequest.route.js",
        _BACKEND_SERVICE_REQUEST_ROUTE.format(route_path="/export-active-ad-units"),
    )
    contract = _contract(
        _endpoint(operation_id="exportActiveAdUnits", method="POST", path=correct_path)
    )

    consumer_refs = {
        "AB-console-admin-2.0": extract_referenced_endpoints(
            frontend_workspace, ["src/apiUtils/serviceRequests.apiUtils.js"]
        ),
        "admanager_console-2.0": [],
    }
    provider_routes = {
        "admanager_console-2.0": extract_registered_routes(
            backend_workspace,
            ["server/routes/index.route.js", "server/routes/serviceRequest.route.js"],
        ),
        "AB-console-admin-2.0": [],
    }

    mismatches = diff_endpoint_manifests(contract, consumer_refs, provider_routes)

    assert mismatches == []


def test_an_untouched_repository_is_not_reported_as_a_mismatch(tmp_path: Path) -> None:
    """No changed files on either side means nothing to compare, not a finding."""
    contract = _contract(_endpoint(operation_id="unrelated", method="GET", path="/unrelated/path"))

    mismatches = diff_endpoint_manifests(
        contract,
        {"frontend": [], "backend": []},
        {"frontend": [], "backend": []},
    )

    assert mismatches == []
