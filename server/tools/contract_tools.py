"""Deterministic OpenAPI generation and contract validation boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import yaml

from artifacts.schemas import ChildWorkflowResultArtifact, IntegrationContractArtifact


@dataclass(frozen=True, slots=True)
class ContractValidationFinding:
    """A generator-level incompatibility that the integration reviewer can route safely."""

    finding_id: str
    severity: str
    responsible_repository_id: str
    affected_repository_ids: tuple[str, ...]
    contract_reference: str
    description: str
    evidence: str
    recommended_fix: str


class ContractCodeGenerator(Protocol):
    """Generate portable contract representations without granting write access to agents."""

    async def generate_openapi(
        self, contract: IntegrationContractArtifact
    ) -> dict[str, Any] | None:
        """Return an OpenAPI document for REST or mixed contracts."""

    async def generate_backend_models(self, contract: IntegrationContractArtifact) -> str:
        """Return a language-neutral backend model representation."""

    async def generate_frontend_client(self, contract: IntegrationContractArtifact) -> str:
        """Return a language-neutral typed-client representation."""

    async def validate_contract(
        self,
        contract: IntegrationContractArtifact,
        child_results: Sequence[ChildWorkflowResultArtifact],
    ) -> list[ContractValidationFinding]:
        """Report contract violations without modifying a child workspace."""


class OpenAPIContractCodeGenerator:
    """Generate a conservative OpenAPI 3.1 projection of approved REST operations."""

    async def generate_openapi(
        self, contract: IntegrationContractArtifact
    ) -> dict[str, Any] | None:
        """Return ``None`` for contracts that do not expose an HTTP API."""
        if contract.api_style not in {"rest", "mixed"}:
            return None
        paths: dict[str, Any] = {}
        for endpoint in contract.endpoints:
            operation: dict[str, Any] = {
                "operationId": endpoint.operation_id,
                "summary": endpoint.summary,
                "responses": {
                    str(code): {
                        "description": "Success response",
                        "content": {"application/json": {"schema": endpoint.response_schema}},
                    }
                    for code in endpoint.success_status_codes
                },
            }
            if endpoint.request_schema is not None:
                operation["requestBody"] = {
                    "required": True,
                    "content": {"application/json": {"schema": endpoint.request_schema}},
                }
            parameters = [
                {**parameter, "in": "query"} for parameter in endpoint.query_parameters
            ] + [{**parameter, "in": "path"} for parameter in endpoint.path_parameters]
            if parameters:
                operation["parameters"] = parameters
            if endpoint.authentication_required:
                operation["security"] = [{"contractAuth": []}]
            paths.setdefault(endpoint.path, {})[endpoint.method.lower()] = operation
        document: dict[str, Any] = {
            "openapi": "3.1.0",
            "info": {
                "title": f"Feature {contract.feature_id}",
                "version": contract.contract_version,
            },
            "paths": paths,
            "components": {
                "schemas": {schema.name: schema.json_schema for schema in contract.shared_schemas},
            },
        }
        if contract.authentication_contract is not None:
            document["components"]["securitySchemes"] = {
                "contractAuth": {
                    "type": "http",
                    "scheme": contract.authentication_contract.scheme,
                    "description": contract.authentication_contract.description,
                }
            }
        return document

    async def generate_backend_models(self, contract: IntegrationContractArtifact) -> str:
        """Expose JSON Schema definitions for a backend validation-model generator."""
        return yaml.safe_dump(
            {schema.name: schema.json_schema for schema in contract.shared_schemas}, sort_keys=True
        )

    async def generate_frontend_client(self, contract: IntegrationContractArtifact) -> str:
        """Expose stable operation IDs for typed-client generation by repository tooling."""
        return yaml.safe_dump(
            {
                "operations": [
                    {"operationId": item.operation_id, "method": item.method, "path": item.path}
                    for item in contract.endpoints
                ]
            },
            sort_keys=True,
        )

    async def validate_contract(
        self,
        contract: IntegrationContractArtifact,
        child_results: Sequence[ChildWorkflowResultArtifact],
    ) -> list[ContractValidationFinding]:
        """Check every result is bound to this revision and every owner is ready."""
        by_repository = {result.repository_id: result for result in child_results}
        findings: list[ContractValidationFinding] = []
        for child_result in child_results:
            contract_artifact_id = child_result.metadata.get("contract_artifact_id")
            contract_version = child_result.metadata.get("contract_version")
            if (
                contract_artifact_id != contract.artifact_id
                or contract_version != contract.contract_version
            ):
                findings.append(
                    ContractValidationFinding(
                        finding_id=f"stale-contract-{child_result.repository_id}",
                        severity="critical",
                        responsible_repository_id=child_result.repository_id,
                        affected_repository_ids=(child_result.repository_id,),
                        contract_reference=contract.artifact_id,
                        description=(
                            "Repository result was reviewed against a different contract revision."
                        ),
                        evidence=(
                            "Child result contract identity does not match the active contract."
                        ),
                        recommended_fix=(
                            "Rerun the repository workstream against the active contract revision."
                        ),
                    )
                )
        for repository_id in contract.owning_workstreams:
            result = by_repository.get(repository_id)
            if result is None:
                findings.append(
                    ContractValidationFinding(
                        finding_id=f"missing-owner-{repository_id}",
                        severity="critical",
                        responsible_repository_id=repository_id,
                        affected_repository_ids=(repository_id,),
                        contract_reference="owning_workstreams",
                        description="No child workflow result was produced for a contract owner.",
                        evidence="The approved contract names this repository as an owner.",
                        recommended_fix="Run the missing required repository workstream.",
                    )
                )
                continue
            if result.status != "approved" or not result.pull_request_readiness:
                findings.append(
                    ContractValidationFinding(
                        finding_id=f"unready-owner-{repository_id}",
                        severity="high",
                        responsible_repository_id=repository_id,
                        affected_repository_ids=(repository_id,),
                        contract_reference="owning_workstreams",
                        description="A contract owner is not ready for a coordinated pull request.",
                        evidence=f"Child result status is {result.status}.",
                        recommended_fix="Resolve the repository review findings and rerun only it.",
                    )
                )
        return findings


class MockContractCodeGenerator(OpenAPIContractCodeGenerator):
    """Deterministic generator used by mock workflows and tests without external tooling."""


__all__ = [
    "ContractCodeGenerator",
    "ContractValidationFinding",
    "MockContractCodeGenerator",
    "OpenAPIContractCodeGenerator",
]
