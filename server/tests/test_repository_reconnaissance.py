"""Reconnaissance grounds a plan in the checkout instead of in what the PRD assumed."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import ImageInput, LLMResponse
from agents.recon.agent import RepositoryReconAgent
from agents.shared.contracts import AgentArtifactError
from artifacts.schemas import (
    RepositoryReconnaissanceArtifact,
    Requirement,
    TechnicalPRDArtifact,
)
from prompts.prompt_loader import PromptLoader
from tools.repository_reconnaissance import inspect_repository_for_planning


class RecordingLLMClient:
    """Return queued responses and keep every rendered instruction for assertion."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        """Queue one JSON payload per expected model call."""
        self._payloads = list(payloads)
        self.calls: list[tuple[str, str]] = []

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> LLMResponse:
        """Return the next queued payload as a model response."""
        self.calls.append((instructions, input_text))
        payload = self._payloads.pop(0)

        return LLMResponse(
            response_id=f"response-{len(self.calls)}",
            model="test-model",
            output_text=json.dumps(payload),
            input_tokens=None,
            output_tokens=None,
            provider="test-provider",
            reasoning_effort="high",
        )


def build_repository(root: Path) -> None:
    """Create a checkout whose real convention differs from what a PRD would assume."""
    (root / "server" / "routes").mkdir(parents=True)
    (root / "server" / "config").mkdir(parents=True)
    (root / "server" / "middlewares").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "jest", "lint": "eslint ."}}), encoding="utf-8"
    )
    # The registry: it assembles the repository's own modules, and it is where auth is
    # applied once, globally -- the exact shape that made three live runs invent per-route
    # handler names that do not exist. Its name resembles no convention a stem list holds.
    (root / "server" / "config" / "express.js").write_text(
        "const authenticate = require('../middlewares/authenticate');\n"
        "const health = require('../routes/health');\n"
        "const billing = require('../routes/billing');\n"
        "function configure(app) {\n"
        "  app.use(authenticate.isAllowed);\n"
        "  app.use('/health', health);\n"
        "  app.use('/billing', billing);\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "server" / "middlewares" / "authenticate.js").write_text(
        "function isAllowed(req, res, next) { next(); }\nmodule.exports = { isAllowed };\n",
        encoding="utf-8",
    )
    (root / "server" / "routes" / "health.js").write_text(
        "function healthCheck(req, res) { res.send('OK'); }\n", encoding="utf-8"
    )
    (root / "server" / "routes" / "billing.js").write_text(
        "function listInvoices(req, res) { res.json([]); }\n", encoding="utf-8"
    )
    (root / "tests" / "health.test.js").write_text(
        "function testHealth() { return true; }\n", encoding="utf-8"
    )
    (root / ".env").write_text("SECRET_TOKEN=live-value\n", encoding="utf-8")
    subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)


def technical_prd() -> TechnicalPRDArtifact:
    """Return a minimal technical PRD to read the checkout against."""
    return TechnicalPRDArtifact(
        schema_version="1.0",
        workflow_id="feature-recon",
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        timestamp=datetime(2026, 8, 24, tzinfo=UTC),
        metadata={},
        validation_status="valid",
        title="Expose admin server status",
        solution_summary="Add a status endpoint the console can read.",
        functional_requirements=[
            Requirement(
                requirement_id="FR-1",
                description="Expose the admin server's current status on a protected route.",
                priority="must",
                # The premise the checkout contradicts: this repository has no per-route auth.
                acceptance_criteria=[
                    "The route is protected the way a comparable protected route is."
                ],
                dependencies=[],
            )
        ],
        non_functional_requirements=[],
        data_requirements=[],
        integration_requirements=[],
        security_requirements=[],
        assumptions=[],
        unresolved_questions=[],
    )


def response_payload(**overrides: Any) -> dict[str, Any]:
    """Return a well-formed reconnaissance response, overridable per test."""
    return {
        "summary": "An Express server whose routes live under server/routes.",
        "source_areas": ["server/routes"],
        "test_areas": ["tests"],
        "conventions": [
            {
                "convention_id": "route",
                "kind": "route",
                "description": "Routes are plain modules registered centrally.",
                "evidence_paths": ["server/routes/health.js"],
                "wiring_path": "server/config/express.js",
            }
        ],
        "shared_utilities": [],
        "contradicted_premises": [],
        **overrides,
    }


def test_the_scan_reports_the_checkout_and_withholds_its_credentials(tmp_path: Path) -> None:
    """Evidence must describe the repository without carrying anything secret into a prompt."""
    build_repository(tmp_path)

    evidence = inspect_repository_for_planning(tmp_path, repository_id="backend")

    assert "server/routes/health.js" in evidence.file_inventory
    assert "server/routes" in evidence.source_directories
    assert evidence.declared_scripts == ["lint", "test"]
    # The registry is what a new route has to be added to, so it must be among the files
    # whose symbols were read rather than one more entry in the inventory. It is found by
    # what it assembles: no filename convention would nominate `config/express.js`.
    wiring_paths = [item.path for item in evidence.wiring_files]
    assert wiring_paths == ["server/config/express.js"]
    assert "configure" in evidence.wiring_files[0].symbols
    # A leaf route imports nothing of its own and is not an assembly point.
    assert "server/routes/health.js" not in wiring_paths
    assert not any(".env" in path for path in evidence.file_inventory)


def test_a_registry_of_dotted_module_names_is_still_recognised(tmp_path: Path) -> None:
    """Imports of files whose names contain dots have to count, and rank as assembly.

    The pilot backend names every module `app.route.js`, `app.service.js`,
    `adUnit.validation.js`. Reference counting replaced every dot with a separator to resolve
    a Python `package.module` import, which turned `./app/app.route` into `//app/app/route`
    and resolved it to `route`, matching nothing. Its route index imports two dozen sibling
    routes and scored zero, so reconnaissance offered large consumers as that repository's
    registries and the file a new route had to be added to was never shown at all.
    """
    routes = tmp_path / "server" / "routes"
    routes.mkdir(parents=True)
    for name in ("auth.route.js", "user.route.js", "report.route.js"):
        (routes / name).write_text("export default {};\n", encoding="utf-8")
    (routes / "index.route.js").write_text(
        "import authRoutes from './auth.route';\n"
        "import userRoutes from './user.route';\n"
        "import reportRoutes from './report.route';\n"
        "export default { authRoutes, userRoutes, reportRoutes };\n",
        encoding="utf-8",
    )
    # More own references than the index, and not an assembly point: this is the shape that
    # outranked the route index when the ordering was by count alone.
    (tmp_path / "server" / "models.js").write_text(
        "import authRoutes from './routes/auth.route';\n"
        "import userRoutes from './routes/user.route';\n"
        "import reportRoutes from './routes/report.route';\n"
        "import indexRoutes from './routes/index.route';\n"
        "export default { authRoutes, userRoutes, reportRoutes, indexRoutes };\n",
        encoding="utf-8",
    )

    evidence = inspect_repository_for_planning(tmp_path, repository_id="backend")

    wiring_paths = [item.path for item in evidence.wiring_files]
    # Counted at all, which the dot-stripping prevented.
    assert "server/routes/index.route.js" in wiring_paths
    # And offered first, ahead of the consumer that references more.
    assert wiring_paths[0] == "server/routes/index.route.js"
    assert wiring_paths.index("server/routes/index.route.js") < wiring_paths.index(
        "server/models.js"
    )


@pytest.mark.asyncio
async def test_reconnaissance_publishes_what_the_checkout_contains(tmp_path: Path) -> None:
    """The artifact carries the repository's real wiring point and its own revision."""
    build_repository(tmp_path)
    client = RecordingLLMClient(response_payload())
    agent = RepositoryReconAgent(prompt_loader=PromptLoader(), llm_client=client)

    artifact = await agent.inspect(
        feature_id="feature-recon",
        repository_id="backend",
        workspace_root=str(tmp_path),
        technical_prd=technical_prd(),
    )

    assert isinstance(artifact, RepositoryReconnaissanceArtifact)
    assert artifact.artifact_id == "015_repository_reconnaissance.backend.json"
    assert artifact.repository_id == "backend"
    assert artifact.repository_revision
    assert artifact.conventions[0].wiring_path == "server/config/express.js"
    instructions, _ = client.calls[0]
    assert "server/routes/health.js" in instructions
    assert "live-value" not in instructions


@pytest.mark.asyncio
async def test_a_path_the_scan_never_read_is_refused(tmp_path: Path) -> None:
    """A guessed path in reconnaissance is worse than none: the planner is told to trust it.

    The whole purpose of this stage is to replace assumed conventions with read ones. A model
    free to name `src/routes/` because that is where routes usually live would reintroduce the
    original failure wearing evidence's authority.
    """
    build_repository(tmp_path)
    invented = response_payload(source_areas=["src/routes"])
    client = RecordingLLMClient(invented, invented)
    agent = RepositoryReconAgent(prompt_loader=PromptLoader(), llm_client=client)

    with pytest.raises(AgentArtifactError, match="absent from the checkout"):
        await agent.inspect(
            feature_id="feature-recon",
            repository_id="backend",
            workspace_root=str(tmp_path),
            technical_prd=technical_prd(),
        )

    # Rejected once, repaired once, then refused -- never silently accepted.
    assert len(client.calls) == 2
    assert "absent from the checkout" in client.calls[1][0]


@pytest.mark.asyncio
async def test_a_repaired_response_is_accepted_and_records_why(tmp_path: Path) -> None:
    """One bad path must not cost the feature its only grounded plan."""
    build_repository(tmp_path)
    client = RecordingLLMClient(response_payload(source_areas=["src/routes"]), response_payload())
    agent = RepositoryReconAgent(prompt_loader=PromptLoader(), llm_client=client)

    artifact = await agent.inspect(
        feature_id="feature-recon",
        repository_id="backend",
        workspace_root=str(tmp_path),
        technical_prd=technical_prd(),
    )

    assert artifact.source_areas == ["server/routes"]
    assert artifact.metadata["repair_of_response_id"] == "response-1"
    assert artifact.metadata["repair_reason"]


@pytest.mark.asyncio
async def test_a_contradicted_premise_is_carried_into_the_artifact(tmp_path: Path) -> None:
    """The finding that saves a run is the one that says the requirement's premise is false."""
    build_repository(tmp_path)
    client = RecordingLLMClient(
        response_payload(
            contradicted_premises=[
                {
                    "premise": "Per-route authentication can be added the way a comparable "
                    "protected route does it.",
                    "contradicted_by": "Authentication is applied once, globally; no route "
                    "file applies it.",
                    "evidence_paths": ["server/config/express.js"],
                    "question": "Should the new route rely on the global middleware, or is "
                    "per-route authorisation being introduced by this feature?",
                }
            ]
        )
    )
    agent = RepositoryReconAgent(prompt_loader=PromptLoader(), llm_client=client)

    artifact = await agent.inspect(
        feature_id="feature-recon",
        repository_id="backend",
        workspace_root=str(tmp_path),
        technical_prd=technical_prd(),
    )

    premise = artifact.contradicted_premises[0]
    assert "globally" in premise.contradicted_by
    assert premise.evidence_paths == ["server/config/express.js"]
    assert premise.question
