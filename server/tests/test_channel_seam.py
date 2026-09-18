"""The review sees the seam the change stands on (72-).

AB-Feature-206's blocker: a change imported a channel package bare instead of the instance
one unchanged module configures, its own unit test mocked the package and asserted the call
config, and every gate passed. The configuration module never entered review evidence -- it
was unchanged -- so the reviewer could not flag a mismatch with a file it never read.

Three mechanisms, tested here end to end: (A) the co-importers of a channel package the
change imports become required evidence for the engineer and the reviewer; (B) a changed test
that mocks a channel package and asserts on that mock is named as laundering in a
review-evidence note that never fails the attempt (47-D's law); (C) the reviewer's prompt
carries the seam clauses whenever the diff touches a channel package.

Repo-agnostic like everything else: no fixture names a package outside the per-ecosystem
tables in `tools.channel_packages` -- every package below is read out of those tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import MockCodingExecutor
from agents.engineer.agent import (
    _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS,
    EngineerAgent,
    _channel_seam_paths,
    _repository_context,
    _required_context_paths,
)
from agents.reviewer.agent import ReviewerAgent
from prompts.prompt_loader import PromptLoader
from tests.test_agents import (
    StaticLLMClient,
    StaticValidationTool,
    agent_state,
    domain_payload,
    review_artifact,
    task_plan_artifact,
    technical_prd_artifact,
    validation_result,
)
from tools.channel_packages import (
    CHANNEL_ECOSYSTEMS,
    channel_seam_scan,
    mocked_channel_packages,
)
from tools.file_tools import WorkspaceFileTools
from tools.scoped_tests import seam_mock_note, seam_mock_notes

# The fixture packages come from the tables, never from a spelled-out name: the tables are
# the one place a package name may live, and these picks are deterministic.
_NODE = next(item for item in CHANNEL_ECOSYSTEMS if ".js" in item.source_suffixes)
_PYTHON = next(item for item in CHANNEL_ECOSYSTEMS if ".py" in item.source_suffixes)
_NODE_PACKAGE = sorted(name for name in _NODE.packages if re.fullmatch(r"[a-z]\w*", name))[0]
_PYTHON_PACKAGE = sorted(_PYTHON.packages)[0]

_CONFIG_MODULE = "src/api/http.js"
_CONFIG_MARKER = "SEAM_CONFIGURED_INSTANCE_MARKER"


def _config_module_source() -> str:
    """The 206 shape: one module creates the configured instance of the channel package."""
    return (
        f"import client from '{_NODE_PACKAGE}';\n"
        f"// {_CONFIG_MARKER}\n"
        "export const configured = client.create({ baseURL: '/internal' });\n"
    )


def _bare_import_source() -> str:
    """The changed file that steps around the configuration: it imports the package bare."""
    return (
        f"import client from '{_NODE_PACKAGE}';\n"
        "export const upload = (payload) => client.post('/upload', payload);\n"
    )


def _laundering_test_source() -> str:
    """A test that mocks the channel package and asserts the call config on that mock."""
    return (
        f"import client from '{_NODE_PACKAGE}';\n"
        f"jest.mock('{_NODE_PACKAGE}');\n"
        "test('uploads', () => {\n"
        "  expect(client.post).toHaveBeenCalledWith('/upload');\n"
        "});\n"
    )


def _reviewer(client: StaticLLMClient) -> ReviewerAgent:
    """One reviewer with passing validation, the way the agent unit tests build one."""
    return ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    )


# --------------------------------------------------------------------------------------
# T1 -- the co-importer set, and the common case paying nothing
# --------------------------------------------------------------------------------------


def test_the_co_importer_set_is_exactly_the_module_sharing_the_channel(tmp_path: Path) -> None:
    """The seam of 206 was two files: the configuration module and the new code itself."""
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "util").mkdir(parents=True)
    (tmp_path / _CONFIG_MODULE).write_text(_config_module_source(), encoding="utf-8")
    (tmp_path / "src" / "util" / "format.js").write_text(
        "export const format = (value) => String(value);\n", encoding="utf-8"
    )
    tools = WorkspaceFileTools(tmp_path)

    scan = channel_seam_scan(
        [("src/features/upload.js", _bare_import_source())],
        [_CONFIG_MODULE, "src/util/format.js", "src/features/upload.js"],
        lambda path: tools.read_file(path),
    )

    assert scan.packages == (_NODE_PACKAGE,)
    assert [item.path for item in scan.co_importers] == [_CONFIG_MODULE]
    assert scan.co_importers[0].channel_packages == (_NODE_PACKAGE,)
    assert scan.co_importer_count == 1
    assert scan.truncated is False


def test_a_change_with_no_channel_import_never_reads_the_checkout() -> None:
    """The common case pays nothing: no channel import means no checkout walk at all."""
    reads: list[str] = []

    def read(path: str) -> str | None:
        reads.append(path)
        return ""

    scan = channel_seam_scan(
        [("src/features/format.js", "import { pad } from './pad';\nexport const f = pad;\n")],
        ["src/api/http.js"],
        read,
    )

    assert scan.packages == ()
    assert scan.co_importers == ()
    assert reads == []


# --------------------------------------------------------------------------------------
# T2 -- the engineer's required context on attempt 0
# --------------------------------------------------------------------------------------


def test_attempt_zero_required_context_carries_the_seam_module(tmp_path: Path) -> None:
    """The tier must not depend on attempt history: 206 died on attempt 0.

    On a first attempt there is no prior diff and no changed set, so the detection reads the
    plan's assigned files from the checkout -- the same source that makes the imported-modules
    tier say anything on attempt 1.
    """
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "features").mkdir(parents=True)
    assigned = "src/features/useUpload.js"
    (tmp_path / assigned).write_text(_bare_import_source(), encoding="utf-8")
    (tmp_path / _CONFIG_MODULE).write_text(_config_module_source(), encoding="utf-8")
    existing = {Path(assigned), Path(_CONFIG_MODULE)}
    tools = WorkspaceFileTools(tmp_path)

    seam = _channel_seam_paths((), existing, tools, "", (assigned,))

    assert seam == (_CONFIG_MODULE,)

    required = _required_context_paths(
        {
            "expected_files_or_areas": [assigned],
            "registry_paths": ["src/routes/index.js"],
        },
        channel_seam_paths=seam,
    )

    # After the imports tier (empty here) and ahead of the registries.
    assert required == (assigned, _CONFIG_MODULE, "src/routes/index.js")


def test_an_oversized_seam_module_is_excerpted_by_the_existing_radius_rules(
    tmp_path: Path,
) -> None:
    """The 61- excerpt rules apply to a seam file unchanged: windowed, never dropped silently."""
    (tmp_path / "src" / "api").mkdir(parents=True)
    # Sized from the per-file constant, not a literal: ~30-character rows, twice the bound.
    row_count = _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS // 15
    filler = "\n".join(f"export const row{index:04d} = {index};" for index in range(row_count))
    oversized = f"import client from '{_NODE_PACKAGE}';\n{filler}\n"
    (tmp_path / _CONFIG_MODULE).write_text(oversized, encoding="utf-8")
    tools = WorkspaceFileTools(tmp_path)

    context = _repository_context({Path(_CONFIG_MODULE)}, tools, frozenset(), (_CONFIG_MODULE,))

    entry = next(item for item in context["files"] if item["path"] == _CONFIG_MODULE)
    assert "excerpt_lines" in entry
    assert context["required_excerpted_paths"] == [_CONFIG_MODULE]
    assert context["required_dropped_paths"] == []


# --------------------------------------------------------------------------------------
# T3 -- reviewer evidence: the banner, and the drop that is declared rather than trimmed
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_evidence_shows_the_seam_module_under_the_unchanged_banner(
    tmp_path: Path,
) -> None:
    """An unchanged co-importer enters evidence, labelled as non-diff source."""
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / _CONFIG_MODULE).write_text(_config_module_source(), encoding="utf-8")
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/features/upload.js": _bare_import_source()}
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await _reviewer(client).run(
        agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion])
    )

    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    assert evidence["limitations"] == []
    seam_entry = next(item for item in evidence["files"] if item["path"] == _CONFIG_MODULE)
    assert seam_entry["content_kind"] == "channel_seam_context"
    assert seam_entry["channel_packages"] == [_NODE_PACKAGE]
    assert seam_entry["content"].startswith(
        f"===== {_CONFIG_MODULE}: unchanged from the revision this change branched from ====="
    )
    assert _CONFIG_MARKER in seam_entry["content"]
    assert evidence["channel_seam"]["packages"] == [_NODE_PACKAGE]
    assert evidence["channel_seam"]["files"] == [_CONFIG_MODULE]
    # The seam module is context, not a reviewed change: the publication fingerprint and the
    # reviewed set stay exactly what the change wrote.
    review = update["artifacts"][0]
    assert review.metadata["reviewed_file_paths"] == ["src/features/upload.js"]
    assert review.verdict == "approved"


@pytest.mark.asyncio
async def test_a_seam_module_past_the_budget_is_dropped_whole_and_declared(
    tmp_path: Path,
) -> None:
    """The 51-A fences, byte-identical: a named limitation and no truncated file."""
    (tmp_path / "src" / "api").mkdir(parents=True)
    filler = "\n".join(f"export const row{index:05d} = '{index:0>40}';" for index in range(500))
    (tmp_path / _CONFIG_MODULE).write_text(
        f"import client from '{_NODE_PACKAGE}';\n// {_CONFIG_MARKER}\n{filler}\n",
        encoding="utf-8",
    )
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/features/upload.js": _bare_import_source()}
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await _reviewer(client).run(
        agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion])
    )

    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    assert "seam_context_omitted" in evidence["limitations"]
    assert all(item["path"] != _CONFIG_MODULE for item in evidence["files"])
    assert evidence["channel_seam"]["omitted"] == [
        {
            "path": _CONFIG_MODULE,
            "channel_packages": [_NODE_PACKAGE],
            "reason": "evidence_budget",
        }
    ]
    # A drop is a drop: not one byte of the file reached the model input.
    assert _CONFIG_MARKER not in client.calls[0][1]
    review = update["artifacts"][0]
    assert review.verdict == "changes_requested"
    assert review.metadata["retryable"] is True
    finding = next(
        item for item in review.findings if item.finding_id == "REVIEW_EVIDENCE_INCOMPLETE"
    )
    assert finding.file_path == _CONFIG_MODULE
    assert _NODE_PACKAGE in finding.description


# --------------------------------------------------------------------------------------
# T4 -- the detector truth table
# --------------------------------------------------------------------------------------


def test_the_seam_mock_detector_truth_table() -> None:
    """Mock of a channel plus an assertion on the mock is the one shape that is named."""
    path = "src/features/upload.test.js"

    laundering = _laundering_test_source()
    assert seam_mock_notes(path, laundering) == (seam_mock_note(_NODE_PACKAGE, path),)

    # A mock with no call-assertions on it: the test isolates, it does not claim the seam.
    effect_only = (
        f"import client from '{_NODE_PACKAGE}';\n"
        f"jest.mock('{_NODE_PACKAGE}');\n"
        "test('formats', () => {\n"
        "  expect(format(upload())).toEqual('ok');\n"
        "});\n"
    )
    assert seam_mock_notes(path, effect_only) == ()

    # A call-assertion on a different double: the channel's own names never appear in it.
    other_double = (
        f"import client from '{_NODE_PACKAGE}';\n"
        f"jest.mock('{_NODE_PACKAGE}');\n"
        "test('notifies', () => {\n"
        "  expect(onUploadComplete).toHaveBeenCalled();\n"
        "});\n"
    )
    assert seam_mock_notes(path, other_double) == ()

    # A boundary fake: nothing mocks the package, the real client is pointed at a stub.
    boundary_fake = (
        "import { createStubServer } from './support/stub';\n"
        "import { upload } from './upload';\n"
        "test('uploads for real', async () => {\n"
        "  const server = await createStubServer();\n"
        "  const response = await upload(server.url);\n"
        "  expect(response.status).toBe(200);\n"
        "});\n"
    )
    assert seam_mock_notes(path, boundary_fake) == ()

    # A mock of the repository's own module: not a channel package, whatever it wraps.
    own_module = (
        "import { post } from './api/client';\n"
        "jest.mock('./api/client');\n"
        "test('uploads', () => {\n"
        "  expect(post).toHaveBeenCalledWith('/upload');\n"
        "});\n"
    )
    assert seam_mock_notes(path, own_module) == ()


def test_the_detector_reads_python_mocks_through_the_same_table() -> None:
    """A patch by string target binds no visible handle; missing that note is the costly
    direction, so it is noted rather than waved through."""
    path = "tests/test_publish.py"
    decorated = (
        "from unittest.mock import patch\n"
        "\n"
        f"@patch('{_PYTHON_PACKAGE}.connect')\n"
        "def test_publishes(mock_connect):\n"
        "    publish()\n"
        "    mock_connect.assert_called_once()\n"
    )
    assert seam_mock_notes(path, decorated) == (seam_mock_note(_PYTHON_PACKAGE, path),)

    aliased_effect_only = (
        "from unittest.mock import patch\n"
        "\n"
        "def test_publishes():\n"
        f"    with patch('{_PYTHON_PACKAGE}.connect') as fake:\n"
        "        assert publish() == 'queued'\n"
    )
    assert seam_mock_notes(path, aliased_effect_only) == ()

    assert mocked_channel_packages(path, decorated) == (_PYTHON_PACKAGE,)


# --------------------------------------------------------------------------------------
# T5 -- the note reaches the review and never fails the attempt (47-D)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_note_reaches_review_input_and_artifact_without_failing_the_attempt(
    tmp_path: Path,
) -> None:
    """The detector informs; it must never fail the attempt -- 47-D, stated as a test."""
    test_path = "src/features/upload.test.js"
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={
                    "src/features/upload.js": _bare_import_source(),
                    test_path: _laundering_test_source(),
                }
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await _reviewer(client).run(
        agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion])
    )

    note = seam_mock_note(_NODE_PACKAGE, test_path)
    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    assert evidence["seam_mock_notes"] == [note]
    test_entry = next(item for item in evidence["files"] if item["path"] == test_path)
    assert test_entry["seam_mock_notes"] == [note]
    review = update["artifacts"][0]
    assert review.metadata["seam_mock_notes"] == [note]
    # The 47-D assertion: the note raised no limitation, forced no verdict, failed nothing.
    assert evidence["limitations"] == []
    assert review.verdict == "approved"
    assert review.metadata["manual_review_required"] is False


# --------------------------------------------------------------------------------------
# T6 -- one registry, one code path
# --------------------------------------------------------------------------------------


def test_both_tables_drive_the_same_detection_code_path(tmp_path: Path) -> None:
    """Node and Python are rows in one module's registry, not branches in the detection."""
    assert {item.name for item in CHANNEL_ECOSYSTEMS} >= {"node", "python"}
    assert all(item.packages for item in CHANNEL_ECOSYSTEMS)

    (tmp_path / "node_repo").mkdir()
    (tmp_path / "py_repo").mkdir()
    (tmp_path / "node_repo" / "http.js").write_text(_config_module_source(), encoding="utf-8")
    (tmp_path / "py_repo" / "session.py").write_text(
        f"import {_PYTHON_PACKAGE}\n\nCONNECTION = {_PYTHON_PACKAGE}.connect()\n",
        encoding="utf-8",
    )
    tools = WorkspaceFileTools(tmp_path)

    node_scan = channel_seam_scan(
        [("changed.js", f"import client from '{_NODE_PACKAGE}';\n")],
        ["node_repo/http.js", "py_repo/session.py"],
        lambda path: tools.read_file(path),
    )
    python_scan = channel_seam_scan(
        [("changed.py", f"from {_PYTHON_PACKAGE} import connect\n")],
        ["node_repo/http.js", "py_repo/session.py"],
        lambda path: tools.read_file(path),
    )

    assert [item.path for item in node_scan.co_importers] == ["node_repo/http.js"]
    assert [item.path for item in python_scan.co_importers] == ["py_repo/session.py"]
    assert node_scan.packages == (_NODE_PACKAGE,)
    assert python_scan.packages == (_PYTHON_PACKAGE,)


# --------------------------------------------------------------------------------------
# T7 -- the 206 replay shape
# --------------------------------------------------------------------------------------


class _RecordingCodingExecutor(MockCodingExecutor):
    """The mock executor, additionally keeping the instructions each execution was given."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.instructions: list[str] = []

    async def execute(self, **kwargs: Any) -> Any:
        self.instructions.append(kwargs["instructions"])
        return await super().execute(**kwargs)


@pytest.mark.asyncio
async def test_the_206_replay_shape_is_caught_at_every_layer(tmp_path: Path) -> None:
    """A bare channel import, a configuration co-importer, a laundering test: A shows the
    config module to the coder and the reviewer, B emits the note, C renders clause one."""
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "features").mkdir(parents=True)
    (tmp_path / _CONFIG_MODULE).write_text(_config_module_source(), encoding="utf-8")
    assigned = "src/features/useUpload.js"
    (tmp_path / assigned).write_text(_bare_import_source(), encoding="utf-8")
    # Enough unrelated files that presence in the prompt is force-inclusion, not accident.
    (tmp_path / "src" / "pages").mkdir(parents=True)
    for index in range(90):
        (tmp_path / "src" / "pages" / f"page_{index:02d}.js").write_text(
            f"// page {index}\n", encoding="utf-8"
        )
    plan = task_plan_artifact()
    plan.metadata["review_scope"] = {
        "repository_id": "repository-1",
        "workstream_id": "workstream-1",
        "role": "frontend",
        "requirement_ids": ["requirement-1"],
        "expected_files_or_areas": [assigned],
    }
    executor = _RecordingCodingExecutor(
        file_updates={
            "src/features/upload.js": _bare_import_source(),
            "src/features/upload.test.js": _laundering_test_source(),
        }
    )

    completion = (
        await EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor).run(
            agent_state(tmp_path, [plan])
        )
    )["artifacts"][0]

    # A, coder side: the configuration module's content reached the attempt's instructions.
    assert _CONFIG_MARKER in executor.instructions[0]

    client = StaticLLMClient(domain_payload(review_artifact()))
    await _reviewer(client).run(agent_state(tmp_path, [technical_prd_artifact(), plan, completion]))

    instructions, input_text = client.calls[0]
    evidence = json.loads(input_text)["workspace_change_evidence"]
    # A, reviewer side: the configuration module is in evidence under the banner.
    seam_entry = next(item for item in evidence["files"] if item["path"] == _CONFIG_MODULE)
    assert _CONFIG_MARKER in seam_entry["content"]
    # B: the laundering test is named.
    assert (
        seam_mock_note(_NODE_PACKAGE, "src/features/upload.test.js")
        in (evidence["seam_mock_notes"])
    )
    # C: clause one is rendered, conditioned on the diff touching the channel package.
    assert _NODE_PACKAGE in instructions
    assert "real channel configuration against a boundary fake" in instructions
    assert "seam_mock_notes" in instructions


# --------------------------------------------------------------------------------------
# T8 -- the wrapper over a channel client is a channel package too (87- A1)
# --------------------------------------------------------------------------------------

# The three names below are literals, and this is the one place in this suite that is
# allowed: A1 *is* a table edit, so the only thing there is to assert is that the rows are
# there. Reading them back out of the table would assert nothing at all.
_HTTP_WRAPPER = "axios-hooks"
_WRAPPER_FAMILY = frozenset({"swr", "react-query", "@tanstack/react-query", "ky-universal"})


def test_a_client_wrapper_earns_a_row_like_the_client_under_it() -> None:
    """AB-Feature-218 imported `axios-hooks` and never `axios`, so nothing matched.

    The table's own rule already covered the case -- a package earns a row when importing it
    *is* opening a side-effect channel, because those are the packages a repository configures
    once. A wrapper is configured once in exactly the same way (`configure({ axios })`, a
    global fetcher, a QueryClient), so importing it bare steps around the same configuration.
    """
    assert _HTTP_WRAPPER in _NODE.packages
    assert _NODE.packages >= _WRAPPER_FAMILY
    # And the rule's other half still holds: a utility package is not a channel, whatever
    # else it is popular. A row here would make every change grow seam context.
    assert not ({"lodash", "date-fns", "classnames", "uuid"} & _NODE.packages)


def test_the_wrapper_row_alone_gives_the_change_a_co_importer(tmp_path: Path) -> None:
    """A1's crude win, on 218's shape: the module that configures the wrapper is reachable.

    Crude because it is a name, and the next repository will import a wrapper this table has
    never heard of. 87- Part A2 is what makes the *configuration* the thing being looked for.
    """
    client_module = "src/config/client.js"
    (tmp_path / "src" / "config").mkdir(parents=True)
    (tmp_path / client_module).write_text(
        f"import Client from '{_NODE_PACKAGE}';\n"
        f"import {{ configure }} from '{_HTTP_WRAPPER}';\n"
        "const client = Client.create({ baseURL: '/internal/api' });\n"
        "configure({ client });\n",
        encoding="utf-8",
    )
    tools = WorkspaceFileTools(tmp_path)

    scan = channel_seam_scan(
        [("src/pages/Upload.js", f"import useClient from '{_HTTP_WRAPPER}';\n")],
        [client_module, "src/pages/Upload.js"],
        lambda path: tools.read_file(path),
    )

    assert scan.packages == (_HTTP_WRAPPER,)
    assert [item.path for item in scan.co_importers] == [client_module]
