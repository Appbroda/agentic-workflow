"""The review sees the client the call goes through (87- Part A).

72- shipped the seam scan: the modules that co-import a channel package a change imports
become the review's evidence. AB-Feature-218 walked straight past it. Its frontend imported
`axios-hooks` and never `axios`, the table held only `axios`, and so `channel_seam_packages`
was recorded `[]` -- while both production blockers that shipped lived in the one module the
change never touched and no gate ever showed: `src/config/axios.js`, whose `baseURL` already
ends in `/api` and whose request interceptor forces `Content-Type: application/json` over a
multipart body.

Two layers are tested here, and they are not alternatives:

* **A1** -- the wrapper packages earn their rows. One table edit; it closes 218's hole and
  nothing more, because the next repository will import a wrapper this table has never heard
  of.
* **A2** -- the seam the change stands on is the module the repository *configured*, found
  structurally: a co-importer that calls a channel package's constructor or configuration
  entry point through a name that package's own import bound. No filename rule anywhere --
  `config/axios.js` and `lib/api.js` are two repositories' conventions and this platform
  carries neither -- and a channel whose configuration module cannot be found says so.

The 218 replay below spells `axios`, `axios-hooks` and `src/config/axios.js` verbatim. That is
deliberate and it is the one place it is allowed: the test's whole job is to replay a run that
happened, and a fixture that abstracted the names could not say that *this* change would now
be reviewed beside *that* file. Every other fixture reads its package out of the tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from adapters.llm_adapter import MockCodingExecutor
from agents.engineer.agent import EngineerAgent
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
    channel_configuration_calls,
    channel_seam_scan,
)
from tools.file_tools import WorkspaceFileTools

_NODE = next(item for item in CHANNEL_ECOSYSTEMS if ".js" in item.source_suffixes)
_PYTHON = next(item for item in CHANNEL_ECOSYSTEMS if ".py" in item.source_suffixes)
# The same deterministic picks 72-'s suite makes, out of the tables and never spelled out.
_NODE_PACKAGE = sorted(name for name in _NODE.packages if re.fullmatch(r"[a-z]\w*", name))[0]
_PYTHON_PACKAGE = sorted(_PYTHON.packages)[0]

# 218's own three names. See the module docstring for why these are literals.
_HTTP_CLIENT = "axios"
_HTTP_WRAPPER = "axios-hooks"
_CLIENT_MODULE = "src/config/axios.js"
_BASE_URL_LINE = "baseURL: `https://${base}.appbroda.com/api`"
_INTERCEPTOR_LINE = "request.headers['Content-Type'] = 'application/json';"


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


def _client_module_source() -> str:
    """218's `src/config/axios.js`, reduced to the two lines that made the feature 404."""
    return (
        f"import Axios from '{_HTTP_CLIENT}';\n"
        f"import {{ configure }} from '{_HTTP_WRAPPER}';\n"
        "\n"
        "const base = process.env.REACT_APP_ENV;\n"
        f"const axios = Axios.create({{ {_BASE_URL_LINE} }});\n"
        "\n"
        "axios.interceptors.request.use((request) => {\n"
        f"  {_INTERCEPTOR_LINE}\n"
        "  return request;\n"
        "});\n"
        "\n"
        "configure({ axios });\n"
        "\n"
        "export default axios;\n"
    )


def _changed_page_source() -> str:
    """218's change: the wrapper imported bare, and a URL that re-prefixes the base."""
    return (
        f"import useAxios from '{_HTTP_WRAPPER}';\n"
        "\n"
        "export const useBulkCreateApps = () =>\n"
        "  useAxios({ url: '/api/apps/bulk', method: 'POST' }, { manual: true });\n"
    )


# --------------------------------------------------------------------------------------
# A2 -- the configuration module, found structurally
# --------------------------------------------------------------------------------------


def test_the_configuration_call_is_read_through_the_binding_the_import_made() -> None:
    """`Axios.create(` counts because `Axios` is what this file's own import of the
    package bound. Nothing here reads the path, and nothing reads a file's name."""
    calls = channel_configuration_calls(
        _CLIENT_MODULE, _client_module_source(), (_HTTP_CLIENT, _HTTP_WRAPPER)
    )

    assert calls == ("Axios.create", "configure")


def test_a_call_site_is_not_a_configuration_module() -> None:
    """The discriminator that makes this usable: `useAxios(...)` is a hook, not a channel.

    Twenty sibling call sites shared 218's channel. If every one of them read as
    configuration, the ordering A2 adds would be worth nothing and the real module would
    still lose the budget to whichever path sorted first.
    """
    calls = channel_configuration_calls(
        "src/pages/AllApps.js", _changed_page_source(), (_HTTP_WRAPPER,)
    )

    assert calls == ()


def test_the_configuration_module_comes_first_and_is_named() -> None:
    """The 218 shape: many co-importers, one of them the configured client.

    The sibling call sites all sort before `src/config/...`, which is what would have pushed
    the configuration module out of a bounded list. Rank, not path order, decides.
    """
    siblings = [f"src/apiUtils/{name}.apiUtils.js" for name in ("allapps", "apps", "users")]
    sources = {path: _changed_page_source() for path in siblings}
    sources[_CLIENT_MODULE] = _client_module_source()

    scan = channel_seam_scan(
        [("src/pages/AllApps.js", _changed_page_source())],
        [*sources, "src/pages/AllApps.js"],
        sources.get,
    )

    assert [item.path for item in scan.co_importers][0] == _CLIENT_MODULE
    assert scan.configuration_paths == (_CLIENT_MODULE,)
    assert scan.co_importers[0].configuration_calls == ("Axios.create", "configure")
    assert scan.configuration_unlocated == ()
    # Every sibling is still reported -- ranking orders the list, it filters nothing.
    assert sorted(item.path for item in scan.co_importers) == sorted([*siblings, _CLIENT_MODULE])


def test_a_checkout_with_no_configuration_module_records_the_omission() -> None:
    """The stop this item was told to prefer over a filename rule: say so, find nothing.

    No finding is manufactured -- the scan still reports the co-importers it did find, and
    the channel it could not locate a configuration for is named rather than left silent.
    """
    scan = channel_seam_scan(
        [("src/pages/AllApps.js", _changed_page_source())],
        ["src/apiUtils/allapps.apiUtils.js", "src/pages/AllApps.js"],
        lambda _path: _changed_page_source(),
    )

    assert [item.path for item in scan.co_importers] == ["src/apiUtils/allapps.apiUtils.js"]
    assert scan.configuration_paths == ()
    assert scan.configuration_unlocated == (_HTTP_WRAPPER,)


def test_the_change_configuring_the_channel_itself_records_no_omission() -> None:
    """A change that edits the configuration module needs nothing pointed out to it: the
    reviewer is already reading those lines in the diff."""
    scan = channel_seam_scan(
        [(_CLIENT_MODULE, _client_module_source())],
        [_CLIENT_MODULE],
        lambda _path: _client_module_source(),
    )

    assert scan.co_importers == ()
    assert scan.configuration_unlocated == ()


def test_a_utility_import_still_grows_nothing() -> None:
    """The module's own warning, kept as the regression guard: `lodash` is not a channel,
    so it must not read the checkout, name a configuration module, or omit one."""
    reads: list[str] = []

    def read(path: str) -> str | None:
        reads.append(path)
        return _client_module_source()

    scan = channel_seam_scan(
        [("src/pages/AllApps.js", "import merge from 'lodash/merge';\nexport const f = merge;\n")],
        [_CLIENT_MODULE],
        read,
    )

    assert reads == []
    assert scan.packages == ()
    assert scan.configuration_paths == ()
    assert scan.configuration_unlocated == ()


def test_a_direct_import_of_the_client_is_unchanged_from_today() -> None:
    """72-'s own shape, asserted here so A2 is additive: the co-importer set is what it was."""
    changed = f"import client from '{_NODE_PACKAGE}';\nexport const post = client.post;\n"
    configured = (
        f"import client from '{_NODE_PACKAGE}';\n"
        "export const configured = client.create({ baseURL: '/internal' });\n"
    )

    scan = channel_seam_scan(
        [("src/features/upload.js", changed)],
        ["src/api/http.js", "src/features/upload.js"],
        lambda _path: configured,
    )

    assert scan.packages == (_NODE_PACKAGE,)
    assert [item.path for item in scan.co_importers] == ["src/api/http.js"]
    assert scan.configuration_paths == ("src/api/http.js",)


def test_both_tables_drive_the_same_configuration_detection(tmp_path: Path) -> None:
    """One code path over the registry, exactly as the package matching is: the Python
    constructor shape (`psycopg2.connect(`, `httpx.Client(`) is rows, not a branch."""
    python_module = "app/db.py"
    python_source = (
        f"import {_PYTHON_PACKAGE}\n\nPOOL = {_PYTHON_PACKAGE}.connect(dsn=DSN, timeout=5)\n"
    )
    (tmp_path / "app").mkdir()
    (tmp_path / python_module).write_text(python_source, encoding="utf-8")

    scan = channel_seam_scan(
        [("app/handlers.py", f"import {_PYTHON_PACKAGE}\n\ndef run():\n    return 1\n")],
        [python_module],
        lambda path: WorkspaceFileTools(tmp_path).read_file(path),
    )

    assert scan.configuration_paths == (python_module,)
    assert channel_configuration_calls(python_module, python_source, (_PYTHON_PACKAGE,)) == (
        f"{_PYTHON_PACKAGE}.connect",
    )


def test_a_channel_reached_through_the_repositorys_own_wrapper_is_still_a_channel(
    tmp_path: Path,
) -> None:
    """One hop, and it is the shape that is more common than 218's: the change imports the
    repository's own module, and *that* module is what imports the client."""
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "api" / "client.js").write_text(
        f"import client from '{_NODE_PACKAGE}';\n"
        "export const configured = client.create({ baseURL: '/internal' });\n",
        encoding="utf-8",
    )
    tools = WorkspaceFileTools(tmp_path)

    scan = channel_seam_scan(
        [("src/pages/Upload.js", "import { configured } from '../api/client';\n")],
        ["src/api/client.js", "src/pages/Upload.js"],
        lambda path: tools.read_file(path),
    )

    assert scan.packages == (_NODE_PACKAGE,)
    assert scan.configuration_paths == ("src/api/client.js",)


# --------------------------------------------------------------------------------------
# The 218 replay, end to end through the review
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ab_feature_218_the_review_reads_the_client_the_call_goes_through(
    tmp_path: Path,
) -> None:
    """The whole item in one test, on 218's own files.

    The change is `src/pages/AllApps.js` importing `axios-hooks` with `url: '/api/apps/bulk'`.
    Twenty sibling modules share the channel, all of them sorting ahead of `src/config`. The
    two lines that decide whether the feature works at all -- the `baseURL` that already ends
    in `/api`, and the interceptor that forces a JSON content type over a multipart body --
    have to be in front of the reviewer, quoted, under the unchanged-source banner. 218's
    review recorded `channel_seam_packages: []` and approved with zero findings.
    """
    (tmp_path / "src" / "config").mkdir(parents=True)
    (tmp_path / "src" / "apiUtils").mkdir(parents=True)
    (tmp_path / _CLIENT_MODULE).write_text(_client_module_source(), encoding="utf-8")
    # The sibling call sites: every one of them sorts before `src/config/axios.js`.
    for index in range(20):
        (tmp_path / "src" / "apiUtils" / f"a{index:02d}.apiUtils.js").write_text(
            _changed_page_source(), encoding="utf-8"
        )
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/pages/AllApps.js": _changed_page_source()}
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await _reviewer(client).run(
        agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion])
    )

    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    seam = next(item for item in evidence["files"] if item["path"] == _CLIENT_MODULE)
    assert seam["content_kind"] == "channel_seam_context"
    assert seam["content"].startswith(
        f"===== {_CLIENT_MODULE}: unchanged from the revision this change branched from ====="
    )
    # The two lines that matter. Neither is inferable from the diff, and 218 shipped because
    # neither was ever read.
    assert _BASE_URL_LINE in seam["content"]
    assert _INTERCEPTOR_LINE in seam["content"]
    # Named as the configuration, not merely present as one co-importer among twenty-one.
    assert evidence["channel_seam"]["configuration_files"] == [_CLIENT_MODULE]
    assert evidence["channel_seam"]["configuration_not_located"] == []
    assert _HTTP_WRAPPER in evidence["channel_seam"]["packages"]

    review = update["artifacts"][0]
    assert review.metadata["channel_seam_configuration_paths"] == [_CLIENT_MODULE]
    # The seam module is context, not a reviewed change (51-A), and the evidence added here
    # decides nothing on its own: the verdict is the reviewer's, unchanged.
    assert review.metadata["reviewed_file_paths"] == ["src/pages/AllApps.js"]
    assert review.verdict == "approved"
