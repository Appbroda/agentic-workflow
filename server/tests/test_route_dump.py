"""Execution-grounded route evidence: requiring a real route module, never guessing from text."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from services.cancellation import MockCancellationToken
from services.process_runner import AsyncioProcessRunner
from tools.api_surface import RegisteredRoute
from tools.route_dump import dump_route_module

# Shaped exactly like what Express itself builds for a `Router()` -- a `.stack` of layers,
# each either a registered route (`.route.path` / `.route.methods`) or something else (plain
# middleware, a nested router) that this walk must skip rather than misread as a route. Built by
# hand rather than requiring the real `express` package: this tests the dump script's own
# parsing of that shape, not `express`'s behaviour, and runs with no dependency installed.
_FAKE_ROUTER_MODULE = """
module.exports = {
  stack: [
    { route: { path: '/export', methods: { post: true } } },
    { route: { path: '/deactivate', methods: { patch: true, get: true } } },
    { handle: function middleware() {} },
  ],
};
"""

_NAMED_EXPORT_MODULE = """
module.exports = {
  router: {
    stack: [
      { route: { path: '/status', methods: { get: true } } },
    ],
  },
};
"""

_THROWS_ON_REQUIRE = """
throw new Error('deliberately broken for the test');
"""

_NOT_ROUTER_SHAPED = """
module.exports = { hello: 'world' };
"""


def _write(tmp_path: Path, relative_path: str, content: str) -> None:
    file_path = tmp_path / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not available")


async def test_a_route_modules_real_stack_is_read_not_guessed(tmp_path: Path) -> None:
    """Methods and paths come from the framework's own object, including a multi-method route."""
    _write(tmp_path, "server/routes/serviceRequest.route.js", _FAKE_ROUTER_MODULE)

    result = await dump_route_module(
        tmp_path,
        "server/routes/serviceRequest.route.js",
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert result.ok
    assert result.error is None
    assert set(result.routes) == {
        RegisteredRoute(
            method="POST", path="/export", source_path="server/routes/serviceRequest.route.js"
        ),
        RegisteredRoute(
            method="PATCH",
            path="/deactivate",
            source_path="server/routes/serviceRequest.route.js",
        ),
        RegisteredRoute(
            method="GET", path="/deactivate", source_path="server/routes/serviceRequest.route.js"
        ),
    }


async def test_a_named_export_router_is_still_found(tmp_path: Path) -> None:
    """`module.exports = { router }` is tried when the direct export is not router-shaped."""
    _write(tmp_path, "server/routes/status.route.js", _NAMED_EXPORT_MODULE)

    result = await dump_route_module(
        tmp_path,
        "server/routes/status.route.js",
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert result.ok
    assert result.routes == (
        RegisteredRoute(method="GET", path="/status", source_path="server/routes/status.route.js"),
    )


async def test_a_missing_file_fails_open_with_no_exception(tmp_path: Path) -> None:
    result = await dump_route_module(
        tmp_path,
        "server/routes/does_not_exist.route.js",
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert not result.ok
    assert result.routes == ()
    assert "does not exist" in (result.error or "")


async def test_a_module_that_throws_on_require_fails_open(tmp_path: Path) -> None:
    _write(tmp_path, "server/routes/broken.route.js", _THROWS_ON_REQUIRE)

    result = await dump_route_module(
        tmp_path,
        "server/routes/broken.route.js",
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert not result.ok
    assert "deliberately broken" in (result.error or "")


async def test_a_module_with_no_router_shaped_export_fails_open(tmp_path: Path) -> None:
    _write(tmp_path, "server/routes/plain.route.js", _NOT_ROUTER_SHAPED)

    result = await dump_route_module(
        tmp_path,
        "server/routes/plain.route.js",
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert not result.ok
    assert "does not export an Express Router" in (result.error or "")
