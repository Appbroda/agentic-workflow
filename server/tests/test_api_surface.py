"""Extraction of a repository's own registered routes from its real source files."""

from __future__ import annotations

from pathlib import Path

from tools.api_surface import RegisteredRoute, extract_registered_routes

_BACKEND_INDEX_ROUTE = """
import serviceRequestRoutes from './serviceRequest.route';
router.use('{prefix}', serviceRequestRoutes);
"""

_BACKEND_SERVICE_REQUEST_ROUTE = """
router
  .route('{route_path}')
  .post(validate(schema), asyncHandler(controller.exportActiveAdUnits));
"""


def _write(tmp_path: Path, relative_path: str, content: str) -> None:
    file_path = tmp_path / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def test_a_mounted_route_is_reported_both_bare_and_under_its_mount_prefix(
    tmp_path: Path,
) -> None:
    """A chained `.route().post()` registration resolves under every observed mount."""
    _write(
        tmp_path,
        "server/routes/index.route.js",
        _BACKEND_INDEX_ROUTE.format(prefix="/service-requests"),
    )
    _write(
        tmp_path,
        "server/routes/serviceRequest.route.js",
        _BACKEND_SERVICE_REQUEST_ROUTE.format(route_path="/export-active-ad-units"),
    )

    routes = extract_registered_routes(
        tmp_path,
        ["server/routes/index.route.js", "server/routes/serviceRequest.route.js"],
    )

    assert (
        RegisteredRoute(
            method="POST",
            path="/export-active-ad-units",
            source_path="server/routes/serviceRequest.route.js",
        )
        in routes
    )
    assert (
        RegisteredRoute(
            method="POST",
            path="/service-requests/export-active-ad-units",
            source_path="server/routes/serviceRequest.route.js",
        )
        in routes
    )


def test_a_routes_mounted_form_is_ordered_ahead_of_the_next_routes_bare_one(
    tmp_path: Path,
) -> None:
    """A cap applied to this list must drop bare duplicates before a complete, mounted path.

    AB-Feature-204 and -199 both hit `_MAX_REGISTERED_ROUTES` in `repository_reconnaissance.py`
    on a repository with more bare routes than the cap, which used to mean every mounted path
    was in a second block the slice never reached -- the contract was then grounded on nothing
    but bare fragments. Interleaved per route, mounted first, a truncating cap now loses routes
    from the tail rather than losing every complete path regardless of how early its route was
    scanned.
    """
    _write(
        tmp_path,
        "server/routes/index.route.js",
        _BACKEND_INDEX_ROUTE.format(prefix="/service-requests"),
    )
    _write(
        tmp_path,
        "server/routes/serviceRequest.route.js",
        _BACKEND_SERVICE_REQUEST_ROUTE.format(route_path="/export-active-ad-units")
        + "\nrouter.route('/floor-price').post(asyncHandler(controller.setFloorPrice));\n",
    )

    routes = extract_registered_routes(
        tmp_path,
        ["server/routes/index.route.js", "server/routes/serviceRequest.route.js"],
    )

    # The first route's mounted (complete) form and its bare form both come before the second
    # route's bare form -- a cap of 3 keeps the first route whole rather than keeping both
    # routes' bare fragments and neither route's complete path.
    assert routes[:3] == [
        RegisteredRoute(
            method="POST",
            path="/service-requests/export-active-ad-units",
            source_path="server/routes/serviceRequest.route.js",
        ),
        RegisteredRoute(
            method="POST",
            path="/export-active-ad-units",
            source_path="server/routes/serviceRequest.route.js",
        ),
        RegisteredRoute(
            method="POST",
            path="/service-requests/floor-price",
            source_path="server/routes/serviceRequest.route.js",
        ),
    ]


def test_a_router_with_no_use_mount_reports_only_the_bare_path(tmp_path: Path) -> None:
    """No observed `.use(...)` mount means no prefix to guess at -- fails open, not closed."""
    _write(
        tmp_path,
        "server/routes/serviceRequest.route.js",
        _BACKEND_SERVICE_REQUEST_ROUTE.format(route_path="/export-active-ad-units"),
    )

    routes = extract_registered_routes(tmp_path, ["server/routes/serviceRequest.route.js"])

    assert routes == [
        RegisteredRoute(
            method="POST",
            path="/export-active-ad-units",
            source_path="server/routes/serviceRequest.route.js",
        )
    ]


_FASTAPI_ROUTES = """
from fastapi import APIRouter
router = APIRouter()

@router.get("/export")
async def export_ad_units():
    ...
"""

_FASTAPI_MAIN = """
from fastapi import FastAPI
from app.routes import router

app = FastAPI()
app.include_router(
    router,
    prefix="/api/ad-units",
    tags=["ad-units"],
)
"""

_FLASK_VIEWS = """
from flask import Blueprint
bp = Blueprint("ad_units", __name__)

@bp.route("/export", methods=["POST"])
def export_ad_units():
    ...

@bp.get("/status")
def status():
    ...
"""

_FLASK_MAIN = """
from flask import Flask
from app.views import bp

app = Flask(__name__)
app.register_blueprint(bp, url_prefix="/api/ad-units")
"""


def test_a_fastapi_route_is_reported_both_bare_and_under_its_include_router_prefix(
    tmp_path: Path,
) -> None:
    """A `@router.get(...)` decorator resolves under a multi-line `include_router(..., prefix=)`."""
    _write(tmp_path, "app/routes.py", _FASTAPI_ROUTES)
    _write(tmp_path, "app/main.py", _FASTAPI_MAIN)

    routes = extract_registered_routes(tmp_path, ["app/routes.py", "app/main.py"])

    assert RegisteredRoute(method="GET", path="/export", source_path="app/routes.py") in routes
    assert (
        RegisteredRoute(method="GET", path="/api/ad-units/export", source_path="app/routes.py")
        in routes
    )


def test_a_flask_route_is_reported_both_bare_and_under_its_blueprint_prefix(
    tmp_path: Path,
) -> None:
    """Flask's `methods=[...]` kwarg and its modern `.get(...)` shorthand both resolve, once."""
    _write(tmp_path, "app/views.py", _FLASK_VIEWS)
    _write(tmp_path, "app/main.py", _FLASK_MAIN)

    routes = extract_registered_routes(tmp_path, ["app/views.py", "app/main.py"])

    assert (
        routes.count(RegisteredRoute(method="POST", path="/export", source_path="app/views.py"))
        == 1
    )
    assert (
        routes.count(RegisteredRoute(method="GET", path="/status", source_path="app/views.py")) == 1
    )
    assert (
        RegisteredRoute(method="POST", path="/api/ad-units/export", source_path="app/views.py")
        in routes
    )
    assert (
        RegisteredRoute(method="GET", path="/api/ad-units/status", source_path="app/views.py")
        in routes
    )


def test_a_python_file_matching_neither_framework_reports_nothing(tmp_path: Path) -> None:
    """Trying both Python frameworks against every `.py` file manufactures no false positives."""
    _write(tmp_path, "app/util.py", "def add(a, b):\n    return a + b\n")

    assert extract_registered_routes(tmp_path, ["app/util.py"]) == []
