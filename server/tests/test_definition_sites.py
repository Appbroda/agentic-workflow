"""The definition-site lookup itself: what it will follow, and what it refuses to.

The end-to-end proof that this reaches a repair prompt and a remediation snapshot is in the
`realrepo` tier (`test_real_repository_gates.py`, A1 to A4 and F1). This file covers the
lookup as a function, in the **default** tier, because the extraction rule and the definition
patterns are the whole of how a name is resolved -- a change to either would otherwise pass
every test anybody runs by habit, and the failure it would cause is a wrong import specifier
that costs a whole attempt.

The asymmetry that shapes these propositions: resolving nothing costs nothing, and resolving
the *wrong* file costs a repair pass spent writing against a shape that was never there. So
the extraction is deliberately narrow and every case below that ends in "resolves to nothing"
is a decision rather than a limitation nobody noticed.
"""

from __future__ import annotations

import logging

import pytest

from tools.definition_sites import (
    DefinitionIndex,
    definition_line,
    definition_sites,
    definition_symbols,
)

_PERMISSIONS = (
    "function canPublish(user) {\n"
    "  return Boolean(user && user.role);\n"
    "}\n"
    "\n"
    "const permissionUtils = { canPublish };\n"
    "\n"
    "module.exports = { permissionUtils };\n"
)
_SETTINGS = (
    '"""Service settings."""\n'
    "\n"
    "RETRY_CEILING = 4\n"
    "\n"
    "\n"
    "def build_client(timeout: float) -> None:\n"
    '    """Build the configured client."""\n'
    "    return None\n"
)
_CHECKOUT = {
    "src/support/permissions.js": _PERMISSIONS,
    "src/routes/publish.js": (
        "function publishRoute(request, response) {\n"
        "  response.json({ allowed: permissionUtils.canPublish(request.user) });\n"
        "}\n"
        "\n"
        "module.exports = { publishRoute };\n"
    ),
    "app/settings.py": _SETTINGS,
    "package.json": '{"name": "service"}\n',
    "docs/permissions.md": "`permissionUtils` is documented here.\n",
    "test/permissions.test.js": (
        "const permissionUtils = require('../src/support/permissions');\n"
        "module.exports = { permissionUtils };\n"
    ),
}


def _index(files: dict[str, str] | None = None) -> DefinitionIndex:
    """Bind the lookup to a checkout described as a dictionary."""
    checkout = _CHECKOUT if files is None else files
    return DefinitionIndex(paths=tuple(checkout), read=checkout.get)


# --------------------------------------------------------------------------------------
# What a diagnostic is read as naming
# --------------------------------------------------------------------------------------


def test_a_quoted_name_in_a_lint_diagnostic_is_the_name_to_resolve() -> None:
    """The case this exists for, written the way every linter writes it.

    `'permissionUtils' is not defined` is the commonest mechanical diagnostic there is, and
    the thirty-line window the repair prompt already quotes around the line it names contains
    the *use*. Nothing in that window says which module exports the name.
    """
    assert definition_symbols(
        ["src/routes/publish.js:2  error  'permissionUtils' is not defined  no-undef"]
    ) == ("permissionUtils",)


def test_prose_about_behaviour_names_nothing_and_that_is_the_documented_limit() -> None:
    """A finding that names no symbol and no path is not something this can rescue.

    Recorded as a proposition rather than left implicit, because it is the boundary of what
    Part A covers: closing it needs either the model asking for files or the reviewer being
    required to cite a path, and neither is in this design. A lookup that followed every word
    of a sentence would resolve whichever file happens to declare a common one, which is worse
    than resolving nothing.
    """
    assert (
        definition_symbols(
            [
                "This duplicates the existing retry helper.",
                "The empty selection case is unhandled and the list still renders.",
            ]
        )
        == ()
    )


def test_a_dotted_filename_a_dependency_api_and_a_sentence_all_resolve_to_nothing() -> None:
    """A2. Three shapes that must contribute nothing, asserted end to end.

    A filename is already resolved by the location reader, which knows the line as well, so
    reading its stem as a symbol would only duplicate that at lower precision. A dependency's
    API is not this repository's to show. And an ordinary sentence names no identifier at all.
    """
    index = _index()
    diagnostics = [
        "helpers.spec.js fails at line 4 and package.json declares no runner for it.",
        "express.Router() is not mounted before the error middleware.",
        "The listing endpoint returns no cursor for an empty page.",
    ]
    assert index.sites(definition_symbols(diagnostics), limit=3) == ()


def test_a_path_and_a_lint_rule_id_are_never_read_as_names() -> None:
    """Both carry a separator, and neither is a name this checkout could define."""
    assert (
        definition_symbols(
            [
                "src/pages/AdminHome.js:12  error  react-hooks/exhaustive-deps",
                "https://example.test/docs/adminHome",
            ]
        )
        == ()
    )


# --------------------------------------------------------------------------------------
# Where the checkout says a name is defined
# --------------------------------------------------------------------------------------


def test_a_name_exported_from_a_file_named_nothing_like_it_still_resolves() -> None:
    """The property that makes this worth having: the filename is not the symbol.

    `permissionUtils` lives in `permissions.js`, so nothing that reasons from filenames finds
    it, and nothing that follows imports finds it either -- the attempt that needs it is the
    one that failed to write the import.
    """
    sites = _index().sites(("permissionUtils",), limit=3)

    assert [site.path for site in sites] == ["src/support/permissions.js"]
    assert _PERMISSIONS.splitlines()[sites[0].line - 1].startswith("const permissionUtils")


def test_a_python_module_level_binding_is_a_definition_here_and_not_an_export() -> None:
    """The one form this lookup accepts that the shared export parser deliberately will not.

    A module-level constant is exactly what an attempt forgets to import. The reachability
    inspection must not treat one as an exported name -- every extra name it accepts makes its
    gate quieter about genuinely dead code -- so the two parsers differ here on purpose.
    """
    sites = _index().sites(("RETRY_CEILING",), limit=3)

    assert [(site.path, site.line) for site in sites] == [("app/settings.py", 3)]


def test_only_production_source_answers_where_a_change_should_import_from() -> None:
    """A test fixture and a document mentioning a name are not where to import it from.

    Both files here declare `permissionUtils` -- the test binds it, the document quotes it --
    and quoting either would send a repair to write an import against a file that exists to
    describe the module rather than to be it.
    """
    sites = _index().sites(("permissionUtils",), limit=5)

    assert [site.path for site in sites] == ["src/support/permissions.js"]


def test_a_local_declaration_that_is_never_exported_is_not_a_definition_site() -> None:
    """What a repair needs is somewhere it can import from, not merely somewhere it occurs."""
    files = {
        "src/a.js": "const sharedTotals = { count: 0 };\n\nmodule.exports = { other: 1 };\n",
        "src/b.js": "export const sharedTotals = { count: 0 };\n",
    }

    sites = definition_sites(("sharedTotals",), tuple(files), files.get, limit=3)

    assert [site.path for site in sites] == ["src/b.js"]


def test_an_ambiguous_name_offers_every_candidate_within_the_bound() -> None:
    """Several quoted candidates beat a blind guess at which module the name came from."""
    files = {
        "src/admin/formatLabel.js": "export function formatLabel(value) { return value; }\n",
        "src/public/formatLabel.js": "export function formatLabel(value) { return value; }\n",
    }

    sites = definition_sites(("formatLabel",), tuple(files), files.get, limit=3)

    assert [site.path for site in sites] == [
        "src/admin/formatLabel.js",
        "src/public/formatLabel.js",
    ]


def test_every_name_gets_a_site_before_any_name_gets_a_second() -> None:
    """The bound is shared, so an ambiguous name must not spend all of it.

    A repair facing two unresolved names and shown three candidates for the first has learned
    nothing about the second, which is the one it is equally likely to have got wrong.
    """
    files = {
        "src/one/alphaHelper.js": "export const alphaHelper = 1;\n",
        "src/two/alphaHelper.js": "export const alphaHelper = 2;\n",
        "src/three/betaHelper.js": "export const betaHelper = 3;\n",
    }

    sites = definition_sites(("alphaHelper", "betaHelper"), tuple(files), files.get, limit=2)

    assert [site.symbol for site in sites] == ["alphaHelper", "betaHelper"]


def test_a_file_that_cannot_be_read_contributes_nothing_rather_than_raising() -> None:
    """Best-effort, like every other read on this path: the attempt proceeds regardless."""
    files: dict[str, str | None] = {
        "src/binary.js": None,
        "src/real.js": "export const widgetTotals = 1;\n",
    }

    sites = definition_sites(("widgetTotals",), tuple(files), files.get, limit=3)

    assert [site.path for site in sites] == ["src/real.js"]


def test_a_name_defined_nowhere_costs_nothing_and_quotes_nothing() -> None:
    """Resolution is the filter, which is what lets extraction stay generous with quotes."""
    assert _index().sites(("lodashDebounce",), limit=3) == ()


def test_the_index_can_be_widened_by_files_the_attempt_itself_just_wrote() -> None:
    """The repair loop needs it: the checkout listing predates the coding call.

    A name defined in a module this very attempt added is exactly as unresolvable to the pass
    that forgot to import it as one that was already committed.
    """
    written = {"src/new/reportTotals.js": "export const reportTotals = () => 0;\n"}
    index = DefinitionIndex(paths=(), read=written.get).including(written)

    assert [site.path for site in index.sites(("reportTotals",), limit=3)] == [
        "src/new/reportTotals.js"
    ]


# --------------------------------------------------------------------------------------
# What is never read, and what is never silent
# --------------------------------------------------------------------------------------


def test_a_credential_shaped_path_is_never_read_by_this_lookup() -> None:
    """The path half of the one key-material rule, applied before anything is opened.

    A file the path rule rejects is not scanned, so it can neither be quoted to a repair nor
    named to the context selector. The *content* half is deliberately not asked here -- the
    two callers apply different policies to it, and one of them has to report the withholding
    rather than fall silent.
    """
    opened: list[str] = []

    def read(path: str) -> str | None:
        opened.append(path)
        return "export const mailerCredential = 'x';\n"

    sites = definition_sites(
        ("mailerCredential",),
        ("mailer.env", "secrets.yaml", "node_modules/pkg/index.js", "src/mailer.js"),
        read,
        limit=3,
    )

    assert opened == ["src/mailer.js"]
    assert [site.path for site in sites] == ["src/mailer.js"]


def test_the_scan_bound_is_logged_rather_than_silently_applied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cap nobody can see reads exactly like "this checkout defines none of these names"."""
    files = {f"src/module{index:05d}.js": "const unrelated = 1;\n" for index in range(2_100)}

    with caplog.at_level(logging.WARNING):
        assert definition_sites(("widgetTotals",), tuple(files), files.get, limit=3) == ()

    assert "definition_scan_truncated" in caplog.text
    assert "widgetTotals" in caplog.text


def test_definition_line_reports_the_line_the_declaration_is_actually_on() -> None:
    """Off by one here quotes a window centred one line away, which is a silent regression."""
    assert definition_line(_PERMISSIONS, "permissionUtils") == 5
    assert definition_line(_SETTINGS, "build_client") == 6
    assert definition_line(_PERMISSIONS, "neverDeclared") is None
