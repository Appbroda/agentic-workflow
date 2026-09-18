"""Generic repository checkouts whose declared commands actually run.

Audit risk P1-8, third tier. `tests/fixtures/__init__.py` already builds real git checkouts,
but nothing ever *executes* what they declare: every test that drives them substitutes a
process runner that returns exit zero without invoking anything. Command selection was
therefore proven and command behaviour was not, so a plan that named a script the repository
could not run looked identical to one that named a script it could.

These fixtures close that gap, under three constraints:

* **Nothing installs.** A repository whose validation needs a package registry is a fixture
  that costs ninety seconds and gets deleted by the first person in a hurry. Every script
  here is a real program run by the runtime the checkout declares -- Node's own `node`,
  Python's `ruff` and `pytest` -- and none of them imports a third-party package. `npm ci`
  still runs, for real, against a lockfile with nothing in it; it takes about half a second.
* **Nothing names a real repository.** Per overview section 4.3 the platform detects
  everything from the checkout, so a fixture modelled on one target's layout would prove
  detection against the very assumption it exists to refute. The names here are generic and
  the shapes are ordinary.
* **The linter is the repository's own.** `scripts/lint.js` is a small but genuine linter:
  it loads a checked-in configuration, resolves the shared configs that configuration
  extends, and reports rule violations in the file:line:rule form every mainstream linter
  uses. That is what lets one fixture family drive both the commit gate (a real violation)
  and the unrepairable-configuration gate (a shared config the manifest never declares).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from tests.fixtures import commit_all, init_git_repository

# A shared constants module far longer than the reviewer's per-file evidence budget, whose
# changes are always a handful of lines. Run 186's frontend lost its delivery to exactly this
# shape: 31,850 bytes over 918 lines, four added lines, committed by an earlier approved
# attempt and therefore absent from `git diff HEAD` -- head-truncated, and refused three times
# for the truncation. Generic, as every fixture here is: a constants module is the most
# ordinary place for a file to be long and a change to it to be small.
LARGE_CONSTANTS_MODULE = "src/support/constants.js"
# Enough entries to put the module well past the 24,000-character per-file budget with room to
# spare, so raising that budget cannot quietly stop this fixture exercising the oversized path.
_LARGE_CONSTANTS_ENTRY_COUNT = 900

# Run 189's shape, generically. `package-lock.json` in that workspace was 1,290,320 bytes --
# past the reviewer's 1 MiB read cap -- and the whole workstream terminated because of it.
# Enough package entries here to land in the same order of magnitude, so a raised read cap
# cannot quietly stop this fixture exercising the path it exists for.
OVERSIZED_LOCKFILE = "package-lock.json"
OVERSIZED_LOCKFILE_MANIFEST = "package.json"
_OVERSIZED_LOCKFILE_PACKAGE_COUNT = 4_200

# Set this in CI. With it, a missing `node`, `npm` or `pnpm` fails the run instead of
# skipping it, because a tier that certifies nothing by skipping is worse than no tier.
REQUIRE_ENVIRONMENT_VARIABLE = "REQUIRE_REAL_REPOSITORY_TESTS"

# What `scripts/lint.js` is asked to extend in the broken-configuration fixture. Deliberately
# a name no registry has: the point is a reference the checkout never declares, not a package
# that happens to be absent from this machine.
UNDECLARED_SHARED_CONFIG = "shared-style-guide"

# The module whose definition the cross-module fixture withholds, and the symbol a
# diagnostic names on it. Deliberately unrelated to anything a requirement about listing
# endpoints would say, so nothing can reach it by matching the task's wording.
CROSS_MODULE_DEFINITION = "src/support/envelope.js"
CROSS_MODULE_SYMBOL = "envelope"
# Enough subject-matching modules to overrun the engineer's file budget, so the definition
# has to be selected rather than merely fit.
_CROSS_MODULE_FILLER_COUNT = 90

# The module the missing-import fixture withholds an import of, and the name it exports.
# Deliberately spelled unlike its own filename, because a symbol that happens to match its
# file's stem can be resolved by machinery that has nothing to do with definition sites.
MISSING_IMPORT_DEFINITION = "src/support/permissions.js"
MISSING_IMPORT_SYMBOL = "permissionUtils"
_PERMISSIONS_MODULE = (
    "function canPublish(user) {\n"
    "  return Boolean(user) && user.role === 'editor';\n"
    "}\n"
    "\n"
    "const permissionUtils = { canPublish };\n"
    "\n"
    "module.exports = { permissionUtils };\n"
)

# Where the typechecked fixture declares what each module exports, and what the committed
# checkout declares. A change that adds a module declares its shape here too, which is what
# makes a mismatch between a value and its own declaration expressible.
DECLARED_TYPES = "types.json"
_COMMITTED_TYPES: dict[str, object] = {
    "src/routes/status.js": {"statusRoute": "function"},
    "src/routes/catalog.js": {
        "CATEGORIES": "object",
        "buildEntry": "function",
        "catalogRoute": "function",
        "normalizeCategory": "function",
        "paginate": "function",
    },
}

# The source file that both defines a symbol and holds a private key, so the two halves of
# the key-material rule are exercised on one file: its path is unremarkable and its bytes
# are not.
KEY_BEARING_DEFINITION = "src/support/signingKeys.js"
KEY_BEARING_SYMBOL = "signingCredentials"

# The key-material fixture's inhabitants. Every filename here would pass a path rule: the
# environment file ends in `.env` rather than beginning with it, and the other two are named
# so that nothing but their content gives them away.
CREDENTIAL_ENVIRONMENT_FILE = "mailer.env"
CREDENTIAL_ENVIRONMENT_TEMPLATE = ".env.example"
CREDENTIAL_SERVICE_ACCOUNT = "analytics-1122334455667-a1b2c3d4e5f6.json"
CREDENTIAL_PEM_FILE = "release-signing.txt"
CREDENTIAL_DOCUMENTATION = "ROTATION.md"
# One string, present in every credential file and nowhere else, so a test can ask the
# delivered prompt whether any key material reached it without enumerating file shapes.
CREDENTIAL_SECRET = "Zm9yYmlkZGVuLWtleS1tYXRlcmlhbA"

# A real 1x1 PNG, decoded rather than described, because the point is bytes no decoder will
# accept as text. An image is the ordinary case -- a feature adding a logo, an icon or a font
# -- and the commit gate briefly refused every one of them for not being UTF-8, so a test
# with a plausible-looking but decodable stand-in would prove nothing.
BINARY_ASSET = "public/logo.png"
BINARY_ASSET_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="
)

__all__ = [
    "BINARY_ASSET",
    "BINARY_ASSET_BYTES",
    "CREDENTIAL_DOCUMENTATION",
    "CREDENTIAL_ENVIRONMENT_FILE",
    "CREDENTIAL_ENVIRONMENT_TEMPLATE",
    "CREDENTIAL_PEM_FILE",
    "CREDENTIAL_SECRET",
    "CREDENTIAL_SERVICE_ACCOUNT",
    "CROSS_MODULE_DEFINITION",
    "CROSS_MODULE_SYMBOL",
    "DECLARED_TYPES",
    "KEY_BEARING_DEFINITION",
    "KEY_BEARING_SYMBOL",
    "LARGE_CONSTANTS_MODULE",
    "MISSING_IMPORT_DEFINITION",
    "MISSING_IMPORT_SYMBOL",
    "OVERSIZED_LOCKFILE",
    "OVERSIZED_LOCKFILE_MANIFEST",
    "REQUIRE_ENVIRONMENT_VARIABLE",
    "UNDECLARED_SHARED_CONFIG",
    "broken_lint_config_repository",
    "credential_bearing_repository",
    "cross_module_dependency_repository",
    "key_bearing_module_repository",
    "large_constants_module_source",
    "large_constants_repository",
    "missing_executables",
    "missing_import_repository",
    "no_tests_repository",
    "node_npm_repository",
    "node_pnpm_repository",
    "oversized_lockfile_repository",
    "oversized_lockfile_source",
    "python_uv_repository",
    "requires_real_repository_tier",
    "slow_suite_repository",
    "two_package_monorepo_repository",
    "typechecked_repository",
    "unavailable_reason",
    "unnarrowable_test_repository",
]


# --------------------------------------------------------------------------------------
# Tool availability
# --------------------------------------------------------------------------------------


def missing_executables(*names: str) -> list[str]:
    """Return which of these executables this machine cannot run."""
    return [name for name in names if shutil.which(name) is None]


def requires_real_repository_tier() -> bool:
    """Report whether a missing toolchain should fail the run rather than skip it."""
    return os.environ.get(REQUIRE_ENVIRONMENT_VARIABLE, "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }


def unavailable_reason(missing: list[str]) -> str:
    """Explain a skip so a reader knows what to install rather than guessing."""
    return (
        f"the real-repository tier needs {', '.join(missing)} on PATH. CI sets "
        f"{REQUIRE_ENVIRONMENT_VARIABLE}=1, which turns this skip into a failure."
    )


# --------------------------------------------------------------------------------------
# The programs the fixtures declare
# --------------------------------------------------------------------------------------


# A real linter: it loads its own configuration, resolves what that configuration extends,
# and then applies two rules to the source roots the configuration names. Written in plain
# CommonJS against Node's standard library so a checkout can be linted without installing
# anything -- which is the only way a fixture stays cheap enough to keep.
_LINT_SCRIPT = """\
const { readFileSync, readdirSync, statSync } = require('node:fs');
const { join, resolve } = require('node:path');

const config = JSON.parse(readFileSync('lintrc.json', 'utf8'));

for (const shared of config.extends || []) {
  try {
    require.resolve(shared);
  } catch {
    process.stderr.write(
      'Failed to load config "' + shared + '" to extend from.\\n' +
      "Cannot find module '" + shared + "'\\n"
    );
    process.exit(2);
  }
}

function sources(directory, found) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) sources(path, found);
    else if (path.endsWith('.js')) found.push(path);
  }
  return found;
}

const problems = [];
for (const root of config.sources) {
  for (const file of sources(root, [])) {
    const text = readFileSync(file, 'utf8');
    text.split('\\n').forEach((line, index) => {
      if (/(^|[^\\w.])var\\s/.test(line)) {
        problems.push(file + ':' + (index + 1) + '  error  Unexpected var, use let or const \
instead  no-var');
      }
      const snake = line.match(/(?:^|[^\\w.])(?:const|let)\\s+([a-z0-9]+_[a-z0-9_]*)\\s*=/);
      if (snake) {
        problems.push(file + ':' + (index + 1) + "  error  Identifier '" + snake[1] + \
"' is not in camel case, rename it  camelcase");
      }
      const uninitialized = line.match(/(?:^|[^\\w.])const\\s+[A-Za-z_$][\\w$]*\\s*;/);
      if (uninitialized) {
        problems.push(resolve(file) + '\\n  ' + (index + 1) + ':' + (line.indexOf(';') + 1) + \
'  error  Parsing error: Unexpected token ;');
      }
    });
    if (text.length > 0 && !text.endsWith('\\n')) {
      problems.push(file + ':1  error  Newline required at end of file  eol-last');
    }
  }
}

if (problems.length > 0) {
  process.stderr.write(problems.join('\\n') + '\\n\\n' + problems.length + ' problems\\n');
  process.exit(1);
}
"""

# A second real linter, for the one diagnostic class the base one cannot produce: a name used
# and never bound. That is the commonest mechanical failure there is -- a missing import --
# and it is the failure the definition-site lookup exists to make repairable, so a fixture
# that cannot produce it cannot test it.
#
# Two rules, and the second is what makes the first worth having. `no-undef` alone would be
# satisfied by *any* import of that name, including one whose path is invented, so a repair
# that guessed a specifier would look identical to one that read it. `import/no-unresolved`
# is what tells the two apart, and it is an ordinary rule in every mainstream JavaScript lint
# configuration.
#
# Written against Node's standard library like the base linter, and kept separate from it
# rather than folded in: the base linter's rules are load-bearing for every other fixture in
# this family, and a scope analysis added underneath them would put all of those at the mercy
# of this one's false positives.
_UNDEFINED_SYMBOL_LINT_SCRIPT = """\
const { existsSync, readFileSync, readdirSync, statSync } = require('node:fs');
const { dirname, join, resolve } = require('node:path');

const config = JSON.parse(readFileSync('lintrc.json', 'utf8'));

const BOUND_ELSEWHERE = new Set([
  'Array', 'Boolean', 'Date', 'Error', 'JSON', 'Map', 'Math', 'Number', 'Object', 'Promise',
  'RegExp', 'Set', 'String', 'Symbol', 'console', 'exports', 'globalThis', 'module',
  'process', 'require', 'setInterval', 'setTimeout',
  'await', 'catch', 'do', 'else', 'for', 'function', 'if', 'in', 'new', 'of', 'return',
  'switch', 'throw', 'try', 'typeof', 'while', 'yield',
]);

function sources(directory, found) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) sources(path, found);
    else if (path.endsWith('.js')) found.push(path);
  }
  return found;
}

function boundNames(text) {
  const bound = new Set(BOUND_ELSEWHERE);
  for (const match of text.matchAll(/(?:const|let|var|function|class)\\s+([A-Za-z_$][\\w$]*)/g)) {
    bound.add(match[1]);
  }
  for (const match of text.matchAll(/(?:const|let|var)\\s*\\{([^}]*)\\}/g)) {
    for (const entry of match[1].split(',')) {
      const name = entry.split(':').pop().trim();
      if (/^[A-Za-z_$][\\w$]*$/.test(name)) bound.add(name);
    }
  }
  for (const match of text.matchAll(/\\(([^()]*)\\)\\s*(?:=>|\\{)/g)) {
    for (const entry of match[1].split(',')) {
      const name = entry.trim();
      if (/^[A-Za-z_$][\\w$]*$/.test(name)) bound.add(name);
    }
  }
  return bound;
}

const problems = [];
for (const root of config.sources) {
  for (const file of sources(root, [])) {
    const text = readFileSync(file, 'utf8');
    const bound = boundNames(text);
    text.split('\\n').forEach((line, index) => {
      const required = line.match(/require\\(\\s*'(\\.[^']*)'\\s*\\)/);
      if (required) {
        const target = resolve(dirname(file), required[1]);
        const found = [target, target + '.js', join(target, 'index.js')].some(existsSync);
        if (!found) {
          problems.push(file + ':' + (index + 1) + "  error  Cannot find module '" + \
required[1] + "'  import/no-unresolved");
        }
      }
      const readable = line.replace(/'[^']*'/g, "''").replace(/"[^"]*"/g, '""');
      for (const use of readable.matchAll(/(?:^|[^\\w$.])([A-Za-z_$][\\w$]*)\\s*[.(]/g)) {
        if (!bound.has(use[1])) {
          problems.push(file + ':' + (index + 1) + "  error  '" + use[1] + \
"' is not defined  no-undef");
        }
      }
    });
    if (text.length > 0 && !text.endsWith('\\n')) {
      problems.push(file + ':1  error  Newline required at end of file  eol-last');
    }
  }
}

if (problems.length > 0) {
  process.stderr.write(problems.join('\\n') + '\\n\\n' + problems.length + ' problems\\n');
  process.exit(1);
}
"""

# A real typechecker, in the only form a registry-free fixture can have one. `tsc` would need
# an install, so instead this loads each module the checkout declares a shape for and compares
# the declaration against what the module actually exports. That is a genuine type check --
# it runs the code, it reads no source text, and it cannot be satisfied by anything except
# making the values agree with the declaration -- and it reports in the `error TS####` form
# and exit code that `tsc` itself uses, so the platform sees the output shape it will see in a
# real checkout.
_TYPECHECK_SCRIPT = """\
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');

const declared = JSON.parse(readFileSync('types.json', 'utf8'));

const problems = [];
for (const [modulePath, shape] of Object.entries(declared)) {
  let loaded;
  try {
    loaded = require(resolve(modulePath));
  } catch (error) {
    problems.push(modulePath + "(1,1): error TS2307: Cannot find module '" + modulePath + "'.");
    continue;
  }
  for (const [name, expected] of Object.entries(shape)) {
    const actual = loaded[name] === undefined ? 'undefined' : typeof loaded[name];
    if (actual !== expected) {
      problems.push(
        modulePath + "(1,1): error TS2322: Type '" + actual + "' is not assignable to type '" +
        expected + "' for exported member '" + name + "'."
      );
    }
  }
}

if (problems.length > 0) {
  process.stdout.write(problems.join('\\n') + '\\n\\nFound ' + problems.length + ' error(s).\\n');
  process.exit(2);
}
"""

# A real build: it writes an artifact, which is what makes "the build ran" observable in the
# worktree rather than only in a recorded argument vector.
_BUILD_SCRIPT = """\
const { mkdirSync, readdirSync, writeFileSync } = require('node:fs');
mkdirSync('dist', { recursive: true });
writeFileSync('dist/manifest.json', JSON.stringify(readdirSync('src').sort()) + '\\n');
"""

# A test command that cannot finish inside any budget a caller would give it. The sleep is
# open-ended on purpose: a fixture that merely takes a while races the machine it runs on.
_SLOW_TEST_SCRIPT = """\
setInterval(() => {}, 1000);
"""

# The first half of a composed test script. A real program, so the composed script the
# fixture declares is one that genuinely runs two things rather than one that looks composed.
_PRETEST_SCRIPT = """\
const { readdirSync } = require('node:fs');
process.stdout.write('checked ' + readdirSync('test').length + ' suite file(s)\\n');
"""

# An existing module with enough behaviour in it to be worth preserving. The rewrite gate
# only fires above a floor of deleted lines, and deliberately so -- a five-line file replaced
# is an edit, and a hundred-line one replaced is work being discarded. A fixture whose only
# source file is five lines therefore cannot reach that gate at all, which is why this exists.
_CATALOG_MODULE = """\
const CATEGORIES = ['hardware', 'software', 'services'];

function normalizeCategory(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const candidate = value.trim().toLowerCase();
  return CATEGORIES.includes(candidate) ? candidate : null;
}

function paginate(items, page, size) {
  const start = Math.max(0, (page - 1) * size);
  return items.slice(start, start + size);
}

function buildEntry(record) {
  return {
    id: record.id,
    name: record.name,
    category: normalizeCategory(record.category),
    available: Boolean(record.available),
  };
}

function catalogRoute(request, response) {
  const query = request.query || {};
  const category = normalizeCategory(query.category);
  const page = Number.parseInt(query.page, 10) || 1;
  const size = Number.parseInt(query.size, 10) || 20;
  const records = request.records || [];
  const entries = records
    .map(buildEntry)
    .filter((entry) => category === null || entry.category === category);
  response.json({
    category,
    page,
    size,
    total: entries.length,
    entries: paginate(entries, page, size),
  });
}

module.exports = { CATEGORIES, buildEntry, catalogRoute, normalizeCategory, paginate };
"""

_GITIGNORE = "node_modules/\ndist/\n"


# --------------------------------------------------------------------------------------
# The fixtures
# --------------------------------------------------------------------------------------


def node_npm_repository(root: Path, *, with_lockfile: bool = True) -> Path:
    """A Node project declaring npm, with lint, test and build scripts that all run.

    The baseline is clean: `npm run lint`, `npm run test` and `npm run build` each exit zero
    against the committed source. That is what makes it usable for the approval path and for
    every gate that has to be reached *after* a healthy checkout.

    `with_lockfile=False` removes the one file that makes a deterministic install possible,
    which is what preflight blocks on. Kept as a variant of this fixture rather than a
    fixture of its own: the difference is a single missing file, and a second near-identical
    checkout would drift away from this one the first time either is changed.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "npm-declared-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "lint": "node scripts/lint.js",
                "test": "node --test",
                "build": "node scripts/build.js",
            },
        },
    )
    if with_lockfile:
        _write_json(root / "package-lock.json", _empty_npm_lockfile("npm-declared-service"))
    _write_node_project(root)
    commit_all(root)
    return root


def node_pnpm_repository(root: Path) -> Path:
    """The same project declaring pnpm instead, so detection cannot be assumption.

    Identical in every respect except the lockfile. A platform that inferred a package
    manager from the presence of `package.json` would select npm here and be wrong, and the
    difference would only ever show up as a command that cannot run in the deployment.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "pnpm-declared-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "lint": "node scripts/lint.js",
                "test": "node --test",
                "build": "node scripts/build.js",
            },
        },
    )
    _write(root / "pnpm-lock.yaml", _empty_pnpm_lockfile())
    _write_node_project(root)
    commit_all(root)
    return root


def python_uv_repository(root: Path, *, with_lockfile: bool = False) -> Path:
    """A Python project configured for ruff and pytest through its own pyproject.

    `with_lockfile` writes a `uv.lock`, which is what makes the plan prefix its commands with
    `uv run --frozen`. It is off by default because resolving a lock needs a package index,
    and a fixture that needs the network is a fixture that fails in an air-gapped CI. The
    lockfile form is therefore used to assert *selection* and never executed.
    """
    init_git_repository(root)
    _write(
        root / "pyproject.toml",
        '[project]\nname = "uv-declared-service"\nversion = "1.0.0"\n'
        'requires-python = ">=3.12"\n\n'
        "[tool.ruff]\nline-length = 100\n\n"
        '[tool.ruff.lint]\nselect = ["E", "F"]\n\n'
        # `pythonpath` so the committed tests can import the package they test without the
        # checkout having to be installed. Every Python repository solves this somehow; this
        # is the cheapest way that is also true of real ones.
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n',
    )
    _write(root / "app" / "__init__.py", '"""Service package."""\n')
    _write(
        root / "app" / "status.py",
        '"""Service status."""\n\n\n'
        "def server_status() -> dict[str, str]:\n"
        '    """Return the current service status."""\n'
        '    return {"status": "ok"}\n',
    )
    _write(
        root / "tests" / "test_status.py",
        '"""Service status tests."""\n\n'
        "from app.status import server_status\n\n\n"
        "def test_server_status_reports_ok() -> None:\n"
        '    """The status endpoint reports a healthy service."""\n'
        '    assert server_status() == {"status": "ok"}\n',
    )
    if with_lockfile:
        _write(root / "uv.lock", 'version = 1\nrequires-python = ">=3.12"\n')
    commit_all(root)
    return root


def two_package_monorepo_repository(root: Path) -> Path:
    """Two independent packages, each with its own manifest and its own scripts.

    Neither package's toolchain may be run in the other's directory, and neither may be run
    at the checkout root. That is a statement about working directories, which is why both
    packages here declare commands that would genuinely fail if run from the wrong one.
    """
    init_git_repository(root)
    _write_json(
        root / "web" / "package.json",
        {
            "name": "monorepo-web",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write_json(root / "web" / "package-lock.json", _empty_npm_lockfile("monorepo-web"))
    _write_node_project(root / "web", with_build=False)
    _write(
        root / "service" / "pyproject.toml",
        '[project]\nname = "monorepo-service"\nversion = "1.0.0"\n'
        'requires-python = ">=3.12"\n\n'
        "[tool.ruff]\nline-length = 100\n\n"
        '[tool.ruff.lint]\nselect = ["E", "F"]\n\n'
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n',
    )
    _write(root / "service" / "app" / "__init__.py", '"""Monorepo service package."""\n')
    _write(
        root / "service" / "app" / "handler.py",
        '"""Monorepo service entry point."""\n\n\n'
        "def handler() -> str:\n"
        '    """Return the service name."""\n'
        '    return "service"\n',
    )
    _write(
        root / "service" / "tests" / "test_handler.py",
        '"""Monorepo service tests."""\n\n'
        "from app.handler import handler\n\n\n"
        "def test_handler_names_the_service() -> None:\n"
        '    """The handler names its own service."""\n'
        '    assert handler() == "service"\n',
    )
    _write(root / ".gitignore", _GITIGNORE)
    commit_all(root)
    return root


def broken_lint_config_repository(root: Path) -> Path:
    """A lint configuration extending a shared config the manifest never declares.

    The checkout is otherwise healthy: dependencies install, the runtime is supported, and
    the source has no violations. Lint still exits before it reads a single source file,
    which is the failure the platform must recognise as the repository's rather than the
    engineer's. Feature -007 spent its entire retry budget rewriting code that was never the
    problem.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "undeclared-config-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("undeclared-config-service"))
    _write_node_project(root, with_build=False, extends=[UNDECLARED_SHARED_CONFIG])
    commit_all(root)
    return root


def no_tests_repository(root: Path) -> Path:
    """A Node project with no test script at all.

    Not a broken checkout: plenty of repositories have not written tests yet, and refusing
    to implement a feature there refuses the very change that would add them. What the
    platform must do is *say so* -- and it must not silently plan a test command the
    repository never configured.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "untested-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("untested-service"))
    _write_node_project(root, with_build=False, with_tests=False)
    commit_all(root)
    return root


def slow_suite_repository(root: Path) -> Path:
    """A test command that never finishes, so its verdict is never available.

    The distinction this proves is between a suite that failed and a suite that could not
    complete. Rewriting the file under test cannot make a command fit inside its budget, so
    a platform that classified this as a source defect would spend a coding budget on it --
    which is exactly what the capacity classification exists to prevent.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "slow-suite-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node scripts/slow-test.js"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("slow-suite-service"))
    _write_node_project(root, with_build=False, with_tests=False)
    _write(root / "scripts" / "slow-test.js", _SLOW_TEST_SCRIPT)
    commit_all(root)
    return root


def unnarrowable_test_repository(root: Path) -> Path:
    """A healthy Node project whose test script chains two commands.

    `node scripts/pretest.js && node --test` is an ordinary shape -- a repository that
    generates or checks something before its suite runs -- and it is one that cannot be
    narrowed: trailing arguments would reach only whichever half ran last, so
    `npm run test -- one.test.js` would hand the file to `node --test` in the lucky case and
    to the pre-step in the unlucky one. `scoped_test_command` returns None here, and the
    correct in-attempt behaviour is to run nothing at all rather than to run the whole suite.

    Everything else about the checkout is the healthy Node baseline, so a test built on it
    isolates the narrowing decision instead of also exercising a broken repository.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "composed-test-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "lint": "node scripts/lint.js",
                "test": "node scripts/pretest.js && node --test",
                "build": "node scripts/build.js",
            },
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("composed-test-service"))
    _write_node_project(root)
    _write(root / "scripts" / "pretest.js", _PRETEST_SCRIPT)
    commit_all(root)
    return root


def cross_module_dependency_repository(root: Path) -> Path:
    """A checkout where the file the plan assigns depends on a module named nothing like it.

    Built from AB-Feature-170, generically. Three properties, and all three are needed
    before the failure it reproduces can happen at all:

    * `src/routes/listing.js` -- the file a plan would assign -- already imports
      `../support/envelope`. *Already*: the import is committed, so an attempt that starts
      calling `envelope.pageBody` adds no import line, and a mechanism reading a diff's
      added lines sees nothing.
    * `src/support/envelope.js` defines that shape and is named after nothing a requirement
      about a listing endpoint would say, so relevance ranking cannot rescue it.
    * `src/app.js` imports six of the repository's own modules. It is the breadth that
      matters: a global budget filled in arrival order is spent here before any file the
      change is about has been read.

    And the checkout is larger than the file budget, which is the condition without which
    none of this is observable: a repository that fits entirely in the snapshot delivers the
    definition whatever the selection does, and a test written against one would pass with
    the defect still in place. The filler modules are named for the same subject a
    requirement about a listing endpoint would use, so relevance ranking prefers them and
    the module that matters -- named after none of it -- is exactly what falls off the end.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "listing-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("listing-service"))
    _write_node_project(root, with_build=False, with_tests=False)
    for name in ("config", "logger", "errors", "sockets", "auth", "metrics"):
        _write(root / "src" / "platform" / f"{name}.js", f"module.exports = {{ {name}: true }};\n")
    _write(
        root / "src" / "app.js",
        "".join(
            f"const {{ {name} }} = require('./platform/{name}');\n"
            for name in ("config", "logger", "errors", "sockets", "auth", "metrics")
        )
        + "\nmodule.exports = { config, logger, errors, sockets, auth, metrics };\n",
    )
    for index in range(_CROSS_MODULE_FILLER_COUNT):
        _write(
            root / "src" / "listing" / f"listingPage{index:02d}.js",
            f"const listingPage{index:02d} = {{ page: {index} }};\n\n"
            f"module.exports = {{ listingPage{index:02d} }};\n",
        )
    _write(
        root / "src" / "support" / "envelope.js",
        "const pageBody = {\n"
        "  describe() {\n"
        "    return { keys: { items: 'array', cursor: 'string' } };\n"
        "  },\n"
        "};\n\n"
        "module.exports = { pageBody };\n",
    )
    _write(
        root / "src" / "routes" / "listing.js",
        "const envelope = require('../support/envelope');\n"
        "const { statusRoute } = require('./status');\n\n"
        "function listingRoute(request, response) {\n"
        "  response.json({ items: [], shape: envelope.pageBody });\n"
        "}\n\n"
        "module.exports = { listingRoute, statusRoute };\n",
    )
    commit_all(root)
    return root


def typechecked_repository(root: Path) -> Path:
    """A healthy Node project that also declares a `typecheck` script, and passes it.

    The baseline matters as much as the failure: `npm run lint`, `npm run typecheck` and
    `npm run test` all exit zero against the committed source, so a test built on this
    isolates the type error it introduces rather than also exercising a broken checkout.

    The declared shapes live in `types.json`, which is where a change that adds a module also
    declares what that module exports. That is what makes a *localized* type mismatch
    expressible here -- a value that disagrees with its own declaration, in the file the
    attempt just wrote, which is the shape of type error a repair pass can actually clear.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "typed-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "lint": "node scripts/lint.js",
                "typecheck": "node scripts/typecheck.js",
                "test": "node --test",
            },
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("typed-service"))
    _write_node_project(root, with_build=False)
    _write(root / "scripts" / "typecheck.js", _TYPECHECK_SCRIPT)
    _write_json(root / DECLARED_TYPES, _COMMITTED_TYPES)
    commit_all(root)
    return root


def missing_import_repository(root: Path) -> Path:
    """A checkout whose linter reports an unbound name, and a module named nothing like it.

    The shape behind the commonest mechanical failure this platform has. `permissionUtils`
    is exported by `src/support/permissions.js`, so:

    * nothing that reasons from filenames finds it -- the symbol and the stem differ, which
      is the ordinary case rather than a contrivance; and
    * nothing that follows imports finds it either, because the attempt that needs it is
      precisely the one that failed to write the import. There is no binding to follow.

    So an attempt using the name without importing it gets `'permissionUtils' is not defined`
    and a thirty-line window around the *use*, and the only way to write the specifier is to
    have been shown where the checkout defines the name. The linter's second rule refuses a
    module path that does not resolve, which is what keeps a guessed specifier from passing
    for a read one.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "publishing-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("publishing-service"))
    _write_node_project(root, with_build=False, lint_script=_UNDEFINED_SYMBOL_LINT_SCRIPT)
    _write(root / MISSING_IMPORT_DEFINITION, _PERMISSIONS_MODULE)
    commit_all(root)
    return root


def key_bearing_module_repository(root: Path) -> Path:
    """A checkout where the file defining a symbol is also the file holding a private key.

    The case both halves of the key-material rule have to survive at once. The path says
    nothing -- `src/support/signingKeys.js` is an ordinary source name in an ordinary source
    directory -- and the bytes say everything, so only the content rule can catch it.

    What must happen is different at each of the two places the definition-site lookup is
    used, and the difference is the point:

    * the repair prompt must quote **nothing** from it, and
    * the context snapshot must withhold its bytes and *say so* through
      ``required_dropped_paths`` (whose union field ``required_omitted_paths`` keeps the old
      readers working), because "the attempt never saw the file it was told about" has to
      stay answerable from the artifact rather than becoming silence.

    The finding is then simply not fixable by that attempt, which is the correct outcome and
    a recorded one.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "signing-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("signing-service"))
    _write_node_project(
        root, with_build=False, with_tests=False, lint_script=_UNDEFINED_SYMBOL_LINT_SCRIPT
    )
    _write(
        root / KEY_BEARING_DEFINITION,
        "const signingKey = [\n"
        "  '-----BEGIN RSA PRIVATE KEY-----',\n"
        f"  '{CREDENTIAL_SECRET}',\n"
        "  '-----END RSA PRIVATE KEY-----',\n"
        "].join('\\n');\n"
        "\n"
        "const signingCredentials = { key: signingKey, algorithm: 'RS256' };\n"
        "\n"
        "module.exports = { signingCredentials };\n",
    )
    commit_all(root)
    return root


def large_constants_module_source(*, extra_entries: Sequence[str] = ()) -> str:
    """Return the constants module's source, optionally with entries a change would add.

    The additions go in the middle rather than at the end, because a change appended to the
    tail of a file is the one case head truncation would happen to survive. A reviewer that
    can only see the head of this module is exactly what has to fail.
    """
    entries = [
        f"  ENTRY_{index:04d}: 'catalog-entry-value-{index:04d}',"
        for index in range(_LARGE_CONSTANTS_ENTRY_COUNT)
    ]
    midpoint = len(entries) // 2
    body = [*entries[:midpoint], *extra_entries, *entries[midpoint:]]
    return "const CONSTANTS = {\n" + "\n".join(body) + "\n};\n\nmodule.exports = { CONSTANTS };\n"


def large_constants_repository(root: Path) -> Path:
    """A healthy Node project carrying one module far too long to be reviewed whole.

    The baseline is clean -- `npm run lint` and `npm run test` exit zero against the committed
    source -- so a test built on this isolates the evidence decision rather than also
    exercising a broken checkout. The module is referenced from the checkout's own entry point
    for the same reason: an unreferenced file is a different finding, and this fixture is about
    what the reviewer can see, not about what the change wired.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "catalog-constants-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("catalog-constants-service"))
    _write_node_project(root, with_build=False)
    _write(root / LARGE_CONSTANTS_MODULE, large_constants_module_source())
    _write(
        root / "src" / "index.js",
        "const { statusRoute } = require('./routes/status');\n"
        "const { catalogRoute } = require('./routes/catalog');\n"
        "const { CONSTANTS } = require('./support/constants');\n\n"
        "module.exports = { statusRoute, catalogRoute, CONSTANTS };\n",
    )
    commit_all(root)
    return root


def oversized_lockfile_source(name: str, *, extra_packages: Sequence[str] = ()) -> str:
    """Return a `package-lock.json` past the reviewer's read cap, in npm's own shape.

    Real resolved-URL and integrity fields, because the size that matters is the size a real
    lockfile has and the reason it has it: a few thousand resolved packages, each carrying a
    registry URL and a hash. `extra_packages` are the entries a dependency change would add,
    which is what makes a numstat against the branch point non-zero.
    """
    packages: dict[str, object] = {"": {"name": name, "version": "1.0.0"}}
    for entry in (
        *(f"generated-package-{index:05d}" for index in range(_OVERSIZED_LOCKFILE_PACKAGE_COUNT)),
        *extra_packages,
    ):
        packages[f"node_modules/{entry}"] = {
            "version": "1.0.0",
            "resolved": f"https://registry.example.test/{entry}/-/{entry}-1.0.0.tgz",
            "integrity": f"sha512-{'0' * 86}==",
            "license": "MIT",
        }
    return (
        json.dumps(
            {
                "name": name,
                "version": "1.0.0",
                "lockfileVersion": 3,
                "requires": True,
                "packages": packages,
            },
            indent=2,
        )
        + "\n"
    )


def oversized_lockfile_repository(root: Path) -> Path:
    """A healthy Node project whose committed lockfile is larger than the reviewer can read.

    Deliberately not a variant of `node_npm_repository`, which promises `npm ci` runs against
    it. This lockfile names several thousand packages that do not exist, so an install would
    need a registry: nothing here may execute it, and nothing does -- the evidence decision
    this fixture exists for is made from `stat`, a streamed hash and `git diff`, none of which
    care whether the packages resolve. The rest of the checkout is ordinary and its own `npm
    run lint` and `npm run test` still pass, so a test built on it isolates the evidence
    decision rather than also exercising a broken checkout.
    """
    init_git_repository(root)
    _write_json(
        root / OVERSIZED_LOCKFILE_MANIFEST,
        {
            "name": "lockfile-carrying-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write(root / OVERSIZED_LOCKFILE, oversized_lockfile_source("lockfile-carrying-service"))
    _write_node_project(root, with_build=False)
    commit_all(root)
    return root


def credential_bearing_repository(root: Path) -> Path:
    """A checkout carrying key material that no filename rule can see.

    Built from AB-Feature-170, generically. The four files it committed were read and sent
    to a provider as content on five engineer calls, and the reason splits in two:

    * `mailer.env` is an environment file the rule matched only as a *prefix*, so `.env`
      and `.env.production` were caught and anything ending in `.env` was not.
    * `analytics-1122334455667-a1b2c3d4e5f6.json` is a service-account key, and it is
      indistinguishable from `package.json` by name. No filename rule can catch it, and
      `.json` cannot be blanket-excluded, so only its content identifies it.

    Everything here sits at the checkout root deliberately. Root files are ranked ahead of
    everything nested, so these are delivered on merit rather than surviving on budget, and
    a test that finds them absent has found the rule working rather than the ceiling.

    The two files that must *survive* are the point of the other half. `.env.example`
    carries the same variable names as `mailer.env` with placeholder values -- committed
    deliberately, holding no key -- and `tsconfig.json` is ordinary configuration. A
    scanner that withholds either has starved the Engineer of context, which is the defect
    the previous task spent itself fixing.
    """
    init_git_repository(root)
    _write_json(
        root / "package.json",
        {
            "name": "mailer-service",
            "version": "1.0.0",
            "private": True,
            "scripts": {"lint": "node scripts/lint.js", "test": "node --test"},
        },
    )
    _write_json(root / "package-lock.json", _empty_npm_lockfile("mailer-service"))
    _write_node_project(root, with_build=False, with_tests=False)
    _write_json(root / "tsconfig.json", {"compilerOptions": {"target": "es2022"}})
    _write(root / CREDENTIAL_ENVIRONMENT_FILE, f"export MAILER_API_KEY='{CREDENTIAL_SECRET}'\n")
    _write(root / CREDENTIAL_ENVIRONMENT_TEMPLATE, "export MAILER_API_KEY=''\n")
    _write_json(
        root / CREDENTIAL_SERVICE_ACCOUNT,
        {
            "type": "service_account",
            "project_id": "mailer-delivery",
            "private_key": f"-----BEGIN PRIVATE KEY-----\n{CREDENTIAL_SECRET}\n"
            "-----END PRIVATE KEY-----\n",
            "client_email": "delivery@mailer-delivery.example.com",
        },
    )
    _write(
        root / CREDENTIAL_PEM_FILE,
        f"-----BEGIN RSA PRIVATE KEY-----\n{CREDENTIAL_SECRET}\n-----END RSA PRIVATE KEY-----\n",
    )
    _write(
        root / CREDENTIAL_DOCUMENTATION,
        "# Rotating the delivery key\n\n"
        "The stored key is a PEM block, so it begins with this line:\n\n"
        "```\n-----BEGIN RSA PRIVATE KEY-----\n```\n\n"
        "Replace the whole block, then restart the workers.\n",
    )
    commit_all(root)
    return root


# --------------------------------------------------------------------------------------
# Shared construction
# --------------------------------------------------------------------------------------


def _write_node_project(
    root: Path,
    *,
    with_build: bool = True,
    with_tests: bool = True,
    extends: list[str] | None = None,
    lint_script: str = _LINT_SCRIPT,
) -> None:
    """Write the source, the linter and its configuration shared by every Node fixture."""
    _write(root / "scripts" / "lint.js", lint_script)
    if with_build:
        _write(root / "scripts" / "build.js", _BUILD_SCRIPT)
    _write_json(root / "lintrc.json", {"extends": extends or [], "sources": ["src"]})
    _write(
        root / "src" / "index.js",
        "const { statusRoute } = require('./routes/status');\n"
        "const { catalogRoute } = require('./routes/catalog');\n\n"
        "module.exports = { statusRoute, catalogRoute };\n",
    )
    _write(
        root / "src" / "routes" / "status.js",
        "function statusRoute(request, response) {\n"
        "  response.json({ status: 'ok' });\n"
        "}\n\n"
        "module.exports = { statusRoute };\n",
    )
    _write(root / "src" / "routes" / "catalog.js", _CATALOG_MODULE)
    if with_tests:
        _write(
            root / "test" / "status.test.js",
            "const { test } = require('node:test');\n"
            "const assert = require('node:assert');\n"
            "const { statusRoute } = require('../src/routes/status');\n\n"
            "test('the status route answers', () => {\n"
            "  let body = null;\n"
            "  statusRoute({}, { json: (value) => { body = value; } });\n"
            "  assert.deepStrictEqual(body, { status: 'ok' });\n"
            "});\n",
        )
    _write(root / ".gitignore", _GITIGNORE)


def _empty_npm_lockfile(name: str) -> dict[str, object]:
    """Return a lockfile npm accepts as in sync with a manifest that declares nothing.

    `npm ci` refuses a lockfile that disagrees with `package.json`, so this is the real
    shape rather than a placeholder: the root package entry is what makes the two agree.
    """
    return {
        "name": name,
        "version": "1.0.0",
        "lockfileVersion": 3,
        "requires": True,
        "packages": {"": {"name": name, "version": "1.0.0"}},
    }


def _empty_pnpm_lockfile() -> str:
    """Return a pnpm lockfile with one importer and no dependencies."""
    return (
        "lockfileVersion: '9.0'\n\n"
        "settings:\n"
        "  autoInstallPeers: true\n"
        "  excludeLinksFromLockfile: false\n\n"
        "importers:\n\n"
        "  .: {}\n"
    )


def _write(path: Path, content: str) -> None:
    """Write one fixture file, creating whatever directories it needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one fixture manifest as formatted JSON."""
    _write(path, f"{json.dumps(payload, indent=2)}\n")
