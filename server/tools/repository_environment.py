"""The environment a repository publishes for running its own checks.

A repository's validation commands often need configuration before they will run at all. Both
DAM repositories are examples: `next build` and `jest` load `next.config.ts`, which imports a
t3-env module that throws unless five `NEXT_PUBLIC_*` variables are set. The platform supplied
one hardcoded variable -- `CI=true` for tests -- and nothing else, so any such repository
failed its own build on an untouched checkout and was reported as an unrepairable defect.

Nothing here invents a value. It reads the file the repository itself commits for the purpose,
which is the only non-guess available: the platform cannot derive somebody's API URL, and a
placeholder of its own choosing would be a decision about their project disguised as a
default. A repository that publishes no such file is unchanged.

Which files may be read is *not* decided here. `is_credential_shaped_path` already answers it
for the reviewer, the engineer's context rule and the commit gate, and it says `.env.example`,
`.env.sample` and `.env.template` are ordinary while `.env`, `.env.local`, `.env.test` and
`.env.defaults` are key material. Delegating means the four rules cannot disagree, and that a
future change to what counts as a secret reaches this reader for free -- which matters more
than it looks, because the mistake this guards against is reading a real `.env` into a
subprocess and then into a log.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from tools.file_tools import is_credential_shaped_path

# Conventional names for a committed, non-secret example environment, most specific first.
# Every candidate is still checked against the key-material rule before it is opened.
_CANDIDATE_FILENAMES = (".env.example", ".env.sample", ".env.template")

# POSIX environment variable names. Deliberately strict: a line that is not a plain assignment
# is skipped rather than repaired, because an example file is documentation and may contain
# prose, shell fragments or interpolation this must not try to evaluate.
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_MAX_FILE_BYTES = 64 * 1024
_MAX_VARIABLES = 256


def example_environment_files(root: Path) -> list[Path]:
    """The committed example environment files at this project root, secrets excluded."""
    return [
        candidate
        for name in _CANDIDATE_FILENAMES
        if (candidate := root / name).is_file()
        and not is_credential_shaped_path(PurePosixPath(name))
    ]


def declared_validation_environment(root: Path) -> dict[str, str]:
    """Parse the repository's own example environment into values its checks can run with.

    Bounded in both directions -- file size and variable count -- because this reads a file the
    repository controls and hands the result to every validation subprocess.

    Values are taken verbatim apart from one pair of surrounding quotes, and no interpolation
    is performed: `${OTHER}` stays literal. Expanding it would either leak the platform's own
    environment into the value or invent one, and an example file exists to be read, not run.
    """
    values: dict[str, str] = {}
    for path in example_environment_files(root):
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            stripped = stripped.removeprefix("export ").strip()
            name, separator, raw = stripped.partition("=")
            if not separator:
                continue
            name = name.strip()
            if not _NAME.fullmatch(name) or name in values:
                continue
            values[name] = _unquoted(raw.strip())
            if len(values) >= _MAX_VARIABLES:
                return values
    return values


def _unquoted(value: str) -> str:
    """Strip one matching pair of surrounding quotes, which is how such files are written."""
    for quote in ('"', "'"):
        if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
            return value[1:-1]
    return value


__all__ = ["declared_validation_environment", "example_environment_files"]
