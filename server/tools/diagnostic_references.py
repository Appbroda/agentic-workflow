"""File-shaped tokens in diagnostic text, extracted once for every consumer.

This lives in its own module because both ends of the failure-evidence path need it and
they import in opposite directions: ``tools.retry_strategy`` (whose callers resolve tokens
against an attempt lineage) already imports from ``tools.validation_tools``, and
``tools.validation_tools`` (which resolves tokens against the workspace it just ran a
command in) must not import ``tools.retry_strategy`` back. The regex is written once, here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# A file-shaped token as a linter, a test runner or a reviewer writes one: an optional
# directory chain, a final segment with a short suffix, and optionally the `:line` or
# `:line:col` a diagnostic attaches. Deliberately permissive -- `2.0` in a package name
# matches too -- because every consumer of these tokens applies its own membership test
# (a workspace, a lineage) and an unresolvable token costs nothing.
_FILE_REFERENCE_PATTERN = re.compile(
    r"(?P<path>/?(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,8})(?::(?P<line>\d+)(?::\d+)?)?"
)
# The second half of ESLint's default (stylish) report shape: the file on a line of its
# own, then one indented `line:col  <message>` per problem. The two lines have to be read
# together or the location is lost -- which is exactly how 176's `179:23` was reported.
_STYLISH_LOCATION_PATTERN = re.compile(r"^\s*(?P<line>\d+):\d+\s")


def diagnostic_file_references(diagnostics: Sequence[str]) -> list[tuple[str, int | None]]:
    """Return the file-shaped tokens these diagnostics name, with the first line each carries.

    The token is returned exactly as the diagnostic spelled it -- absolute, repo-relative,
    or bare -- because only the caller knows what to resolve it against: the engineer holds a
    workspace, the terminal triage holds an attempt lineage. First-seen order, one entry per
    distinct token. The line number is the first one attached to that token, whether inline
    (`path:line:col`) or in ESLint's stylish shape, where the path heads its own line and
    each `line:col` sits indented beneath it.
    """
    seen: dict[str, int | None] = {}
    order: list[str] = []

    def record(token: str, line: int | None) -> None:
        if token not in seen:
            seen[token] = line
            order.append(token)
        elif seen[token] is None and line is not None:
            # A stylish header records the token before its locations arrive; the first
            # location completes it rather than being read as a second file.
            seen[token] = line

    for text in diagnostics:
        header: str | None = None
        for raw_line in text.splitlines():
            location = _STYLISH_LOCATION_PATTERN.match(raw_line)
            if location is not None and header is not None:
                record(header, int(location.group("line")))
                continue
            matches = list(_FILE_REFERENCE_PATTERN.finditer(raw_line))
            for match in matches:
                token = match.group("path").rstrip(".")
                if not token:
                    continue
                line = match.group("line")
                record(token, int(line) if line else None)
            # A line that is nothing but one path token is a stylish header: the locations
            # on the following lines belong to it.
            stripped = raw_line.strip()
            if len(matches) == 1 and matches[0].group(0).strip() == stripped:
                header = matches[0].group("path").rstrip(".")
    return [(token, seen[token]) for token in order]
