"""How prose points at an image: `[image:login-error]`.

One syntax and one regular expression, used by the submission validator, the prompt renderer
and the interface, so the three cannot come to disagree about what a marker is. The
alternative -- each of them recognising "roughly that shape" -- is how a submission is
accepted with a reference the prompt then renders as literal text.

A marker is a slug: lowercase letters and digits, hyphens inside, at most forty characters.
Short enough to type in the middle of a sentence, long enough to be `checkout-error-state`,
and with nothing in it that needs escaping wherever it is rendered.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# What a marker may be called. Anchored, because this is also the field validator: a value
# that merely *contains* a valid marker is not one.
MARKER_PATTERN = r"^[a-z0-9][a-z0-9-]{0,39}$"
MARKER_REGEX = re.compile(MARKER_PATTERN)

# How a marker appears in prose. The inner group is deliberately the same character class as
# above rather than a loose `[^\]]+`: `[image:Login Error]` is not a reference to anything,
# and matching it here would turn a typo into a rejection naming a marker nobody declared.
REFERENCE_REGEX = re.compile(r"\[image:([a-z0-9][a-z0-9-]{0,39})\]")

# The longest a caption may be. A caption is a sentence about a screenshot -- "the error that
# appears after a failed login" -- and it travels into a prompt, so it is bounded.
MAX_CAPTION_LENGTH = 500


def is_marker(value: str) -> bool:
    """Return whether one string is a well-formed marker."""
    return MARKER_REGEX.match(value) is not None


def markers_in(text: str) -> list[str]:
    """Return every marker referenced in one piece of prose, in the order they appear."""
    return REFERENCE_REGEX.findall(text)


def markers_in_all(texts: Iterable[str]) -> list[str]:
    """Return every marker referenced across many pieces of prose, first occurrence first."""
    seen: dict[str, None] = {}
    for text in texts:
        for marker in markers_in(text):
            seen.setdefault(marker, None)
    return list(seen)


def reference(marker: str) -> str:
    """Return the text that references one marker, so nothing builds the string by hand."""
    return f"[image:{marker}]"


__all__ = [
    "MARKER_PATTERN",
    "MARKER_REGEX",
    "MAX_CAPTION_LENGTH",
    "REFERENCE_REGEX",
    "is_marker",
    "markers_in",
    "markers_in_all",
    "reference",
]
