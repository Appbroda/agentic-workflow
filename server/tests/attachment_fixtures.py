"""The committed image bytes the attachment tests use, read from disk and never generated.

`sha256` is asserted in several places, and a fixture regenerated at test time is a test that
passes for the wrong reason: whatever the encoder in the running environment produces becomes
the expected value, so a hash mismatch -- exactly the failure the column exists to catch --
cannot happen. These are five real files, checked in.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_DIRECTORY = Path(__file__).resolve().parent / "fixtures" / "attachments"

# A 1x1 PNG, a 1x1 JPEG and a 1x1 WebP: one of each accepted type, as small as the format
# allows, so a test that reads them costs nothing.
PNG = _DIRECTORY / "pixel.png"
JPEG = _DIRECTORY / "pixel.jpg"
WEBP = _DIRECTORY / "pixel.webp"
# An SVG, for the refusal that has its own message.
SVG = _DIRECTORY / "square.svg"
# A zip archive named `.png`, for the rule that a declared type is a claim.
ZIP_NAMED_PNG = _DIRECTORY / "archive-named-png.png"


def content(path: Path) -> bytes:
    """Return one fixture's bytes."""
    return path.read_bytes()


def digest(path: Path) -> str:
    """Return one fixture's sha256, computed from the committed bytes."""
    return hashlib.sha256(content(path)).hexdigest()


ACCEPTED = ((PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp"))
