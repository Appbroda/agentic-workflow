"""Bounded windows of source text, for prompts that cannot carry a whole file.

Two agents quote source they are unable to show whole, and they have to do it the same way.
The Engineer excerpts a required context file around the line a diagnostic named; the
Reviewer excerpts a changed file around each hunk the change touched. One implementation, so
the radius, the trimming order and the reported line range are a single decision rather than
two that drift apart -- and so a reader of either prompt can rely on `excerpt_lines` meaning
the same thing in both.
"""

from __future__ import annotations

__all__ = ["SOURCE_REGION_RADIUS_LINES", "source_region"]

# How much unchanged source surrounds the line -- or the span -- a window is anchored on.
# Thirty lines is what makes a quoted region answerable rather than merely indicative: a
# repair pass reading `const nestedDocument;` beside the eslint line that names it is a
# one-edit fix, and a review reading a changed hunk needs the declarations around it to judge
# whether the change is correct.
SOURCE_REGION_RADIUS_LINES = 30


def source_region(
    content: str,
    line_number: int | None,
    *,
    max_characters: int,
    end_line: int | None = None,
    radius_lines: int | None = None,
) -> tuple[str, int, int]:
    """Return a bounded window of ``content`` around one line or span, with its line range.

    Centered on the anchor so trimming for the character bound sheds the edges and never the
    middle; a missing or out-of-range line number falls back to the head of the file, which
    still beats dropping the file entirely. Line numbers are one-based, as every linter
    reports them, and the returned range is inclusive of both ends.

    ``end_line`` extends the anchor from a single line to a span, which is what a changed
    hunk is. Without it the window is radius-bounded on both sides of one line, so a hunk
    longer than the radius would be quoted with its middle missing while the returned range
    still claimed to cover it -- an excerpt that lies about what it contains is worse for a
    reviewer than no excerpt at all. The span is protected from trimming ahead of the context
    around it, but it is not guaranteed to survive a bound too small to hold it: a caller
    that must know whether the whole span is present compares the returned range against the
    span it asked for.

    ``radius_lines`` overrides the module's thirty-line radius for callers whose real bound
    is ``max_characters`` rather than answerability: a required context file entitled to a
    twelve-thousand-character budget was being cut to sixty-one lines because the *radius*
    bound first (199's `app.service.js`, shown lines 1-61 of 643 on eight straight
    attempts). Widening the radius never changes what ``excerpt_lines`` means -- the
    returned range is always exactly what the window contains -- so both existing callers
    keep reading it the same way.
    """
    radius = SOURCE_REGION_RADIUS_LINES if radius_lines is None else radius_lines
    lines = content.splitlines()
    if line_number is None or not 1 <= line_number <= len(lines):
        start = 1
        selected = lines[: 2 * radius + 1]
        anchor_start = anchor_end = 1
    else:
        anchor_start = line_number
        anchor_end = min(max(end_line or line_number, line_number), len(lines))
        start = max(1, line_number - radius)
        selected = lines[start - 1 : anchor_end + radius]
    end = start + len(selected) - 1
    if not start <= anchor_start <= end:
        anchor_start = anchor_end = start
    # The window length is tracked incrementally: joining the selection on every trim is
    # quadratic, which the thirty-line radius hid and a budget-derived one would not.
    selected_length = sum(len(line) for line in selected) + max(0, len(selected) - 1)
    while len(selected) > 1 and selected_length > max_characters:
        # Shed from whichever side carries more context beyond the anchor, so the anchor is
        # the last thing to go rather than the first.
        if end - anchor_end >= anchor_start - start:
            selected_length -= len(selected.pop()) + 1
            end -= 1
        else:
            selected_length -= len(selected.pop(0)) + 1
            start += 1
    region = "\n".join(selected)
    if len(region) > max_characters:
        region = region[:max_characters]
    return region, start, end
