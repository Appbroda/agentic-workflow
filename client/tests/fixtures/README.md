# Captured payloads

Every file here is a genuine response from the real server — the same router, the same response
model and the same serialization a deployment runs. A hand-written double agrees with the code
reading it by construction, which is how this client silently lost a dozen fields once.

## `design-snapshot.json` and `design-snapshot.omissions.json`

Two `GET /features/{id}/artifacts?artifact_type=design_snapshot` responses, written by
`server/tests/capture_design_snapshot_fixture.py`. Regenerate with, from `server/`:

    uv run python -m tests.capture_design_snapshot_fixture

**What they are.** Real all the way down except the network. Each was produced by the real
`/features/start` path, the real design-resolution step and the shipped extraction, over the
live Figma payload committed at `server/tests/fixtures/figma/figma_file_read.json` — and read
back through the real artifact endpoint. Only the Figma adapter is stubbed, and it answers out
of that committed capture rather than inventing a shape.

**Why two.** Being honest about a bound that bit is the renderer's whole job, and the two
states look completely different:

- `design-snapshot.json` — the file's three real frames, all quoted, nothing left out.
- `design-snapshot.omissions.json` — the same file at a per-node character bound of 8,000,
  which sits between the two largest frames' measured sizes (8,994 and 7,270 characters), so
  exactly one frame is omitted and named with its size and its reason.

**What they are not.** They come from a file that uses no auto layout and defines no named
styles, so they cannot exercise the renderer's `style_names` block. What the live API returns
for a file that *does* was verified separately; see `.codex-prompts/89-design-report.md`.
