# Saved payloads

## `feature_197_integration_contract.json`

Run 197's approved integration contract, `009_integration_contract.json` of
`feature-53d226e2-6021-4758-9281-790bba726ac7`, exactly as the deployment database holds it.

**What it is.** A genuine capture: read out of `feature_artifacts` and re-serialized with sorted
keys, nothing else changed. It is the contract whose `BulkImportSuccess` — `code` a
`const BULK_IMPORT_CREATED`, `created` an integer at `minimum: 1`, both `required` — an
independent Reviewer blocked a conformant frontend for restating in code, while holding only
the section's name. That defect and its fix are 00-todo item 25;
`test_contract_sections_for_judges.py` reads this file through
`IntegrationContractArtifact.model_validate_json`, the same deserialization the feature store
performs, so a field this repository renames or drops fails at load rather than at an assertion
written to agree with it.

**Why the whole artifact.** The selection under test reads five different contract fields, and
the bound it spends is measured against a real contract's real sizes — sixteen sections and
16,455 characters. A trimmed fixture would set those bounds against a contract nobody shipped.

**What it is not.** One contract from one feature. It says nothing about how large contracts get
in general; the bounds in `agents/shared/contract_sections.py` are calibrated to it and stated
there as such.

## Saved provider payloads

## `anthropic_messages_response.json`

One Messages API response, in the shape a coding-role call returns: a thinking block carrying
no text, then the JSON object the Engineer's contract requires.

**What it is.** The wire shape as the installed `anthropic` SDK defines it. Every test that
reads it loads it through `anthropic.types.Message.model_validate`, so the vendor's own model
is the gate: a field this repository invented, renamed, or dropped fails at load rather than
at an assertion that was written to agree with it. That is the failure this fixture exists to
catch — the web client lost a dozen fields to hand-written doubles that agreed with the code
reading them.

**What it is not.** It was not captured from a live call. No Anthropic credential is available
in this checkout, so the payload was constructed and then validated against the SDK's model
rather than recorded off the wire. That is weaker than a capture in exactly one way: it proves
the shape this SDK version accepts, not that the service currently emits every field in it.
Replace it with a genuine capture the first time a live key is available — the tests need no
change, because they already read it through the SDK.

## Saved Figma payloads

### `figma/figma_file_read.json`

`GET https://api.figma.com/v1/files/VGULlnz44R0Ooe4FZKDxlhh4`, captured live on 2026-09-07 with
a read-only Figma personal access token.

**What it is.** A genuine capture, re-serialized with sorted keys, with exactly one field
removed: `thumbnailUrl`. That field is a presigned S3 URL carrying `X-Amz-Credential` and an
`AKIA…` access key id, so it is a credential and committing it would be storing one in the
repository. Nothing else was touched — no field added, none renamed, none trimmed.

**Why a file read rather than a node read.** The extraction consumes a `FigmaNodeSubtree`,
which is what `GET /v1/files/:key/nodes` returns per node. The file endpoint returns the *same*
node shape and the *same* `styles` / `components` / `componentSets` maps, one level up — which
was verified live against both endpoints on the same file before this fixture was chosen. So
the test builds a subtree from this payload's own frames rather than from an invented one.

**Why this file.** It is one of the few real Figma files a personal access token can read
without organisation membership: it is the file Figma's own `figma-api-demo` repository
documents. It exercises TEXT nodes with their actual characters and typography, fills, strokes,
effects, constraints, `INSTANCE` nodes with `componentId`, a `COMPONENT`, `GROUP`, `VECTOR` and
`RECTANGLE`.

**What it is not.** It is not a design-system file. It uses no auto layout and defines no named
styles, so it cannot exercise `style_names` or the `layout` block — both of those were verified
against the live API separately (`28gd2JrZO28FCN9PCKM4qK`, "OpenCRM Design": 369 auto-layout
nodes, 68 named styles, `Primary / 100` and `Title/XL` returned by the nodes endpoint for an
ordinary frame with a read token) and the capture of *that* file was refused by Figma's
rate limiter for the rest of the session. See `.codex-prompts/89-design-report.md`.

### `figma/figma_nodes_absent.json`

`GET https://api.figma.com/v1/files/VGULlnz44R0Ooe4FZKDxlhh4/nodes?ids=1:2`, captured live on
2026-09-07, with `thumbnailUrl` removed for the reason above.

**What it is.** Figma's answer to a well-formed node id the file does not define: `200 OK` with
`nodes["1:2"] = null`. That is an *answer*, not a failure, and the whole reason
`design_nodes_absent` is a list rather than an error: this payload is what makes the difference
testable against the real thing rather than against an assumption about it.

### `figma/figma_product_frame.json`

`GET https://api.figma.com/v1/files/EqhdoZR1UKVZsaX7jplDsP/nodes?ids=73996:19746`, captured live
on 2026-09-10 with a read-only Figma personal access token (`role: viewer` is sufficient — the
resolution only ever reads).

**What it is.** A genuine capture, re-serialized with sorted keys and `indent=2`, with exactly
one field removed for the reason above: `thumbnailUrl`, which was again a presigned S3 URL
carrying `X-Amz-Credential` and an `AKIA…` key id. Nothing else was touched. It is 2.1 MB, which
is large for a fixture and is the point — this is what one real product screen actually weighs.

**Why a second Figma file.** `figma_file_read.json` above cannot exercise the two things every
size decision in 96- turns on: it uses no auto layout and defines no named styles. This file is
the opposite — a product archive with a design system inside it. Its frame `73996:19746` is 93
rendered nodes at depth 8, of which 83 are auto-layout containers; all 16 of its TEXT nodes
carry typography; 16 nodes reference a named type style (`Label/Label-3/Medium`,
`Body/Body-4/Regular`); 13 are component instances naming their sets (`Text Field`,
`Button Contained`). The real tree is 411 nodes at depth 17.

**What it measures.** Every bound in `artifacts/design_references.py` that 96- added or changed,
and the index/build size comparison the two-tier design rests on. Measured through the shipped
extraction and `_characters` (`indent=2` — the serialization `design_snapshot_context_json`
actually hands a model), the frame is **136,816 characters at build fidelity and 41,934 in index
mode**, 450 characters a node. Those exact numbers are asserted in `test_design_snapshot.py` so
a change to `_render` shows up as a drift rather than as a slow return to one tier.

**One node inside it is worth knowing about.** `74046:28253` is a self-contained sign-in form —
36 nodes, 35,359 characters, **natural depth 5**. It is the only real screen measured on this
file that fits today's bounds whole, with nothing truncated or omitted, which is what made it
the citation for the design path's first live run. `test_design_snapshot.py` pins it for that
reason: if a bound change stops it fitting, the design path loses its live smoke test.
