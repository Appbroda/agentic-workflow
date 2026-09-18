# Repository validation

Live child validation is selected from checked-out repository evidence, never from an LLM guess.
`tools/technology_detection.py` detects languages, package managers, framework/tool configuration,
and standard manifests. `DefaultValidationPlanBuilder` then creates fixed non-shell commands.

Before validation or coding, live children run [repository preflight](REPOSITORY_PREFLIGHT.md).
Node dependency installation is a separate journaled bootstrap operation, not a validation command.

For JavaScript and TypeScript, configured `package.json` scripts are used in the order
`format:check`, `format`, `lint`, `typecheck`, `test`, `test:ci`, and `build`. npm, pnpm, and yarn
are selected from the repository lockfile. Python commands are used only when Python manifests and
configured Ruff, mypy/pyright, or pytest evidence exist. Mixed repositories are planned per
manifest directory, so a JavaScript child never receives `ruff` or `pytest` merely because another
repository is Python.

Each result records command, validation type, repository identifier, combined revision fingerprint,
working-tree fingerprint, exit code, bounded output summaries, duration, and journal operation ID.
`passed`, `failed`, `not_configured`, `no_tests_found`, `skipped`, and `cancelled` have distinct
meanings. In particular, pytest exit code 5 is `no_tests_found`, not a successful test run.
When no native test script exists, the result code is `TEST_COMMAND_NOT_CONFIGURED`; it must never
be presented as passed. ESLint bootstrap/configuration failures are distinct from source lint
violations.

Validation is re-run after any source revision change. The operation key includes workspace,
revision, command/configuration fingerprint, and environment fingerprint; a result can only be
reused for the exact same state.
