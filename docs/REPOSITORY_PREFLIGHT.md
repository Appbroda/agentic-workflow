# Repository preflight

Every live child workstream runs repository preflight after cloning and before the coding executor.
It records the checkout revision, package manager, configured scripts, source/test directories,
entrypoints, frameworks, validation readiness, and actionable issues.

For Node repositories, lockfiles select the only permitted bootstrap command:

- `package-lock.json` → `npm ci`
- `pnpm-lock.yaml` → `pnpm install --frozen-lockfile`
- `yarn.lock` → `yarn install --frozen-lockfile`

The installation is bounded, cancellable, and journaled as `install_dependencies`. A missing
lockfile, failed deterministic installation, or undeclared shared lint configuration is a setup
issue, not an Engineer or source-code failure. The system never adds an undeclared dependency
automatically.

ESLint failures are classified as `lint_violation`, `lint_configuration_error`,
`missing_dependency`, or `unsupported_runtime`. `airbnb-base` references are checked against
`package.json` and the selected lockfile before code is changed. A declared locked dependency may
be restored by deterministic installation; an undeclared one requires approved repository repair.

`TEST_COMMAND_NOT_CONFIGURED` is emitted when a package has no `test` or `test:ci` script. It is
not a passing test result. The workstream API exposes readiness, setup issues, package manager,
and safe configured validation commands; process environments and credentials are never exposed.
